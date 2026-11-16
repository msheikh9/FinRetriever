"""Leakage-safe training-data preparation: one config, versioned artifacts, one manifest.

``prepare`` turns the frozen corpus into the passages that may give training examples:

1. overlap groups (``finretriever.training.overlap``);
2. the evaluation blocklist, expanded to whole groups (``finretriever.training.blocklist``);
3. each passage's tokenizer-visible span (``finretriever.training.visible_text``);
4. with a ``reuse`` section (manifest format 2): sentence- and fact-level evidence reuse
   (``finretriever.training.evidence_reuse``). Passages substantially reusing blocked evidence
   are blocked with their groups, and groups substantially reusing each other's evidence form
   one split unit. The result is verified: no remaining passage substantially reuses evaluation
   evidence and no link crosses the split;
5. a train/validation split of the remaining groups or units (``finretriever.training.splits``),
   optionally with per-company validation shares and with pinned units (``split.pins``, a
   split-pins file, ``finretriever.training.split_pins``) that keep an earlier preparation's
   training and validation data on their sides; the manifest records the pins file's SHA-256.

It writes, under ``output_dir`` (which must be inside the gitignored ``data/processed/``):
``overlap_groups.jsonl``, ``overlap_edges.jsonl``, ``blocklist.jsonl``, ``passages.jsonl``, in
format 2 ``evidence_reuse.jsonl`` (what each passage shares with evaluation passages) and
``reuse_links.jsonl`` (the links that joined groups), and ``manifest.json`` (corpus identity,
config, every setting and statistic, every file's SHA-256).
Outputs carry no timestamps, so the same corpus, evaluation sources, config and tokenizer give
byte-identical files; a version directory is never overwritten with different content.
``open_prepared`` reloads a preparation after checking its corpus and file hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from finretriever.data.schemas import Passage
from finretriever.eval_set.annotation_store import verify_corpus
from finretriever.evaluation.provenance import CorpusIdentity, sha256_file
from finretriever.training.blocklist import (
    BlocklistSources,
    build_blocklist,
    expand_by_evidence_reuse,
)
from finretriever.training.evidence_reuse import ReuseIndex, ReuseSettings
from finretriever.training.overlap import OverlapSettings, build_overlap_groups
from finretriever.training.query_generation import GenerationSettings, SourcePassage
from finretriever.training.source_quality import SourceQualitySettings
from finretriever.training.split_pins import SplitPinsError, load_split_pins
from finretriever.training.splits import SplitSettings, assign_splits, fiscal_year
from finretriever.training.visible_text import VisibleTextSettings, tokenizer_identity, visible_span
from finretriever.utils.io import read_yaml

MANIFEST_FILE = "manifest.json"
GROUPS_FILE = "overlap_groups.jsonl"
EDGES_FILE = "overlap_edges.jsonl"
BLOCKLIST_FILE = "blocklist.jsonl"
PASSAGES_FILE = "passages.jsonl"
REUSE_FILE = "evidence_reuse.jsonl"
LINKS_FILE = "reuse_links.jsonl"
DATA_FILES = (GROUPS_FILE, EDGES_FILE, BLOCKLIST_FILE, PASSAGES_FILE)
DATA_FILES_V2 = (*DATA_FILES, REUSE_FILE, LINKS_FILE)
REUSE_METHOD = (
    "spans (span_words consecutive case-folded [a-z0-9]+ words, BLAKE2b 64-bit hashes), sentence units (prose "
    "sentences of >= min_sentence_words words) and fact units (ticker, significant number, nearest label word); "
    "spans and sentences in passages of > max_text_companies companies, or any unit in > max_unit_passages "
    "passages, are generic; substantial reuse = visible words inside shared spans >= substantial_word_share of the "
    "visible words (spans only an adjacent chunk of the same section shares excluded), or >= substantial_facts "
    "shared facts that are >= substantial_fact_share of the passage's facts; visible text of the candidate against "
    "full text of blocked passages (blocklist, one step) or visible text of other passages (split links)"
)
NOTICE = (
    "Training-data preparation. Evaluation sources contributed passage IDs only (no grades, judgments or "
    "question texts); blocked passages and their overlap groups never become training examples or negatives."
)
OVERLAP_METHOD = (
    "word 5-gram shingles (case-folded [a-z0-9]+ words, BLAKE2b 64-bit hashes); candidates from an inverted index "
    "over shingles in at most max_shingle_df passages with >= min_shared_shingles shared; exact Jaccard and "
    "containment |A&B|/min(|A|,|B|) on full shingle sets; edges at containment >= containment_threshold or "
    "identical word sequences; groups = connected components"
)


class DataPrepError(ValueError):
    """The configuration, corpus or an existing preparation is inconsistent; nothing is written."""


class LeakageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_question_files: list[str] = Field(min_length=1)
    lexical_overlap_threshold: float = Field(default=0.8, gt=0, le=1)


class DataPrepConfig(BaseModel):
    """``configs/train/data_prep_<version>.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    corpus_path: str
    corpus_version: str
    corpus_sha256: str
    output_dir: str
    min_source_words: int = Field(default=40, ge=1)  # visible words a passage needs to give queries
    overlap: OverlapSettings = Field(default_factory=OverlapSettings)
    blocklist: BlocklistSources
    split: SplitSettings
    visible_text: VisibleTextSettings
    generation: GenerationSettings
    leakage: LeakageSettings
    # Sentence- and fact-level evidence reuse (manifest format 2); absent in format-1 preparations.
    reuse: ReuseSettings | None = None
    # Which approved passages are sampled for generation; not part of the preparation itself.
    source_quality: SourceQualitySettings = Field(default_factory=SourceQualitySettings)

    @field_validator("version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]+", v):
            raise ValueError("use lowercase letters, digits and underscores")
        return v

    @field_validator("output_dir")
    @classmethod
    def _output_dir(cls, v: str) -> str:
        parts = Path(v).parts
        if not any(
            a == "data" and b == "processed" for a, b in zip(parts, parts[1:], strict=False)
        ):
            raise ValueError(
                "generated training data must be written under data/processed/ (gitignored)"
            )
        if any(a == "data" and b == "eval" for a, b in zip(parts, parts[1:], strict=False)):
            raise ValueError("training data must never be written under data/eval/")
        return v


def load_data_prep_config(path: str | Path) -> tuple[DataPrepConfig, str]:
    file = Path(path)
    if not file.is_file():
        raise DataPrepError(f"config not found: {file}")
    try:
        return DataPrepConfig.model_validate(read_yaml(file)), sha256_file(file)
    except ValidationError as exc:
        raise DataPrepError(f"{file}: {exc}") from None


def verify_frozen_corpus(config: DataPrepConfig) -> CorpusIdentity:
    """The corpus checked against its manifest and the frozen SHA-256 and version in the config."""
    return verify_corpus(
        config.corpus_path,
        expected_sha256=config.corpus_sha256,
        expected_version=config.corpus_version,
    )


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for r in rows
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(
    config: DataPrepConfig,
    *,
    config_path: str,
    config_sha256: str,
    corpus: CorpusIdentity,
    passages: Mapping[str, Passage],
    tokenizer: Any,
) -> dict[str, bytes]:
    """Every preparation file (manifest included) as bytes; see the module docstring."""
    ordered = list(passages.values())
    pins, pins_record = None, None
    if config.split.pins is not None:
        try:
            pins, pins_record = load_split_pins(config.split.pins, corpus, passages)
        except SplitPinsError as exc:
            raise DataPrepError(str(exc)) from None
    groups = build_overlap_groups(ordered, config.overlap)
    blocklist = build_blocklist(config.blocklist, corpus=corpus, passages=passages, groups=groups)
    vt = config.visible_text
    spans = {
        p.passage_id: visible_span(
            p.text, tokenizer, max_seq_length=vt.max_seq_length, prefix=vt.passage_prefix
        )
        for p in ordered
    }
    reuse_files: dict[str, bytes] = {}
    reuse_manifest: dict[str, Any] | None = None
    if config.reuse is None:
        split = assign_splits(passages, groups, blocklist, config.split, pins=pins)
    else:
        index = ReuseIndex.build(
            ordered, {pid: span.chars for pid, span in spans.items()}, config.reuse
        )
        evaluation = set(blocklist.reasons)
        blocklist, reuse = expand_by_evidence_reuse(blocklist, index, groups, passages)
        free = [pid for pid in index.order if pid not in blocklist]
        links = {
            pair: r
            for pair, r in index.links(free).items()
            if groups.group_of[pair[0]] != groups.group_of[pair[1]]
        }
        split = assign_splits(passages, groups, blocklist, config.split, links=list(links), pins=pins)
        reuse_files, reuse_manifest = _reuse_outputs(
            index, evaluation, reuse, links, split.split_of, config.reuse
        )
    rows = []
    for p in ordered:
        span = spans[p.passage_id]
        rows.append(
            {
                "passage_id": p.passage_id,
                "split": split.split_of[p.passage_id],
                "group_id": groups.group_of[p.passage_id],
                "ticker": p.ticker,
                "fiscal_year": fiscal_year(p),
                "visible_chars": span.chars,
                "visible_tokens": span.tokens,
                "truncated": span.truncated,
                "visible_words": len(p.text[: span.chars].split()),
            }
        )
    usable = [r for r in rows if r["split"] != "blocked"]
    sources = {
        s: sum(
            1 for r in usable if r["split"] == s and r["visible_words"] >= config.min_source_words
        )
        for s in ("train", "validation")
    }
    files = {
        GROUPS_FILE: _jsonl(groups.group_rows()),
        EDGES_FILE: _jsonl([e.to_dict() for e in groups.edges]),
        BLOCKLIST_FILE: _jsonl(blocklist.rows([p.passage_id for p in ordered])),
        PASSAGES_FILE: _jsonl(rows),
        **reuse_files,
    }
    manifest = {
        "format_version": 1 if reuse_manifest is None else 2,
        "kind": "training_data_preparation",
        "version": config.version,
        "notice": NOTICE,
        "corpus": {
            "path": str(corpus.corpus_path),
            "sha256": corpus.corpus_sha256,
            "version": corpus.corpus_version,
            "passages": corpus.passages,
        },
        "config": {"path": config_path, "sha256": config_sha256},
        "overlap": {
            "method": OVERLAP_METHOD,
            "settings": config.overlap.model_dump(),
            "stats": groups.stats,
        },
        "blocklist": {
            "sources": list(blocklist.sources),
            "stats": blocklist.stats,
            "expanded_to": (
                "overlap_group" if reuse_manifest is None else "overlap_group+evidence_reuse"
            ),
        },
        "split": {
            "settings": config.split.manifest_settings(),
            "stats": split.stats,
            **({"pins": pins_record} if pins_record is not None else {}),
        },
        "visible_text": {
            "identity": tokenizer_identity(tokenizer, vt),
            "truncated": sum(r["truncated"] for r in rows),
            "truncated_share": (
                round(sum(r["truncated"] for r in rows) / len(rows), 4) if rows else 0.0
            ),
            "visible_tokens_median": (
                statistics.median(r["visible_tokens"] for r in rows) if rows else 0
            ),
            "visible_words_median": (
                statistics.median(r["visible_words"] for r in rows) if rows else 0
            ),
        },
        "generation_sources": {"min_source_words": config.min_source_words, **sources},
        "files": {name: _sha256(data) for name, data in files.items()},
    }
    if reuse_manifest is not None:
        manifest["reuse"] = reuse_manifest
    files[MANIFEST_FILE] = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    return files


def _reuse_outputs(
    index: ReuseIndex,
    evaluation: set[str],
    reuse: Mapping[str, Any],
    links: Mapping[tuple[str, str], Any],
    split_of: Mapping[str, str],
    settings: ReuseSettings,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """The reuse files and manifest section, after verifying the blocklist and split against them."""
    remaining = [pid for pid in index.order if split_of[pid] != "blocked"]
    if leaked := [pid for pid in remaining if reuse[pid].substantial(settings)]:
        raise DataPrepError(
            f"passages substantially reusing evaluation evidence are not blocked: {leaked[:3]}"
        )
    if crossing := [pair for pair in links if split_of[pair[0]] != split_of[pair[1]]]:
        raise DataPrepError(f"evidence-reuse links cross the split: {crossing[:3]}")

    def any_reuse(r: Any) -> bool:
        return bool(r.sentences or r.facts or r.span_words)

    residual = [pid for pid in remaining if any_reuse(reuse[pid])]
    by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    for pid in remaining:
        by_split[split_of[pid]].add(pid)
    cross = sum(
        any_reuse(r)
        for r in index.reuse_against(
            by_split["train"], by_split["validation"], full_text=False
        ).values()
    )
    reuse_rows = [
        {
            "passage_id": pid,
            "split": split_of[pid],
            "blocked_by_evidence_reuse": split_of[pid] == "blocked"
            and reuse[pid].substantial(settings),
            **reuse[pid].to_dict(),
        }
        for pid in index.order
        if pid in reuse and any_reuse(reuse[pid])
    ]
    link_rows = [
        {"passage_a": a, "passage_b": b, **{k: v for k, v in r.to_dict().items() if k != "sources"}}
        for (a, b), r in sorted(
            links.items(), key=lambda kv: (index.position[kv[0][0]], index.position[kv[0][1]])
        )
    ]
    files = {REUSE_FILE: _jsonl(reuse_rows), LINKS_FILE: _jsonl(link_rows)}
    section = {
        "method": REUSE_METHOD,
        "settings": settings.model_dump(),
        "index": {"digest": index.digest(), **index.stats()},
        "evaluation_passages": len(evaluation),
        "split_links": len(link_rows),
        "verification": {
            "substantial_evaluation_reuse_remaining": 0,
            "links_crossing_split": 0,
            "residual_passages_sharing_units_with_evaluation": len(residual),
            "residual_max_word_share": max((reuse[p].word_share for p in residual), default=0.0),
            "residual_train_passages_sharing_units_with_validation": cross,
            "note": "residual reuse below the passage threshold is enforced per pair: build-pairs rejects a pair "
            "whose evidence is held exactly by a blocked or other-split passage",
        },
    }
    return files, section


def prepared_status(directory: str | Path, files: Mapping[str, bytes]) -> str:
    """How ``files`` compare with what ``directory`` holds: ``absent``, ``identical``, ``config_edited``
    (every data file and recorded setting identical, only the config file's SHA-256 differs: the file was
    edited in sections the preparation does not use, such as generation settings) or ``different``.
    """
    directory = Path(directory)
    existing = {name: directory / name for name in files if (directory / name).exists()}
    if not existing:
        return "absent"
    if len(existing) != len(files):
        return "different"
    if all(p.read_bytes() == files[n] for n, p in existing.items()):
        return "identical"
    data_same = all(p.read_bytes() == files[n] for n, p in existing.items() if n != MANIFEST_FILE)
    old = json.loads(existing[MANIFEST_FILE].read_bytes())
    new = json.loads(files[MANIFEST_FILE])
    old["config"].pop("sha256", None)
    new["config"].pop("sha256", None)
    return "config_edited" if data_same and old == new else "different"


def write_prepared(directory: str | Path, files: Mapping[str, bytes]) -> bool:
    """Write the files; ``False`` if the same preparation exists (see ``prepared_status``, which treats a
    config edit outside the preparation's settings as the same). A different version is never overwritten.
    """
    directory = Path(directory)
    status = prepared_status(directory, files)
    if status in ("identical", "config_edited"):
        return False
    if status == "different":
        raise DataPrepError(f"{directory} already holds a different preparation; use a new version")
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(n for n in files if n != MANIFEST_FILE):
        (directory / name).write_bytes(files[name])
    (directory / MANIFEST_FILE).write_bytes(
        files[MANIFEST_FILE]
    )  # last: a manifest means a complete set
    return True


@dataclass(frozen=True)
class PreparedData:
    """A verified preparation: its manifest and one row per corpus passage."""

    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    rows: dict[str, dict[str, Any]]

    def split_of(self, passage_id: str) -> str:
        return self.rows[passage_id]["split"]

    def reuse_links(self) -> dict[str, frozenset[str]]:
        """Each passage's direct evidence-reuse links: passages it substantially reuses, or that reuse it,
        across overlap groups (``reuse_links.jsonl``; none in format 1)."""
        cached = self.__dict__.get("_links")
        if cached is None:
            linked: dict[str, set[str]] = {}
            path = self.directory / LINKS_FILE
            if self.manifest.get("format_version") == 2:
                for line in path.read_text(encoding="utf-8").splitlines():
                    link = json.loads(line)
                    linked.setdefault(link["passage_a"], set()).add(link["passage_b"])
                    linked.setdefault(link["passage_b"], set()).add(link["passage_a"])
            cached = {pid: frozenset(v) for pid, v in linked.items()}
            object.__setattr__(self, "_links", cached)
        return cached

    def split_units(self) -> dict[str, str]:
        """Each passage's split unit: its overlap group joined with groups linked by evidence reuse, named after
        the unit's first group (in format 1, its overlap group)."""
        cached = self.__dict__.get("_units")
        if cached is None:
            parent: dict[str, str] = {}

            def find(g: str) -> str:
                while parent.get(g, g) != g:
                    parent[g] = parent.get(parent[g], parent[g])
                    g = parent[g]
                return g

            for a, linked in self.reuse_links().items():
                for b in linked:
                    ra, rb = find(self.rows[a]["group_id"]), find(self.rows[b]["group_id"])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
            cached = {pid: find(row["group_id"]) for pid, row in self.rows.items()}
            object.__setattr__(self, "_units", cached)
        return cached

    @property
    def reuse_settings(self) -> ReuseSettings | None:
        section = self.manifest.get("reuse")
        return None if section is None else ReuseSettings.model_validate(section["settings"])

    def reuse_index(self, passages: Mapping[str, Passage]) -> ReuseIndex:
        """The evidence-reuse index, rebuilt and checked against the digest recorded at preparation.

        Fails closed: a format-1 preparation has no reuse checks, and an index that differs from the
        recorded one means the leakage checks would not test what the preparation verified.
        """
        settings = self.reuse_settings
        if settings is None:
            raise DataPrepError(
                f"{self.directory} was prepared without evidence-reuse checks (manifest format 1), so "
                "sentence- and fact-level leakage cannot be checked; prepare a newer version (a config with a "
                "reuse section, e.g. configs/train/data_prep_v2.yaml)"
            )
        index = ReuseIndex.build(
            list(passages.values()),
            {pid: row["visible_chars"] for pid, row in self.rows.items()},
            settings,
        )
        if index.digest() != self.manifest["reuse"]["index"]["digest"]:
            raise DataPrepError(
                f"the evidence-reuse index of {self.directory} differs from the one recorded at preparation; "
                "the leakage checks cannot be completed"
            )
        return index

    def source_passages(
        self, split: str, passages: Mapping[str, Passage], *, min_words: int
    ) -> list[SourcePassage]:
        """Approved passages of ``split`` with at least ``min_words`` visible words, in corpus order."""
        if split not in ("train", "validation"):
            raise DataPrepError(f"no generation from the {split!r} split")
        return [
            self.source_passage(pid, passages)
            for pid, row in self.rows.items()
            if row["split"] == split and row["visible_words"] >= min_words
        ]

    def source_passage(self, passage_id: str, passages: Mapping[str, Passage]) -> SourcePassage:
        """A passage as the generator sees it: its metadata and its visible text (train or validation)."""
        row = self.rows[passage_id]
        if row["split"] not in ("train", "validation"):
            raise DataPrepError(f"{passage_id} is {row['split']}; it never gives training examples")
        p = passages[passage_id]
        item = p.metadata.get("item_number")
        return SourcePassage(
            passage_id=passage_id,
            split=row["split"],
            group_id=row["group_id"],
            ticker=p.ticker,
            fiscal_year=row["fiscal_year"],
            company=str(p.metadata.get("company_name") or p.ticker),
            form_type=p.form_type,
            filing_date=p.filing_date,
            report_date=str(p.metadata.get("report_date") or ""),
            section=(f"Item {item} " if item else "") + (p.section_title or ""),
            visible_text=p.text[: row["visible_chars"]],
            item=str(item or ""),
        )


def open_prepared(
    directory: str | Path, corpus: CorpusIdentity, passages: Mapping[str, Passage]
) -> PreparedData:
    """Reload a preparation after checking its corpus binding, its file hashes and its rows."""
    directory = Path(directory)
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise DataPrepError(f"no preparation at {directory} (run prepare first)")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    recorded = manifest.get("corpus", {})
    if (recorded.get("sha256"), recorded.get("version")) != (
        corpus.corpus_sha256,
        corpus.corpus_version,
    ):
        raise DataPrepError(f"{path} was prepared from another corpus ({recorded.get('version')})")
    format_version = manifest.get("format_version")
    if format_version not in (1, 2):
        raise DataPrepError(f"{path} has unknown format_version {format_version!r}")
    for name in DATA_FILES if format_version == 1 else DATA_FILES_V2:
        if not (directory / name).is_file() or sha256_file(directory / name) != manifest[
            "files"
        ].get(name):
            raise DataPrepError(f"{directory / name} is missing or does not match {MANIFEST_FILE}")
    rows = {}
    for line in (directory / PASSAGES_FILE).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["passage_id"]] = row
    if list(rows) != list(passages):
        raise DataPrepError(
            f"{directory / PASSAGES_FILE} does not cover the corpus in corpus order"
        )
    blocked = {
        json.loads(line)["passage_id"]
        for line in (directory / BLOCKLIST_FILE).read_text(encoding="utf-8").splitlines()
    }
    if wrong := sorted(
        pid for pid, r in rows.items() if (pid in blocked) != (r["split"] == "blocked")
    ):
        raise DataPrepError(f"split and blocklist disagree for {wrong[:3]}")
    sides: dict[str, set[str]] = {}
    for r in rows.values():
        sides.setdefault(r["group_id"], set()).add(r["split"])
    if crossing := sorted(g for g, s in sides.items() if len(s) > 1):
        raise DataPrepError(f"overlap groups cross splits: {crossing[:3]}")
    if format_version == 2:
        for line in (directory / LINKS_FILE).read_text(encoding="utf-8").splitlines():
            link = json.loads(line)
            if rows[link["passage_a"]]["split"] != rows[link["passage_b"]]["split"]:
                raise DataPrepError(
                    f"an evidence-reuse link crosses the split: {link['passage_a']}"
                )
        for line in (directory / REUSE_FILE).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["blocked_by_evidence_reuse"] and rows[row["passage_id"]]["split"] != "blocked":
                raise DataPrepError(
                    f"{row['passage_id']} reuses evaluation evidence but is not blocked"
                )
    return PreparedData(directory, manifest, _sha256(raw), rows)


__all__ = [
    "BLOCKLIST_FILE",
    "DATA_FILES",
    "DATA_FILES_V2",
    "EDGES_FILE",
    "GROUPS_FILE",
    "LINKS_FILE",
    "MANIFEST_FILE",
    "PASSAGES_FILE",
    "REUSE_FILE",
    "DataPrepConfig",
    "DataPrepError",
    "LeakageSettings",
    "PreparedData",
    "load_data_prep_config",
    "open_prepared",
    "prepare",
    "prepared_status",
    "verify_frozen_corpus",
    "write_prepared",
]
