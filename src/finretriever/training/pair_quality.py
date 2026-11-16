"""Synthetic training pairs: built from generation records, checked, and recorded in a manifest.

``build_pairs`` reads each split's generation records (whose recorded run must match the current
prompt, model, parameters and preparation, request settings aside; each pair names the parameters
its record was generated with), keeps the accepted candidates in corpus order, and
applies the global checks the generator cannot: a query repeated anywhere (train first, then
validation) is rejected as ``duplicate_query``, and a query whose word-set Jaccard with any
evaluation question reaches the threshold is rejected as ``evaluation_overlap``. Evaluation
questions are read only for this comparison and never enter a pair.

Evidence-level checks (preparations with evidence-reuse checks, manifest format 2) apply
``finretriever.training.evidence_reuse`` to every pair's evidence: a pair whose evidence a
blocked passage holds exactly (the quote, a whole sentence or the answer's figure) is rejected as
``evaluation_evidence``, and one whose evidence the visible text of a passage in the other split
holds exactly is rejected as ``cross_split_evidence``. Every same-split passage whose visible text
holds the evidence is recorded with the pair (``equivalents``: ``exact`` and ``uncertain``), so a
trainer can keep it out of the pair's negatives (``finretriever.training.false_negatives``). A
preparation without these checks fails closed: no pair is built from it.

``check_pairs`` then re-verifies every pair from scratch and the build fails closed on any
violation: the positive exists and belongs to the pair's split; it is not blocked; its overlap
group lies wholly in that split; no (query, passage) pair or query repeats; the candidate checks
of ``finretriever.training.query_generation`` still pass (visible evidence, no invented
numbers, company named); no query overlaps an evaluation question; and the evidence checks above
still hold, with the same equivalents. The lexical check catches copies and light rewordings
only: a paraphrase of an evaluation question can still pass, so semantic leakage remains
possible (docs/training_data_v1.md).

``write_pairs`` writes the pairs once: a build that would change them is refused (see there).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finretriever.data.schemas import Passage, Query
from finretriever.eval_set.qrels_builder import find_gold_overlap
from finretriever.evaluation.provenance import sha256_file
from finretriever.training.data_prep import (
    BLOCKLIST_FILE,
    DataPrepConfig,
    DataPrepError,
    PreparedData,
)
from finretriever.training.evidence_reuse import Equivalent, EvidenceLocator
from finretriever.training.pair_contract import contract_problems, make_training_pair
from finretriever.training.query_generation import (
    LEGACY_VALIDATOR_VERSION,
    VALIDATOR_VERSION,
    CompanyAliases,
    GenerationError,
    SourcePassage,
    check_candidate,
    check_run,
    continuations_path,
    latest_records,
    prompt_sha256,
    question_key,
    read_records,
    records_path,
    run_path,
)

PAIRS_DIRNAME = "pairs"
GENERATION_DIRNAME = "generation"
PAIR_FILES = {"train": "train_pairs.jsonl", "validation": "validation_pairs.jsonl"}
PAIRS_META_FILE = "pairs.meta.json"
SPLITS = ("train", "validation")
NOTICE = (
    "Synthetic query-passage pairs written by an LLM from approved training passages only. They are training and "
    "validation data for the bi-encoder, never evaluation data, and contain no evaluation question or label."
)
SEMANTIC_LEAKAGE_NOTE = (
    "Leakage against evaluation questions is checked lexically (word-set Jaccard); paraphrases that change the "
    "wording can pass, so semantic leakage is not ruled out. Evaluation passages are excluded separately by the "
    "blocklist."
)


class PairQualityError(ValueError):
    """Pairs cannot be built, or a built pair breaks a rule."""


def load_evaluation_questions(paths: Sequence[str]) -> list[Query]:
    """Every evaluation question text (``query_id`` and ``text`` of each JSONL row); fails closed if a file is missing."""
    questions = []
    for path in map(Path, paths):
        if not path.is_file():
            raise PairQualityError(
                f"evaluation question file {path} is missing; the leakage check fails closed"
            )
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                    questions.append(
                        Query(
                            query_id=f"{path}:{row['query_id']}",
                            text=row["text"],
                            source="evaluation",
                        )
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    raise PairQualityError(
                        f"{path}:{line_no} is not a question row ({exc!r})"
                    ) from None
    return questions


def aliases_for(
    passages: Mapping[str, Passage], extra: Mapping[str, Sequence[str]]
) -> CompanyAliases:
    companies = {}
    for p in passages.values():
        companies.setdefault(p.ticker, str(p.metadata.get("company_name") or p.ticker))
    return CompanyAliases.from_companies(companies, extra)


@dataclass
class PairSet:
    pairs: dict[str, list[dict[str, Any]]]
    stats: dict[str, Any]
    runs: dict[str, dict[str, str]] = field(default_factory=dict)
    evidence_index_digest: str = ""


def evidence_locator(prepared: PreparedData, passages: Mapping[str, Passage]) -> EvidenceLocator:
    """The evidence locator of a preparation; fails closed if its leakage checks cannot run."""
    try:
        return EvidenceLocator(passages, prepared.reuse_index(passages))
    except DataPrepError as exc:
        raise PairQualityError(str(exc)) from None


def evidence_leak(
    equivalents: Sequence[Equivalent], rows: Mapping[str, Mapping[str, Any]], split: str
) -> str | None:
    """``evaluation_evidence`` or ``cross_split_evidence`` if another passage holds the evidence
    exactly where it must not (a blocked passage's text, another split's visible text), else None.
    """
    if any(e.exact and rows[e.passage_id]["split"] == "blocked" for e in equivalents):
        return "evaluation_evidence"
    if any(
        e.visible_exact and rows[e.passage_id]["split"] not in (split, "blocked")
        for e in equivalents
    ):
        return "cross_split_evidence"
    return None


def equivalent_ids(
    equivalents: Sequence[Equivalent], rows: Mapping[str, Mapping[str, Any]], split: str
) -> dict[str, list[str]]:
    """Same-split passages whose visible text holds the evidence, exactly or uncertainly."""
    same = [
        e
        for e in equivalents
        if rows[e.passage_id]["split"] == split and e.visible_match is not None
    ]
    return {
        "exact": [e.passage_id for e in same if e.visible_exact],
        "uncertain": [e.passage_id for e in same if not e.visible_exact],
    }


def _max_similarity(questions: Sequence[Query], text: str) -> float:
    found = find_gold_overlap(questions, [text], threshold=0.0)
    return max((o.similarity for o in found), default=0.0)


def build_pairs(
    config: DataPrepConfig,
    prepared: PreparedData,
    passages: Mapping[str, Passage],
    evaluation_questions: Sequence[Query],
) -> PairSet:
    """Pairs for both splits from the generation records; see the module docstring."""
    settings = config.generation
    aliases = aliases_for(passages, settings.company_aliases)
    locator = evidence_locator(prepared, passages)  # before anything else: fail closed
    generation_dir = prepared.directory / GENERATION_DIRNAME
    threshold = config.leakage.lexical_overlap_threshold
    pairs: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    rejected: Counter[str] = Counter()
    per_split: dict[str, dict[str, int]] = {}
    runs: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for split in SPLITS:
        records_file, run_file = records_path(generation_dir, split), run_path(
            generation_dir, split
        )
        try:
            run = check_run(generation_dir, split, settings, prepared.manifest_sha256)
        except GenerationError as exc:
            raise PairQualityError(str(exc)) from None
        if run.state == "new":
            per_split[split] = {
                "passages_done": 0,
                "passages_failed": 0,
                "proposed": 0,
                "accepted_by_generator": 0,
                "done_by_validator": {},
            }
            continue
        bound = [b["params_sha256"] for b in run.bindings]  # the run, then each continuation
        continuations = continuations_path(generation_dir, split)
        runs[split] = {
            "run_sha256": sha256_file(run_file),
            "continuations_sha256": sha256_file(continuations) if continuations.exists() else "",
            "records_sha256": sha256_file(records_file) if records_file.exists() else "",
            "params_sha256": bound,
        }
        state = latest_records(read_records(records_file))
        sources = {
            p.passage_id: p
            for p in prepared.source_passages(split, passages, min_words=config.min_source_words)
        }
        done = failed = proposed = generated = 0
        validators: Counter[str] = (
            Counter()
        )  # an older one means `revalidate` may recover rejections
        for pid in prepared.rows:
            record = state.get(pid)
            if record is None:
                continue
            if pid not in sources:
                raise PairQualityError(
                    f"generation record for {pid}, which is not an approved {split} passage"
                )
            if record["status"] != "done":
                failed += 1
                continue
            params = record.get(
                "params_sha256", bound[0]
            )  # records before format 3 predate any continuation
            if params not in bound:
                raise PairQualityError(
                    f"generation record for {pid} names parameters {params} that the run never recorded"
                )
            done += 1
            validators[record.get("validator_version", LEGACY_VALIDATOR_VERSION)] += 1
            proposed += record["proposed"]
            rejected.update(r["reason"] for r in record["rejected"])
            for raw in record["accepted"]:
                generated += 1
                candidate, reason = check_candidate(raw, sources[pid], settings, aliases)
                if candidate is None:
                    rejected[f"recheck_{reason}"] += 1
                    continue
                key = question_key(candidate.question)
                if key in seen:
                    rejected["duplicate_query"] += 1
                    continue
                similarity = _max_similarity(evaluation_questions, candidate.question)
                if similarity >= threshold:
                    rejected["evaluation_overlap"] += 1
                    continue
                row = prepared.rows[pid]
                equivalents = locator.locate(
                    candidate.evidence, candidate.answer, row["ticker"], pid
                )
                if leak := evidence_leak(equivalents, prepared.rows, split):
                    rejected[leak] += 1
                    continue
                seen.add(key)
                version = record.get("prompt_version", settings.prompt_version)
                pairs[split].append(
                    make_training_pair(
                        pair_id=f"{split}_{len(pairs[split]) + 1:06d}",
                        split=split,
                        candidate=candidate.to_dict(),
                        positive=pid,
                        prepared=prepared,
                        passages=passages,
                        similarity=similarity,
                        equivalents=equivalent_ids(equivalents, prepared.rows, split),
                        provider=settings.provider,
                        model=record.get("model", settings.model),
                        prompt_version=version,
                        prompt_sha=prompt_sha256(version),
                        validator_version=VALIDATOR_VERSION,
                        params_sha256=params,
                        origin="build-pairs",
                    )
                )
        per_split[split] = {
            "passages_done": done,
            "passages_failed": failed,
            "proposed": proposed,
            "accepted_by_generator": generated,
            "done_by_validator": dict(sorted(validators.items())),
        }
    violations = check_pairs(
        pairs, config, prepared, passages, evaluation_questions, locator=locator
    )
    if violations:
        raise PairQualityError(
            f"{len(violations)} pair violations, nothing written: {violations[:5]}"
        )
    stats = {
        "by_split": {s: {**per_split[s], "pairs": len(pairs[s])} for s in SPLITS},
        "rejected_by_reason": dict(sorted(rejected.items())),
        "question_types": {
            s: dict(sorted(Counter(p["question_type"] for p in pairs[s]).items())) for s in SPLITS
        },
        "by_ticker": {
            s: dict(sorted(Counter(p["ticker"] for p in pairs[s]).items())) for s in SPLITS
        },
        "by_fiscal_year": {
            s: dict(sorted(Counter(p["fiscal_year"] for p in pairs[s]).items())) for s in SPLITS
        },
        "max_evaluation_similarity": max(
            (p["max_evaluation_similarity"] for s in SPLITS for p in pairs[s]), default=0.0
        ),
        "pairs_with_evaluation_similarity_over_0_5": sum(
            p["max_evaluation_similarity"] > 0.5 for s in SPLITS for p in pairs[s]
        ),
    }
    return PairSet(pairs, stats, runs, locator.index.digest())


def check_pairs(
    pairs: Mapping[str, Sequence[Mapping[str, Any]]],
    config: DataPrepConfig,
    prepared: PreparedData,
    passages: Mapping[str, Passage],
    evaluation_questions: Sequence[Query],
    *,
    locator: EvidenceLocator | None = None,
) -> list[str]:
    """Every rule broken by ``pairs`` (empty when they are clean); see the module docstring."""
    settings = config.generation
    aliases = aliases_for(passages, settings.company_aliases)
    if locator is None:
        try:
            locator = evidence_locator(prepared, passages)
        except PairQualityError as exc:
            return [f"evidence leakage checks unavailable: {exc}"]
    blocked = {
        json.loads(line)["passage_id"]
        for line in (prepared.directory / BLOCKLIST_FILE).read_text(encoding="utf-8").splitlines()
    }
    group_sides: dict[str, set[str]] = {}
    for row in prepared.rows.values():
        group_sides.setdefault(row["group_id"], set()).add(row["split"])
    sources = {
        s: {p.passage_id: p for p in prepared.source_passages(s, passages, min_words=1)}
        for s in SPLITS
    }
    violations: list[str] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    seen_queries: set[str] = set()
    for split in SPLITS:
        for pair in pairs.get(split, ()):
            pid, pair_id = pair.get("positive_passage_id"), pair.get("pair_id")
            where = f"{pair_id} ({split})"
            if pair.get("split") != split:
                violations.append(f"{where}: recorded split {pair.get('split')!r}")
            if pair_id in seen_ids:
                violations.append(f"{where}: duplicate pair_id")
            seen_ids.add(pair_id)
            if pid not in passages:
                violations.append(f"{where}: positive {pid} is not in the corpus")
                continue
            row = prepared.rows[pid]
            if pid in blocked or row["split"] == "blocked":
                violations.append(f"{where}: positive {pid} is on the evaluation blocklist")
            if row["split"] != split:
                violations.append(f"{where}: positive {pid} belongs to {row['split']}")
            if pair.get("group_id") != row["group_id"] or group_sides[row["group_id"]] != {split}:
                violations.append(
                    f"{where}: overlap group {row['group_id']} is not wholly in {split}"
                )
            key = question_key(str(pair.get("query", "")))
            if (key, pid) in seen_pairs:
                violations.append(f"{where}: duplicate query-passage pair")
            elif key in seen_queries:
                violations.append(f"{where}: query repeats another pair's query")
            seen_pairs.add((key, pid))
            seen_queries.add(key)
            source: SourcePassage | None = sources[split].get(pid)
            if source is None:
                violations.append(f"{where}: positive {pid} is not an approved {split} passage")
                continue
            raw = {
                "question": pair.get("query"),
                "type": pair.get("question_type"),
                "answer": pair.get("answer"),
                "evidence": pair.get("evidence"),
            }
            _, reason = check_candidate(raw, source, settings, aliases)
            if reason is not None:
                violations.append(f"{where}: fails the candidate check ({reason})")
            equivalents = locator.locate(
                str(pair.get("evidence", "")), str(pair.get("answer", "")), row["ticker"], pid
            )
            if leak := evidence_leak(equivalents, prepared.rows, split):
                violations.append(f"{where}: {leak.replace('_', ' ')} (another passage holds it)")
            recorded = pair.get("equivalents") or {}
            expected = equivalent_ids(equivalents, prepared.rows, split)
            if {k: recorded.get(k) for k in ("exact", "uncertain")} != expected:
                violations.append(f"{where}: equivalent passages do not match its evidence")
            violations.extend(contract_problems(pair, prepared, passages))
            if find_gold_overlap(
                evaluation_questions,
                [str(pair.get("query", ""))],
                threshold=config.leakage.lexical_overlap_threshold,
            ):
                violations.append(f"{where}: query overlaps an evaluation question")
    return violations


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for r in rows
    ).encode()


def write_pairs(
    pair_set: PairSet,
    config: DataPrepConfig,
    prepared: PreparedData,
    *,
    config_path: str,
    config_sha256: str,
) -> tuple[bool, dict[str, Path]]:
    """Write both pair files and ``pairs.meta.json`` (no timestamps) under ``<preparation>/pairs/``;
    False if an identical copy is present.

    Written once: pairs that differ from those present (new records, a changed check, an edited
    config) are refused and nothing is written, because quality decisions name pairs by ID and
    candidate datasets record these files' SHA-256 (a rebuild renumbers the pairs after any
    rejected one). Pairs from new generation records belong in a new pair set.
    """
    directory = prepared.directory / PAIRS_DIRNAME
    data = {name: _jsonl(pair_set.pairs[split]) for split, name in PAIR_FILES.items()}
    question_files = []
    for path in config.leakage.evaluation_question_files:
        rows = [
            line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        question_files.append({"path": path, "sha256": sha256_file(path), "questions": len(rows)})
    manifest = prepared.manifest
    meta = {
        "format_version": 2,
        "kind": "synthetic_training_pairs",
        "version": config.version,
        "notice": NOTICE,
        "corpus": manifest["corpus"],
        "config": {"path": config_path, "sha256": config_sha256},
        "preparation": {
            "manifest_sha256": prepared.manifest_sha256,
            "blocklist_sha256": manifest["files"][BLOCKLIST_FILE],
            "blocklist_stats": manifest["blocklist"]["stats"],
            "split": manifest["split"]["settings"],
            "overlap": manifest["overlap"]["settings"],
            "visible_text": manifest["visible_text"]["identity"],
        },
        "generation": {
            "provider": config.generation.provider,
            "model": config.generation.model,
            "prompt_version": config.generation.prompt_version,
            "prompt_sha256": prompt_sha256(config.generation.prompt_version),
            "params": config.generation.params(),
            "params_sha256": config.generation.params_sha256(),
            "runs": pair_set.runs,
        },
        "leakage": {
            "method": "word-set Jaccard against every evaluation question",
            "threshold": config.leakage.lexical_overlap_threshold,
            "question_files": question_files,
            "note": SEMANTIC_LEAKAGE_NOTE,
            "evidence": {
                "method": "evidence located in other passages (quote, sentence or the answer's figure under its label, compared by amount: exact; other "
                "figure, answer value or 8-word span: uncertain); exact in a blocked passage or in another "
                "split's visible text rejects the pair",
                "reuse_settings": manifest["reuse"]["settings"],
                "index_digest": pair_set.evidence_index_digest,
            },
        },
        "stats": pair_set.stats,
        "quality_checks": {"passed": True, "violations": 0},
        "files": {name: hashlib.sha256(blob).hexdigest() for name, blob in data.items()},
    }
    blobs = {
        **data,
        PAIRS_META_FILE: (json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(),
    }
    paths = {name: directory / name for name in blobs}
    present = [name for name, path in paths.items() if path.exists()]
    if present:
        changed = [name for name in blobs if not paths[name].is_file() or paths[name].read_bytes() != blobs[name]]
        if not changed:
            return False, paths
        raise PairQualityError(
            f"{directory} already holds different pairs ({', '.join(changed)} would change); written pairs are never "
            "replaced, because quality decisions and candidate datasets are bound to them, so nothing was written"
        )
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(data):
        paths[name].write_bytes(blobs[name])
    paths[PAIRS_META_FILE].write_bytes(blobs[PAIRS_META_FILE])  # last: a manifest means a complete set
    return True, paths


def read_pairs(prepared: PreparedData) -> dict[str, list[dict[str, Any]]]:
    """The written pairs, after checking them against ``pairs.meta.json``."""
    directory = prepared.directory / PAIRS_DIRNAME
    meta_path = directory / PAIRS_META_FILE
    if not meta_path.is_file():
        raise PairQualityError(f"no pairs at {directory} (run build-pairs first)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["preparation"]["manifest_sha256"] != prepared.manifest_sha256:
        raise PairQualityError(f"{meta_path} was built from a different preparation")
    out = {}
    for split, name in PAIR_FILES.items():
        path = directory / name
        if sha256_file(path) != meta["files"][name]:
            raise PairQualityError(f"{path} does not match {PAIRS_META_FILE}")
        out[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return out


__all__ = [
    "GENERATION_DIRNAME",
    "PAIRS_DIRNAME",
    "PAIRS_META_FILE",
    "PAIR_FILES",
    "PairQualityError",
    "PairSet",
    "aliases_for",
    "build_pairs",
    "check_pairs",
    "load_evaluation_questions",
    "read_pairs",
    "write_pairs",
]
