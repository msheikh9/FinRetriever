"""Expanded development pools: stored runs' top results added to a judged pool, and their labels.

The judged pools of the pilot came mostly from BM25, so passages a dense retriever ranks
highly were often never judged and counted as non-relevant. Pool deepening (methodology §7)
adds the top results of the stored zero-shot runs. It happens in three derived steps, each
pinned to its inputs by SHA-256 and never edited by hand:

1. ``expand_pool`` writes ``<version>/pool/``:

   * ``pool.jsonl`` — one ``DevPoolEntry`` per (question, passage): every passage the base
     label set's pool held, plus every passage a stored run ranked within ``depth``; with the
     original candidate sources, every run that ranked it (run, mode, rank, score), and the
     label version that judged it (``judged_in``, ``null`` when unjudged);
   * ``pool.meta.json`` — ``DevPoolInfo``: corpus binding, every input with its SHA-256 (base
     labels, annotation pool, each run file and its metrics), the depth, coverage statistics
     and the SHA-256 of ``pool.jsonl``.

   Adding a passage to a pool is not a judgment: a new entry is unjudged until a label set
   grades it, and until then it is absent from the qrels (never grade 0).

2. AI proposals for the unjudged entries go into their own directory (``PoolProposalSet``,
   the same records as ``finretriever.eval_set.ai_proposals``). A passage the base labels
   already judged can never be proposed for, so earlier labels are never re-graded.

3. ``derive_expanded_labels`` writes a development label set (the files of
   ``finretriever.eval_set.dev_labels``): the base set's labels, changes, flags and audit
   dispositions carried over unchanged, plus the proposals for new entries under the same
   rubric rules (P2 deterministic changes; P1, low-confidence and curated flags). Entries
   without a proposal stay unjudged. ``export-dev`` exports it like any other label set.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from finretriever.data.schemas import Passage
from finretriever.eval_set.ai_proposals import PROPOSALS_INFO_FILE, ProposalError, ProposalSet
from finretriever.eval_set.annotation_store import utc_now
from finretriever.eval_set.dev_labels import (
    DATA_FILES,
    DEV_STATUS,
    LABELS_INFO_FILE,
    P1_FLAG_REASON,
    BasisRule,
    DevLabel,
    DevLabelError,
    DevLabelSet,
    DevLabelSetInfo,
    GradeChange,
    LabelFlag,
    NonEmptyStr,
    RubricDecisions,
    SpecFlag,
    make_flag,
    p2_change,
    validate_label_set,
)
from finretriever.eval_set.schemas import (
    CONFIDENCE_LEVELS,
    RELEVANCE_GRADES,
    AIProposal,
    CandidateSource,
    Confidence,
    GoldQuestion,
    PoolEntry,
)
from finretriever.evaluation.provenance import (
    CorpusIdentity,
    EvaluationMetadata,
    check_annotations_match_corpus,
    sha256_file,
)
from finretriever.evaluation.stored_runs import StoredRun
from finretriever.utils.io import read_yaml

POOL_DIRNAME = "pool"
PROPOSALS_DIRNAME = "proposals"
POOL_FILE = "pool.jsonl"
POOL_INFO_FILE = "pool.meta.json"
POOL_NOTICE = (
    "Expanded development pool. Adding a passage to a pool is not a judgment: an entry with "
    "judged_in null is unjudged and stays out of the qrels until a label set grades it. Entries "
    "proposed by stored zero-shot runs are for development pools only, never for the gold "
    "annotation workspace."
)

_STRICT = ConfigDict(extra="forbid")
_VERSION_RE = re.compile(r"^[a-z0-9_]+$")


class DevPoolError(ValueError):
    """An expanded pool, its spec or its inputs break the derivation rules."""


def _explain(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render_jsonl(rows: Iterable[BaseModel]) -> bytes:
    return "".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def _render_json(model: BaseModel) -> bytes:
    return (json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _check_version(value: str) -> str:
    if not _VERSION_RE.match(value):
        raise ValueError("use lowercase letters, digits and underscores (e.g. pilot_v3)")
    return value


def _load_spec(path: str | Path, model: type[BaseModel], error: type[ValueError]) -> tuple[Any, str]:
    file = Path(path)
    if not file.is_file():
        raise error(f"spec not found: {file}")
    try:
        return model.model_validate(read_yaml(file)), sha256_file(file)
    except ValidationError as exc:
        raise error(f"{file}: {_explain(exc)}") from None


# --- pool records -------------------------------------------------------------------------


class RunRef(BaseModel):
    """A stored run to pool from: ``<runs_dir>/<mode>/<run>/``."""

    model_config = _STRICT

    run: NonEmptyStr
    mode: NonEmptyStr


class DevPoolSpec(BaseModel):
    """How an expanded pool is derived (``configs/eval/dev_pool_<version>.yaml``)."""

    model_config = _STRICT

    pool_version: NonEmptyStr
    annotation_version: NonEmptyStr
    description: NonEmptyStr
    base_labels_dir: NonEmptyStr  # its labelled passages are kept; its SHA-256 is pinned
    base_pool_path: NonEmptyStr  # the annotation pool the base labels came from: original sources
    runs_dir: NonEmptyStr
    runs: list[RunRef] = Field(min_length=1)
    depth: int = Field(default=10, ge=1)
    output_dir: NonEmptyStr

    @field_validator("pool_version")
    @classmethod
    def _version(cls, v: str) -> str:
        return _check_version(v)


def load_dev_pool_spec(path: str | Path) -> tuple[DevPoolSpec, str]:
    """The pool spec at ``path`` and the SHA-256 of its bytes."""
    return _load_spec(path, DevPoolSpec, DevPoolError)


class RunSource(BaseModel):
    """A stored run that ranked the passage within the pool depth."""

    model_config = _STRICT

    run: NonEmptyStr
    mode: NonEmptyStr
    rank: int = Field(ge=1)
    score: float


class DevPoolEntry(BaseModel):
    """A passage pooled for one question, with every way it entered and whether it is judged."""

    model_config = _STRICT

    query_id: NonEmptyStr
    passage_id: NonEmptyStr
    position: int = Field(ge=1)
    ticker: NonEmptyStr
    in_scope: bool  # its company is one the question asks about
    origin: Literal["base_pool", "run_expansion"]
    base_position: int | None = Field(default=None, ge=1)
    base_sources: list[CandidateSource] = Field(default_factory=list)
    run_sources: list[RunSource] = Field(default_factory=list)
    judged_in: str | None  # the label version that judged it; None: unjudged


class DevPoolInfo(BaseModel):
    """``pool.meta.json``: what an expanded pool is, where it came from, and its file."""

    model_config = _STRICT

    format_version: Literal[1] = 1
    kind: Literal["expanded_development_pool"] = "expanded_development_pool"
    status: Literal["provisional_development"] = DEV_STATUS
    pool_version: NonEmptyStr
    description: NonEmptyStr
    evaluation: EvaluationMetadata
    notice: NonEmptyStr = POOL_NOTICE
    spec: dict[str, str]
    base: dict[str, str]
    runs: list[dict[str, Any]]
    depth: int = Field(ge=1)
    coverage: dict[str, Any]
    files: dict[str, str]


def read_annotation_pool(path: str | Path) -> list[PoolEntry]:
    """The ``PoolEntry`` records of an annotation workspace's ``pool.jsonl``."""
    entries = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                entries.append(PoolEntry.model_validate_json(line))
            except ValidationError as exc:
                raise DevPoolError(f"{path}:{line_no}: {_explain(exc)}") from None
    return entries


@dataclass(frozen=True)
class DevPool:
    """An expanded pool in memory: its info and its entries, in (question, position) order."""

    directory: Path
    info: DevPoolInfo
    entries: tuple[DevPoolEntry, ...]

    def entry(self, query_id: str, passage_id: str) -> DevPoolEntry | None:
        return next((e for e in self.entries if e.query_id == query_id and e.passage_id == passage_id), None)

    def pool(self, query_id: str) -> list[DevPoolEntry]:
        return [e for e in self.entries if e.query_id == query_id]

    def unjudged(self, query_id: str | None = None) -> list[DevPoolEntry]:
        return [e for e in self.entries if e.judged_in is None and query_id in (None, e.query_id)]

    def rendered(self) -> dict[str, bytes]:
        return {POOL_FILE: _render_jsonl(self.entries), POOL_INFO_FILE: _render_json(self.info)}

    def write(self) -> bool:
        """Write to ``directory``; ``False`` if an identical copy is there. Never overwrites a different pool."""
        files = self.rendered()
        existing = {name: self.directory / name for name in files if (self.directory / name).exists()}
        if existing:
            if len(existing) == len(files) and all(p.read_bytes() == files[n] for n, p in existing.items()):
                return False
            raise DevPoolError(
                f"{self.directory} already holds a different pool; pool versions are immutable, so derive a "
                "new version instead"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (self.directory / name).write_bytes(data)
        return True

    @classmethod
    def open(cls, directory: str | Path, corpus: CorpusIdentity, passages: Mapping[str, Passage]) -> DevPool:
        """Load a pool after checking its corpus binding, its file hash and its entries."""
        directory = Path(directory)
        info_path, pool_path = directory / POOL_INFO_FILE, directory / POOL_FILE
        if not info_path.is_file() or not pool_path.is_file():
            raise DevPoolError(f"no expanded pool at {directory}")
        try:
            info = DevPoolInfo.model_validate_json(info_path.read_bytes())
        except ValidationError as exc:
            raise DevPoolError(f"{info_path}: {_explain(exc)}") from None
        check_annotations_match_corpus(info.evaluation, corpus)
        raw = pool_path.read_bytes()
        if _sha256(raw) != info.files.get(POOL_FILE):
            raise DevPoolError(f"{pool_path} does not match the SHA-256 in {POOL_INFO_FILE}; it was modified")
        entries = []
        for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            try:
                entries.append(DevPoolEntry.model_validate_json(line))
            except ValidationError as exc:
                raise DevPoolError(f"{POOL_FILE}:{line_no}: {_explain(exc)}") from None
        pool = cls(directory, info, tuple(entries))
        validate_pool(pool, passages)
        return pool


def validate_pool(pool: DevPool, passages: Mapping[str, Passage]) -> None:
    """Raise ``DevPoolError`` unless every entry is unique, in the corpus and has a recorded source."""
    keys = Counter((e.query_id, e.passage_id) for e in pool.entries)
    if duplicates := sorted(key for key, n in keys.items() if n > 1):
        raise DevPoolError(f"pooled more than once: {duplicates[:5]}")
    positions: dict[str, list[int]] = {}
    for entry in pool.entries:
        key = (entry.query_id, entry.passage_id)
        passage = passages.get(entry.passage_id)
        if passage is None:
            raise DevPoolError(f"pool entry {key}: passage not in the corpus")
        if passage.ticker != entry.ticker:
            raise DevPoolError(f"pool entry {key}: ticker {entry.ticker}, the corpus says {passage.ticker}")
        if entry.origin == "base_pool" and not entry.base_sources:
            raise DevPoolError(f"pool entry {key}: from the base pool but without its candidate sources")
        if entry.origin == "run_expansion" and (not entry.run_sources or entry.judged_in is not None):
            raise DevPoolError(f"pool entry {key}: an expansion entry needs a run source and no judgment")
        positions.setdefault(entry.query_id, []).append(entry.position)
    for query_id, seen in positions.items():
        if seen != list(range(1, len(seen) + 1)):
            raise DevPoolError(f"pool positions of {query_id} must run 1, 2, 3, ... in file order")


# --- expansion ---------------------------------------------------------------------------


def _systems(run_sources: Iterable[RunSource]) -> str:
    return "+".join(sorted({s.run for s in run_sources}))


def _retriever_summary(stored: StoredRun) -> dict[str, Any]:
    retriever = stored.metrics.get("retriever") or {}
    keys = ("type", "tokenizer", "backend", "model_name", "model_revision", "embeddings_sha256")
    summary = {key: retriever[key] for key in keys if retriever.get(key) is not None}
    if cache := retriever.get("cache"):
        summary["cache_key_sha256"] = cache.get("key_sha256")
    return summary


def expand_pool(
    spec: DevPoolSpec,
    *,
    spec_path: str,
    spec_sha256: str,
    base: DevLabelSet,
    base_pool: Sequence[PoolEntry],
    runs: Sequence[StoredRun],
    passages: Mapping[str, Passage],
) -> DevPool:
    """The pool ``spec`` describes: the base pool plus every passage a run ranked within ``depth``.

    Every passage is checked against the corpus. A stored run first scored with the base
    labels must report the judged@k those labels give, or the inputs do not belong together.
    """
    depth = spec.depth
    questions = sorted(base.questions, key=lambda q: q.query_id)
    question_ids = {q.query_id for q in questions}
    tickers = {q.query_id: set(q.tickers) for q in questions}
    judged = {(x.query_id, x.passage_id) for x in base.labels}

    base_entries: dict[tuple[str, str], PoolEntry] = {}
    for entry in base_pool:
        if entry.query_id not in question_ids:
            continue  # the workspace may hold questions the base label set does not cover
        if entry.passage_id not in passages:
            raise DevPoolError(f"base pool entry {(entry.query_id, entry.passage_id)}: passage not in the corpus")
        base_entries[entry.query_id, entry.passage_id] = entry
    if stray := sorted(judged - base_entries.keys()):
        raise DevPoolError(f"base labels for passages the base pool never held: {stray[:5]}")

    expected = [(ref.mode, ref.run) for ref in spec.runs]
    if [(r.mode, r.run) for r in runs] != expected:
        raise DevPoolError(f"runs {[(r.mode, r.run) for r in runs]} differ from the spec's {expected}")
    if len(set(expected)) != len(expected):
        raise DevPoolError(f"the spec lists a run twice: {expected}")
    run_sources: dict[tuple[str, str], list[RunSource]] = {}
    for stored in runs:
        ranked = stored.ranked(depth)
        if missing := sorted(question_ids - ranked.keys()):
            raise DevPoolError(f"stored run {stored.run_id} has no results for {missing}")
        for query_id, hits in sorted(ranked.items()):
            if query_id not in question_ids:
                raise DevPoolError(f"stored run {stored.run_id} ranks unknown question {query_id}")
            for hit in hits:
                if hit.passage_id not in passages:
                    raise DevPoolError(f"stored run {stored.run_id}: {hit.passage_id} is not in the corpus")
                run_sources.setdefault((query_id, hit.passage_id), []).append(
                    RunSource(run=stored.run, mode=stored.mode, rank=hit.rank, score=hit.score)
                )
        _check_judged_at_k(stored, ranked, judged, base.info.label_version, depth)

    entries: list[DevPoolEntry] = []
    for question in questions:
        qid = question.query_id
        old = sorted((e for k, e in base_entries.items() if k[0] == qid), key=lambda e: e.position)
        new = sorted(
            (pid for (q, pid) in run_sources if q == qid and (q, pid) not in base_entries),
            key=lambda pid: (min(s.rank for s in run_sources[qid, pid]), pid),
        )
        rows = [(e.passage_id, "base_pool", e) for e in old] + [(pid, "run_expansion", None) for pid in new]
        for position, (pid, origin, entry) in enumerate(rows, start=1):
            passage = passages[pid]
            entries.append(
                DevPoolEntry(
                    query_id=qid,
                    passage_id=pid,
                    position=position,
                    ticker=passage.ticker,
                    in_scope=not tickers[qid] or passage.ticker in tickers[qid],
                    origin=origin,
                    base_position=None if entry is None else entry.position,
                    base_sources=[] if entry is None else list(entry.sources),
                    run_sources=run_sources.get((qid, pid), []),
                    judged_in=base.info.label_version if (qid, pid) in judged else None,
                )
            )

    base_dir = Path(spec.base_labels_dir)
    draft = DevPoolInfo(
        pool_version=spec.pool_version,
        description=spec.description,
        evaluation=EvaluationMetadata(
            corpus_sha256=base.info.evaluation.corpus_sha256,
            corpus_version=base.info.evaluation.corpus_version,
            annotation_version=spec.annotation_version,
        ),
        spec={"path": spec_path, "sha256": spec_sha256},
        base={
            "labels_dir": str(base_dir),
            "label_version": base.info.label_version,
            "labels_meta_sha256": sha256_file(base_dir / LABELS_INFO_FILE),
            "pool_path": spec.base_pool_path,
            "pool_sha256": sha256_file(spec.base_pool_path),
        },
        runs=[{**stored.provenance(), "retriever": _retriever_summary(stored)} for stored in runs],
        depth=depth,
        coverage=pool_coverage(entries, runs, depth=depth),
        files={},
    )
    files = {POOL_FILE: _sha256(_render_jsonl(entries))}
    pool = DevPool(Path(spec.output_dir), draft.model_copy(update={"files": files}), tuple(entries))
    validate_pool(pool, passages)
    return pool


def _check_judged_at_k(
    stored: StoredRun,
    ranked: Mapping[str, list[Any]],
    judged: set[tuple[str, str]],
    label_version: str,
    depth: int,
) -> None:
    """A run scored with the base labels must report the judged@k they give for every question."""
    settings = stored.metrics.get("settings") or {}
    if stored.metrics.get("label_version") != label_version or settings.get("judged_k") != depth:
        return
    for row in stored.metrics.get("per_query", []):
        hits = ranked.get(row["query_id"], [])
        share = sum((row["query_id"], h.passage_id) in judged for h in hits) / len(hits) if hits else 0.0
        if share != row[f"judged@{depth}"]:
            raise DevPoolError(
                f"stored run {stored.run_id} reports judged@{depth} {row[f'judged@{depth}']} for "
                f"{row['query_id']}, but the base labels {label_version} give {share}; the run was not "
                "scored with these labels"
            )


def pool_coverage(entries: Sequence[DevPoolEntry], runs: Sequence[StoredRun], *, depth: int) -> dict[str, Any]:
    """What the expansion added, which systems proposed it, and how judged each run's top results were."""
    by_key = {(e.query_id, e.passage_id): e for e in entries}
    new = [e for e in entries if e.origin == "run_expansion"]
    unjudged = [e for e in entries if e.judged_in is None]
    in_runs = [e for e in entries if e.run_sources]
    per_query: dict[str, dict[str, int]] = {}
    for e in entries:
        row = per_query.setdefault(
            e.query_id,
            {"base_pool": 0, "new": 0, "new_in_scope": 0, "new_out_of_scope": 0, "pool": 0, "run_top_k_union": 0},
        )
        row["pool"] += 1
        row["base_pool"] += e.origin == "base_pool"
        row["new"] += e.origin == "run_expansion"
        row["new_in_scope"] += e.origin == "run_expansion" and e.in_scope
        row["new_out_of_scope"] += e.origin == "run_expansion" and not e.in_scope
        row["run_top_k_union"] += bool(e.run_sources)
    per_run: dict[str, Any] = {}
    for stored in runs:
        top = stored.ranked(depth)
        rows = {}
        for qid, hits in sorted(top.items()):
            states = [by_key[qid, h.passage_id] for h in hits]
            rows[qid] = {
                f"judged@{depth}": sum(e.judged_in is not None for e in states) / len(states) if states else 0.0,
                "unjudged": sum(e.judged_in is None for e in states),
                "unjudged_in_scope": sum(e.judged_in is None and e.in_scope for e in states),
                "new": sum(e.origin == "run_expansion" for e in states),
            }
        per_run[stored.run_id] = {
            f"judged@{depth}_before": sum(r[f"judged@{depth}"] for r in rows.values()) / len(rows) if rows else 0.0,
            "unjudged_top_k": sum(r["unjudged"] for r in rows.values()),
            "unjudged_top_k_in_scope": sum(r["unjudged_in_scope"] for r in rows.values()),
            "new_entries_contributed": sum(r["new"] for r in rows.values()),
            "per_query": rows,
        }
    systems = sorted({s.run for e in in_runs for s in e.run_sources})
    return {
        "depth": depth,
        "questions": len(per_query),
        "entries": len(entries),
        "base_pool_entries": len(entries) - len(new),
        "judged_entries": len(entries) - len(unjudged),
        "new_entries": len(new),
        "new_in_scope": sum(e.in_scope for e in new),
        "new_out_of_scope": sum(not e.in_scope for e in new),
        "unjudged_entries": len(unjudged),
        "run_top_k_pairs": len(in_runs),
        "run_top_k_pairs_unjudged": sum(e.judged_in is None for e in in_runs),
        "new_by_run": {
            r.run_id: sum(any((s.run, s.mode) == (r.run, r.mode) for s in e.run_sources) for e in new) for r in runs
        },
        "new_by_system": {name: sum(any(s.run == name for s in e.run_sources) for e in new) for name in systems},
        "new_by_system_set": dict(sorted(Counter(_systems(e.run_sources) for e in new).items())),
        "top_k_by_system_set": dict(sorted(Counter(_systems(e.run_sources) for e in in_runs).items())),
        "per_query": dict(sorted(per_query.items())),
        "per_run": per_run,
    }


# --- proposals for the unjudged entries ---------------------------------------------------


class PoolProposalSet(ProposalSet):
    """AI proposals for the unjudged entries of an expanded pool (``store`` is the ``DevPool``).

    A passage the base labels judged is never proposed for, so its label is carried over,
    not re-graded. The records and the append-only file are those of ``ProposalSet``.
    """

    def _check(
        self, proposal: AIProposal, *, where: str, existing: Mapping[tuple[str, str], AIProposal] | None = None
    ) -> None:
        existing = self._proposals if existing is None else existing
        key = (proposal.query_id, proposal.passage_id)
        entry = self.store.entry(*key)
        if entry is None:
            raise ProposalError(f"{where}: {proposal.passage_id} is not pooled for {proposal.query_id}")
        if entry.judged_in is not None:
            raise ProposalError(f"{where}: {key} was judged in {entry.judged_in}; that label is kept, not re-proposed")
        if key in existing:
            raise ProposalError(f"{where}: a proposal for {key} already exists; proposals are not revised")

    def coverage(self) -> dict[str, Any]:
        """Per question: pool size, entries judged before, proposed and still unlabelled, grades, confidence."""
        rows = {}
        for query_id in sorted({e.query_id for e in self.store.entries}):
            pool = self.store.pool(query_id)
            proposed = self.proposals(query_id)
            grades = Counter(p.grade for p in proposed)
            confidence = Counter(p.confidence for p in proposed)
            judged = sum(e.judged_in is not None for e in pool)
            rows[query_id] = {
                "pool": len(pool),
                "judged_before": judged,
                "ai_proposed": len(proposed),
                "unlabelled": len(pool) - judged - len(proposed),
                "grades": {str(g): grades.get(g, 0) for g in RELEVANCE_GRADES},
                "confidence": {c: confidence.get(c, 0) for c in CONFIDENCE_LEVELS},
            }
        return rows


def create_pool_proposals(
    directory: str | Path,
    pool: DevPool,
    *,
    model: str,
    method: str,
    method_description: str,
    created_by: str,
    clock: Callable[[], str] = utc_now,
) -> PoolProposalSet:
    """Start an empty proposal set for ``pool``'s unjudged entries; refuses to overwrite one."""
    return PoolProposalSet.create(
        directory, pool, model=model, method=method, method_description=method_description,  # type: ignore[arg-type]
        created_by=created_by, clock=clock,
    )


def open_pool_proposals(directory: str | Path, pool: DevPool) -> PoolProposalSet:
    """Open a proposal set after checking it labels ``pool`` (corpus and annotation version)."""
    return PoolProposalSet.open(directory, pool)  # type: ignore[arg-type]


# --- labels -------------------------------------------------------------------------------


class ExpandedLabelSpec(BaseModel):
    """How a label version is derived from a base label set and an expanded pool."""

    model_config = _STRICT

    label_version: NonEmptyStr
    annotation_version: NonEmptyStr
    description: NonEmptyStr
    base_labels_dir: NonEmptyStr
    pool_dir: NonEmptyStr
    proposals_dir: NonEmptyStr
    output_dir: NonEmptyStr
    rubric: RubricDecisions  # must be the base set's rubric version
    p2_basis: list[BasisRule] = Field(default_factory=list)
    p1_open_questions: list[NonEmptyStr] = Field(default_factory=list)
    review_confidence: list[Confidence] = Field(default_factory=lambda: ["low"])
    flags: list[SpecFlag] = Field(default_factory=list)  # curated; only on newly labelled passages

    @field_validator("label_version")
    @classmethod
    def _version(cls, v: str) -> str:
        return _check_version(v)


def is_expanded_label_spec(path: str | Path) -> bool:
    """Whether the spec at ``path`` derives labels from a base label set (it names ``base_labels_dir``)."""
    raw = read_yaml(Path(path)) if Path(path).is_file() else None
    return isinstance(raw, dict) and "base_labels_dir" in raw


def load_expanded_label_spec(path: str | Path) -> tuple[ExpandedLabelSpec, str]:
    return _load_spec(path, ExpandedLabelSpec, DevLabelError)


def derive_expanded_labels(
    spec: ExpandedLabelSpec,
    *,
    spec_path: str,
    spec_sha256: str,
    base: DevLabelSet,
    pool: DevPool,
    proposals: PoolProposalSet,
    passages: Mapping[str, Passage],
) -> DevLabelSet:
    """The base labels carried over unchanged plus the proposals for the pool's unjudged entries."""
    version = base.info.label_version
    base_dir = Path(spec.base_labels_dir)
    if pool.info.base.get("label_version") != version or pool.info.base.get("labels_meta_sha256") != sha256_file(
        base_dir / LABELS_INFO_FILE
    ):
        raise DevLabelError(f"the pool at {pool.directory} was not expanded from the base labels at {base_dir}")
    if spec.rubric.version != base.info.rubric.version:
        raise DevLabelError(
            f"spec rubric {spec.rubric.version} differs from the base labels' {base.info.rubric.version}; "
            "carried and new labels must follow one rubric"
        )
    if spec.label_version == version:
        raise DevLabelError(f"label version {version} already exists; derive a new version")
    questions = sorted(base.questions, key=lambda q: q.query_id)
    question_ids = {q.query_id for q in questions}
    for query_id in [r.query_id for r in spec.p2_basis] + list(spec.p1_open_questions):
        if query_id not in question_ids:
            raise DevLabelError(f"spec names unknown question {query_id}")
    judged_in_pool = {(e.query_id, e.passage_id) for e in pool.entries if e.judged_in is not None}
    if judged_in_pool != {(x.query_id, x.passage_id) for x in base.labels}:
        raise DevLabelError("the pool's judged entries differ from the base labels")

    carried = [label.model_copy(update={"introduced_in": version}) for label in base.labels]
    changes: list[GradeChange] = list(base.changes)
    flags: list[LabelFlag] = list(base.flags)
    basis = {rule.query_id: rule for rule in spec.p2_basis}
    applied_by = f"deterministic rule in {spec_path} (sha256 {spec_sha256[:12]})"
    new: list[DevLabel] = []
    unlabelled: Counter[str] = Counter()
    confidence: dict[tuple[str, str], str] = {}
    for entry in pool.unjudged():
        key = (entry.query_id, entry.passage_id)
        proposal = proposals.proposal(*key)
        if proposal is None:
            unlabelled[entry.query_id] += 1  # never invented: stays unjudged
            continue
        confidence[key] = proposal.confidence
        label = DevLabel(
            query_id=key[0], passage_id=key[1], grade=proposal.grade, label_source="ai_proposal",
            ai_generated=True, assistance="not_applicable", labelled_by=proposal.model,
            base_grade=proposal.grade, changed=False, introduced_in=spec.label_version,
        )
        rule = basis.get(entry.query_id)
        change = None if rule is None else p2_change(
            label, explanation=proposal.explanation, rule=rule, text=passages[entry.passage_id].text,
            change_id=f"chg_{len(changes) + 1:03d}", applied_by=applied_by,
        )
        if change is not None:
            changes.append(change)
            label = label.model_copy(update={"grade": 1, "changed": True})
        new.append(label)

    by_key = {(x.query_id, x.passage_id): x for x in new}

    def add_flag(key: tuple[str, str], rule: str, candidate: int | None, origin: str, reason: str) -> None:
        label = by_key[key]
        flag = make_flag(
            label, flag_id=f"flg_{len(flags) + 1:03d}", rule=rule, candidate=candidate, origin=origin, reason=reason
        )
        flags.append(flag)
        by_key[key] = label.model_copy(update={"flags": [*label.flags, flag.flag_id]})

    open_questions = set(spec.p1_open_questions)
    review = set(spec.review_confidence)
    for label in new:
        key = (label.query_id, label.passage_id)
        if label.query_id in open_questions and label.grade == 1:
            add_flag(key, "P1", 2, "rule", P1_FLAG_REASON)
        if confidence[key] in review:
            add_flag(
                key, "low_confidence", None, "rule",
                f"The labelling model gave this grade with {confidence[key]} confidence; a reader should check it.",
            )
    for spec_flag in spec.flags:
        key = (spec_flag.query_id, spec_flag.passage_id)
        if key not in by_key:
            raise DevLabelError(f"spec flag refers to {key}, which is not a newly labelled passage")
        add_flag(key, spec_flag.rule, spec_flag.candidate_grade, "spec", spec_flag.reason)
    new = [by_key[x.query_id, x.passage_id] for x in new]
    order = {(e.query_id, e.passage_id): i for i, e in enumerate(pool.entries)}
    labels = sorted(carried + new, key=lambda x: order[x.query_id, x.passage_id])

    counts = _expanded_counts(questions, labels, new, pool, unlabelled, confidence, changes, flags)
    proposals_info = proposals.info
    audit = {k: v for k, v in base.info.derived_from.items() if k.startswith("audit_")}
    derived_from = {
        "base_labels_dir": str(base_dir),
        "base_label_version": version,
        "base_labels_meta_sha256": sha256_file(base_dir / LABELS_INFO_FILE),
        "pool_dir": str(pool.directory),
        "pool_sha256": sha256_file(pool.directory / POOL_FILE),
        "pool_meta_sha256": sha256_file(pool.directory / POOL_INFO_FILE),
        "proposals_dir": str(proposals.directory),
        "proposals_sha256": proposals.sha256(),
        "proposals_meta_sha256": sha256_file(proposals.directory / PROPOSALS_INFO_FILE),
        "proposal_model": proposals_info.model,
        "proposal_method": proposals_info.method,
        **audit,
    }
    rules_applied = [
        f"Carried from {version} unchanged: {len(carried)} labels with their grades, sources, changes, flags and "
        "audit evidence; no earlier grade is re-graded or changed",
        f"New pool entries (from the stored runs in {pool.directory}) are labelled only by the AI proposals in "
        f"{proposals.directory}; entries without a proposal stay unjudged",
        f"P2 (deterministic) on new AI grade-2 labels for {', '.join(sorted(basis)) or 'no question'}: the passage "
        "must match the spec's period/basis patterns; otherwise grade 1",
        f"P1 (flag only) on new AI grade-1 labels for open questions {', '.join(sorted(open_questions)) or 'none'}",
        f"Confidence (flag only): new AI labels with confidence {', '.join(sorted(review))} are flagged for review",
        "Curated spec flags on new labels: flagged for review; grades unchanged",
        "No human judgment is created or changed",
    ]
    draft = DevLabelSet(
        info=DevLabelSetInfo(
            label_version=spec.label_version,
            description=spec.description,
            evaluation=EvaluationMetadata(
                corpus_sha256=base.info.evaluation.corpus_sha256,
                corpus_version=base.info.evaluation.corpus_version,
                annotation_version=spec.annotation_version,
            ),
            derived_from=derived_from,
            audit_independence=base.info.audit_independence,
            audit_independence_note=base.info.audit_independence_note,
            rubric=spec.rubric,
            spec={"path": spec_path, "sha256": spec_sha256},
            rules_applied=rules_applied,
            counts=counts,
            audit_dispositions=list(base.info.audit_dispositions),
            files={},
        ),
        questions=tuple(questions),
        labels=tuple(labels),
        changes=tuple(changes),
        flags=tuple(flags),
    )
    rendered = draft.rendered()
    files = {name: _sha256(rendered[name]) for name in DATA_FILES}
    label_set = DevLabelSet(
        draft.info.model_copy(update={"files": files}), draft.questions, draft.labels, draft.changes, draft.flags
    )
    validate_label_set(label_set, passages)
    return label_set


def _expanded_counts(
    questions: Sequence[GoldQuestion],
    labels: Sequence[DevLabel],
    new: Sequence[DevLabel],
    pool: DevPool,
    unlabelled: Mapping[str, int],
    confidence: Mapping[tuple[str, str], str],
    changes: Sequence[GradeChange],
    flags: Sequence[LabelFlag],
) -> dict[str, Any]:
    entries = {(e.query_id, e.passage_id): e for e in pool.entries}
    new_ids = {(x.query_id, x.passage_id) for x in new}

    def grades(rows: Iterable[DevLabel]) -> dict[str, int]:
        counts = Counter(x.grade for x in rows)
        return {str(g): counts.get(g, 0) for g in RELEVANCE_GRADES}

    per_question: dict[str, dict[str, Any]] = {}
    for question in questions:
        qid = question.query_id
        rows = [x for x in labels if x.query_id == qid]
        added = [x for x in new if x.query_id == qid]
        per_question[qid] = {
            "pool": len(pool.pool(qid)),
            "labelled": len(rows),
            "carried": len(rows) - len(added),
            "new": len(added),
            "unlabelled": unlabelled.get(qid, 0),
            "grades": grades(rows),
            "new_grades": grades(added),
            "positives": sum(1 for x in rows if x.grade > 0),
            "new_positives": sum(1 for x in added if x.grade > 0),
            "changed": sum(1 for x in rows if x.changed),
            "flagged": sum(1 for x in rows if x.flags),
        }
    new_positive_systems = Counter(
        _systems(entries[x.query_id, x.passage_id].run_sources) for x in new if x.grade > 0
    )
    new_flags = [f for f in flags if (f.query_id, f.passage_id) in new_ids]
    return {
        "labels": len(labels),
        "carried": len(labels) - len(new),
        "new": len(new),
        "unlabelled": sum(unlabelled.values()),
        "by_source": dict(sorted(Counter(x.label_source for x in labels).items())),
        "grades": grades(labels),
        "new_grades": grades(new),
        "new_confidence": {c: sum(1 for k in new_ids if confidence[k] == c) for c in CONFIDENCE_LEVELS},
        "new_in_scope_grades": grades(x for x in new if entries[x.query_id, x.passage_id].in_scope),
        "new_out_of_scope_grades": grades(x for x in new if not entries[x.query_id, x.passage_id].in_scope),
        "new_positives_by_system_set": dict(sorted(new_positive_systems.items())),
        "changes": dict(sorted(Counter(c.rule for c in changes).items())),
        "new_changes": dict(sorted(Counter(c.rule for c in changes if (c.query_id, c.passage_id) in new_ids).items())),
        "flags": dict(sorted(Counter(f.rule for f in flags).items())),
        "new_flags": dict(sorted(Counter(f.rule for f in new_flags).items())),
        "questions": per_question,
        "questions_without_positive": sorted(q for q, row in per_question.items() if row["positives"] == 0),
        "questions_with_new_positives": sorted(q for q, row in per_question.items() if row["new_positives"]),
    }


__all__ = [
    "POOL_DIRNAME",
    "POOL_FILE",
    "POOL_INFO_FILE",
    "POOL_NOTICE",
    "PROPOSALS_DIRNAME",
    "DevPool",
    "DevPoolEntry",
    "DevPoolError",
    "DevPoolInfo",
    "DevPoolSpec",
    "ExpandedLabelSpec",
    "PoolProposalSet",
    "RunRef",
    "RunSource",
    "create_pool_proposals",
    "derive_expanded_labels",
    "expand_pool",
    "is_expanded_label_spec",
    "load_dev_pool_spec",
    "load_expanded_label_spec",
    "open_pool_proposals",
    "pool_coverage",
    "read_annotation_pool",
    "validate_pool",
]
