"""Gold qrels export, judgment-coverage diagnostics and leakage checks.

The export follows TREC conventions: ``gold_qrels.jsonl`` lists only passages
someone graded (0, 1 or 2). Skipped and never-judged pool entries are left
out, so they stay *unjudged*. Retrieval metrics score an unjudged passage like
any passage outside the pool — as non-relevant — so the export report carries
the judgment coverage needed to read those metrics honestly, and
``judged_at_k`` measures how much of a run's top-k was ever judged.

A question is exported only when it is marked complete and has at least one
positive (grade >= 1) judgment: a question with no relevant passage scores 0
for every retriever and cannot tell them apart. Everything excluded is listed
with its reason.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from finretriever.data.schemas import Qrel, Query, SearchHit
from finretriever.eval_set.annotation_store import AnnotationStore
from finretriever.eval_set.schemas import (
    AUTHORING,
    COMPARISONS,
    QUESTION_TYPES,
    RELEVANCE_GRADES,
    TOPICS,
)
from finretriever.evaluation.provenance import EvaluationMetadata
from finretriever.utils.io import write_jsonl

GOLD_QUERY_SOURCE = "gold"
QUERIES_FILENAME = "gold_queries.jsonl"
QRELS_FILENAME = "gold_qrels.jsonl"
METADATA_FILENAME = "gold_eval.metadata.json"
REPORT_FILENAME = "gold_eval.report.json"

FLAG_EMPTY_POOL = "empty_pool"
FLAG_SMALL_POOL = "small_pool"
FLAG_PENDING = "pending_candidates"
FLAG_SKIPPED = "skipped_candidates"
FLAG_NO_POSITIVE = "no_positive"


class GoldExportError(ValueError):
    """The judgments cannot be exported as valid qrels."""


class GoldLeakageError(ValueError):
    """Training data contains gold evaluation questions or near-copies of them."""


# --- coverage diagnostics ----------------------------------------------------------


@dataclass(frozen=True)
class QueryCoverage:
    """Judgment counts for one question. ``pending`` were never graded or skipped."""

    query_id: str
    status: str
    pool_size: int
    graded: int
    skipped: int
    pending: int
    grades: dict[int, int]
    sources: dict[str, int]  # candidate method -> pooled passages it proposed
    flags: tuple[str, ...]

    @property
    def positives(self) -> int:
        return sum(n for grade, n in self.grades.items() if grade > 0)


@dataclass(frozen=True)
class CoverageReport:
    """Per-question judgment coverage plus totals, flags and the question mix."""

    queries: tuple[QueryCoverage, ...]
    target_queries: int | None
    small_pool_below: int
    status_counts: dict[str, int]
    totals: dict[str, int]
    flagged: dict[str, list[str]]
    distribution: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["queries"] = [{**asdict(q), "positives": q.positives} for q in self.queries]
        return out


def judgment_coverage(
    store: AnnotationStore,
    *,
    small_pool_below: int = 10,
    target_queries: int | None = None,
    query_ids: Collection[str] | None = None,
) -> CoverageReport:
    """Judgment coverage of every question in ``store`` (only ``query_ids``, when given).

    Flags: ``empty_pool``; ``small_pool`` (fewer than ``small_pool_below``
    candidates); ``pending_candidates`` and ``skipped_candidates`` (pool
    entries without a grade); ``no_positive`` (no grade >= 1 among the graded
    ones). Withdrawn questions are counted but never flagged.
    """
    questions = [q for q in store.questions() if query_ids is None or q.query_id in query_ids]
    rows: list[QueryCoverage] = []
    for question in questions:
        pool = store.pool(question.query_id)
        judgments = store.judgments(question.query_id)
        grades = Counter(j.grade for j in judgments if j.grade is not None)
        graded = sum(grades.values())
        skipped = sum(j.grade is None for j in judgments)
        sources = Counter(method for e in pool for method in dict.fromkeys(s.method for s in e.sources))
        flags: list[str] = []
        if question.status != "withdrawn":
            if not pool:
                flags.append(FLAG_EMPTY_POOL)
            elif len(pool) < small_pool_below:
                flags.append(FLAG_SMALL_POOL)
            if len(pool) - graded - skipped:
                flags.append(FLAG_PENDING)
            if skipped:
                flags.append(FLAG_SKIPPED)
            if not any(grade > 0 for grade in grades):
                flags.append(FLAG_NO_POSITIVE)
        rows.append(
            QueryCoverage(
                query_id=question.query_id,
                status=question.status,
                pool_size=len(pool),
                graded=graded,
                skipped=skipped,
                pending=len(pool) - graded - skipped,
                grades={grade: grades.get(grade, 0) for grade in RELEVANCE_GRADES},
                sources=dict(sorted(sources.items())),
                flags=tuple(flags),
            )
        )

    active = [q for q in questions if q.status != "withdrawn"]
    flagged = {
        flag: [r.query_id for r in rows if flag in r.flags]
        for flag in (FLAG_EMPTY_POOL, FLAG_SMALL_POOL, FLAG_PENDING, FLAG_SKIPPED, FLAG_NO_POSITIVE)
    }
    totals = {
        "pooled": sum(r.pool_size for r in rows),
        "graded": sum(r.graded for r in rows),
        "skipped": sum(r.skipped for r in rows),
        "pending": sum(r.pending for r in rows),
        **{f"grade_{grade}": sum(r.grades[grade] for r in rows) for grade in RELEVANCE_GRADES},
    }

    def mix(values: Iterable[str], names: Sequence[str]) -> dict[str, int]:
        counts = Counter(values)
        return {name: counts.get(name, 0) for name in names}

    distribution = {
        "topic": mix((q.topic for q in active), TOPICS),
        "question_type": mix((q.question_type for q in active), QUESTION_TYPES),
        "comparison": mix((q.comparison for q in active), COMPARISONS),
        "authoring": mix((q.authoring for q in active), AUTHORING),
        "ai_assisted": mix(("yes" if q.ai_assisted else "no" for q in active), ("yes", "no")),
        "ticker": dict(sorted(Counter(t for q in active for t in q.tickers or ["(any)"]).items())),
    }
    return CoverageReport(
        queries=tuple(rows),
        target_queries=target_queries,
        small_pool_below=small_pool_below,
        status_counts=dict(Counter(q.status for q in questions)),
        totals=totals,
        flagged=flagged,
        distribution=distribution,
    )


# --- export ------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldExport:
    """Benchmark-ready queries and qrels, with what was left out and why."""

    metadata: EvaluationMetadata
    queries: list[Query]
    qrels: list[Qrel]
    excluded: dict[str, str]
    coverage: CoverageReport
    unjudged_in_pool: dict[str, int] = field(default_factory=dict)  # exported query -> skipped + pending


def validate_gold_qrels(queries: Sequence[Query], qrels: Sequence[Qrel], passage_ids: Mapping[str, Any]) -> None:
    """Raise ``GoldExportError`` unless queries and qrels form a consistent gold set."""
    ids = [q.query_id for q in queries]
    if duplicates := sorted(qid for qid, n in Counter(ids).items() if n > 1):
        raise GoldExportError(f"duplicate query IDs: {duplicates}")
    known = set(ids)
    pairs = Counter((r.query_id, r.passage_id) for r in qrels)
    if conflicts := sorted(pair for pair, n in pairs.items() if n > 1):
        raise GoldExportError(f"more than one judgment for (query, passage): {conflicts[:5]}")
    if unknown_queries := sorted({r.query_id for r in qrels} - known):
        raise GoldExportError(f"qrels for queries not in the query set: {unknown_queries}")
    if unknown_passages := sorted({r.passage_id for r in qrels} - passage_ids.keys()):
        raise GoldExportError(f"judged passages not in the corpus: {unknown_passages[:5]}")
    if bad := sorted({r.relevance for r in qrels} - RELEVANCE_GRADES.keys()):
        raise GoldExportError(f"relevance grades outside {sorted(RELEVANCE_GRADES)}: {bad}")
    if unjudged := sorted(known - {r.query_id for r in qrels}):
        raise GoldExportError(f"queries without any judgment: {unjudged}")


def build_gold_export(
    store: AnnotationStore,
    *,
    include_open: bool = False,
    small_pool_below: int = 10,
    target_queries: int | None = None,
    ai_identities: Iterable[str] = (),
) -> GoldExport:
    """Queries and qrels of every exportable question in ``store``.

    Exported: questions marked complete (or still open, with
    ``include_open``) that have at least one positive grade. Every graded
    passage becomes a qrel; skipped and pending passages do not. Any
    judgment recorded under one of ``ai_identities`` stops the export: gold
    judgments come only from people.
    """
    ai = {name.casefold() for name in ai_identities}
    if machine := [j for q in store.questions() for j in store.judgments(q.query_id) if j.annotator.casefold() in ai]:
        raise GoldExportError(
            f"{len(machine)} judgments were recorded by AI identities ({sorted({j.annotator for j in machine})}); "
            "AI labels are development data, never gold"
        )
    queries: list[Query] = []
    qrels: list[Qrel] = []
    excluded: dict[str, str] = {}
    unjudged: dict[str, int] = {}
    version = store.info.evaluation.annotation_version
    for question in store.questions():
        if question.status == "withdrawn":
            excluded[question.query_id] = "withdrawn"
            continue
        if question.status != "complete" and not include_open:
            excluded[question.query_id] = "not marked complete"
            continue
        judgments = store.judgments(question.query_id)
        graded = sorted((j for j in judgments if j.grade is not None), key=lambda j: j.passage_id)
        if not any(j.grade for j in graded):
            excluded[question.query_id] = "no positive judgment"
            continue
        queries.append(
            Query(
                query_id=question.query_id,
                text=question.text,
                source=GOLD_QUERY_SOURCE,
                metadata={
                    "annotation_version": version,
                    "topic": question.topic,
                    "question_type": question.question_type,
                    "comparison": question.comparison,
                    "tickers": ",".join(question.tickers),
                    "fiscal_years": ",".join(map(str, question.fiscal_years)),
                    "authoring": question.authoring,
                    "ai_assisted": question.ai_assisted,
                    "reviewed_by": question.reviewed_by,
                    "status": question.status,
                },
            )
        )
        qrels += [Qrel(query_id=j.query_id, passage_id=j.passage_id, relevance=j.grade) for j in graded]
        unjudged[question.query_id] = len(store.pool(question.query_id)) - len(graded)
    validate_gold_qrels(queries, qrels, store.passages)
    return GoldExport(
        metadata=store.info.evaluation,
        queries=queries,
        qrels=qrels,
        excluded=excluded,
        coverage=judgment_coverage(store, small_pool_below=small_pool_below, target_queries=target_queries),
        unjudged_in_pool=unjudged,
    )


def write_gold_export(export: GoldExport, out_dir: str | Path) -> dict[str, Path]:
    """Write queries, qrels, metadata and the coverage report; returns their paths.

    The output holds no timestamps, so exporting unchanged judgments twice
    gives identical files.
    """
    if not export.queries:
        raise GoldExportError(
            "nothing to export: no complete question has a positive judgment "
            f"(excluded: {dict(Counter(export.excluded.values()))})"
        )
    out = Path(out_dir)
    paths = {
        "queries": out / QUERIES_FILENAME,
        "qrels": out / QRELS_FILENAME,
        "metadata": out / METADATA_FILENAME,
        "report": out / REPORT_FILENAME,
    }
    write_jsonl(paths["queries"], export.queries)
    write_jsonl(paths["qrels"], export.qrels)
    grades = Counter(r.relevance for r in export.qrels)
    report = {
        "evaluation": export.metadata.model_dump(),
        "exported": {
            "queries": len(export.queries),
            "qrels": len(export.qrels),
            "grades": {str(g): grades.get(g, 0) for g in RELEVANCE_GRADES},
            "unjudged_pool_entries": export.unjudged_in_pool,
        },
        "excluded": export.excluded,
        "unjudged_policy": (
            "Only graded passages are qrels. Skipped and never-judged pool entries, and every passage "
            "outside the pool, are unjudged: metrics count them as non-relevant, which is an assumption, "
            "not a judgment. Report judged@k alongside every metric."
        ),
        "coverage": export.coverage.to_dict(),
    }
    for key, payload in (("metadata", export.metadata.model_dump()), ("report", report)):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


# --- evaluation-time diagnostics ----------------------------------------------------


def judged_at_k(qrels: Sequence[Qrel], hits: Sequence[SearchHit], k: int = 10) -> dict[str, float]:
    """Share of each query's top-``k`` hits that carry a judgment (any grade).

    Low values mean a metric rests on unjudged passages counted as
    non-relevant; compare it across retrievers before comparing their scores.
    A run that returns fewer than ``k`` hits is scored on what it returned;
    queries without hits score 0.0.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    judged: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels:
        judged[qrel.query_id].add(qrel.passage_id)
    ranked: dict[str, list[str]] = defaultdict(list)
    for hit in sorted(hits, key=lambda h: (h.query_id, h.rank)):
        if hit.passage_id not in ranked[hit.query_id]:
            ranked[hit.query_id].append(hit.passage_id)
    shares: dict[str, float] = {}
    for qid, pids in judged.items():
        top = ranked.get(qid, [])[:k]
        shares[qid] = sum(pid in pids for pid in top) / len(top) if top else 0.0
    return shares


# --- leakage ------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.casefold()))


@dataclass(frozen=True)
class GoldOverlap:
    query_id: str  # the gold question
    index: int  # position of the overlapping training text
    similarity: float


def find_gold_overlap(
    gold_queries: Sequence[Query], texts: Sequence[str], *, threshold: float = 0.8
) -> list[GoldOverlap]:
    """Training texts that repeat a gold question (word-set Jaccard >= ``threshold``).

    Lexical only: it catches copies and light rewordings, not paraphrases that
    change the wording; see docs/eval_methodology.md for the other safeguards.
    """
    gold = [(q.query_id, _words(q.text)) for q in gold_queries]
    overlaps: list[GoldOverlap] = []
    for index, text in enumerate(texts):
        words = _words(text)
        for query_id, gold_words in gold:
            union = words | gold_words
            similarity = len(words & gold_words) / len(union) if union else 0.0
            if similarity >= threshold:
                overlaps.append(GoldOverlap(query_id, index, round(similarity, 3)))
    return overlaps


def check_no_gold_leakage(gold_queries: Sequence[Query], texts: Sequence[str], *, threshold: float = 0.8) -> None:
    """Raise ``GoldLeakageError`` if any training text repeats a gold question."""
    if overlaps := find_gold_overlap(gold_queries, texts, threshold=threshold):
        shown = ", ".join(f"text {o.index} ~ {o.query_id} ({o.similarity})" for o in overlaps[:5])
        raise GoldLeakageError(f"{len(overlaps)} training texts overlap gold questions: {shown}")


__all__ = [
    "GOLD_QUERY_SOURCE",
    "METADATA_FILENAME",
    "QRELS_FILENAME",
    "QUERIES_FILENAME",
    "REPORT_FILENAME",
    "CoverageReport",
    "GoldExport",
    "GoldExportError",
    "GoldLeakageError",
    "GoldOverlap",
    "QueryCoverage",
    "build_gold_export",
    "check_no_gold_leakage",
    "find_gold_overlap",
    "judged_at_k",
    "judgment_coverage",
    "validate_gold_qrels",
    "write_gold_export",
]
