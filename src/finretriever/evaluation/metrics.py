"""Retrieval metrics for FinRetriever.

This module is load-bearing: every recruiter-facing number in the project
(nDCG@10, MRR@10, Recall@100) flows through here. Errors here silently
invalidate the benchmark, so the implementation is deliberately explicit and
tested against hand-computed fixtures.

One implementation, whatever is installed (decided 2026-11-12). Every metric
is computed by the pure-Python kernels below. ``pytrec_eval`` is never used,
even when it is installed. trec_eval, which it wraps, by default takes the
grade itself as the gain, not ``2**rel - 1``. For graded judgments, results
would then depend on the environment. The parity test that would have shown
this never ran here, because pytrec_eval was not installed. ``metric_backend()``
returns ``METRIC_BACKEND`` everywhere.

Conventions (authoritative)
---------------------------

* Relevance is an integer; ``relevance > 0`` means "relevant" for MRR and
  Recall. nDCG uses graded relevance directly.
* **Gain:** a passage of grade ``rel`` at rank ``i`` (1-indexed) contributes
  ``(2**rel - 1) / log2(i + 1)`` to DCG@k: gain 0, 1 and 3 for grades 0, 1
  and 2; discount 1 at rank 1, 1/log2(3) at rank 2, and so on.
* nDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of the query's judged
  grades sorted in decreasing order (top k); if IDCG is zero, nDCG is 0.0.
  A retrieved passage without a judgment counts as grade 0.
* Recall@k = ``|relevant ∩ top_k| / |relevant|``; if there are no relevant
  documents for a query, Recall is 0.0.
* MRR@k = ``1 / rank_of_first_relevant`` within the top-k, else 0.0.
* All averages are arithmetic means over the queries present in the qrels.
"""

from __future__ import annotations

import math
from collections import defaultdict

from finretriever.data.schemas import Qrel, SearchHit

METRIC_BACKEND = "finretriever_python"  # the only backend: the kernels below (see the module docstring)

_DEFAULT_METRICS: tuple[str, ...] = ("ndcg@10", "mrr@10", "recall@100")
_SUPPORTED_BASES: frozenset[str] = frozenset({"ndcg", "mrr", "recall"})


# ---------------------------------------------------------------------------
# Backend probe
# ---------------------------------------------------------------------------


def metric_backend() -> str:
    """The metric implementation in use: always ``METRIC_BACKEND`` (the pure-Python kernels), whatever is
    installed."""
    return METRIC_BACKEND


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def qrels_to_trec(qrels: list[Qrel]) -> dict[str, dict[str, int]]:
    """Convert a flat list of :class:`Qrel` into the nested TREC qrel format.

    On duplicate ``(query_id, passage_id)`` pairs the maximum relevance is
    kept; the lower grade is dropped so it cannot quietly downgrade a
    higher-graded judgement.
    """
    out: dict[str, dict[str, int]] = {}
    for q in qrels:
        per_query = out.setdefault(q.query_id, {})
        existing = per_query.get(q.passage_id)
        rel = int(q.relevance)
        if existing is None or rel > existing:
            per_query[q.passage_id] = rel
    return out


def hits_to_trec_run(hits: list[SearchHit]) -> dict[str, dict[str, float]]:
    """Convert a flat list of :class:`SearchHit` into the nested TREC run format.

    On duplicate ``(query_id, passage_id)`` pairs the hit with the lowest
    ``rank`` wins; ties on rank are broken by the higher ``score``. Scores
    are preserved as floats. Input order is not assumed.
    """
    best: dict[tuple[str, str], SearchHit] = {}
    for h in hits:
        key = (h.query_id, h.passage_id)
        cur = best.get(key)
        if cur is None or h.rank < cur.rank or (h.rank == cur.rank and h.score > cur.score):
            best[key] = h
    out: dict[str, dict[str, float]] = {}
    for (qid, pid), h in best.items():
        out.setdefault(qid, {})[pid] = float(h.score)
    return out


def group_hits_by_query(hits: list[SearchHit]) -> dict[str, list[SearchHit]]:
    """Group ``hits`` by ``query_id``, sorted by ``(rank asc, score desc)``."""
    grouped: dict[str, list[SearchHit]] = defaultdict(list)
    for h in hits:
        grouped[h.query_id].append(h)
    for items in grouped.values():
        items.sort(key=lambda h: (h.rank, -h.score))
    return dict(grouped)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def average_metric(per_query: dict[str, float]) -> float:
    """Arithmetic mean over query-level scores. Returns 0.0 for an empty dict."""
    if not per_query:
        return 0.0
    return sum(per_query.values()) / len(per_query)


# ---------------------------------------------------------------------------
# Metric name parsing
# ---------------------------------------------------------------------------


def _parse_metric(name: str) -> tuple[str, int]:
    """Split ``"ndcg@10"`` into ``("ndcg", 10)`` after lower-casing.

    Raises :class:`ValueError` with a clear message on unsupported bases,
    missing cutoffs, non-integer cutoffs, or cutoffs < 1.
    """
    if not isinstance(name, str) or "@" not in name:
        raise ValueError(
            f"Invalid metric name {name!r}: expected '<base>@<cutoff>', e.g. 'ndcg@10'."
        )
    base_raw, _, cutoff_raw = name.partition("@")
    base = base_raw.strip().lower()
    cutoff_str = cutoff_raw.strip()
    if base not in _SUPPORTED_BASES:
        raise ValueError(
            f"Unsupported metric base {base_raw!r} in {name!r}: "
            f"expected one of {sorted(_SUPPORTED_BASES)}."
        )
    try:
        cutoff = int(cutoff_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid cutoff {cutoff_str!r} in metric {name!r}: must be a positive integer."
        ) from exc
    if cutoff < 1:
        raise ValueError(
            f"Invalid cutoff {cutoff} in metric {name!r}: must be >= 1."
        )
    return base, cutoff


def _normalize_metric(name: str) -> str:
    base, cutoff = _parse_metric(name)
    return f"{base}@{cutoff}"


# ---------------------------------------------------------------------------
# Pure-Python metric kernels (also used by tests for backend parity)
# ---------------------------------------------------------------------------


def _mrr_at_k(qrels_for_query: dict[str, int], ranked_passages: list[str], k: int) -> float:
    for position, pid in enumerate(ranked_passages[:k], start=1):
        if qrels_for_query.get(pid, 0) > 0:
            return 1.0 / position
    return 0.0


def _recall_at_k(qrels_for_query: dict[str, int], ranked_passages: list[str], k: int) -> float:
    relevant = {pid for pid, rel in qrels_for_query.items() if rel > 0}
    if not relevant:
        return 0.0
    # set() naturally deduplicates the top-k, so duplicate hits cannot inflate recall.
    retrieved_relevant = len(set(ranked_passages[:k]) & relevant)
    return retrieved_relevant / len(relevant)


def _ndcg_at_k(qrels_for_query: dict[str, int], ranked_passages: list[str], k: int) -> float:
    dcg = 0.0
    for position, pid in enumerate(ranked_passages[:k], start=1):
        rel = qrels_for_query.get(pid, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(position + 1)
    ideal_rels = sorted(
        (r for r in qrels_for_query.values() if r > 0), reverse=True
    )[:k]
    idcg = 0.0
    for position, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1) / math.log2(position + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Public metric computation
# ---------------------------------------------------------------------------


def _dedup_ranked(hits_for_query: list[SearchHit]) -> list[str]:
    """Return passage IDs in ranked order, preserving the first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for h in hits_for_query:
        if h.passage_id not in seen:
            seen.add(h.passage_id)
            out.append(h.passage_id)
    return out


def compute_per_query_metrics(
    qrels: list[Qrel],
    hits: list[SearchHit],
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Return ``{query_id: {metric_name: value}}`` over every query in ``qrels``.

    See module docstring for the metric definitions and conventions. Query
    IDs present only in ``hits`` are ignored — evaluation is anchored to the
    gold set.
    """
    metric_names = list(metrics) if metrics is not None else list(_DEFAULT_METRICS)
    parsed: list[tuple[str, str, int]] = []  # (normalized_name, base, cutoff)
    for name in metric_names:
        base, cutoff = _parse_metric(name)
        parsed.append((f"{base}@{cutoff}", base, cutoff))

    qrels_by_q = qrels_to_trec(qrels)
    grouped_hits = group_hits_by_query(hits)
    ranked_by_q: dict[str, list[str]] = {
        qid: _dedup_ranked(hs) for qid, hs in grouped_hits.items()
    }

    out: dict[str, dict[str, float]] = {}
    for qid, q_qrels in qrels_by_q.items():
        ranked = ranked_by_q.get(qid, [])
        q_scores: dict[str, float] = {}
        for norm_name, base, cutoff in parsed:
            if base == "mrr":
                q_scores[norm_name] = _mrr_at_k(q_qrels, ranked, cutoff)
            elif base == "ndcg":
                q_scores[norm_name] = _ndcg_at_k(q_qrels, ranked, cutoff)
            elif base == "recall":
                q_scores[norm_name] = _recall_at_k(q_qrels, ranked, cutoff)
            else:  # pragma: no cover - guarded by _parse_metric
                raise ValueError(f"Unsupported metric base: {base}")
        out[qid] = q_scores
    return out


def compute_metrics(
    qrels: list[Qrel],
    hits: list[SearchHit],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Return mean per-query scores keyed by the normalized metric name.

    Defaults to ``["ndcg@10", "mrr@10", "recall@100"]``. Empty qrels or
    empty hits return 0.0 for every requested metric.
    """
    metric_names = list(metrics) if metrics is not None else list(_DEFAULT_METRICS)
    normalized = [_normalize_metric(name) for name in metric_names]

    per_query = compute_per_query_metrics(qrels, hits, metric_names)
    if not per_query:
        return {name: 0.0 for name in normalized}

    out: dict[str, float] = {}
    for name in normalized:
        out[name] = average_metric(
            {qid: scores[name] for qid, scores in per_query.items()}
        )
    return out


__all__ = [
    "METRIC_BACKEND",
    "average_metric",
    "compute_metrics",
    "compute_per_query_metrics",
    "group_hits_by_query",
    "hits_to_trec_run",
    "metric_backend",
    "qrels_to_trec",
]
