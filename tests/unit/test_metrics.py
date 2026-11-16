"""Tests for finretriever.evaluation.metrics.

Metric correctness is non-negotiable: every recruiter-facing number depends
on it. Tests therefore:

* compare against *hand-computed* values from the docstring formula;
* pin the one backend (the pure-Python kernels), whatever is installed;
* check the conversion helpers in isolation so a bug in qrel/hit handling
  cannot quietly skew the metric.
"""

from __future__ import annotations

import math
import sys

import pytest

from finretriever.data.schemas import Qrel, SearchHit
from finretriever.evaluation.metrics import (
    METRIC_BACKEND,
    _mrr_at_k,
    _ndcg_at_k,
    _recall_at_k,
    average_metric,
    compute_metrics,
    compute_per_query_metrics,
    group_hits_by_query,
    hits_to_trec_run,
    metric_backend,
    qrels_to_trec,
)


def _hit(query_id: str, passage_id: str, rank: int, score: float) -> SearchHit:
    return SearchHit(
        query_id=query_id,
        passage_id=passage_id,
        rank=rank,
        score=score,
        retriever="test",
    )


# ---------------------------------------------------------------------------
# qrels_to_trec
# ---------------------------------------------------------------------------


def test_qrels_to_trec_groups_by_query_id() -> None:
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=2),
        Qrel(query_id="q1", passage_id="p2", relevance=1),
        Qrel(query_id="q2", passage_id="p3", relevance=1),
    ]
    assert qrels_to_trec(qrels) == {
        "q1": {"p1": 2, "p2": 1},
        "q2": {"p3": 1},
    }


def test_qrels_to_trec_keeps_max_relevance_for_duplicates() -> None:
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=1),
        Qrel(query_id="q1", passage_id="p1", relevance=3),
        Qrel(query_id="q1", passage_id="p1", relevance=2),
    ]
    assert qrels_to_trec(qrels) == {"q1": {"p1": 3}}


# ---------------------------------------------------------------------------
# hits_to_trec_run
# ---------------------------------------------------------------------------


def test_hits_to_trec_run_groups_by_query_id() -> None:
    hits = [
        _hit("q1", "p1", rank=1, score=0.9),
        _hit("q1", "p2", rank=2, score=0.5),
        _hit("q2", "p1", rank=1, score=0.8),
    ]
    run = hits_to_trec_run(hits)
    assert run == {
        "q1": {"p1": 0.9, "p2": 0.5},
        "q2": {"p1": 0.8},
    }


def test_hits_to_trec_run_keeps_best_rank_for_duplicates() -> None:
    hits = [
        _hit("q1", "p1", rank=5, score=0.99),
        _hit("q1", "p1", rank=1, score=0.10),
        _hit("q1", "p1", rank=3, score=0.50),
    ]
    run = hits_to_trec_run(hits)
    assert run == {"q1": {"p1": 0.10}}  # rank=1 wins despite the low score


def test_hits_to_trec_run_tie_rank_keeps_higher_score() -> None:
    hits = [
        _hit("q1", "p1", rank=1, score=0.3),
        _hit("q1", "p1", rank=1, score=0.9),
        _hit("q1", "p1", rank=1, score=0.5),
    ]
    run = hits_to_trec_run(hits)
    assert run == {"q1": {"p1": 0.9}}


# ---------------------------------------------------------------------------
# group_hits_by_query
# ---------------------------------------------------------------------------


def test_group_hits_by_query_sorts_by_rank_ascending() -> None:
    hits = [
        _hit("q1", "p3", rank=3, score=0.5),
        _hit("q1", "p1", rank=1, score=0.9),
        _hit("q1", "p2", rank=2, score=0.7),
    ]
    grouped = group_hits_by_query(hits)
    assert [h.passage_id for h in grouped["q1"]] == ["p1", "p2", "p3"]


def test_group_hits_by_query_tie_rank_sorts_by_score_descending() -> None:
    hits = [
        _hit("q1", "pA", rank=1, score=0.4),
        _hit("q1", "pB", rank=1, score=0.9),
        _hit("q1", "pC", rank=1, score=0.6),
    ]
    grouped = group_hits_by_query(hits)
    assert [h.passage_id for h in grouped["q1"]] == ["pB", "pC", "pA"]


# ---------------------------------------------------------------------------
# average_metric
# ---------------------------------------------------------------------------


def test_average_metric_returns_arithmetic_mean() -> None:
    assert math.isclose(
        average_metric({"q1": 1.0, "q2": 0.5, "q3": 0.0}),
        0.5,
        rel_tol=1e-9,
    )


def test_average_metric_returns_zero_for_empty_dict() -> None:
    assert average_metric({}) == 0.0


# ---------------------------------------------------------------------------
# metric_backend
# ---------------------------------------------------------------------------


def test_metric_backend_is_the_python_kernels_whatever_is_installed() -> None:
    assert metric_backend() == METRIC_BACKEND == "finretriever_python"


def test_pytrec_eval_is_never_used_even_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in pytrec_eval that fails if touched: the metrics must not change and must not use it."""

    class _Refuse:
        def __getattr__(self, name: str) -> None:
            raise AssertionError(f"pytrec_eval.{name} was used")

    monkeypatch.setitem(sys.modules, "pytrec_eval", _Refuse())
    qrels = [Qrel(query_id="q", passage_id="a", relevance=1), Qrel(query_id="q", passage_id="b", relevance=2)]
    hits = [_hit("q", "a", 1, 0.9), _hit("q", "b", 2, 0.8)]
    scores = compute_per_query_metrics(qrels, hits, ["ndcg@10", "recall@100", "mrr@10"])["q"]
    assert scores["ndcg@10"] == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert (scores["recall@100"], scores["mrr@10"]) == (1.0, 1.0)


# ---------------------------------------------------------------------------
# Known answers for graded judgments (the authoritative convention: gain 2**grade - 1, discount log2(rank + 1))
# ---------------------------------------------------------------------------

_LOG = {r: math.log2(r + 1) for r in range(1, 12)}


@pytest.mark.parametrize(
    ("grades", "ranked", "expected"),
    [
        # grade 2 at rank 1, grade 1 at rank 3: DCG = 3/1 + 1/2; IDCG = 3/1 + 1/log2(3)
        ({"a": 2, "b": 1}, ["a", "x", "b"], (3 + 1 / 2) / (3 + 1 / _LOG[2])),
        # the reverse order: grade 1 first, grade 2 at rank 3
        ({"a": 2, "b": 1}, ["b", "x", "a"], (1 + 3 / 2) / (3 + 1 / _LOG[2])),
        # grade 3 (a scale above the gold rubric's): gain 7
        ({"a": 3, "b": 1}, ["b", "a"], (1 + 7 / _LOG[2]) / (7 + 1 / _LOG[2])),
        # a judged 0 counts like an unjudged passage; the grade-2 passage is never retrieved
        ({"a": 0, "b": 1, "c": 2}, ["a", "b"], (1 / _LOG[2]) / (3 + 1 / _LOG[2])),
        # only grade 1s: the gain equals the binary case
        ({"a": 1, "b": 1}, ["x", "a", "b"], (1 / _LOG[2] + 1 / _LOG[3]) / (1 + 1 / _LOG[2])),
        # the cut at 10: a grade-2 passage at rank 11 adds nothing
        ({"a": 2}, [f"x{i}" for i in range(10)] + ["a"], 0.0),
        # the grade-2 passage at rank 10 exactly
        ({"a": 2}, [f"x{i}" for i in range(9)] + ["a"], (3 / _LOG[10]) / 3),
        # no relevant judgment at all: nDCG is 0
        ({"a": 0}, ["a"], 0.0),
        # more than 10 relevant: the ideal ranking is cut at 10 too
        ({f"r{i}": 2 for i in range(12)}, [f"r{i}" for i in range(12)], 1.0),
    ],
)
def test_ndcg_at_10_known_answers(grades: dict[str, int], ranked: list[str], expected: float) -> None:
    qrels = [Qrel(query_id="q", passage_id=p, relevance=g) for p, g in grades.items()]
    hits = [_hit("q", p, i, 1.0 / i) for i, p in enumerate(ranked, start=1)]
    assert compute_per_query_metrics(qrels, hits, ["ndcg@10"])["q"]["ndcg@10"] == pytest.approx(expected, abs=1e-12)
    assert _ndcg_at_k(grades, ranked, 10) == pytest.approx(expected, abs=1e-12)


def test_mrr_and_recall_known_answers_with_grades() -> None:
    """MRR@10 and Recall@100 treat grades 1 and 2 alike; duplicates count once; the cut-offs hold."""
    qrels = [Qrel(query_id="q", passage_id=p, relevance=g) for p, g in {"a": 0, "b": 1, "c": 2, "d": 2}.items()]
    hits = [_hit("q", p, i, 1.0 / i) for i, p in enumerate(["a", "b", "b", "x", "c"], start=1)]
    scores = compute_per_query_metrics(qrels, hits, ["mrr@10", "recall@100", "recall@2"])["q"]
    assert scores["mrr@10"] == pytest.approx(1 / 2)  # b at rank 2 (a is judged 0)
    assert scores["recall@100"] == pytest.approx(2 / 3)  # b and c of b, c, d; the duplicate b counts once
    assert scores["recall@2"] == pytest.approx(1 / 3)  # the top 2 after de-duplication: a, b


def test_every_system_is_scored_by_the_same_function() -> None:
    """Two systems with the same ranking get identical numbers, and a mean is the mean of the per-query values."""
    qrels = [Qrel(query_id=q, passage_id=f"{q}-r", relevance=g) for q, g in (("q1", 2), ("q2", 1))]
    ranked = {"q1": ["x", "q1-r"], "q2": ["q2-r"]}
    runs = {name: [SearchHit(query_id=q, passage_id=p, rank=i, score=1.0 / i, retriever=name)
                   for q, ps in ranked.items() for i, p in enumerate(ps, start=1)] for name in ("A0", "A1")}
    a0, a1 = (compute_per_query_metrics(qrels, runs[n]) for n in ("A0", "A1"))
    assert a0 == a1
    assert compute_metrics(qrels, runs["A0"])["ndcg@10"] == pytest.approx((a0["q1"]["ndcg@10"] + 1.0) / 2)


# ---------------------------------------------------------------------------
# compute_metrics / compute_per_query_metrics behaviour
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_exact_default_keys() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p1", relevance=1)]
    hits = [_hit("q1", "p1", rank=1, score=1.0)]
    result = compute_metrics(qrels, hits)
    assert set(result.keys()) == {"ndcg@10", "mrr@10", "recall@100"}


def test_perfect_ranking_gives_one_for_all_default_metrics() -> None:
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=2),
        Qrel(query_id="q1", passage_id="p2", relevance=1),
    ]
    hits = [
        _hit("q1", "p1", rank=1, score=0.99),
        _hit("q1", "p2", rank=2, score=0.80),
    ]
    result = compute_metrics(qrels, hits)
    for name in ("ndcg@10", "mrr@10", "recall@100"):
        assert math.isclose(result[name], 1.0, rel_tol=1e-9, abs_tol=1e-12)


def test_query_in_qrels_but_missing_from_hits_contributes_zero() -> None:
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=1),
        Qrel(query_id="q2", passage_id="pX", relevance=1),  # never retrieved
    ]
    hits = [_hit("q1", "p1", rank=1, score=1.0)]
    result = compute_metrics(qrels, hits)
    # q1 contributes 1.0, q2 contributes 0.0 → mean = 0.5
    for name in ("ndcg@10", "mrr@10", "recall@100"):
        assert math.isclose(result[name], 0.5, rel_tol=1e-9, abs_tol=1e-12)


def test_extra_hit_queries_not_in_qrels_do_not_affect_average() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p1", relevance=1)]
    hits = [
        _hit("q1", "p1", rank=1, score=1.0),
        _hit("q99", "pX", rank=1, score=1.0),  # not in qrels at all
        _hit("q99", "pY", rank=2, score=0.5),
    ]
    result = compute_metrics(qrels, hits)
    for name in ("ndcg@10", "mrr@10", "recall@100"):
        assert math.isclose(result[name], 1.0, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Hand-computed metric values
# ---------------------------------------------------------------------------


def test_mrr_at_10_when_first_relevant_is_at_rank_three() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p3", relevance=1)]
    hits = [
        _hit("q1", "p1", rank=1, score=0.9),
        _hit("q1", "p2", rank=2, score=0.8),
        _hit("q1", "p3", rank=3, score=0.7),
    ]
    result = compute_metrics(qrels, hits, metrics=["mrr@10"])
    assert math.isclose(result["mrr@10"], 1.0 / 3.0, rel_tol=1e-9)


def test_mrr_at_2_returns_zero_when_first_relevant_is_beyond_cutoff() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p3", relevance=1)]
    hits = [
        _hit("q1", "p1", rank=1, score=0.9),
        _hit("q1", "p2", rank=2, score=0.8),
        _hit("q1", "p3", rank=3, score=0.7),
    ]
    result = compute_metrics(qrels, hits, metrics=["mrr@2"])
    assert result["mrr@2"] == 0.0


def test_recall_at_100_for_partial_retrieval() -> None:
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=1),
        Qrel(query_id="q1", passage_id="p2", relevance=1),
        Qrel(query_id="q1", passage_id="p3", relevance=1),
        Qrel(query_id="q1", passage_id="p4", relevance=1),
    ]
    hits = [
        _hit("q1", "p1", rank=1, score=0.9),
        _hit("q1", "p2", rank=2, score=0.8),
        # plus three non-relevant hits
        _hit("q1", "pX", rank=3, score=0.7),
        _hit("q1", "pY", rank=4, score=0.6),
        _hit("q1", "pZ", rank=5, score=0.5),
    ]
    result = compute_metrics(qrels, hits, metrics=["recall@100"])
    assert math.isclose(result["recall@100"], 2.0 / 4.0, rel_tol=1e-9)


def test_ndcg_at_10_hand_computed_graded_example() -> None:
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=3),
        Qrel(query_id="q1", passage_id="p2", relevance=2),
        Qrel(query_id="q1", passage_id="p3", relevance=1),
    ]
    # Ranked order p2, p1, p3 (a small inversion vs. ideal).
    hits = [
        _hit("q1", "p2", rank=1, score=0.9),
        _hit("q1", "p1", rank=2, score=0.8),
        _hit("q1", "p3", rank=3, score=0.7),
    ]
    dcg = (
        (2**2 - 1) / math.log2(2)
        + (2**3 - 1) / math.log2(3)
        + (2**1 - 1) / math.log2(4)
    )
    idcg = (
        (2**3 - 1) / math.log2(2)
        + (2**2 - 1) / math.log2(3)
        + (2**1 - 1) / math.log2(4)
    )
    expected = dcg / idcg
    result = compute_metrics(qrels, hits, metrics=["ndcg@10"])
    assert math.isclose(result["ndcg@10"], expected, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Metric name handling
# ---------------------------------------------------------------------------


def test_metric_names_are_case_insensitive() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p1", relevance=1)]
    hits = [_hit("q1", "p1", rank=1, score=1.0)]
    result = compute_metrics(qrels, hits, metrics=["NDCG@10", "MRR@10", "Recall@100"])
    assert set(result.keys()) == {"ndcg@10", "mrr@10", "recall@100"}
    for v in result.values():
        assert math.isclose(v, 1.0, rel_tol=1e-9)


def test_unsupported_metric_raises_value_error() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p1", relevance=1)]
    hits = [_hit("q1", "p1", rank=1, score=1.0)]
    with pytest.raises(ValueError, match="Unsupported metric base"):
        compute_metrics(qrels, hits, metrics=["map@10"])


def test_metric_name_without_cutoff_raises_value_error() -> None:
    with pytest.raises(ValueError, match="expected '<base>@<cutoff>'"):
        compute_metrics([], [], metrics=["ndcg"])


def test_invalid_cutoff_raises_value_error() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p1", relevance=1)]
    hits = [_hit("q1", "p1", rank=1, score=1.0)]
    with pytest.raises(ValueError, match="must be >= 1"):
        compute_metrics(qrels, hits, metrics=["ndcg@0"])
    with pytest.raises(ValueError, match="must be >= 1"):
        compute_metrics(qrels, hits, metrics=["ndcg@-5"])
    with pytest.raises(ValueError, match="must be a positive integer"):
        compute_metrics(qrels, hits, metrics=["ndcg@abc"])


# ---------------------------------------------------------------------------
# Per-query view
# ---------------------------------------------------------------------------


def test_compute_per_query_metrics_includes_every_qrel_query() -> None:
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=1),
        Qrel(query_id="q2", passage_id="pX", relevance=1),
        Qrel(query_id="q3", passage_id="pY", relevance=2),
    ]
    hits = [_hit("q1", "p1", rank=1, score=1.0)]
    per_q = compute_per_query_metrics(qrels, hits)
    assert set(per_q.keys()) == {"q1", "q2", "q3"}
    for name in ("ndcg@10", "mrr@10", "recall@100"):
        assert name in per_q["q2"]
        assert name in per_q["q3"]


def test_compute_per_query_metrics_returns_zeros_for_query_with_no_hits() -> None:
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=1),
        Qrel(query_id="q2", passage_id="pX", relevance=1),
    ]
    hits = [_hit("q1", "p1", rank=1, score=1.0)]
    per_q = compute_per_query_metrics(qrels, hits)
    assert per_q["q2"] == {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@100": 0.0}


# ---------------------------------------------------------------------------
# Robustness to duplicate qrels / hits
# ---------------------------------------------------------------------------


def test_duplicate_hits_do_not_inflate_recall() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p1", relevance=1)]
    hits = [
        _hit("q1", "p1", rank=1, score=0.99),
        _hit("q1", "p1", rank=2, score=0.50),
        _hit("q1", "p1", rank=3, score=0.10),
    ]
    result = compute_metrics(qrels, hits, metrics=["recall@100"])
    assert math.isclose(result["recall@100"], 1.0, rel_tol=1e-9)


def test_duplicate_qrels_do_not_break_graded_ndcg() -> None:
    # Lower-graded duplicate must not override the higher grade for the same passage.
    qrels = [
        Qrel(query_id="q1", passage_id="p1", relevance=1),
        Qrel(query_id="q1", passage_id="p1", relevance=3),
    ]
    hits = [_hit("q1", "p1", rank=1, score=0.99)]
    result = compute_metrics(qrels, hits, metrics=["ndcg@10"])
    # With graded relevance 3 and a single perfect-ranking hit, nDCG is 1.0.
    assert math.isclose(result["ndcg@10"], 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_empty_qrels_returns_zero_aggregates_for_requested_metrics() -> None:
    hits = [_hit("q1", "p1", rank=1, score=1.0)]
    assert compute_metrics([], hits) == {
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
        "recall@100": 0.0,
    }


def test_empty_hits_with_nonempty_qrels_returns_zero_metrics() -> None:
    qrels = [Qrel(query_id="q1", passage_id="p1", relevance=1)]
    assert compute_metrics(qrels, []) == {
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
        "recall@100": 0.0,
    }


# ---------------------------------------------------------------------------
# Direct kernel sanity checks (they pin the one implementation)
# ---------------------------------------------------------------------------


def test_mrr_kernel_handles_no_relevant_in_topk() -> None:
    assert _mrr_at_k({"p1": 1}, ["p2", "p3"], 10) == 0.0


def test_recall_kernel_handles_no_relevant_docs_in_qrels() -> None:
    # All qrels are graded 0 → no relevant docs → recall must be 0.0.
    assert _recall_at_k({"p1": 0, "p2": 0}, ["p1", "p2"], 10) == 0.0


def test_ndcg_kernel_is_zero_when_idcg_is_zero() -> None:
    assert _ndcg_at_k({"p1": 0}, ["p1"], 10) == 0.0
