"""Unit tests for finretriever.eval_set.qrels_builder (synthetic corpus, offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finretriever.data.schemas import Qrel, Query, SearchHit
from finretriever.eval_set.annotation_store import AnnotationStore
from finretriever.eval_set.qrels_builder import (
    GoldExportError,
    GoldLeakageError,
    build_gold_export,
    check_no_gold_leakage,
    find_gold_overlap,
    judged_at_k,
    judgment_coverage,
    validate_gold_qrels,
    write_gold_export,
)
from finretriever.eval_set.schemas import Candidate, CandidateSource
from finretriever.evaluation.benchmark import load_benchmark_inputs
from finretriever.evaluation.provenance import verify_evaluation_corpus
from tests.unit.eval_set_fixtures import new_store, passage_ids

IDS = passage_ids()
BM25 = CandidateSource(method="bm25", rank=1, score=1.0, detail="query='x'")


def setup_question(store: AnnotationStore, text: str, grades: dict[str, int | None], *, status: str = "complete",
                   extra_pool: tuple[str, ...] = (), **kw) -> str:
    """A question whose pool is ``grades`` plus ``extra_pool``; ``None`` marks a skip."""
    values = {"topic": "risk_factors", "question_type": "semantic", "authoring": "independent", "author": "T1"}
    question = store.add_question(text, **{**values, **kw})
    qid = question.query_id
    names = [*grades, *extra_pool]
    store.add_candidates(qid, [Candidate(passage_id=IDS[n], source=BM25) for n in names], added_by="T1")
    for name, grade in grades.items():
        if grade is None:
            store.skip(qid, IDS[name], annotator="T1")
        else:
            store.judge(qid, IDS[name], grade, annotator="T1")
    if status != "open":
        store.set_status(qid, status, annotator="T1")
    return qid


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    store = new_store(tmp_path)
    setup_question(  # q_0001: complete, positives, a 0 and a skip
        store, "How dependent is Acme on one supplier?",
        {"acme24_supplier": 2, "acme23_supplier": 1, "acme24_competition": 0, "acme24_sales": None},
        tickers=["ACME"],
    )
    setup_question(  # q_0002: complete but nothing relevant was found
        store, "Does Beta disclose cyber incidents?", {"beta24_rates": 0, "beta24_business": 0},
        topic="other", tickers=["BETA"],
    )
    setup_question(  # q_0003: still open, one candidate pending
        store, "What were Acme's net sales in 2024?", {"acme24_table": 2},
        status="open", extra_pool=("acme24_sales",), topic="revenue_profitability", question_type="table",
        comparison="cross_year", fiscal_years=[2024],
    )
    setup_question(  # q_0004: withdrawn
        store, "How large is Beta's liquidity buffer?", {"beta24_liquidity": 2},
        status="withdrawn", topic="debt_liquidity", authoring="passage_inspired",
    )
    return store


def test_export_holds_only_graded_passages_of_complete_questions(store: AnnotationStore) -> None:
    export = build_gold_export(store)
    assert [q.query_id for q in export.queries] == ["q_0001"]
    assert {(r.passage_id, r.relevance) for r in export.qrels} == {
        (IDS["acme24_supplier"], 2), (IDS["acme23_supplier"], 1), (IDS["acme24_competition"], 0),
    }
    assert IDS["acme24_sales"] not in {r.passage_id for r in export.qrels}  # skipped stays unjudged
    assert export.excluded == {
        "q_0002": "no positive judgment", "q_0003": "not marked complete", "q_0004": "withdrawn",
    }
    assert export.unjudged_in_pool == {"q_0001": 1}
    query = export.queries[0]
    assert query.source == "gold"
    assert query.metadata["tickers"] == "ACME" and query.metadata["annotation_version"] == "gold_test"


def test_include_open_exports_open_questions_with_positives(store: AnnotationStore) -> None:
    export = build_gold_export(store, include_open=True)
    assert [q.query_id for q in export.queries] == ["q_0001", "q_0003"]
    assert [(r.query_id, r.relevance) for r in export.qrels if r.query_id == "q_0003"] == [("q_0003", 2)]


def test_written_export_loads_into_the_benchmark(tmp_path: Path, store: AnnotationStore) -> None:
    paths = write_gold_export(build_gold_export(store), tmp_path / "eval")
    _, queries, qrels = load_benchmark_inputs(store.corpus.corpus_path, paths["queries"], paths["qrels"])
    assert len(queries) == 1 and len(qrels) == 3
    assert verify_evaluation_corpus(paths["metadata"], store.corpus.corpus_path) == store.corpus
    report = json.loads(paths["report"].read_text())
    assert report["exported"]["grades"] == {"0": 1, "1": 1, "2": 1}
    assert report["excluded"]["q_0002"] == "no positive judgment"
    assert "not a judgment" in report["unjudged_policy"]


def test_export_is_deterministic(tmp_path: Path, store: AnnotationStore) -> None:
    first = write_gold_export(build_gold_export(store), tmp_path / "a")
    second = write_gold_export(build_gold_export(store), tmp_path / "b")
    assert all(first[k].read_bytes() == second[k].read_bytes() for k in first)


def test_nothing_to_export_is_an_error(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    setup_question(store, "Is Acme profitable?", {"acme24_sales": None}, status="open")
    with pytest.raises(GoldExportError, match="nothing to export"):
        write_gold_export(build_gold_export(store), tmp_path / "eval")


@pytest.mark.parametrize(
    ("queries", "qrels", "message"),
    [
        (["q_0001", "q_0001"], [("q_0001", "acme24_supplier", 2)], "duplicate query IDs"),
        (["q_0001"], [("q_0001", "acme24_supplier", 2), ("q_0001", "acme24_supplier", 0)], "more than one judgment"),
        (["q_0001"], [("q_0001", "acme24_supplier", 2), ("q_0002", "acme24_sales", 1)], "not in the query set"),
        (["q_0001"], [("q_0001", None, 2)], "not in the corpus"),
        (["q_0001"], [("q_0001", "acme24_supplier", 3)], "relevance grades outside"),
        (["q_0001", "q_0002"], [("q_0001", "acme24_supplier", 2)], "without any judgment"),
    ],
)
def test_validation_rejects_inconsistent_qrels(queries, qrels, message: str) -> None:
    query_rows = [Query(query_id=qid, text=f"question {i}", source="gold") for i, qid in enumerate(queries)]
    qrel_rows = [Qrel(query_id=q, passage_id=IDS[p] if p else "passage_gone", relevance=r) for q, p, r in qrels]
    with pytest.raises(GoldExportError, match=message):
        validate_gold_qrels(query_rows, qrel_rows, {pid: None for pid in IDS.values()})


def test_coverage_diagnostics(store: AnnotationStore) -> None:
    report = judgment_coverage(store, small_pool_below=3, target_queries=100)
    rows = {r.query_id: r for r in report.queries}
    q1, q2, q3, q4 = (rows[f"q_000{i}"] for i in range(1, 5))
    assert (q1.pool_size, q1.graded, q1.skipped, q1.pending, q1.grades, q1.positives) == (4, 3, 1, 0, {0: 1, 1: 1, 2: 1}, 2)
    assert q1.flags == ("skipped_candidates",)
    assert q2.flags == ("small_pool", "no_positive") and q2.grades == {0: 2, 1: 0, 2: 0}
    assert q3.flags == ("small_pool", "pending_candidates") and q3.pending == 1
    assert q4.flags == ()  # withdrawn questions are not flagged
    assert q1.sources == {"bm25": 4}
    assert report.flagged["no_positive"] == ["q_0002"]
    assert report.flagged["small_pool"] == ["q_0002", "q_0003"]
    assert report.status_counts == {"complete": 2, "open": 1, "withdrawn": 1}
    assert report.totals == {
        "pooled": 9, "graded": 7, "skipped": 1, "pending": 1, "grade_0": 3, "grade_1": 1, "grade_2": 3,
    }
    assert report.distribution["topic"]["risk_factors"] == 1  # withdrawn q_0004 not counted
    assert report.distribution["comparison"] == {"none": 2, "cross_year": 1, "cross_company": 0}
    assert report.distribution["authoring"] == {"independent": 3, "passage_inspired": 0}
    assert report.to_dict()["queries"][0]["positives"] == 2


def test_ai_assisted_provenance_is_exported(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    setup_question(
        store, "Which supplier risks does Acme describe?", {"acme24_supplier": 2}, tickers=["ACME"],
        authoring="passage_inspired", ai_assisted=True, reviewed_by="RV", author="claude",
    )
    setup_question(store, "Does Beta report credit losses?", {"beta24_credit": 2}, tickers=["BETA"])
    export = build_gold_export(store)
    assert [(q.metadata["ai_assisted"], q.metadata["reviewed_by"]) for q in export.queries] == [(True, "RV"), (False, "")]
    assert export.coverage.distribution["ai_assisted"] == {"yes": 1, "no": 1}


def test_judged_at_k_measures_unjudged_hits() -> None:
    qrels = [Qrel(query_id="q1", passage_id="a", relevance=2), Qrel(query_id="q1", passage_id="b", relevance=0),
             Qrel(query_id="q2", passage_id="c", relevance=1)]

    def hit(qid: str, pid: str, rank: int) -> SearchHit:
        return SearchHit(query_id=qid, passage_id=pid, rank=rank, score=1.0 / rank, retriever="bm25")

    hits = [hit("q1", "a", 1), hit("q1", "x", 2), hit("q1", "b", 3), hit("q1", "y", 4)]
    assert judged_at_k(qrels, hits, k=2) == {"q1": 0.5, "q2": 0.0}
    assert judged_at_k(qrels, hits, k=10) == {"q1": 0.5, "q2": 0.0}
    with pytest.raises(ValueError):
        judged_at_k(qrels, hits, k=0)


def test_gold_questions_are_detected_in_training_text() -> None:
    gold = [Query(query_id="q_0001", text="How dependent is Acme on one semiconductor supplier?", source="gold")]
    texts = [
        "How dependent is Acme on one semiconductor supplier?",
        "how DEPENDENT is acme on one semiconductor supplier",
        "Is Acme dependent on one semiconductor supplier?",
        "What was Beta's allowance for credit losses?",
    ]
    assert [(o.query_id, o.index) for o in find_gold_overlap(gold, texts)] == [("q_0001", 0), ("q_0001", 1), ("q_0001", 2)]
    with pytest.raises(GoldLeakageError, match="3 training texts overlap gold questions"):
        check_no_gold_leakage(gold, texts)
    check_no_gold_leakage(gold, texts[3:])
