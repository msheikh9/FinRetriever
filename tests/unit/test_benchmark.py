"""Tests for finretriever.evaluation.benchmark.

These tests pin the contract the benchmark harness exposes to the
``04_run_baselines.py`` CLI and to future dense baselines:

* JSONL inputs are loaded and validated through the existing schemas.
* The benchmark runs end-to-end on a tiny obvious fixture and yields the
  expected perfect metrics — proof the wiring between BM25 and the metric
  kernels is correct.
* Output files have the exact shape consumed by the markdown table on
  ``results/final_ablation_table.md`` and the JSON consumed by humans.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finretriever.data.schemas import Passage, Qrel, Query, RunResult, SearchHit
from finretriever.evaluation.benchmark import (
    format_results_markdown,
    load_benchmark_inputs,
    load_metrics_from_config,
    load_top_k_from_config,
    results_to_jsonable,
    run_bm25_baseline,
    run_retriever_benchmark,
    write_results,
)
from finretriever.retrieval.bm25 import BM25Retriever
from finretriever.utils.io import write_jsonl

# ---------------------------------------------------------------------------
# Shared tiny fixture
# ---------------------------------------------------------------------------


def _make_passage(passage_id: str, ticker: str, text: str) -> Passage:
    return Passage(
        passage_id=passage_id,
        filing_id=f"filing_{ticker.lower()}",
        ticker=ticker,
        form_type="10-K",
        filing_date="2023-11-03",
        text=text,
    )


@pytest.fixture
def small_corpus() -> list[Passage]:
    return [
        _make_passage("p1", "AAPL", "Apple supply chain risk in semiconductors"),
        _make_passage("p2", "JPM", "JPMorgan credit risk and loan loss provisions"),
        _make_passage("p3", "TSLA", "Tesla manufacturing capacity Fremont Shanghai"),
    ]


@pytest.fixture
def small_queries() -> list[Query]:
    return [Query(query_id="q1", text="Apple supply chain", source="gold")]


@pytest.fixture
def small_qrels() -> list[Qrel]:
    return [Qrel(query_id="q1", passage_id="p1", relevance=2)]


# ---------------------------------------------------------------------------
# load_benchmark_inputs
# ---------------------------------------------------------------------------


def _write_corpus_files(
    tmp_path: Path,
    corpus: list[Passage],
    queries: list[Query],
    qrels: list[Qrel],
) -> tuple[Path, Path, Path]:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    write_jsonl(corpus_path, corpus)
    write_jsonl(queries_path, queries)
    write_jsonl(qrels_path, qrels)
    return corpus_path, queries_path, qrels_path


def test_load_benchmark_inputs_loads_models(
    tmp_path: Path,
    small_corpus: list[Passage],
    small_queries: list[Query],
    small_qrels: list[Qrel],
) -> None:
    corpus_path, queries_path, qrels_path = _write_corpus_files(
        tmp_path, small_corpus, small_queries, small_qrels
    )
    passages, queries, qrels = load_benchmark_inputs(
        corpus_path, queries_path, qrels_path
    )
    assert all(isinstance(p, Passage) for p in passages)
    assert all(isinstance(q, Query) for q in queries)
    assert all(isinstance(r, Qrel) for r in qrels)
    assert [p.passage_id for p in passages] == ["p1", "p2", "p3"]


def test_load_benchmark_inputs_rejects_empty_corpus(
    tmp_path: Path,
    small_queries: list[Query],
    small_qrels: list[Qrel],
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    corpus_path.write_text("", encoding="utf-8")
    write_jsonl(queries_path, small_queries)
    write_jsonl(qrels_path, small_qrels)
    with pytest.raises(ValueError, match="Empty corpus"):
        load_benchmark_inputs(corpus_path, queries_path, qrels_path)


def test_load_benchmark_inputs_rejects_empty_queries(
    tmp_path: Path,
    small_corpus: list[Passage],
    small_qrels: list[Qrel],
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    write_jsonl(corpus_path, small_corpus)
    queries_path.write_text("", encoding="utf-8")
    write_jsonl(qrels_path, small_qrels)
    with pytest.raises(ValueError, match="Empty query set"):
        load_benchmark_inputs(corpus_path, queries_path, qrels_path)


def test_load_benchmark_inputs_rejects_empty_qrels(
    tmp_path: Path,
    small_corpus: list[Passage],
    small_queries: list[Query],
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    write_jsonl(corpus_path, small_corpus)
    write_jsonl(queries_path, small_queries)
    qrels_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty qrels"):
        load_benchmark_inputs(corpus_path, queries_path, qrels_path)


# ---------------------------------------------------------------------------
# run_retriever_benchmark / run_bm25_baseline
# ---------------------------------------------------------------------------


def test_run_retriever_benchmark_returns_run_result(
    small_corpus: list[Passage],
    small_queries: list[Query],
    small_qrels: list[Qrel],
) -> None:
    result = run_retriever_benchmark(
        BM25Retriever(retriever_name="bm25"),
        small_corpus,
        small_queries,
        small_qrels,
        top_k=10,
    )
    assert isinstance(result, RunResult)
    assert all(isinstance(h, SearchHit) for h in result.hits)
    assert result.metadata["corpus_size"] == 3
    assert result.metadata["query_count"] == 1
    assert result.metadata["qrel_count"] == 1
    assert result.metadata["top_k"] == 10
    assert result.metadata["retriever"] == "bm25"


def test_run_retriever_benchmark_computes_perfect_metrics(
    small_corpus: list[Passage],
    small_queries: list[Query],
    small_qrels: list[Qrel],
) -> None:
    # The query terms "Apple", "supply", "chain" appear only in p1, so BM25
    # must rank it first and every default metric must hit 1.0.
    result = run_retriever_benchmark(
        BM25Retriever(retriever_name="bm25"),
        small_corpus,
        small_queries,
        small_qrels,
    )
    for name in ("ndcg@10", "mrr@10", "recall@100"):
        assert result.metrics[name] == pytest.approx(1.0, rel=1e-9, abs=1e-12)


def test_run_retriever_benchmark_validates_top_k(
    small_corpus: list[Passage],
    small_queries: list[Query],
    small_qrels: list[Qrel],
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        run_retriever_benchmark(
            BM25Retriever(retriever_name="bm25"),
            small_corpus,
            small_queries,
            small_qrels,
            top_k=0,
        )


def test_run_bm25_baseline_sets_run_name_to_bm25(
    small_corpus: list[Passage],
    small_queries: list[Query],
    small_qrels: list[Qrel],
) -> None:
    result = run_bm25_baseline(small_corpus, small_queries, small_qrels, top_k=10)
    assert result.run_name == "bm25"
    assert result.metadata["retriever"] == "bm25"


def test_run_retriever_benchmark_does_not_mutate_inputs(
    small_corpus: list[Passage],
    small_queries: list[Query],
    small_qrels: list[Qrel],
) -> None:
    corpus_snapshot = [p.model_dump() for p in small_corpus]
    queries_snapshot = [q.model_dump() for q in small_queries]
    qrels_snapshot = [r.model_dump() for r in small_qrels]
    run_retriever_benchmark(
        BM25Retriever(retriever_name="bm25"),
        small_corpus,
        small_queries,
        small_qrels,
        top_k=10,
    )
    assert [p.model_dump() for p in small_corpus] == corpus_snapshot
    assert [q.model_dump() for q in small_queries] == queries_snapshot
    assert [r.model_dump() for r in small_qrels] == qrels_snapshot


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _example_run_result() -> RunResult:
    return RunResult(
        run_name="bm25",
        metrics={"ndcg@10": 0.5123, "mrr@10": 0.6111, "recall@100": 0.7832},
        hits=[
            SearchHit(
                query_id="q1",
                passage_id="p1",
                rank=1,
                score=1.23,
                retriever="bm25",
            ),
        ],
        metadata={"corpus_size": 3, "query_count": 1, "qrel_count": 1, "top_k": 10},
    )


def test_results_to_jsonable_includes_all_required_fields() -> None:
    payload = results_to_jsonable(_example_run_result())
    assert set(payload.keys()) >= {"run_name", "metrics", "metadata", "hits"}
    assert payload["run_name"] == "bm25"
    assert isinstance(payload["hits"], list)
    assert isinstance(payload["hits"][0], dict)
    assert payload["hits"][0]["passage_id"] == "p1"
    assert payload["metrics"]["ndcg@10"] == pytest.approx(0.5123)


def test_write_results_writes_pretty_json(tmp_path: Path) -> None:
    output_json = tmp_path / "nested" / "out.json"
    write_results(_example_run_result(), output_json)
    assert output_json.exists()
    text = output_json.read_text(encoding="utf-8")
    # Pretty-printed JSON has indented lines, not a single one-liner.
    assert "\n  " in text
    payload = json.loads(text)
    assert payload["run_name"] == "bm25"
    assert payload["metrics"]["ndcg@10"] == pytest.approx(0.5123)


def test_write_results_writes_markdown_table_when_requested(tmp_path: Path) -> None:
    output_json = tmp_path / "out.json"
    output_table = tmp_path / "table.md"
    write_results(_example_run_result(), output_json, output_table)
    assert output_table.exists()
    md = output_table.read_text(encoding="utf-8")
    assert "# Baseline Results" in md
    assert "bm25" in md
    assert "ndcg@10" in md
    assert "0.5123" in md


def test_write_results_skips_markdown_when_no_table_path(tmp_path: Path) -> None:
    output_json = tmp_path / "out.json"
    write_results(_example_run_result(), output_json, None)
    assert output_json.exists()
    assert list(tmp_path.iterdir()) == [output_json]


def test_format_results_markdown_has_expected_columns_and_decimals() -> None:
    md = format_results_markdown([_example_run_result()])
    # Columns appear in the header row.
    assert "| run | ndcg@10 | mrr@10 | recall@100 |" in md
    # Floats formatted to 4 decimals.
    assert "0.5123" in md
    assert "0.6111" in md
    assert "0.7832" in md


def test_format_results_markdown_fills_missing_metrics_as_zero() -> None:
    partial = RunResult(
        run_name="bm25",
        metrics={"ndcg@10": 0.4},  # mrr@10 and recall@100 missing
    )
    md = format_results_markdown([partial])
    # Missing metrics render as 0.0000.
    assert "0.0000" in md
    assert "0.4000" in md


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def test_load_metrics_from_config_defaults() -> None:
    assert load_metrics_from_config({}) == ["ndcg@10", "mrr@10", "recall@100"]


def test_load_metrics_from_config_reads_from_config() -> None:
    assert load_metrics_from_config({"metrics": ["ndcg@5", "recall@50"]}) == [
        "ndcg@5",
        "recall@50",
    ]


def test_load_top_k_from_config_default_is_100() -> None:
    assert load_top_k_from_config({}) == 100


def test_load_top_k_from_config_reads_from_config() -> None:
    assert load_top_k_from_config({"top_k": 25}) == 25


def test_load_top_k_from_config_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        load_top_k_from_config({"top_k": 0})
    with pytest.raises(ValueError, match="positive integer"):
        load_top_k_from_config({"top_k": -5})
    with pytest.raises(ValueError, match="positive integer"):
        load_top_k_from_config({"top_k": "100"})
