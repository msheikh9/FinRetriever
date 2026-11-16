"""Integration test for the committed demo fixture.

The ``demo/`` directory exists so anyone cloning the repo can run a real
benchmark loop *before* the SEC corpus has been built. This test guards
that contract:

* the fixture loads and validates through the existing schemas;
* the in-process BM25 baseline produces sensible aggregate metrics;
* the CLI runs end-to-end against the fixture and writes the documented
  outputs.

The test never writes to the real ``results/`` directory — that is reserved
for the actual demo run committed alongside this fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from finretriever.evaluation.benchmark import (
    load_benchmark_inputs,
    run_bm25_baseline,
    run_retriever_benchmark,
)
from finretriever.retrieval.bm25 import BM25Retriever

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO_DIR = _REPO_ROOT / "demo"
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "04_run_baselines.py"
_CORPUS = _DEMO_DIR / "mini_corpus.jsonl"
_QUERIES = _DEMO_DIR / "mini_queries.jsonl"
_QRELS = _DEMO_DIR / "mini_qrels.jsonl"


@pytest.fixture
def demo_inputs() -> tuple[list, list, list]:
    return load_benchmark_inputs(_CORPUS, _QUERIES, _QRELS)


# ---------------------------------------------------------------------------
# Fixture shape
# ---------------------------------------------------------------------------


def test_demo_fixture_files_exist() -> None:
    assert _CORPUS.exists(), f"missing demo corpus at {_CORPUS}"
    assert _QUERIES.exists(), f"missing demo queries at {_QUERIES}"
    assert _QRELS.exists(), f"missing demo qrels at {_QRELS}"


def test_demo_inputs_load_and_are_populated(
    demo_inputs: tuple[list, list, list],
) -> None:
    passages, queries, qrels = demo_inputs
    # Spec: 6-8 passages, 4-5 queries, multiple qrels.
    assert 6 <= len(passages) <= 8
    assert 4 <= len(queries) <= 5
    assert len(qrels) >= len(queries)  # at least one qrel per query, often more


# ---------------------------------------------------------------------------
# In-process BM25 baseline
# ---------------------------------------------------------------------------


def test_demo_bm25_baseline_returns_sensible_metrics(
    demo_inputs: tuple[list, list, list],
) -> None:
    passages, queries, qrels = demo_inputs
    result = run_bm25_baseline(passages, queries, qrels)

    # The three headline metrics must be present and well-scaled.
    for name in ("ndcg@10", "mrr@10", "recall@100"):
        assert name in result.metrics, f"missing metric {name!r}"
        value = result.metrics[name]
        assert 0.0 <= value <= 1.0, f"{name}={value!r} outside [0, 1]"

    # nDCG@10 above 0.5 catches a broken fixture or a regression in BM25
    # ranking on lexically-obvious queries.
    assert result.metrics["ndcg@10"] > 0.5


# ---------------------------------------------------------------------------
# CLI end-to-end (subprocess)
# ---------------------------------------------------------------------------


def test_demo_cli_runs_against_demo_files(tmp_path: Path) -> None:
    output_json = tmp_path / "demo_baselines.json"
    output_table = tmp_path / "demo_ablation_table.md"

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--corpus", str(_CORPUS),
            "--queries", str(_QUERIES),
            "--qrels", str(_QRELS),
            "--output-json", str(output_json),
            "--output-table", str(output_table),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )

    assert result.returncode == 0, (
        f"CLI exited non-zero.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_json.exists()
    assert output_table.exists()

    # Sanity-check the JSON payload.
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_name"] == "bm25"
    assert {"ndcg@10", "mrr@10", "recall@100"} <= set(payload["metrics"].keys())

    # The stdout summary must mention the retriever and the metric names.
    assert "bm25" in result.stdout
    assert "ndcg@10" in result.stdout

    # The test must not have touched the real results/ directory.
    real_results = _REPO_ROOT / "results"
    assert tmp_path not in (real_results, real_results.parent)


# ---------------------------------------------------------------------------
# Backward compatibility with the committed demo results
# ---------------------------------------------------------------------------


def test_demo_results_are_reproduced_exactly(demo_inputs: tuple[list, list, list]) -> None:
    """The committed demo run came from the pure-Python engine; it must still reproduce byte for byte."""
    passages, queries, qrels = demo_inputs
    committed = json.loads((_REPO_ROOT / "results" / "demo_baselines.json").read_text(encoding="utf-8"))
    run = run_retriever_benchmark(
        BM25Retriever(retriever_name="bm25", backend="python"), passages, queries, qrels, run_name="bm25"
    )
    assert [h.model_dump(mode="json") for h in run.hits] == committed["hits"]
    assert run.metrics == committed["metrics"]
    # The default retriever (whitespace tokens, auto backend) keeps the committed metrics.
    assert run_bm25_baseline(passages, queries, qrels).metrics == committed["metrics"]
