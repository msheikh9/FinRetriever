"""Integration tests for scripts/04_run_baselines.py.

Runs the script as a subprocess against a tiny tmp_path fixture so the test
exercises argument parsing, the sys.path bootstrap, and the full
config → load → BM25 → write pipeline exactly as a user would.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.checkpoint_fixtures import candidate, write_checkpoint, write_training_side
from tests.unit.eval_set_fixtures import passage_ids, write_dev_eval_set

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "04_run_baselines.py"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record))
            fh.write("\n")


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    config_path = tmp_path / "config.yaml"

    _write_jsonl(
        corpus_path,
        [
            {
                "passage_id": "p1",
                "filing_id": "filing_aapl",
                "ticker": "AAPL",
                "form_type": "10-K",
                "filing_date": "2023-11-03",
                "text": "Apple supply chain risk in semiconductors",
                "metadata": {},
            },
            {
                "passage_id": "p2",
                "filing_id": "filing_jpm",
                "ticker": "JPM",
                "form_type": "10-K",
                "filing_date": "2023-02-21",
                "text": "JPMorgan credit risk and loan loss provisions",
                "metadata": {},
            },
            {
                "passage_id": "p3",
                "filing_id": "filing_tsla",
                "ticker": "TSLA",
                "form_type": "10-K",
                "filing_date": "2024-01-26",
                "text": "Tesla manufacturing capacity Fremont Shanghai",
                "metadata": {},
            },
        ],
    )
    _write_jsonl(
        queries_path,
        [
            {
                "query_id": "q1",
                "text": "Apple supply chain",
                "source": "gold",
                "metadata": {},
            },
        ],
    )
    _write_jsonl(
        qrels_path,
        [{"query_id": "q1", "passage_id": "p1", "relevance": 2}],
    )
    config_path.write_text(
        "metrics:\n"
        "  - ndcg@10\n"
        "  - mrr@10\n"
        "  - recall@100\n"
        "top_k: 10\n",
        encoding="utf-8",
    )
    return corpus_path, queries_path, qrels_path, config_path


def _run_script(
    args: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
        env=env,
    )


def _offline_env(tmp_path: Path) -> dict[str, str]:
    """No network and an empty Hugging Face cache, so a dense model can never be fetched or loaded."""
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "hf_home"),
    }


@pytest.fixture
def fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    return _build_fixture(tmp_path)


def test_script_writes_json_and_markdown(
    tmp_path: Path,
    fixture_paths: tuple[Path, Path, Path, Path],
) -> None:
    corpus_path, queries_path, qrels_path, config_path = fixture_paths
    output_json = tmp_path / "baselines.json"
    output_table = tmp_path / "table.md"

    result = _run_script(
        [
            "--corpus", str(corpus_path),
            "--queries", str(queries_path),
            "--qrels", str(qrels_path),
            "--config", str(config_path),
            "--output-json", str(output_json),
            "--output-table", str(output_table),
        ]
    )

    assert result.returncode == 0, (
        f"Script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_json.exists()
    assert output_table.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_name"] == "bm25"
    assert {"ndcg@10", "mrr@10", "recall@100"} <= set(payload["metrics"].keys())

    # Stdout summary mentions the retriever and the metric names.
    assert "bm25" in result.stdout
    assert "ndcg@10" in result.stdout
    assert "mrr@10" in result.stdout
    assert "recall@100" in result.stdout

    md = output_table.read_text(encoding="utf-8")
    assert "# Baseline Results" in md
    assert "bm25" in md


def test_script_no_table_flag_skips_markdown(
    tmp_path: Path,
    fixture_paths: tuple[Path, Path, Path, Path],
) -> None:
    corpus_path, queries_path, qrels_path, config_path = fixture_paths
    output_json = tmp_path / "baselines.json"
    output_table = tmp_path / "table.md"

    result = _run_script(
        [
            "--corpus", str(corpus_path),
            "--queries", str(queries_path),
            "--qrels", str(qrels_path),
            "--config", str(config_path),
            "--output-json", str(output_json),
            "--output-table", str(output_table),
            "--no-table",
        ]
    )

    assert result.returncode == 0, (
        f"Script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output_json.exists()
    assert not output_table.exists()


def test_script_top_k_override(
    tmp_path: Path,
    fixture_paths: tuple[Path, Path, Path, Path],
) -> None:
    corpus_path, queries_path, qrels_path, config_path = fixture_paths
    output_json = tmp_path / "baselines.json"

    result = _run_script(
        [
            "--corpus", str(corpus_path),
            "--queries", str(queries_path),
            "--qrels", str(qrels_path),
            "--config", str(config_path),
            "--output-json", str(output_json),
            "--top-k", "3",
            "--no-table",
        ]
    )

    assert result.returncode == 0, (
        f"Script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["metadata"]["top_k"] == 3


# ---------------------------------------------------------------------------
# Named evaluation sets (--eval-set) and config-driven defaults
# ---------------------------------------------------------------------------


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def test_eval_set_mode_writes_provisional_results_under_results_dev(tmp_path: Path) -> None:
    dev = write_dev_eval_set(tmp_path)
    committed = sorted((_REPO_ROOT / "results").glob("*.*"))
    gold = sorted((_REPO_ROOT / "data" / "eval").glob("gold_*"))
    before = _hashes(committed + gold)

    result = _run_script(
        ["--config", str(dev["config_path"]), "--eval-set", "dev"], env=_offline_env(tmp_path)
    )

    # Exit 1: results written, but the dense model could not be loaded (offline, empty cache).
    assert result.returncode == 1, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "PROVISIONAL DEVELOPMENT METRICS" in result.stdout
    assert "run=bm25 mode=global ndcg@10=" in result.stdout and "judged@10=" in result.stdout
    # Without the model backend the import fails; with it, the empty offline cache has no model.
    assert re.search(r"skipped bge_zero_shot: (cannot|could not) load BAAI/bge-base-en-v1\.5", result.stdout)
    out = tmp_path / "results" / "dev" / "test_v2"
    assert (out / "global" / "bm25" / "metrics.json").exists() and (out / "report.md").exists()
    payload = json.loads((out / "global" / "bm25" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "provisional_development" and len(payload["per_query"]) == 2
    assert payload["mode"] == "global" and payload["skipped_retrievers"][0]["name"] == "bge_zero_shot"
    assert not (tmp_path / "hf_home").exists() or not list((tmp_path / "hf_home").rglob("*.safetensors"))
    assert not list(tmp_path.rglob("baselines.json")) and not list(tmp_path.rglob("final_ablation_table.md"))
    assert _hashes(committed + gold) == before  # the committed results and gold files are untouched


def test_eval_set_mode_refuses_per_run_flags_and_corpus_mismatch(tmp_path: Path) -> None:
    dev = write_dev_eval_set(tmp_path)
    result = _run_script(["--config", str(dev["config_path"]), "--eval-set", "dev", "--corpus", "x.jsonl"])
    assert result.returncode == 2 and "remove --corpus" in result.stderr

    metadata = dev["labels"] / "dev_eval.metadata.json"
    metadata.write_text(json.dumps({**json.loads(metadata.read_text()), "corpus_sha256": "f" * 64}))
    result = _run_script(
        ["--config", str(dev["config_path"]), "--eval-set", "dev"], env=_offline_env(tmp_path)
    )
    assert result.returncode == 2 and "error" in result.stderr
    assert not (tmp_path / "results").exists()


def test_legacy_mode_reads_omitted_paths_from_config(
    tmp_path: Path,
    fixture_paths: tuple[Path, Path, Path, Path],
) -> None:
    corpus_path, queries_path, qrels_path, config_path = fixture_paths
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + f"corpus_path: {corpus_path}\ngold_queries_path: {queries_path}\ngold_qrels_path: {qrels_path}\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "baselines.json"
    result = _run_script(["--config", str(config_path), "--output-json", str(output_json), "--no-table"])
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_name"] == "bm25" and payload["metadata"]["corpus_size"] == 3


def test_check_only_inspects_checkpoints_without_loading_or_writing(tmp_path: Path) -> None:
    dev = write_dev_eval_set(tmp_path)
    judged = sorted({json.loads(line)["passage_id"] for line in (dev["labels"] / "dev_qrels.jsonl").read_text().splitlines()})
    inputs = write_training_side(tmp_path / "training", Path(dev["config"]["corpus_path"]), blocked=judged,
                                 candidates=[candidate("train_000001", passage_ids()["acme24_table"], "What does the table show?")])
    checkpoint = write_checkpoint(tmp_path / "models" / "fixture-seed42", 5, inputs)
    config = {**dev["config"], "baselines": dev["config"]["baselines"][:1]}  # BM25 only: the stored reference
    stored = tmp_path / "results" / "dev" / "test_v2"
    config["evaluation_sets"] = {
        **config["evaluation_sets"],
        "ckpt": {**config["evaluation_sets"]["dev"], "reference_runs_from": str(stored), "paired_reference": "bm25",
                 "retrievers": [{"name": "ckpt5", "type": "dense_checkpoint", "checkpoint_path": str(checkpoint)}],
                 "output_dir": str(tmp_path / "results" / "dev" / "test_ckpt")},
    }
    config_path = tmp_path / "benchmark_ckpt.yaml"
    config_path.write_text(json.dumps(config))
    env = _offline_env(tmp_path)
    assert _run_script(["--config", str(config_path), "--eval-set", "dev"], env=env).returncode == 0

    result = _run_script(["--config", str(config_path), "--eval-set", "ckpt", "--check-only"], env=env)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "retriever=ckpt5 type=dense_checkpoint" in result.stdout and "step=5/15" in result.stdout
    assert "evaluation isolation: 4 of 4 judged passages on the training blocklist" in result.stdout
    assert "stored reference run=bm25 mode=global ndcg@10=" in result.stdout
    assert "no model was loaded and nothing was written" in result.stdout
    assert not (tmp_path / "results" / "dev" / "test_ckpt").exists()

    # Results are written once: the dev set run again over an edited report changes nothing.
    report = stored / "report.md"
    report.write_text(report.read_text() + "edited\n")
    before = _hashes(sorted(p for p in stored.rglob("*") if p.is_file()))
    result = _run_script(["--config", str(config_path), "--eval-set", "dev"], env=env)
    assert result.returncode == 2 and "never overwritten" in result.stderr
    assert _hashes(sorted(p for p in stored.rglob("*") if p.is_file())) == before
