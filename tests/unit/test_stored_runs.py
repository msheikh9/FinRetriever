"""Tests for reading stored retrieval runs back with their provenance checked (synthetic corpus, offline)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from finretriever.eval_set.annotation_store import load_corpus_passages, verify_corpus
from finretriever.evaluation.provenance import CorpusIdentity
from finretriever.evaluation.stored_runs import StoredRunError, load_stored_run
from tests.unit.eval_set_fixtures import passage_ids, write_corpus, write_stored_run

IDS = passage_ids()
RANKING = {"q_0001": ["acme24_supplier", "acme23_supplier", "beta24_credit"], "q_0002": ["beta24_liquidity"]}


@pytest.fixture
def corpus(tmp_path: Path) -> CorpusIdentity:
    return verify_corpus(write_corpus(tmp_path / "corpus"))


def stored(tmp_path: Path, corpus: CorpusIdentity, **kwargs: Any) -> Path:
    return write_stored_run(tmp_path / "runs", "global", "dense", RANKING, corpus_path=corpus.corpus_path, **kwargs)


def rewrite(run_dir: Path, change: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> None:
    """Edit run.jsonl and re-pin its SHA-256, as if the run had been written that way."""
    rows = change([json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()])
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (run_dir / "run.jsonl").write_bytes(data)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    metrics["run_file"]["sha256"] = hashlib.sha256(data).hexdigest()
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_a_stored_run_is_read_back_exactly(tmp_path: Path, corpus: CorpusIdentity) -> None:
    run_dir = stored(tmp_path, corpus)
    run = load_stored_run(tmp_path / "runs", "global", "dense", corpus=corpus, passages=load_corpus_passages(corpus))
    assert run.run_id == "global/dense" and run.top_k == 10
    assert [(h.query_id, h.rank, h.passage_id) for h in run.hits] == [
        (q, rank, IDS[name]) for q, names in RANKING.items() for rank, name in enumerate(names, start=1)
    ]
    assert run.hits[2].metadata == {"ticker": "BETA"} and run.hits[0].retriever == "dense"
    assert run.ranked(2)["q_0001"] == list(run.hits[:2])
    provenance = run.provenance()
    assert provenance["run_sha256"] == hashlib.sha256((run_dir / "run.jsonl").read_bytes()).hexdigest()
    assert provenance["metrics_sha256"] == hashlib.sha256((run_dir / "metrics.json").read_bytes()).hexdigest()
    assert (provenance["label_version"], provenance["run_file"]) == ("earlier_labels", str(run_dir / "run.jsonl"))


def test_modified_or_foreign_runs_are_refused(tmp_path: Path, corpus: CorpusIdentity) -> None:
    passages = load_corpus_passages(corpus)

    def load(mode: str = "global", run: str = "dense") -> Any:
        return load_stored_run(tmp_path / "runs", mode, run, corpus=corpus, passages=passages)

    run_dir = stored(tmp_path, corpus)
    with pytest.raises(StoredRunError, match="missing"):
        load("company")
    original = (run_dir / "run.jsonl").read_bytes()
    (run_dir / "run.jsonl").write_bytes(original.replace(b'"rank":2', b'"rank":3', 1))
    with pytest.raises(StoredRunError, match="was modified"):
        load()
    (run_dir / "run.jsonl").write_bytes(original)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    (run_dir / "metrics.json").write_text(json.dumps({**metrics, "mode": "company"}))
    with pytest.raises(StoredRunError, match="not 'dense' in 'global'"):
        load()
    (run_dir / "metrics.json").write_text(json.dumps({**metrics, "corpus": {"sha256": "f" * 64, "version": "x"}}))
    with pytest.raises(StoredRunError, match="was produced on corpus"):
        load()
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert len(load().hits) == 4


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: [rows[0], rows[2], rows[3]], "rank 3 follows rank 1"),
        (lambda rows: [rows[0], {**rows[1], "passage_id": rows[0]["passage_id"]}, rows[2], rows[3]], "ranked twice"),
        (lambda rows: [{**rows[0], "passage_id": "passage_unknown"}, *rows[1:]], "not in the corpus"),
        (lambda rows: [{**rows[0], "ticker": "BETA"}, *rows[1:]], "the corpus says ACME"),
        (lambda rows: [{k: v for k, v in rows[0].items() if k != "score"}, *rows[1:]], "not a ranked hit"),
    ],
)
def test_inconsistent_rankings_are_refused(
    tmp_path: Path, corpus: CorpusIdentity, change: Callable[[list[dict[str, Any]]], list[dict[str, Any]]], message: str
) -> None:
    rewrite(stored(tmp_path, corpus), change)
    with pytest.raises(StoredRunError, match=message):
        load_stored_run(tmp_path / "runs", "global", "dense", corpus=corpus, passages=load_corpus_passages(corpus))


def test_ranks_beyond_the_recorded_depth_are_refused(tmp_path: Path, corpus: CorpusIdentity) -> None:
    stored(tmp_path, corpus, top_k=2)
    with pytest.raises(StoredRunError, match="beyond the recorded top_k 2"):
        load_stored_run(tmp_path / "runs", "global", "dense", corpus=corpus, passages=load_corpus_passages(corpus))
