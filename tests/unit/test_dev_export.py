"""Tests for the ``dev-labels`` / ``export-dev`` workflow (synthetic corpus, offline)."""

from __future__ import annotations

import hashlib
import io
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from finretriever.data.schemas import Qrel, Query
from finretriever.eval_set.annotation_cli import main
from finretriever.eval_set.annotation_store import load_corpus_passages, verify_corpus
from finretriever.eval_set.dev_export import (
    DEV_METADATA_FILENAME,
    DEV_PROVENANCE_FILENAME,
    DEV_QRELS_FILENAME,
    DEV_QUERIES_FILENAME,
    DEV_REPORT_FILENAME,
    GOLD_FILENAMES,
    DevExportError,
    build_dev_export,
    write_dev_export,
)
from finretriever.eval_set.dev_labels import DevLabelSet
from finretriever.evaluation.provenance import load_evaluation_metadata, verify_evaluation_corpus
from tests.unit.eval_set_fixtures import DevInputs, build_dev_inputs, passage_ids

IDS = passage_ids()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def inputs(tmp_path: Path) -> DevInputs:
    return build_dev_inputs(tmp_path)


def cli(inputs: DevInputs, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main([argv[0], "--config", str(inputs.config), *argv[1:]], out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def file_hashes(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


def test_dev_labels_then_export_dev_is_reproducible_and_never_gold(inputs: DevInputs, tmp_path: Path) -> None:
    spec = inputs.write_spec()
    labels_dir = tmp_path / "dev_ai_v2"
    code, out, err = cli(inputs, "dev-labels", "--spec", str(spec))
    assert code == 0, err
    assert "wrote development labels test_v2" in out and "{'P2': 1}" in out
    assert "never as gold" in out
    code, out, err = cli(inputs, "dev-labels", "--spec", str(spec))
    assert code == 0 and "unchanged" in out  # re-deriving the same version is a no-op

    code, out, err = cli(inputs, "export-dev", "--labels", str(labels_dir))
    assert code == 0, err
    assert "provisional development labels, not gold" in out
    first = file_hashes(labels_dir)
    code, _, err = cli(inputs, "export-dev", "--labels", str(labels_dir))
    assert code == 0, err
    assert file_hashes(labels_dir) == first  # byte-identical re-export

    queries = [Query.model_validate_json(line) for line in (labels_dir / DEV_QUERIES_FILENAME).read_text().splitlines()]
    assert [q.source for q in queries] == ["dev_ai", "dev_ai"]
    assert {q.metadata["status"] for q in queries} == {"provisional_development"}
    qrels = [Qrel.model_validate_json(line) for line in (labels_dir / DEV_QRELS_FILENAME).read_text().splitlines()]
    assert len(qrels) == 10
    assert ("q_0002", IDS["acme24_table"]) not in {(r.query_id, r.passage_id) for r in qrels}  # stays unjudged
    provenance = [json.loads(line) for line in (labels_dir / DEV_PROVENANCE_FILENAME).read_text().splitlines()]
    assert [(p["query_id"], p["passage_id"], p["relevance"]) for p in provenance] == [
        (r.query_id, r.passage_id, r.relevance) for r in qrels
    ]
    assert sum(p["ai_generated"] for p in provenance) == 9
    (human,) = [p for p in provenance if not p["ai_generated"]]
    assert (human["label_source"], human["assistance"]) == ("human_judgment", "not_recorded")

    corpus_path = tmp_path / "corpus" / "corpus.jsonl"
    verify_evaluation_corpus(labels_dir / DEV_METADATA_FILENAME, corpus_path)  # bound to the frozen corpus
    assert load_evaluation_metadata(labels_dir / DEV_METADATA_FILENAME).annotation_version == "dev_test_v2"
    report = json.loads((labels_dir / DEV_REPORT_FILENAME).read_text())
    assert report["status"] == "provisional_development" and "never gold" in report["notice"]
    assert report["audit"]["independence"] == "ai_assisted"
    assert report["labels"]["by_source"] == {"ai_proposal": 9, "human_judgment": 1}
    assert report["labels"]["changed_by_rule"] == {"P2": 1} and report["labels"]["open_flags_by_rule"] == {"P1": 1}
    assert report["exceptions"] == {"no_positive": [], "no_grade_2": ["q_0002"]}  # explicit, not silent
    for name in (DEV_QUERIES_FILENAME, DEV_QRELS_FILENAME, DEV_PROVENANCE_FILENAME, DEV_METADATA_FILENAME):
        assert report["files"][name] == hashlib.sha256((labels_dir / name).read_bytes()).hexdigest()
    assert not any(p.name in GOLD_FILENAMES for p in tmp_path.rglob("*"))  # nothing gold was written


def test_export_dev_refuses_gold_and_protected_paths(inputs: DevInputs, tmp_path: Path) -> None:
    spec = inputs.write_spec()
    assert cli(inputs, "dev-labels", "--spec", str(spec))[0] == 0
    labels_dir = tmp_path / "dev_ai_v2"
    protected = [tmp_path / "export", tmp_path / "annotation", tmp_path / "dev_ai", tmp_path / "audit" / "sub"]
    before = {d: file_hashes(d) for d in (tmp_path / "annotation", tmp_path / "dev_ai", tmp_path / "audit")}
    for target in protected:
        code, _, err = cli(inputs, "export-dev", "--labels", str(labels_dir), "--out-dir", str(target))
        assert code == 1 and "refusing" in err, target
    assert {d: file_hashes(d) for d in before} == before
    assert not (tmp_path / "export").exists()

    spec_into_workspace = inputs.write_spec(output_dir=str(tmp_path / "annotation" / "dev"))
    code, _, err = cli(inputs, "dev-labels", "--spec", str(spec_into_workspace))
    assert code == 1 and "refusing" in err


def test_export_rejects_conflicts_unknown_passages_and_other_corpora(inputs: DevInputs, tmp_path: Path) -> None:
    assert cli(inputs, "dev-labels", "--spec", str(inputs.write_spec()))[0] == 0
    corpus = verify_corpus(tmp_path / "corpus" / "corpus.jsonl")
    passages = load_corpus_passages(corpus)
    label_set = DevLabelSet.open(tmp_path / "dev_ai_v2", corpus, passages)

    duplicate = DevLabelSet(label_set.info, label_set.questions, (*label_set.labels, label_set.labels[0]), (), ())
    with pytest.raises(DevExportError, match="conflicting labels"):
        build_dev_export(duplicate, passages)
    missing = {pid: p for pid, p in passages.items() if pid != IDS["beta24_credit"]}
    with pytest.raises(DevExportError, match="not in the corpus"):
        build_dev_export(label_set, missing)
    export = build_dev_export(label_set, passages)
    with pytest.raises(DevExportError, match="gold export directory"):
        write_dev_export(export, tmp_path / "export", gold_export_dir=tmp_path / "export")

    # A corpus with other bytes is refused before anything is read or written.
    corpus_file = tmp_path / "corpus" / "corpus.jsonl"
    corpus_file.write_bytes(corpus_file.read_bytes() + b"\n")
    code, _, err = cli(inputs, "export-dev", "--labels", str(tmp_path / "dev_ai_v2"))
    assert code == 2 and "error" in err


def test_questions_without_positive_evidence_are_flagged(inputs: DevInputs, tmp_path: Path) -> None:
    assert cli(inputs, "dev-labels", "--spec", str(inputs.write_spec()))[0] == 0
    corpus = verify_corpus(tmp_path / "corpus" / "corpus.jsonl")
    passages = load_corpus_passages(corpus)
    label_set = DevLabelSet.open(tmp_path / "dev_ai_v2", corpus, passages)
    # A version in which q_0002 has no positive label must say so, not drop the question silently.
    negatives = tuple(
        x.model_copy(update={"grade": 0, "base_grade": 0, "changed": False}) if x.query_id == "q_0002" else x
        for x in label_set.labels
    )
    export = build_dev_export(DevLabelSet(label_set.info, label_set.questions, negatives, (), ()), passages)
    assert [q.query_id for q in export.queries] == ["q_0001", "q_0002"]
    assert export.report["exceptions"]["no_positive"] == ["q_0002"]
    (row,) = [q for q in export.report["queries"] if q["query_id"] == "q_0002"]
    assert row["positives"] == 0 and "no_positive" in row["exceptions"]
