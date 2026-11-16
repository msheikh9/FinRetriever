"""Tests for versioned development labels (synthetic corpus, offline)."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from finretriever.eval_set.ai_proposals import PROPOSALS_FILE
from finretriever.eval_set.annotation_store import load_corpus_passages, verify_corpus
from finretriever.eval_set.dev_labels import (
    CHANGES_FILE,
    FLAGS_FILE,
    LABELS_FILE,
    LABELS_INFO_FILE,
    DevLabelError,
    DevLabelSet,
    DevLabelSpec,
    derive_dev_labels,
)
from finretriever.evaluation.provenance import CorpusMismatchError
from tests.unit.eval_set_fixtures import DevInputs, build_dev_inputs, passage_ids, write_corpus

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


def derive(inputs: DevInputs, **overrides: Any) -> DevLabelSet:
    spec = DevLabelSpec.model_validate(inputs.spec(**overrides))
    return derive_dev_labels(
        spec, spec_path="dev_labels.yaml", spec_sha256="0" * 64,
        store=inputs.store, proposals=inputs.proposals, audit=inputs.audit,
    )


def hashes(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


def test_derivation_is_reproducible_and_records_every_change(inputs: DevInputs) -> None:
    before = hashes(inputs.proposals.directory)
    first, second = derive(inputs), derive(inputs)
    assert first.rendered() == second.rendered()  # same inputs, byte-identical files

    labels = {(x.query_id, x.passage_id): x for x in first.labels}
    assert len(labels) == 10  # 9 AI proposals + 1 human grade; the unproposed acme24_table is not invented
    assert ("q_0002", IDS["acme24_table"]) not in labels
    human = labels["q_0001", IDS["acme24_competition"]]
    assert (human.label_source, human.ai_generated, human.assistance, human.labelled_by) == (
        "human_judgment", False, "not_recorded", "RV",
    )
    ai = labels["q_0001", IDS["acme24_supplier"]]
    assert (ai.label_source, ai.ai_generated, ai.labelled_by, ai.changed) == ("ai_proposal", True, "claude-opus-5-5", False)

    (change,) = first.changes
    assert (change.query_id, change.passage_id) == ("q_0002", IDS["beta24_liquidity"])
    assert (change.previous_grade, change.new_grade, change.rule) == (2, 1, "P2")
    assert change.check == {"type": "p2_basis", "requires_any": ["December 31, 2024"], "matched": []}
    assert change.previous_explanation == "EXPLANATION-beta24_liquidity"
    assert "P2" in change.reason and "dev_labels.yaml" in change.applied_by
    assert change.evidence and "evidence, not validation" in change.evidence[0]
    changed = labels["q_0002", IDS["beta24_liquidity"]]
    assert (changed.grade, changed.base_grade, changed.changed) == (1, 2, True)
    assert hashes(inputs.proposals.directory) == before  # the base proposals are never touched


def test_flags_and_audit_grades_never_change_grades(inputs: DevInputs) -> None:
    label_set = derive(inputs)
    labels = {(x.query_id, x.passage_id): x for x in label_set.labels}
    (flag,) = label_set.flags
    assert (flag.rule, flag.origin, flag.current_grade, flag.candidate_grade) == ("P1", "rule", 1, 2)
    sales = labels["q_0001", IDS["acme24_sales"]]
    assert sales.grade == 1 and sales.flags == [flag.flag_id]  # the audit says 2: evidence, not a label
    assert sales.audit is not None and (sales.audit.human_grade, sales.audit.agrees_with_base) == (2, False)
    assert sales.audit.independence == "ai_assisted"
    older = labels["q_0001", IDS["acme23_supplier"]]
    assert older.grade == 0 and older.audit.human_grade == 1  # kept under P4, not overwritten
    dispositions = {d.item: d for d in label_set.info.audit_dispositions}
    assert {d.disposition for d in dispositions.values()} == {"changed", "flagged", "kept"}
    assert dispositions[inputs.audit_items["q_0001", "acme23_supplier"]].rule == "P4"


def test_every_audit_disagreement_needs_an_explicit_disposition(inputs: DevInputs) -> None:
    with pytest.raises(DevLabelError, match="no rule, flag or kept_audit_disagreements entry"):
        derive(inputs, kept_audit_disagreements=[])
    item = inputs.audit_items["q_0001", "acme24_supplier"]  # the auditor agreed with this one
    stale = [*inputs.spec()["kept_audit_disagreements"], {"item": item, "rule": "P3", "reason": "not a disagreement"}]
    with pytest.raises(DevLabelError, match="not open disagreements"):
        derive(inputs, kept_audit_disagreements=stale)


def test_spec_flags_are_validated(inputs: DevInputs) -> None:
    unlabelled = {"query_id": "q_0002", "passage_id": IDS["acme24_table"], "rule": "P5", "candidate_grade": 2, "reason": "x"}
    with pytest.raises(DevLabelError, match="unlabelled passage"):
        derive(inputs, flags=[unlabelled])
    same = {"query_id": "q_0001", "passage_id": IDS["acme24_supplier"], "rule": "P5", "candidate_grade": 2, "reason": "x"}
    with pytest.raises(DevLabelError, match="already has"):
        derive(inputs, flags=[same])
    with pytest.raises(ValueError, match="lowercase"):
        DevLabelSpec.model_validate(inputs.spec(label_version="Pilot V2"))


def test_conflicting_human_and_ai_labels_are_rejected(inputs: DevInputs) -> None:
    inputs.store.judge("q_0002", IDS["beta24_rates"], 1, annotator="RV")  # a human grade on a proposed passage
    with pytest.raises(DevLabelError, match="conflicting labels"):
        derive(inputs)


def test_written_set_opens_and_detects_tampering(inputs: DevInputs, tmp_path: Path) -> None:
    corpus = verify_corpus(tmp_path / "corpus" / "corpus.jsonl")
    passages = load_corpus_passages(corpus)
    label_set = derive(inputs)
    out = tmp_path / "dev_ai_v2"
    assert label_set.write(out) is True
    assert label_set.write(out) is False  # identical copy: nothing rewritten
    reopened = DevLabelSet.open(out, corpus, passages)
    assert reopened.rendered() == label_set.rendered()
    reopened.verify_inputs()

    other = derive(inputs, description="a different derivation")
    with pytest.raises(DevLabelError, match="immutable"):
        other.write(out)

    labels_path = out / LABELS_FILE
    original = labels_path.read_bytes()
    labels_path.write_bytes(original.replace(b'"grade":0', b'"grade":2', 1))
    with pytest.raises(DevLabelError, match="was modified"):
        DevLabelSet.open(out, corpus, passages)

    # Even with matching hashes, a grade that changed without a change record is refused.
    info = json.loads((out / LABELS_INFO_FILE).read_text())
    info["files"][LABELS_FILE] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    (out / LABELS_INFO_FILE).write_text(json.dumps(info))
    with pytest.raises(DevLabelError, match="'changed' disagrees|without a change record"):
        DevLabelSet.open(out, corpus, passages)


def test_open_rejects_unknown_passages_and_orphan_changes(inputs: DevInputs, tmp_path: Path) -> None:
    corpus = verify_corpus(tmp_path / "corpus" / "corpus.jsonl")
    passages = load_corpus_passages(corpus)
    out = tmp_path / "dev_ai_v2"
    derive(inputs).write(out)
    info_path = out / LABELS_INFO_FILE

    def rewrite(name: str, data: bytes) -> None:
        (out / name).write_bytes(data)
        info = json.loads(info_path.read_text())
        info["files"][name] = hashlib.sha256(data).hexdigest()
        info_path.write_text(json.dumps(info))

    labels = (out / LABELS_FILE).read_bytes()
    first = json.loads(labels.splitlines()[0])
    rewrite(LABELS_FILE, labels + json.dumps({**first, "passage_id": "passage_unknown"}).encode() + b"\n")
    with pytest.raises(DevLabelError, match="not in the corpus"):
        DevLabelSet.open(out, corpus, passages)
    rewrite(LABELS_FILE, labels)

    flags = (out / FLAGS_FILE).read_bytes()
    rewrite(FLAGS_FILE, b"")  # the label still lists its flag
    with pytest.raises(DevLabelError, match="lists flags"):
        DevLabelSet.open(out, corpus, passages)
    rewrite(FLAGS_FILE, flags)

    rewrite(CHANGES_FILE, b"")
    with pytest.raises(DevLabelError, match="without a change record"):
        DevLabelSet.open(out, corpus, passages)


def test_open_and_inputs_are_bound_to_corpus_and_base_files(inputs: DevInputs, tmp_path: Path) -> None:
    corpus = verify_corpus(tmp_path / "corpus" / "corpus.jsonl")
    passages = load_corpus_passages(corpus)
    out = tmp_path / "dev_ai_v2"
    derive(inputs).write(out)
    other = verify_corpus(write_corpus(tmp_path / "other", version="fixture_corpus@9.9"))
    with pytest.raises(CorpusMismatchError):
        DevLabelSet.open(out, other, passages)
    label_set = DevLabelSet.open(out, corpus, passages)
    proposals_file = inputs.proposals.directory / PROPOSALS_FILE
    proposals_file.write_bytes(proposals_file.read_bytes() + b"\n")
    with pytest.raises(DevLabelError, match="must stay immutable"):
        label_set.verify_inputs()
