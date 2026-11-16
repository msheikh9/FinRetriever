"""Tests for AI-proposed development labels and the blind human audit (synthetic corpus, offline)."""

from __future__ import annotations

import hashlib
import io
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from finretriever.eval_set.ai_proposals import PROPOSALS_FILE, ProposalError, ProposalSet
from finretriever.eval_set.annotation_cli import main
from finretriever.eval_set.annotation_store import AnnotationStore, verify_corpus
from finretriever.eval_set.audit import (
    AUDIT_FILE,
    REPORT_FILE,
    AuditError,
    AuditWorkspace,
    audit_agreement,
    cohen_kappa,
    create_audit,
    select_audit_sample,
)
from finretriever.eval_set.qrels_builder import GoldExportError, build_gold_export
from finretriever.eval_set.schemas import Candidate, CandidateSource
from tests.unit.eval_set_fixtures import fixed_clock, passage_ids, write_config, write_corpus

IDS = passage_ids()
SOURCE = CandidateSource(method="bm25", rank=1, score=1.0, detail="query='x'")
AI_GRADES = {
    "q_0001": {"acme24_supplier": 2, "acme23_supplier": 0, "acme24_sales": 1, "acme24_liquidity": 0, "acme24_table": 0},
    "q_0002": {"beta24_liquidity": 2, "beta24_credit": 1, "beta24_rates": 0, "beta24_business": 0},
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.delenv("FINRETRIEVER_ANNOTATOR", raising=False)


def explanation(name: str) -> str:
    return f"EXPLANATION-{name}"


def proposal(query_id: str, name: str, grade: int, **overrides: Any) -> dict[str, Any]:
    row = {
        "query_id": query_id, "passage_id": IDS[name], "grade": grade, "confidence": "high",
        "explanation": explanation(name), "model": "claude-opus-5-5", "method": "test-read",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    return {**row, **overrides}


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(tmp_path / "gold.yaml", write_corpus(tmp_path / "corpus"))


@pytest.fixture
def store(config: Path) -> AnnotationStore:
    corpus = verify_corpus(config.parent / "corpus" / "corpus.jsonl")
    store = AnnotationStore.create(
        config.parent / "annotation", corpus, annotation_version="gold_test", created_by="T1", clock=fixed_clock()
    )
    common = {"question_type": "semantic", "authoring": "independent", "author": "T1"}
    store.add_question("How dependent is Acme on one supplier?", topic="risk_factors", tickers=["ACME"], **common)
    store.add_question("How large is Beta's liquidity buffer?", topic="debt_liquidity", tickers=["BETA"], **common)
    pools = {
        "q_0001": [*AI_GRADES["q_0001"], "acme24_competition"],
        "q_0002": [*AI_GRADES["q_0002"], "acme24_table"],  # acme24_table stays unproposed for q_0002
    }
    for query_id, names in pools.items():
        store.add_candidates(query_id, [Candidate(passage_id=IDS[n], source=SOURCE) for n in names], added_by="T1")
    store.judge("q_0001", IDS["acme24_competition"], 0, annotator="RV")  # an existing human judgment
    return store


@pytest.fixture
def proposals(config: Path, store: AnnotationStore) -> ProposalSet:
    proposals = ProposalSet.create(
        config.parent / "dev_ai", store, model="claude-opus-5-5", method="test-read",
        method_description="read each passage in full against the rubric", created_by="claude-opus-5-5",
        clock=fixed_clock(),
    )
    proposals.add([proposal(q, name, g) for q, grades in AI_GRADES.items() for name, g in grades.items()])
    return proposals


def file_hashes(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


# --- proposals --------------------------------------------------------------------------


def test_proposals_record_provenance_and_stay_apart(proposals: ProposalSet, store: AnnotationStore) -> None:
    meta = json.loads((proposals.directory / "proposals.meta.json").read_text())
    assert meta["label_source"] == "ai_proposal" and "never be exported or reported as gold" in meta["notice"]
    assert meta["evaluation"] == store.info.evaluation.model_dump() and meta["model"] == "claude-opus-5-5"
    reopened = ProposalSet.open(proposals.directory, store)
    assert len(reopened.proposals()) == 9
    assert reopened.proposal("q_0002", IDS["beta24_credit"]).grade == 1
    coverage = reopened.coverage()
    assert coverage["q_0001"] == {
        "pool": 6, "human_graded": 1, "ai_proposed": 5, "unlabelled": 0,
        "grades": {"0": 3, "1": 1, "2": 1}, "confidence": {"low": 0, "medium": 0, "high": 5},
    }
    assert coverage["q_0002"]["unlabelled"] == 1
    assert store.judgment("q_0001", IDS["acme24_supplier"]) is None  # proposals never touch the store
    with pytest.raises(ProposalError, match="must not live in the human annotation workspace"):
        ProposalSet.create(store.directory, store, model="m", method="m", method_description="d", created_by="m")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (proposal("q_0001", "acme24_supplier", 1), "already exists; proposals are not revised"),
        (proposal("q_0001", "acme24_competition", 0), "already has a human grade"),
        (proposal("q_0001", "beta24_credit", 0), "is not pooled for q_0001"),
        (proposal("q_0009", "acme24_supplier", 0), "unknown query_id q_0009"),
        (proposal("q_0002", "acme24_table", 3), "relevance grade must be one of"),
        (proposal("q_0002", "acme24_table", 0, confidence="certain"), "confidence"),
        (proposal("q_0002", "acme24_table", 0, explanation=" "), "must be non-empty"),
    ],
)
def test_invalid_proposals_are_rejected_without_writing(proposals: ProposalSet, row: dict, message: str) -> None:
    before = proposals.path.read_bytes()
    with pytest.raises(ProposalError, match=message):
        proposals.add([proposal("q_0002", "acme24_table", 0), row])  # the valid first row is not written either
    assert proposals.path.read_bytes() == before


def test_proposals_must_match_the_labelled_workspace(tmp_path: Path, proposals: ProposalSet, store: AnnotationStore) -> None:
    other = AnnotationStore.create(tmp_path / "other", store.corpus, annotation_version="other", created_by="T1")
    with pytest.raises(ProposalError, match="but the workspace is"):
        ProposalSet.open(proposals.directory, other)


# --- sampling ---------------------------------------------------------------------------


def test_sample_is_reproducible_and_stratified(proposals: ProposalSet) -> None:
    sample = select_audit_sample(proposals.proposals(), size=6, seed=11)
    assert sample == select_audit_sample(proposals.proposals(), size=6, seed=11)
    for query_id in AI_GRADES:
        assert sorted(p.grade for p in sample if p.query_id == query_id) == [0, 1, 2]
    assert sorted(p.passage_id for p in select_audit_sample(proposals.proposals(), size=6, seed=12)) != []
    with pytest.raises(AuditError, match="exceeds the 9 proposals"):
        select_audit_sample(proposals.proposals(), size=10, seed=11)


# --- audit workspace --------------------------------------------------------------------


def test_audit_workspace_is_blind_and_leaves_the_source_untouched(
    config: Path, store: AnnotationStore, proposals: ProposalSet
) -> None:
    before = file_hashes(store.directory)
    audit = create_audit(
        config.parent / "audit", store, proposals, size=6, seed=11, created_by="claude", annotation_version="audit"
    )
    assert file_hashes(store.directory) == before
    assert [q.query_id for q in audit.store.questions()] == ["q_0001", "q_0002"]
    assert [q.text for q in audit.store.questions()] == [q.text for q in store.questions()]
    assert len(audit.items()) == 6 and audit.progress() == {"items": 6, "graded": 0, "skipped": 0, "pending": 6}
    every_byte = "".join(p.read_text() for p in audit.directory.iterdir())
    assert "EXPLANATION-" not in every_byte and "confidence" not in every_byte and "ai_proposal" not in every_byte
    info = json.loads((audit.directory / AUDIT_FILE).read_text())
    assert info["proposals_sha256"] == proposals.sha256() and info["seed"] == 11
    with pytest.raises(AuditError, match="already holds an audit"):
        create_audit(config.parent / "audit", store, proposals, size=6, seed=11, created_by="claude", annotation_version="a")
    with pytest.raises(AuditError, match="its own directory"):
        create_audit(store.directory, store, proposals, size=6, seed=11, created_by="claude", annotation_version="a")


def test_agreement_statistics(config: Path, store: AnnotationStore, proposals: ProposalSet) -> None:
    audit = create_audit(config.parent / "audit", store, proposals, size=6, seed=11, created_by="c", annotation_version="a")
    with pytest.raises(AuditError, match="6 of 6 audit items still need"):
        audit_agreement(audit, proposals)
    human = {}
    for n, item in enumerate(audit.items()):
        ai = proposals.proposal(item.query_id, item.passage_id).grade
        if n == 0:
            audit.store.skip(item.query_id, item.passage_id, annotator="RV")
            continue
        grade = ai if n < 4 else (ai + 1) % 3  # items 2-4 agree, items 5-6 disagree
        audit.store.judge(item.query_id, item.passage_id, grade, annotator="RV")
        human[item.item] = (grade, ai)
    report = audit_agreement(AuditWorkspace.open(audit.directory, store.corpus), proposals)
    assert (report.items, report.graded, report.skipped) == (6, 5, 1)
    assert report.raw_agreement == pytest.approx(3 / 5)
    assert sum(sum(row.values()) for row in report.confusion.values()) == 5
    for grade, ai in human.values():
        assert report.confusion[str(grade)][str(ai)] >= 1
    assert report.cohen_kappa == pytest.approx(cohen_kappa(list(human.values())))
    assert report.binary_agreement == pytest.approx(sum((h > 0) == (a > 0) for h, a in human.values()) / 5)
    assert len(report.disagreements) == 2 and all("EXPLANATION-" in d["ai_explanation"] for d in report.disagreements)
    assert sum(cell["items"] for cell in report.by_question_type.values()) == 5
    assert sum(cell["disagree"] for cell in report.by_ai_grade.values()) == 2
    assert 0 <= report.weighted_raw_agreement <= 1


def test_cohen_kappa_matches_hand_computation() -> None:
    # observed 3/4; expected (2*1 + 1*2 + 1*1) / 16
    assert cohen_kappa([(0, 0), (1, 1), (2, 2), (0, 1)]) == pytest.approx((0.75 - 5 / 16) / (1 - 5 / 16))
    assert cohen_kappa([(0, 0), (0, 0)]) is None
    assert cohen_kappa([]) is None


def test_changed_proposals_invalidate_the_audit(config: Path, store: AnnotationStore, proposals: ProposalSet) -> None:
    audit = create_audit(config.parent / "audit", store, proposals, size=6, seed=11, created_by="c", annotation_version="a")
    proposals.add([proposal("q_0002", "acme24_table", 0)])
    with pytest.raises(AuditError, match="changed after the audit sample was drawn"):
        audit_agreement(audit, proposals)


# --- command line -----------------------------------------------------------------------


def cli(config: Path, *argv: str, keys: list[str] | None = None, annotator: str | None = "RV") -> tuple[int, str, str]:
    pending = list(keys or [])

    def type_key(_prompt: str) -> str:
        if not pending:
            raise EOFError
        return pending.pop(0)

    out, err = io.StringIO(), io.StringIO()
    args = [*argv, "--config", str(config)] + (["--annotator", annotator] if annotator else [])
    return main(args, input_fn=type_key, out=out, err=err), out.getvalue(), err.getvalue()


def test_blind_audit_workflow(config: Path, store: AnnotationStore, proposals: ProposalSet) -> None:
    code, out, _ = cli(config, "audit-init", annotator="claude")
    assert code == 0 and "blind audit of 6 passages across 2 questions" in out and "contains no AI labels" in out

    code, _, err = cli(config, "audit", annotator="claude")
    assert code == 1 and "is an AI identity" in err
    code, _, err = cli(config, "audit-report")
    assert code == 1 and "AI labels stay hidden until the audit is complete" in err

    code, out, _ = cli(config, "audit", keys=["2", "0 not the asked year", "q"])
    assert code == 0 and "AI labels hidden" in out
    assert "EXPLANATION-" not in out and "score" not in out and "source" not in out and "confidence" not in out
    assert "saved audit: graded 2 · skipped 0 · pending 4 of 6 (2 changes this session)" in out

    code, out, _ = cli(config, "audit", keys=["g 1", "1 on reflection", "q"])  # resumes at item 3; corrects item 1
    assert "audit item 3/6" in out.split("audit> ")[0]
    assert "saved audit: graded 2 · skipped 0 · pending 4 of 6 (1 changes this session)" in out
    _, history, _ = cli(config, "history", "q_0001", "--workspace", str(config.parent / "audit"))
    _, history2, _ = cli(config, "history", "q_0002", "--workspace", str(config.parent / "audit"))
    assert "-> 1 — on reflection" in history + history2

    code, out, _ = cli(config, "audit-status")
    assert code == 0 and "graded 2 · skipped 0 · pending 4" in out and "AI labels stay hidden" in out
    code, out, _ = cli(config, "audit", keys=["0", "s unsure", "0", "1"])
    assert "all 6 audit items are graded or skipped" in out

    code, out, _ = cli(config, "audit-report")
    assert code == 0 and "5 graded, 1 skipped of 6" in out and "confusion (rows: human grade, columns: AI grade)" in out
    report = json.loads((config.parent / "audit" / REPORT_FILE).read_text())
    assert report["graded"] == 5 and sum(sum(r.values()) for r in report["confusion"].values()) == 5


def test_ai_labels_and_audits_never_become_gold(config: Path, store: AnnotationStore, proposals: ProposalSet) -> None:
    code, _, err = cli(config, "annotate", "q_0001", annotator="claude")
    assert code == 1 and "relevance judgments must come from a person" in err
    assert cli(config, "audit-init", annotator="claude")[0] == 0
    code, _, err = cli(config, "export", "--workspace", str(config.parent / "audit"))
    assert code == 1 and "audit judgments are never exported as gold" in err
    store.judge("q_0002", IDS["acme24_table"], 0, annotator="claude")  # bypassing the CLI
    with pytest.raises(GoldExportError, match="recorded by AI identities"):
        build_gold_export(store, ai_identities=["claude"])
    assert (config.parent / "dev_ai" / PROPOSALS_FILE).exists() and not (config.parent / "export").exists()
