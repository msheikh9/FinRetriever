"""Tests for the blind priority audit of an expanded pool's proposals (synthetic corpus, offline)."""

from __future__ import annotations

import hashlib
import io
import json
import re
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

from finretriever.eval_set.annotation_cli import main
from finretriever.eval_set.audit import AUDIT_FILE, REPORT_FILE, AuditError, AuditWorkspace
from finretriever.eval_set.dev_labels import DevLabelSet
from finretriever.eval_set.dev_pool import DevPool, open_pool_proposals
from finretriever.eval_set.pool_audit import (
    AuditCandidate,
    audit_candidates,
    select_priority_sample,
)
from finretriever.evaluation.stored_runs import load_stored_run
from tests.unit.eval_set_fixtures import ExpandedInputs, build_expanded_inputs, passage_ids

IDS = passage_ids()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def expanded(tmp_path: Path) -> ExpandedInputs:
    return build_expanded_inputs(tmp_path)


def cli(expanded: ExpandedInputs, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main([argv[0], "--config", str(expanded.dev.config), *argv[1:]], out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def spec(expanded: ExpandedInputs, **overrides: Any) -> Path:
    values = {
        "audit_version": "test_v3",
        "labels_dir": str(expanded.labels_dir),
        "pool_dir": str(expanded.pool_dir),
        "proposals_dir": str(expanded.proposals_dir),
        "output_dir": str(expanded.root / "audit_v3"),
        "size": 2,
        "seed": 5,
        "in_scope_only": True,
        "lexical_run": {"run": "bm25", "mode": "company"},
        "lexical_disagreement_rank": 100,
        **overrides,
    }
    path = expanded.root / "audit_v3.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def candidate(query_id: str, name: str, tier: int, confidence: str, *features: str) -> AuditCandidate:
    return AuditCandidate(query_id, name, tier, confidence, tuple(features))


def test_priority_selection_puts_tier_one_first_then_balances_questions() -> None:
    candidates = [
        candidate("q1", "a", 1, "low"),
        candidate("q1", "b", 1, "medium", "table"),
        candidate("q1", "c", 2, "high", "table", "other_year", "rank_disagreement"),
        candidate("q1", "d", 2, "medium"),
        candidate("q2", "e", 2, "high", "table"),
        candidate("q2", "f", 2, "high"),
    ]
    sample = select_priority_sample(candidates, size=4, seed=3)
    # Tier 1 first; then q2 (nothing chosen yet) before q1's feature-rich c.
    assert {c.passage_id for c in sample} == {"a", "b", "e", "f"}
    assert select_priority_sample(list(reversed(candidates)), size=4, seed=3) == sample  # input order is irrelevant
    assert {c.passage_id for c in select_priority_sample(candidates, size=5, seed=3)} == {"a", "b", "e", "f", "d"}
    assert {c.passage_id for c in select_priority_sample(candidates, size=4, seed=99)} == {"a", "b", "e", "f"}
    with pytest.raises(AuditError, match="exceeds the 6 eligible"):
        select_priority_sample(candidates, size=7, seed=3)


def test_review_features_come_from_the_passage_the_question_and_the_runs(expanded: ExpandedInputs) -> None:
    pool = DevPool.open(expanded.pool_dir, expanded.corpus, expanded.passages)
    proposals = open_pool_proposals(expanded.proposals_dir, pool)
    questions = DevLabelSet.open(expanded.labels_dir, expanded.corpus, expanded.passages).questions
    lexical = load_stored_run(expanded.root / "runs", "company", "bm25", corpus=expanded.corpus, passages=expanded.passages)

    def run(**kwargs: Any) -> dict[tuple[str, str], AuditCandidate]:
        found = audit_candidates(pool, proposals, questions, expanded.passages, lexical, lexical_disagreement_rank=100, **kwargs)
        return {(c.query_id, c.passage_id): c for c in found}

    in_scope = run()
    assert set(in_scope) == {("q_0001", IDS["acme23_sales"]), ("q_0001", IDS["acme23_table"])}
    sales = in_scope["q_0001", IDS["acme23_sales"]]
    assert (sales.tier, sales.confidence, sales.features) == (1, "low", ("other_year", "rank_disagreement"))
    table = in_scope["q_0001", IDS["acme23_table"]]
    assert (table.tier, table.features) == (2, ("table", "other_year", "rank_disagreement"))
    everything = run(in_scope_only=False)
    assert len(everything) == 5
    assert everything["q_0002", IDS["acme24_table"]].features == ("table", "rank_disagreement")
    assert everything["q_0001", IDS["beta24_credit"]].features == ("rank_disagreement",)


def test_audit_init_pool_writes_a_blind_workspace_and_audit_report_compares_it(expanded: ExpandedInputs) -> None:
    root = expanded.root
    protected = {name: tree(root / name) for name in ("dev_ai_v3", "dev_ai_v2", "annotation", "audit", "runs")}
    code, out, err = cli(expanded, "audit-init-pool", "--spec", str(spec(expanded)), "--annotator", "claude-opus-5-5")
    assert code == 0, err
    assert "blind priority audit of 2 passages across 1 questions" in out and "contains no AI labels" in out
    assert "audit --audit-dir" in out and "low" not in out and "tier" not in out
    audit_dir = root / "audit_v3"
    for path in (p for p in audit_dir.rglob("*") if p.is_file()):  # no per-item AI grade, confidence, reason
        text = path.read_text()
        assert "EXPLANATION-" not in text and not re.search(r'"grade":\s*[012]', text), path
        assert not re.search(r'"(confidence|tier|features)":', text), path
    info = json.loads((audit_dir / AUDIT_FILE).read_text())
    assert info["sample_size"] == 2 and info["seed"] == 5 and info["source_workspace"] == str(expanded.pool_dir)
    assert info["proposals_sha256"] == hashlib.sha256((expanded.proposals_dir / "proposals.jsonl").read_bytes()).hexdigest()
    assert info["selection"].startswith("Purposive priority sample")
    assert {(i["query_id"], i["passage_id"]) for i in info["items"]} == {("q_0001", IDS["acme23_sales"]), ("q_0001", IDS["acme23_table"])}
    assert {name: tree(root / name) for name in protected} == protected  # only the new audit directory is written

    code, _, err = cli(expanded, "audit-init-pool", "--spec", str(spec(expanded)), "--annotator", "claude-opus-5-5")
    assert code == 1 and "already holds an audit" in err
    code, out, _ = cli(expanded, "audit-status", "--audit-dir", str(audit_dir))
    assert code == 0 and "pending 2" in out
    code, _, err = cli(expanded, "audit-report", "--audit-dir", str(audit_dir))
    assert code == 1 and "still need a human grade" in err  # AI labels stay hidden until the audit is complete

    # A person grades both items (simulated here; the tool never records grades by itself).
    audit = AuditWorkspace.open(audit_dir, expanded.corpus)
    audit.store.judge("q_0001", IDS["acme23_sales"], 1, annotator="RV")
    audit.store.judge("q_0001", IDS["acme23_table"], 1, annotator="RV", note="2023 table")
    code, out, err = cli(expanded, "audit-report", "--audit-dir", str(audit_dir))
    assert code == 0, err
    report = json.loads((audit_dir / REPORT_FILE).read_text())
    assert (report["graded"], report["raw_agreement"], report["binary_agreement"]) == (2, 0.5, 0.5)
    (disagreement,) = report["disagreements"]
    assert (disagreement["passage_id"], disagreement["human_grade"], disagreement["ai_grade"]) == (IDS["acme23_table"], 1, 0)
    assert {name: tree(root / name) for name in ("dev_ai_v3", "dev_ai_v2")} == {
        name: protected[name] for name in ("dev_ai_v3", "dev_ai_v2")
    }


def test_audit_init_pool_refuses_protected_outputs_and_changed_inputs(expanded: ExpandedInputs) -> None:
    for target in (expanded.labels_dir, expanded.pool_dir, expanded.root / "annotation" / "audit"):
        code, _, err = cli(expanded, "audit-init-pool", "--spec", str(spec(expanded, output_dir=str(target))), "--annotator", "c")
        assert code == 1 and "refusing to write an audit" in err
    code, _, err = cli(expanded, "audit-init-pool", "--spec", str(spec(expanded, output_dir=str(expanded.root / "audit"))), "--annotator", "c")
    assert code == 1 and "already holds an audit" in err  # an earlier audit is never overwritten
    code, _, err = cli(expanded, "audit-init-pool", "--spec", str(spec(expanded, size=3)), "--annotator", "c")
    assert code == 1 and "exceeds the 2 eligible" in err
    run_file = expanded.root / "runs" / "company" / "bm25" / "run.jsonl"
    run_file.write_bytes(run_file.read_bytes() + b"\n")
    code, _, err = cli(expanded, "audit-init-pool", "--spec", str(spec(expanded)), "--annotator", "c")
    assert code == 1 and "was modified" in err
    assert not (expanded.root / "audit_v3").exists()
