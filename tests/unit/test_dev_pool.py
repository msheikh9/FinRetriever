"""Tests for expanded development pools and the labels derived from them (synthetic corpus, offline)."""

from __future__ import annotations

import hashlib
import io
import json
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

from finretriever.eval_set.ai_proposals import PROPOSALS_FILE, ProposalError
from finretriever.eval_set.annotation_cli import main
from finretriever.eval_set.annotation_store import load_corpus_passages, verify_corpus
from finretriever.eval_set.dev_export import DEV_PROVENANCE_FILENAME, DEV_QRELS_FILENAME
from finretriever.eval_set.dev_labels import LABELS_INFO_FILE, DevLabel, DevLabelError, DevLabelSet
from finretriever.eval_set.dev_pool import (
    POOL_FILE,
    DevPool,
    DevPoolError,
    create_pool_proposals,
    open_pool_proposals,
)
from tests.unit.eval_set_fixtures import (
    DevInputs,
    build_dev_inputs,
    fixed_clock,
    passage_ids,
    write_stored_run,
)

IDS = passage_ids()
DENSE = {"q_0001": ["acme24_supplier", "acme23_sales", "beta24_credit"], "q_0002": ["beta24_liquidity", "acme24_table", "acme23_table"]}
BM25 = {"q_0001": ["acme24_supplier", "acme24_sales"], "q_0002": ["beta24_liquidity", "beta24_rates"]}
# New AI proposals: a P1 + low-confidence case, a grade 2 the P2 check turns into 1, and a 0.
# beta24_credit (q_0001) gets no proposal and must stay unjudged.
NEW = [("q_0001", "acme23_sales", 1, "low"), ("q_0002", "acme24_table", 2, "high"), ("q_0002", "acme23_table", 0, "medium")]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


def cli(inputs: DevInputs, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main([argv[0], "--config", str(inputs.config), *argv[1:]], out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def inputs(tmp_path: Path) -> DevInputs:
    inputs = build_dev_inputs(tmp_path)
    code, _, err = cli(inputs, "dev-labels", "--spec", str(inputs.write_spec()))  # the base set, test_v2
    assert code == 0, err
    corpus = tmp_path / "corpus" / "corpus.jsonl"
    write_stored_run(tmp_path / "runs", "global", "dense", DENSE, corpus_path=corpus)
    write_stored_run(tmp_path / "runs", "company", "bm25", BM25, corpus_path=corpus)
    return inputs


def pool_spec(inputs: DevInputs, **overrides: Any) -> Path:
    root = inputs.config.parent
    values = {
        "pool_version": "test_v3",
        "annotation_version": "dev_test_v3",
        "description": "test_v2 pool plus the stored runs' top results",
        "base_labels_dir": str(root / "dev_ai_v2"),
        "base_pool_path": str(root / "annotation" / "pool.jsonl"),
        "runs_dir": str(root / "runs"),
        "runs": [{"run": "dense", "mode": "global"}, {"run": "bm25", "mode": "company"}],
        "depth": 10,
        "output_dir": str(root / "dev_ai_v3" / "pool"),
        **overrides,
    }
    path = root / "dev_pool.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def label_spec(inputs: DevInputs, **overrides: Any) -> Path:
    root = inputs.config.parent
    base = inputs.spec()
    values = {
        "label_version": "test_v3",
        "annotation_version": "dev_test_v3",
        "description": "test_v2 carried over plus proposals for the expanded pool",
        "base_labels_dir": str(root / "dev_ai_v2"),
        "pool_dir": str(root / "dev_ai_v3" / "pool"),
        "proposals_dir": str(root / "dev_ai_v3" / "proposals"),
        "output_dir": str(root / "dev_ai_v3"),
        "rubric": base["rubric"],
        "p2_basis": base["p2_basis"],
        "p1_open_questions": base["p1_open_questions"],
        "review_confidence": ["low"],
        "flags": [],
        **overrides,
    }
    path = root / "dev_labels_v3.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def open_pool(inputs: DevInputs) -> DevPool:
    corpus = verify_corpus(inputs.config.parent / "corpus" / "corpus.jsonl")
    return DevPool.open(inputs.config.parent / "dev_ai_v3" / "pool", corpus, load_corpus_passages(corpus))


def propose(inputs: DevInputs, rows: list[tuple[str, str, int, str]] = NEW) -> None:
    proposals = create_pool_proposals(
        inputs.config.parent / "dev_ai_v3" / "proposals", open_pool(inputs), model="claude-opus-5-5",
        method="test-read", method_description="read each new passage in full", created_by="claude-opus-5-5",
        clock=fixed_clock(),
    )
    proposals.add(
        [
            {"query_id": q, "passage_id": IDS[name], "grade": grade, "confidence": confidence,
             "explanation": f"EXPLANATION-{name}", "model": "claude-opus-5-5", "method": "test-read",
             "created_at": "2026-01-01T00:00:00+00:00"}
            for q, name, grade, confidence in rows
        ]
    )


def test_expansion_adds_run_results_as_unjudged_entries_with_their_provenance(inputs: DevInputs) -> None:
    root = inputs.config.parent
    protected = {name: tree(root / name) for name in ("dev_ai_v2", "annotation", "runs", "dev_ai", "audit")}
    code, out, err = cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs)))
    assert code == 0, err
    assert "11 from the test_v2 pool + 3 new from the top 10 of 2 runs (1 from the companies asked about, 2 from others)" in out
    assert "adding a passage to a pool is not a judgment" in out
    assert {name: tree(root / name) for name in protected} == protected  # inputs are only read

    pool = open_pool(inputs)
    entries = {(e.query_id, e.passage_id): e for e in pool.entries}
    assert [e.position for e in pool.pool("q_0001")] == list(range(1, 9))
    supplier = entries["q_0001", IDS["acme24_supplier"]]
    assert (supplier.origin, supplier.judged_in, [s.method for s in supplier.base_sources]) == ("base_pool", "test_v2", ["bm25"])
    assert [(s.run, s.mode, s.rank) for s in supplier.run_sources] == [("dense", "global", 1), ("bm25", "company", 1)]
    new = [e for e in pool.entries if e.origin == "run_expansion"]
    assert [(e.query_id, e.passage_id, e.in_scope) for e in new] == [
        ("q_0001", IDS["acme23_sales"], True), ("q_0001", IDS["beta24_credit"], False), ("q_0002", IDS["acme23_table"], False),
    ]
    assert all(e.judged_in is None and not e.base_sources for e in new)
    assert entries["q_0002", IDS["acme24_table"]].judged_in is None  # pooled before but never labelled: still unjudged
    assert {(e.query_id, e.passage_id) for e in pool.unjudged()} == {(e.query_id, e.passage_id) for e in new} | {
        ("q_0002", IDS["acme24_table"])
    }

    coverage = pool.info.coverage
    assert (coverage["entries"], coverage["new_entries"], coverage["unjudged_entries"]) == (14, 3, 4)
    assert coverage["new_by_run"] == {"global/dense": 3, "company/bm25": 0}
    assert coverage["new_by_system"] == {"bm25": 0, "dense": 3} and coverage["new_by_system_set"] == {"dense": 3}
    assert coverage["top_k_by_system_set"] == {"bm25": 2, "bm25+dense": 2, "dense": 4}
    dense = coverage["per_run"]["global/dense"]
    assert dense["judged@10_before"] == pytest.approx(1 / 3) and dense["unjudged_top_k"] == 4
    assert dense["unjudged_top_k_in_scope"] == 1 and coverage["per_run"]["company/bm25"]["judged@10_before"] == 1.0
    assert coverage["per_query"]["q_0002"] == {
        "base_pool": 5, "new": 1, "new_in_scope": 0, "new_out_of_scope": 1, "pool": 6, "run_top_k_union": 4,
    }
    info = pool.info
    assert info.base["label_version"] == "test_v2" and info.evaluation.annotation_version == "dev_test_v3"
    assert info.base["labels_meta_sha256"] == hashlib.sha256((root / "dev_ai_v2" / LABELS_INFO_FILE).read_bytes()).hexdigest()
    assert [r["run_sha256"] for r in info.runs] == [
        hashlib.sha256((root / "runs" / m / r / "run.jsonl").read_bytes()).hexdigest() for m, r in (("global", "dense"), ("company", "bm25"))
    ]
    assert info.runs[0]["retriever"]["model_name"] == "fake/model"

    code, out, _ = cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs)))
    assert code == 0 and "unchanged" in out  # same inputs, byte-identical pool
    code, _, err = cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs, depth=1)))
    assert code == 1 and "immutable" in err


def test_expansion_refuses_modified_runs_mismatched_labels_and_protected_outputs(inputs: DevInputs) -> None:
    root = inputs.config.parent
    run_file = root / "runs" / "global" / "dense" / "run.jsonl"
    original = run_file.read_bytes()
    run_file.write_bytes(original.replace(b'"rank":2', b'"rank":3', 1))
    code, _, err = cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs)))
    assert code == 1 and "was modified" in err
    assert not (root / "dev_ai_v3").exists()
    run_file.write_bytes(original)

    # A run scored with the base labels must report the judged@10 those labels give.
    corpus = root / "corpus" / "corpus.jsonl"
    wrong = [{"query_id": "q_0001", "judged@10": 1.0}, {"query_id": "q_0002", "judged@10": 1 / 3}]
    write_stored_run(root / "runs", "global", "dense", DENSE, corpus_path=corpus, label_version="test_v2", per_query=wrong)
    code, _, err = cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs)))
    assert code == 1 and "was not scored with these labels" in err
    right = [{"query_id": "q_0001", "judged@10": 1 / 3}, {"query_id": "q_0002", "judged@10": 1 / 3}]
    write_stored_run(root / "runs", "global", "dense", DENSE, corpus_path=corpus, label_version="test_v2", per_query=right)

    for target in (root / "dev_ai_v2" / "pool", root / "annotation" / "x", root / "runs" / "pool", root / "audit"):
        code, _, err = cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs, output_dir=str(target))))
        assert code == 1 and "refusing to write an expanded pool" in err
    code, _, err = cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs, runs=[{"run": "dense", "mode": "company"}])))
    assert code == 1 and "missing" in err
    code, _, err = cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs)))
    assert code == 0, err


def test_pool_proposals_cover_only_unjudged_entries(inputs: DevInputs) -> None:
    assert cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs)))[0] == 0
    proposals = create_pool_proposals(
        inputs.config.parent / "dev_ai_v3" / "proposals", open_pool(inputs), model="claude-opus-5-5",
        method="test-read", method_description="read in full", created_by="claude-opus-5-5",
    )
    row = {"grade": 0, "confidence": "high", "explanation": "x", "model": "claude-opus-5-5", "method": "test-read", "created_at": "t"}
    with pytest.raises(ProposalError, match="that label is kept, not re-proposed"):
        proposals.add([{**row, "query_id": "q_0001", "passage_id": IDS["acme24_supplier"]}])
    with pytest.raises(ProposalError, match="not pooled"):
        proposals.add([{**row, "query_id": "q_0002", "passage_id": IDS["acme24_supplier"]}])
    proposals.add([{**row, "query_id": "q_0002", "passage_id": IDS["acme24_table"]}])
    with pytest.raises(ProposalError, match="already exists"):
        proposals.add([{**row, "query_id": "q_0002", "passage_id": IDS["acme24_table"]}])
    reopened = open_pool_proposals(proposals.directory, open_pool(inputs))
    assert len(reopened.proposals()) == 1
    assert reopened.coverage()["q_0002"] == {
        "pool": 6, "judged_before": 4, "ai_proposed": 1, "unlabelled": 1,
        "grades": {"0": 1, "1": 0, "2": 0}, "confidence": {"low": 0, "medium": 0, "high": 1},
    }
    with pytest.raises(ProposalError, match="already holds a proposal set"):
        create_pool_proposals(proposals.directory, open_pool(inputs), model="m", method="m", method_description="m", created_by="m")


def test_expanded_labels_carry_the_base_unchanged_and_add_only_new_labels(inputs: DevInputs) -> None:
    root = inputs.config.parent
    assert cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs)))[0] == 0
    propose(inputs)
    protected = {name: tree(root / name) for name in ("dev_ai_v2", "annotation", "runs")}
    code, out, err = cli(inputs, "dev-labels", "--spec", str(label_spec(inputs)))
    assert code == 0, err
    assert "13 labels = 10 carried unchanged from test_v2 + 3 new AI labels" in out and "1 pool entries still unjudged" in out
    assert "never as gold" in out
    assert "unchanged (identical copy present)" in cli(inputs, "dev-labels", "--spec", str(label_spec(inputs)))[1]

    corpus = verify_corpus(root / "corpus" / "corpus.jsonl")
    passages = load_corpus_passages(corpus)
    base = DevLabelSet.open(root / "dev_ai_v2", corpus, passages)
    expanded = DevLabelSet.open(root / "dev_ai_v3", corpus, passages)
    expanded.verify_inputs()
    carried = {(x.query_id, x.passage_id): x for x in expanded.labels if x.introduced_in == "test_v2"}
    assert {k: v.model_copy(update={"introduced_in": None}) for k, v in carried.items()} == {
        (x.query_id, x.passage_id): x for x in base.labels
    }
    assert expanded.changes[: len(base.changes)] == base.changes and expanded.flags[: len(base.flags)] == base.flags
    assert expanded.info.audit_dispositions == base.info.audit_dispositions

    new = {(x.query_id, x.passage_id): x for x in expanded.labels if x.introduced_in == "test_v3"}
    assert ("q_0001", IDS["beta24_credit"]) not in new  # no proposal: unjudged, never a 0
    table = new["q_0002", IDS["acme24_table"]]
    assert (table.base_grade, table.grade, table.changed) == (2, 1, True)
    (change,) = expanded.changes[len(base.changes):]
    assert (change.change_id, change.rule, change.previous_explanation) == ("chg_002", "P2", "EXPLANATION-acme24_table")
    sales = new["q_0001", IDS["acme23_sales"]]
    flags = {f.flag_id: f for f in expanded.flags}
    assert [(flags[f].rule, flags[f].candidate_grade) for f in sales.flags] == [("P1", 2), ("low_confidence", None)]
    assert (sales.grade, sales.labelled_by, sales.ai_generated) == (1, "claude-opus-5-5", True)
    counts = expanded.info.counts
    assert (counts["carried"], counts["new"], counts["unlabelled"]) == (10, 3, 1)
    assert counts["new_grades"] == {"0": 1, "1": 2, "2": 0} and counts["new_confidence"] == {"low": 1, "medium": 1, "high": 1}
    assert counts["new_flags"] == {"P1": 1, "low_confidence": 1} and counts["new_changes"] == {"P2": 1}
    assert counts["new_positives_by_system_set"] == {"dense": 2}
    derived = expanded.info.derived_from
    assert derived["base_label_version"] == "test_v2" and derived["proposals_dir"] == str(root / "dev_ai_v3" / "proposals")
    assert {name: tree(root / name) for name in protected} == protected

    code, out, err = cli(inputs, "export-dev", "--labels", str(root / "dev_ai_v3"))
    assert code == 0, err
    qrels = {(r["query_id"], r["passage_id"]): r["relevance"] for r in map(json.loads, (root / "dev_ai_v3" / DEV_QRELS_FILENAME).read_text().splitlines())}
    base_qrels = {(x.query_id, x.passage_id): x.grade for x in base.labels}
    assert len(qrels) == 13 and all(qrels[k] == g for k, g in base_qrels.items())
    assert ("q_0001", IDS["beta24_credit"]) not in qrels
    provenance = [json.loads(line) for line in (root / "dev_ai_v3" / DEV_PROVENANCE_FILENAME).read_text().splitlines()]
    assert {p["introduced_in"] for p in provenance} == {"test_v2", "test_v3"}
    report = json.loads((root / "dev_ai_v3" / "dev_eval.report.json").read_text())
    assert report["labels"]["by_introduced_in"] == {"test_v2": 10, "test_v3": 3}


def test_expanded_label_derivation_guards(inputs: DevInputs) -> None:
    root = inputs.config.parent
    assert cli(inputs, "expand-pool", "--spec", str(pool_spec(inputs)))[0] == 0
    propose(inputs)
    carried = {"query_id": "q_0001", "passage_id": IDS["acme24_supplier"], "rule": "P5", "candidate_grade": 1, "reason": "x"}
    code, _, err = cli(inputs, "dev-labels", "--spec", str(label_spec(inputs, flags=[carried])))
    assert code == 1 and "not a newly labelled passage" in err  # earlier labels are never re-flagged here
    rubric = {**inputs.spec()["rubric"], "version": "v2.0"}
    code, _, err = cli(inputs, "dev-labels", "--spec", str(label_spec(inputs, rubric=rubric)))
    assert code == 1 and "one rubric" in err
    code, _, err = cli(inputs, "dev-labels", "--spec", str(label_spec(inputs, output_dir=str(root / "dev_ai_v3" / "pool"))))
    assert code == 1 and "refusing to write development labels into the pool" in err
    code, _, err = cli(inputs, "dev-labels", "--spec", str(label_spec(inputs, label_version="test_v2")))
    assert code == 1 and "already exists" in err
    assert cli(inputs, "dev-labels", "--spec", str(label_spec(inputs)))[0] == 0

    # Every recorded input is re-checked: the proposals, the pool and the base labels' own inputs.
    corpus = verify_corpus(root / "corpus" / "corpus.jsonl")
    expanded = DevLabelSet.open(root / "dev_ai_v3", corpus, load_corpus_passages(corpus))
    for path in (root / "dev_ai_v3" / "proposals" / PROPOSALS_FILE, root / "dev_ai_v3" / "pool" / POOL_FILE, root / "dev_ai" / PROPOSALS_FILE):
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(DevLabelError, match="must stay immutable"):
            expanded.verify_inputs()
        path.write_bytes(original)
    expanded.verify_inputs()
    pool_file = root / "dev_ai_v3" / "pool" / POOL_FILE
    pool_file.write_bytes(pool_file.read_bytes().replace(b'"in_scope":true', b'"in_scope":false', 1))
    with pytest.raises(DevPoolError, match="was modified"):
        open_pool(inputs)


def test_labels_without_an_origin_render_as_before() -> None:
    label = DevLabel(
        query_id="q_0001", passage_id="p", grade=1, label_source="ai_proposal", ai_generated=True,
        assistance="not_applicable", labelled_by="m", base_grade=1, changed=False,
    )
    assert "introduced_in" not in label.model_dump(mode="json")  # earlier label sets stay byte-identical
    assert label.model_copy(update={"introduced_in": "v3"}).model_dump(mode="json")["introduced_in"] == "v3"
