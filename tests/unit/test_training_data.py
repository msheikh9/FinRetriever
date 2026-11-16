"""Tests for the training-pair contract, fail-closed loading and false-negative-safe batch planning (offline)."""

from __future__ import annotations

import hashlib
import io
import json
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from finretriever.eval_set.annotation_store import load_corpus_passages
from finretriever.training.data_cli import main
from finretriever.training.data_prep import (
    load_data_prep_config,
    open_prepared,
    verify_frozen_corpus,
)
from finretriever.training.pair_contract import (
    PAIR_CONTRACT_VERSION,
    TrainingPair,
    contract_problems,
)
from finretriever.training.pair_quality import load_evaluation_questions
from finretriever.training.query_generation import VALIDATOR_VERSION, prompt_sha256
from finretriever.training.training_data import (
    NegativeGuard,
    TrainingDataError,
    hard_negative_candidates,
    load_training_pairs,
    plan_batches,
)
from tests.unit.eval_set_fixtures import build_expanded_inputs
from tests.unit.training_fixtures import (
    SECRET,
    MockTransport,
    WhitespaceTokenizer,
    generic_candidates,
    training_extra_passages,
    write_prep_config,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


def cli(config: Path, *argv: str, transport: Any = None) -> int:
    out, err = io.StringIO(), io.StringIO()
    return main([argv[0], "--config", str(config), *argv[1:]], tokenizer_loader=lambda s: WhitespaceTokenizer(),
                transport=transport, sleep=lambda seconds: None, out=out, err=err)


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The synthetic corpus prepared, generated (mocked) and paired."""
    build_expanded_inputs(tmp_path, training_extra_passages())
    config_path = write_prep_config(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    transport = MockTransport(lambda payload: {"questions": generic_candidates(payload)})
    for argv in (("prepare",), ("generate", "--split", "train", "--confirm-api-use"),
                 ("generate", "--split", "validation", "--confirm-api-use"), ("build-pairs",)):
        assert cli(config_path, *argv, transport=transport) == 0
    config = load_data_prep_config(config_path)[0]
    corpus = verify_frozen_corpus(config)
    passages = load_corpus_passages(corpus)
    prepared = open_prepared(config.output_dir, corpus, passages)
    return {"config": config, "config_path": config_path, "passages": passages, "prepared": prepared,
            "questions": load_evaluation_questions(config.leakage.evaluation_question_files)}


def pairs_file(prepared: Any, split: str = "train") -> Path:
    return prepared.directory / "pairs" / f"{split}_pairs.jsonl"


def test_built_pairs_follow_the_contract_and_load_only_into_their_split(built: dict[str, Any]) -> None:
    prepared, passages, config = built["prepared"], built["passages"], built["config"]
    records = [json.loads(line) for line in pairs_file(prepared).read_text().splitlines()]
    assert records
    for record in records:
        pair = TrainingPair.model_validate(record)
        assert pair.contract_version == PAIR_CONTRACT_VERSION and contract_problems(record, prepared, passages) == []
        prov = pair.provenance
        assert (prov.preparation_manifest_sha256, prov.corpus_sha256) == (
            prepared.manifest_sha256, prepared.manifest["corpus"]["sha256"])
        assert (prov.validator_version, prov.prompt_sha256) == (VALIDATOR_VERSION, prompt_sha256(prov.prompt_version))
        assert pair.equivalents.index_digest == prepared.manifest["reuse"]["index"]["digest"]
        assert pair.split_unit == prepared.split_units()[pair.positive_passage_id]
        assert pair.source.visible_chars == prepared.rows[pair.positive_passage_id]["visible_chars"]
    train = load_training_pairs(prepared, passages, config, "train", evaluation_questions=built["questions"])
    validation = load_training_pairs(prepared, passages, config, "validation", evaluation_questions=built["questions"])
    assert {p.split for p in train} == {"train"} and {p.split for p in validation} <= {"validation"}
    assert all(prepared.rows[p.positive_passage_id]["split"] == "validation" for p in validation)
    # A validation passage can never be a train pair's positive, and vice versa.
    if validation:
        moved = {**validation[0].model_dump(mode="json"), "split": "train"}
        assert any("positive is in validation" in p for p in contract_problems(moved, prepared, passages))


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda r: r.pop("equivalents"), "not a contract-1 pair"),
        (lambda r: r["equivalents"].update(index_digest="0" * 64), "another evidence-reuse index"),
        (lambda r: r["provenance"].update(validator_version="sec-qcheck-v2"), "rebuild or re-check"),
        (lambda r: r["provenance"].update(prompt_sha256="0" * 64), "prompt SHA-256 does not match"),
        (lambda r: r["provenance"].update(preparation_manifest_sha256="0" * 64), "another preparation"),
        (lambda r: r.update(group_id="grp_99999"), "is not the positive's"),
        (lambda r: r["source"].update(company="Someone Else"), "source metadata differs"),
    ],
)
def test_missing_or_incompatible_metadata_is_refused(built: dict[str, Any], change: Any, problem: str) -> None:
    prepared, passages = built["prepared"], built["passages"]
    record = json.loads(pairs_file(prepared).read_text().splitlines()[0])
    change(record)
    assert any(problem in p for p in contract_problems(record, prepared, passages)), problem


def test_loading_fails_closed_on_stale_equivalents_and_untrusted_preparations(built: dict[str, Any]) -> None:
    prepared, passages, config, questions = built["prepared"], built["passages"], built["config"], built["questions"]
    path, meta_path = pairs_file(prepared), prepared.directory / "pairs" / "pairs.meta.json"
    original, original_meta = path.read_text(), meta_path.read_text()
    records = [json.loads(line) for line in original.splitlines()]
    other = next(pid for pid, r in prepared.rows.items() if r["split"] == "train"
                 and pid != records[0]["positive_passage_id"] and pid not in records[0]["equivalents"]["exact"])
    records[0]["equivalents"]["uncertain"] = [*records[0]["equivalents"]["uncertain"], other]
    data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records)
    path.write_text(data)
    meta = json.loads(original_meta)
    meta["files"]["train_pairs.jsonl"] = hashlib.sha256(data.encode()).hexdigest()  # re-pinned, as if written so
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(TrainingDataError, match="equivalents are not the ones"):
        load_training_pairs(prepared, passages, config, "train", evaluation_questions=questions)
    path.write_text(original)
    meta_path.write_text(original_meta)
    manifest_path = prepared.directory / "manifest.json"
    manifest = manifest_path.read_text()
    manifest_path.write_text(manifest.replace(prepared.manifest["reuse"]["index"]["digest"], "0" * 64))
    tampered = open_prepared(prepared.directory, verify_frozen_corpus(config), passages)
    with pytest.raises(TrainingDataError, match="different preparation"):  # the pairs no longer match it
        load_training_pairs(tampered, passages, config, "train", evaluation_questions=questions)
    with pytest.raises(TrainingDataError, match="cannot be completed"):  # nor does the evidence-reuse index
        load_training_pairs(tampered, passages, config, "train", evaluation_questions=questions, include_built=False)
    manifest_path.write_text(manifest)
    with pytest.raises(TrainingDataError, match="no training pairs"):
        load_training_pairs(prepared, passages, config, "blocked", evaluation_questions=questions)


# --- the guard and batch planning, on a small stub preparation ------------------------------------

DIGEST, MANIFEST = "d" * 64, "e" * 64


@dataclass
class StubPrepared:
    rows: dict[str, dict[str, str]]
    links: dict[str, frozenset[str]] = field(default_factory=dict)
    manifest_sha256: str = MANIFEST

    def reuse_links(self) -> dict[str, frozenset[str]]:
        return self.links


def stub() -> StubPrepared:
    groups = {"A": "gA", "A2": "gA", "B": "gB", "C": "gC", "D": "gD", "E": "gE", "F": "gF", "G": "gG", "H": "gH",
              "X": "gX", "Z": "gZ"}
    split = {"X": "validation", "Z": "blocked"}
    rows = {pid: {"group_id": g, "split": split.get(pid, "train")} for pid, g in groups.items()}
    return StubPrepared(rows, {"D": frozenset({"E"}), "E": frozenset({"D"})})


def pair(pair_id: str, positive: str, exact: tuple[str, ...] = (), uncertain: tuple[str, ...] = (),
         manifest: str = MANIFEST, split: str = "train", group: str | None = None) -> TrainingPair:
    return TrainingPair.model_validate({
        "contract_version": 1, "pair_id": pair_id, "split": split, "query": f"What does {positive} say?",
        "positive_passage_id": positive, "group_id": group or stub().rows[positive]["group_id"], "split_unit": "u",
        "question_type": "other", "answer": "a", "evidence": "e", "max_evaluation_similarity": 0.0,
        "ticker": "AAA", "fiscal_year": "2024",
        "source": {"ticker": "AAA", "company": "AAA Inc.", "fiscal_year": "2024", "form_type": "10-K",
                   "filing_date": "2025-02-01", "report_date": "2024-12-31", "section": "Item 7", "item": "7",
                   "visible_chars": 100},
        "equivalents": {"exact": list(exact), "uncertain": list(uncertain), "method": "evidence-reuse-v2",
                        "index_digest": DIGEST},
        "provenance": {"origin": "build-pairs", "corpus_version": "c", "corpus_sha256": "c" * 64,
                       "preparation_version": "v", "preparation_manifest_sha256": manifest, "provider": "groq",
                       "model": "m", "prompt_version": "sec-qgen-v2", "prompt_sha256": prompt_sha256("sec-qgen-v2"),
                       "validator_version": VALIDATOR_VERSION, "generation_params_sha256": "f" * 64},
    })


PAIRS = [
    pair("p1", "A", exact=("B",), uncertain=("C",)),  # B also holds the evidence; C may answer
    pair("p2", "B"),  # an equivalent positive of p1
    pair("p3", "C"),  # an uncertain equivalent of p1
    pair("p4", "A2"),  # same overlap group as p1's positive
    pair("p5", "D"),
    pair("p6", "E"),  # linked to D by substantial evidence reuse
    pair("p7", "F"),
    pair("p8", "G"),
    pair("p9", "H", exact=("F",)),  # two positives: H and F
]


def test_the_guard_knows_every_passage_that_answers_a_pair() -> None:
    guard = NegativeGuard(stub())  # type: ignore[arg-type]
    p1 = PAIRS[0]
    assert guard.positives(p1) == {"A", "B"}  # multi-positive: the positive and its exact equivalents
    assert {"A", "A2", "B", "C"} <= guard.exclusions(p1)
    by_id = {p.pair_id: p for p in PAIRS}
    expected = {("p1", "p2"), ("p1", "p3"), ("p1", "p4"), ("p5", "p6"), ("p7", "p9")}
    found = {(a.pair_id, b.pair_id) for i, a in enumerate(PAIRS) for b in PAIRS[i + 1:] if guard.conflict(a, b)}
    assert found == expected
    assert not guard.conflict(by_id["p7"], by_id["p8"])
    ranked = ["A", "B", "C", "A2", "X", "Z", "D", "F", "E"]
    assert hard_negative_candidates(p1, ranked, guard) == ["D", "F", "E"]  # no equivalent, other split or blocked
    assert hard_negative_candidates(by_id["p5"], ranked, guard, limit=2) == ["A", "B"]  # E is linked to D
    with pytest.raises(TrainingDataError, match="another preparation"):
        guard.exclusions(pair("q", "G", manifest="0" * 64))


def test_batch_plans_are_deterministic_and_never_hold_false_negatives() -> None:
    guard = NegativeGuard(stub())  # type: ignore[arg-type]
    by_id = {p.pair_id: p for p in PAIRS}
    for seed in range(25):
        plan = plan_batches(PAIRS, guard, batch_size=3, seed=seed)
        assert plan == plan_batches(list(reversed(PAIRS)), guard, batch_size=3, seed=seed)  # input order irrelevant
        placed = [pid for batch in plan.batches for pid in batch]
        assert sorted(placed) == sorted(by_id) and all(len(b) <= 3 for b in plan.batches)
        for batch in plan.batches:
            assert not any(guard.conflict(by_id[a], by_id[b]) for a in batch for b in batch if a < b)
        assert plan.stats["pairs"] == 9 and plan.stats["batches"] == len(plan.batches)
    assert plan_batches(PAIRS, guard, batch_size=3, seed=1) != plan_batches(PAIRS, guard, batch_size=3, seed=2)
    whole = plan_batches(PAIRS, guard, batch_size=9, seed=0)
    assert whole.stats["batches"] >= 2  # five conflicting combinations cannot share one batch
    with pytest.raises(TrainingDataError, match="one split"):
        plan_batches([*PAIRS, pair("v1", "X", split="validation")], guard, batch_size=3, seed=0)
    with pytest.raises(TrainingDataError, match="repeat"):
        plan_batches([PAIRS[0], PAIRS[0]], guard, batch_size=3, seed=0)
    with pytest.raises(TrainingDataError, match="no pairs"):
        plan_batches([], guard, batch_size=3, seed=0)


def test_balanced_plans_fill_equal_batches_under_the_same_protections() -> None:
    guard = NegativeGuard(stub())  # type: ignore[arg-type]
    by_id = {p.pair_id: p for p in PAIRS}
    for seed in range(25):
        plan = plan_batches(PAIRS, guard, batch_size=3, seed=seed, method="balanced")
        assert plan == plan_batches(list(reversed(PAIRS)), guard, batch_size=3, seed=seed, method="balanced")
        assert sorted(pid for batch in plan.batches for pid in batch) == sorted(by_id)  # every pair, once
        assert [len(b) for b in plan.batches] == [3, 3, 3]  # ceil(9 / 3) batches, full
        for batch in plan.batches:  # the guard's conflicts (equivalents, overlap groups, reuse links) never meet
            assert not any(guard.conflict(by_id[a], by_id[b]) for a in batch for b in batch if a < b)
    assert plan_batches(PAIRS, guard, batch_size=3, seed=1, method="balanced") != plan_batches(
        PAIRS, guard, batch_size=3, seed=2, method="balanced")
    assert plan_batches(PAIRS, guard, batch_size=3, seed=4) == plan_batches(PAIRS, guard, batch_size=3, seed=4,
                                                                           method="first_fit")  # the default
    with pytest.raises(TrainingDataError, match="unknown batching method"):
        plan_batches(PAIRS, guard, batch_size=3, seed=0, method="masked")


def _clustered(clique: int, others: int) -> tuple[StubPrepared, list[TrainingPair]]:
    """``clique`` pairs whose positives share an overlap group (no two may share a batch) and ``others`` free pairs."""
    rows = {f"K{i}": {"group_id": "gK", "split": "train"} for i in range(clique)}
    rows |= {f"P{i}": {"group_id": f"g{i}", "split": "train"} for i in range(others)}
    pairs = [pair(f"k{i}", f"K{i}", group="gK") for i in range(clique)]
    return StubPrepared(rows), pairs + [pair(f"q{i}", f"P{i}", group=f"g{i}") for i in range(others)]


@pytest.mark.parametrize(("clique", "others", "batch_size", "sizes"), [
    (3, 17, 8, [6, 7, 7]),  # ceil(20 / 8) = 3 batches, as even as 20 allows
    (5, 15, 8, [4, 4, 4, 4, 4]),  # a 5-clique needs 5 batches: the planner grows to the fewest that fit
    (0, 7, 64, [7]),  # fewer pairs than a batch: one batch
])
def test_balanced_plans_use_the_fewest_batches_the_conflicts_allow(clique: int, others: int, batch_size: int,
                                                                     sizes: list[int]) -> None:
    prepared, pairs = _clustered(clique, others)
    guard = NegativeGuard(prepared)  # type: ignore[arg-type]
    by_id = {p.pair_id: p for p in pairs}
    for seed in range(10):
        plan = plan_batches(pairs, guard, batch_size=batch_size, seed=seed, method="balanced")
        assert sorted(len(b) for b in plan.batches) == sizes and plan.stats["smallest_batch"] == min(sizes)
        assert sorted(pid for b in plan.batches for pid in b) == sorted(by_id)
        assert all(sum(pid.startswith("k") for pid in b) <= 1 for b in plan.batches)  # clique members apart
        for batch in plan.batches:
            assert not any(guard.conflict(by_id[a], by_id[b]) for a in batch for b in batch if a < b)


def test_carried_over_pairs_load_through_the_same_contract(built: dict[str, Any], tmp_path: Path) -> None:
    prepared, passages, config = built["prepared"], built["passages"], built["config"]
    train = [json.loads(line) for line in pairs_file(prepared).read_text().splitlines()]
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("".join(json.dumps({
        "pair_id": p["pair_id"], "passage_id": p["positive_passage_id"], "decision": "accept", "issues": [],
        "question_sha256": hashlib.sha256(p["query"].encode()).hexdigest(), "reviewer": "ai_review", "gold": False,
    }) + "\n" for p in train))
    target_path = tmp_path / "target.yaml"
    write_prep_config(tmp_path, target_path, output_dir=str(tmp_path / "data" / "processed" / "train" / "target"))
    assert cli(target_path, "prepare") == 0
    assert cli(target_path, "carry-over", "--from-config", str(built["config_path"]), "--decisions", str(decisions),
               "--name", "pilot") == 0
    target_config = load_data_prep_config(target_path)[0]
    target = open_prepared(target_config.output_dir, verify_frozen_corpus(target_config), passages)
    carried = load_training_pairs(target, passages, target_config, "train", evaluation_questions=built["questions"],
                                  carryovers=["pilot"], include_built=False)
    assert carried and {p.provenance.origin for p in carried} == {"carryover:pilot"}
    assert all(p.provenance.preparation_manifest_sha256 == target.manifest_sha256 for p in carried)
    with pytest.raises(TrainingDataError, match="no carry-over"):
        load_training_pairs(target, passages, target_config, "train", evaluation_questions=built["questions"],
                            carryovers=["missing"], include_built=False)
    del config
