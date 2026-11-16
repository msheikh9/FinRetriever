"""The validation-pair set: built once from a validation plan's records, apart from the training data (offline)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from finretriever.training.candidates import (
    load_candidates,
    read_candidates_manifest,
    verify_candidate_provenance,
)
from finretriever.training.evidence_reuse import Equivalent, EvidenceLocator
from finretriever.training.pair_quality import evidence_leak
from finretriever.training.sampling_plan import (
    PlanError,
    load_plan,
    make_stop_record,
    read_stop_record,
    stop_path,
    write_stop_record,
)
from finretriever.training.training_data import (
    NegativeGuard,
    TrainingDataError,
    plan_batches,
    verify_pairs,
)
from finretriever.training.validation_set import (
    ValidationSetError,
    isolation_problems,
    load_validation_config,
    load_validation_decisions,
    load_validation_set,
    pair_id_for,
)
from tests.unit.training_fixtures import MockTransport
from tests.unit.validation_fixtures import (
    EXTRA,
    PASS,
    PLAN,
    World,
    block_network,
    build_world,
    cli,
    log_of,
    tree,
    validation_reply,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    block_network(monkeypatch)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return build_world(tmp_path, monkeypatch)


def test_a_validation_set_is_written_once_apart_from_the_training_data(world: World) -> None:
    training = {name: tree(world.directory / name) for name in ("pairs", "candidates")}
    generation = tree(world.directory / "generation")
    config_path = world.validation_config("val_v1", world.review("val_decisions"))
    assert set(world.plan_ids()) == {EXTRA["gamma24_debt"], EXTRA["gamma23_revenue"], EXTRA["gamma22_liquidity"]}

    code, out, err = cli(world.config_path, "build-validation", "--validation", str(config_path))
    assert code == 0, err
    assert "3 passages, 3 questions accepted by the generator, 2 included" in out
    assert "not human-validated, not gold labels, never training or evaluation data" in out
    directory = world.directory / "validation_sets" / "val_v1"
    assert sorted(p.name for p in directory.iterdir()) == ["manifest.json", "validation_log.jsonl", "validation_pairs.jsonl"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["kind"] == "validation_pairs" and "not gold labels" in manifest["notice"]
    assert manifest["decisions"]["reviewer"] == "ai_review" and manifest["labels"].endswith("not gold")
    assert [c["version"] for c in manifest["training"]["candidates"]] == ["cand_v1"]
    assert manifest["generation"]["records"]["lines"] == 3 and manifest["plan"]["method"] == "balanced_years"

    # gamma23_revenue's figure is the prior year of a train passage (gamma24_revenue): an uncertain match, which the
    # training-pair rule lets through, but never a validation question.
    log = log_of(world, "val_v1")
    revenue = log[EXTRA["gamma23_revenue"]]
    assert not revenue["included"] and revenue["reasons"] == ["possible_training_evidence"]
    assert [(e["passage_id"], e["match"]) for e in revenue["evidence"]["outside_validation"]] == [
        (EXTRA["gamma24_revenue"], "value")]
    assert [r["plan_position"] for r in log.values()] == [1, 2, 3]
    pairs = [json.loads(line) for line in (directory / "validation_pairs.jsonl").read_text().splitlines()]
    assert {p["positive_passage_id"] for p in pairs} == {EXTRA["gamma24_debt"], EXTRA["gamma22_liquidity"]}
    for pair in pairs:
        assert pair["split"] == "validation" == world.prepared.rows[pair["positive_passage_id"]]["split"]
        assert pair["pair_id"].startswith(f"{PLAN}:validation_") and pair["provenance"]["origin"] == f"validation-plan:{PLAN}"

    # Nothing of the training data or of the generation records changed; the training candidates still verify.
    assert {name: tree(world.directory / name) for name in ("pairs", "candidates")} == training
    assert tree(world.directory / "generation") == generation
    train = load_candidates(world.prepared, world.passages, world.config, "cand_v1", "train")
    assert verify_candidate_provenance(world.prepared, "cand_v1")["candidates"] == len(train)
    code, _, err = cli(world.config_path, "build-pairs")
    assert code == 1 and "already holds different pairs" in err

    # Written once: an identical rebuild writes nothing, a different one is refused.
    written = tree(directory)
    code, out, _ = cli(world.config_path, "build-validation", "--validation", str(config_path))
    assert code == 0 and "unchanged (identical copy present)" in out
    rejecting = world.review("val_decisions_2", lambda pid: ("reject", {**PASS, "financial_relevance": "fail"}))
    code, _, err = cli(world.config_path, "build-validation", "--validation", str(world.validation_config("val_v1", rejecting)))
    assert code == 1 and "already holds a different validation set" in err and "nothing was written" in err
    assert tree(directory) == written

    # A trainer cannot open it: candidates are found by name under candidates/ and by kind.
    with pytest.raises(TrainingDataError, match="no candidate dataset"):
        read_candidates_manifest(world.prepared, "val_v1")
    shutil.copytree(directory, world.directory / "candidates" / "val_v1")
    with pytest.raises(TrainingDataError, match="not a format-1 candidate manifest"):
        load_candidates(world.prepared, world.passages, world.config, "val_v1", "train")
    shutil.rmtree(world.directory / "candidates" / "val_v1")

    # Loaded for checkpoint selection, it shares nothing with the training candidates.
    validation = load_validation_set(world.prepared, world.passages, world.config, "val_v1", training=train)
    assert [p.pair_id for p in validation] == [p["pair_id"] for p in pairs]
    assert isolation_problems(validation, train, threshold=0.8) == []
    guard = NegativeGuard(world.prepared)
    assert not any(guard.allowed_negative(t, pid) for t in train for v in validation for pid in (v.positive_passage_id,))
    with pytest.raises(TrainingDataError, match="validation pair among train pairs"):
        verify_pairs(pairs, world.prepared, world.passages, world.config, "train", evaluation_questions=None)
    with pytest.raises(TrainingDataError, match="pairs of one split only"):  # never in a training batch
        plan_batches([*train, *validation], guard, batch_size=len(train) + len(validation), seed=0)


def test_a_written_set_lets_records_grow_but_refuses_any_changed_input(world: World) -> None:
    config_path = world.validation_config("val_v1", world.review("val_decisions"))
    assert cli(world.config_path, "build-validation", "--validation", str(config_path))[0] == 0

    def load() -> list[Any]:
        return load_validation_set(world.prepared, world.passages, world.config, "val_v1")

    records = world.directory / "generation" / "validation.records.jsonl"
    original = records.read_bytes()
    records.write_bytes(original + original.splitlines(keepends=True)[-1])  # a later run appends: still bound
    assert len(load()) == 2
    records.write_bytes(original.replace(b'"attempts":1', b'"attempts": 1', 1) if b'"attempts":1' in original
                        else original.replace(b'"attempts": 1', b'"attempts":1', 1))
    with pytest.raises(TrainingDataError, match="generation records are append-only"):
        load()
    records.write_bytes(original)

    pairs_file = world.directory / "validation_sets" / "val_v1" / "validation_pairs.jsonl"
    kept = pairs_file.read_bytes()
    pairs_file.write_bytes(kept.replace(b"Gamma Energy", b"Gamma", 1))
    with pytest.raises(TrainingDataError, match="does not match manifest.json"):
        load()
    pairs_file.write_bytes(kept)
    decisions = world.root / "val_decisions.jsonl"
    decisions.write_text(decisions.read_text() + "\n")
    with pytest.raises(TrainingDataError, match="changed since it was built"):
        load()


def test_evidence_any_passage_outside_validation_holds_excludes_the_question(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = world.prepared.rows
    blocked = next(pid for pid, row in rows.items() if row["split"] == "blocked")
    added = {  # matches the training-pair rule lets through: uncertain, or exact beyond the visible text
        "$900 million of borrowings": Equivalent(blocked, "span", None),
        "$410 million of cash": Equivalent(EXTRA["gamma23_capex"], "quote", None),
    }
    assert rows[EXTRA["gamma23_capex"]]["split"] == "train"
    locate = EvidenceLocator.locate

    def with_outside(self: EvidenceLocator, evidence: str, answer: str, ticker: str, positive: str) -> list[Equivalent]:
        text = world.passages[positive].text
        return [*locate(self, evidence, answer, ticker, positive), *(e for k, e in added.items() if k in text)]

    monkeypatch.setattr(EvidenceLocator, "locate", with_outside)
    assert all(evidence_leak([e], rows, "validation") is None for e in added.values())
    built = world.build(world.validation_config("val_v2", world.review("val_decisions")))
    assert built.pairs == [] and built.manifest["stats"]["included"] == 0
    log = {r["positive_passage_id"]: r for r in built.log}
    assert log[EXTRA["gamma24_debt"]]["reasons"] == ["possible_evaluation_evidence"]
    assert log[EXTRA["gamma22_liquidity"]]["reasons"] == ["possible_training_evidence"]
    outside = log[EXTRA["gamma24_debt"]]["evidence"]["outside_validation"]
    assert outside == [{"passage_id": blocked, "split": "blocked", "match": "span", "visible_match": None,
                        "blocked_for": outside[0]["blocked_for"]}] and outside[0]["blocked_for"]
    assert not (world.directory / "validation_sets").exists()  # building writes nothing


def test_decisions_are_an_ai_review_bound_to_each_question(world: World) -> None:
    good = world.review("val_decisions")
    rows = [json.loads(line) for line in good.read_text().splitlines()]

    def broken(change: Callable[[dict[str, Any]], None]) -> Path:
        changed = [dict(r) for r in rows]
        change(changed[0])
        path = world.root / "broken.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in changed))
        return path

    cases = [
        (lambda r: r.update(gold=True), "never gold labels"),
        (lambda r: r.update(reviewer="human"), "AI review"),
        (lambda r: r.pop("aspects"), "needs one outcome for each"),
        (lambda r: r.update(aspects={**PASS, "units": "unclear"}), "unknown aspect outcomes"),
        (lambda r: r.update(aspects={**PASS, "figures": "fail"}), "accepted although figures failed"),
        (lambda r: r.update(decision="revise", reason="x"), "names no failed aspect"),
    ]
    for change, message in cases:
        with pytest.raises(ValidationSetError, match=message):
            load_validation_decisions(broken(change))
    with pytest.raises(ValidationSetError, match="never move between questions"):
        world.build(world.validation_config("val_v3", broken(lambda r: r.update(question_sha256="0" * 64))))
    with pytest.raises(ValidationSetError, match="does not have"):
        world.build(world.validation_config("val_v3", broken(lambda r: r.update(pair_id=pair_id_for(9)))))

    def revise_debt(pid: str) -> tuple[str, dict[str, str]]:
        return ("revise", {**PASS, "period": "fail"}) if pid == EXTRA["gamma24_debt"] else ("accept", PASS)

    built = world.build(world.validation_config("val_v3", world.review("partial", revise_debt)))
    log = {r["positive_passage_id"]: r for r in built.log}
    assert log[EXTRA["gamma24_debt"]]["reasons"] == ["quality_revise"]
    assert log[EXTRA["gamma24_debt"]]["quality"]["gold"] is False
    assert [p["positive_passage_id"] for p in built.pairs] == [EXTRA["gamma22_liquidity"]]


def test_isolation_problems_find_shared_passages_and_questions(world: World) -> None:
    config_path = world.validation_config("val_v1", world.review("val_decisions"))
    assert cli(world.config_path, "build-validation", "--validation", str(config_path))[0] == 0
    validation = load_validation_set(world.prepared, world.passages, world.config, "val_v1")
    train = load_candidates(world.prepared, world.passages, world.config, "cand_v1", "train")
    v, t = validation[0], train[0]
    assert isolation_problems(validation, train, threshold=0.8) == []
    same_question = t.model_copy(update={"query": v.query})
    shared = t.model_copy(update={"equivalents": t.equivalents.model_copy(update={"uncertain": [v.positive_passage_id]})})
    near = t.model_copy(update={"query": v.query.replace("How much", "How much money")})
    assert "repeats the question" in isolation_problems([v], [same_question], threshold=0.8)[0]
    assert "shares passage" in isolation_problems([v], [shared], threshold=0.8)[0]
    assert "overlaps training pair" in isolation_problems([v], [near], threshold=0.8)[0]
    assert "a validation pair among the training pairs" in isolation_problems([v], [v], threshold=0.8)[0]
    with pytest.raises(TrainingDataError, match="repeats the question"):
        load_validation_set(world.prepared, world.passages, world.config, "val_v1", training=[same_question])


def _write_config(world: World, version: str, **fields: Any) -> Path:
    path = world.root / f"{version}.yaml"
    path.write_text(yaml.safe_dump({"version": version, "data_prep_config": str(world.config_path), "plan": PLAN,
                                    "training_candidates": ["cand_v1"], **fields}))
    return path


def test_a_set_takes_a_plan_prefix_and_decisions_from_several_files(world: World) -> None:
    rows = world.review("all_positions").read_text().splitlines(keepends=True)
    files = []
    for position in (1, 2, 3):
        files.append(world.root / f"batch{position}.jsonl")
        files[-1].write_text(rows[position - 1])
    config_path = _write_config(world, "val_prefix", plan_positions=2, decisions=[str(files[0]), str(files[1])])
    code, out, err = cli(world.config_path, "build-validation", "--validation", str(config_path))
    assert code == 0, err
    assert "2 passages (its first 2)" in out
    log = log_of(world, "val_prefix")
    assert sorted(r["plan_position"] for r in log.values()) == [1, 2]  # position 3 is not part of the set
    assert world.plan_ids()[2] not in log
    manifest = json.loads((world.directory / "validation_sets" / "val_prefix" / "manifest.json").read_text())
    assert manifest["plan"]["passages"] == 3 and manifest["plan"]["positions_used"] == 2
    assert [f["path"] for f in manifest["decisions"]["files"]] == [str(files[0]), str(files[1])]
    assert manifest["decisions"]["decided"] == 2 and manifest["decisions"]["reviewer"] == "ai_review"
    assert len(load_validation_set(world.prepared, world.passages, world.config, "val_prefix")) == sum(
        r["included"] for r in log.values())

    files[1].write_text(files[1].read_text() + "\n")  # every decisions file is bound, not only the first
    with pytest.raises(TrainingDataError, match="changed since it was built"):
        load_validation_set(world.prepared, world.passages, world.config, "val_prefix")
    with pytest.raises(ValidationSetError, match="in its first 2 positions"):
        world.build(_write_config(world, "val_x", plan_positions=2, decisions=[str(files[0]), str(files[2])]))
    twice = world.root / "twice.jsonl"
    twice.write_text(rows[0])
    with pytest.raises(ValidationSetError, match="decided in an earlier decisions file"):
        world.build(_write_config(world, "val_x", decisions=[str(files[0]), str(twice)]))
    with pytest.raises(ValidationSetError, match="plan_positions 4"):
        world.build(_write_config(world, "val_x", plan_positions=4, decisions=[str(files[0])]))
    for bad in ({"decisions": []}, {"decisions": [str(files[0]), str(files[0])]}):
        with pytest.raises(ValidationSetError, match="each once"):
            load_validation_config(_write_config(world, "val_x", **bad))


def test_a_stopped_plan_is_never_generated_again_and_sets_stay_within_it(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = world.directory / "generation" / "validation.records.jsonl"
    before = records.read_bytes()
    code, out, err = cli(world.config_path, "stop-plan", "--split", "validation", "--plan", PLAN, "--reason", "budget")
    assert code == 0, err
    assert "stopped after position 3 of 3" in out and "generate --plan now refuses this plan" in out
    stop = read_stop_record(world.directory, "validation", PLAN)
    assert stop is not None and (stop["position"], stop["reason"], stop["records"]["lines"]) == (3, "budget", 3)
    assert stop["statuses"] == {"done": 3} and stop["later_positions_with_records"] == []

    # generate refuses before reading the credential: no request, no record
    monkeypatch.delenv("GROQ_API_KEY")
    transport = MockTransport(validation_reply)
    for argv in (("--confirm-api-use",), ()):
        code, _, err = cli(world.config_path, "generate", "--split", "validation", "--plan", PLAN, *argv,
                           transport=transport)
        assert code == 1 and "was stopped at position 3 of 3" in err and "no request is sent" in err
    assert transport.calls == [] and records.read_bytes() == before

    code, out, _ = cli(world.config_path, "stop-plan", "--split", "validation", "--plan", PLAN, "--reason", "again")
    assert code == 0 and "unchanged (already stopped there)" in out

    # a plan stopped earlier: the stop is the last position with a record, and a set must end there
    sources = {p.passage_id: p for p in world.prepared.source_passages("validation", world.passages, min_words=1)}
    plan, planned = load_plan(world.directory, "validation", PLAN, sources,
                              prepared_manifest_sha256=world.prepared.manifest_sha256)
    state = {r["passage_id"]: r for r in world.records() if r["passage_id"] != planned[1].passage_id}
    earlier = make_stop_record(plan, planned, state, plan_sha256=stop["plan_sha256"], records=stop["records"],
                               reason="test", stopped_at="2026-11-04T00:00:00Z")
    assert earlier["position"] == 1 and earlier["later_positions_with_records"] == [3]
    with pytest.raises(PlanError, match="already stops plan"):
        write_stop_record(world.directory, earlier)
    stop_path(world.directory, "validation", PLAN).unlink()
    assert write_stop_record(world.directory, earlier)
    decisions = world.review("stopped")
    with pytest.raises(ValidationSetError, match="stopped at position 1; set plan_positions to 1"):
        world.build(_write_config(world, "val_stopped", decisions=str(decisions)))
    first = world.root / "first.jsonl"
    first.write_text(decisions.read_text().splitlines(keepends=True)[0])
    built = world.build(_write_config(world, "val_stopped", plan_positions=1, decisions=str(first)))
    assert built.manifest["plan"]["stop"]["position"] == 1 and len(built.log) == 1

    stop_file = stop_path(world.directory, "validation", PLAN)
    stop_file.write_text(stop_file.read_text().replace(earlier["plan_sha256"], "0" * 64))
    with pytest.raises(PlanError, match="another version of"):
        read_stop_record(world.directory, "validation", PLAN)
    with pytest.raises(PlanError, match="needs a reason"):
        make_stop_record(plan, planned, state, plan_sha256="", records={}, reason=" ", stopped_at="")
