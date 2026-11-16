"""Validation-based checkpoint selection: the fixed rule, its record and hashes, the validated trainer run, seeds and
resumption, and the validation evaluator (offline, CPU, synthetic data)."""

from __future__ import annotations

import builtins
import inspect
import io
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from finretriever.data.schemas import Passage, Qrel, SearchHit  # noqa: E402
from finretriever.evaluation.metrics import compute_per_query_metrics  # noqa: E402
from finretriever.training import biencoder_trainer, trainer_setup, validation_eval  # noqa: E402
from finretriever.training.biencoder_trainer import TrainerError, main, train  # noqa: E402
from finretriever.training.checkpoint_selection import (  # noqa: E402
    PATIENCE,
    SELECTION_FILE,
    SelectionError,
    SelectionState,
    checkpoint_files,
    evaluation_steps,
    files_digest,
    is_candidate_step,
    read_selection_record,
    select,
    select_through,
    verify_selected_checkpoint,
)
from finretriever.training.trainer_setup import TrainerSetupError, plan_steps  # noqa: E402
from tests.unit.trainer_fixtures import (  # noqa: E402
    TinyEncoder,
    validated_config,
    validated_inputs,
    validated_world,
)
from tests.unit.validation_fixtures import World  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return validated_world(tmp_path_factory.mktemp("selection"))


def scripted(values: Mapping[int, float]) -> Any:
    """An evaluator returning fixed whole-corpus MRR@10 values by step (the rule's input), no model involved."""

    def evaluate(encoder: Any, step: int) -> dict[str, Any]:
        return {"global": {"mrr@10": values[step]}, "company": {"mrr@10": values[step]}}

    return evaluate


def steps_of(run_dir: Path) -> list[int]:
    return sorted(int(p.name.rsplit("-", 1)[1]) for p in run_dir.glob("checkpoint-step-*"))


# --- the rule -----------------------------------------------------------------------------------


def test_the_rule_evaluates_step_0_and_every_5_steps_and_reports_a_final_step_only() -> None:
    assert evaluation_steps(12) == [0, 5, 10, 12] and evaluation_steps(10) == [0, 5, 10] and evaluation_steps(3) == [0, 3]
    assert [s for s in range(13) if is_candidate_step(s)] == [0, 5, 10]


def test_the_best_wins_ties_keep_the_earlier_step_and_three_misses_stop_training() -> None:
    state = SelectionState()
    for step, value in ((0, 0.40), (5, 0.55), (10, 0.55), (15, 0.54999999999999)):
        state.record(step, {"global": {"mrr@10": value}})
    assert (state.best_step, state.since_best, state.should_stop) == (5, 2, False)
    entry = state.record(20, {"global": {"mrr@10": 0.55 + 1e-14}})  # equal to 12 decimal places: a tie
    assert entry["improved"] is False and state.should_stop and state.since_best == PATIENCE
    with pytest.raises(SelectionError, match="stopped early"):
        state.record(25, {"global": {"mrr@10": 0.9}})
    assert select(state.evaluations) == (5, True)
    # A final step that is not a multiple of 5 is reported but never selected, however high.
    final = SelectionState()
    for step, value in ((0, 0.3), (5, 0.4), (7, 0.99)):
        final.record(step, {"global": {"mrr@10": value}})
    assert final.evaluations[-1]["candidate"] is False and final.evaluations[-1]["improved"] is None
    assert select(final.evaluations) == (5, False)
    # Nothing beats step 0: the base model is selected.
    base = SelectionState()
    for step, value in ((0, 0.5), (5, 0.5), (10, 0.4), (15, 0.49)):
        base.record(step, {"global": {"mrr@10": value}})
    assert select(base.evaluations) == (0, True)
    with pytest.raises(SelectionError, match="not after"):
        base.record(15, {"global": {"mrr@10": 0.1}})


# --- the validated trainer run ------------------------------------------------------------------


def test_a_validated_run_evaluates_the_rule_steps_and_records_every_checkpoint_hash(world: World,
                                                                                     tmp_path: Path) -> None:
    inputs = validated_inputs(world, tmp_path)
    assert len(inputs.validation) == 2 and inputs.identity["validation"]["version"] == "val_v1"
    said: list[str] = []
    result = train(inputs, TinyEncoder(), say=said.append)  # the real evaluator, over the whole synthetic corpus
    assert steps_of(result.run_dir) == [0, 5, 10, 12] and result.steps_done == 12
    record = read_selection_record(result.run_dir / SELECTION_FILE)
    assert [e["step"] for e in record["evaluations"]] == [0, 5, 10, 12]
    assert [e["candidate"] for e in record["evaluations"]] == [True, True, True, False]
    values = {e["step"]: e["value"] for e in record["evaluations"] if e["candidate"]}
    best = max(values.values())
    assert record["selected"]["step"] == min(s for s, v in values.items() if round(v, 12) == round(best, 12))
    assert record["validation"]["version"] == "val_v1" and "not gold" in record["notice"]
    assert record["seeds"] == [42] and record["stopped"] == {"early": False, "at_step": 12, "planned_steps": 12}
    for entry in record["evaluations"]:  # each digest is the checkpoint as it is on disk
        directory = result.run_dir / entry["checkpoint"]
        assert checkpoint_files(directory) == entry["files"] and files_digest(entry["files"]) == entry["digest"]
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["validation"]["mrr@10_global"] == entry["value"]
        assert "only to choose among this run's checkpoints" in metadata["notice"]
        per_question = entry["metrics"]["per_question"]
        assert [q["pair_id"] for q in per_question] == [p.pair_id for p in inputs.validation]
    assert json.loads((result.run_dir / "checkpoint-step-000000" / "metadata.json").read_text())["loss_history"] == []
    assert "never selected" in json.loads((result.run_dir / "checkpoint-step-000012" / "metadata.json").read_text())[
        "selection"]
    log = [json.loads(line) for line in (result.run_dir / "validation_log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in log] == [0, 5, 10, 12] and log[1]["digest"] == record["evaluations"][1]["digest"]
    assert any("validation at step 5 (a selection candidate)" in s for s in said)
    run_record = json.loads((result.run_dir / "run.json").read_text())
    assert run_record["selection_rule"]["evaluated_steps"] == [0, 5, 10, 12]
    # The benchmark side: the selected checkpoint verifies; any other one, or a changed one, does not.
    selected = result.run_dir / record["selected"]["checkpoint"]
    summary = verify_selected_checkpoint(result.run_dir / SELECTION_FILE, selected)
    assert summary["selected_step"] == record["selected"]["step"] and summary["seeds"] == [42]
    other = next(result.run_dir / e["checkpoint"] for e in record["evaluations"] if e["step"] != summary["selected_step"])
    with pytest.raises(SelectionError, match="is not the checkpoint run"):
        verify_selected_checkpoint(result.run_dir / SELECTION_FILE, other)
    original = (selected / "encoder.json").read_bytes()
    try:
        (selected / "encoder.json").write_text("{}")
        with pytest.raises(SelectionError, match="differs from the checkpoint the selection hashed"):
            verify_selected_checkpoint(result.run_dir / SELECTION_FILE, selected)
    finally:
        (selected / "encoder.json").write_bytes(original)
    # Written once: the record is never written over, and an edited record is refused when read.
    saved = (result.run_dir / SELECTION_FILE).read_bytes()
    edited = json.loads(saved)
    edited["selected"]["step"] = next(e["step"] for e in edited["evaluations"] if e["step"] != edited["selected"]["step"])
    try:
        (result.run_dir / SELECTION_FILE).write_text(json.dumps(edited))
        with pytest.raises(SelectionError, match="the rule selects step"):
            read_selection_record(result.run_dir / SELECTION_FILE)
    finally:
        (result.run_dir / SELECTION_FILE).write_bytes(saved)


def test_early_stopping_fixes_the_selection_and_training_continues_to_the_planned_end(world: World,
                                                                                    tmp_path: Path) -> None:
    inputs = validated_inputs(world, tmp_path / "tie", max_steps=40)
    values = {0: 0.50, 5: 0.60, 10: 0.60, 15: 0.55, 20: 0.60, 25: 0.99, 30: 0.98, 35: 0.97, 40: 0.96}
    said: list[str] = []
    result = train(inputs, TinyEncoder(), evaluate=scripted(values), say=said.append)
    assert steps_of(result.run_dir) == [0, 5, 10, 15, 20, 25, 30, 35, 40] and result.steps_done == 40
    record = read_selection_record(result.run_dir / SELECTION_FILE)
    # The rule stopped at step 20 (a tie at 10 and 20 keeps step 5); 0.99 at step 25 cannot change that.
    assert record["selected"]["step"] == 5 and record["stopped"] == {"early": True, "at_step": 40, "planned_steps": 40}
    assert [e["step"] for e in record["evaluations"]] == [0, 5, 10, 15, 20] and record["rule_stop"]["step"] == 20
    after = record["after_stop_evaluations"]
    assert [e["step"] for e in after] == [25, 30, 35, 40] and [e["value"] for e in after] == [0.99, 0.98, 0.97, 0.96]
    assert all(e["selectable"] is False and e["after_stop"] is True and e["candidate"] is False for e in after)
    assert record["last_evaluated"]["step"] == 40 and record["environment"]["device_type"] == "cpu"
    for entry in after:  # hashed and marked on disk like every evaluated checkpoint
        directory = result.run_dir / entry["checkpoint"]
        assert checkpoint_files(directory) == entry["files"]
        metadata = json.loads((directory / "metadata.json").read_text())
        assert "never selectable" in metadata["selection"] and metadata["validation"]["selectable"] is False
    log = [json.loads(line) for line in (result.run_dir / "validation_log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in log] == [0, 5, 10, 15, 20, 25, 30, 35, 40]
    assert any("the rule stops selecting at step 20" in s and "step 5 stays selected" in s for s in said)
    assert any("validation at step 25 (after the rule's stop: recorded, never selectable)" in s for s in said)
    summary = verify_selected_checkpoint(result.run_dir / SELECTION_FILE, result.run_dir / "checkpoint-step-000005")
    assert summary["selected_step"] == 5
    with pytest.raises(SelectionError, match="is not the checkpoint run"):  # an after-stop checkpoint never is
        verify_selected_checkpoint(result.run_dir / SELECTION_FILE, result.run_dir / "checkpoint-step-000025")
    # Nothing improves on the base model: step 0 is selected, and its checkpoint is the base weights.
    base = validated_inputs(world, tmp_path / "base", max_steps=40)
    result = train(base, TinyEncoder(), evaluate=scripted({0: 0.5, 5: 0.4, 10: 0.5, 15: 0.45, **{
        s: 0.9 for s in range(20, 41, 5)}}))
    record = read_selection_record(result.run_dir / SELECTION_FILE)
    assert record["selected"] == {**record["selected"], "step": 0, "checkpoint": "checkpoint-step-000000",
                                  "base_model": True}
    initial = TinyEncoder().state_dict()
    saved = torch.load(result.run_dir / "checkpoint-step-000000" / "weights.pt", weights_only=True)
    assert all(torch.equal(initial[k], saved[k]) for k in initial)
    assert record["rule_stop"]["step"] == 15 and [e["step"] for e in record["after_stop_evaluations"]] == [
        20, 25, 30, 35, 40]


def test_evaluations_after_the_stop_are_recorded_apart_and_never_selectable() -> None:
    state = SelectionState()
    with pytest.raises(SelectionError, match="has not stopped"):
        state.record_after_stop(0, {"global": {"mrr@10": 0.5}})
    for step, value in ((0, 0.5), (5, 0.6), (10, 0.5), (15, 0.5), (20, 0.5)):
        state.record(step, {"global": {"mrr@10": value}})
    assert state.should_stop
    entry = state.record_after_stop(25, {"global": {"mrr@10": 0.99}})
    assert entry["selectable"] is False and entry["candidate"] is False and entry["after_stop"] is True
    with pytest.raises(SelectionError, match="not after"):
        state.record_after_stop(25, {"global": {"mrr@10": 0.1}})
    assert (state.best_step, select(state.evaluations)) == (5, (5, True))  # unchanged by the later evaluation
    again = SelectionState.from_json(json.loads(json.dumps(state.to_json())))  # the trainer state round-trips
    assert again.after_stop == state.after_stop and again.evaluations == state.evaluations
    # The rule over every planned evaluation stops and selects as a run records it, ignoring what follows the stop.
    full = [{"step": e["step"], "value": e["value"]} for e in [*state.evaluations, *state.after_stop]]
    assert select_through(full) == (5, 20)
    assert select_through([{"step": 0, "value": 0.5}, {"step": 5, "value": 0.6}, {"step": 7, "value": 0.9}]) == (5, None)
    with pytest.raises(SelectionError, match="nothing to select"):
        select_through([])


def test_a_record_must_hold_every_planned_evaluation_and_cannot_be_edited(world: World, tmp_path: Path) -> None:
    inputs = validated_inputs(world, tmp_path, max_steps=30)
    values = {0: 0.5, 5: 0.7, 10: 0.6, 15: 0.6, 20: 0.65, 25: 0.9, 30: 0.8}
    result = train(inputs, TinyEncoder(), evaluate=scripted(values))
    path = result.run_dir / SELECTION_FILE
    saved = path.read_bytes()
    assert read_selection_record(path)["selected"]["step"] == 5

    def refused(edit: Any, message: str) -> None:
        record = json.loads(saved)
        edit(record)
        path.write_text(json.dumps(record))
        try:
            with pytest.raises(SelectionError, match=message):
                read_selection_record(path)
        finally:
            path.write_bytes(saved)

    refused(lambda r: r["after_stop_evaluations"].pop(), "every planned evaluation step")  # incomplete
    refused(lambda r: r["after_stop_evaluations"][0].update(selectable=True), "never selectable")
    refused(lambda r: r["after_stop_evaluations"][0].update(candidate=True), "never selectable")
    refused(lambda r: r.update(rule_stop=None), "rule stop")
    refused(lambda r: r["stopped"].update(at_step=25), "every planned evaluation step")
    refused(lambda r: r["after_stop_evaluations"][0].update(digest="0" * 64), "digest of step 25")
    # Moving an after-stop evaluation into the rule's evaluations would make it a candidate: the rule then selects it.
    refused(lambda r: r["evaluations"].append(r["after_stop_evaluations"].pop(0)), "follows the early stop")
    assert read_selection_record(path)["selected"]["step"] == 5


def test_a_run_interrupted_after_the_stop_resumes_its_after_stop_evaluations(world: World, tmp_path: Path) -> None:
    values = {0: 0.5, 5: 0.7, 10: 0.6, 15: 0.6, 20: 0.65, 25: 0.9, 30: 0.8}
    straight = train(validated_inputs(world, tmp_path / "a", max_steps=30), TinyEncoder(), evaluate=scripted(values))
    interrupted = validated_inputs(world, tmp_path / "b", max_steps=30)
    first = train(interrupted, TinyEncoder(), evaluate=scripted(values), stop_after=22)
    assert first.steps_done == 22 and not (first.run_dir / SELECTION_FILE).exists()
    resumed = train(interrupted, TinyEncoder(), evaluate=scripted(values), resume=True)
    a = read_selection_record(straight.run_dir / SELECTION_FILE)
    b = read_selection_record(resumed.run_dir / SELECTION_FILE)
    assert [(e["step"], e["value"]) for e in a["after_stop_evaluations"]] == [
        (e["step"], e["value"]) for e in b["after_stop_evaluations"]] == [(25, 0.9), (30, 0.8)]
    assert a["selected"]["step"] == b["selected"]["step"] == 5 and a["rule_stop"] == {
        **a["rule_stop"], "step": 20} and b["rule_stop"]["step"] == 20
    assert [h["loss"] for h in straight.history] == [h["loss"] for h in resumed.history]


class Diverging(TinyEncoder):
    """Encodes normally until its ``bad_call``-th forward pass, which returns NaN (a diverged or overflowing step)."""

    def __init__(self, bad_call: int) -> None:
        super().__init__()
        self.calls, self.bad_call = 0, bad_call

    def encode(self, texts: Sequence[str], max_length: int) -> Any:
        self.calls += 1
        out = super().encode(texts, max_length)
        return out * float("nan") if self.calls == self.bad_call else out


def test_a_validated_run_stops_on_non_finite_gradients_without_an_optimizer_step(world: World,
                                                                                 tmp_path: Path) -> None:
    inputs = validated_inputs(world, tmp_path, max_steps=12)
    values = {s: 0.5 for s in (0, 5, 10, 12)}
    encoder = Diverging(bad_call=2 * 7 - 1)  # two encodes (queries, passages) per step: step 7's first one
    with pytest.raises(TrainerError, match="step 7: the gradients are not finite.*before the optimizer step"):
        train(inputs, encoder, evaluate=scripted(values))
    run_dir = Path(inputs.config.checkpoints.output_dir) / inputs.config.run_name
    assert steps_of(run_dir) == [0, 5] and not (run_dir / SELECTION_FILE).exists()
    log = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    assert [h["step"] for h in log] == [1, 2, 3, 4, 5, 6]  # step 7 was never taken
    assert all(torch.isfinite(p).all() for p in encoder.parameters())  # no NaN reached the weights


def test_an_interrupted_validated_run_resumes_to_the_same_checkpoints_and_selection(world: World,
                                                                                     tmp_path: Path) -> None:
    straight = train(validated_inputs(world, tmp_path / "a"), TinyEncoder())
    interrupted = validated_inputs(world, tmp_path / "b")
    first = train(interrupted, TinyEncoder(), stop_after=7)
    assert first.steps_done == 7 and steps_of(first.run_dir) == [0, 5, 7]
    assert "interruption checkpoint" in json.loads((first.run_dir / "checkpoint-step-000007" / "metadata.json")
                                                   .read_text())["selection"]
    assert not (first.run_dir / SELECTION_FILE).exists()  # the run has not ended
    with pytest.raises(TrainerError, match="--resume"):
        train(interrupted, TinyEncoder())  # never written over
    resumed = train(interrupted, TinyEncoder(), resume=True)
    assert resumed.steps_done == 12 and steps_of(resumed.run_dir) == [0, 5, 7, 10, 12]
    a = read_selection_record(straight.run_dir / SELECTION_FILE)
    b = read_selection_record(resumed.run_dir / SELECTION_FILE)
    assert [(e["step"], e["value"]) for e in a["evaluations"]] == [(e["step"], e["value"]) for e in b["evaluations"]]
    assert a["selected"]["step"] == b["selected"]["step"]
    assert [h["loss"] for h in straight.history] == [h["loss"] for h in resumed.history]
    final_a = torch.load(straight.run_dir / "checkpoint-step-000012" / "weights.pt", weights_only=True)
    final_b = torch.load(resumed.run_dir / "checkpoint-step-000012" / "weights.pt", weights_only=True)
    assert all(torch.equal(final_a[k], final_b[k]) for k in final_a)
    # A finished run resumed again trains nothing and writes nothing.
    before = {p: p.stat().st_mtime_ns for p in resumed.run_dir.rglob("*") if p.is_file() and p.name != "log.jsonl"}
    again = train(interrupted, TinyEncoder(), resume=True)
    assert again.selection == resumed.run_dir / SELECTION_FILE and not again.checkpoints
    assert {p: p.stat().st_mtime_ns for p in before} == before


def test_three_seeds_train_one_run_each_in_their_own_directories(world: World, tmp_path: Path) -> None:
    config, path = validated_config(world, tmp_path, seeds=[42, 1337, 2026], max_steps=5)
    assert config.seed_list == (42, 1337, 2026)
    with pytest.raises(TrainerSetupError, match="choose one with --seed"):
        _ = config.run_name
    with pytest.raises(TrainerSetupError, match="not one of the config's seeds"):
        config.for_seed(7)
    out = io.StringIO()
    assert main(["--config", str(path)], out=out) == 0  # a dry run shows each seed's run and schedule
    text = out.getvalue()
    assert all(f"run tiny-seed{s}:" in text for s in (42, 1337, 2026)) and "dry run" in text
    assert "checkpoint selection only, by the fixed rule: whole-corpus MRR@10 at step 0 and every 5 steps" in text
    err = io.StringIO()
    assert main(["--config", str(path), "--confirm-training"], encoder_loader=lambda m: TinyEncoder(),
                out=io.StringIO(), err=err) == 1
    assert "train one seed per run: pass --seed" in err.getvalue() and not (tmp_path / "models").exists()
    assert main(["--config", str(path), "--seed", "7"], out=io.StringIO(), err=io.StringIO()) == 1
    schedules = {}
    for seed in (42, 1337, 2026):
        out = io.StringIO()
        assert main(["--config", str(path), "--seed", str(seed), "--confirm-training"],
                    encoder_loader=lambda m: TinyEncoder(), out=out) == 0
        run_dir = tmp_path / "models" / f"tiny-seed{seed}"
        record = json.loads((run_dir / "run.json").read_text())
        assert record["config"]["training"]["seed"] == seed and record["config"]["training"]["seeds"] == [42, 1337, 2026]
        selection = read_selection_record(run_dir / SELECTION_FILE)
        assert (selection["seed"], selection["seeds"]) == (seed, [42, 1337, 2026])
        schedules[seed] = record["schedule"]["sha256"]
        assert "checkpoints chosen only by the validation rule" in out.getvalue()
    assert len(set(schedules.values())) == 3  # each seed orders the batches its own way


def test_the_validation_set_is_checked_against_the_pairs_trained_on(world: World, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    real = trainer_setup.load_validation_set

    def spy(*args: Any, training: Sequence[Any] | None = None, **kwargs: Any) -> Any:
        seen["training"] = training
        return real(*args, training=training, **kwargs)

    monkeypatch.setattr(trainer_setup, "load_validation_set", spy)
    inputs = validated_inputs(world, tmp_path, max_pairs=3, subset_seed=1)
    assert [p.pair_id for p in seen["training"]] == [p.pair_id for p in inputs.pairs] and len(inputs.pairs) == 3
    guard = inputs.guard
    assert not any(guard.allowed_negative(t, pid) for t in inputs.pairs for v in inputs.validation
                   for pid in v.positive_ids)  # a validation passage is never a training negative
    config = validated_config(world, tmp_path / "missing")[0]
    missing = config.model_copy(update={"validation": trainer_setup.ValidationConfig(version="missing_v1")})
    with pytest.raises(ValueError, match="missing_v1"):
        trainer_setup.prepare_inputs(missing, "0" * 64)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checkpoints": {"output_dir": "models/x", "save_every_steps": 2}}, "leave save_every_steps unset"),
        ({"validation": {"version": "val_v1", "metric": "ndcg@10"}}, "Extra inputs"),
        ({"validation": {"version": "val_v1", "every_steps": 2}}, "Extra inputs"),
        ({"training": {"seeds": [42, 42]}}, "repeat a seed"),
        ({"training": {"seed": 7, "seeds": [42, 1337]}}, "not one of seeds"),
        ({"training": {"mini_batch_size": 4}}, "exceeds batch_size"),
    ],
)
def test_the_rule_cannot_be_configured_away(world: World, tmp_path: Path, change: dict[str, Any], message: str) -> None:
    import yaml

    _, path = validated_config(world, tmp_path)
    raw = yaml.safe_load(path.read_text())
    for section, values in change.items():
        raw[section] = {**(raw.get(section) or {}), **values}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(TrainerSetupError, match=message):
        trainer_setup.load_trainer_config(path)


def test_validated_training_reads_no_evaluation_file(world: World, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    real_open = io.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.path.realpath(file))
        return real_open(file, *args, **kwargs)

    _, path = validated_config(world, tmp_path, max_steps=6)
    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    assert main(["--config", str(path), "--confirm-training"], encoder_loader=lambda m: TinyEncoder(),
                out=io.StringIO()) == 0
    root = os.path.realpath(world.root)
    evaluation = [os.path.join(root, name) for name in ("annotation", "dev_ai_v3", "audit", "gold_qrels.jsonl",
                                                        "gold_queries.jsonl", "gold_eval.metadata.json")]
    assert any("validation_sets" in f for f in opened)  # the validation set was read, from the preparation
    assert not [f for f in opened if any(f == p or f.startswith(p + os.sep) for p in evaluation)]
    # The semantic leakage screen reads evaluation question texts; nothing in training or selection uses it.
    for module in (biencoder_trainer, trainer_setup, validation_eval):
        assert "semantic_screen" not in inspect.getsource(module)


# --- the validation evaluator -------------------------------------------------------------------


class Fixed:
    """Embeds each text as a fixed vector (no parameters)."""

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: Sequence[str], max_length: int) -> Any:
        return torch.tensor([self.vectors[t] for t in texts], dtype=torch.float32)


def _passage(pid: str, ticker: str) -> Passage:
    return Passage(passage_id=pid, filing_id="f", section_id="s", ticker=ticker, form_type="10-K",
                   filing_date="2024-01-01", section_title="t", text=pid, token_count=1)


def test_rankings_follow_the_records_and_the_metrics_are_the_benchmarks() -> None:
    corpus = {pid: _passage(pid, t) for pid, t in (("p1", "AAA"), ("p2", "AAA"), ("p3", "BBB"), ("p4", "AAA"),
                                                   ("p5", "AAA"))}
    vectors = {"Q:q1": [1, 0], "Q:q2": [0, 1], "p1": [0.9, 0.1], "p2": [0.95, 0.05], "p3": [0.99, 0.01],
               "p4": [0.9, 0.1], "p5": [0.1, 0.9]}
    targets = validation_eval.ValidationTargets(
        pair_ids=("v1", "v2"), queries=("Q:q1", "Q:q2"), tickers=("AAA", "AAA"),
        relevant=(frozenset({"p4"}), frozenset({"p5", "p1"})), removed=(frozenset({"p2"}), frozenset()),
        passage_ids=tuple(corpus), passage_texts=tuple(corpus), passage_tickers=tuple(p.ticker for p in corpus.values()),
    )
    result = validation_eval.evaluate(Fixed(vectors), targets, query_max_length=8, passage_max_length=8, batch_size=2,
                                      device=torch.device("cpu"))
    q1, q2 = result["per_question"]
    # q1: p2 is removed (it may answer); p3 ranks first globally; p1 and p4 tie and keep corpus order.
    assert q1["removed"] == 1 and q1["global"]["first_relevant_rank"] == 3 and q1["company"]["first_relevant_rank"] == 2
    assert q1["global"]["mrr@10"] == pytest.approx(1 / 3) and q1["company"]["mrr@10"] == pytest.approx(1 / 2)
    assert q2["global"]["first_relevant_rank"] == 1 and result["passages_ranked"] == 5 and result["questions"] == 2
    ranked = validation_eval.rank(torch.tensor([0.9, 0.95, 0.99, 0.9, 0.1], dtype=torch.float64),
                                  torch.tensor([True, False, True, True, True]), list(corpus))
    assert ranked == ["p3", "p1", "p4", "p5"]
    hits = [SearchHit(query_id="v1", passage_id=pid, score=float(10 - i), rank=i + 1, retriever="r")
            for i, pid in enumerate(ranked)]
    benchmark = compute_per_query_metrics([Qrel(query_id="v1", passage_id="p4", relevance=1)], hits,
                                          ["mrr@10", "ndcg@10", "recall@100"])["v1"]
    assert {k: q1["global"][k] for k in benchmark} == pytest.approx(benchmark)
    assert result["global"]["mrr@10"] == pytest.approx((1 / 3 + 1) / 2)


def test_evaluating_changes_no_weight_mode_or_random_state(world: World, tmp_path: Path) -> None:
    inputs = validated_inputs(world, tmp_path)
    targets = validation_eval.build_targets(inputs.validation, inputs.guard, inputs.corpus,
                                            query_prefix="Find: ", passage_prefix="")
    for relevant, removed, pair in zip(targets.relevant, targets.removed, inputs.validation, strict=True):
        assert pair.positive_passage_id in relevant and not relevant & removed
        assert set(pair.equivalents.exact) <= relevant and set(pair.equivalents.uncertain) <= removed
    model = TinyEncoder()
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.get_rng_state()
    result = validation_eval.evaluate(model, targets, query_max_length=64, passage_max_length=40, batch_size=3,
                                      device=torch.device("cpu"))
    assert model.training and torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert result["passages_ranked"] == len(inputs.corpus) and 0 <= result["global"]["mrr@10"] <= 1
    again = validation_eval.evaluate(model, targets, query_max_length=64, passage_max_length=40, batch_size=5,
                                     device=torch.device("cpu"))
    assert again["global"] == pytest.approx(result["global"])  # eval mode: no dropout, batching does not matter
    assert plan_steps(inputs).steps  # the inputs still plan a schedule
