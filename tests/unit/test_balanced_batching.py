"""Balanced batching in the trainer (``training.batching: balanced``): every pair in every epoch, the same
false-negative protections, and unchanged records for configs that do not set it (offline, CPU, synthetic data)."""

from __future__ import annotations

import dataclasses
import io
import json
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from finretriever.training.biencoder_trainer import main, train  # noqa: E402
from finretriever.training.trainer_setup import (  # noqa: E402
    TrainerSetupError,
    check_batch,
    load_trainer_config,
    plan_steps,
)
from tests.unit.trainer_fixtures import (  # noqa: E402
    TinyEncoder,
    validated_config,
    validated_inputs,
    validated_world,
)
from tests.unit.validation_fixtures import World  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return validated_world(tmp_path_factory.mktemp("balanced"))


def test_every_pair_trains_in_every_epoch_in_near_equal_batches(world: World, tmp_path: Path) -> None:
    inputs = validated_inputs(world, tmp_path, batching="balanced", epochs=4, max_steps=None)
    schedule = plan_steps(inputs)
    assert schedule.batching == "balanced" and schedule.dropped_batches == 0
    ids = sorted(p.pair_id for p in inputs.pairs)
    for epoch in range(4):
        batches = [b for s in schedule.steps if s.epoch == epoch for b in s.batches]
        assert sorted(i for b in batches for i in b) == ids  # each pair exactly once per epoch
        assert max(map(len, batches)) - min(map(len, batches)) <= 1 and max(map(len, batches)) <= 3
        for batch in batches:
            check_batch(inputs, batch)  # the trainer's own check: only allowed in-batch negatives
    assert len(set(schedule.batches_per_epoch)) == 1
    negatives = schedule.negatives
    assert negatives["min"] >= 1 and negatives["mean"] <= 2
    # Each seed plans its own batches.
    seven = inputs.config.model_copy(update={"training": inputs.config.training.model_copy(update={"seed": 7})})
    assert plan_steps(dataclasses.replace(inputs, config=seven)).sha256 != schedule.sha256


def test_configs_without_batching_keep_their_schedule_and_records(world: World, tmp_path: Path) -> None:
    unset = validated_inputs(world, tmp_path / "unset", validation=False, max_steps=3)
    explicit = validated_inputs(world, tmp_path / "explicit", validation=False, max_steps=3, batching="first_fit")
    assert plan_steps(unset).sha256 == plan_steps(explicit).sha256 and plan_steps(unset).batching == "first_fit"
    result = train(unset, TinyEncoder())
    record = json.loads((result.run_dir / "run.json").read_text())
    assert "batching" not in record["config"]["training"] and "batching" not in record["schedule"]
    meta = json.loads((result.run_dir / "checkpoint-step-000003" / "metadata.json").read_text())
    assert "batching" not in meta["objective"]
    balanced = validated_inputs(world, tmp_path / "balanced", validation=False, max_steps=3, batching="balanced")
    result = train(balanced, TinyEncoder())
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["config"]["training"]["batching"] == "balanced" and record["schedule"]["batching"] == "balanced"
    assert record["schedule"]["in_batch_negatives_per_query"] == plan_steps(balanced).negatives
    meta = json.loads((result.run_dir / "checkpoint-step-000003" / "metadata.json").read_text())
    assert meta["objective"]["batching"] == "balanced"


def test_the_dry_run_reports_the_planner_and_the_real_negatives(world: World, tmp_path: Path) -> None:
    _, path = validated_config(world, tmp_path, batching="balanced", seeds=[42, 1337, 2026], max_steps=None, epochs=3)
    out = io.StringIO()
    assert main(["--config", str(path)], out=out) == 0
    text = out.getvalue()
    runs = [line for line in text.splitlines() if "optimizer steps" in line]
    assert len(runs) == 3 and all("balanced batches" in r and "0 single-pair batches dropped" in r for r in runs)
    assert all("in-batch negatives per query: at least" in r for r in runs)
    steps = Counter(int(r.split(": ")[1].split()[0]) for r in runs)
    assert len(steps) == 1  # balanced batching gives every seed the same number of steps


def test_an_unknown_planner_is_refused(world: World, tmp_path: Path) -> None:
    import yaml

    _, path = validated_config(world, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["training"]["batching"] = "masked"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(TrainerSetupError, match="batching"):
        load_trainer_config(path)
