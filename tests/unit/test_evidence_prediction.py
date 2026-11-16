"""Tests for the evidence-pass predictor: unique figures, the calibrated fit, the write-once calibration, and its use
by a validation plan that skips everything already generated (offline, synthetic corpus)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from finretriever.training.data_prep import load_data_prep_config
from finretriever.training.evidence_prediction import (
    PREDICTOR_VERSION,
    PredictionError,
    auc,
    brier,
    build_calibration,
    figure_isolation,
    fit,
    load_calibration,
    mixture,
    write_calibration,
)
from finretriever.training.pair_quality import evidence_locator
from tests.unit.test_preparation_v3 import migrate
from tests.unit.validation_fixtures import EXTRA, World, block_network, build_world, cli


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    block_network(monkeypatch)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return build_world(tmp_path, monkeypatch)


def observation(pilot: str, position: int, kind: str, unique: int, passed: bool | None, review: str | None = "accept") -> dict:
    generated = passed is not None
    return {"pilot": pilot, "plan_position": position, "passage_id": f"p{pilot}{position}", "ticker": "AAA", "kind": kind,
            "figures": 4, "unique_figures": unique, "sentence_share": 0.5, "generated": generated,
            **({"question_sha256": "a" * 64, "quote_has_figure": unique > 1, "evidence_pass": passed, "review": review}
               if generated else {})}


def test_the_fit_metrics_and_mixture_are_deterministic_and_bounded() -> None:
    scores, outcomes = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], [False, False, True, False, True, True]
    a, b = fit(scores, outcomes)
    assert (a, b) == fit(scores, outcomes) and b > 0  # higher scores, likelier passes
    assert fit([0.1, 0.9], [False, True]) != (0.0, 1.0)  # separable data still converges, held near the identity
    assert auc(scores, outcomes) == pytest.approx(8 / 9) and auc([0.5], [True]) is None
    assert brier([1.0, 0.0], [True, False]) == 0.0 and brier([0.5, 0.5], [True, False]) == 0.25
    rate = {"narrative": 0.5, "table": 0.7}
    assert mixture("table", 0, 0, 0.4, rate) == 0.4  # no significant figure: the sentence share alone
    assert mixture("table", 4, 1, 0.4, rate) == pytest.approx(0.7 * 0.25 + 0.3 * 0.4)
    with pytest.raises(PredictionError):
        fit([], [])


def test_a_calibration_records_its_observations_and_is_written_once(tmp_path: Path) -> None:
    obs = [observation("v1", i, "table" if i % 2 else "narrative", unique=i % 4, passed=bool(i % 3)) for i in range(1, 13)]
    obs.append(observation("v1", 13, "table", unique=0, passed=None))  # rejected by the generator
    doc = build_calibration(obs, [{"plan": "v1"}])
    assert doc["rates"]["generator_acceptance"] == round(12 / 13, 4) and doc["rates"]["review_acceptance"] == 1.0
    assert doc["metrics"]["questions"] == 12 and len(doc["observations"]) == 13 and "relevance label" in doc["notice"]
    path = tmp_path / "cal.json"
    assert write_calibration(path, doc) is True and write_calibration(path, doc) is False
    with pytest.raises(PredictionError, match="never changed"):
        write_calibration(path, {**doc, "rates": {"generator_acceptance": 0.5, "review_acceptance": 0.5}})
    calibration = load_calibration(path)
    assert calibration.probability(0.9) > calibration.probability(0.1)
    assert calibration.usable(1.0) == pytest.approx(calibration.generator_acceptance * calibration.review_acceptance)
    assert PREDICTOR_VERSION in calibration.method() and calibration.sha256[:12] in calibration.method()
    (tmp_path / "other.json").write_text(json.dumps({**doc, "predictor": "evidence-pass-v0"}))
    with pytest.raises(PredictionError, match="calibrates"):
        load_calibration(tmp_path / "other.json")
    with pytest.raises(PredictionError, match="both pass and fail"):
        build_calibration([observation("v1", 1, "table", 1, True)], [])


def test_figures_held_outside_validation_are_not_unique(world: World) -> None:
    locator = evidence_locator(world.prepared, world.passages)
    revenue = world.prepared.source_passage(EXTRA["gamma23_revenue"], world.passages)
    n, unique = figure_isolation(revenue, locator, world.prepared.rows)
    assert n >= 2 and unique < n  # a train passage states its $3,036 million as the prior year


def test_a_calibrated_plan_skips_everything_generated_and_records_how_it_ranked(world: World) -> None:
    decisions = world.review("val_decisions")
    cal = world.root / "evidence_pass.json"
    argv = ("calibrate-predictor", "--out", str(cal), "--pilot", str(world.config_path), "val_a", str(decisions), "3")
    code, out, err = cli(world.config_path, *argv)
    assert code == 0 and "wrote" in out and "AUC" in out, err
    assert "unchanged (identical calibration present)" in cli(world.config_path, *argv)[1]
    doc = json.loads(cal.read_text())
    assert [o["evidence_pass"] for o in doc["observations"]].count(False) >= 1  # the revenue question
    assert doc["sources"][0]["records"]["lines"] == 3 and doc["sources"][0]["decisions"]["sha256"]
    code, _, err = cli(world.config_path, "calibrate-predictor", "--out", str(world.root / "x.json"), "--pilot",
                       str(world.config_path), "val_a", str(decisions), "9")
    assert code == 1 and "has 3 passages" in err

    # A later preparation with a larger validation split; the plan skips the three generated passages' groups.
    config_v1 = world.validation_config("val_v1", decisions)
    assert cli(world.config_path, "build-validation", "--validation", str(config_v1))[0] == 0
    pins = world.root / "pins.json"
    assert cli(world.config_path, "pin-split", "--candidates", "cand_v1", "--pairs", "--keep-validation",
               "--out", str(pins))[0] == 0
    new_path, new = migrate(world, "test_v2", pins, validation_share_by_ticker={"GAMMA": 0.45})
    generated = {r["passage_id"] for r in world.records()}
    plan_argv = ("plan", "--split", "validation", "--name", "val_c", "--size", "1", "--method", "balanced_years",
                 "--max-company-share", "1.0", "--one-per", "group", "--max-per-unit", "3", "--predictor", str(cal),
                 "--score-priority", "--administrative-rule", "--after-generated", str(world.config_path),
                 "--exclude-equivalents-of", f"{world.config_path}:val_v1")
    code, out, err = cli(new_path, *plan_argv)
    assert code == 0 and "predicted evidence pass" in out and "estimated usable questions" in out, err
    plan = json.loads((new.directory / "plans" / "validation.val_c.json").read_text())
    groups = {new.rows[p]["group_id"] for p in generated}
    assert all(p["passage_id"] not in generated and p["group_id"] not in groups for p in plan["passages"])
    c = plan["constraints"]
    assert c["score_priority"] is True and c["prior_counted"]["units"] == 3
    assert PREDICTOR_VERSION in c["scores"]["method"] and 0 <= plan["passages"][0]["score"] <= 1
    assert f"generated:validation:{world.config_path}:3 passages" in c["excluded_from"]
    assert plan["source_quality"]["settings"]["administrative_rules"] is True
    code, _, err = cli(new_path, "plan", "--split", "validation", "--name", "val_d", "--size", "1", "--method",
                       "balanced_years", "--min-predicted-pass", "0.5")
    assert code == 1 and "needs --predictor" in err
    config_new = load_data_prep_config(new_path)[0]
    assert yaml.safe_load(Path(new_path).read_text())["source_quality"] == {"context_rules": True}  # config unchanged
    assert config_new.source_quality.administrative_rules is False
