"""The A0-versus-A1 decision rule on synthetic selection records (readiness check T7), and its simulation (offline)."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from finretriever.training import decision_rule, decision_simulation
from finretriever.training.checkpoint_selection import (
    SelectionState,
    evaluation_steps,
    files_digest,
    make_selection_record,
    write_selection_record,
)
from finretriever.training.decision_rule import (
    DecisionError,
    analyse,
    decide,
    main,
    seed_runs,
    write_decision,
)

SEEDS = (42, 1337, 2026)
STEPS = evaluation_steps(55)  # 0, 5, ..., 55: every planned evaluation of the experiment's schedule
Q = 40
VALIDATION = {"version": "validation_pairs_v4", "manifest_sha256": "a" * 64, "pairs": Q}
ENVIRONMENT = {"python": "3.11.13", "torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1", "cuda": "12.4",
               "cudnn": 90100, "device_type": "cuda", "gpu_name": "Tesla T4", "gpu_capability": "7.5",
               "gpu_memory_gib": 14.74, "platform": "Linux"}
RR = np.array([1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 1 / 10, 0.0])  # rank 1..10, then none


def curves(rng: np.random.Generator, gain_ranks: float, noise: float = 0.0) -> np.ndarray:
    """Per-question reciprocal ranks (questions x steps): ranks improve by ``gain_ranks`` from step 10 on, plus noise."""
    base = rng.integers(0, 11, size=Q)  # index into RR (10: not in the top 10)
    out = np.empty((Q, len(STEPS)))
    for k, step in enumerate(STEPS):
        shift = gain_ranks if step >= 10 else 0.0
        jitter = np.zeros(Q) if step == 0 else rng.normal(0.0, noise, Q)
        out[:, k] = RR[np.clip(np.round(base - shift + jitter), 0, 10).astype(int)]
    out[:, 0] = RR[base]
    return out


def write_records(root: Path, by_seed: Mapping[int, np.ndarray], *, seeds: Sequence[int] = SEEDS,
                  environment: Mapping[str, Any] | None = None, validation: Mapping[str, Any] = VALIDATION,
                  steps: Sequence[int] = STEPS) -> list[Path]:
    """Each seed's selection record, as the trainer writes it: the rule's evaluations, then the after-stop ones."""
    paths = []
    for seed, rr in by_seed.items():
        state = SelectionState()
        for k, step in enumerate(steps):
            per = [{"pair_id": f"v{q:03d}", "relevant": 1, "removed": 0,
                    "global": {"mrr@10": float(rr[q, k]), "ndcg@10": float(rr[q, k]), "recall@100": 1.0,
                               "first_relevant_rank": round(1 / rr[q, k]) if rr[q, k] else None},
                    "company": {"mrr@10": float(rr[q, k]), "first_relevant_rank": None}} for q in range(len(rr))]
            column = [p["global"]["mrr@10"] for p in per]
            metrics = {"global": {"mrr@10": sum(column) / len(column), "ndcg@10": sum(column) / len(column),
                                  "recall@100": 1.0},
                       "company": {"mrr@10": sum(column) / len(column)}, "per_question": per}
            entry = state.record_after_stop(step, metrics) if state.should_stop else state.record(step, metrics)
            files = {"model.safetensors": hashlib.sha256(f"{seed}:{step}".encode()).hexdigest()}
            entry.update(checkpoint=f"checkpoint-step-{step:06d}", files=files, digest=files_digest(files))
        record = make_selection_record(
            run=f"a1-seed{seed}", seed=seed, seeds=list(seeds), config_sha256="c" * 64, schedule_sha256="d" * 64,
            run_record_sha256="e" * 64, validation=validation, state=state, planned_steps=steps[-1],
            steps_done=steps[-1], environment=dict(environment or ENVIRONMENT),
        )
        run_dir = root / f"a1-seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_selection_record(run_dir, record)
        paths.append(run_dir / "selection.json")
    return paths


def gains(rng: np.random.Generator, ranks: float, noise: float) -> dict[int, np.ndarray]:
    base = curves(rng, 0.0)  # the shared step 0
    out = {}
    for seed in SEEDS:
        c = curves(rng, ranks, noise)
        c[:, 0] = base[:, 0]
        out[seed] = c
    return out


def test_a_clear_gain_is_decided_and_the_decision_is_reproducible(tmp_path: Path) -> None:
    paths = write_records(tmp_path, gains(np.random.default_rng(1), ranks=2.0, noise=0.5))
    decision = analyse(paths, flips=999)
    result = decision["result"]
    assert result["a1_beats_a0"] is True and result["conclusion"] == "A1 beats A0 on the validation set"
    assert result["p_value"] <= 0.05 and result["every_seed_selected_after_step_0"]
    assert result["statistic"] > result["selection_adjusted_estimate"] > 0  # the flip null's mean is the optimism
    assert set(result["seeds"]) == {"42", "1337", "2026"} and result["questions"] == Q
    assert all(s["evaluated_steps"] == STEPS for s in result["seeds"].values())
    low, high = result["intervals"]["selected_minus_step_0"]
    assert low < result["statistic"] < high
    assert decision == analyse(list(reversed(paths)), flips=999)  # argument order does not matter
    assert decision["inputs"][0]["sha256"] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert "not gold" in decision["notice"] and decision["method"]["flips"] == 999


def test_the_command_writes_the_decision_once_with_the_fixed_settings(tmp_path: Path) -> None:
    paths = write_records(tmp_path, gains(np.random.default_rng(2), ranks=2.0, noise=0.5))
    out_path = tmp_path / "decisions" / "a1_vs_a0.json"
    args = [a for p in paths for a in ("--selection", str(p))] + ["--out", str(out_path)]
    out = io.StringIO()
    assert main(args, out=out) == 0 and "A1 beats A0" in out.getvalue()
    record = json.loads(out_path.read_text())
    assert record["method"]["flips"] == decision_rule.FLIPS == 10_000 and record["method"]["alpha"] == 0.05
    assert record["method"]["rng"].startswith("numpy PCG64(20261109)")
    again = io.StringIO()
    assert main(args, out=again) == 0 and "unchanged" in again.getvalue()  # the same inputs, the same p-value
    with pytest.raises(DecisionError, match="never written over"):
        write_decision(out_path, {**record, "result": {}})
    err = io.StringIO()
    assert main([*args[:-2], "--out", str(tmp_path / "data" / "eval" / "x.json")], out=io.StringIO(), err=err) == 1
    assert "data/eval" in err.getvalue()


def test_no_effect_shows_no_improvement(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    by_seed = gains(rng, ranks=0.0, noise=1.0)  # every later checkpoint only moves ranks at random
    result = analyse(write_records(tmp_path, by_seed), flips=999)["result"]
    assert result["a1_beats_a0"] is False and result["conclusion"] == "no improvement shown"
    assert result["p_value"] > 0.05 and result["statistic"] >= 0  # selection still finds a "gain"


def test_a_seed_that_keeps_the_base_model_blocks_the_decision(tmp_path: Path) -> None:
    by_seed = gains(np.random.default_rng(4), ranks=3.0, noise=0.3)
    worse = by_seed[2026].copy()
    worse[:, 1:] = np.minimum(worse[:, 1:], worse[:, :1]) * 0.5  # every later checkpoint is worse than step 0
    by_seed[2026] = worse
    result = analyse(write_records(tmp_path, by_seed), flips=999)["result"]
    assert result["seeds"]["2026"]["selected_step"] == 0 and not result["every_seed_selected_after_step_0"]
    assert result["p_value"] <= 0.05 and result["a1_beats_a0"] is False  # significant, but not consistent
    assert "2026" in result["reason"] and "base model" in result["reason"]


def test_the_null_reapplies_the_fixed_rule_with_its_stop(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    by_seed = gains(rng, ranks=1.0, noise=0.8)
    records = [json.loads(p.read_text()) for p in write_records(tmp_path, by_seed)]
    runs, pair_ids = seed_runs(records)
    assert pair_ids == sorted(f"v{q:03d}" for q in range(Q))
    for run, record in zip(runs, sorted(records, key=lambda r: r["seed"]), strict=True):
        base = run.value_at(0)
        identity = decision_rule._flipped_gains(run, np.ones((1, Q)), base)[0]  # the unflipped data
        assert identity == pytest.approx(run.value_at(run.selected_step) - base, abs=1e-12)
        mirrored = decision_rule._flipped_gains(run, -np.ones((1, Q)), base)[0]
        curve = base - (run.per_question - run.per_question[:, :1]).mean(axis=0)
        best, chosen, misses = -1.0, 0, 0  # the rule written out independently: candidates every 5 steps,
        for step, value in zip(run.steps, curve, strict=True):  # 12-decimal ties to the earlier step, patience 3
            if step % 5:
                continue
            if round(float(value), 12) > round(best, 12):
                best, chosen, misses = float(value), step, 0
            else:
                misses += 1
                if misses == 3:
                    break
        assert mirrored == pytest.approx(curve[run.steps.index(chosen)] - base, abs=1e-12)
        assert record["selected"]["step"] == run.selected_step
    # The decision depends only on the records and the fixed seeds.
    assert decide(runs, flips=499) == decide(runs, flips=499)


def _break(path: Path, edit: Any) -> None:
    record = json.loads(path.read_text())
    edit(record)
    path.write_text(json.dumps(record))


@pytest.mark.parametrize(("change", "message"), [
    ("step0", "step 0 differs between the seeds"),
    ("environment", "different hardware or software"),
    ("no_environment", "holds no environment"),
    ("validation", "differ in validation"),
    ("seeds", "different seed lists|not each of"),
    ("per_question", "no per-question metrics"),
    ("incomplete", "every planned evaluation step"),
])
def test_unsound_records_are_refused_without_a_decision(tmp_path: Path, change: str, message: str) -> None:
    rng = np.random.default_rng(6)
    by_seed = gains(rng, ranks=2.0, noise=0.5)
    if change == "step0":
        shifted = by_seed[1337].copy()
        shifted[0, 0] = 1.0 if shifted[0, 0] != 1.0 else 0.5  # one question's base rank differs in one seed
        by_seed[1337] = shifted
    environment = {**ENVIRONMENT, "gpu_name": "Tesla P100-PCIE-16GB"} if change == "environment" else None
    paths = write_records(tmp_path, by_seed)
    if environment is not None:
        paths[1:2] = write_records(tmp_path / "other", {1337: by_seed[1337]}, environment=environment)
    if change == "no_environment":
        _break(paths[2], lambda r: r.update(environment=None))
    if change == "validation":
        paths[1:2] = write_records(tmp_path / "other", {1337: by_seed[1337]},
                                   validation={**VALIDATION, "manifest_sha256": "b" * 64})
    if change == "seeds":
        paths = paths[:2]
    if change == "per_question":
        _break(paths[0], lambda r: r["evaluations"][1]["metrics"].pop("per_question"))
    if change == "incomplete":
        _break(paths[0], lambda r: (r["after_stop_evaluations"] or r["evaluations"]).pop())
    with pytest.raises(DecisionError, match=message):
        analyse(paths, flips=99)


def test_the_decision_reads_only_the_selection_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_records(tmp_path, gains(np.random.default_rng(7), ranks=2.0, noise=0.5))
    with pytest.raises(DecisionError, match="data/eval"):
        analyse([tmp_path / "data" / "eval" / "selection.json"])
    opened: list[str] = []
    real_open = io.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, str | os.PathLike):
            opened.append(os.path.realpath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    analyse(paths, flips=99)
    assert set(opened) <= {os.path.realpath(p) for p in paths}
    code = ("import sys, finretriever.training.decision_rule, finretriever.training.decision_simulation; "
            "print('torch' in sys.modules)")
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONPATH": str(Path(decision_rule.__file__).parents[2])})
    assert found.stdout.strip() == "False"  # the analysis needs no deep-learning stack


def test_the_simulation_runs_the_implemented_rule_and_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decision_simulation, "CALIBRATION_DRAWS", 40_000)
    null = decision_simulation.Scenario("null: base")
    first = decision_simulation.run_scenario(null, experiments=6, flips=99, seed=11)
    assert first == decision_simulation.run_scenario(null, experiments=6, flips=99, seed=11)
    assert 0 <= first["rejection_rate"] <= first["p_at_most_alpha"] <= 1
    assert max(abs(g) for g in first["true_gain_by_step"].values()) < 0.003  # calibrated: no effect on average
    assert first["mean_statistic"] > 0 and first["selection_optimism"] > 0  # selection alone finds a gain
    plateau = decision_simulation.run_scenario(
        decision_simulation.Scenario("plateau", effect="plateau", peak_gain=0.06), experiments=4, flips=99, seed=12)
    assert plateau["true_gain_by_step"]["55"] == pytest.approx(0.06, abs=0.006)
    model = decision_simulation.calibrate(null, np.random.default_rng(0))
    runs = decision_simulation.simulate_runs(model, np.random.default_rng(1))
    assert [r.seed for r in runs] == list(SEEDS) and all(r.steps == tuple(STEPS) for r in runs)
    assert len({tuple(r.per_question[:, 0]) for r in runs}) == 1  # step 0 is the same model for every seed
