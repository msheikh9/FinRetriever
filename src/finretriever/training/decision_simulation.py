"""Simulations of the decision rule (``decision_rule``) on synthetic validation results: false-positive rate under no
effect, power under effects, and the optimism selection adds.

Reads no data: no validation, development or evaluation file. Every simulated experiment goes through the
implemented rule: ``checkpoint_selection.select_through`` selects each seed's checkpoint over every planned
evaluation (as the trainer now records them), and ``decision_rule.decide`` runs the sign-flip test and the decision.

**The model** (``Scenario``; the assumptions of docs/training_b1_b3_proposal.md §1.3):
- **Questions.** 93 questions, each with a latent log-rank u at step 0, shared by all seeds. Its per-question
  MRR@10 value is 1/floor(e^u) within the top 10, else 0. That keeps the metric's discreteness, ceiling and floor.
  The mean of u is calibrated to the base MRR@10, and its SD is 1.3.
- **Training paths.** Training moves u along a Brownian path in training time t/T, so nearby checkpoints are strongly
  correlated (or, with ``independent``, uncorrelated at every evaluation). A fraction ``shared`` of the path is common
  to the seeds. Each question's susceptibility g is lognormal (log-SD 0.5, mean 1).
- **No effect** (``null``):
  - "zero": a drift is calibrated so the expected MRR@10 equals step 0's at every step. Changes are mean-zero,
    heterogeneous and asymmetric per question.
  - "noise": no drift. Training noise then hurts on average, a conservative null.
- **Effects:**
  - "plateau": the gain is reached at 40% of training and kept;
  - "overfit": the gain peaks at 40% and halves by the end.

  Each is sized by its population MRR@10 gain at the peak (``peak_gain``).
- **The latent noise scale** is calibrated so the per-question SD of (last step − step 0) is ``question_sd``. The
  default 0.2 is the plan's ±0.04 paired-interval estimate for 93 questions.

**Limitations.** This is a model of how per-question ranks move, not of the real runs: the real correlations
across checkpoints, seeds and questions are unknown. The rule's computation does not depend on the model; its
*estimated* error rates do. The scenarios therefore bracket the unknowns (base MRR@10 0.4–0.8, per-question SD
0.1–0.3, seed sharing 0–0.9, correlated or independent checkpoints, 55 or 64 steps).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from finretriever.training.checkpoint_selection import evaluation_steps, select_through
from finretriever.training.decision_rule import ALPHA, SeedRun, decide

QUESTIONS = 93
SEEDS = (42, 1337, 2026)
BASE_SD = 1.3
SUSCEPTIBILITY_SDLOG = 0.5
CALIBRATION_DRAWS = 300_000
PEAK = 0.4  # fraction of training at which an effect peaks


@dataclass(frozen=True)
class Scenario:
    name: str
    base_mrr: float = 0.6
    question_sd: float = 0.2
    shared: float = 0.5
    steps: int = 55
    null: str = "zero"  # "zero" or "noise"
    effect: str | None = None  # None, "plateau" or "overfit"
    peak_gain: float = 0.0
    independent: bool = False


SCENARIOS = (
    Scenario("null: base"),
    Scenario("null: pure noise", null="noise"),
    Scenario("null: per-question SD 0.1", question_sd=0.1),
    Scenario("null: per-question SD 0.3", question_sd=0.3),
    Scenario("null: base MRR@10 0.4", base_mrr=0.4),
    Scenario("null: base MRR@10 0.8", base_mrr=0.8),
    Scenario("null: seeds independent", shared=0.0),
    Scenario("null: seeds 90% shared", shared=0.9),
    Scenario("null: checkpoints independent", independent=True),
    Scenario("null: 64 steps", steps=64),
    Scenario("plateau +0.02", effect="plateau", peak_gain=0.02),
    Scenario("plateau +0.04", effect="plateau", peak_gain=0.04),
    Scenario("plateau +0.06", effect="plateau", peak_gain=0.06),
    Scenario("overfit +0.04", effect="overfit", peak_gain=0.04),
    Scenario("overfit +0.06", effect="overfit", peak_gain=0.06),
)


def reciprocal_rank(u: np.ndarray) -> np.ndarray:
    rank = np.floor(np.exp(np.clip(u, 0.0, None)))
    return np.where(rank <= 10, 1.0 / np.maximum(rank, 1.0), 0.0)


def _shape(kind: str | None, tau: np.ndarray) -> np.ndarray:
    if kind == "plateau":
        return np.minimum(1.0, tau / PEAK)
    if kind == "overfit":
        return np.where(tau <= PEAK, tau / PEAK, 1.0 - 0.5 * (tau - PEAK) / (1.0 - PEAK))
    return np.zeros_like(tau)


def _bisect(f: Any, lo: float, hi: float, target: float, iterations: int = 60) -> float:
    """The x in [lo, hi] where the increasing function f reaches target."""
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


@dataclass(frozen=True)
class Model:
    """A scenario's calibrated latent parameters."""

    scenario: Scenario
    steps: tuple[int, ...]
    tau: np.ndarray
    mean_u: float
    sigma: float
    drift: np.ndarray  # per step: the no-effect drift plus the effect
    true_gain: np.ndarray  # per step: the population MRR@10 gain over step 0


def calibrate(scenario: Scenario, rng: np.random.Generator) -> Model:
    u0 = rng.normal(0.0, BASE_SD, CALIBRATION_DRAWS)
    g = rng.lognormal(-SUSCEPTIBILITY_SDLOG**2 / 2, SUSCEPTIBILITY_SDLOG, CALIBRATION_DRAWS)
    z = rng.normal(size=CALIBRATION_DRAWS)
    mean_u = _bisect(lambda m: -reciprocal_rank(u0 + m).mean(), -5.0, 5.0, -scenario.base_mrr)
    u0 = u0 + mean_u
    base = reciprocal_rank(u0)

    def gain(sigma: float, tau: float, shift: float) -> float:
        return float((reciprocal_rank(u0 - g * shift + sigma * np.sqrt(tau) * z) - base).mean())

    def compensation(sigma: float, tau: float) -> float:  # the drift that keeps the expected MRR@10 at step 0's
        if tau == 0 or scenario.null == "noise":
            return 0.0
        return _bisect(lambda c: gain(sigma, tau, c), -10.0, 10.0, 0.0)

    def question_sd(sigma: float) -> float:
        return float((reciprocal_rank(u0 - g * compensation(sigma, 1.0) + sigma * z) - base).std())

    sigma = _bisect(question_sd, 0.01, 6.0, scenario.question_sd, iterations=40)
    steps = tuple(evaluation_steps(scenario.steps))
    tau = np.array(steps) / scenario.steps
    drift = np.array([compensation(sigma, t) for t in tau])
    if scenario.effect:
        at_peak = compensation(sigma, PEAK)
        size = _bisect(lambda d: gain(sigma, PEAK, at_peak + d), 0.0, 10.0, scenario.peak_gain, iterations=50)
        drift = drift + size * _shape(scenario.effect, tau)
    true_gain = np.array([gain(sigma, t, d) for t, d in zip(tau, drift, strict=True)])
    return Model(scenario, steps, tau, mean_u, sigma, drift, true_gain)


def simulate_runs(model: Model, rng: np.random.Generator) -> list[SeedRun]:
    """One synthetic experiment: each seed's per-question MRR@10 at every planned evaluation, and the rule's
    selection over them."""
    sc, k = model.scenario, len(model.steps)
    u0 = rng.normal(model.mean_u, BASE_SD, QUESTIONS)
    g = rng.lognormal(-SUSCEPTIBILITY_SDLOG**2 / 2, SUSCEPTIBILITY_SDLOG, QUESTIONS)
    if sc.independent:
        shared = rng.normal(size=(QUESTIONS, k)) * np.sqrt(model.tau)
        own = rng.normal(size=(len(SEEDS), QUESTIONS, k)) * np.sqrt(model.tau)
    else:
        dt = np.diff(model.tau, prepend=0.0)
        shared = np.cumsum(rng.normal(size=(QUESTIONS, k)) * np.sqrt(dt), axis=1)
        own = np.cumsum(rng.normal(size=(len(SEEDS), QUESTIONS, k)) * np.sqrt(dt), axis=2)
    u = (u0[None, :, None] - g[None, :, None] * model.drift[None, None, :]
         + model.sigma * (np.sqrt(sc.shared) * shared[None] + np.sqrt(1 - sc.shared) * own))
    rr = reciprocal_rank(u)
    rr[:, :, 0] = reciprocal_rank(u0)[None, :]  # step 0 is the base model, the same for every seed
    runs = []
    for s, seed in enumerate(SEEDS):
        values = rr[s].mean(axis=0)
        selected, stop = select_through([{"step": st, "value": float(v)} for st, v in zip(model.steps, values,
                                                                                          strict=True)])
        runs.append(SeedRun(seed, model.steps, values, rr[s], selected, stop))
    return runs


def run_scenario(scenario: Scenario, *, experiments: int, flips: int, seed: int) -> dict[str, Any]:
    """Rates and means over ``experiments`` simulated experiments of one scenario."""
    rng = np.random.default_rng(seed)
    model = calibrate(scenario, rng)
    gain_at = dict(zip(model.steps, model.true_gain, strict=True))
    rows = []
    for _ in range(experiments):
        runs = simulate_runs(model, rng)
        result = decide(runs, flips=flips, rng_seed=int(rng.integers(2**31)), bootstrap=0)
        rows.append((result["a1_beats_a0"], result["p_value"] <= ALPHA, result["every_seed_selected_after_step_0"],
                     result["statistic"], result["null_mean"], float(np.mean([gain_at[r.selected_step] for r in runs])),
                     float(np.mean([r.values[-1] - r.values[0] for r in runs])),
                     float(np.mean([r.rule_stop_step is not None for r in runs]))))
    a = np.array(rows, dtype=float)
    return {
        "scenario": asdict(scenario),
        "experiments": experiments,
        "flips": flips,
        "rejection_rate": round(float(a[:, 0].mean()), 4),
        "p_at_most_alpha": round(float(a[:, 1].mean()), 4),
        "every_seed_after_step_0": round(float(a[:, 2].mean()), 4),
        "mean_statistic": round(float(a[:, 3].mean()), 4),
        "mean_null_mean": round(float(a[:, 4].mean()), 4),
        "mean_selection_adjusted": round(float((a[:, 3] - a[:, 4]).mean()), 4),
        "mean_true_gain_at_selected": round(float(a[:, 5].mean()), 4),
        "selection_optimism": round(float((a[:, 3] - a[:, 5]).mean()), 4),
        "mean_last_minus_step_0": round(float(a[:, 6].mean()), 4),
        "rule_stopped_early_share": round(float(a[:, 7].mean()), 4),
        "monte_carlo_se": round(float(np.sqrt(max(a[:, 0].mean() * (1 - a[:, 0].mean()), 1e-12) / experiments)), 4),
        "true_gain_by_step": {str(s): round(float(x), 4) for s, x in zip(model.steps, model.true_gain, strict=True)},
    }


def simulate(scenarios: Sequence[Scenario] = SCENARIOS, *, null_experiments: int = 2000,
             effect_experiments: int = 1000, flips: int = 999, workers: int = 1, seed: int = 20261109) -> list[dict]:
    """Every scenario, each with its own fixed seed (``seed`` + its index), optionally in parallel processes."""
    jobs = [(sc, null_experiments if sc.effect is None else effect_experiments, flips, seed + i)
            for i, sc in enumerate(scenarios)]
    if workers <= 1:
        return [run_scenario(sc, experiments=n, flips=f, seed=s) for sc, n, f, s in jobs]
    from multiprocessing import get_context

    with get_context("spawn").Pool(workers) as pool:
        return pool.starmap(_job, jobs)


def _job(scenario: Scenario, experiments: int, flips: int, seed: int) -> dict[str, Any]:
    return run_scenario(scenario, experiments=experiments, flips=flips, seed=seed)


__all__ = ["SCENARIOS", "Model", "Scenario", "calibrate", "reciprocal_rank", "run_scenario", "simulate",
           "simulate_runs"]
