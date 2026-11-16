"""The A0-versus-A1 decision rule: a selection-aware sign-flip test over the seeds' selection records.

Approved on 2026-11-09 (docs/training_b1_b3_proposal.md §1.5, K1). A0 is the zero-shot base model, which is step 0 of
every run. A1 is the procedure "fine-tune, then select a checkpoint by the fixed rule" (``checkpoint_selection``).
Both are compared on the 93 synthetic validation questions that also choose A1's checkpoint, so the test re-applies
the selection to every resampled dataset instead of treating the selected checkpoint as fixed.

**Inputs.** Only the seeds' ``selection.json`` records (``read_selection_record``: kind, format, rule, hashes,
completeness and the re-derived selection are checked on reading). No development or evaluation file is read; a path
under ``data/eval`` is refused.

**Preconditions** (``seed_runs``). Any failure stops the analysis: no decision is made.

1. The records name the same seed list, and each of its seeds (at least two) appears once, with the same config
   SHA-256 and validation set.
2. Each record holds every planned evaluation (the rule's evaluations, then the after-stop ones).
3. Every evaluation scores the same questions, and its per-question whole-corpus MRR@10 values average to its value.
4. Step 0 is identical across the seeds: every question's first relevant rank and MRR@10 (readiness check N10).
5. The seeds ran on the same hardware and software (``checkpoint_selection.ENVIRONMENT_IDENTITY``).

**Statistic.** T = (1/S) Σ_s [V_s(k*_s) − V(0)]: whole-corpus MRR@10 at the step the rule selected for seed s, minus
the shared step-0 value, averaged over the S seeds.

**Null distribution.** ``FLIPS`` sign vectors over the questions (sorted by pair ID) from
``numpy.random.Generator(PCG64(RNG_SEED))``. Each flip reverses all of a question's differences from step 0 at once,
across every evaluation and every seed. The fixed rule is re-applied to each flipped curve over every recorded
evaluation (``checkpoint_selection.select_through``: candidates, 12-decimal ties to the earlier step, patience 3),
and T*_b is computed as T is. One-sided p = (1 + #{T*_b ≥ T − TOLERANCE}) / (FLIPS + 1).

**Decision.** "A1 beats A0 on the validation set" if and only if p ≤ ``ALPHA`` and every seed selected a checkpoint
after step 0. Otherwise "no improvement shown". The rule never concludes that A0 beats A1.

**Reported, never deciding:**
- T, which is optimistic by construction;
- the null mean E*[T*] and T − E*[T*], a selection-adjusted estimate;
- per seed: the selected step and value, the rule's stop, the last step;
- the seed-minimum statistic and its flip p-value;
- question-bootstrap 95% intervals (``BOOTSTRAP`` resamples, ``BOOTSTRAP_SEED``) for T and for the last-step
  difference, both conditional on these runs and not selection-adjusted;
- company-mode and other metrics at the selected checkpoints.

**Assumptions and limits** (simulated in ``decision_simulation``):
- The sign flip is exact when each question's differences are symmetric under no effect. For mean-zero but
  asymmetric changes it holds the level only approximately: 3.4–5.4% in the approved simulation.
- The inference covers the sampling of these synthetic, AI-reviewed questions, given these runs. Three seeds are a
  consistency check, not an estimate of seed variance.
- A pass means "better on this synthetic validation distribution", not a retrieval improvement on real questions.

The record (``write_decision``) is written once and holds its inputs' SHA-256, so the same records always give the
same p-value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from finretriever.training.checkpoint_selection import (
    ENVIRONMENT_IDENTITY,
    METRIC,
    MODE,
    RULE,
    SelectionError,
    read_selection_record,
    select_through,
)

KIND = "finretriever_a1_vs_a0_decision"
FORMAT_VERSION = 1
FLIPS = 10_000
RNG_SEED = 20261109
ALPHA = 0.05
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261110
TOLERANCE = 1e-12
PREFIX = "[07_decide_a1_vs_a0]"
METHOD = {
    "statistic": "mean over seeds of (whole-corpus MRR@10 at the selected checkpoint - at step 0)",
    "null": "sign flips of each question's differences from step 0, jointly across every evaluation and seed, with the "
    "fixed selection rule re-applied to every flipped curve over all recorded evaluations",
    "flips": FLIPS,
    "rng": f"numpy PCG64({RNG_SEED}); columns in sorted pair-ID order",
    "p_value": "one-sided, (1 + #{T* >= T - 1e-12}) / (flips + 1)",
    "alpha": ALPHA,
    "consistency": "every seed selects a checkpoint after step 0",
    "decision": "A1 beats A0 on the validation set if and only if p <= alpha and the consistency condition holds",
    "bootstrap": {"resamples": BOOTSTRAP, "rng": f"numpy PCG64({BOOTSTRAP_SEED})", "interval": "percentile 95%"},
    "selection_rule": RULE,
}
NOTICE = (
    "A selection-aware comparison on synthetic validation questions accepted by an AI review (not human-validated, not "
    "gold labels), the same questions that chose each run's checkpoint. It covers that question distribution, given "
    "these runs. It is not a retrieval result on real questions or the benchmark, and no development or evaluation "
    "data was read."
)


class DecisionError(ValueError):
    """The records cannot support a decision; nothing is concluded."""


@dataclass(frozen=True)
class SeedRun:
    """One seed's evaluations, as the rule and the test read them."""

    seed: int
    steps: tuple[int, ...]  # every evaluated step, in order (the rule's, then after its stop)
    values: np.ndarray  # whole-corpus MRR@10 at each step, as recorded
    per_question: np.ndarray  # questions x steps: per-question whole-corpus MRR@10 (rows in sorted pair-ID order)
    selected_step: int
    rule_stop_step: int | None

    def value_at(self, step: int) -> float:
        return float(self.values[self.steps.index(step)])


def _under_evaluation_data(path: str | Path) -> bool:
    parts = Path(path).resolve().parts
    return any(parts[i : i + 2] == ("data", "eval") for i in range(len(parts) - 1))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(paths: Sequence[str | Path]) -> list[tuple[Path, str, dict[str, Any]]]:
    """Each selection record, read and verified, with its path and SHA-256."""
    out = []
    for raw in paths:
        path = Path(raw)
        if _under_evaluation_data(path):
            raise DecisionError(f"{path} is under data/eval: the decision reads selection records only")
        try:
            record = read_selection_record(path)
        except SelectionError as exc:
            raise DecisionError(str(exc)) from None
        out.append((path, _sha256(path), record))
    return out


def seed_runs(records: Sequence[Mapping[str, Any]]) -> tuple[list[SeedRun], list[str]]:
    """The preconditions (module docstring), then each seed's evaluations; raises ``DecisionError`` with every
    problem found. Returns the runs in seed order and the questions' pair IDs (the rows of each ``per_question``)."""
    problems: list[str] = []
    if len(records) < 2:
        raise DecisionError(f"{len(records)} selection record(s): the rule compares at least two seeds")
    seed_lists = {tuple(r["seeds"]) for r in records}
    seeds = sorted(r["seed"] for r in records)
    if len(seed_lists) != 1:
        problems.append(f"the records name different seed lists {sorted(seed_lists)}")
    elif sorted(next(iter(seed_lists))) != seeds:
        problems.append(f"the records are seeds {seeds}, not each of {list(next(iter(seed_lists)))} once")
    for key in ("config_sha256", "validation", "rule"):
        if len({json.dumps(r[key], sort_keys=True) for r in records}) != 1:
            problems.append(f"the records differ in {key}")
    environments = []
    for r in records:
        env = r.get("environment")
        if not env:
            problems.append(f"seed {r['seed']}: the record holds no environment")
        else:
            environments.append({k: env.get(k) for k in ENVIRONMENT_IDENTITY})
    if len({json.dumps(e, sort_keys=True) for e in environments}) > 1:
        problems.append(f"the seeds ran on different hardware or software: {environments}")
    pair_ids: list[str] | None = None
    runs: list[SeedRun] = []
    step0: dict[int, list[tuple[Any, float]]] = {}
    for r in sorted(records, key=lambda r: r["seed"]):
        evaluations = [*r["evaluations"], *r["after_stop_evaluations"]]
        rows: list[list[float]] = []
        for e in evaluations:
            per = e["metrics"].get("per_question")
            if not per:
                problems.append(f"seed {r['seed']} step {e['step']}: no per-question metrics")
                continue
            by_id = {q["pair_id"]: q for q in per}
            ids = sorted(by_id)
            if len(ids) != len(per):
                problems.append(f"seed {r['seed']} step {e['step']}: a question is scored twice")
            if pair_ids is None:
                pair_ids = ids
            if ids != pair_ids:
                problems.append(f"seed {r['seed']} step {e['step']}: scores other questions than the first record")
                continue
            column = [float(by_id[q][MODE][METRIC]) for q in ids]
            if abs(sum(column) / len(column) - float(e["value"])) > TOLERANCE:
                problems.append(f"seed {r['seed']} step {e['step']}: the per-question values do not average to "
                                f"{e['value']}")
            if e["step"] == 0:
                step0[r["seed"]] = [(by_id[q][MODE].get("first_relevant_rank"), float(by_id[q][MODE][METRIC]))
                                    for q in ids]
            rows.append(column)
        if problems:
            continue
        runs.append(SeedRun(
            seed=int(r["seed"]),
            steps=tuple(int(e["step"]) for e in evaluations),
            values=np.array([float(e["value"]) for e in evaluations]),
            per_question=np.array(rows).T,
            selected_step=int(r["selected"]["step"]),
            rule_stop_step=None if r["rule_stop"] is None else int(r["rule_stop"]["step"]),
        ))
    if len({json.dumps(v) for v in step0.values()}) > 1:
        problems.append("step 0 differs between the seeds (first relevant ranks or MRR@10): the base model was not "
                        "evaluated identically, so A0 is not one model (N10)")
    if pair_ids is not None and len(pair_ids) < 2:
        problems.append(f"{len(pair_ids)} validation question(s): a sign-flip test needs at least two")
    if problems:
        raise DecisionError(f"{len(problems)} precondition(s) failed, no decision: " + "; ".join(problems))
    assert pair_ids is not None
    return runs, pair_ids


def _flipped_gains(run: SeedRun, signs: np.ndarray, base: float) -> np.ndarray:
    """Each flip's selected gain over step 0 for one seed: the fixed rule re-applied to every flipped curve."""
    questions = run.per_question.shape[0]
    differences = run.per_question - run.per_question[:, :1]
    curves = base + signs @ differences / questions  # flips x steps
    gains = np.empty(len(signs))
    for b, curve in enumerate(curves):
        chosen, _ = select_through([{"step": s, "value": float(v)} for s, v in zip(run.steps, curve, strict=True)])
        gains[b] = curve[run.steps.index(chosen)] - base
    return gains


def _interval(values: np.ndarray, resamples: int, seed: int) -> list[float] | None:
    if resamples <= 0:
        return None
    rng = np.random.Generator(np.random.PCG64(seed))
    index = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[index].mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 6), round(float(np.quantile(means, 0.975)), 6)]


def decide(runs: Sequence[SeedRun], *, flips: int = FLIPS, rng_seed: int = RNG_SEED, alpha: float = ALPHA,
           bootstrap: int = BOOTSTRAP, bootstrap_seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """The test and the decision on verified runs (``seed_runs``); see the module docstring."""
    base = runs[0].value_at(0)
    questions = runs[0].per_question.shape[0]
    gains = np.array([run.value_at(run.selected_step) - base for run in runs])
    observed, observed_min = float(gains.mean()), float(gains.min())
    signs = (np.random.Generator(np.random.PCG64(rng_seed)).integers(0, 2, size=(flips, questions)) * 2 - 1).astype(
        float)
    null = np.stack([_flipped_gains(run, signs, base) for run in runs])  # seeds x flips
    null_mean, null_min = null.mean(axis=0), null.min(axis=0)
    p = (1 + int(np.sum(null_mean >= observed - TOLERANCE))) / (flips + 1)
    p_min = (1 + int(np.sum(null_min >= observed_min - TOLERANCE))) / (flips + 1)
    after_step0 = all(run.selected_step > 0 for run in runs)
    accepted = p <= alpha and after_step0
    selected_q = np.mean([run.per_question[:, run.steps.index(run.selected_step)] - run.per_question[:, 0]
                          for run in runs], axis=0)
    last_q = np.mean([run.per_question[:, -1] - run.per_question[:, 0] for run in runs], axis=0)
    if accepted:
        reason = f"p = {p:.4f} <= {alpha} and every seed selected a checkpoint after step 0"
    elif not after_step0:
        reason = (f"seed(s) {[r.seed for r in runs if r.selected_step == 0]} selected the base model (step 0)"
                  + (f"; p = {p:.4f}" if p <= alpha else f" and p = {p:.4f} > {alpha}"))
    else:
        reason = f"p = {p:.4f} > {alpha}"
    return {
        "a1_beats_a0": accepted,
        "conclusion": "A1 beats A0 on the validation set" if accepted else "no improvement shown",
        "reason": reason,
        "p_value": round(p, 6),
        "statistic": round(observed, 6),
        "null_mean": round(float(null_mean.mean()), 6),
        "selection_adjusted_estimate": round(observed - float(null_mean.mean()), 6),
        "every_seed_selected_after_step_0": after_step0,
        "seed_minimum": {"statistic": round(observed_min, 6), "p_value": round(p_min, 6)},
        "step_0_value": round(base, 6),
        "questions": questions,
        "seeds": {str(run.seed): {
            "selected_step": run.selected_step,
            "selected_value": round(run.value_at(run.selected_step), 6),
            "gain_over_step_0": round(run.value_at(run.selected_step) - base, 6),
            "rule_stop_step": run.rule_stop_step,
            "rule_stop_value": None if run.rule_stop_step is None else round(run.value_at(run.rule_stop_step), 6),
            "last_step": run.steps[-1],
            "last_value": round(float(run.values[-1]), 6),
            "evaluated_steps": list(run.steps),
        } for run in runs},
        "intervals": {
            "selected_minus_step_0": _interval(selected_q, bootstrap, bootstrap_seed),
            "last_step_minus_step_0": _interval(last_q, bootstrap, bootstrap_seed + 1),
            "note": "question bootstrap, conditional on these runs; not selection-adjusted",
        },
        "flips": flips,
        "alpha": alpha,
    }


def _metrics_at(records: Sequence[Mapping[str, Any]], step_of: Mapping[int, int]) -> dict[str, dict[str, float]]:
    """Mean over seeds of every aggregate metric at each seed's given step, in both modes."""
    found: dict[str, dict[str, list[float]]] = {}
    for r in records:
        entry = next(e for e in [*r["evaluations"], *r["after_stop_evaluations"]] if e["step"] == step_of[r["seed"]])
        for mode in ("global", "company"):
            for name, value in (entry["metrics"].get(mode) or {}).items():
                if isinstance(value, int | float) and name != "first_relevant_rank":
                    found.setdefault(mode, {}).setdefault(name, []).append(float(value))
    return {mode: {name: round(sum(v) / len(v), 6) for name, v in names.items()} for mode, names in found.items()}


def analyse(paths: Sequence[str | Path], **settings: Any) -> dict[str, Any]:
    """The decision record for the given selection records (see the module docstring)."""
    loaded = load_records(paths)
    records = [record for _, _, record in loaded]
    runs, pair_ids = seed_runs(records)
    result = decide(runs, **settings)
    return {
        "kind": KIND,
        "format_version": FORMAT_VERSION,
        "notice": NOTICE,
        "method": {**METHOD, **({"flips": settings["flips"]} if "flips" in settings else {})},
        "inputs": [{"path": str(path), "sha256": digest, "seed": record["seed"], "run": record["run"]}
                   for path, digest, record in sorted(loaded, key=lambda x: x[2]["seed"])],
        "validation": records[0]["validation"],
        "environment": records[0]["environment"],
        "questions_sha256": hashlib.sha256("\n".join(pair_ids).encode()).hexdigest(),
        "result": result,
        "reported_metrics": {
            "step_0": _metrics_at(records, {r["seed"]: 0 for r in records}),
            "selected": _metrics_at(records, {r["seed"]: r["selected"]["step"] for r in records}),
            "note": "means over seeds; reported, never deciding. Development diagnostics are separate (the benchmark).",
        },
    }


def write_decision(path: str | Path, decision: Mapping[str, Any]) -> bool:
    """Write the decision record once; ``False`` if an identical one is there, an error if a different one is."""
    path = Path(path)
    if _under_evaluation_data(path):
        raise DecisionError(f"{path} is under data/eval; the decision is never written there")
    data = (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() == data:
            return False
        raise DecisionError(f"{path} already records a different decision; it is never written over")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="07_decide_a1_vs_a0.py",
        description="The selection-aware A0-versus-A1 decision from the seeds' selection records (reads nothing else).",
    )
    parser.add_argument("--selection", action="append", required=True, metavar="SELECTION_JSON",
                        help="One seed's selection.json (repeat for every seed).")
    parser.add_argument("--out", required=True, help="The decision record to write once (JSON).")
    return parser


def main(argv: list[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        decision = analyse(args.selection)
        written = write_decision(args.out, decision)
    except DecisionError as exc:
        print(f"{PREFIX} refused: {exc}", file=err)
        return 1
    result = decision["result"]
    print(f"{PREFIX} {'wrote' if written else 'unchanged (identical record present)'} {args.out}", file=out)
    print(f"{PREFIX} {result['conclusion']}: {result['reason']}", file=out)
    print(f"{PREFIX} selected minus step 0 (mean over {len(result['seeds'])} seeds) {result['statistic']:+.4f}; "
          f"selection-adjusted {result['selection_adjusted_estimate']:+.4f}; flip p = {result['p_value']:.4f}", file=out)
    print(f"{PREFIX} {NOTICE}", file=out)
    return 0


__all__ = [
    "ALPHA",
    "BOOTSTRAP",
    "FLIPS",
    "KIND",
    "METHOD",
    "NOTICE",
    "RNG_SEED",
    "DecisionError",
    "SeedRun",
    "analyse",
    "decide",
    "load_records",
    "main",
    "seed_runs",
    "write_decision",
]
