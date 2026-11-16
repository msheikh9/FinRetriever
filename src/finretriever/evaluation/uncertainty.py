"""Statistical uncertainty of per-question retrieval metrics.

Every metric of a run is a mean over questions, so its uncertainty comes from which questions were
asked. Both functions resample questions, never passages, and are deterministic (a fixed seed):

* :func:`mean_interval`: the mean and a percentile bootstrap interval (questions drawn with
  replacement), with the standard error of the mean;
* :func:`paired_difference`: run A against run B on the same questions: the mean per-question
  difference, a paired bootstrap interval, wins, ties and losses, and a two-sided sign-flip
  (randomization) test of "no difference". With at most ``exact_limit`` questions every sign
  pattern is enumerated (an exact test); above it a fixed number of seeded random flips is used.

With few questions the intervals are wide and a percentile bootstrap understates them, which is the
point of reporting them: a difference whose interval includes zero is not evidence of a change.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261104
DEFAULT_LEVEL = 0.95
EXACT_LIMIT = 16  # 2**16 sign patterns
INTERVAL_METHOD = (
    "percentile bootstrap over questions ({resamples} resamples, seed {seed}); paired runs resample the same "
    "questions together; sign-flip randomization test of no difference ({test})"
)


def _bootstrap_means(values: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(resamples, len(values)))
    return values[picks].mean(axis=1)


def mean_interval(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, Any]:
    """Mean of ``values`` with a percentile bootstrap interval and the standard error."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("no values to summarise")
    means = _bootstrap_means(data, resamples, seed)
    tail = (1 - level) / 2
    return {
        "n": int(data.size),
        "mean": float(data.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1 - tail)),
        "level": level,
        "standard_error": float(data.std(ddof=1) / math.sqrt(data.size)) if data.size > 1 else None,
    }


def _sign_flip_p(diffs: np.ndarray, *, exact_limit: int, resamples: int, seed: int) -> tuple[float, str]:
    observed = abs(diffs.mean())
    if not diffs.any():
        return 1.0, "exact"
    if diffs.size <= exact_limit:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=diffs.size)))
        stats = np.abs((signs * diffs).mean(axis=1))
        return float((stats >= observed - 1e-12).mean()), "exact"
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(resamples, diffs.size))
    stats = np.abs((signs * diffs).mean(axis=1))
    return float(((stats >= observed - 1e-12).sum() + 1) / (resamples + 1)), "monte_carlo"


def paired_difference(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
    exact_limit: int = EXACT_LIMIT,
) -> dict[str, Any]:
    """Run A minus run B per question (both keyed by question ID, the same questions)."""
    if set(a) != set(b):
        raise ValueError(f"paired runs must score the same questions; they differ in {sorted(set(a) ^ set(b))[:5]}")
    keys = sorted(a)
    diffs = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
    interval = mean_interval(diffs, resamples=resamples, seed=seed, level=level)
    p_value, test = _sign_flip_p(diffs, exact_limit=exact_limit, resamples=resamples, seed=seed)
    return {
        "n": len(keys),
        "mean_a": float(np.mean([a[k] for k in keys])),
        "mean_b": float(np.mean([b[k] for k in keys])),
        "mean_difference": interval["mean"],
        "low": interval["low"],
        "high": interval["high"],
        "level": level,
        "wins": int((diffs > 1e-12).sum()),
        "ties": int((np.abs(diffs) <= 1e-12).sum()),
        "losses": int((diffs < -1e-12).sum()),
        "p_value": p_value,
        "test": test,
        "interval_includes_zero": interval["low"] <= 0.0 <= interval["high"],
    }


def describe_method(resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED, exact_limit: int = EXACT_LIMIT) -> str:
    return INTERVAL_METHOD.format(
        resamples=resamples, seed=seed, test=f"exact for up to {exact_limit} questions, else seeded Monte Carlo"
    )


__all__ = [
    "DEFAULT_LEVEL",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "EXACT_LIMIT",
    "describe_method",
    "mean_interval",
    "paired_difference",
]
