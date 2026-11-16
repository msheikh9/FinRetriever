#!/usr/bin/env python
"""Simulate the A0-versus-A1 decision rule: false-positive rate under no effect and power under effects.

See ``finretriever.training.decision_simulation`` (the model, its assumptions and limits). Every simulated experiment
goes through the implemented rule (``decision_rule.decide`` with the fixed selection rule). Reads no data; writes the
results to ``--out`` (JSON) and prints a table. Fixed seeds: the same arguments give the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running directly from the repo root without ``pip install -e .``.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.training.decision_simulation import SCENARIOS, simulate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simulate_decision_rule.py", description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Results file (JSON).")
    parser.add_argument("--null-experiments", type=int, default=2000)
    parser.add_argument("--effect-experiments", type=int, default=1000)
    parser.add_argument("--flips", type=int, default=999, help="Sign flips per simulated test (the rule uses 10,000).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scenario", action="append", default=[], help="Run only these scenarios (by name).")
    args = parser.parse_args(argv)
    chosen = [s for s in SCENARIOS if not args.scenario or s.name in args.scenario]
    started = time.time()
    results = simulate(chosen, null_experiments=args.null_experiments, effect_experiments=args.effect_experiments,
                       flips=args.flips, workers=args.workers)
    Path(args.out).write_text(json.dumps({"results": results, "seconds": round(time.time() - started)}, indent=1))
    print(f"{'scenario':32s} {'decided A1':>10s} {'p<=0.05':>8s} {'SE':>6s} {'T':>8s} {'optimism':>9s} {'adjusted':>9s}")
    for r in results:
        print(f"{r['scenario']['name']:32s} {r['rejection_rate']:10.3f} {r['p_at_most_alpha']:8.3f} "
              f"{r['monte_carlo_se']:6.3f} {r['mean_statistic']:+8.4f} {r['selection_optimism']:+9.4f} "
              f"{r['mean_selection_adjusted']:+9.4f}")
    print(f"{round(time.time() - started)} s; written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
