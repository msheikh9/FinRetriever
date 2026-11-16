#!/usr/bin/env python
"""Step 6: benchmark every retriever on the gold set.

Scores BM25, both zero-shot dense baselines, and the fine-tuned FinRetriever
checkpoint on the gold queries. Writes per-system JSON under ``results/`` and
updates ``results/final_ablation_table.md``.

v1 scaffold: this is a stub. It prints what it *would* do and exits 0.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="configs/eval/benchmark.yaml",
        help="Path to the benchmark YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[06_run_benchmark] config={args.config}")
    print("[06_run_benchmark] not implemented yet (v1 scaffold).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
