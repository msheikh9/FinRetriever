#!/usr/bin/env python
"""Step 4: run the retrieval baselines against an evaluation set.

Two modes:

* ``--eval-set NAME`` scores a named evaluation set from the benchmark config
  (``evaluation_sets``) with every configured retriever (BM25 and the
  zero-shot dense baselines) in every configured retrieval mode (``global``,
  ``company``): corpus and label paths come from the config, the corpus is
  verified against the labels' metadata before scoring or loading a model,
  and every query gets its own record with judged@10 and its unjudged
  passages counted apart from confirmed negatives. Development sets write
  under ``results/dev/`` and are labelled provisional development metrics.
  Dense models need torch and sentence-transformers; passage embeddings are
  cached under ``embedding_cache_dir``. A set may run its own ``retrievers``,
  such as local fine-tuned checkpoints (``dense_checkpoint``), each checked for
  evaluation isolation before it is loaded, with the stored baseline rankings
  scored beside them (``reference_runs_from``). Every run prints 95% bootstrap
  intervals, and a paired comparison when the set names ``paired_reference``.
  Results are written once: an output file holding different results is never
  replaced (exit 2, nothing written).
* Without ``--eval-set`` (the original mode, used by the mini-demo), the
  BM25 baseline runs on the corpus, queries and qrels from the flags — or
  from ``corpus_path``, ``gold_queries_path`` and ``gold_qrels_path`` in the
  config when a flag is omitted — and the script writes
  ``results/baselines.json`` and ``results/final_ablation_table.md`` as before.

Exit codes: 0 every configured retriever ran; 1 results were written but a
dense model could not be loaded (listed as skipped, with the reason); 2
configuration, provenance, isolation or input error, or results already
written, nothing written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running directly from the repo root without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.evaluation.benchmark import (  # noqa: E402
    STATUS_DEV,
    EvalSetResult,
    checkpoint_cautions,
    load_benchmark_inputs,
    load_metrics_from_config,
    load_top_k_from_config,
    run_bm25_baseline,
    run_eval_set,
    write_eval_set_outputs,
    write_results,
)
from finretriever.evaluation.isolation import isolation_summary  # noqa: E402
from finretriever.utils.io import read_yaml  # noqa: E402

# Used when neither a flag nor the config names a path (the original defaults).
_FALLBACK_PATHS = {
    "corpus": ("corpus_path", "data/processed/corpus.jsonl"),
    "queries": ("gold_queries_path", "data/eval/gold_queries.jsonl"),
    "qrels": ("gold_qrels_path", "data/eval/gold_qrels.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--eval-set",
        default=None,
        help=(
            "Score a named evaluation set from the config (evaluation_sets): paths from the "
            "config, corpus provenance verified first, per-query results with judged@10. "
            "Development sets write under results/dev/."
        ),
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to the corpus JSONL (default: corpus_path in the config).",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="Path to the gold queries JSONL (default: gold_queries_path in the config).",
    )
    parser.add_argument(
        "--qrels",
        default=None,
        help="Path to the gold qrels JSONL (default: gold_qrels_path in the config).",
    )
    parser.add_argument(
        "--config",
        default="configs/eval/benchmark.yaml",
        help="Path to the benchmark YAML config.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write the BM25 results JSON (default: results/baselines.json).",
    )
    parser.add_argument(
        "--output-table",
        default=None,
        help="Markdown summary table (default: results/final_ablation_table.md).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the top_k from the config.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip writing the markdown table.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "With --eval-set: verify provenance, inspect every checkpoint and check its evaluation isolation, "
            "and score the stored reference runs; load no model and write nothing."
        ),
    )
    return parser.parse_args()


def run_named_eval_set(args: argparse.Namespace, config: dict) -> int:
    """``--eval-set``: everything comes from the config, so per-run path flags are refused."""
    conflicting = [
        flag for flag, value in (
            ("--corpus", args.corpus), ("--queries", args.queries), ("--qrels", args.qrels),
            ("--output-json", args.output_json), ("--output-table", args.output_table),
            ("--top-k", args.top_k),
        ) if value is not None
    ]
    if conflicting:
        print(
            f"[04_run_baselines] error: --eval-set takes its paths and settings from the config; "
            f"remove {', '.join(conflicting)}",
            file=sys.stderr,
        )
        return 2
    # Progress of long steps (dense encoding) is logged by the finretriever modules.
    logging.basicConfig(format="[%(name)s] %(message)s", stream=sys.stderr)
    logging.getLogger("finretriever").setLevel(logging.INFO)
    try:
        result = run_eval_set(config, args.eval_set, config_path=args.config, load_models=not args.check_only)
        paths = {} if args.check_only else write_eval_set_outputs(result)
    except (ValueError, OSError) as exc:
        print(f"[04_run_baselines] error: {exc}", file=sys.stderr)
        return 2
    if args.check_only:
        return report_checks(result)
    eval_set = result.eval_set
    if eval_set.status == STATUS_DEV:
        print(
            "[04_run_baselines] PROVISIONAL DEVELOPMENT METRICS "
            "(AI-generated development labels; not gold)"
        )
    print(
        f"[04_run_baselines] eval_set={eval_set.name} corpus={result.corpus.corpus_version} "
        f"sha256={result.corpus.corpus_sha256} labels={result.annotation_version} "
        f"queries={len(result.runs[0].per_query)}"
    )
    headline = ("ndcg@10", "mrr@10", "recall@100", "judged@10")
    for caution in checkpoint_cautions(result.runs):
        print(f"[04_run_baselines] {caution}")
    for run in result.runs:
        agg = run.aggregate
        metrics_str = " ".join(f"{name}={agg[name]:.4f}" for name in headline if name in agg)
        tag = (" (diagnostic)" if run.diagnostic else "") + (" (stored reference)" if run.reference else "")
        print(
            f"[04_run_baselines] run={run.run_name}{tag} mode={run.mode} {metrics_str} "
            f"unjudged@10={agg['top10_unjudged_total']}"
        )
        if run.uncertainty:
            intervals = " ".join(
                f"{name}=[{u['low']:.3f},{u['high']:.3f}]" for name, u in run.uncertainty.items()
            )
            print(f"[04_run_baselines]   95% intervals: {intervals}")
        if run.paired:
            diffs = " ".join(
                f"{name}={d['mean_difference']:+.3f}[{d['low']:+.3f},{d['high']:+.3f}] p={d['p_value']:.3f}"
                for name, d in run.paired["metrics"].items()
            )
            print(f"[04_run_baselines]   minus {run.paired['reference']}: {diffs}")
        if isolation := run.retriever.get("evaluation_isolation"):
            print(f"[04_run_baselines]   evaluation isolation: {isolation_summary(isolation)}")
        insufficient = agg["queries_with_insufficient_coverage"]
        print(f"[04_run_baselines] insufficient coverage: {', '.join(insufficient) or 'none'}")
        failed = agg.get("queries_without_relevant_in_top10")
        if failed is not None:
            print(f"[04_run_baselines] no relevant passage in the top 10: {', '.join(failed) or 'none'}")
    for skipped in result.skipped:
        print(f"[04_run_baselines] skipped {skipped['name']}: {skipped['reason']}")
    for name, path in paths.items():
        print(f"[04_run_baselines] wrote {name}: {path}")
    return 1 if result.skipped else 0


def report_checks(result: EvalSetResult) -> int:
    """``--check-only``: what was verified, and the stored reference runs; nothing was loaded or written."""
    print(
        f"[04_run_baselines] check only: eval_set={result.eval_set.name} corpus={result.corpus.corpus_version} "
        f"sha256={result.corpus.corpus_sha256} labels={result.annotation_version}"
    )
    for name, checked in (result.checked or {}).items():
        line = f"[04_run_baselines] retriever={name} type={checked['type']}"
        if checkpoint := checked["checkpoint"]:
            line += (
                f" checkpoint={checkpoint['path']} step={checkpoint['step']}/{checkpoint['total_steps']} "
                f"digest={checkpoint['digest'][:16]}"
            )
        print(line)
        if isolation := checked["evaluation_isolation"]:
            print(f"[04_run_baselines]   evaluation isolation: {isolation_summary(isolation)}")
    for run in result.runs:
        u = run.uncertainty or {}
        intervals = " ".join(f"{k}={v['mean']:.3f}[{v['low']:.3f},{v['high']:.3f}]" for k, v in u.items())
        print(f"[04_run_baselines] stored reference run={run.run_name} mode={run.mode} {intervals}")
    print("[04_run_baselines] check only: no model was loaded and nothing was written")
    return 0


def main() -> int:
    args = parse_args()

    config = read_yaml(args.config)
    if args.eval_set is not None:
        return run_named_eval_set(args, config)
    if args.check_only:
        print("[04_run_baselines] error: --check-only needs --eval-set", file=sys.stderr)
        return 2

    metrics = load_metrics_from_config(config)
    top_k = args.top_k if args.top_k is not None else load_top_k_from_config(config)
    for name, (key, fallback) in _FALLBACK_PATHS.items():
        if getattr(args, name) is None:
            setattr(args, name, config.get(key) or fallback)
    if args.output_json is None:
        args.output_json = "results/baselines.json"
    if args.output_table is None:
        args.output_table = "results/final_ablation_table.md"

    passages, queries, qrels = load_benchmark_inputs(
        args.corpus, args.queries, args.qrels
    )

    print(
        f"[04_run_baselines] corpus_size={len(passages)} "
        f"query_count={len(queries)} qrel_count={len(qrels)} top_k={top_k}"
    )

    result = run_bm25_baseline(
        passages, queries, qrels, top_k=top_k, metrics=metrics
    )

    output_table = None if args.no_table else args.output_table
    write_results(result, args.output_json, output_table)

    metrics_str = " ".join(f"{name}={value:.4f}" for name, value in result.metrics.items())
    print(f"[04_run_baselines] run={result.run_name} metrics: {metrics_str}")
    print(f"[04_run_baselines] wrote json: {args.output_json}")
    if not args.no_table:
        print(f"[04_run_baselines] wrote table: {args.output_table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
