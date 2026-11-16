# Results

Benchmark outputs land in this directory. Files are committed so that every
reported number can be traced back to a specific run.

## Files

- `baselines.json` — BM25, zero-shot BGE and zero-shot e5 scores.
- `biencoder_v1.json` — Fine-tuned FinRetriever scores (filled after
  training).
- `final_ablation_table.md` — Human-readable summary table comparing all
  systems on nDCG@10, MRR@10 and Recall@100.

All three are placeholders. The gold benchmark they were planned for was not
completed: gold set v1 was closed as an incomplete pilot on 2026-11-15
([docs/gold_set_v1_closure.md](../docs/gold_set_v1_closure.md)). No gold-set
number may be written into them.

## Development runs

`dev/<label version>/` holds runs on AI-generated development labels
(`scripts/04_run_baselines.py --eval-set dev_pilot_v2`). It contains one
directory per retrieval mode (`global`, `company`) and, inside it, one per
retriever, each with `metrics.json` (aggregate and per-query metrics,
judged@10, retriever and model provenance) and `run.jsonl` (the ranked hits,
with each hit's judgment or `null` if unjudged). A `report.md` compares all
runs. They are **provisional development metrics**, never benchmark results,
and they never feed the files above.

`dev/pilot_v3/` runs no retriever: it re-scores the stored rankings of
`dev/pilot_v2/` against the `pilot_v3` labels (`runs_from` in the benchmark
config). Its `run.jsonl` files have the same ranks, passages and scores,
each `metrics.json` records the source run file and its SHA-256
(`source_run`) and the `pilot_v2` scores (`compared_with`), and
`report.md` shows both side by side.

`dev/pilot_v3_smoke_checkpoints/` (written by
`scripts/04_run_baselines.py --eval-set dev_pilot_v3_smoke_checkpoints`, not
yet run) holds the three checkpoints of the 15-step trainer smoke test on
the `pilot_v3` labels.
- Each checkpoint is checked for evaluation isolation before it is loaded.
- The stored `pilot_v2` zero-shot rankings are scored beside them, not
  written again, and every run is compared with `bge_zero_shot` question by
  question.
- Every metric comes with a 95% bootstrap interval.
- No checkpoint is selected, and no improvement is claimed. See
  [docs/checkpoint_evaluation.md](../docs/checkpoint_evaluation.md).

`dev/pilot_v3_a1/` (written by `scripts/04_run_baselines.py --config
configs/eval/retrieval_analysis_v1.yaml --eval-set dev_pilot_v3_a1`) holds
the A0 checkpoint and the three official A1 checkpoints on the `pilot_v3`
labels, beside the stored zero-shot rankings. Their scores are lower bounds:
the judged pools do not include A1's results.

Every development directory is written once: a run whose files would differ
from those present is refused and writes nothing.

## Retrieval analysis

`analysis/retrieval_analysis_v1/` (written by
`scripts/09_retrieval_analysis.py` and
`scripts/10_retrieval_analysis_supplement.py`) holds the data behind
[docs/retrieval_analysis_v1.md](../docs/retrieval_analysis_v1.md):

- `results.json` and `supplement.json`;
- the per-question files and every system's top 10;
- the generated tables.

It is not a benchmark. It reads no human relevance label, and its real-question
part is a metadata diagnostic. Re-running the scripts rewrites these files; the
per-question outputs are deterministic, and two runs were byte-identical.

Dense passage embeddings are cached in `data/cache/embeddings/`, which is
gitignored and rebuilt when missing.

## Retrieval experiments (v2 and v2.1)

`experiments/` holds one directory per experiment suite, each written by
`scripts/11_v2_experiments.py` from its own config. Neither is a benchmark;
see [docs/retrieval_v2_metadata_hybrid.md](../docs/retrieval_v2_metadata_hybrid.md).

- `experiments/v2_metadata_hybrid/`: 26 experiments over three A1 seeds
  (`configs/experiments/v2_metadata_hybrid.yaml`, parser qu-v1). Frozen.
- `experiments/v2_1_year_roles/`: the v2 experiments that the fiscal-year
  parser affects, re-run with parser qu-v2
  (`configs/experiments/v2_1_year_roles.yaml`). Each run is compared question
  by question with the same experiment and seed in the frozen v2 run.

Each directory contains:

- `manifest.json`: the settings, input and code hashes, checkpoint digests,
  cache keys and library versions;
- `summary.json`, `parser_accuracy.json` and `year_window_choice.json`;
- `runs/<experiment>/<seed>/{validation,development,real}.jsonl`: per-question
  rankings and scope traces (passage IDs only, no passage text);
- `tables.md`, generated from the files above (`--tables-only` re-renders it).

Two runs of each suite were byte-identical.
