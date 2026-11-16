# Demo Fixture

A synthetic, hand-written mini benchmark used purely as a smoke test for the
BM25 baseline pipeline. **This is not the v1 SEC gold eval set.** It exists
so that anyone cloning the repo can run a real benchmark loop end-to-end
before the SEC corpus has been built.

The texts here are hand-written to look SEC-flavored; they are not actual
filings and they are not financial advice.

## Files

- [`mini_corpus.jsonl`](mini_corpus.jsonl) — 7 synthetic passages across
  AAPL, JPM, TSLA, PFE, AMZN, XOM. Schema:
  [`finretriever.data.schemas.Passage`](../src/finretriever/data/schemas.py).
- [`mini_queries.jsonl`](mini_queries.jsonl) — 5 queries with
  `source: "demo"`. Schema: `Query`.
- [`mini_qrels.jsonl`](mini_qrels.jsonl) — relevance grades (`2` = directly
  relevant, `1` = partially relevant) linking demo queries to passages.
  Schema: `Qrel`. At least one query has multiple relevant passages so the
  graded-nDCG path is exercised.

## Run

```bash
python3 scripts/04_run_baselines.py \
    --corpus demo/mini_corpus.jsonl \
    --queries demo/mini_queries.jsonl \
    --qrels demo/mini_qrels.jsonl \
    --output-json results/demo_baselines.json \
    --output-table results/demo_ablation_table.md
```

This writes:

- `results/demo_baselines.json` — full `RunResult` (metrics + hits + metadata).
- `results/demo_ablation_table.md` — README-ready markdown summary.

## What this fixture is not

- Not the SEC 10-K corpus (that lands later via `scripts/01_build_corpus.py`).
- Not the hand-annotated gold eval set (lands later via
  `scripts/03_annotate_gold_set.py`).
- Not a benchmark we report numbers from. The point is to prove the pipeline
  is wired up; the v1 headline numbers come from the real gold set.
