# Full fine-tuning setup (P0) and the first training-data batch plan (P1)

Status: 2026-11-04.

**Done:**
- P0 is implemented and tested.
- P1 batch 1 is planned, with a generation dry run.

**Not done:**
- No model was trained.
- No Groq request was sent.

**Unchanged:**
- the frozen corpus;
- the evaluation data and labels;
- the historical generation records, pairs and candidates;
- the validation sets and every result.

**Labels:**
- The training and validation questions are synthetic and were accepted by an AI review. They are not
  human-validated and not gold labels.
- The validation set only chooses among one run's checkpoints.

| | What | Where |
|---|---|---|
| 1 | GradCache, with exact gradients, dropout replay and fp16 scaling | `training/biencoder_trainer.py` |
| 2 | Validation evaluator on `validation_pairs_v4` | `training/validation_eval.py` |
| 3 | The fixed checkpoint-selection rule, its record and hashes | `training/checkpoint_selection.py` |
| 4 | Three seeds (42, 1337, 2026), one run each | `training/trainer_setup.py`, `--seed` |
| 5 | Safe checkpoints and resumption with the selection state | `training/biencoder_trainer.py` |
| 6 | Final models in the benchmark, with selected-checkpoint provenance | `evaluation/benchmark.py`, `evaluation/isolation.py`, `retrieval/checkpoint.py` |
| 7 | Kaggle bundle, preflight and results collector for validated runs | `training/smoke_bundle.py`, `smoke_preflight.py`, `smoke_results.py`, [kaggle_training.md](kaggle_training.md) |
| – | Semantic leakage screen | `training/semantic_screen.py`, `screen-questions` |
| P1 | Plan `train_v3_p1_batch1` (100 train passages) | `bge_base_ft_v3/plans/train.train_v3_p1_batch1.json` |

## 1. GradCache

`mini_batch_size` in the training section turns it on (Gao et al., 2021). Each step's loss is the in-batch InfoNCE of the
whole batch. `batch_size` sets the negatives (63 at batch 64), and `mini_batch_size` sets the memory.

`cached_contrastive_backward` runs each step in three passes:
1. **Cached pass.** Every mini-batch, queries first and then passages, is encoded without gradients. Its random state
   is saved first: CPU, plus the device's on CUDA.
2. **Loss.** The loss is computed on the concatenated cached embeddings, scaled by the GradScaler exactly as the plain
   loss is, and back-propagated to those embeddings.
3. **Replay.** Each mini-batch is encoded again from its saved random state, so dropout draws the same masks. Its
   embeddings' gradients are then back-propagated into the model. Afterwards the random state is where the uncached
   passes would leave it.

**Exactness.** The parameter gradients equal those of the same forward passes without caching.
- **Checked on every step:** each replayed mini-batch is compared with its cached embeddings, and a difference above
  `REPLAY_TOLERANCE` (1e-5 in fp32, 1e-2 in fp16, relative to the largest value) refuses the step with a
  `TrainerError` before any gradient of it is used. Each step's largest difference is logged
  (`gradcache_replay_difference`).
- **Attention dropout:** GradCache runs load the base model with eager attention, so attention dropout is drawn by
  the same generator in both passes. The metadata records `attention: eager`.
- **fp16:** the cached embeddings' gradients are scaled by the GradScaler's factor, so the parameters get the same
  scaled gradients as plain training, and the optimizer step unscales and checks them as before.

**Verified:**

| Check | Result |
|---|---|
| Tiny encoder with dropout, mini-batches 1, 2, 3 and 7 | Gradients equal the uncached passes (1e-7); losses equal; random state ends where the uncached passes leave it; another state gives other gradients |
| Without dropout | Equal to the plain full-batch loss |
| CPU GradScaler (2^12) | Scaled and unscaled gradients equal plain training's |
| A replay that cannot match | Refused before any gradient accumulates |
| A training run, mini-batch 1 against the plain run (no dropout) | Same losses and weights (1e-5) |
| Tiny BERT, hidden and attention dropout 0.1, eager attention | Relative difference < 1e-5 (dense venv) |
| **Real BGE-base** (109.5M parameters, dropout on, eager attention), 4 real pairs at 512 tokens, CPU | Relative gradient difference **5.5e-8**, cosine 1.0, replay difference 0.0, identical loss |

The CUDA and fp16 version of the same check runs in the Kaggle preflight (§7).

## 2. Validation evaluator (`validation_eval`)

- **Inputs.** `validation_pairs_v4` (93 questions) is loaded by `trainer_setup.prepare_inputs` with
  `load_validation_set(training=<the pairs trained on>)`. That fails closed if a training pair shares a passage or
  equivalent with it, or a lexically overlapping question. The evaluator itself gets everything in memory and opens
  no file.
- **Relevance**, from each pair's own records:
  - relevant: the positive and its exact equivalents (`NegativeGuard.positives`);
  - removed from the ranking: every passage the guard keeps from being a negative (uncertain equivalents, overlap
    groups, reuse links). In `validation_pairs_v4` that is 50 exact equivalents in 25 questions and 29 uncertain
    ones.
- **Ranking.**
  - The whole corpus (10,137 passages) is ranked by the cosine of normalised embeddings.
  - Queries carry the BGE instruction; passages are truncated at 512 tokens as in training.
  - Ties keep corpus order.
  - `global` mode ranks everything; `company` mode ranks only the question's company.
- **Metrics.** The benchmark's own kernels: MRR@10, nDCG@10 and Recall@100, aggregate and per question.
- **No side effects.** Evaluation runs in eval mode without gradients, inside a forked random state, and restores the
  training mode. Weights are unchanged (tested).

## 3. The selection rule (`checkpoint_selection.RULE`, fixed in code)

- **Candidates.** The validation set is evaluated at step 0 (the base model, before any update) and every 5 optimizer
  steps.
- **Metric.** Whole-corpus (`global`) MRR@10; company-mode metrics are reported only.
- **Ties.** The highest value wins; a tie (equal to 12 decimal places) keeps the earlier step.
- **Early stopping.** Selection stops after 3 candidate evaluations in a row without improvement: no later
  checkpoint is selectable. Since K1 (2026-11-09, [training_a1_implementation_report.md](training_a1_implementation_report.md)),
  training does not stop there. It continues to the planned last step, and every remaining evaluation step is
  evaluated and recorded as "after the rule's stop, never selectable". The selected checkpoint is unchanged, and the
  decision rule gets every planned evaluation.
- **The final step.** When the last step is not a multiple of 5, it is also evaluated and reported, but it is never
  a candidate. The rule as approved names step 0 and every 5 steps. Scoring the last checkpoint is what lets the
  experiment's decision rule compare it, and it cannot be selected.
- **Base model.** If step 0 wins, the base model is selected: `checkpoint-step-000000`, which holds the base weights
  the trainer writes before its first update.
- **Not configurable.** A config cannot change the rule: `validation` accepts only `version` and `batch_size`.
  `save_every_steps` is refused with validation, because checkpoints follow the evaluated steps.

**The record (`selection.json`, format 2).** Written once, atomically, in the run directory when the run reaches its
planned last step. It holds:
- every evaluation up to the rule's stop: step, candidate or not, value, improvement, the patience count,
  aggregate and per-question metrics;
- `after_stop_evaluations`: the same for every later evaluation, each marked `selectable: false`;
- each evaluated checkpoint's file hashes (every file except `trainer_state.pt`) and their digest;
- the selected checkpoint (step, directory, digest, value, whether it is the base model);
- the rule's stop, the last evaluated checkpoint, and the planned and completed steps;
- the run's GPU and software environment;
- the run record's SHA-256, the config and schedule SHA-256, the seeds, the validation set's identity, and the rule.

**Reading the record back.** `read_selection_record` checks that:
- the recorded steps are every planned evaluation step;
- the after-stop evaluations are marked never selectable;
- the rule re-applied to every evaluation stops and selects where the record says.

So an edited or incomplete record is refused. `verify_selected_checkpoint` re-hashes a checkpoint and checks it, its `run.json` and its metadata
against the record.

## 4. Seeds

- **Config.** `training.seeds: [42, 1337, 2026]`; `seed` and `seeds` cannot conflict, and seeds cannot repeat.
- **One run per seed.** `--seed N` trains one run, in `<output_dir>/<name>-seed<N>/`. Training a multi-seed config
  without `--seed` is refused.
- **Dry run.** A dry run shows every seed's schedule.
- **What a seed changes:** the batch order (the conflict-free planner is seeded with `seed + epoch`) and dropout. The
  data, the validation set and the rule are the same for every seed.
- **Records.** Each run records its seed and the config's seed list.

## 5. Checkpoints and resumption

**Writing:**
- With validation, a checkpoint is written at step 0 and at every evaluated step. Each holds the model, the
  sentence-transformers module files and `metadata.json` with that step's validation metrics.
- `trainer_state.pt` (optimizer, scheduler, scaler, random state and the selection state) stays in the newest
  checkpoint only.
- An interruption (`stop_after`) writes an unevaluated checkpoint, marked as never selected.

**Never written over:**
- Checkpoint writes are atomic, and an existing checkpoint is refused.
- A different configuration in an existing run directory is refused.

**Resuming:**
- `--resume` restores the selection state, so evaluations, the best step and early stopping continue exactly. It is
  tested equal to an uninterrupted run on CPU: same losses, weights, evaluations and selection.
- A run with `selection.json` has ended. Resuming it trains, loads and writes nothing.

**Unchanged for configs without the new settings:**
- `config_record` leaves unset new settings out of `run.json`. The smoke run's stored `run.json` is reproduced
  byte-for-byte by the new code (checked on the real data).
- Its checkpoints' metadata has the same format.
- The smoke-checkpoint evaluation pre-flight (`04_run_baselines.py --check-only`) still passes, unchanged.

## 6. Final models in the benchmark

A `dense_checkpoint` entry may carry `role: final` and `selection_record: <run dir>/selection.json`.

**The harness refuses a final model:**
- without a selection record. This covers the smoke-test checkpoints and any fixed-step checkpoint; they remain
  usable as `diagnostic` entries;
- whose checkpoint is not the one its record selected, whose files differ from the hashes recorded, or whose
  `run.json` changed;
- whose own metadata says it was saved at a fixed step without validation, even with a forged record;
- unless every seed of its run is listed, once each (`check_final_seeds`). No seed is left out or picked.

**Isolation** (`checkpoint_isolation`) now also checks the validation set that selected the checkpoint:
- its recorded SHA-256;
- no judged passage among its questions' positives and equivalents;
- no validation question repeating an evaluation question.

Earlier isolation records are unchanged.

**Reporting.**
- Final runs carry `role`, the verified selection summary (step, seed, seeds, value, record SHA-256) and
  `SELECTED_CHECKPOINT_CAUTION` (scores on development labels are lower bounds).
- No evaluation set lists a final model yet, since none exists. The entry for the experiment looks like this, with one
  entry per seed:

```yaml
retrievers:
  - name: ft_v3_seed42
    type: dense_checkpoint
    role: final
    checkpoint_path: models/<run>/<name>-seed42/checkpoint-step-NNNNNN   # the one selection.json names
    selection_record: models/<run>/<name>-seed42/selection.json
```

The methodology's pooling rule is unchanged: pools are never deepened with a fine-tuned model's output. No label is
changed or added.

## 7. Kaggle bundle, preflight and results

**Bundle.** A validated config builds a `finretriever_training_bundle`. Discovery traces what the run reads, so it
adds:
- the validation set, with the carried set it verifies;
- the plan and stop record;
- the validation generation records and decision files;
- every seed's schedule.

The real bundle of `biencoder_bge_base_ft_v3_rehearsal.yaml` held 95 files (9.0 MB) and no evaluation data. It was
built and checked locally, then removed after a later code change; rebuild it with the command in
[kaggle_training.md](kaggle_training.md).

**Preflight** adds three checks:
- **validation_set:** it re-verifies, is the bundled set, and is isolated from the pairs trained on;
- **schedule:** every seed's schedule matches the bundle;
- **gradcache:** §1's check on the loaded base model, plus, on CUDA, the measured peak of one cached step over a full
  mini-batch.

The memory check uses that measurement plus AdamW's state when it exists. On the extracted real bundle (CPU, offline)
every check passed, with the CUDA checks skipped.

**Results collector.** For validated runs it reads `selection.json`, re-hashes every evaluated checkpoint, and reports
the evaluations and the selected checkpoint, under its own statement. The smoke output is unchanged.

## 8. Semantic leakage screen (`semantic_screen`, `screen-questions`)

**What it does.** Each training question is embedded with the preparation's pinned zero-shot encoder (BGE-base at
`a5beb1e3…`, no query instruction), as is each permitted evaluation question. A question is flagged at cosine ≥ 0.9,
the threshold of [training_plan_v1.md](training_plan_v1.md).

**What it reads.**
- Only the files `leakage.evaluation_question_files` names (`data/eval/annotation/questions.jsonl`,
  `data/eval/gold_queries.jsonl`), and only `query_id` and `text` from each row.
- A file named like a qrels file, or a row with a passage, grade, judgment, label or pool field, is refused before
  anything is encoded.
- No qrels, relevance labels, judged passages, pools or audits are opened. A spy test checks the files opened.

**Isolation.**
- Only `build-candidates` consults it, when a candidates config names `semantic_screening`. Flagged and unscreened
  pairs are then excluded, with the reason logged.
- The candidate loader never opens it (tested). The trainer, the validation evaluator and selection never import it
  (tested).
- Candidate sets without the field rebuild byte-identically: `train_candidates_v1` and `v2` were rebuilt as
  unchanged.

**Provenance.** `<preparation>/screening/<name>/screening.json` records:
- the question files and their SHA-256;
- the fields read and never read;
- the encoder, its revision and library versions;
- the threshold and the subjects' sources.

`screening.jsonl` has one row per training question: its SHA-256, the source pair, the highest similarity and the
nearest evaluation question's ID, never that question's text. The record is written once.

**Not run yet** on real data. It runs on the batch-1 pairs once they exist (§9).

## 9. P1: plan `train_v3_p1_batch1`

```bash
python3 scripts/02_generate_queries.py plan --config configs/train/data_prep_v3.yaml --split train \
    --name train_v3_p1_batch1 --size 100 --method balanced_years --max-company-share 0.25 --one-per group \
    --exclude-candidate-groups train_candidates_v2 \
    --after-generated configs/train/data_prep_v2.yaml configs/train/data_prep_v1.yaml      # written once
```

**The file.** `data/processed/train/bge_base_ft_v3/plans/train.train_v3_p1_batch1.json`, SHA-256 `3b9210c3…cec0`,
seed 20261029. It matches an in-memory simulation exactly.

**Sources.** 4,775 train passages are long enough. Source quality, including v3's context rules, keeps 3,224.

**Exclusions:** 139 overlap groups:
- every group a passage of the 66 candidates uses (positives, exact and uncertain equivalents), with the new
  `--exclude-candidate-groups` option;
- every group of the 75 train passages already generated under `bge_base_ft_v2` (50) and `bge_base_ft_v1` (25).

So no new question is asked of a passage an existing candidate uses, and the 66 candidates are only read: their
dataset and source pairs are unchanged.

**Balance:** at most 25% per company, one passage per overlap group.

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | All | Tables |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 2 | 2 | 2 | 2 | 2 | 2 | 2 | – | 14 | 7 |
| AMZN | 2 | 2 | 2 | 2 | 2 | 2 | 2 | – | 14 | 7 |
| JPM | 2 | 2 | 2 | 2 | 2 | 2 | 2 | – | 14 | 7 |
| PFE | 3 | 2 | 2 | 2 | 2 | 2 | 2 | – | 15 | 7 |
| TSLA | 2 | 2 | 2 | 2 | 2 | 2 | 2 | – | 14 | 7 |
| WMT | – | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 15 | 7 |
| XOM | 2 | 2 | 2 | 2 | 2 | 2 | 2 | – | 14 | 7 |

- **Kinds:** 51 narrative and 49 table passages.
- **Section families:** statements 31, MD&A 29, risk 20, business 10, other 10.
- **Spread:** 75 split units, the largest giving 5 passages (Walmart, linked by evidence reuse). Every company covers
  every fiscal year it has.

**Generation settings.** `sec-qgen-v2` (prompt `f42bdc54…`), validator `sec-qcheck-v3`, Qwen through Groq, two
questions per passage. These are the settings of the validation questions and `pilot_v3`. The validator and evidence
locator versions are unchanged, so `validation_pairs_v4` and the candidates still load.

**Dry run** (no key in the environment): no request was sent, and no file was written.

```bash
python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v3.yaml --split train \
    --plan train_v3_p1_batch1 --max-requests 100                     # add --confirm-api-use only after approval
```

**Expected yield.** These are estimates from `pilot_v3`'s measured rates, not measurements:

| Stage | `pilot_v3` (50 passages) | Batch 1 (100 passages), estimate |
|---|---|---|
| Questions passing the validator | 78 of 100 | about 155 |
| Pairs built | 70 | about 140 |
| AI review accepts | 57 (81%) | about 115 (more if v3's context rules help as they did for validation) |
| Candidates after the evidence and leakage checks | 51 | **about 100 (range about 80–115)** |

The planned passages' evidence is slightly less isolated from validation and blocked passages than `pilot_v3`'s.
The mean share of visible sentences that no passage outside the train split holds is 0.55, against 0.60. So the
evidence rules may exclude a few more pairs than `pilot_v3`'s 6 of 57.

**Cost.**
- **Tokens:** about 143,000. That is 100 requests at `pilot_v3`'s 1,426 tokens each (median: about 1,200 prompt and
  230 completion); at most about 200,000 if every reply used its 768-token limit.
- **Requests:** `--max-requests 100` caps every attempt, retries included.
- **Time:** about 100 minutes at the configured pacing (1 request per minute when each reserves 768 output tokens;
  `pilot_v3`'s median gap was 59 s).
- **Review:** about 140 pairs need an AI review.

**After generation, each step offline:**
1. `build-pairs`.
2. The AI review: decisions in `data/train_review/`, `ai_review`, `gold: false`.
3. `screen-questions --pairs --candidates train_candidates_v2`.
4. A new `candidates_v3.yaml` with `semantic_screening`, building `train_candidates_v3` (the 66, re-checked, plus
   batch 1).

## 10. Tests and checks

**Test files:**

| File | Tests |
|---|---|
| `tests/unit/test_gradcache.py` | Exactness with dropout, the plain loss, GradScaler scaling, refusal of an inexact replay, the preflight check, a cached training run, a tiny BERT |
| `tests/unit/test_checkpoint_selection.py` | The rule (candidates, ties, patience, final step, base model), a validated run's checkpoints and hashes, tampering, early stopping, resume equality, finished runs, three seeds, isolation from the trained pairs, config strictness, no evaluation file opened, evaluator rankings and metrics against the benchmark kernels, no side effects |
| `tests/unit/test_final_models.py` | Every seed's selected checkpoint scored with its verified selection; smoke, unselected, changed or forged checkpoints refused; every seed required; validation isolation |
| `tests/unit/test_semantic_screen.py` | Only permitted question texts read; flagging and provenance; refusal of label-bearing files; write-once; candidate exclusion; the loader never opens the screen |
| `tests/unit/test_smoke_bundle.py` (added) | Training bundle contents; preflight with validation, seeds and GradCache; a validated run collected with its selection |
| `tests/unit/test_training_data_cli.py` (added) | `--exclude-candidate-groups` |

**Real data** (each read-only, or write-once with an unchanged result):
- the smoke `run.json` is reproduced byte-for-byte;
- `train_candidates_v1`, `train_candidates_v2` and `validation_pairs_v4` rebuild unchanged;
- the smoke-checkpoint evaluation pre-flight passes;
- the rehearsal dry run passes;
- the real bundle preflight passes;
- the real-BGE GradCache check passes.

## 11. Remaining risks

1. **The rule's step 0 can win.** With little data, validation may select the base model. That is a valid outcome,
   recorded as such.
2. **GradCache on CUDA is checked, not yet measured.** The fp16 replay tolerance (1e-2) and the per-step replay check
   have only run on CPU. The Kaggle preflight's GPU check is the first CUDA run.
3. **Kaggle disk.** Checkpoints are kept at every evaluated step: about 438 MB each, up to about 12 per seed. Run one
   seed per session, or download between seeds.
4. **Validation encodes the whole corpus at each evaluation.** That is about 1–1.5 min on a T4 (estimated), and
   dominates a run's time.
5. **The validation and training questions come from the same generator.** Validation gains may partly reflect its
   style.
6. **The semantic screen's threshold (0.9) is not calibrated on this corpus.** Its distribution will be reported when
   it first runs.
7. **Generation needs approval, and yield is uncertain.** The figures above are estimates, and the Groq account's
   daily token limit is unknown.

## 12. Changes

| File | Change |
|---|---|
| `src/finretriever/training/checkpoint_selection.py` (new) | The rule, its state, `selection.json`, checkpoint hashing and verification |
| `src/finretriever/training/validation_eval.py` (new) | The validation evaluator |
| `src/finretriever/training/semantic_screen.py` (new) | The semantic leakage screen |
| `src/finretriever/training/trainer_setup.py` | `seeds`, `mini_batch_size`, `validation`; `for_seed`, `config_record`; the validation set loaded with isolation |
| `src/finretriever/training/biencoder_trainer.py` | GradCache and its check; validation, selection and records in `train`; `--seed`; eager attention for GradCache |
| `src/finretriever/training/candidates.py` | Optional `semantic_screening` |
| `src/finretriever/training/data_cli.py` | `screen-questions`; `plan --exclude-candidate-groups` |
| `src/finretriever/training/smoke_bundle.py`, `smoke_preflight.py`, `smoke_results.py` | Training bundles, the validation, seed and GradCache checks, validated results |
| `src/finretriever/evaluation/benchmark.py`, `isolation.py`; `src/finretriever/retrieval/checkpoint.py` | Final models, the seed rule, validation isolation, cautions |
| `scripts/04_run_baselines.py`, `scripts/smoke_bundle.py` | The caution helper; the smoke fallback only for the smoke config |
| `configs/train/biencoder_bge_base_ft_v3_rehearsal.yaml` (new) | The mechanics rehearsal at the real settings (66 candidates; not the experiment) |
| `docs/training_full_v1.md`, `docs/kaggle_training.md` (new) | This document; the Kaggle guide |
| `data/processed/train/bge_base_ft_v3/plans/train.train_v3_p1_batch1.json` (new, gitignored) | The batch-1 plan |
| Tests | Five new or extended files; `TinyEncoder` moved to `tests/unit/trainer_fixtures.py` |
