# A1 three-seed experiment: consolidated results and the final analysis plan

> **Closed, incomplete (2026-11-15).** Gold set v1 human annotation was stopped after one graded question (q_0011, 45 judgments). The 100-question gold benchmark, the inter-annotator agreement analysis and the gold-set decision test were not completed, and no result comes from gold set v1. Stage C below therefore never happened: the only retrieval claim remains Stage A's, on synthetic validation questions, and no improvement on real questions is claimed. See [gold_set_v1_closure.md](gold_set_v1_closure.md).

**Date:** 2026-11-12.

**Status (updated 2026-11-12, after your decisions D1–D7):**
- **Training is complete, and all three seeds are official:** seed 42 step 55, seed 1337 step 40, seed 2026 step 40.
  Each was trained once, as a saved version on a Kaggle Tesla T4, and independently audited.
- **Stage A, the approved validation decision, has been run** (§6): **"A1 beats A0 on the validation set"**, p =
  0.0001. This applies only to the 93 synthetic, AI-reviewed validation questions. It is not an independent
  retrieval benchmark.
- **No retrieval improvement is claimed.** Only the gold set (Stage C) can support one. Its protocol, query plan and
  frozen analysis are prepared ([gold_set_v1_protocol.md](gold_set_v1_protocol.md)), and annotation has not started.
- **Nothing else was done:** no training, no Kaggle run, no API request, no commit or push. No evaluation label was
  opened, and the three official checkpoints and their records are unchanged.

## 1. The experiment

**The question:** does **A1** beat **A0**?
- **A0** is the zero-shot base model, BGE-base-en-v1.5. It is step 0 of every run.
- **A1** is the procedure "fine-tune BGE-base on `train_candidates_v3`, then select a checkpoint by the fixed rule".

| Fixed before training | Value |
|---|---|
| Base model | `BAAI/bge-base-en-v1.5`, revision `a5beb1e3…`; BGE query instruction; CLS pooling, L2-normalised; passages up to 512 tokens |
| Configuration | `configs/train/biencoder_bge_base_ft_v3_a1.yaml` (`7a4c4618…`):<br>• learning rate 2e-5; batch 64 (balanced 61–62); GradCache mini-batch 16;<br>• fp16 with the default GradScaler (65,536); temperature 0.02;<br>• 5 epochs = 55 steps, 6 of warmup; weight decay 0.01; `max_grad_norm` 1.0 |
| Training data | `train_candidates_v3`: 678 synthetic pairs, AI-reviewed (manifest `0586c7ca…`) |
| Selection data | `validation_pairs_v4`: 93 synthetic questions, AI-reviewed, isolated from the training pairs (manifest `40238cb0…`, IDs `b83abf1a…`) |
| Corpus | `corpus_v1@1.2`: 10,137 passages (`ff9dadf3…`) |
| Selection rule (K1) | Candidates are step 0 and every 5 steps, scored by whole-corpus MRR@10. Ties at 12 decimals go to the earlier step; the patience is 3. Evaluation continues to step 55. |
| Decision rule (K1) | A selection-aware sign-flip test of the selected checkpoints against step 0 (§5) |
| Environment | Tesla T4 (7.5), CUDA 13.0, cuDNN 92700, driver 580.178.04, Python 3.13.16, torch 2.11.0+cu130, transformers 5.3.0, tokenizers 0.22.2; `pip freeze` `5432fefb…` |
| Bundle | r2: archive `efc377d9…`, manifest `a6615850…` |

## 2. The three official runs

| | Seed 42 | Seed 1337 | Seed 2026 |
|---|---|---|---|
| Status | accepted 2026-11-10 | accepted 2026-11-11 | accepted 2026-11-12 (one training execution in the Kaggle version history) |
| Execution | the saved version, 2026-11-10, 15:25–15:58 UTC (notebook version 1 records no execution ID) | `b98f183e…`, batch, 2026-11-11, 00:00–00:30 UTC | `24187107…`, batch, 2026-11-11, 23:23–23:59 UTC |
| Notebook | version 1, `4447559f…` | version 2, `19bbf437…`: the batch-only gate and the execution identity | version 3, `c6ca9e5b…`: version 2 plus the dataset provenance |
| Archive (SHA-256) | `4a3b9d4a…` (four identical downloads) | `7f0317af…` | `7e79fa35…` |
| `selection.json` | `19cca1cb…` | `165064f5…` | `2afbdae2…` |
| **Selected checkpoint** | **step 55** | **step 40** | **step 40** |
| Its digest | `24c2aaae…` | `8237ec0b…` | `cba9110b…` |
| Its weights | `3dfbfa92…` | `6d9952c3…` | `8f9dbd38…` |
| Validation MRR@10 (whole corpus), selected | 0.669039 | 0.667273 | 0.658001 |
| The rule's stop | none: no third miss | step 55, the planned last step | step 55, the planned last step |
| Value at step 55 | 0.669039 | 0.660719 | 0.655210 |
| Single-rank moves that would change the selection | 251 of 1,374 | 0 of 1,354 | 75 of 1,397 |
| Audit | [seed42_final_audit.md](seed42_final_audit.md) | [seed1337_final_audit.md](seed1337_final_audit.md) | [seed2026_final_audit.md](seed2026_final_audit.md) |

**A0, step 0:** validation MRR@10 0.265886. It is identical in all three runs: weights `0d4db737…`, and the same
first-relevant rank for every question (`0de690c4…`, equal to the rehearsal's).

## 3. Validation results (descriptive only)

**Read these numbers with care:**
- **Synthetic.** They are on the 93 synthetic, AI-reviewed validation questions.
- **Optimistic.** The same questions chose each checkpoint, so the selected values are optimistic by construction.
- **In-distribution.** The questions come from the same generator, prompt and filings as the training pairs, with
  repeated templates. A large gain here can reflect learning that distribution.
- **Not a result.** They are not a retrieval result. The decision test on them is in §6 (Stage A).

**MRR@10 at every evaluation:**

| Step | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seed 42 | 0.2659 | 0.4251 | 0.5868 | 0.6324 | 0.6435 | 0.6583 | 0.6628 | 0.6686 | 0.6665 | 0.6681 | 0.6689 | **0.6690** |
| Seed 1337 | 0.2659 | 0.4096 | 0.5696 | 0.6103 | 0.6338 | 0.6340 | 0.6390 | 0.6505 | **0.6673** | 0.6577 | 0.6585 | 0.6607 |
| Seed 2026 | 0.2659 | 0.4110 | 0.5659 | 0.6315 | 0.6472 | 0.6526 | 0.6512 | 0.6529 | **0.6580** | 0.6490 | 0.6493 | 0.6552 |

**Other metrics at the selected checkpoints** (the decision rule reports these, but they never decide):

| | Step 0 (A0) | Seed 42, step 55 | Seed 1337, step 40 | Seed 2026, step 40 |
|---|---:|---:|---:|---:|
| MRR@10, whole corpus | 0.2659 | 0.6690 | 0.6673 | 0.6580 |
| nDCG@10, whole corpus | 0.3059 | 0.7176 | 0.7040 | 0.7119 |
| Recall@100, whole corpus | 0.8369 | 1.0000 | 0.9969 | 1.0000 |
| MRR@10, the question's company | 0.2824 | 0.7373 | 0.7380 | 0.7167 |
| nDCG@10, the question's company | 0.3295 | 0.7750 | 0.7694 | 0.7683 |

**The curves:**
- **The plateau.** All three rise steeply to step 15, then flatten. From step 35 on, every seed stays within 0.017
  of its own maximum.
- **The spread.** The three selected values span 0.0110. From step 40 on, each seed's checkpoints are within 0.01
  of one another.
- **The selection is sensitive to single ranks for two seeds.** For seeds 42 and 2026, one question moving one rank
  could change the selected step, among checkpoints about 0.005 apart or less. For seed 1337 it could not. The rule
  was applied as written in every case, and the decision test re-applies it to every resample.

## 4. Integrity and consistency across the seeds

**One procedure, three seeds.**
- **`run.json`:** each pair of seeds differs only in the seed, the run name and the schedule hash. The
  hyperparameters, base model, inputs, validation set and learning-rate sequence are identical.
- **The schedule:** each seed has its own (`44b0ed7c…`, `795095dc…`, `49b2b153…`), all with the same shape: 55 steps,
  6 of warmup, balanced 11 × 5, none dropped. Each equals the hash recorded before training.
- **The environment:** equal on all nine identity keys; `pip freeze` byte-identical; the same driver.
- **Step 0:** the same weights and the same 93 ranks in all three runs, so A0 is one model.

**Integrity.**
- **The archives:** every listed file in each archive was re-hashed from inside the zip, with no mismatch, no
  unlisted file and no missing file.
- **The checkpoints:** every digest matches its selection record.
- **The resume states:** each was read independently and agrees with its records.
- **No change since:** all six archive files and every extracted record are byte-identical to their recorded hashes
  ([seed2026_final_audit.md](seed2026_final_audit.md) §7).

**Training behaviour:**

| | Seed 42 | Seed 1337 | Seed 2026 |
|---|---|---|---|
| Steps | 55 of 55 | 55 of 55 | 55 of 55 |
| Loss, first → last (in-batch InfoNCE) | 1.2130 → 0.1513 | 1.0889 → 0.1041 | 1.5173 → 0.1056 |
| Mean loss by epoch | 0.881, 0.338, 0.157, 0.117, 0.110 | 0.831, 0.308, 0.170, 0.132, 0.106 | 0.842, 0.321, 0.166, 0.138, 0.113 |
| Gradient norm before clipping: min / median / max | 2.11 / 4.88 / 16.90 | 2.80 / 4.79 / 14.63 | 1.50 / 5.08 / 15.95 |
| Steps clipped | 55 of 55 | 55 of 55 | 55 of 55 |
| Non-finite loss or gradient; fp16 overflow | none | none | none |
| Loss scale | 65,536 throughout | 65,536 throughout | 65,536 throughout |
| GradCache replay difference | 0.0 | 0.0 | 0.0 |
| Time per step; peak GPU memory | 3.63 s; 9.22 GiB | 3.34 s; 9.22 GiB | 4.02 s; 9.22 GiB |

**Isolation from evaluation data.**
- **Absent from the bundle:** it holds no `data/eval/`. The preflight's `evaluation_isolation` check passed on the T4
  in every run: none of the 8 evaluation sources is present or was read.
- **Selection:** it used only `validation_pairs_v4`.

## 5. The approved decision rule: review

### 5.1 What it is

The rule was approved on 2026-11-09 (K1, [training_b1_b3_proposal.md](training_b1_b3_proposal.md) §1.5). It is
implemented in `finretriever.training.decision_rule` and run by `scripts/07_decide_a1_vs_a0.py`.

**Inputs:** the three `selection.json` records only. A path under `data/eval` is refused.

**The statistic:** T = (1/3) Σ_s [V_s(k*_s) − V(0)], the mean gain of the selected checkpoints over step 0 in
whole-corpus MRR@10.

**The null distribution:**
- 10,000 sign vectors over the 93 questions, from `PCG64(20261109)`.
- Each vector flips every question's differences from step 0 at once, across all evaluations and seeds.
- The fixed rule is re-applied to every flipped curve, so selection is part of the null.

**The p-value:** one-sided, p = (1 + #{T*_b ≥ T − 1e-12}) / 10,001.

**The decision:**
- "A1 beats A0 on `validation_pairs_v4`" if and only if p ≤ 0.05 and every seed selected a step after 0.
- Otherwise the result is "no improvement shown".
- It never concludes that A0 beats A1.

**Reported, never deciding:**
- T (optimistic), and the selection-adjusted T − E*[T*];
- per seed: the selected step, the rule's stop and the last step;
- the seed-minimum statistic and its p-value;
- question-bootstrap 95% intervals (10,000 resamples, seed 20261110);
- the other metrics.

**The record** is written once and holds its inputs' SHA-256, so it is reproducible.

### 5.2 The preconditions, checked now (without running the test)

I ran only the module's own precondition code (`load_records`, `seed_runs`) on the three official records. No flip,
statistic, p-value or record was produced.

| Precondition | Result |
|---|---|
| Each record verifies under the fixed rule (`read_selection_record`) | pass: `19cca1cb…`, `165064f5…`, `2afbdae2…` |
| The same seed list, each seed once: [42, 1337, 2026] | pass |
| The same config, validation set and rule | pass |
| Every planned evaluation is present | pass: 12 per seed. There are no after-stop evaluations, because no seed stopped before step 55. |
| The same 93 questions in every evaluation, and per-question values that average to the recorded values | pass |
| Step 0 identical across seeds (N10) | pass |
| The same hardware and software (nine identity keys) | pass |

### 5.3 Review findings

1. **The rule is implemented as approved, and every precondition holds.** I recommend no change. Changing it now,
   with the per-seed values visible, would itself compromise it.
2. **What a pass would mean.** "A1 beats A0 on this synthetic validation distribution". It is not a retrieval
   improvement on real questions, and the rule's own text says so.
   - The validation questions share the training pairs' generator, prompt, filings and templates (top word overlap
     0.765).
   - So the size of a validation gain says little about the benchmark. This is why a pass cannot support a retrieval
     claim; §6, Stage C, is the independent evaluation.
3. **The fragile selections are handled.** The test re-applies the rule to every flipped curve, so the sensitivity of
   seeds 42 and 2026 (§3) is inside the null distribution. Nothing needs adjusting.
4. **Three seeds are a consistency check,** not an estimate of seed variance. The inference covers the sampling of
   these 93 questions, given these three runs.
5. **Paths.** The proposal's example command read the records from `models/…` on the training machine. Here they are
   in the extracted archives under `dist/`. They are the same files: verified by hash, and the decision record stores
   each input's SHA-256.

## 6. The final analysis plan (proposed 2026-11-12; decided the same day)

Three stages, kept apart. Only Stage C can support a retrieval claim.
- **Stage A:** approved and run (D3).
- **Stage B:** skipped (D4).
- **Stage C:** its rule is approved (D5, D6) and its annotation protocol is prepared (D7). It waits for two open
  decisions (D8, D9) before annotation can start.

### Stage A: the decision on validation (K1), run 2026-11-12

**Result: "A1 beats A0 on the validation set."**
- It holds on `validation_pairs_v4` only: 93 synthetic questions written by an LLM and accepted by an AI review, not
  gold labels, and the same questions that chose each checkpoint.
- **It is not an independent retrieval benchmark, and it supports no retrieval claim.**

| | |
|---|---|
| Decision record | `results/decisions/bge_base_ft_v3_a1_vs_a0.json`, SHA-256 `ea079827a9b25d3bc527c28e4f0a8bd73faf285cf7cc5adbbed6d2380c6b9a55`. It was written once; a second run reproduced it byte for byte (write-once check: "unchanged"). |
| Implementation | Exactly the pre-approved code: `decision_rule.py` `5d943450…`, `checkpoint_selection.py` `a612636e…`, `07_decide_a1_vs_a0.py` `17cd24f0…`. All are unchanged since commit `f3b2ff9` (2026-11-10 09:10 +04:00), before the first seed trained. Its 34 tests pass, and nothing was modified after the result. |
| Inputs | Only the three official selection records, `19cca1cb…`, `165064f5…` and `2afbdae2…`, byte-identical before and after. The 93 questions hash to `b83abf1a…`. No development or evaluation file was read. |
| Test | Selection-aware sign flip: 10,000 flips, `PCG64(20261109)`; the fixed rule re-applied to every flipped curve |
| **p (one-sided)** | **0.0001** (= 1/10,001). No flipped dataset reached the observed statistic, so this is the smallest p 10,000 flips can give. |
| **Observed effect T** | **+0.3989** MRR@10: from 0.2659 at step 0 to a mean of 0.6648 at the selected checkpoints. It is optimistic by construction. |
| Selection optimism under the null, E*[T*] | +0.0339 |
| **Selection-adjusted estimate**, T − E*[T*] | **+0.3649** |
| **95% intervals** (question bootstrap, 10,000 resamples; not selection-adjusted) | selected minus step 0: [+0.3215, +0.4780]; last step minus step 0: [+0.3182, +0.4744] |
| Every seed selected a step after 0 | yes |
| **Per seed** | 42: step 55, 0.6690, +0.4032 (no rule stop)<br>1337: step 40, 0.6673, +0.4014 (rule stop 55)<br>2026: step 40, 0.6580, +0.3921 (rule stop 55) |
| Seed minimum | +0.3921, p 0.0001 |
| Other metrics (means over seeds; reported, never deciding) | nDCG@10 0.3059 → 0.7111; Recall@100 0.8369 → 0.9990; company-mode MRR@10 0.2824 → 0.7307 |

**How to read it:**
- **What it shows:** on the synthetic validation distribution, fine-tuning then selecting ranks the positive
  passages far higher than the base model. All three seeds agree, and the selection optimism (+0.034) is small
  beside the effect.
- **What it does not show:** an improvement on real questions.
  - The validation questions share the training pairs' generator, prompt, filings and templates, so a gain this
    large may partly reflect learning that distribution.
  - Only the gold set (Stage C) tests real questions.
- **The p-value is at its floor.** More flips would only resolve it more finely; the decision would not change.

**The command that was run** (it writes once and reads only the three records):

```bash
PY=python   # the environment of requirements/lock-*.txt (docs/environment.md)
$PY scripts/07_decide_a1_vs_a0.py \
    --selection dist/kaggle_a1_seed42_results/kaggle_a1_seed42_outputs/run/bge_base_ft_v3_a1-seed42/selection.json \
    --selection dist/kaggle_a1_seed1337_results/kaggle_a1_seed1337_outputs/run/bge_base_ft_v3_a1-seed1337/selection.json \
    --selection dist/kaggle_a1_seed2026_results/kaggle_a1_seed2026_outputs/run/bge_base_ft_v3_a1-seed2026/selection.json \
    --out results/decisions/bge_base_ft_v3_a1_vs_a0.json
```

**Output:** the decision ("A1 beats A0 on `validation_pairs_v4`", or "no improvement shown"), p, T, the
selection-adjusted estimate, the per-seed values, the seed-minimum test and the intervals.

**Wording, whatever the outcome:** "on the synthetic validation set". It takes seconds and needs no GPU.

### Stage B: development diagnostics on `dev_pilot_v3`: **skipped (D4)**

**Not run, and no file was deleted to make space for it.** The proposal is kept below as it was.
- **The pipeline check it offered** (final-model loading, isolation, A0 identity) moves to the gold benchmark itself.
- **The A0 identity** is now handled by defining A0 as the step-0 checkpoint
  ([gold_set_v1_protocol.md](gold_set_v1_protocol.md) §3.2, D9).

**What:** score the three selected checkpoints with the existing benchmark harness, against the stored zero-shot
runs, on the 10 AI-labelled development questions.

**Why it is useful:**
- it exercises the final-model path: `role: final`, all three seeds listed once, the selection-record checks and
  the isolation checks;
- with the step-0 checkpoint added as a `diagnostic` entry, it checks that the harness's BGE zero-shot run equals
  step 0 (the same rankings), which Stage C needs. A cheaper partial check is to compare step 0's weights
  tensor by tensor with the cached base model; it needs no encoding.

**Why it cannot decide anything:**
- there are only 10 questions;
- the labels are AI development labels;
- the pools were built from the zero-shot runs only, so a fine-tuned model's unjudged passages count against it, and
  its scores are lower bounds. Report judged@10 with them.

**What it needs:**
- **Your approval.** It reads AI development labels, through the harness only; I would not open them.
- **Disk space.** It extracts the three selected checkpoints, step 0 and their records from the archives into
  `models/bge_base_ft_v3_a1/…`, about 1.8 GB, each verified against its digest. **This Mac has 2.1 GiB free,** which
  is too little to do this safely.
  - The three extra seed 42 downloads are byte-identical copies of the first (5.7 GB each).
  - Removing them, or moving them to another disk, is your call.
- **The time:** about 30 minutes of CPU encoding per checkpoint, so about 2 hours for four.
- **A new eval-set entry** in `configs/eval/benchmark.yaml`, writing to `results/dev/pilot_v3_a1_final/` once.

### Stage C: the independent evaluation (the only basis for a retrieval claim)

**Decided 2026-11-12:**
- **D5:** the primary analysis below is kept: nDCG@10, A1 against A0, a one-sided paired sign-flip test, every seed
  must improve. It is to be frozen in code before any gold label exists.
- **D6:** the methodology's treatment of unjudged passages is kept, with A1's downward bias stated explicitly.
- **D7:** the annotation protocol is prepared, but annotation is not started.

All of it is in [gold_set_v1_protocol.md](gold_set_v1_protocol.md):
- the frozen implementation, `gold_decision.py` `d7215c30…`, with 22 tests;
- the query plan;
- the relevance guidelines;
- the blinded workflow, with the new `annotate --blind` and `leakage-check`;
- the power table;
- the readiness checklist.

**Two points surfaced while preparing it** (each now an open decision):
- **D8 (passages A1 trained on).** The benchmark's isolation check refuses to score A1 on any judged passage it
  trained on, and on any judged passage missing from the frozen training blocklist. Under today's code, A1 could not
  be scored on any new gold set at all. 1,745 passages (17.2% of the corpus) were training or validation positives
  or equivalents. The recommended resolution: a residual-collection primary scope, with the full corpus reported.
- **D9 (A0).** A0 should be the step-0 checkpoint, scored through the same code path as A1, rather than the
  separately loaded BGE zero-shot model. Then the A0–A1 difference is the fine-tuning alone.

**The rule as first proposed** (kept for the record; the protocol's §3 has the implemented version):

**The state of the gold set.** It does not exist yet ([eval_methodology.md](eval_methodology.md)).
- `gold_queries.jsonl` and `gold_qrels.jsonl` are empty.
- Human grading of the 10-question pilot stopped after 4 grades.

The independent evaluation therefore needs, in order:
1. **The gold set.** Build gold set v1 by the methodology: about 100 questions, mainly new and written by people.
   - Pools come from BM25, manual, stratified and random sources only. No model, and never the fine-tuned model,
     selects candidates.
   - The grading is human (tier 3), and the result is a frozen export.
2. **The benchmark.** Run it once, on that export: BM25, BGE zero-shot (A0), E5 zero-shot, and A1 as the three seeds'
   selected checkpoints (`role: final`, every seed listed once).
3. **The analysis.** Apply the gold-set rule below.

**The gold-set rule** (my proposal; it must be approved before any gold label is exported or seen):

| # | Rule | Proposal |
|---|---|---|
| G1 | Primary metric | nDCG@10 (graded 0/1/2), whole-corpus mode, over the exported questions that have at least one positive |
| G2 | Primary comparison | A1 against A0. A1's score per question is the mean over the three seeds' selected checkpoints. A0 is BGE zero-shot, which must equal step 0: the same revision `a5beb1e3…`, instruction, pooling and lengths, and the same rankings (checked in Stage B). Otherwise no decision is made. |
| G3 | Test | One-sided paired sign-flip test over questions: 10,000 flips, a fixed seed, α = 0.05. The checkpoints are fixed in advance and the gold set selects nothing, so no selection adjustment is needed. |
| G4 | Consistency | Each seed's own mean difference must also be above 0. Like K1's "every seed" condition, this is required for the claim. |
| G5 | Unjudged passages | Unjudged passages count as non-relevant, and pools are never deepened with A1's output (the methodology). Report judged@10 for every system, and judged-only scores as a sensitivity check. This biases the comparison against A1: a pass stands, but a fail with A1's judged@10 clearly below BM25's is inconclusive, not negative. |
| G6 | Reported, never deciding | MRR@10, Recall@100, company mode, per company and per question type; A1 against BM25 and E5; each seed separately; bootstrap 95% intervals |
| G7 | The pilot questions | `q_0001`–`q_0010` are AI-drafted. Exclude them from G1–G4, and report the results with and without them, as the methodology requires. |
| G8 | One look | The gold set is scored once for this comparison. Nothing is retrained, reselected or re-thresholded on it. Any further model (for example A2) needs its own plan and a multiplicity rule. |
| G9 | Power | Before freezing G1–G8, estimate the detectable effect for the expected number of questions, using the zero-shot development spread. No gold label is needed for this. |

**The claim, if G3 and G4 pass:** "A1 improves nDCG@10 over A0 on gold set v1". It covers these questions, this
corpus and these three runs. If they fail: "no improvement shown on gold set v1".

### Stage D: the final report

It keeps the three layers apart, each with its label:
- the validation decision (synthetic; Stage A);
- the development diagnostics (AI labels, lower bounds; Stage B);
- the gold benchmark, the only basis for a retrieval claim (Stage C).

## 7. Disclosures carried into any report

1. **Seed 42 was trained twice.**
   - The earlier execution, reported to select step 35, is disclosed and non-authoritative. It is never pooled or
     substituted.
   - Its console output was never provided ([seed42_final_audit.md](seed42_final_audit.md)).
2. **GPU training is not bit-reproducible.** Each seed was trained once, and its saved version is the run of record.
3. **Every optimizer step was clipped** (`max_grad_norm` 1.0 at temperature 0.02), in all three seeds. This is the
   approved configuration, and it shapes the training dynamics.
4. **Two selections are fragile:** seeds 42 and 2026 (§3).
5. **The notebook versions differ** in safeguards only; the trainer, configuration, data and command are the same.
   - Seed 42 (version 1) has no execution-identity record, so its run of record rests on its audit.
   - Seed 1337 (version 2) has no dataset-provenance record.
6. **The validation set:** it is synthetic and AI-reviewed (not gold), has 93 questions, and comes from the same
   distribution as the training data.

## 8. Decisions

**Decided 2026-11-12:**

| # | Decision | Outcome |
|---|---|---|
| D1 | Seed 2026's run of record | **Accepted:** execution `24187107`, `checkpoint-step-000040`, digest `cba9110b…` |
| D2 | Kaggle execution history | **Confirmed:** one seed 2026 training execution. No further Kaggle work, so N1 and L3 stay as documented in the audit. |
| D3 | Stage A | **Approved and run:** A1 beats A0 on the validation set, p = 0.0001 (§6) |
| D4 | Stage B | **Skipped;** no file deleted |
| D5 | The gold-set primary analysis | **Kept as drafted;** implemented and to be frozen before any gold label ([gold_set_v1_protocol.md](gold_set_v1_protocol.md) §3) |
| D6 | Unjudged A1 passages | **The methodology is kept,** with the downward bias stated explicitly (protocol §5). A positive result can support an improvement; a negative one is not evidence of no improvement. |
| D7 | The gold-set annotation | **The protocol is prepared** (query set, guidelines, blinded workflow). Annotation starts only when the protocol's §9 checklist is complete. |

**Decided 2026-11-12, after the protocol review:**

| # | Decision | Outcome |
|---|---|---|
| D8 | Passages A1 trained on | **Approved:** the primary gold analysis ranks only the eligible corpus. The 1,745 training and validation passages are removed before ranking, for every system (8,392 of 10,137 passages remain). The full corpus is a sensitivity analysis, labelled potentially contaminated. Implemented and frozen ([gold_set_v1_protocol.md](gold_set_v1_protocol.md) §4). |
| D9 | A0 | **Approved:** the original step-0 BGE checkpoint; A1 the three official checkpoints; the same pipeline and eligible corpus. Zero-shot BGE is a separate reference. |
| — | The metric definition | **Required and done:** gain 2^grade − 1 is authoritative, and every metric uses the project's kernels whatever is installed. The frozen decision pins that implementation (protocol §5). |

**The frozen analysis:** `gold_decision.py` `1f5d81b4…`. Annotation waits for your approval
([gold_set_v1_readiness.md](gold_set_v1_readiness.md)).
