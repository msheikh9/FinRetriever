# Blockers B1 and B3: the decision rule, the training configuration and the GPU rehearsal

**Status:** proposal, 2026-11-09. Nothing here is implemented. Every change below waits for your approval.
**Inputs, all frozen:**
- `train_candidates_v3` (678 candidates);
- `validation_pairs_v4` (93 questions);
- the fixed selection rule ([training_full_v1.md](training_full_v1.md) §3);
- blockers B1 and B3 of [final_training_readiness_audit.md](final_training_readiness_audit.md).

**What this work did:**
- **Analysis only.** It read the code and the frozen datasets.
- **Simulation in a scratch directory.** It simulated the selection procedure and decision rules on synthetic data. It
  read no validation, development or evaluation score.
- **Batching replay.** It replayed the trainer's batch planner on `train_candidates_v3`.
- **Prototype.** It prototyped an alternative planner in scratch.
- **Dry runs and checks.** It dry-ran the proposed configuration and ran the tests.

**What it did not do:**
- change code, configs, candidates, validation data or results;
- send API requests;
- train;
- run on a GPU;
- commit.

The training and validation questions are synthetic and were accepted by an AI review. They are not human-validated
and not gold labels.

## Summary and your decisions

| # | Decision | Recommendation |
|---|---|---|
| K1 | The A0-versus-A1 decision rule (B1) | Adopt the **selection-aware sign-flip test on every planned evaluation**, plus "every seed selects a checkpoint after step 0" (§1.5). It needs one trainer change: after the fixed rule stops, keep training and evaluating to the planned last step, and record those evaluations as never selectable (§1.6). **Which checkpoint is selected does not change.** |
| K1-alt | If K1's trainer change is declined | Decide on the **last-checkpoint test** instead (§1.7). It is valid but weaker. |
| K2 | Batching (the dropped single-pair batches) | A **balanced conflict-free planner**: 11 batches of 61–62 pairs per epoch, every pair in every epoch, nothing dropped, every protection kept (§3). It would be a new config value, `training.batching: balanced`. Existing configs keep today's planner. |
| K3 | The experiment config | `configs/train/biencoder_bge_base_ft_v3_a1.yaml` as in §2: the approved hyperparameters, with 55 optimizer steps per seed under K2. |
| K4 | The GPU rehearsal (B3) | A new **no-training rehearsal tool** with 10 checks (§4), run on Kaggle after `smoke_preflight`. It builds no optimizer step; any call to `optimizer.step` raises. |

Each change is approved separately. After approval comes implementation with tests, then the rehearsal, then the
training approval.

## 1. The decision rule (B1)

### 1.1 What the procedure does

**Per run (one seed):**
- The validation set is scored on whole-corpus MRR@10 at step 0, the base model (A0), and every 5 optimizer steps.
- The highest value wins, and ties at 12 decimal places keep the earlier step.
- Training stops after 3 candidate evaluations in a row without improvement.

**Under the configuration of §2:** the plan is 55 steps, so there are 12 candidates (0, 5, …, 55). The earliest
possible stop is at step 15.

**A1** is the procedure "fine-tune, then select". The comparison is made on the same 93 questions that did the
selecting.

### 1.2 Why the 2026-11-04 rule is not defensible

The rule from chat said "A1 beats A0" if all three of these hold:
1. all three seeds' selected checkpoints beat step 0;
2. the 95% interval of the mean paired difference excludes 0;
3. the last checkpoints point the same way.

**Condition 1 is half-automatic.** A selected checkpoint never scores below step 0, so condition 1 holds in 50% of
experiments with no true effect.

**Condition 2 inherits the selection.** The interval is for the maximum of up to 12 correlated evaluations, all on the
same questions. On its own it is "significant" in 16% of null experiments, against the 2.5% it suggests. Used as a 5%
test of the selected checkpoint, it is wrong 20–41% of the time.

**Together, the three conditions** give a 6–10% false-positive rate (table below). That is up to four times the level
the 95% interval implies, and it is only held down by condition 3.

### 1.3 How the alternatives were compared

**The simulation** is synthetic: `selection_sim.py` in the scratchpad. It uses only parameters already in the
2026-11-04 plan, and no data.

- **Questions:** 93. Each has a latent log-rank at step 0, shared by all seeds. Its MRR@10 value is 1/rank within the
  top 10, else 0. That gives the metric's discreteness, its ceiling and its floor.
- **Training paths:** each question's latent rank moves as a Brownian path over training time, so nearby checkpoints
  are strongly correlated.
  - Part of the path is shared by the seeds (ρ).
  - Questions differ in how much training moves them.
  - Under "no effect", a drift is calibrated so the expected MRR@10 equals step 0's at every step. The pure-noise
    variant is also run: with no drift, training noise hurts on average.
- **Settings:**

  | Setting | Value | Sensitivity runs |
  |---|---|---|
  | Base MRR@10 | 0.6 | 0.4, 0.8 |
  | Per-question SD of (end − step 0) | 0.2, from the plan's ±0.04 estimate for 93 questions | 0.1, 0.3 |
  | Seed-shared fraction ρ | 0.5 | 0, 0.9 |
  | Steps | 55 | 64 (today's planner) |
  | Checkpoints | correlated | independent |

- **Effects:**
  - "plateau": the gain is reached by 40% of training and kept;
  - "overfit": the gain peaks at 40% of training and halves by the end.

  Each is sized by its population MRR@10 gain at the peak.
- **The rule:** it is applied to every seed, with ties and patience. The vectorised version equals
  `checkpoint_selection.select` on 3,000 random sequences.
- **Runs:**
  - 2,000 simulated experiments per null scenario, so the Monte Carlo SE of a 5% rate is about 0.005;
  - 1,000 per alternative;
  - 999 sign flips each.

**The rules compared:**

| Rule | What it does |
|---|---|
| Naive | A one-sided paired t-test on the selected checkpoints. |
| 2026-11-04 | All three conditions above. |
| Sign-flip, recorded evaluations only | The audit's D7 as written. |
| **Proposed** | The same sign-flip test over every planned evaluation, plus "every seed selects a checkpoint after step 0". |
| Last checkpoint | A paired t-test on the checkpoint where the rule stopped. |
| Split half | Selection on 47 questions, test on the other 46. It needs a change to how selection works. |
| Bonferroni | The naive test at α/12. |
| Seed-minimum sign-flip | The smallest of the three seeds' gains as the statistic. |

### 1.4 Results

**False-positive rate under no effect** (target ≤ 0.05; 2,000 simulated experiments each, Monte Carlo SE ≈ 0.005):

| Scenario | Naive test on selected | 2026-11-04 rule | Sign-flip, recorded evaluations only | **Proposed rule** | Last checkpoint | Split half | Bonferroni | Seed-minimum sign-flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base (mean-zero null) | 0.285 | 0.103 | 0.071 | 0.049 | 0.029 | 0.036 | 0.032 | 0.050 |
| Pure-noise null | 0.233 | 0.085 | 0.058 | 0.034 | 0.018 | 0.035 | 0.022 | 0.041 |
| Per-question SD 0.1 | 0.249 | 0.058 | 0.065 | 0.037 | 0.027 | 0.033 | 0.009 | 0.047 |
| Per-question SD 0.3 | 0.272 | 0.097 | 0.067 | 0.049 | 0.032 | 0.040 | 0.036 | 0.051 |
| Base MRR@10 0.4 | 0.297 | 0.105 | 0.078 | 0.053 | 0.034 | 0.044 | 0.033 | 0.056 |
| Base MRR@10 0.8 | 0.266 | 0.091 | 0.060 | 0.043 | 0.034 | 0.038 | 0.025 | 0.040 |
| Seeds independent (ρ = 0) | 0.377 | 0.075 | 0.075 | 0.049 | 0.029 | 0.042 | 0.053 | 0.046 |
| Seeds 90% shared (ρ = 0.9) | 0.202 | 0.095 | 0.057 | 0.044 | 0.030 | 0.035 | 0.021 | 0.043 |
| Checkpoints independent | 0.409 | 0.079 | 0.068 | 0.046 | 0.009 | 0.048 | 0.061 | 0.048 |
| 64 steps (current planner) | 0.274 | 0.094 | 0.071 | 0.045 | 0.023 | 0.045 | 0.024 | 0.042 |

**Power** (1,000 simulated experiments each):

| Scenario | Naive test on selected | 2026-11-04 rule | Sign-flip, recorded evaluations only | **Proposed rule** | Last checkpoint | Split half | Bonferroni | Seed-minimum sign-flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plateau +0.02 | 0.749 | 0.512 | 0.465 | 0.406 | 0.295 | 0.203 | 0.298 | 0.387 |
| Plateau +0.04 | 0.977 | 0.915 | 0.912 | 0.870 | 0.759 | 0.577 | 0.778 | 0.853 |
| Plateau +0.06 | 0.998 | 0.994 | 0.989 | 0.984 | 0.970 | 0.866 | 0.970 | 0.983 |
| Plateau +0.08 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.962 | 0.999 | 1.000 |
| Overfit, peak +0.04 | 0.972 | 0.847 | 0.850 | 0.810 | 0.577 | 0.547 | 0.721 | 0.804 |
| Overfit, peak +0.06 | 0.999 | 0.961 | 0.991 | 0.985 | 0.866 | 0.813 | 0.962 | 0.981 |
| Overfit, peak +0.08 | 1.000 | 0.997 | 1.000 | 0.999 | 0.980 | 0.949 | 0.998 | 1.000 |

The naive test and the 2026-11-04 rule have high power only because they reject too often under no effect. Their
power is not comparable with the valid rules'.

**What selection does to the estimate** (MRR@10, mean over simulated experiments):

| Scenario | Selected − step 0 | True gain at the selected step | Selection optimism | Mean of the flip null | Selected − null mean | Rule-stop checkpoint − step 0 | All three seeds select a non-base checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base (mean-zero null) | +0.0122 | +0.0000 | +0.0122 | +0.0102 | +0.0020 | -0.0016 | 0.502 |
| Pure-noise null | +0.0111 | -0.0010 | +0.0121 | +0.0101 | +0.0010 | -0.0034 | 0.500 |
| Per-question SD 0.1 | +0.0063 | +0.0000 | +0.0063 | +0.0054 | +0.0009 | -0.0012 | 0.516 |
| Per-question SD 0.3 | +0.0176 | -0.0000 | +0.0176 | +0.0148 | +0.0028 | -0.0029 | 0.482 |
| Base MRR@10 0.4 | +0.0121 | +0.0000 | +0.0121 | +0.0099 | +0.0022 | -0.0014 | 0.495 |
| Base MRR@10 0.8 | +0.0120 | -0.0000 | +0.0120 | +0.0102 | +0.0018 | -0.0016 | 0.496 |
| Seeds independent (ρ = 0) | +0.0126 | +0.0000 | +0.0126 | +0.0102 | +0.0024 | -0.0012 | 0.466 |
| Seeds 90% shared (ρ = 0.9) | +0.0117 | +0.0000 | +0.0117 | +0.0102 | +0.0015 | -0.0021 | 0.565 |
| Checkpoints independent | +0.0147 | -0.0000 | +0.0147 | +0.0128 | +0.0019 | -0.0047 | 0.641 |
| 64 steps (current planner) | +0.0115 | -0.0000 | +0.0115 | +0.0098 | +0.0017 | -0.0019 | 0.500 |
| Plateau +0.02 | +0.0290 | +0.0155 | +0.0135 | +0.0114 | +0.0176 | +0.0171 | 0.829 |
| Plateau +0.04 | +0.0501 | +0.0371 | +0.0131 | +0.0123 | +0.0378 | +0.0382 | 0.974 |
| Plateau +0.06 | +0.0705 | +0.0580 | +0.0125 | +0.0132 | +0.0573 | +0.0580 | 0.998 |
| Plateau +0.08 | +0.0895 | +0.0780 | +0.0115 | +0.0139 | +0.0756 | +0.0772 | 1.000 |
| Overfit, peak +0.04 | +0.0452 | +0.0338 | +0.0115 | +0.0122 | +0.0330 | +0.0280 | 0.981 |
| Overfit, peak +0.06 | +0.0622 | +0.0534 | +0.0087 | +0.0128 | +0.0494 | +0.0420 | 1.000 |
| Overfit, peak +0.08 | +0.0804 | +0.0730 | +0.0074 | +0.0134 | +0.0670 | +0.0571 | 1.000 |

**What the results show:**
- **Selection optimism.** Repeated evaluation on the same 93 questions inflates the selected checkpoint's gain by about
  **+0.012 MRR@10**. It ranges from +0.006 to +0.018 as the per-question spread goes from 0.1 to 0.3, and is
  +0.007–0.014 when there is a real effect. With no effect at all, A1 still "gains" +0.012.
- **The D7 test on recorded evaluations only is anti-conservative: 5.7–7.8%.**
  - Under no effect, 93% of runs stop early, and the evaluations after the stop never exist.
  - Many flipped sequences "would" have gone on to a higher maximum, so the null distribution is cut short.
  - Lowering α to 0.025 brings it under 5%, but only by a calibration that depends on this simulation's model.
- **A conservative correction does not work either.** It counts every flipped sequence whose stop is unknown as
  extreme. That covers about 70% of flips, so the test rejects almost never: power is 0.05 at +0.04.
- **The same test with every planned evaluation recorded holds the level** (3.4–5.4%). It keeps most of the power:
  0.87 at a +0.04 plateau and 0.81 at a +0.04 peak that later declines.
- **The valid alternatives are weaker.**
  - Last-checkpoint: 0.76 and 0.58.
  - Split half: 0.58 and 0.55. It also needs selection on only half the questions.
  - Bonferroni: 0.78 and 0.72.
- **Requiring every seed to select a non-base checkpoint** costs almost nothing: 0.870 against 0.872.
- **The seed-minimum statistic** behaves like the mean but is slightly weaker (0.85).
- **A selection-adjusted estimate.** "Selected minus the mean of the flip null" is close to the true gain at the
  selected step: +0.038 against +0.037 at a +0.04 plateau, and +0.033 against +0.034 when overfitting. It is +0.001 to
  +0.003 under no effect.

### 1.5 The proposed rule, to be fixed in the repository before any training

**Inputs.** The three `selection.json` records of `bge_base_ft_v3_a1`, for seeds 42, 1337 and 2026, and nothing else.
No development or evaluation label is read.

**Preconditions.** If any of these fails, no decision is made and the failure is reported.
1. Each record verifies (`read_selection_record`), under the fixed `RULE`. The seeds are 42, 1337 and 2026, once each,
   with the same config SHA-256 and the same validation set (`validation_pairs_v4`, its manifest SHA-256).
2. Each record holds every planned evaluation. These are the candidates up to the rule's stop, then every later
   evaluation to the planned last step (§1.6).
3. Every evaluation scores the same 93 pair IDs. The per-question global `mrr@10` values average to the recorded value
   (within 1e-12).
4. **Step 0 is identical across the seeds** (N10): every question's global first relevant rank is the same.

**Statistic.** T = (1/3) Σ_s [V_s(k*_s) − V(0)], where:
- V is whole-corpus MRR@10 over the 93 questions;
- V(0) is the shared step-0 value (A0);
- k*_s is the step the fixed rule selected for seed s.

**Null distribution.**
1. Draw B = 10,000 sign vectors ε_b ∈ {−1, +1}^93 from `numpy.random.Generator(PCG64(20261109))`, with columns in
   sorted pair-ID order.
2. For each b and seed s, flip each question's differences from step 0 at once, across every evaluation and every
   seed: V*_{b,s}(k) = V(0) + (1/93) Σ_q ε_{b,q} [r_{s,q}(k) − r_q(0)].
3. Re-apply the fixed rule to each flipped curve over all recorded evaluations, with its candidates, 12-decimal ties to
   the earlier step and patience 3. Use `checkpoint_selection`'s own code, not a re-implementation.
4. T*_b = (1/3) Σ_s [V*_{b,s}(k*_{b,s}) − V(0)].

**The p-value** is p = (1 + #{b : T*_b ≥ T − 1e-12}) / (B + 1), one-sided.

**Decision.** "A1 beats A0 on `validation_pairs_v4`" if and only if:
- p ≤ 0.05; and
- every seed selected a checkpoint after step 0.

Otherwise the result is "no improvement shown". The rule never concludes that A0 beats A1.

**Reported, never deciding:**
- **T**, labelled as optimistic by construction.
- **The selection-adjusted estimate.** The null mean E*[T*] and T − E*[T*].
- **Per seed:** the selected step and value, the rule's stop step and value, and the value at the planned last step.
- **Seed consistency:** the seed-minimum statistic and its flip p-value.
- **Question-bootstrap 95% intervals** (10,000 resamples, a fixed seed):
  - for the last-step difference;
  - for T, labelled as not selection-adjusted.

  Both are conditional on these three runs.
- **Other metrics:** company-mode MRR@10, nDCG@10 and Recall@100 at the selected checkpoints, and per-company
  differences.
- **Development diagnostics,** reported separately: `dev_pilot_v3` in both modes, with judged@10 (10 AI-labelled
  questions; scores are lower bounds). They are never used for selection or the decision.

**Reproducibility.** The metric, mode, ties and patience come from `checkpoint_selection.RULE` itself. The flips, B,
the RNG seed, the tolerance and the bootstrap are fixed in code. The analysis writes a write-once decision record
holding its inputs' SHA-256, so rerunning it gives the same p-value.

### 1.6 The change K1 needs, and what it does not change

Today the trainer stops training at the rule's stop, so the later evaluations, which the valid test needs, never
exist. The proposal:

- **Trainer:**
  - After the rule stops, keep training to the planned last step, and evaluate at every remaining multiple of 5 and
    at the last step.
  - Record those evaluations as **post-stop**: never candidates, never selected.
  - The newest checkpoint keeps the model and `trainer_state.pt` for resuming. Earlier post-stop checkpoints keep only
    `metadata.json`, with a weights digest, to save disk.
- **`checkpoint_selection`:**
  - `SELECTION_FORMAT` 2 adds `post_stop_evaluations`.
  - `RULE` gains one entry: "after_stop: training and evaluation continue to the planned last step; those evaluations
    are never candidates".
  - `select`, the selected checkpoint, `read_selection_record` (which re-derives the selection from the candidate
    evaluations only) and `verify_selected_checkpoint` are otherwise unchanged.
  - No format-1 record of a real run exists.
- **A decision module and CLI** implement §1.5. They are tested on synthetic selection records, which is readiness
  check T7. The tests check:
  - the re-applied rule equals `select`;
  - the p-values reproduce;
  - each precondition refuses bad input.

**What this does not change:** the metric, the candidates, the tie-breaking, the patience or which checkpoint each
seed selects.

**What it costs:** about 5–8 more validation passes per seed (about 1–1.5 minutes each, estimated) and the remaining
training steps.

### 1.7 If K1's trainer change is declined

**Use the last-checkpoint test instead.** Statistic: (1/3) Σ_s [V_s(stop_s) − V(0)], where stop_s is the
checkpoint at which the rule stopped. It is a one-sided paired test at 0.05 (a sign-flip or t-test over questions),
plus the same preconditions 1, 3 and 4.
- **Valid:** 0.9–3.4% false positives in the simulation. It is biased against A1, because the stop comes three
  evaluations after the best.
- **Weaker:** power 0.76 at a +0.04 plateau and 0.58 when the model overfits.
- **No code change.**

The selected checkpoint is still the final model. The test only says whether fine-tuning, stopped by the rule, beats
A0.

### 1.8 Limitations of the proposed rule

1. **What it generalises over.** The inference covers the sampling of these 93 synthetic, AI-reviewed questions, given
   these three runs.
   - Three seeds are a consistency check (the non-base condition), not an estimate of seed variance. A seed-level
     interval would have 2 degrees of freedom.
   - Another set of three seeds could disagree.
2. **The sign-flip assumption.** The test treats each question's change as equally likely to be up or down under no
   effect. That is exact only for symmetric differences. In the simulation, mean-zero but asymmetric and heterogeneous
   changes kept the level at 3.4–5.4%, but the simulation is a model and the real correlations are unknown. The model
   affects the *estimated* error rates, not how the test is computed.
3. **Power.**
   - A +0.02 gain is detected about 40% of the time, so "no improvement shown" is not evidence of no effect.
   - A +0.04 gain is detected about 87% of the time.
4. **Validation is not the benchmark.** The validation questions come from the same generator (Qwen, same prompt) and
   the same filings as the training questions. Templates repeat: the top training–validation word overlap is 0.765,
   below the 0.80 threshold, on the same template for another year.
   - So a pass means "better on this synthetic distribution". It is not a retrieval improvement on real questions or
     on the benchmark, and it cannot be one until gold labels exist.
   - Relevance is the positive plus its exact equivalents, so other relevant passages count against both A0 and A1.
5. **One comparison.** The rule covers A1 against A0 only. A2 (hard negatives) would need its own written rule and a
   multiplicity plan, and `validation_pairs_v4` stays frozen (no v5).
6. **The environment.** N10 requires an identical step 0 across seeds. All three seeds must run on the same GPU type
   and software stack (T4, the bundled requirements), or precondition 4 fails and no decision is made.

## 2. The proposed training configuration (K3)

`configs/train/biencoder_bge_base_ft_v3_a1.yaml`, to be written once K2 is implemented. A scratch copy has been
dry-run (§7).

```yaml
name: bge_base_ft_v3_a1

base_model:
  name: BAAI/bge-base-en-v1.5
  revision: a5beb1e3e68b9ab74eb54cfd186867f64f240e1a   # the zero-shot baseline's pinned commit
  query_prefix: "Represent this sentence for searching relevant passages: "
  passage_prefix: ""
  pooling: cls

data:
  data_prep_config: configs/train/data_prep_v3.yaml
  candidates: train_candidates_v3   # 678 candidates: the 66 and 612 from P1

training:
  seeds: [42, 1337, 2026]           # one run per seed (--seed)
  batch_size: 64                    # at most 64 pairs per batch; balanced batching gives 61–62
  batching: balanced                # NEW (K2): equal-size conflict-free batches; none dropped
  mini_batch_size: 16               # GradCache: texts per forward pass (memory, not negatives)
  gradient_accumulation_steps: 1
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1                 # 6 of 55 steps
  epochs: 5                         # 55 optimizer steps per seed
  temperature: 0.02                 # BGE's fine-tuning setting (scale 50)
  query_max_length: 128
  passage_max_length: 512           # the preparation's visible-text limit
  max_grad_norm: 1.0
  fp16: true                        # T4 has no bf16
  gradient_checkpointing: false     # GradCache bounds memory by the mini-batch
  device: cuda

checkpoints:
  output_dir: models/bge_base_ft_v3_a1   # gitignored; one directory per seed, never written over

validation:
  version: validation_pairs_v4      # 93 AI-reviewed synthetic questions (not gold); checkpoint selection only
  batch_size: 64
```

**Unchanged from the approved settings:**
- the base model and revision;
- the learning rate (2e-5), AdamW with weight decay 0.01, 10% linear warmup then linear decay;
- 5 epochs, temperature 0.02, fp16 and gradient clipping at 1.0;
- the 128/512 token limits;
- batch 64 with GradCache mini-batch 16;
- the three seeds and the validation set.

**What K2 changes in the schedule:**

| | Today's planner | Balanced (K2) |
|---|---|---|
| Steps per seed | 64, 59, 62 | 55 for every seed |
| Warmup | 7, 6, 7 | 6 |
| Evaluations | 14, 13, 14 (steps 0, 5, …, plus a final non-candidate step) | 12, at steps 0, 5, …, 55, all candidates |

**Throughput estimate** (to be measured by the rehearsal): about 2.3 s per GradCache step and 1–1.5 minutes per
validation pass. With K1, that is about 20 minutes per seed and about 1 GPU hour for three seeds, plus setup. Disk is
about 438 MB per kept checkpoint, plus 871 MB of optimizer state in the newest.

## 3. Single-pair batches (K2)

### 3.1 What happens today

`plan_batches` packs pairs first-fit, taking them in a hashed order. Each pair goes into the first open batch it does
not conflict with, or opens a new one.

**The conflict graph of the 678 candidates:**
- 913 edges: pairs that may not share a batch, because one's positive is excluded as the other's negative;
- 102 pairs have no conflict, and the highest degree is 19;
- **the largest clique is 10.** It is made of Walmart cash-flow-statement questions asked of several fiscal years'
  filings. Others are Apple share repurchases (9) and Walmart's opioid MDL (8).

**Every epoch of every seed** has the same shape: 10 full batches of 64, one of 28–36, and then **a tail of 1 to 4
batches of 1–6 pairs**. The tail holds the leftover members of those cliques.

| Seed | Steps | Single-pair batches dropped | Batches of 2–7 pairs | Pairs not trained in some epoch | Largest/smallest per-pair weight in a step |
|---:|---:|---:|---:|---:|---:|
| 42 | 64 | 7 | 9 | 7 | 32× |
| 1337 | 59 | 5 | 4 | 5 | 21× |
| 2026 | 62 | 1 | 7 | 1 | 32× |

**The dropped pairs.** 13 pair-epochs in all (12 distinct pairs) are dropped. Every one is highly conflicted (degree
7–19), and one pair is dropped twice.

**A single-pair batch is useless as a step.** With no negative, its in-batch loss is log 1 = 0, so dropping it is
correct. But its pair goes untrained that epoch.

**The small batches are the larger problem.**
- The loss is the batch mean, so a pair in a batch of 2 gets 32 times the gradient weight of a pair in a batch of 64.
- AdamW then takes a full-size step in the direction of those 2–3 pairs, with only 1–5 negatives.
- These are the same clique members: repeated statements of one table across years.

### 3.2 The options

| Option | Every pair every epoch | Batch sizes | Protections | Code change | Assessment |
|---|---|---|---|---|---|
| Keep today's planner | No: 13 pair-epochs dropped | 1–64 | Kept | None | Biases the schedule toward a few repeated tables and gives each seed a different step count |
| A smaller final batch | Not by itself | Still 2–6 at the tail | Kept | Small | Does not apply: the tail is 2–4 conflict-forced batches, not one remainder |
| **Balanced conflict-free planner** | **Yes: 3,390 of 3,390 pair-epochs** | **61–62** | **Kept: the same conflict relation and `check_batch`** | A planner plus one config value | **Recommended** |
| Full batches with conflicting logits masked | Yes | 58–62 | Kept by masking 68–95 pairs per epoch | Changes the loss and GradCache's input; the batch check becomes a mask check | Correct, but it changes the objective and the verified GradCache path. Not needed. |

### 3.3 The recommended planner

**The algorithm:**
1. K = ⌈N / batch_size⌉ batches: 11 for 678 pairs.
2. The capacity is ⌈N / K⌉ = 62.
3. Pairs are taken most-constrained first: by conflict degree, descending, with ties broken by
   SHA-256(`"{seed}:{pair_id}"`).
4. Each pair goes into the **smallest** open batch it does not conflict with, the lower index on ties.
5. If some pair fits nowhere, K rises by one and the planner starts again. That never happened here.
6. The batches are then ordered by SHA-256(`"{seed}:"` + their sorted pair IDs).
7. The trainer seeds each epoch with seed + epoch, as today.

**The scratch prototype** on `train_candidates_v3`, for all three seeds × 5 epochs:
- **11 batches of 61–62** in every epoch: 35 of 62 and 20 of 61 per seed;
- every batch passes the trainer's own `check_batch`;
- 0 dropped, and all 678 pairs trained in all 5 epochs;
- at least 60 in-batch negatives per query, 60.6 on average, against 61.0–61.2 today;
- the per-pair weight ratio is 1.0;
- batch-mates repeat between epochs 8.9% of the time, against 9.0% today, which is what random batches of 62 give;
- deterministic: replanning gives the same schedule digest.

**Compatibility.**
- `training.batching` defaults to `first_fit`, today's planner. So the smoke and rehearsal configs keep their schedules
  and their `run.json` byte-for-byte.
- `balanced` changes the schedule hash and is recorded in `run.json` and the checkpoint metadata.
- The dry run should print the real per-batch negatives (minimum and mean). Today it prints the nominal
  "63 in-batch negatives per query".

## 4. The GPU rehearsal (B3, K4): no optimizer training

### 4.1 What `smoke_preflight` already checks, and what it does not

**Already checked** (pass/fail on Kaggle):
- the bundle, config, corpus, candidates, provenance, schedule and validation set;
- evaluation isolation, the model and tokenizer;
- `cuda_fp16`, a toy autocast-and-GradScaler test;
- `gradcache`: 4 real pairs, mini-batch 2, fp16 with a fixed ×1024 scale, eager attention, dropout on;
- `gpu_memory`, from one 16-pair cached step plus AdamW's estimated state.

**Not checked:**
- a full 64-pair step at the longest lengths;
- the GradScaler at its real initial scale (2^16), where an overflow skips a step;
- attention dropout on its own;
- the full-corpus validation pass (time, memory, side effects, determinism);
- checkpoint writing and reloading on the GPU;
- a guarantee that nothing was trained.

### 4.2 The rehearsal tool

The tool is `scripts/gpu_rehearsal.py` with module `training/gpu_rehearsal.py`, added to the bundle's script list.
- **It runs the bundled code** on the experiment config, after `smoke_preflight` passes.
- **No optimizer step is taken.** AdamW is built only to save and reload its (empty) state. Its `step` is replaced by a
  function that raises. Gradients are cleared after each check.
- **No retrieval value is printed.** The report goes to `rehearsal_outputs/rehearsal.json`.

| # | Check | Method | Passes if |
|---|---|---|---|
| G1 | Environment | GPU name, compute capability, memory, CUDA, torch and transformers versions. | A T4 (compute 7.5), CUDA available, fp16 autocast works. Recorded for N10. |
| G2 | Real BGE with dropout | The pinned model, loaded as for GradCache: eager attention. Hidden and attention dropout probabilities read from the model, then `train()`. Two forward passes of the same 8 real texts from different random states, and two from the same state. | Eager attention and both dropout probabilities 0.1. Different states give different embeddings (relative difference > 1e-3). The same state gives identical embeddings (fp32: 0; fp16: ≤ 1e-2). |
| G3 | Attention-dropout replay | As G2, with the hidden-dropout modules set to p = 0 for the check only (restored and verified afterwards), so only attention dropout acts. | Different states differ (attention dropout is active); the same state replays exactly. |
| G4 | GradCache equivalence, fp32 on CUDA | `gradcache_check` on the 8 pairs with the longest positives (512 tokens), mini-batch 4, against the uncached passes from the same random state. | Relative gradient difference ≤ 1e-5, cosine ≥ 0.999999, equal losses, replay difference ≤ 1e-5. |
| G5 | GradCache equivalence, fp16 | As G4 under autocast, with the loss scaled by the GradScaler's real initial scale (65,536) and by 1,024. | Relative ≤ 1e-3, cosine ≥ 0.999 and replay ≤ 1e-2 at a scale whose gradients are finite. An overflow is reported, and handled by G6. |
| G6 | fp16 loss scaling | The full first batch of each seed's schedule (61–62 pairs, mini-batch 16), cached backward, at scales 2^16, 2^15, …, 2^10. Each time it checks whether every scaled gradient is finite, as `GradScaler`'s check does. The unscaled fp16 gradient is compared with the fp32 GradCache gradient of the same batch from the same random state. | It reports the predicted skipped steps: the halvings from 2^16 to the first finite scale. The scale never grows within 55 steps (growth interval 2,000). Passes if at most 2 steps would be skipped and fp16 against fp32 has cosine ≥ 0.99. Otherwise it warns, with a proposed fixed initial scale for your decision. |
| G7 | Peak GPU memory | One full cached step on the worst-case batch: the 62 longest positives at 512 tokens and the longest queries, fp16. Two fp32 tensors per parameter are allocated as AdamW's moments (0.82 GiB). `max_memory_allocated` and `max_memory_reserved` are read, and the step is timed. | Peak reserved ≤ 85% of total memory. The measured step time gives the training-time estimate. |
| G8 | Full-corpus validation | `validation_eval.evaluate` at step 0 on `validation_pairs_v4` over all 10,137 passages, with the config's settings (fp16, batch 64), run twice. Also fp16 against fp32 embeddings on a 512-passage slice. | Each pass is timed and its memory recorded. No side effects: same weights digest before and after, training mode restored, CPU and CUDA random states unchanged. The two passes give identical per-question first relevant ranks. The values are not printed: their SHA-256 and the full result are stored for N10. |
| G9 | Checkpoint save and resume | `_save_checkpoint` writes `checkpoint-step-000000` into a scratch run directory, with the base weights, an un-stepped AdamW, the scheduler, the GradScaler, the random states and a selection state holding the step-0 evaluation. Then: re-hash the files; reload the weights into a fresh base encoder; load through the benchmark's `load_checkpoint_encoder`; reload `trainer_state.pt`; save the same step again. | `checkpoint_files` equals the recorded digest. Every weight tensor is bitwise equal. Embeddings of 8 texts match in fp32 eval (≤ 1e-6). The optimizer, scheduler, scaler and selection states round-trip. Restoring the random state reproduces a dropout forward pass bitwise. A second save of the same step is refused, and no `.tmp` directory remains. Checkpoint size is recorded. |
| G10 | Nothing trained | `optimizer.step` raises if called. The final weights digest is compared with the digest at load. The files written are listed. | No step was called. The final weights equal the base weights. Files exist only in `rehearsal_outputs/` and the scratch run directory. |

**Run time:** about 10–15 minutes of T4 time, estimated: model load, G6's 8 cached steps, and G8's two full passes.
**Tests:** the CPU parts are unit-tested with a tiny BERT. CUDA-only assertions are skipped off-GPU, as the existing
preflight tests do.

### 4.3 What the rehearsal cannot show

Anything that needs real updates stays untested until the first training run:
- the interplay of AdamW, the scheduler and the scaler over steps;
- memory with a real AdamW state;
- resuming mid-training on CUDA, which is close, not bit-identical;
- loss behaviour.

Seed 42 of the experiment is therefore the first end-to-end CUDA run. Its log, `smoke_results` and `selection.json`
should be reviewed before seeds 1337 and 2026 start.

## 5. Remaining technical risks

1. **Statistical power.** About 40% at +0.02 MRR@10 and 87% at +0.04. A null result does not show that fine-tuning
   fails.
2. **Model dependence of the error rates.** The simulation's latent-rank model is a stand-in for the real per-question
   changes. The rule's computation does not depend on it, but its estimated 3.4–5.4% level does.
3. **One generator, the same filings.** Validation measures within-filing generalisation on Qwen-style questions
   (§1.8, item 4). Template learning can raise A1 on validation without helping real questions.
4. **fp16 overflow at the initial scale.** If G6 predicts more than 2 skipped steps, the initial scale needs your
   decision. `LambdaLR` also steps on a skipped step, which shifts warmup by one step per skip.
5. **N10 across sessions.** A different GPU type or library version changes step 0 and blocks the decision
   (precondition 4). Pin the T4, and install from the bundle's requirements.
6. **Kaggle limits.** With K1 the runs are a little longer. Keep one seed per session or download between seeds
   (readiness N9). Disk use is checked by G9.
7. **Concentrated data.** The 10-clique and its relatives are a few tables repeated across years. Balanced batching
   stops them skewing steps, but they remain heavily weighted per epoch. Apple and Amazon remain underweight (the
   build report).
8. **AI labels throughout.** No result can be described as human-validated (the readiness audit, D5).
9. **The code changes (K1, K2, K4).**
   - Each needs tests that keep today's records and schedules byte-for-byte for configs that do not opt in.
   - The fixed `RULE` gains only the after-stop entry, and selection is unchanged.

## 6. Commands for the next stage

**After your approval of K1, K2 and K4: implement, test, write the config.**

```bash
export HF_HUB_OFFLINE=1
PY=python   # the environment of requirements/lock-*.txt (docs/environment.md)
$PY -m pytest                                    # the full suite, plus the new tests
python3 -m ruff check src tests scripts && git diff --check
$PY scripts/05_train_biencoder.py --config configs/train/biencoder_bge_base_ft_v3_a1.yaml   # dry run: 55 steps per seed
python3 scripts/smoke_bundle.py build --config configs/train/biencoder_bge_base_ft_v3_a1.yaml \
    --name finretriever_train_v3_a1             # writes dist/kaggle/finretriever_train_v3_a1.tar.gz and .sha256
```

**On Kaggle** (T4, internet on, the private dataset). Set up as in [kaggle_training.md](kaggle_training.md) §3, then:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
python scripts/smoke_preflight.py --config configs/train/biencoder_bge_base_ft_v3_a1.yaml    # must end READY FOR TRAINING
python scripts/gpu_rehearsal.py --config configs/train/biencoder_bge_base_ft_v3_a1.yaml \
    --out rehearsal_outputs 2>&1 | tee rehearsal_outputs/console.log                       # trains nothing
```

**Then stop.** Download `rehearsal_outputs/` and `smoke_outputs/preflight.json` for review.

**Only after a separate training approval,** one seed per session, seed 42 first and reviewed:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_train_biencoder.py \
    --config configs/train/biencoder_bge_base_ft_v3_a1.yaml --seed 42 --confirm-training 2>&1 | tee smoke_outputs/train_seed42.log
python scripts/smoke_results.py --run-dir models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed42 \
    --console-log smoke_outputs/train_seed42.log --out-dir smoke_outputs/seed42
```

**After all three seeds, locally.** The decision, from the three selection records only (the proposed script name):

```bash
python scripts/07_decide_a1_vs_a0.py \
    --selection models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed42/selection.json \
    --selection models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed1337/selection.json \
    --selection models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed2026/selection.json \
    --out results/decisions/bge_base_ft_v3_a1_vs_a0.json        # write-once
```

## 7. Checks run for this proposal

| Check | Result |
|---|---|
| `pytest`, the full suite (dense venv) | **1,267 passed, 25 skipped**, 0 failed. The skips are environmental: the Apple 10-K fixture and `pytrec_eval`. |
| Targeted: `test_checkpoint_selection`, `test_gradcache`, `test_training_data`, `test_biencoder_trainer`, `test_smoke_bundle`, `test_final_models`, `test_semantic_screen` | 93 passed |
| `ruff check src tests scripts`; `git diff --check` | Clean; no tracked file changed |
| Selection rule, vectorised against `checkpoint_selection.select` | Equal on 3,000 random sequences with ties, patience and final steps |
| Simulation | 10 null scenarios × 2,000 and 7 alternatives × 1,000, each with 999 flips. Two runs with the same seeds agree exactly. |
| The batching replay of `plan_batches` (3 seeds × 5 epochs) | Reproduces the dry run's 64/59/62 steps and 7/5/1 drops |
| The balanced-planner prototype | 11 × 61–62 in all 15 epoch plans, every batch through `check_batch`, deterministic |
| Dry run of the proposed config without `batching` (scratch output directory) | Passes: 678 candidates, all seeds planned, `validation_pairs_v4` isolated, nothing written |
| Dry run of the proposed config with `batching` | Refused today ("extra inputs are not permitted"), as expected until K2 is implemented |

The scratch files are in
`<session scratch directory>/b1b3/`:
- the simulation: `selection_sim.py` and `sim_final.json`;
- the batching analysis: `batching_analysis.py` and `batching.json`;
- the prototype: `balanced_prototype.py` and `balanced.json`;
- the two config variants.

No validation, development or evaluation score was computed or read.
