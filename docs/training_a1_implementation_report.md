# K1–K4 implementation: the decision rule, balanced batching, the A1 configuration and the GPU rehearsal

**Status:** implemented and verified on CPU, 2026-11-09. Your approval of K1–K4 is in
[training_b1_b3_proposal.md](training_b1_b3_proposal.md).

**Not done:**
- no GPU rehearsal was run;
- no optimizer step was taken and no model was trained;
- no API request was sent;
- nothing was committed or pushed.

**Unchanged:** `train_candidates_v3`, `validation_pairs_v4`, every generation record and decision, the historical
results and the smoke checkpoints are byte-for-byte unchanged: nothing under `data/` changed (§6).

The training and validation questions are synthetic and accepted by an AI review. They are not human-validated and
not gold labels.

## Summary

| | What was built | Verified |
|---|---|---|
| K1 | The trainer evaluates every planned checkpoint to the planned end; after the rule's stop those evaluations are recorded and marked never selectable. The selection record is format 2. `decision_rule` implements the approved sign-flip test and decision. `decision_simulation` runs the implemented rule on synthetic data. | **The selected checkpoint is unchanged**, even when a later value is much higher. **False positives under no effect: 5.05% pooled** over 20,000 simulated experiments, and 3.7–5.7% per scenario. |
| K2 | `training.batching: balanced`, a conflict-free planner of near-equal batches; first-fit stays the default | On `train_candidates_v3`, every seed and epoch has **11 batches of 61–62**: all 678 pairs once, 0 dropped, every batch through the trainer's `check_batch`. The smoke run's `run.json` is reproduced byte-for-byte. |
| K3 | `configs/train/biencoder_bge_base_ft_v3_a1.yaml` | Its values equal the approved proposal. The dry run gives **55 steps per seed** (warmup 6) and ≥ 60 in-batch negatives per query. |
| K4 | `gpu_rehearsal`, ten checks, plus `scripts/gpu_rehearsal.py`, shipped in the bundle | Every `optimizer.step` and `GradScaler.step` raises. The final weights must equal the starting ones. An fp16 overflow stops the rehearsal. The CPU-runnable checks pass from a real extracted bundle. |

Tests: **1,301 passed, 25 skipped** (dense venv; 34 new). Ruff and `git diff --check` are clean.

## 1. K1: the selection-aware decision rule

### 1.1 What changed in training and selection

**The trainer** (`biencoder_trainer.train`):
- When the fixed rule stops (3 candidate evaluations without improvement), training **does not stop**. It continues
  to the planned last step, 55 under K3.
- Every remaining evaluation step (every 5 steps and the last) is evaluated and recorded through
  `SelectionState.record_after_stop`, as `candidate: false`, `selectable: false`, `after_stop: true`.
- Each such checkpoint is written and hashed like the others. Its metadata says "after the rule's stop … never
  selectable".
- The console reports the stop once: "the rule stops selecting at step N … step S stays selected".
- Resuming continues the after-stop evaluations (tested).

**The selection** is unchanged:
- **Unchanged:** the candidates, the metric, the 12-decimal ties, the patience and therefore the selected step.
- **Tested:**
  - a run with 0.99 at step 25, after the rule stopped at step 20, still selects step 5;
  - `verify_selected_checkpoint` refuses any after-stop checkpoint.

**The record** (`selection.json`, format 2):
- **Holds:**
  - `evaluations` (up to the rule's stop) and `after_stop_evaluations`;
  - `rule_stop`;
  - the run's `environment`: GPU name and compute capability, CUDA, cuDNN and library versions.
- **Checked on reading** (`read_selection_record`):
  - the recorded steps are every planned evaluation step;
  - the after-stop entries are marked never selectable;
  - the rule re-applied to every evaluation (`select_through`) stops and selects where the record says.
- **Refused** (tested): a missing evaluation, an after-stop entry marked selectable or as a candidate, a moved
  entry, a forged rule stop or digest.
- `RULE` gains an `after_stop` entry, and its early-stopping text now says what stops: selection.

**Overflow.** A validated run stops at the first step whose gradients are not finite, before the optimizer step. The
GradScaler is never left to skip the step and lower its scale, and the message says no scaler setting changed. Each
step logs its loss scale. This is tested with an encoder that diverges at step 7: steps 1–6 are logged, step 7 is
never taken, and no NaN reaches the weights. Configs without validation keep their behaviour.

### 1.2 The rule, as implemented (`decision_rule`, `scripts/07_decide_a1_vs_a0.py`)

**Inputs.** The three `selection.json` files only, each verified on reading. A path under `data/eval` is refused.

**Preconditions.** If any fails, there is no decision and every problem is listed:
1. Every record names the same seed list, and each seed appears once.
2. The records share the config SHA-256, the validation set and the rule.
3. Every planned evaluation is present.
4. Every evaluation scores the same questions, and the per-question MRR@10 values average to the recorded value.
5. Step 0 is identical across seeds: each question's first relevant rank and MRR@10 (N10).
6. **The same GPU type and software** (`ENVIRONMENT_IDENTITY`: python, torch, transformers, tokenizers, CUDA,
   cuDNN, device, GPU name, compute capability).

**Statistic.** T = (1/3) Σ_s [V_s(k*_s) − V(0)].

**Null distribution.**
- 10,000 sign vectors from `numpy` PCG64(20261109), over the questions in sorted pair-ID order.
- Each flip reverses a question's differences from step 0 jointly across every evaluation and seed.
- The fixed rule is re-applied to every flipped curve through `checkpoint_selection.select_through`, the module's
  own `SelectionState` code.

**The p-value** is one-sided: p = (1 + #{T* ≥ T − 1e-12}) / 10,001.

**Decision.** "A1 beats A0 on the validation set" if and only if p ≤ 0.05 **and** every seed selected a checkpoint
after step 0. Otherwise "no improvement shown". The rule never concludes that A0 is better.

**Reported, never deciding:**
- T;
- the null mean and the selection-adjusted estimate T − E*[T*];
- per seed: the selected, rule-stop and last steps and values;
- the seed-minimum statistic and its p-value;
- question-bootstrap 95% intervals (10,000 resamples, PCG64(20261110)), labelled not selection-adjusted;
- company-mode and other metrics at the selected checkpoints and at step 0.

**The record** is written once, with each input's SHA-256, so the same records give the same p-value (tested).

### 1.3 Null-effect and power simulations of the implemented rule

`decision_simulation` keeps the approved model ([training_b1_b3_proposal.md](training_b1_b3_proposal.md) §1.3) and
runs every simulated experiment through the implemented code:
- `select_through` selects each seed's checkpoint over every planned evaluation, as the trainer now records them;
- `decision_rule.decide` runs the test and the decision.

The simulations use 999 flips per test, and the rule itself uses 10,000. Fixed seeds:
`python scripts/simulate_decision_rule.py --out <file> --workers 8` reproduces these numbers (201 s here).

| Scenario | Experiments | A1 decided (rate) | Monte Carlo SE | p ≤ 0.05 | Every seed after step 0 | Selected − step 0 | Selection optimism | Selection-adjusted estimate | Rule stopped early |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| null: base | 2,000 | **0.054** | 0.005 | 0.054 | 0.493 | +0.0121 | +0.0121 | -0.0001 | 0.95 |
| null: pure noise | 2,000 | **0.037** | 0.004 | 0.037 | 0.465 | +0.0108 | +0.0114 | -0.0014 | 0.97 |
| null: per-question SD 0.1 | 2,000 | **0.051** | 0.005 | 0.052 | 0.520 | +0.0064 | +0.0064 | +0.0001 | 0.96 |
| null: per-question SD 0.3 | 2,000 | **0.057** | 0.005 | 0.057 | 0.478 | +0.0178 | +0.0178 | -0.0001 | 0.95 |
| null: base MRR@10 0.4 | 2,000 | **0.051** | 0.005 | 0.052 | 0.489 | +0.0116 | +0.0116 | -0.0004 | 0.95 |
| null: base MRR@10 0.8 | 2,000 | **0.052** | 0.005 | 0.052 | 0.507 | +0.0124 | +0.0124 | +0.0002 | 0.95 |
| null: seeds independent | 2,000 | **0.047** | 0.005 | 0.049 | 0.448 | +0.0121 | +0.0121 | -0.0000 | 0.95 |
| null: seeds 90% shared | 2,000 | **0.054** | 0.005 | 0.054 | 0.575 | +0.0122 | +0.0122 | +0.0000 | 0.95 |
| null: checkpoints independent | 2,000 | **0.055** | 0.005 | 0.056 | 0.621 | +0.0146 | +0.0146 | -0.0002 | 0.97 |
| null: 64 steps | 2,000 | **0.046** | 0.005 | 0.046 | 0.518 | +0.0117 | +0.0117 | +0.0000 | 0.97 |
| plateau +0.02 | 1,000 | **0.412** | 0.016 | 0.416 | 0.833 | +0.0292 | +0.0136 | +0.0169 | 0.87 |
| plateau +0.04 | 1,000 | **0.852** | 0.011 | 0.856 | 0.966 | +0.0490 | +0.0120 | +0.0362 | 0.80 |
| plateau +0.06 | 1,000 | **0.988** | 0.003 | 0.988 | 0.999 | +0.0689 | +0.0108 | +0.0555 | 0.79 |
| overfit +0.04 | 1,000 | **0.827** | 0.012 | 0.828 | 0.976 | +0.0453 | +0.0116 | +0.0326 | 0.96 |
| overfit +0.06 | 1,000 | **0.980** | 0.004 | 0.980 | 0.998 | +0.0620 | +0.0084 | +0.0488 | 0.98 |

**What the simulations show:**
- **False-positive rate.**
  - Pooled over the 10 null scenarios (20,000 experiments): **5.05% (SE 0.15 points)**.
  - Per scenario: 3.7–5.7%.
  - Seven of the ten scenarios are above 5%, by 0.1–0.7 points, each within 1.5 SE. The pooled rate is consistent
    with the nominal level. The small excesses fit the sign flip being approximate, not exact, when changes are
    mean-zero but asymmetric per question.
  - Under the conservative pure-noise null, where training noise hurts on average, it is 3.7%.
- **Selection optimism.** Selecting on the same 93 questions adds about +0.012 MRR@10, ranging from +0.006 to
  +0.018 with the per-question spread.
- **The selection-adjusted estimate.**
  - Under no effect, it is centred on 0 (−0.0014 to +0.0002).
  - With an effect, it is slightly below the true gain at the selected step, which is conservative.
- **Power.** 0.41 at +0.02, **0.85** at +0.04 and 0.99 at +0.06 for a plateau; 0.83 and 0.98 when the model
  overfits. So "no improvement shown" is weak evidence against effects below about +0.03.
- **The consistency condition** (every seed after step 0) holds in about half of null experiments and in ≥ 0.97 at
  +0.04. It costs almost no power.
- **Agreement with the proposal's estimate.** The proposal's independent prototype gave 3.4–5.4% and 0.87/0.81. The
  implemented rule agrees within Monte Carlo error.

### 1.4 Assumptions and limitations (also in the module docstrings)

1. **A model, not the real runs.** The simulation describes how per-question ranks move with a latent-rank model, a
   Brownian path over training, a seed-shared fraction and heterogeneous questions. The real correlations are
   unknown. The rule's computation does not depend on the model; its estimated error rates do. The scenarios bracket
   the unknowns.
2. **The sign-flip assumption.** The test is exact for symmetric per-question differences under no effect, and
   approximate otherwise (§1.3).
3. **What it generalises over.** It covers the sampling of these 93 synthetic, AI-reviewed questions, given these
   three runs. Three seeds are a consistency check, not an estimate of seed variance.
4. **What a pass means.** It means "better on this synthetic validation distribution": same generator, same filings.
   It is not a retrieval improvement on real questions or the benchmark, and it cannot be one until gold labels
   exist.
5. **One comparison.** The rule covers A1 against A0. Another ablation needs its own written rule.

### 1.5 Tests

**`test_checkpoint_selection.py`** (20 tests, 4 new, 1 rewritten). The run trains past the rule's stop and keeps the
selection, and it records and marks the later evaluations. The other tests cover:
- `select_through`;
- the record's completeness and tamper checks;
- resuming after the stop;
- the stop on non-finite gradients.

**`test_decision_rule.py`** (14 tests) uses records built and written through the real `SelectionState` and
`make_selection_record`. It covers:
- **Decisions:** a clear gain is decided; no effect is not; a seed that keeps the base model blocks the decision even
  at p ≤ 0.05.
- **Reproducibility:** the p-value does not depend on argument order.
- **The null:** the identity and mirrored flips re-apply the rule exactly as an independent hand implementation does.
- **Preconditions:** each one refuses: a different step 0, GPU, missing environment, validation set or seed list, a
  missing per-question result, an incomplete record.
- **The command:** the record is written once with 10,000 flips and PCG64(20261109).
- **Reads only the selection records:** a spy on `open`; no torch import.
- **The simulation:** reproducible and calibrated (no effect on average; a +0.06 plateau reaches +0.06).

## 2. K2: balanced batching

**The implementation:**
- `false_negatives.balanced_conflict_free_batches`, as proposed:
  - K = ⌈N / batch_size⌉ batches of at most ⌈N / K⌉ pairs;
  - pairs taken most-constrained first, ties broken by SHA-256(`"{seed}:{pair_id}"`);
  - each pair into the smallest batch it does not conflict with;
  - K grows if needed;
  - the batches are ordered by SHA-256 of the seed and their members.
- It uses the same conflict relation as the first-fit planner: the guard's positives and exclusions, so equivalents,
  uncertain equivalents, overlap groups and reuse links are all respected.
- `training_data.plan_batches(method=...)` still verifies every batch pair by pair. `plan_steps` still checks every
  batch with `check_batch`, and the trainer checks it again before each forward pass.
- **Config.** `training.batching: first_fit | balanced`. Unset means first-fit, which every historical config uses,
  and is left out of `run.json`. When set, the run record and checkpoint metadata record the planner and the measured
  in-batch negatives.
- **The dry run** now prints the real negatives (minimum and mean) instead of the nominal "63".

**On `train_candidates_v3`, through the trainer's own `plan_steps`:**

| Seed | Steps | Warmup | Batches per epoch | Sizes (55 batches) | Pairs per epoch | Dropped | In-batch negatives per query |
|---:|---:|---:|---|---|---|---:|---|
| 42 | 55 | 6 | 11, 11, 11, 11, 11 | 35 × 62, 20 × 61 | all 678, once | 0 | at least 60, 60.64 on average |
| 1337 | 55 | 6 | 11, 11, 11, 11, 11 | 35 × 62, 20 × 61 | all 678, once | 0 | at least 60, 60.64 on average |
| 2026 | 55 | 6 | 11, 11, 11, 11, 11 | 35 × 62, 20 × 61 | all 678, once | 0 | at least 60, 60.64 on average |

Every batch passed `check_batch`. Today's planner, by comparison, gives 64/59/62 steps, 7/5/1 single-pair batches
dropped and batches as small as 2.

**Tests.** The planner:
- keeps every conflict apart (the guard's fixture, all 25 seeds);
- fills ⌈N / B⌉ equal batches;
- grows to 5 batches for a 5-clique;
- is deterministic whatever the input order;
- refuses an unknown method.

In the trainer:
- every pair trains in every epoch;
- the three seeds get the same step count;
- configs without `batching` keep their schedule and records;
- the dry run shows the planner and the real negatives;
- `batching: masked` is refused.

## 3. K3: the A1 configuration

`configs/train/biencoder_bge_base_ft_v3_a1.yaml`:
- the pinned BGE-base `a5beb1e3…`;
- `train_candidates_v3`, `validation_pairs_v4`;
- seeds 42, 1337, 2026;
- batch size 64 with `batching: balanced`, GradCache mini-batch 16;
- learning rate 2e-5, weight decay 0.01, warmup 0.1, 5 epochs, temperature 0.02;
- 128/512 tokens, gradient clipping 1.0, fp16, CUDA;
- output `models/bge_base_ft_v3_a1`.

**The values are identical to the approved proposal.** Against the mechanics-rehearsal config, only the name, the
candidates, `batching` and the output directory differ: no hyperparameter changed.

**Dry run** (`scripts/05_train_biencoder.py --config …`):
- 678 candidates;
- 55 optimizer steps per seed (warmup 6);
- evaluations at steps 0, 5, …, 55, all 12 candidates;
- `validation_pairs_v4` isolated from the 678;
- nothing loaded or written.

**The real bundle,** built into a scratch directory, not `dist/`:
- 104 files: the a1 config, the rehearsal script and module, the checkpoint loader and the guide; no evaluation file;
- `smoke_bundle verify` passes;
- its recorded schedules are 55 steps each, with the dry run's schedule hashes.

**The preflight, run from the extracted bundle on CPU** (`--skip-cuda`): every non-CUDA check passes:
- the bundle, the corpus and preparation;
- all 678 candidates and their provenance;
- the schedule;
- validation and evaluation isolation;
- real BGE and its tokenizer;
- GradCache on real BGE, at 5.4e-8.

## 4. K4: the GPU rehearsal

`src/finretriever/training/gpu_rehearsal.py` and `scripts/gpu_rehearsal.py`, added to the bundle's scripts and
traced modules. `rehearsal_outputs/` is a runtime output, so the bundle still verifies after a rehearsal.

| # | Check | What it does | Passes if |
|---|---|---|---|
| 1 | `environment` | GPU, compute capability, CUDA, cuDNN and library versions (`run_environment`), fp16 autocast | CUDA and fp16 autocast work. Recorded, because every seed must use this GPU type. |
| 2 | `dropout` | Real BGE in training mode: different random states, and a restored state, in fp32 and fp16 | Eager attention; another state changes the embeddings (> 1e-3); a restored state replays them (≤ 1e-5 fp32, ≤ 1e-2 fp16) |
| 3 | `attention_dropout_replay` | Every non-attention dropout set to p = 0 for the check only (restored and verified) | Attention dropout alone changes the embeddings and replays exactly |
| 4 | `gradcache_fp32` | GradCache against the uncached passes on the 8 pairs with the longest positives (512 tokens), mini-batch 4 | Relative ≤ 1e-5, cosine ≥ 0.999999, replay ≤ 1e-5 |
| 5 | `gradcache_fp16` | The same under autocast, at the GradScaler's **real initial scale** (read from `GradScaler`, 65,536 by default) and at 1,024 | Finite, relative ≤ 1e-3, cosine ≥ 0.999, replay ≤ 1e-2. An overflow stops the rehearsal. |
| 6 | `fp16_loss_scale` | A full cached backward at the initial scale on each seed's first batch (61–62 pairs) and on the **64 longest pairs**; fp16 against fp32 gradients from the same random state | Every gradient finite and cosine ≥ 0.99. An overflow **stops the rehearsal and reports it**; smaller scales are probed only to inform your decision. |
| 7 | `peak_memory` | One cached step of the **64 longest pairs** with AdamW's two moment buffers allocated (0.82 GiB), timed | Peak reserved memory ≤ 85% of the GPU's (measured) |
| 8 | `validation_full_corpus` | `validation_eval.evaluate` at step 0 over all 10,137 passages, twice; fp16 against fp32 embeddings on 256 passages | Identical on repeat; weights, mode and random state unchanged. The value is written to `validation_step0.json` for N10, never printed. |
| 9 | `checkpoint_resume` | The trainer's `_save_checkpoint` writes `checkpoint-step-000000` with an un-stepped AdamW, the scheduler, the scaler, the random state and the step-0 evaluation. Then the files are re-hashed, the trainer's `restore_checkpoint` reloads into a fresh base model, the benchmark's sentence-transformers loader loads it, and the same step is saved again. | Hashes match; weights bitwise equal; optimizer, scheduler, scaler, random and selection state restored; the random state replays; sentence-transformers agrees (≤ 1e-4); a second save is refused with no partial write |
| 10 | `no_training` | Counts attempted steps; compares weight digests | No optimizer step was attempted, and the final weights equal the weights at load |

**Safeguards:**
- **No optimizer step.** For the whole rehearsal, every `torch.optim` optimizer's `step` and `GradScaler.step` are
  replaced by a function that raises `RehearsalError` (`no_optimizer_steps`). It is tested with AdamW, SGD and
  GradScaler, and restored afterwards. A check that tries to step fails, and so does check 10.
- **fp16 overflow.** An overflow at the initial scale (checks 5 or 6) stops the rehearsal. Checks 7–9 are skipped,
  check 10 still runs, and the exit code is 1. The scaler is never changed.
- **No evaluation data.** It reads the bundle's config, preparation, candidates and validation set, as training does.
- **Write-once output.** A rehearsal refuses a non-empty output directory.

**Exit codes:** 0 every check passed; 1 a check failed; 3 nothing failed but checks were skipped (no CUDA).
`--dry-run` verifies the config, inputs and schedules and lists the checks without loading a model. Run on the real
config, it lists 55 steps per seed and first batches of 62, 62 and 61 pairs.

**Tests** (`test_gpu_rehearsal.py`, 7, and 1 in `test_smoke_bundle.py`):
- **Step guard:** steps raise inside the guard and work after it.
- **A full CPU run** with an attention-carrying test encoder: every CPU-runnable check passes, the CUDA ones are
  skipped, exit code 3. Checkpoint and resume pass with every flag, the weights are identical, and the value is
  never printed.
- **Refusals:** a run without CUDA unless allowed; an unvalidated config; a reused output directory.
- **An overflow** stops the rehearsal, with checks 7–9 skipped and check 10 run.
- **A check that tries to train** fails, and so does check 10.
- **Real tiny BERT:** checks 2–4 pass with eager attention, and SDPA attention fails check 2.
- **An encoder without attention dropout** fails check 3.
- **From a real extracted bundle:** the rehearsal runs, and the bundle still verifies.

**What only the GPU can show.** On CPU, checks 1, 5, 6 and 7 are skipped. Their fp16 and memory behaviour is the
reason for B3, and they run only on the T4. Their code shares the paths the CPU tests exercise:
- `gradcache_check`, now with an explicit loss scale;
- `cached_contrastive_backward`;
- the step guard.

The estimated run time is 10–15 minutes on a T4, not measured.

## 5. The additional requirements

| Requirement | How it is met |
|---|---|
| All three seeds on the same GPU type | Each selection record holds its environment. The decision rule refuses seeds that differ in GPU name, compute capability, CUDA, cuDNN or library versions. The guide says so. |
| Hardware and software metadata | `run_environment` in every selection record and in the rehearsal report; GPU memory in each checkpoint's metadata, as before |
| An fp16 overflow stops and reports; the scaler is never silently changed | Training: a validated run stops at the first non-finite gradient, before the step. Rehearsal: an overflow at the initial scale stops it. Both are tested. No scaler setting exists or changed. |
| `train_candidates_v3` and `validation_pairs_v4` byte-for-byte | Nothing under `data/` changed (§6) |
| Historical results and smoke checkpoints | `models/bge_base_ft_v2_smoke` untouched; its `run.json` reproduced byte-for-byte by the current code; `dist/kaggle` untouched |
| Evaluation labels isolated | The trainer, rehearsal and decision read no evaluation file (spy tests). The decision reads only the selection records and imports no torch. The bundle holds no evaluation data. |

## 6. Verification

| Check | Result |
|---|---|
| `pytest`, the dense venv | **1,301 passed, 25 skipped**, 0 failed (34 new tests). The skips are environmental: the Apple 10-K fixture, `pytrec_eval`. |
| `pytest`, global `python3` | 1,297 passed, 29 skipped (its broken scipy also skips the BERT tests), 0 failed |
| `ruff check src tests scripts` | All checks passed |
| `git diff --check` | Clean |
| Protected files (`data configs src scripts`, SHA-256) | 365 → 372. **Nothing under `data/` changed.** 9 source modules modified; added: 3 modules, 3 scripts, 1 config. |
| The smoke run's stored `run.json` | Reproduced byte-for-byte |
| The a1 dry run, rehearsal dry run, bundle build and verify, CPU preflight from the bundle | Pass (§2–§4) |

## 7. Remaining blockers and risks

**Blockers:**
- **B1 (decision rule): resolved in code.** It is implemented, tested and simulated. It can only be applied after the
  three runs.
- **B3 (CUDA path): tooling ready, not resolved.** It is resolved when the GPU rehearsal passes on a Kaggle T4 (your
  approval), and the first seed then trains.
- B2 and B4 are resolved (P1 closed at 678).

**Risks:**
1. **fp16 at the initial scale (65,536) is untested on BGE.** If check 5 or 6 overflows, the rehearsal stops. Then a
   fixed initial scale needs your decision and a config or code change, because none exists today.
2. **Things the rehearsal cannot show,** because they need real updates:
   - the AdamW, scheduler and scaler interplay over steps;
   - memory with a real AdamW state (emulated here by allocated buffers);
   - mid-run resumption on CUDA, which is close, not bit-identical.

   Seed 42 is therefore the first end-to-end CUDA run and should be reviewed before seeds 1337 and 2026.
3. **Disk.** Every evaluated step writes a full checkpoint (438 MB), after-stop steps included: 12 per seed, about
   5.3 GB plus 0.87 GB of optimizer state. That is the same worst case as a run that never stops early. The
   proposal's idea of trimming after-stop checkpoints to metadata was not adopted: full, hashed checkpoints keep every
   after-stop evaluation verifiable. So run one seed per Kaggle session and download between seeds.
4. **Run length.** With K1 every seed trains all 55 steps and 12 validation passes, about 20 minutes per seed
   (estimated; checks 7 and 8 measure it).
5. **The same GPU type for every seed.** A session that lands on another GPU cannot be used for the decision; check
   `nvidia-smi` first.
6. **Statistical limits** (§1.4): about 41% power at +0.02, a synthetic validation set, and AI labels throughout.

## 8. Kaggle commands for the rehearsal (each stage after your approval)

**Locally, build the bundle** (writes `dist/kaggle/finretriever_train_v3_a1.tar.gz`, its `.sha256` and a metadata
template):

```bash
python3 scripts/smoke_bundle.py build --config configs/train/biencoder_bge_base_ft_v3_a1.yaml \
    --name finretriever_train_v3_a1
```

**Upload** the archive and its `.sha256` as a **private** Kaggle dataset. Then use a notebook with a T4 GPU, internet
on and that dataset as input.

**On Kaggle, set up** as in [kaggle_smoke_test.md](kaggle_smoke_test.md) steps 4–6, with the bundle name
`finretriever_train_v3_a1`:

```python
import hashlib, pathlib, shutil, tarfile
src = next(p for p in pathlib.Path("/kaggle/input").rglob("*") if p.name in ("finretriever_train_v3_a1.tar.gz", "BUNDLE_MANIFEST.json"))
dst = pathlib.Path("/kaggle/working")
if src.name.endswith(".tar.gz"):
    expected = (src.parent / (src.name + ".sha256")).read_text().split()[0]
    actual = hashlib.sha256(src.read_bytes()).hexdigest()
    assert actual == expected, f"archive SHA-256 {actual} != {expected}"
    with tarfile.open(src) as tar:
        tar.extractall(dst, filter="data")
else:  # Kaggle unpacked it
    shutil.copytree(src.parent, dst / "finretriever_train_v3_a1", dirs_exist_ok=False)
```

```
%cd /kaggle/working/finretriever_train_v3_a1
!sha256sum --quiet -c SHA256SUMS && echo "all checksums match"
!python -c "import torch; print('torch==' + torch.__version__)" > /tmp/torch.txt
!pip install -q -r requirements-kaggle.txt -c /tmp/torch.txt
!nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

**Run the preflight, then the rehearsal.** Neither trains, and the rehearsal takes no optimizer step:

```
!python scripts/smoke_preflight.py --config configs/train/biencoder_bge_base_ft_v3_a1.yaml
!python scripts/gpu_rehearsal.py --config configs/train/biencoder_bge_base_ft_v3_a1.yaml --out rehearsal_outputs 2>&1 | tee rehearsal_console.log
```

- The preflight must end with `READY FOR TRAINING`.
- The rehearsal must end with `REHEARSAL PASSED`. Exit 1 means a failure or an fp16 overflow: stop and report.

**Download** `rehearsal_outputs/` (`rehearsal.json`, `validation_step0.json`, `run/`), `rehearsal_console.log` and
`smoke_outputs/preflight.json`, and **stop for review.** Training then needs your separate approval, one seed per
session on a T4 ([kaggle_training.md](kaggle_training.md) §7–§9).

## 9. Files

**Modified:**
- `src/finretriever/training/`:
  - `biencoder_trainer.py`: post-stop training, the overflow stop, `run_environment`, `restore_checkpoint`,
    `gradcache_check(loss_scale)`, batching records and the dry-run text;
  - `checkpoint_selection.py`: format 2, `record_after_stop`, `select_through`, completeness, `ENVIRONMENT_IDENTITY`;
  - `trainer_setup.py`: `batching`, `Schedule.negatives`;
  - `training_data.py`: `BATCHING_METHODS`;
  - `false_negatives.py`: the balanced planner;
  - `smoke_results.py`: after-stop evaluations and the environment;
  - `smoke_preflight.py`: per-seed batching details;
  - `smoke_bundle.py`: the rehearsal script, module and output;
  - `__init__.py`.
- `docs/kaggle_training.md`, `docs/training_full_v1.md` (§3).
- Tests: `test_checkpoint_selection.py`, `test_training_data.py`, `test_evidence_reuse.py`, `test_smoke_bundle.py`,
  `trainer_fixtures.py`.

**Added:**
- `src/finretriever/training/decision_rule.py`, `decision_simulation.py`, `gpu_rehearsal.py`;
- `scripts/07_decide_a1_vs_a0.py`, `scripts/gpu_rehearsal.py`, `scripts/simulate_decision_rule.py`;
- `configs/train/biencoder_bge_base_ft_v3_a1.yaml`;
- `tests/unit/test_decision_rule.py`, `test_gpu_rehearsal.py`, `test_balanced_batching.py`;
- this report.

Nothing is committed.
