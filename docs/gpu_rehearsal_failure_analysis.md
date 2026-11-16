# GPU rehearsal on a Kaggle T4: failure analysis and fixes

**Status, 2026-11-10:** both failures of the first T4 rehearsal are explained, fixed and tested on CPU. A rebuilt
bundle (`finretriever_train_v3_a1_r2`) and notebook are verified locally.

> **Update, 2026-11-10: the r2 rehearsal passed all ten checks on a T4.** `fp16_loss_scale`: cosine 0.99998 with
> dropout off; mask agreement across dtypes 0.819. `checkpoint_resume`: every flag true; the benchmark's loader agrees
> to 1.4e-07. The results are in [training_a1_kaggle_rehearsal_report.md](training_a1_kaggle_rehearsal_report.md).

**Not done:**
- no GPU request and no new rehearsal;
- no training and no optimizer step outside CPU unit tests;
- no change to `train_candidates_v3`, `validation_pairs_v4`, the learning rate, the schedule or the GradScaler;
- no commit or push.

The T4 numbers below are as you reported them from the run. I did not receive its output archive.

## 1. Summary

| | T4 result | Root cause | Fix | Effect on training |
|---|---|---|---|---|
| `fp16_loss_scale` | "fp16 gradients differ from fp32 ones (cosine 0.4461)" | **The comparison was invalid.** It compared an fp16 and an fp32 pass with dropout on, assuming one random state gives both the same dropout masks. On CPU it does (measured). On CUDA the T4 result shows it does not, and with different masks these gradients are not comparable at all (cosine −0.035 on CPU). Precision alone gives cosine 0.99998. | The pass/fail comparison now runs **with dropout off in both passes**, so both compute the same function. The mask agreement across dtypes, the old dropout-on comparison and the mask-only effect are recorded, never deciding. | None. Training never mixes fp32 and fp16: every pass and its GradCache replay run in fp16, and the T4 confirmed that replay is exact (check 5). |
| `checkpoint_resume` | strict reload: missing `*.LayerNorm.weight/bias`, unexpected `*.LayerNorm.gamma/beta` | **transformers 5.3.0's `save_pretrained`** reverses its "legacy" renaming on save by default (`save_original_format=True`), for a model loaded with `from_pretrained`. The checkpoint stored the 50 LayerNorm tensors as `gamma`/`beta`, and the trainer's strict reload refused them. No value was lost. | Checkpoints are written under the model's own parameter names (`save_original_format=False`), the names BGE's hub file uses. Every save is checked name by name, shape and dtype. The reload stays strict, and a legacy-named file is refused, never renamed. | **Real.** `--resume` of any real run would have failed. Evaluation was unaffected, because it loads through `from_pretrained`, which renames back. |

**Preserved from the T4 run** (8 of 10 checks passed):

| Check | Result |
|---|---|
| Environment | Tesla T4, compute capability 7.5; CUDA 13.0; PyTorch 2.11.0+cu130; transformers 5.3.0; tokenizers 0.22.2 |
| `dropout`, `attention_dropout_replay` | pass |
| `gradcache_fp32` | pass |
| `gradcache_fp16` | pass at both 65,536 and 1,024: relative 2.2e-08, cosine 1.00000, replay 0 |
| `fp16_loss_scale`, first part | Every gradient finite at the initial scale 65,536 on each seed's first batch and on the 64 longest pairs. The check reports its comparison only after every batch is finite, and the failure message came from the comparison. |
| `peak_memory` | 9.82 GiB reserved of 14.56 GiB (67%) |
| `validation_full_corpus` | 93 questions over 10,137 passages, about 121 s per pass |
| `no_training` | 0 optimizer steps; the starting and final weights identical |

## 2. Failure 1: the fp16-against-fp32 comparison

### 2.1 What the comparison did

Check 6 took the 16 pairs with the longest positives and did the following:
1. saved the random state;
2. ran one cached backward in fp16 at loss scale 65,536 and divided the gradients by 65,536;
3. restored the random state;
4. ran the same cached backward in fp32;
5. compared the two gradient vectors (cosine ≥ 0.99 to pass).

### 2.2 The five questions

1. **Identical inputs, masks, random state and loss normalization?**
   - **The same:** the pairs, the weights, the loss (in-batch InfoNCE on float embeddings, mean over the batch,
     divisor 1) and the starting random state.
   - **Not guaranteed the same: the dropout masks.** One random state is replayed, but the kernels that consume it
     draw masks for fp16 tensors (attention probabilities and the hidden states after each dense layer, under
     autocast) and for fp32 tensors.
   - **On CPU** the masks are identical: the agreement is 1.000 on a hidden-state-sized tensor.
   - **On the T4** the numbers below show they cannot have been. The likely mechanism is that CUDA's fused dropout
     maps random numbers to elements by a vector width that depends on the element size. This is a hypothesis: the
     r2 rehearsal measures the agreement directly (§4).
   - **Measured in r2 (2026-11-10, T4):** from one random state, the fp16 and fp32 masks agree on 0.819 of the
     elements, so they are not identical. The dropout-on comparison gave cosine 0.5311, and two fp32 dropout states
     gave 0.4001. With dropout off, the deciding comparison gave cosine 0.99998, as on CPU. The diagnosis is
     confirmed; the exact kernel mechanism was not inspected.
2. **Too sensitive to fp16 rounding, or to a different random path?** It is entirely a random-path effect. On real
   BGE, same pairs and mini-batch:

   | Comparison (CPU, real BGE) | Cosine | Relative difference |
   |---|---:|---:|
   | fp16 against fp32, identical masks (what the check meant to measure) | 0.99998 | 0.006 |
   | fp16 against fp32, dropout off (precision alone) | 0.99998 | 0.006 |
   | fp32 against fp32, different masks (a mask mismatch alone) | −0.035 | 5.05 |
   | fp32 against fp32, the same mask replayed | 1.0 | 0 |

   With these 16 pairs, the gradient is dominated by the dropout noise: two masks give nearly orthogonal gradients
   of very different norms. A cosine of 0.4461 is well inside what a mask mismatch produces. It is far outside what
   fp16 rounding produces.
3. **Unscaled correctly?** Yes. The fp16 gradients were divided by the same 65,536 that multiplied the loss. A test
   shows that multiplying by 65,536 and dividing back leaves fp32 gradients **bitwise** unchanged.
4. **Overflow, underflow, non-finite values?**
   - **None.** Every gradient was finite on the T4 at 65,536, and on CPU.
   - **About 21.3% of gradient entries are exactly zero in both precisions.** These are the word-embedding rows of
     tokens absent from the batch: 23.4M of 109.5M parameters. fp16 adds 0.002 percentage points, which is no
     material underflow.
5. **Equivalent quantities?** No. With dropout on, the two passes compute two different stochastic functions
   unless their masks agree, and on CUDA they need not. The comparison was not measuring precision.

**Conclusion.**
- **The test was invalid, not the training path.** The gradient computation itself was right, and nothing points
  to an fp16 problem in training: fp16 GradCache equals uncached fp16 to 2.2e-8 on the T4.
- **The GradScaler setting was not involved** and is unchanged.

### 2.3 The fix

`gpu_rehearsal.precision_comparison`, used by check 6.

**Deciding:** fp16 against fp32 gradients with dropout off in both passes. `dropout_disabled` sets every dropout
module, attention included, to p = 0, and then restores and verifies them. Both passes therefore compute the same
function on the same pairs, weights, loss and normalization. The fp16 loss is scaled by the GradScaler's real initial
scale, and its gradients are divided back. Both must be finite, and the cosine must be ≥ 0.99, the same threshold as
before. Each precision also records its zero share and its largest gradient.

**Recorded, never deciding:**
- **`dropout_masks_across_dtypes`:** whether one random state draws the same mask for an fp16 and an fp32 tensor of
  hidden-state size on the device. This tests the explanation on the T4.
- **`dropout_on_same_state`:** the old comparison, marked "a precision measure only if the masks agree".
- **`fp32_two_dropout_states`:** what a mask difference alone does.

The overflow part of check 6 is unchanged, and an overflow still stops the rehearsal with the scaler untouched.

**Tests** (`tests/unit/test_gpu_rehearsal.py`, 4 new, with a tiny BERT and CPU fp16 autocast):
- **Masks decide dropout-on comparisons:** the same masks give bitwise-equal gradients; other masks move them far
  (cosine < 0.99).
- **Unscaling is exact:** a 65,536 scale, applied and removed, changes nothing.
- **The new comparison is valid whatever the dropout kernel does.**
  - With CPU dropout: identical masks, dropout-off and dropout-on cosines both ≥ 0.99.
  - With a simulated dtype-dependent dropout (the mask lands on other elements in an fp16 pass), the mask
    diagnostic reports the mismatch, and the old dropout-on comparison fails as on the T4.
  - The new dropout-off comparison gives exactly the same result as without the simulation.
  - Weights, gradients and dropout probabilities are unchanged afterwards.
- **Dropout is restored** even when the comparison raises.

## 3. Failure 2: checkpoint save and reload

### 3.1 Reproduced on CPU

With the cached BGE and the same transformers 5.3.0:
- **BGE's hub file** (`model.safetensors`) holds 200 tensors named `LayerNorm.weight/bias`, plus the non-persistent
  `embeddings.position_ids` buffer.
- **The loaded model** has 199 tensors with the same names.
- **`HFEncoder.save`** (`save_pretrained`) wrote 199 tensors, 50 of them as `LayerNorm.gamma/beta`.
- **`load_weights`** (strict `load_state_dict`) failed with the T4's exact missing and unexpected keys.

### 3.2 Root cause

In transformers 5.3.0, `save_pretrained(..., save_original_format=True)` is the default, and it runs
`revert_weight_conversion` on the state dict. `conversion_mapping.py` defines a global "legacy" rule:
`LayerNorm.gamma` → `LayerNorm.weight` and `LayerNorm.beta` → `LayerNorm.bias`. Its reverse is applied on save to any
model loaded with `from_pretrained`, "even if the model was loaded from a non-legacy checkpoint" (its docstring).
- **`from_pretrained` on such a checkpoint** applies the rule forward again. That is why the benchmark and
  sentence-transformers loads worked.
- **The trainer's `load_weights`** reads the file straight into `load_state_dict(strict=True)` and refused it.
- **Why earlier tests missed it.** They saved a tiny BERT built from a config, which carries no conversion and is
  saved under its own names. Loaded through `from_pretrained`, the same tiny model reproduces the failure: 10 of 39
  tensors become `gamma`/`beta`, and the strict load fails.

**Correspondence, checked before changing anything:**
- under gamma→weight and beta→bias, the saved key set equals the model's;
- every saved tensor is bitwise equal to its parameter;
- `from_pretrained` on the checkpoint restores identical weights.

So no value was lost; only the names broke the strict reload.

### 3.3 The fix (`biencoder_trainer.HFEncoder`)

- **`save`** writes with `save_original_format=False` (when transformers has the option), so the file uses the
  model's own parameter names, which are also the hub file's.
- **The written file is checked against the model's state dict:** names, shapes and safetensors dtypes. A file that
  differs is an error at save time.
- **`load_weights`** checks the file the same way before a strict `load_state_dict`. A legacy-named file is refused
  with "(transformers' legacy LayerNorm names gamma/beta) … it cannot be resumed". `strict=False` is never used, and
  nothing is renamed or dropped.

**Verified on CPU with the real BGE** (the rehearsal's own check 9, under its no-step guard):
- **pass**, with every flag true: hashes match, weights identical, and the optimizer, scheduler, scaler, random and
  selection state restored;
- the random state replays a dropout pass bitwise;
- the benchmark's loader agrees to 1.7e-07;
- a second save is refused, with no partial write;
- 0 optimizer steps, and the weights unchanged;
- the checkpoint's 199 tensor names equal the hub file's, without `position_ids`, with no `gamma`/`beta`.

This run used a stand-in step-0 validation record, because a CPU validation pass takes about 26 minutes; check 8
produces the real one on the GPU.

**Tests** (`tests/unit/test_checkpoint_serialization.py`, 4 new, a tiny BERT loaded with `from_pretrained` like BGE):
- every parameter keeps its name and value, including trained LayerNorm values; the strict reload and
  `from_pretrained` agree;
- a legacy-named file is refused with the weights untouched, and a save that writes other names is caught;
- **a resumed run continues exactly as the uninterrupted one:**
  - identical weights;
  - AdamW's moments, step and parameter groups;
  - the scheduler's state and position;
  - the GradScaler;
  - the next step's loss and weights, bitwise;
- check 9 passes end to end. Here the benchmark's input-format check is stood in by sentence-transformers itself,
  because the fixture's base model is not BGE.

### 3.4 Side finding: the historical smoke checkpoints

- **What was saved.** The smoke run's checkpoints (`models/bge_base_ft_v2_smoke`) were saved by the same code, so
  they store their LayerNorm tensors as `gamma`/`beta`: 50 of 199.
- **Retrieval evaluations stand.** They load through `from_pretrained`, which renames correctly.
- **One reported field was incomplete.** In `smoke_outputs/results.json`, `weights_vs_base` compared **149 of 199
  tensors**. It listed the LayerNorm tensors as "missing_from_checkpoint"; they were present under the legacy names.
- **Nothing historical was changed.** The new strict `load_weights` refuses those checkpoints, but they are never
  resumed.

## 4. The rebuilt bundle and notebook (verified locally)

| | |
|---|---|
| Bundle | `dist/kaggle_a1_r2/finretriever_train_v3_a1_r2.tar.gz`, SHA-256 `efc377d93f61b3bb5fbfb6706ecd2b08507d0ed5d30a13f597ba3ca9517e6f65`; manifest `a6615850495541bcfb2cb7d74cf0fa8402c9edfef46e6c752dad25fe9819aeba` |
| Checks | archive against its `.sha256`, every file against `SHA256SUMS` and `verify_bundle`: 104 files, all byte-identical to the repository |
| Against the first A1 bundle | exactly two files changed: `training/biencoder_trainer.py` and `training/gpu_rehearsal.py`. `train_candidates_v3` and `validation_pairs_v4` are byte-identical; no evaluation file. |
| Preserved | the first A1 bundle (`dist/kaggle_a1`, `2c3f7a64…`, the one that ran on the T4) and the smoke bundle are unchanged |
| Notebook | `dist/kaggle_a1_r2_notebook/finretriever_a1_rehearsal_r2.ipynb`: the same cells as before, with the r2 bundle name, hashes, dataset slug `finretriever-train-v3-a1-r2` and outputs `kaggle_a1_r2_outputs.zip`. No training command. |
| Notebook, tested locally | the extraction, pinned-hash and bundle-verification cells pass; off a T4 it stops with "the A1 runs need a T4"; the rehearsal cell's refusal path still packages and hashes the outputs; `check_outputs.py` verifies the result |

**What the r2 rehearsal will show for check 6:**
- the dropout-off fp16-against-fp32 cosine, the deciding number (expected near the CPU's 0.99998);
- the mask agreement across dtypes on CUDA, which confirms or refutes the explanation;
- the old dropout-on number.

Check 9 runs the fixed save and reload.

## 5. Verification

| Check | Result |
|---|---|
| `pytest`, the dense venv | **1,309 passed, 25 skipped**, 0 failed: 8 new tests (4 serialization, 4 precision) |
| `pytest`, global `python3` | 1,297 passed, 37 skipped (its broken scipy skips the BERT tests), 0 failed |
| `ruff check src tests scripts`; `git diff --check` | clean |
| Real BGE on CPU: check 9; the precision and mask diagnostics | as in §2.2 and §3.3 |
| Frozen data | `train_candidates_v3` and `validation_pairs_v4` unchanged: byte-identical in the repository and in the r2 bundle |

## 6. Remaining blockers and next step

- **B3 (CUDA path): still open.** It closes when the r2 rehearsal ends with `REHEARSAL PASSED` on a T4.
- **The environment to keep for the experiment.** The T4 image ran PyTorch 2.11.0+cu130 and CUDA 13.0. The decision
  rule requires every seed to run on the same GPU type and stack.
- **The fp16 initial scale is unchanged at 65,536.** The T4 found every gradient finite at that scale.

**To run the r2 rehearsal (after your approval):**
1. Upload `dist/kaggle_a1_r2/finretriever_train_v3_a1_r2.tar.gz` and its `.sha256` as a new **private** dataset,
   `finretriever-train-v3-a1-r2`.
2. Import `dist/kaggle_a1_r2_notebook/finretriever_a1_rehearsal_r2.ipynb`. Set GPU T4, internet on, and add that
   dataset.
3. Run All.
4. Download `kaggle_a1_r2_outputs.zip` and its `.sha256`, then run
   `python3 dist/kaggle_a1_r2_notebook/check_outputs.py <zip>`.

## 7. Files changed

- `src/finretriever/training/biencoder_trainer.py`: `HFEncoder.save` (own parameter names, checked);
  `HFEncoder.load_weights` (checked, strict, legacy refused); `_SAFETENSORS_DTYPES`.
- `src/finretriever/training/gpu_rehearsal.py`:
  - `precision_comparison`, `dropout_disabled`, `dropout_mask_agreement`;
  - check 6 uses them;
  - the docstring explains why.
- Tests: `tests/unit/test_checkpoint_serialization.py` (new) and `tests/unit/test_gpu_rehearsal.py` (4 new tests).
- Gitignored, outside the repository: `dist/kaggle_a1_r2/`, `dist/kaggle_a1_r2_notebook/`.
