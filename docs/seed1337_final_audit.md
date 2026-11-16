# Seed 1337: final audit

## Official record (2026-11-11)

**Seed 1337 is officially accepted:** execution `b98f183e-0560-45e0-9e10-cacfed7d1c9f`, selected checkpoint
`checkpoint-step-000040`, digest `8237ec0b82a12830c5e41653f3597fb82516475203b8e61b262462a40597b675`, as proposed below.

**Open items closed:**
- **L2:** the user confirmed that the Kaggle version history shows exactly one training execution of seed 1337.
- **L5:** the dataset provenance was approved for seed 2026, as a notebook-only change. It is not retroactive: seed
  1337's `execution.json` still has no dataset record. Its notebook read only the bundle from the inputs, and Kaggle
  verified it by hash before training (§8). See
  [training_a1_seed2026_readiness.md](training_a1_seed2026_readiness.md).

The audit below is unchanged.

## Run of record (as proposed by this audit; accepted above)

| | |
|---|---|
| Selected checkpoint | `checkpoint-step-000040`, digest `8237ec0b82a12830c5e41653f3597fb82516475203b8e61b262462a40597b675`, whole-corpus validation MRR@10 0.6672725721 |
| Its weights | `model.safetensors` SHA-256 `6d9952c3574ea3a6873ef7397e542bce96c074cf5da8b01f68427040dac84284` |
| Selection record | `selection.json` (SHA-256 `165064f5…`), unchanged. It holds 12 candidate evaluations; the rule's third miss falls at step 55, the planned last step, so there is no after-stop evaluation. |
| Archive | `kaggle_a1_seed1337_outputs.zip`, 6,136,992,269 bytes, SHA-256 `7f0317afea45065fb0712fea2fe9490ea899f97b3a18fc8e598c145047742048` (one download in `~/Downloads`). Records are extracted in `dist/kaggle_a1_seed1337_results/`. |
| Execution | `b98f183e-0560-45e0-9e10-cacfed7d1c9f`, a saved version (`KAGGLE_KERNEL_RUN_TYPE = "Batch"`). Training ran 2026-11-11 00:00:19–00:30:14 UTC on a Tesla T4: CUDA 13.0, driver 580.178.04, torch 2.11.0+cu130, Python 3.13.16. |
| Notebook | `dist/kaggle_a1_seed_v2/finretriever_a1_train_seed1337.ipynb`, as delivered (SHA-256 `19bbf437…`); `START_OPTIMIZER = True` |

**Date of the audit:** 2026-11-11.

**Scope:** the eight items requested. Everything below was recomputed on this Mac from the archive itself, independently
of the notebook's own verification, except where a section says otherwise.

**Nothing was changed.**
- **Not done:** no training, no API request, no seed 2026, no commit or push. No frozen dataset, historical record or
  selection record was changed.
- **Re-checked after the audit:**
  - seed 42's 119 extracted records and 6 notebook files re-hash to their snapshots;
  - the 122 extracted seed 1337 records are byte-identical before and after this audit.

## 1. Conclusions

1. **The archive is complete and intact.**
   - All 134 listed files were re-hashed from inside the zip: none mismatched, none unlisted, none missing.
   - The zip's SHA-256 equals its `.sha256`.
2. **The selection is correct: `checkpoint-step-000040`.**
   - All 12 values were recomputed exactly from the 93 per-question ranks, and each equals the record.
   - The K1 rule, re-implemented without the repository's code, selects step 40.
   - The repository's own reader accepts the record; `select_through` returns `(40, 55)`.
3. **The stopping rule worked, and changed nothing here.**
   - Steps 45, 50 and 55 each fail to beat step 40. The third miss is step 55, the planned last step.
   - So `rule_stop` is step 55, `stopped.early` is true, and there are no after-stop evaluations.
   - Every checkpoint was a selectable candidate. The plain arg-max over all 12 is also step 40.
4. **The selection is robust: unlike seed 42's, no one-rank change on one question can flip it.**
   - Step 40 leads the next best checkpoint (step 55) by 0.0066 MRR@10, which is 0.61 in summed reciprocal rank.
   - One question moving up one rank gains at most 0.5, so no such move at a later step could overtake step 40.
   - Seed 42's selection turned on margins of 0.033 and 0.011.
5. **The execution identity is verified.**
   - It is a saved-version (batch) execution, positively identified, with the gate open.
   - The execution code is the version verified on Kaggle, and the execution ID is new.
   - Its record binds `run.json` and `selection.json` byte for byte.
   - All five pre-training checks passed on the T4 before the optimizer started.
6. **Every checkpoint hash and binding holds.**
   - **Digests:** all 12 checkpoint digests, rebuilt from the zip's bytes, match `selection.json`,
     `validation_log.jsonl` and the metadata.
   - **Resume state:** it is only in step 55, and its hash matches. Read independently, it is consistent with the
     records: the optimizer, scheduler and scaler at step 55, and the selection state with step 40 best.
7. **Training was numerically clean.**
   - All 55 of 55 steps ran; the loss fell from 1.0889 to 0.1041.
   - There was no non-finite loss or gradient norm.
   - The loss scale stayed at 65,536 with no overflow.
   - The GradCache replay difference was 0.0 at every step.
   - Every step was clipped, as in seed 42.
8. **The configuration, schedule procedure and environment are the same as seed 42's.**
   - `run.json` differs only in the seed, the run name and the schedule hash.
   - The environment is equal on all nine identity keys, `pip freeze` is byte-identical, and the GPU driver is the same.
9. **No development or evaluation labels were used.**
   - The bundle contains no evaluation data.
   - The trainer's inputs are bound by hash.
   - Selection used only `validation_pairs_v4`.
10. **No discrepancy affects validity.** §9 lists two wording points and seven limitations.

## 2. Evidence

| Item | Value |
|---|---|
| Archive | 6,136,992,269 bytes, SHA-256 `7f0317af…`, equal to its `.sha256`. It is stored without compression, with one top directory, `kaggle_a1_seed1337_outputs/`. |
| Contents | `OUTPUT_SHA256SUMS` lists 134 files; with the list itself, the zip has 135 members. There are 12 checkpoints with 9 hashed files each, and one `trainer_state.pt`. |
| Extraction | `dist/kaggle_a1_seed1337_results/`: 122 records, each equal to its listed hash. The weights and the resume state stay in the zip. |
| Timeline (UTC) | notebook start 2026-11-10 23:54:29; gate opened and training started 2026-11-11 00:00:19; training ended 00:30:14 (29 min 55 s). Then results, verification (362.6 s) and packaging. `status.json` reports exit 0 for training, results and verification. |
| The notebook's own verification | `run_verification.json`: 5 of 5 checks pass (run identity, checkpoints, resume state, re-evaluation, execution identity), and the run directory is unchanged |

## 3. Checkpoint selection, recomputed (items 1 and 2)

**Columns:**
- **Σ RR:** the sum over the 93 questions of the reciprocal rank, cut at 10 (MRR@10 × 93).
- **Misses:** candidate evaluations in a row without improvement.
- **Digest:** the checkpoint digest, rebuilt from the zip.
- **Seed 42:** that seed's value at the same step, for reference only.

Every row is a candidate and selectable, both in `selection.json` and in the checkpoint's `metadata.json`.

| Step | MRR@10 | Σ RR | Improved | Misses | Digest | Weights | Seed 42 |
|---:|---:|---:|---|---:|---|---|---:|
| 0 | 0.2658858167 | 24.7274 | yes | 0 | `e1735b1e…` | `0d4db737…` | 0.265886 |
| 5 | 0.4096432838 | 38.0968 | yes | 0 | `ac6fc5b2…` | `15300168…` | 0.425098 |
| 10 | 0.5696321898 | 52.9758 | yes | 0 | `84034160…` | `57269c8c…` | 0.586781 |
| 15 | 0.6103046595 | 56.7583 | yes | 0 | `dc8ba8b0…` | `918f8604…` | 0.632433 |
| 20 | 0.6338240314 | 58.9456 | yes | 0 | `656eff02…` | `319eed01…` | 0.643544 |
| 25 | 0.6340160437 | 58.9635 | yes | 0 | `cf14bff6…` | `7bc14e79…` | 0.658252 |
| 30 | 0.6390211640 | 59.4290 | yes | 0 | `6e0c1cb7…` | `8b0025cf…` | 0.662826 |
| 35 | 0.6505333675 | 60.4996 | yes | 0 | `395c7807…` | `c97d98b4…` | 0.668570 |
| **40** | **0.6672725721** | 62.0563 | yes | 0 | `8237ec0b…` | `6d9952c3…` | 0.666500 |
| 45 | 0.6577444956 | 61.1702 | **no** | 1 | `32e47fed…` | `3b37a354…` | 0.668070 |
| 50 | 0.6585082779 | 61.2413 | **no** | 2 | `9ebacb97…` | `6128db34…` | 0.668920 |
| 55 | 0.6607185527 | 61.4468 | **no** | 3 | `11c4b675…` | `de1b5536…` | 0.669039 |

### Independent recomputation

**Values.**
- Each value was recomputed with exact fractions from the 93 whole-corpus `first_relevant_rank` values: 1/rank if the
  rank is at most 10, otherwise 0.
- All 12 equal the recorded values (difference below 1e-12), and so does every per-question `mrr@10`.

**Questions.**
- There are 93 unique pair IDs, the same ones in the same order in all 12 evaluations.
- Their hash, SHA-256 of the IDs joined by newlines, is `b83abf1a…`, the pinned `validation_pairs_v4` value.
- The IDs come only from `val_pilot_v1`, `val_pilot_v2e` and `val_pilot_v2f`.

**Step 0.** The base model is identical across seeds:
- its ranks hash to `0de690c4…`, the rehearsal's value;
- its per-question results are identical to seed 42's step 0;
- its weights (`0d4db737…`) are byte-identical to seed 42's step 0.

**The rule, re-implemented from its approved text:**
- candidates are step 0 and every 5 steps;
- a value counts as an improvement only if it is strictly greater at 12 decimals;
- selection stops after 3 candidate evaluations in a row without improvement.

The result is: select step 40, with the rule stopping at step 55.

**Flags.**
- `candidate`, `improved` and `since_best` in `selection.json` equal the re-implementation at all 12 steps.
- `validation_log.jsonl` equals the record entry for entry: step, value and digest.
- Every checkpoint's `metadata.json` has `candidate: true` and `selectable: true`, the same value, and set
  `validation_pairs_v4`.

**The repository's code.** `read_selection_record` accepts the record (it re-derives the rule and checks completeness),
and `select_through` returns `(40, 55)`.

### Stopping and eligibility

**The record:**
- `selected` is step 40;
- `rule_stop` is step 55 (`11c4b675…`, 0.6607185527);
- `stopped` is `{early: true, at_step: 55, planned_steps: 55}`;
- `after_stop_evaluations` is empty.

**What `early: true` means here.**
- It is the rule's own flag, from `select()`: the patience of 3 misses ran out.
- `at_step` is the number of steps done.
- The stop coincides with the planned last step. So training was not shortened, and no evaluation came after the stop.

**Eligibility.**
- Every checkpoint from 0 to 55 was evaluated before the stop and was a selectable candidate.
- Step 55 is itself a candidate: it is the third miss.
- No checkpoint was excluded by the stop.

**After-stop evaluations.** None exist in this run. That they can never be selected is guaranteed in code and in tests,
as shown in [seed42_final_audit.md](seed42_final_audit.md) §5.

### Margins and counterfactuals

| Against step 40 | ΔMRR@10 | ΔΣ RR | Questions whose RR changes | Can one question moving one rank overtake? |
|---|---:|---:|---:|---|
| step 35 | −0.016739 | −1.5567 | 14 | no |
| step 45 | −0.009528 | −0.8861 | 7 | no |
| step 50 | −0.008764 | −0.8151 | 10 | no |
| step 55 | −0.006554 | −0.6095 | 10 | no |

**Closing the gap needs at least two rank positions.** The largest gain from one question moving up one rank is 0.5
(rank 2 → 1), which is less than every gap above. Closing even the smallest gap, 0.61 to step 55, needs one question to
move two ranks or two questions to move one rank each. In seed 42, a single rank on a single question decided the
selection.

**Counterfactuals,** with the repository's `select_through` applied to this run's recorded values:

| Change | Rule outcome |
|---|---|
| none (as recorded) | select 40; stop at 55 |
| any one of steps 45, 50, 55 exactly equal to step 40 | select 40 (ties go to the earlier step); stop at 55 |
| any one of steps 45, 50, 55 above step 40 by a single rank-9-to-10 move | that step is selected; no stop |
| step 40 no better than step 35 | select 55; no stop |
| arg-max over all 12, ignoring the stop | 40 |

**Step 40 is a local peak.** It is 0.0167 above step 35, and step 45 falls back by 0.0095. If step 40 had not improved
on step 35, the later steps would have counted as improvements, and the rule would have selected step 55.

## 4. Execution identity (item 3)

**`execution.json`**, and its byte-identical copy in the run directory:

| Field | Value | Check |
|---|---|---|
| Execution type | `batch`, from `KAGGLE_KERNEL_RUN_TYPE = "Batch"` | The gate requires exactly `"Batch"`. |
| Indicators | 12 `KAGGLE_` variable names (names only, no values); `/kaggle/src/script.ipynb`; a papermill parent process | The same pattern as the verified batch probe (`8e34c7d6`), not the interactive pattern (`jupyter-server`). |
| Execution code | `3cd0bc44…` | The code verified by the Kaggle probes |
| Execution ID | `b98f183e…` | New; distinct from both probes |
| Gate | open; decided 00:00:19 UTC; `start_optimizer` true | |
| Training | 00:00:19 → 00:30:14 UTC, exit 0 | Times in order, inside the GPU monitor's window (00:00:19.4–00:30:13.8) |
| Environment | Tesla T4 (compute 7.5, 14.56 GiB), CUDA 13.0, cuDNN 92700, driver 580.178.04, torch 2.11.0+cu130, transformers 5.3.0, tokenizers 0.22.2, Python 3.13.16 | Equal to `selection.json`'s environment on every identity key |
| Seed, config, schedule, bundle, manifest | 1337, `7a4c4618…`, `795095dc…`, `efc377d9…`, `a6615850…` | Equal to `pins.json`, `run.json`, `selection.json` and `pretraining_checks.json` |
| `run_files` | `run.json` `46dd8e34…`, `selection.json` `165064f5…` | Equal to those files' bytes in the zip |

**Pre-training checks** (`pretraining_checks.json`). All five passed on the T4 before the optimizer was allowed to
start, with `mismatch: {}`.

| Check | Found on the T4 |
|---|---|
| 1. Python and packages | Python 3.13.16; the 13 pinned packages equal seed 42's; `pip freeze` identical to seed 42's |
| 2. GPU and CUDA | T4, compute 7.5, CUDA 13.0, cuDNN 92700, driver 580.178.04; preflight `cuda_fp16`, `gradcache` and `gpu_memory` all pass |
| 3. Config and schedule | config `7a4c4618…`; schedule `795095dc…` (55 steps, warmup 6, balanced 11×5, none dropped, at least 60 negatives) |
| 4. Candidates and validation | 678 pairs (`train_candidates_v3`, manifest `0586c7ca…`), 93 questions (`validation_pairs_v4`, manifest `40238cb0…`, IDs `b83abf1a…`), 10,137 passages |
| 5. Execution mode | `batch` (`"Batch"`) |

**Python version.** The Kaggle probe sessions reported Python 3.13.18. This session ran 3.13.16, the same as seed 42,
so the risk did not materialise.

## 5. Checkpoint hashes and bindings (item 4)

**Every file.** Re-hashed from inside the zip against `OUTPUT_SHA256SUMS`: 134 of 134 match.

**Checkpoint digests.**
- **Method:** each checkpoint's file map was rebuilt from the zip's bytes. It covers all 9 files except
  `trainer_state.pt`, including `metadata.json`. Its digest was recomputed with an independent canonical JSON (sorted
  keys, compact separators).
- **Result:** for all 12 checkpoints, the file map and digest equal `selection.json`, and the digest equals
  `validation_log.jsonl`. The selected entry's digest equals its evaluation's digest.

| | Digest | `model.safetensors` |
|---|---|---|
| step 0 | `e1735b1e6dfbef5092f5140c278244f502963956ec3eeec26914520f973657ad` | `0d4db737f56aaea90796b5a8d219de0eee958295a575c611f6b417ad340151da` (= seed 42's step 0) |
| **step 40 (selected)** | `8237ec0b82a12830c5e41653f3597fb82516475203b8e61b262462a40597b675` | `6d9952c3574ea3a6873ef7397e542bce96c074cf5da8b01f68427040dac84284` |
| step 55 (last) | `11c4b67594a8b7acad0f1af9bc422204caf67e5b14a858bb13d39b763a3b4f71` | `de1b55364edd7439560dbcbc4cedfd54232d388180fc6e89ff9098c70d597d04` |

**The record chain.**
- `execution.json`'s `run_files` bind `run.json` and `selection.json`.
- `selection.json`'s `run_record` binds `run.json` (`46dd8e34…`).
- Every record carries config `7a4c4618…` and schedule `795095dc…`.
- So does each checkpoint's `metadata.json`, and that file is part of the checkpoint's digest.

**Resume state.**
- **Location:** only in `checkpoint-step-000055`, as designed.
- **Hash:** 871,638,951 bytes, SHA-256 `48b7d6a954aa14ea78c3b83af1b4a1129ebd54d024d24b6c5ec2e9fb6bfc376a`, as listed.
- **Read independently:** on CPU with torch's restricted loader (`weights_only=True`), as the trainer itself loads it.

| Part | Contents | Agrees with |
|---|---|---|
| Step, schedule | 55; `795095dc…` | `run.json` |
| AdamW | moments for 197 of 199 parameters, all at step 55, all finite. The 2 without moments are consistent with the BERT pooler, which CLS pooling never uses; seed 42 shows the same pattern. Two groups: weight decay 0.01 (76 parameters) and 0.0 (123); initial learning rate 2e-5, betas (0.9, 0.999), eps 1e-8; learning rate now 0 | the config |
| Scheduler | `last_epoch` 55, learning rate 0 | the planned schedule |
| Scaler | scale 65,536; growth tracker 55 (55 overflow-free steps in a row; the growth interval is 2,000) | `log.jsonl` (65,536 at every step) |
| Random state | Python, torch and CUDA | |
| Selection state | best step 40 (0.6672725721), 3 misses, evaluations 0–55, none after the stop; values and digests equal `selection.json` | the record |
| History | equal to `log.jsonl` | |

**Checked on the T4 only** (the archive's own verification; not repeated here, because this Mac has no CUDA):
- the step-55 state restores exactly, and its AdamW moments equal the saved ones;
- the reloaded checkpoints 40 and 55 reproduce their recorded validation results exactly;
- step 0 ranks the validation set as the rehearsal did.

## 6. Training behaviour (item 5)

| | Seed 1337 | Seed 42 |
|---|---|---|
| Steps | 55 of 55 (5 epochs × 11) | 55 of 55 |
| Pairs per epoch | 678 (batches of 61–62) | 678 |
| Loss, first → last | 1.0889 → 0.1041 | 1.2130 → 0.1513 |
| Loss minimum | 0.0674 (step 41) | 0.0517 (step 53) |
| Mean loss by epoch | 0.831, 0.308, 0.170, 0.132, 0.106 | 0.881, 0.338, 0.157, 0.117, 0.110 |
| Gradient norm before clipping: min / median / max | 2.80 / 4.79 / 14.63 (step 1) | 2.11 / 4.88 / 16.90 (step 1) |
| Mean gradient norm by epoch | 10.08, 5.69, 4.62, 4.24, 3.69 | 10.64, 5.95, 4.26, 3.73, 3.96 |
| Steps clipped (norm above 1.0) | 55 of 55 | 55 of 55 |
| Non-finite loss / gradient norm | 0 / 0 | 0 / 0 |
| Loss scale | 65,536 throughout | 65,536 throughout |
| GradCache replay difference | 0.0 at every step | 0.0 at every step |
| Learning rate | as planned: linear warmup to 2e-5 over 6 steps, then linear decay; the same sequence as seed 42 | as planned |
| Time per step | 3.34 s (184 s of updates) | 3.63 s (200 s) |
| Peak GPU memory | 9.22 GiB allocated; 10,309 MiB on `nvidia-smi` (GPU 0; GPU 1 idle) | 9.22 GiB; 10,335 MiB |
| Highest temperature; mean utilisation | 83 °C; 91.7 % | 81 °C; 91.7 % |

**Loss.**
- This is in-batch InfoNCE on each step's own batch. It is noisy from step to step (for example 1.227 at step 3, 0.237
  at step 40) because the batches differ.
- Its epoch means fall steadily. It is not a retrieval metric.

**Gradient clipping.**
- Every update was clipped, as in seed 42. This is the approved configuration: at temperature 0.02, the raw gradient
  norm is always above 1.0.
- The norm falls through training (epoch means 10.1 → 3.7). The clip sets the size of every update, which is part of
  the training dynamics and should be disclosed in the report.

**Numerical stability.**
- There was no fp16 overflow: the scaler never halved its scale, and its growth tracker counted 55 finite steps in a
  row. No update was skipped.
- The GradCache replay was exact at every step.

**Validation curve.**
- Seed 1337 rises more slowly than seed 42 (0.4096 against 0.4251 at step 5).
- It peaks sharply at step 40, then falls back 0.0065–0.0095 for steps 45–55.
- Seed 42 instead plateaued at 0.6665–0.6690 from step 35 on.
- The selected values are 0.6673 (seed 1337, step 40) and 0.6690 (seed 42, step 55), a difference of 0.0018.

**How to read these values.** The validation questions are synthetic and AI-reviewed. They choose a checkpoint within a
run; they are not an evaluation result, and no improvement is claimed from them.

## 7. Against seed 42: configuration, schedule and environment (item 6)

| Item | Seed 1337 against seed 42 |
|---|---|
| Config | `7a4c4618…` for both: the same file |
| `run.json` | Differs only in `config.training.seed` (1337 against 42), `run` and `schedule.sha256`. Every other field is identical, including:<br>• **hyperparameters:** learning rate 2e-5, batch 64, mini-batch 16, temperature 0.02, warmup ratio 0.1, weight decay 0.01, `max_grad_norm` 1.0, fp16, 5 epochs;<br>• **base model:** revision `a5beb1e3…`;<br>• **inputs:** corpus `ff9dadf3…`, candidates manifest `0586c7ca…`, training pair IDs `58df5d9d…`, validation manifest `40238cb0…` and IDs `b83abf1a…`, preparation `21c2293a…`. |
| Schedule | `795095dc…` against `44b0ed7c…`. Both have the same shape: 55 steps, warmup 6, balanced 11×5, none dropped, at least 60 in-batch negatives (mean 60.64). Only the order differs, by seed, as designed. Each matches its pinned hash. |
| Learning rate sequence | identical |
| Selection rule and validation record | identical. `selection.json` differs only in `seed`, `run` and `schedule_sha256`, apart from the evaluations themselves. |
| Checkpoint metadata (step 55) | differs only in the seed, run, schedule, measured memory and validation values |
| Environment identity | equal on all nine keys: Python, torch, transformers, tokenizers, CUDA, cuDNN, device, GPU and compute capability |
| `pip freeze` | byte-identical (`5432fefb…`) |
| Driver | 580.178.04 for both (seed 42's is in its `environment.json`, from `nvidia-smi`) |
| Preflight | both ready, with no failed check |
| Notebook | Seed 1337 used version 2, seed 42 version 1. The trainer, config and data are the same. Version 2 adds only notebook-side safeguards, approved 2026-11-10: the batch-only gate, the execution identity record and the five pre-training checks. |

## 8. No development or evaluation labels (item 7)

**Not available to the run.**
- **The bundle:** archive `efc377d9…`, manifest `a6615850…`. Kaggle verified both before training, with 104 files by
  role.
- **What it contains:**
  - code;
  - configs;
  - under `data/processed/`: the corpus, the training and validation pairs, the candidates and the preparation records;
  - under `data/train_review/` and `data/validation_review/`: the AI quality-review decisions. These are preparation
    inputs, not gold labels.
- **What it lacks:** it has no `data/eval/` directory. The repository's evaluation material all lives there: the gold
  qrels and queries, the AI development labels, the annotations and the audits.
- **How this was established:** I listed only the file names under `data/eval/`, to confirm it is absent from the
  bundle. No label file was opened.

**Not read.** The trainer's inputs are bound by hash in `run.json`:
- the corpus, `ff9dadf3…` (10,137 passages);
- `train_candidates_v3`, `0586c7ca…` (678 pairs, IDs `58df5d9d…`);
- `validation_pairs_v4`, `40238cb0…` (93 questions, IDs `b83abf1a…`);
- the preparation, `21c2293a…`.

**The code path.** From the evaluation packages, the training code imports only:
- metric formulas (MRR, nDCG, recall);
- a corpus-passage loader;
- a file hasher.

None of them loads labels.

**Selection.**
- It used only `validation_pairs_v4`: the 93 per-question IDs hash to the pinned value.
- The run records its isolation check against the 678 training pairs: no shared passage or equivalent, and no
  lexically overlapping question.

**No trace.**
- No record or log in the archive mentions qrels, development labels or the evaluation set.
- `selection.json`, every checkpoint's metadata and the verification each state that no development or evaluation
  metric was computed.

**This audit** opened no label, qrels or judged-passage file, and made no API request.

## 9. Discrepancies and limitations (item 8)

No discrepancy affects the run's validity or its selection.

| # | Finding | Effect | Proposed handling |
|---|---|---|---|
| W1 | `stopped.early: true` and `rule_stop: step 55` appear although all 55 steps ran. Here `early` means the rule's patience ran out, which happened at the planned last step. | Could be misread as a shortened run | Explain it in the seed 1337 report, as here. No code change: the trainer is shared by all seeds. |
| W2 | At step 55 the console prints its generic stop message: "training and evaluation continue to step 55, recorded as never selectable". Nothing followed. | Cosmetic | None; the trainer stays unchanged. |
| L1 | GPU training is not bit-reproducible, as with seed 42. | Another execution could select differently, though that is less likely here, given the margins (§3). | Designate execution `b98f183e` as the run of record. |
| L2 | One execution per seed is only partly enforced. The gate refuses interactive runs, but would also allow a second saved version. Only one seed 1337 archive exists on this machine. | It cannot be confirmed from here that no other saved version trained. | Please confirm in the Kaggle notebook's version history that exactly one version ran this notebook. |
| L3 | The notebook file is not in the outputs. Its identity is bound indirectly: the embedded execution code `3cd0bc44…` and verification script `df3e4038…` (identical in the outputs), and the pinned seed, config and schedule. | A weaker binding to `19bbf437…` | Optional: download the saved version's notebook and compare its code cells. |
| L4 | The CUDA-only checks, re-evaluation and the resume restore, ran on the T4 only. Here, the resume state was read on CPU, and the hashes and step-0 identity were checked. | None in practice; step 0 equals the rehearsal and seed 42 exactly. | None |
| L5 | Datasets attached on Kaggle other than the bundle are not recorded. | The trainer reads only hash-bound bundle files, but the list of attached datasets is not on record. | Optional, for seed 2026: record the `/kaggle/input` names. This is a notebook-only change and needs approval. |
| L6 | The validation set is synthetic, AI-reviewed and has 93 questions. | It is for selection only; no improvement is claimed. | As designed |
| L7 | Every step was clipped (the approved config). | The clip shapes the training dynamics. | Disclose it in the report, as for seed 42. |

## 10. Recommendation

1. **Designate execution `b98f183e` as seed 1337's run of record.** Its selected checkpoint is
   `checkpoint-step-000040`, digest `8237ec0b82a12830c5e41653f3597fb82516475203b8e61b262462a40597b675`.
2. **Keep `~/Downloads/kaggle_a1_seed1337_outputs.zip` unchanged.** It is the only copy of the 12 checkpoints and of
   the resume state. Keep it with seed 42's archive.
3. **Before seed 2026:**
   - confirm L2;
   - decide on L5 (optional);
   - run it in the same pinned Kaggle environment, because check 1 halts on any other Python or package version.
