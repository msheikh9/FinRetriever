# Seed 2026: final audit

## Official record (2026-11-12)

**Seed 2026 is officially accepted (D1):** execution `24187107-8edd-4f78-a67a-5d93d8ec3eaf`, selected checkpoint
`checkpoint-step-000040`, digest `cba9110b7f744b0c47c3a8d428b61f3fd44224ef8aea779a5ce8b54e83c54d6c`, as proposed below.

**Open items:**
- **L2, closed (D2):** the user checked the Kaggle version history; it shows exactly one seed 2026 training
  execution.
- **N1 and L3, closed without further evidence:** no additional Kaggle work is required (D2).
  - The NameError's source stays as analysed in §6: it cannot have come from the batch execution's packaging cell,
    and it does not affect the run.
  - The notebook's binding to `c6ca9e5b…` stays indirect, as described in L3.

All three seeds are now official: seed 42 step 55, seed 1337 step 40, seed 2026 step 40. The three-seed validation
decision has been run ([a1_three_seed_summary.md](a1_three_seed_summary.md) §6).

The audit below is unchanged.

## Run of record (as proposed by this audit; accepted above)

| | |
|---|---|
| Selected checkpoint | `checkpoint-step-000040`, digest `cba9110b7f744b0c47c3a8d428b61f3fd44224ef8aea779a5ce8b54e83c54d6c`, whole-corpus validation MRR@10 0.6580005120 |
| Its weights | `model.safetensors` SHA-256 `8f9dbd383b01074f5e56c03b9420155d29a758396a65d1b515e8ac95251a927b` |
| Selection record | `selection.json` (SHA-256 `2afbdae2…`), unchanged. It holds 12 candidate evaluations. The rule's third miss falls at step 55, the planned last step, so there is no after-stop evaluation. |
| Archive | `kaggle_a1_seed2026_outputs.zip`: 6,137,035,311 bytes, SHA-256 `7e79fa35c8a54e9582154124546458c767583fbea741fc8acf72b8555273f502`; one download, in `~/Downloads`. Its records are extracted in `dist/kaggle_a1_seed2026_results/`. |
| Execution | `24187107-8edd-4f78-a67a-5d93d8ec3eaf`, a saved version (`KAGGLE_KERNEL_RUN_TYPE = "Batch"`). Training ran 2026-11-11 23:23:08–23:59:35 UTC on a Tesla T4: CUDA 13.0, driver 580.178.04, torch 2.11.0+cu130, Python 3.13.16. |
| Notebook | `dist/kaggle_a1_seed_v3/finretriever_a1_train_seed2026.ipynb` as delivered (SHA-256 `c6ca9e5b…`), `START_OPTIMIZER = True`. The binding is indirect (§8, L3). |

**Date of the audit:** 2026-11-12.

**Scope:** your items 1–6 and 8; item 7, the three-seed report, is [a1_three_seed_summary.md](a1_three_seed_summary.md).
I recomputed everything below on this Mac from the archive itself, independently of the notebook's own verification,
except where a section says otherwise.

**Nothing was changed.**
- **Not done:** no training, no API request, no commit or push. No frozen dataset, historical record or selection
  record was changed. No label, qrels or judged-passage file was opened.
- **Re-checked after the audit (§7):** the six archives and every extracted record are byte-identical to their
  earlier hashes.

## 1. Conclusions

1. **The archive is complete and intact.**
   - All 134 listed files were re-hashed from inside the zip: none mismatched, none unlisted, none missing.
   - The zip's SHA-256 equals its `.sha256`.
2. **The selection is correct: `checkpoint-step-000040`** (item 1).
   - All 12 values were recomputed exactly from the 93 per-question ranks, and each equals the record.
   - The K1 rule, re-implemented without the repository's code, selects step 40.
   - The repository's own reader accepts the record, and `select_through` returns `(40, 55)`.
3. **Eligibility and stopping are correct** (item 2).
   - Step 30 was a miss, and step 35 reset the count. Steps 45, 50 and 55 are three misses in a row, and the third
     falls at step 55, the planned last step.
   - So `rule_stop` is step 55, `stopped.early` is true and there are no after-stop evaluations.
   - All 12 checkpoints were selectable candidates. The plain arg-max over all 12 is also step 40.
4. **The selection is fragile, like seed 42's and unlike seed 1337's.**
   - Step 40 leads step 55 by 0.0028 MRR@10 (0.26 in summed reciprocal rank) and step 35 by 0.0051 (0.47).
   - Of 1,397 possible single-question, single-rank moves, 75 would change the selected checkpoint, to step 35 or
     step 55. All three candidates lie within 0.0051 MRR@10 of one another.
   - This does not affect validity: the rule was applied as written. The decision test re-applies the rule to every
     resample, so this sensitivity is part of what it measures.
5. **The execution identity and every checkpoint binding hold** (item 3).
   - It is a positively identified saved-version (batch) execution, with the gate open.
   - All six pre-training checks passed on the T4 before the optimizer started.
   - All 12 checkpoint digests, rebuilt from the zip's bytes, match `selection.json`, `validation_log.jsonl` and the
     checkpoint metadata.
   - The resume state is only in step 55, its hash matches, and read independently it agrees with the records.
6. **The dataset provenance is confirmed** (item 4).
   - The training dataset `finretriever-train-v3-a1-r2` was identified and confirmed against the pinned manifest
     before training.
   - The record is bound to the gate by its digest. The attached-input listing is byte-for-byte the one seen in the
     interactive check: the r2 bundle and its `.sha256`, nothing else.
7. **The packaging-cell NameError does not affect the run** (item 5, §6).
   - The archive proves that, in the batch execution, `train_code` was defined and equal to 0 when the packaging cell
     began.
   - Packaging then completed, and its output verifies.
   - A NameError for `train_code` cannot have come from that execution's packaging cell. It most likely came from
     another session in which the training cell never ran. Either way, nothing it could have affected is part of the
     run's evidence.
8. **Training was numerically clean** (§5).
   - All 55 of 55 steps ran; the loss fell from 1.5173 to 0.1056.
   - There was no non-finite loss or gradient norm, and the loss scale stayed at 65,536 with no overflow.
   - The GradCache replay difference was 0.0 at every step. Every step was clipped, as in the other seeds.
9. **The configuration, procedure and environment are the same as seeds 42's and 1337's.**
   - `run.json` differs from each only in the seed, the run name and the schedule hash.
   - The environment identity is equal, `pip freeze` is byte-identical, and the driver is the same.
   - Step 0 is identical across all three seeds: the same weights and the same 93 ranks.
10. **No development or evaluation labels were used.**
    - The bundle contains no evaluation data, and the preflight's isolation check passed on the T4.
    - Selection used only `validation_pairs_v4`.
11. **Proposed run of record** (item 6): execution `24187107`, `checkpoint-step-000040`. §8 lists one open question
    and the limitations.

## 2. Evidence

| Item | Value |
|---|---|
| Archive | 6,137,035,311 bytes (5.72 GiB), SHA-256 `7e79fa35…`, equal to its `.sha256`. It is stored without compression, with one top directory, `kaggle_a1_seed2026_outputs/`. |
| Contents | `OUTPUT_SHA256SUMS` lists 134 files; with the list itself, the zip has 135 members. There are 12 checkpoints with 9 hashed files each, and one `trainer_state.pt`. |
| Extraction | `dist/kaggle_a1_seed2026_results/`: 122 records, each equal to its listed hash. The weights and the resume state stay in the zip. |
| Timeline (UTC) | notebook start 2026-11-11 23:16:53; dataset identified 23:16:53 and confirmed 23:17:02; gate opened and training started 23:23:08; training ended 23:59:35 (36 min 27 s). Then results, verification (417.9 s) and packaging. `status.json`: exit 0 for training, results and verification. |
| The notebook's own verification | `run_verification.json`: 6 of 6 checks pass (run identity, checkpoints, resume state, re-evaluation, execution identity, dataset provenance), and the run directory is unchanged |
| Logs | No error, traceback or warning line in any of the 8 console logs in the archive |

## 3. Checkpoint selection, recomputed (items 1 and 2)

**Columns:**
- **Σ RR:** the sum over the 93 questions of the reciprocal rank, cut at 10 (MRR@10 × 93).
- **Misses:** candidate evaluations in a row without improvement.
- **Digest:** the checkpoint digest, rebuilt from the zip.
- **Seed 42, seed 1337:** those seeds' values at the same step, for reference only.

Every row is a candidate and selectable, both in `selection.json` and in the checkpoint's `metadata.json`.

| Step | MRR@10 | Σ RR | Improved | Misses | Digest | Weights | Seed 42 | Seed 1337 |
|---:|---:|---:|---|---:|---|---|---:|---:|
| 0 | 0.2658858167 | 24.7274 | yes | 0 | `e080e213…` | `0d4db737…` | 0.265886 | 0.265886 |
| 5 | 0.4110343062 | 38.2262 | yes | 0 | `fe1c4c23…` | `1c7c21eb…` | 0.425098 | 0.409643 |
| 10 | 0.5659455538 | 52.6329 | yes | 0 | `17d22885…` | `5296e087…` | 0.586781 | 0.569632 |
| 15 | 0.6315113501 | 58.7306 | yes | 0 | `9b2d3893…` | `6b79eabd…` | 0.632433 | 0.610305 |
| 20 | 0.6471923536 | 60.1889 | yes | 0 | `38035ca5…` | `9cfc8aaf…` | 0.643544 | 0.633824 |
| 25 | 0.6525644308 | 60.6885 | yes | 0 | `9887fb91…` | `f86d70ac…` | 0.658252 | 0.634016 |
| 30 | 0.6511606076 | 60.5579 | **no** | 1 | `330dab0d…` | `cd0a0bf8…` | 0.662826 | 0.639021 |
| 35 | 0.6529399215 | 60.7234 | yes | 0 | `ca042cec…` | `8be60713…` | 0.668570 | 0.650533 |
| **40** | **0.6580005120** | 61.1940 | yes | 0 | `cba9110b…` | `8f9dbd38…` | 0.666500 | 0.667273 |
| 45 | 0.6489631336 | 60.3536 | **no** | 1 | `0f86fd71…` | `bcc72bf1…` | 0.668070 | 0.657744 |
| 50 | 0.6493215566 | 60.3869 | **no** | 2 | `ec8e4baf…` | `ab015474…` | 0.668920 | 0.658508 |
| 55 | 0.6552099334 | 60.9345 | **no** | 3 | `07093d38…` | `2445296c…` | 0.669039 | 0.660719 |

### Independent recomputation (item 1)

**Values.**
- Each value was recomputed with exact fractions from the 93 whole-corpus `first_relevant_rank` values: 1/rank if
  the rank is at most 10, otherwise 0.
- All 12 equal the recorded values (difference below 1e-12), and so does every per-question `mrr@10`.

**Questions.**
- There are 93 unique pair IDs, the same ones in the same order in all 12 evaluations.
- Their hash is `b83abf1a…`, the pinned `validation_pairs_v4` value.
- The IDs come only from `val_pilot_v1`, `val_pilot_v2e` and `val_pilot_v2f`.

**Step 0.** The base model is identical across all three seeds:
- its ranks hash to `0de690c4…`, the rehearsal's value;
- its per-question results equal seed 42's and seed 1337's step 0;
- its weights (`0d4db737…`) are byte-identical to theirs.

**The rule, re-implemented from its approved text:**
- candidates are step 0 and every 5 steps;
- a value counts as an improvement only if it is strictly greater at 12 decimals;
- selection stops after 3 candidate evaluations in a row without improvement.

The result is: select step 40, with the rule stopping at step 55.

**Flags.**
- `candidate`, `improved` and `since_best` in `selection.json` equal the re-implementation at all 12 steps.
- `validation_log.jsonl` equals the record entry for entry: step, value and digest.
- Every checkpoint's `metadata.json` has `candidate: true`, `selectable: true`, the same value and set
  `validation_pairs_v4`.

**The repository's code.** `read_selection_record` accepts the record (it re-derives the rule and checks
completeness), and `select_through` returns `(40, 55)`.

### K1 eligibility and stopping (item 2)

**The record:**
- `selected` is step 40;
- `rule_stop` is step 55 (`07093d38…`, 0.6552099334);
- `stopped` is `{early: true, at_step: 55, planned_steps: 55}`;
- `after_stop_evaluations` is empty.

**How the count ran.** Step 30 (0.65116) fell below step 25 (0.65256): miss 1. Step 35 (0.65294) beat step 25, so
the count reset. Step 40 improved again. Steps 45, 50 and 55 each fail to beat step 40, and the third miss is step
55.

**What `early: true` means here,** as for seed 1337:
- it is the rule's own flag: the patience of 3 misses ran out;
- the stop coincides with the planned last step, so training was not shortened, and no evaluation came after it.

**Eligibility.**
- Every checkpoint from 0 to 55 was evaluated before the stop and was a selectable candidate.
- Step 55 is itself a candidate: it is the third miss.
- No checkpoint was excluded by the stop.

**After-stop evaluations.** None exist in this run. That they could never be selected is guaranteed in code and in
tests ([seed42_final_audit.md](seed42_final_audit.md) §5).

**The decision rule's completeness precondition** ("every planned evaluation") holds: all 12 planned evaluations are
recorded.

### Margins and sensitivity

| Against step 40 | ΔMRR@10 | ΔΣ RR | Questions whose RR differs | Can one question moving one rank change the selection? |
|---|---:|---:|---:|---|
| step 25 | −0.005436 | −0.5056 | 26 | no |
| step 30 | −0.006840 | −0.6361 | 18 | no |
| step 35 | −0.005061 | −0.4706 | 13 | **yes**: a rank 2 → 1 move at step 35 (gain 0.5) |
| step 45 | −0.009037 | −0.8405 | 7 | no |
| step 50 | −0.008679 | −0.8071 | 8 | no |
| step 55 | −0.002791 | −0.2595 | 9 | **yes**: a rank 2 → 1 move at step 55 |

**An exhaustive check.** I tried every single-question, single-rank move (up or down) at every evaluation, one at a
time, and re-applied the repository's rule (`select_through`). Of 1,397 such moves, **75 change the selection**:

| Move | Count | Would select |
|---|---:|---|
| a question at step 40 drops from rank 1 to 2 | 50 | step 35; the rule would then stop at step 50 |
| a question at step 35 rises from rank 2 to 1 | 13 | step 35 |
| a question at step 55 rises from rank 2 to 1 | 12 | step 55 |

**For comparison,** the same check on the other seeds:

| Seed | Moves tried | Moves that change the selection |
|---|---:|---:|
| 42 | 1,374 | 251 |
| 1337 | 1,354 | 0 |
| 2026 | 1,397 | 75 |

**How to read this.**
- **No error.** The selection follows the pre-registered rule exactly.
- **Small stakes.** Steps 35, 40 and 55 are within 0.0051 MRR@10 of each other. Selecting another of them would
  change the reported value by at most that much.
- **Not overridden.** Choosing another checkpoint after seeing these values would override the rule. That was
  rejected for seed 42 for the same reason ([seed42_final_audit.md](seed42_final_audit.md)).
- **Accounted for.** The decision test re-applies the rule to every sign-flipped curve, so selection noise of this
  kind is part of its null distribution.

**Counterfactuals,** with the repository's `select_through` applied to this run's recorded values:

| Change | Rule outcome |
|---|---|
| none (as recorded) | select 40; stop at 55 |
| any one of steps 45, 50, 55 exactly equal to step 40 | select 40 (ties go to the earlier step); stop at 55 |
| any one of steps 45, 50, 55 above step 40 by a single rank-9-to-10 move | that step is selected; no stop |
| step 40 no better than step 35 | select 35; stop at 50 |
| arg-max over all 12, ignoring the stop | 40 |

## 4. Execution identity and checkpoint hashes (item 3)

### Execution identity

**`execution.json`**, and its byte-identical copy in the run directory:

| Field | Value | Check |
|---|---|---|
| Execution type | `batch`, from `KAGGLE_KERNEL_RUN_TYPE = "Batch"` | The gate requires exactly `"Batch"`. |
| Indicators | 12 `KAGGLE_` variable names (names only, no values); `/kaggle/src/script.ipynb` present; a papermill parent process | The same pattern as the verified batch probe and seed 1337; not the interactive pattern (`jupyter-server`) |
| Execution code | `3cd0bc44…` | The code the Kaggle probes verified |
| Execution ID | `24187107…` | New; distinct from the probes, seed 1337 (`b98f183e…`) and the interactive check (`95bee077…`) |
| Gate | open; decided 23:23:08 UTC; `start_optimizer` true; `training_dataset_confirmed` true; `datasets_sha256` `5e799308…` | The digest equals the one recomputed from the recorded provenance (§5). |
| Training | 23:23:08 → 23:59:35 UTC, exit 0 | Times in order, inside the GPU monitor's window (23:23:08.1–23:59:33.0) |
| Environment | Tesla T4 (compute 7.5, 14.56 GiB), CUDA 13.0, cuDNN 92700, driver 580.178.04, torch 2.11.0+cu130, transformers 5.3.0, tokenizers 0.22.2, Python 3.13.16 | Equal to `selection.json`'s environment on every identity key, and to seeds 42 and 1337 |
| Seed, config, schedule, bundle, manifest | 2026, `7a4c4618…`, `49b2b153…`, `efc377d9…`, `a6615850…` | Equal to `pins.json`, `run.json`, `selection.json` and `pretraining_checks.json`. The schedule is the one both earlier T4 preflights recorded for seed 2026. |
| `run_files` | `run.json` `ea1d1d2f…`, `selection.json` `2afbdae2…` | Equal to those files' bytes in the zip |

**Pre-training checks** (`pretraining_checks.json`). All six passed on the T4 before the optimizer was allowed to
start, with `mismatch: {}`.

| Check | Found on the T4 |
|---|---|
| 0. Training dataset | `finretriever-train-v3-a1-r2` at `datasets/yousefkndesh/finretriever-train-v3-a1-r2`, nested, unpacked, manifest `a6615850…` |
| 1. Python and packages | Python 3.13.16; the 13 pinned packages equal seed 42's; `pip freeze` identical to seed 42's (`5432fefb…`) |
| 2. GPU and CUDA | T4, compute 7.5, CUDA 13.0, cuDNN 92700, driver 580.178.04; the preflight's `cuda_fp16`, `gradcache` and `gpu_memory` all pass |
| 3. Config and schedule | config `7a4c4618…`; schedule `49b2b153…` (55 steps, warmup 6, balanced 11×5, none dropped) |
| 4. Candidates and validation | 678 pairs (`train_candidates_v3`, manifest `0586c7ca…`), 93 questions (`validation_pairs_v4`, manifest `40238cb0…`, IDs `b83abf1a…`), 10,137 passages |
| 5. Execution mode | `batch` (`"Batch"`) |

### Checkpoint hashes

**Every file.** Re-hashed from inside the zip against `OUTPUT_SHA256SUMS`: 134 of 134 match.

**Checkpoint digests.**
- **Method:** each checkpoint's file map was rebuilt from the zip's bytes. It covers all 9 files except
  `trainer_state.pt`, including `metadata.json`. Its digest was recomputed with an independent canonical JSON.
- **Result:** for all 12 checkpoints, the file map and digest equal `selection.json`, and the digest equals
  `validation_log.jsonl`. The selected entry's digest equals its evaluation's digest.

| | Digest | `model.safetensors` |
|---|---|---|
| step 0 | `e080e21367974b32baa0ea6468fb96d30eda2497e6ebfdc0897d244cd5c43ef9` | `0d4db737f56aaea90796b5a8d219de0eee958295a575c611f6b417ad340151da` (= seeds 42 and 1337, step 0) |
| **step 40 (selected)** | `cba9110b7f744b0c47c3a8d428b61f3fd44224ef8aea779a5ce8b54e83c54d6c` | `8f9dbd383b01074f5e56c03b9420155d29a758396a65d1b515e8ac95251a927b` |
| step 55 (last) | `07093d38d60dacaced2f46300404341810ec27ebf67a040daf9489913ad6afd0` | `2445296c4f6886d2bd2215d7246e85b5600254aabbc867909b09562b639877ac` |

Step 0's digest differs from the other seeds' only because its `metadata.json` names the seed and run. Its weights
are identical.

**The record chain.**
- `execution.json`'s `run_files` bind `run.json` and `selection.json`.
- `selection.json`'s `run_record` binds `run.json` (`ea1d1d2f…`).
- Every record carries config `7a4c4618…` and schedule `49b2b153…`. So does each checkpoint's `metadata.json`, which
  is part of the checkpoint's digest.

**Resume state.**
- **Location:** only in `checkpoint-step-000055`, as designed.
- **Hash:** 871,639,399 bytes, SHA-256 `80794e00f4eca8a65e8842273c7d2e4323aa8ddf16a8a389d57f2fdd74e46ae1`, as listed.
- **Read independently:** on CPU with torch's restricted loader (`weights_only=True`), as the trainer loads it.

| Part | Contents | Agrees with |
|---|---|---|
| Step, schedule | 55; `49b2b153…` | `run.json` |
| AdamW | moments for 197 of 199 parameters, all at step 55, all finite (the same pooler pattern as seeds 42 and 1337). Two groups: weight decay 0.01 (76 parameters) and 0.0 (123); initial learning rate 2e-5, betas (0.9, 0.999), eps 1e-8; learning rate now 0 | the config |
| Scheduler | `last_epoch` 55, learning rate 0 | the planned schedule |
| Scaler | scale 65,536; growth tracker 55 (55 overflow-free steps in a row) | `log.jsonl` (65,536 at every step) |
| Random state | Python, torch and CUDA | |
| Selection state | best step 40 (0.6580005120), 3 misses, evaluations 0–55, none after the stop; values and digests equal `selection.json` | the record |
| History | equal to `log.jsonl` | |

**Checked on the T4 only** (the archive's own verification; not repeated here, because this Mac has no CUDA):
- the step-55 state restores exactly, and its AdamW moments equal the saved ones;
- the reloaded checkpoints 40 and 55 reproduce their recorded validation results exactly;
- step 0 ranks the validation set as the rehearsal did.

## 5. Dataset provenance (item 4)

This is the first seed with the version 3 provenance record
([training_a1_seed2026_readiness.md](training_a1_seed2026_readiness.md)).

| Check | Result |
|---|---|
| Recorded before training | identified 23:16:53 and confirmed 23:17:02, before the gate decision and training start at 23:23:08. The order is identified ≤ confirmed ≤ gate ≤ training. |
| Training dataset | `finretriever-train-v3-a1-r2` at `datasets/yousefkndesh/finretriever-train-v3-a1-r2`: `nested`, `unpacked`, one copy (`found` has one entry), no problems |
| Manifest | `a6615850…`, equal to the pin in `execution.json`, `pins.json` and the bundle the verification ran from. It names the bundle `finretriever_train_v3_a1_r2`. |
| Archive hash | `null`, as designed for the unpacked form: Kaggle unpacked the archive. The pinned manifest binds every file, and the bundle-verification cell checked each one. |
| Name pin | the recorded name equals the expected name and `pins.json`'s `dataset` |
| Attached inputs | one entry, `datasets`: 107 files, 56,908,199 bytes, listing digest `55628bac…`. This is identical to the interactive check, which I matched to the r2 bundle plus its `.sha256`, and nothing else. |
| Gate binding | `training_dataset_confirmed: true`. `datasets_sha256` `5e799308…` equals the digest recomputed here from the recorded provenance. |
| Run verification | check 6, `dataset_provenance`, passed on the T4. No input `run.json` records is missing from the manifest. |
| Kaggle's numeric identifiers | Not available. The saved version's `/kaggle/src/script.ipynb` existed but carried no `metadata.kaggle.dataSources`, so they are recorded as unavailable. It did carry `accelerator: gpu`, `isInternetEnabled` and `isGpuEnabled`. As designed, nothing depends on them (§8, L8). |
| No secrets | only variable names are recorded, never values; no file content was recorded |

## 6. The packaging-cell NameError (item 5)

**What you reported:** the last cell raised a NameError because `train_code` was undefined. The training, results
and verification exits were nevertheless recorded as 0, and the archive was created and verifies.

**What the packaging cell does,** in order:
1. `status = {"training_exit": train_code, "results_exit": results_code, "verification_exit": verify_code}`;
2. write `status.json`;
3. `package()`: hash every file into `OUTPUT_SHA256SUMS` (including `status.json`), then write the zip and its
   `.sha256`;
4. print the status, then the final message: `if train_code != 0: … elif … else: print("SEED 2026 TRAINED AND
   VERIFIED …")`.

**What the archive proves about the batch execution.**
- **`status.json` exists,** with `training_exit: 0`. Statement 1 reads `train_code`, so `train_code` was defined
  and equal to 0 when the cell began. It was set by the training cell, which also recorded `exit: 0` in
  `execution.json` at 23:59:35.
- **Packaging completed:** `OUTPUT_SHA256SUMS` lists `status.json` with its exact bytes, and the zip verifies
  against both.
- **Nothing in between can undefine the name.** `package()` reads the output paths but never assigns or deletes a
  global name. No cell deletes `train_code`, and only the training cell assigns it.

So in the batch execution, `train_code` could not have been undefined at step 4. If statement 1 had raised a
NameError, `status.json` and the zip would not exist.

**The likely source** is a different session, one in which the training cell never ran. Examples: the editor's
interactive session, if the last cell was run there before or after the saved version, or after the interactive
gate refused. In such a session:
- the first statement of the cell raises exactly "name 'train_code' is not defined";
- no training can occur, because the interactive gate refuses;
- its kernel and `/kaggle/working` are separate from the saved version's, so it cannot touch the archive.

**Why it does not invalidate the run,** wherever it came from:
- **The exits:** training, results and verification are each recorded twice: in `status.json`, and in
  `execution.json` and `run_verification.json`. All are 0 or passing.
- **The archive:** it is complete. It verifies here against `OUTPUT_SHA256SUMS`, and against its `.sha256`, which
  `package()` wrote after the zip was closed.
- **The independent checks:** every check this audit repeated (§1–§5) passes from the archive alone, with no use of
  the final message.
- **The only thing missing:** at most, the final printed line. It is a message, not a record.

**To close it (optional, item N1 in §8):** download the saved version's notebook with its outputs (the version
page's notebook download, or *File → Download Notebook* while viewing that version), and its log. This shows which
cell and session raised the error. It also binds the version's code cells to `c6ca9e5b…`, which closes L3. No
notebook change is proposed: the seed notebooks are frozen records, and no further seed is planned.

## 7. Archives and historical records unchanged (item 8)

The hashes after this audit were compared with the hashes taken before it, and before the seed 2026 work.

| What | Before | After this audit |
|---|---|---|
| Seed 42: the four downloads | `4a3b9d4a…` | `4a3b9d4a…` ×4 |
| Seed 1337 archive | `7f0317af…` | `7f0317af…` |
| Seed 2026 archive | `7e79fa35…` (its `.sha256`) | `7e79fa35…` |
| Protected files: seed 42 and 1337 extracted records, both notebook directories, the r2 bundle, the probe records, `configs/train/` | 282 files | byte-identical |
| Seed 2026 extracted records (`dist/kaggle_a1_seed2026_results/`), as you extracted them | 122 files | byte-identical |
| `train_candidates_v3` and `validation_pairs_v4` | 7 files | byte-identical |
| Selection records | seed 42 `19cca1cb…`, seed 1337 `165064f5…`, seed 2026 `2afbdae2…` | unchanged |
| Tracked files (345) | as at the end of the seed 2026 preparation | none changed. This audit adds two new, untracked documents: this one and [a1_three_seed_summary.md](a1_three_seed_summary.md). |

## 8. Discrepancies and limitations

No discrepancy affects the run's validity or its selection.

| # | Finding | Effect | Proposed handling |
|---|---|---|---|
| N1 | A NameError for `train_code` was reported from the packaging cell. The archive shows it cannot have come from the batch execution's packaging cell (§6). | None on the run. The only possible loss is the final printed line. | Optional: download the saved version's notebook with outputs and its log, to locate it. |
| W1 | `stopped.early: true` and `rule_stop: step 55` appear although all 55 steps ran. As for seed 1337, `early` means the rule's patience ran out, here at the planned last step. | Could be misread as a shortened run | Explain it in the report, as here |
| W2 | At step 55 the console prints the generic "training and evaluation continue to step 55, recorded as never selectable"; nothing followed. | Cosmetic | None; the trainer stays unchanged |
| L1 | GPU training is not bit-reproducible, as with the other seeds. | Another execution could select differently. Given L5, that is more likely here than for seed 1337. | Designate execution `24187107` as the run of record |
| L2 | One execution per seed is only partly enforced. The gate refuses interactive runs but would allow a second saved version. One seed 2026 archive exists on this machine. | It cannot be confirmed from here that no other saved version trained | Please confirm in the Kaggle version history that exactly one saved version ran this notebook |
| L3 | The notebook file is not in the outputs. Its identity is bound indirectly: the execution code `3cd0bc44…`, the verification script `679ff007…` (identical in the outputs), the "generator v3" label, the version 3 provenance record, and the pinned seed, config and schedule. | A weaker binding to `c6ca9e5b…` | The download proposed for N1 also closes this |
| L4 | The CUDA-only checks (re-evaluation, the resume restore) ran on the T4 only. | None in practice. Step 0 equals the rehearsal and both other seeds exactly. | None |
| L5 | The selection is fragile: 75 of 1,397 single-rank moves change it (§3). | The selected value would change by at most 0.0051 MRR@10 | Disclose it; the decision test accounts for it. Never override the rule. |
| L6 | The validation set is synthetic, AI-reviewed, 93 questions, from the same generator and filings as the training pairs. | It selects checkpoints only; no improvement is claimed from it. | As designed |
| L7 | Every step was clipped (the approved config). | The clip shapes the training dynamics | Disclose it in the report, as for the other seeds |
| L8 | Kaggle's numeric dataset identifiers were not available in the saved version's notebook copy (readiness P1). | None. The name, the pinned manifest and the listing digest identify the dataset. | As designed |

## 9. Recommendation (item 6)

1. **Designate execution `24187107-8edd-4f78-a67a-5d93d8ec3eaf` as seed 2026's run of record.** Its selected
   checkpoint is `checkpoint-step-000040`, digest
   `cba9110b7f744b0c47c3a8d428b61f3fd44224ef8aea779a5ce8b54e83c54d6c`, weights `8f9dbd38…`.
2. **Keep `~/Downloads/kaggle_a1_seed2026_outputs.zip` unchanged,** with the seed 42 and 1337 archives. It is the
   only copy of the 12 checkpoints and of the resume state.
3. **Confirm L2** (one saved version in the Kaggle version history). N1 and L3 can optionally be closed with one
   notebook download.

**Audit tools** (scratch, not in the repository), in
`<session scratch directory>/seed2026audit/scripts/`:
- `archive_audit.py`, `recompute.py`, `state_check.py` and `compare.py`: the seed 1337 tools, generalised to any
  seed;
- `sensitivity.py`: the exhaustive single-rank check;
- `three_seeds.py`: the side-by-side tables;
- `preconditions.py`: the decision rule's preconditions only, with no test.

Their JSON outputs are in the directory above.
