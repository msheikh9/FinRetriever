# Seed 42: final selection audit

## Official record (decision approved 2026-11-10)

**Seed 42's official result is `checkpoint-step-000055`** of the verified Save & Run All execution.

| | |
|---|---|
| Selected checkpoint | `checkpoint-step-000055`, digest `24c2aaae977ce319110a7dd979cb365fccfdac6cc89d63b1ba566e31a678beb9`, whole-corpus validation MRR@10 0.6690390852 |
| Selection record | `selection.json` in the archive, unchanged: 12 candidate evaluations, no rule stop, no after-stop evaluation |
| Archive | `kaggle_a1_seed42_outputs.zip`, SHA-256 `4a3b9d4a2ba9e54e95a2f0598d7005ad4342b149f90148d37c8c8c1701ebb8c3` (the four identical downloads in `~/Downloads`); records extracted in `dist/kaggle_a1_seed42_results/` |
| Execution | the saved version (Save & Run All); training ran 15:25–15:58 UTC on 2026-11-10 (Kaggle clock); Tesla T4, CUDA 13.0, driver 580.178.04, torch 2.11.0+cu130 |

**The earlier execution: disclosed, non-authoritative.**
- **What happened:** seed 42 was also trained in an earlier, separate Kaggle execution. By your report, it selected
  step 35, the rule stopped at step 50, and step 55 was marked after-stop and non-selectable.
- **Status:** it is not the run of record, and it is never substituted for, or pooled with, the official run.
- **What remains of it:**
  - no archive, selection record or checkpoint of it exists on this machine;
  - its console output is to be preserved unchanged if Kaggle still shows it (procedure below).
- **What it is for:** measuring run-to-run variation only, never choosing.
- **Its weights differ:** its step-35 checkpoint is a different model from the official run's
  `checkpoint-step-000035`.

**Preserving its console output, if it is still available.**
1. Open the seed 42 notebook in the Kaggle editor.
2. If the training cell still shows the earlier execution's output, use *File → Download Notebook*. The `.ipynb`
   holds the cell outputs exactly as shown.
3. Do not edit or re-run anything first: re-running would replace the output.
4. Put the file, unchanged, in `logs/seed42_earlier_execution/`, and record its SHA-256
   (`shasum -a 256 <file>`) here.

**Status, 2026-11-11:** not yet provided; nothing of it is on this machine.

**GPU training is not bit-reproducible across separate executions.**
- **The setup:** the trainer seeds Python, torch and CUDA, but does not enable deterministic algorithms
  (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`). This is unchanged and kept for every seed, so
  all seeds share one procedure.
- **What it means:** two executions with the same seed, code and GPU type can end with different weights.
- **What is reproducible:**
  - inference: step 0 equals the rehearsal's ranking, and a reloaded checkpoint re-scores identically;
  - a CPU run.
- **The consequence:** each seed is trained exactly once, and the saved version is its run of record. The version 2
  notebooks enforce this (see [training_a1_execution_safeguards.md](training_a1_execution_safeguards.md)).

**Gradient clipping.**
- **What happened:** all 55 of 55 steps were clipped. Gradient norms before clipping ran from 2.11 to 16.90 (median
  4.88), against `max_grad_norm` 1.0.
- **Is it a defect?** No: this is the approved configuration, unchanged. With temperature 0.02 (logit scale 50), the
  raw gradient is always above 1, so every update is clipped.
- **How to read it:** the clip sets the size of every gradient before AdamW. This is part of the training dynamics,
  and the seed 1337 and 2026 runs use the same setting.

---

**Date of the audit:** 2026-11-10.

**Question:** an earlier execution of seed 42 was reported to select step 35, with selection stopping at step 50 and
step 55 marked non-selectable. The verified Save & Run All archive selects step 55. Why, and which result stands?

**Nothing was changed:** no training, no API request, no change to a frozen dataset, a historical result or a
selection record, no commit, no other seed. This audit reads the verified archive and recomputes from it.

## 1. Conclusions

1. **The verified archive's `selection.json` is correct.**
   - It holds 12 candidate evaluations (steps 0–55), no after-stop evaluation, and `rule_stop: null`.
   - It selects `checkpoint-step-000055`, at MRR@10 0.669039.
2. **Every eligibility flag is consistent** across `selection.json`, `validation_log.jsonl` and each checkpoint's
   `metadata.json`. All 12 checkpoints are candidates and selectable, which is right, because the rule never stopped.
3. **An independent recomputation agrees.**
   - Each MRR@10 was recomputed exactly from its 93 per-question first-relevant ranks.
   - The K1 rule was re-applied without the repository's code.
   - The result is step 55, with no stop, identical to the record and to the repository's own reader.
4. **The stopping rule worked as specified.**
   - After step 35, steps 40 and 45 did not improve: 2 misses in a row.
   - Step 50 then improved on step 35 by 0.00035, which reset the count.
   - Step 55 improved again, by 0.00012.
   - Three misses in a row never occurred, so the rule correctly did not stop.
5. **After-stop checkpoints cannot be selected.** This run has none, but the guarantee holds in code and in its
   checks (§5). Lowering step 50 by a single rank on a single question makes the repository's rule produce exactly
   the earlier execution's described outcome: select 35, stop at 50, step 55 never selectable.
6. **Why the two executions differ:**
   - **Two trainings ran.** Seed 42 was trained in two separate Kaggle executions, the earlier one and the Save & Run
     All one.
   - **The runs aren't bit-identical.** CUDA training here is not configured to be deterministic, so the same seed
     on the same stack does not guarantee identical weights.
   - **Selection is fragile here.** Steps 35, 50 and 55 lie within 0.00047 MRR@10 of each other. The selection
     between them turns on a single question moving by one rank.
   - **What isn't verified:** no record of the earlier execution exists on this machine, so its values are not
     verified here (§6).
7. **The latest run is methodologically valid.** It is complete and verified, and the rule was applied correctly.
   Its result is one valid outcome of the approved procedure. The earlier execution is a second outcome of the same
   procedure, and it must be disclosed, not substituted.
8. **No implementation or recording bug** was found in the rule, the flags, the hashes or the records. Three gaps
   were found (§7):
   - the procedure allowed one seed to be trained twice, and the plan did not warn about it (my omission);
   - same-seed reproducibility on CUDA was never established, and the documents implied it only by silence;
   - the outputs do not identify which Kaggle execution produced them.
9. **The defensible selected checkpoint is `checkpoint-step-000055` of the verified run** (digest `24c2aaae…`). The
   earlier execution should be recorded as a disclosed, non-designated replicate.

## 2. Evidence

| Item | Value |
|---|---|
| Archive | `kaggle_a1_seed42_outputs.zip`, 6,136,987,429 bytes, SHA-256 `4a3b9d4a2ba9e54e95a2f0598d7005ad4342b149f90148d37c8c8c1701ebb8c3`, equal to its `.sha256` |
| Downloads | four in `~/Downloads` (`… .zip`, `(1)`, `(2)`, `(3)`), all **byte-identical**: one execution, downloaded four times |
| Extraction | `dist/kaggle_a1_seed42_results/` (records only); 131 files listed in `OUTPUT_SHA256SUMS`, all verified by `check_outputs.py` |
| Execution | GPU monitor from 2026-11-10 15:25:15 to 15:57:55 (Kaggle clock); training, results and verification each exit 0 |
| Bindings | config `7a4c4618…`; schedule `44b0ed7c…`; `run.json` `c2483843…`; the environment equals the rehearsal's on every identity key (T4 7.5, CUDA 13.0, cuDNN 92700, Python 3.13.16, torch 2.11.0+cu130, transformers 5.3.0, tokenizers 0.22.2) |
| The earlier execution | **No artifact of it exists on this machine.** Nothing turned up in `~/Downloads`, `~/Documents`, `~/Desktop` or this project's session records: no zip, no `selection.json`, no log. Its description comes from your report only. |

## 3. The recorded evaluations and their eligibility

Columns:
- **Σ RR:** the sum over the 93 questions of the reciprocal rank, cut at 10 (MRR@10 × 93).
- **Selectable:** from each checkpoint's `metadata.json`.

| Step | MRR@10 | Σ RR | Candidate | Improved | Misses in a row | After stop | Selectable | Digest |
|---:|---:|---:|---|---|---:|---|---|---|
| 0 | 0.2658858167 | 24.7274 | yes | yes | 0 | no | yes | `38633262…` |
| 5 | 0.4250981396 | 39.5341 | yes | yes | 0 | no | yes | `455b108b…` |
| 10 | 0.5867810207 | 54.5706 | yes | yes | 0 | no | yes | `6c4a7557…` |
| 15 | 0.6324330090 | 58.8163 | yes | yes | 0 | no | yes | `6217595b…` |
| 20 | 0.6435441202 | 59.8496 | yes | yes | 0 | no | yes | `02da55a7…` |
| 25 | 0.6582522615 | 61.2175 | yes | yes | 0 | no | yes | `5529d86d…` |
| 30 | 0.6628264209 | 61.6429 | yes | yes | 0 | no | yes | `a3f40591…` |
| 35 | 0.6685697218 | 62.1770 | yes | yes | 0 | no | yes | `37e89be4…` |
| 40 | 0.6665002560 | 61.9845 | yes | **no** | 1 | no | yes | `14672a65…` |
| 45 | 0.6680704898 | 62.1306 | yes | **no** | 2 | no | yes | `95d325ad…` |
| 50 | 0.6689196109 | 62.2095 | yes | yes | 0 | no | yes | `248f3832…` |
| **55** | **0.6690390852** | 62.2206 | yes | yes | 0 | no | yes | `24c2aaae…` |

- **Record:** `selected` = step 55; `rule_stop` = null; `stopped` = {at_step 55, early false, planned 55};
  `after_stop_evaluations` = [].
- **Consistency:** `validation_log.jsonl` equals the record entry for entry (step, value, digest). The console prints
  the same sequence and "selected checkpoint-step-000055".

## 4. Independent recomputation

**Values.**
- Each evaluation's MRR@10 was recomputed with exact fractions from the 93 `first_relevant_rank` values: 1/rank if
  the rank is ≤ 10, else 0.
- All 12 equal the recorded values (difference < 1e-12).
- The values were therefore not altered after evaluation, and they are the mean over all 93 questions.

**The rule, re-implemented from its approved text.**
- Candidates are step 0 and every 5 steps.
- An evaluation improves only if strictly greater at 12 decimals; ties go to the earlier step.
- Selection stops after 3 candidate evaluations in a row without improvement, and later evaluations are never
  selectable.

| Step | Outcome |
|---:|---|
| 0–35 | each improves; best = step 35 (0.668570) |
| 40 | miss 1 (best stays 35) |
| 45 | miss 2 (best stays 35) |
| 50 | improves on 35 by **+0.00034989** → best = 50, misses reset to 0 |
| 55 | improves on 50 by **+0.00011947** → best = 55 |

**Result:** selected step 55, no stop.
- This equals the record.
- The repository's `read_selection_record` accepts the record; it re-derives the rule and checks completeness.
- `select_through` returns `(55, None)`.

## 5. The stopping rule and after-stop selection

**In code.** `SelectionState` counts misses and resets the count on improvement; `should_stop` means 3 or more
misses.
- **Before the stop:** the trainer calls `record` while `should_stop` is false.
- **After the stop:** it calls `record_after_stop`, which writes `candidate: false, selectable: false, after_stop:
  true` and never touches the best.

`read_selection_record` re-applies the rule to every evaluation (`select_through`). It rejects any record whose
selection or stop differs, including one that selected an after-stop checkpoint.

**Exercised:**
- `tests/unit/test_checkpoint_selection.py`;
- the seed-run verification tests, where the fixture run stopped early and evaluated after the stop.

**Counterfactuals,** with the repository's `SelectionState` and `select_through` on this run's recorded values:

| Change | Rule outcome |
|---|---|
| none (as recorded) | select 55; no stop |
| step 50 lower by one rank on one question (5 → 6, −0.0333/93) | **select 35; stop at 50; step 55 after stop, never selectable** |
| step 50 exactly equal to step 35 | select 35 (tie to the earlier step); stop at 50; 55 never selectable |
| as the row above, with step 55 at 0.99 | still select 35: after-stop evaluations are never selectable |

**What the second row shows:** the earlier execution's described outcome is exactly what the approved rule produces
when step 50 falls at or below step 35. The described behaviour is correct behaviour, under slightly different
values.

## 6. Why the two executions selected different checkpoints

**1. Two trainings of seed 42 ran.**
- **How it can happen:** a Kaggle *Save & Run All* starts a fresh, independent execution of the whole notebook.
- **What it means here:** if the notebook also ran interactively with `START_OPTIMIZER = True`, which is my reading
  of "earlier" and "latest Save & Run All", seed 42 was trained twice. Each training went from the base model to step
  55 in its own session.
- **No mixing:** neither execution resumed or reused the other. The notebook refuses an existing run directory, and
  it never passes `--resume`.

**2. The two trainings need not produce identical weights.**
- **What the seed fixes:** `seed_everything` seeds Python, torch and CUDA. That fixes the data order and the dropout
  masks.
- **What it doesn't:** the trainer does not call `torch.use_deterministic_algorithms(True)` or set
  `CUBLAS_WORKSPACE_CONFIG`. PyTorch documents that CUDA results are then not guaranteed reproducible from run to
  run, even with the same seed and hardware.
- **This archive shows inference is reproducible:**
  - step 0 ranks the validation set exactly as the r2 rehearsal did, in a different session;
  - the reloaded step-55 checkpoint reproduces its recorded result exactly;
  - GradCache's replay difference is 0.0 at every step.
- **So any difference must arise in training:** in the backward pass or the optimizer, after the first update. The
  exact kernel was not identified; that would need GPU experiments.

**3. The selection margins are smaller than one rank change on one question.**

| Comparison | ΔMRR@10 | ΔΣ RR | Questions whose RR changes |
|---|---:|---:|---:|
| 50 against 35 | +0.000350 | +0.0325 | 12 (moves of ±1–6 ranks, netting almost to zero) |
| 55 against 50 | +0.000119 | +0.0111 | 1: pair …000087 moves from rank 10 to rank 9 |
| 55 against 35 | +0.000469 | +0.0437 | 12 |

**Conclusion.** A nondeterministic difference small enough to move one question by one rank at step 50 switches the
outcome between "select 55, no stop" and "select 35, stop at 50". The described earlier result is therefore the
expected kind of difference, not a sign of a defect.

**What remains unverified.**
- The earlier execution's values are not available here.
- **Confirmation would need** its console output (still visible in the Kaggle editor if the interactive session's
  cell output was kept) or its archive (if it was a saved version).
- **The check:** step 1's loss should be identical in both executions, since the weights, data and dropout are the
  same before the first update. The losses should diverge slightly from step 2 on.
- **If step 1 already differs,** something other than backward-pass nondeterminism differed, and that would need
  investigating.

## 7. Bugs and gaps

**No bug** in the rule's implementation, its recording, the eligibility flags, the hashes, the resume state or the
results tools. Each was checked directly (§3–§5), and the archive's own verification passed:
- 12 of 12 checkpoints re-hash;
- the step-55 resume state restores on the T4;
- the step-55 re-evaluation is identical;
- step 0 equals the rehearsal.

**Gaps:**

| # | Gap | Effect | Proposed remedy (not implemented; needs your approval) |
|---|---|---|---|
| G1 | The procedure allowed one seed to be trained twice. The notebook does not distinguish an interactive run from a Save & Run All, and my plan did not warn that running interactively with the gate open, then saving the version, trains twice. | Two outcomes for seed 42, and the temptation to choose between them after seeing both | Refuse training unless Kaggle reports a batch run (`KAGGLE_KERNEL_RUN_TYPE == "Batch"`; confirm on Kaggle that it is set, before relying on it), so only the saved version trains, and state it in the instructions |
| G2 | Same-seed reproducibility on CUDA was never established. The documents say a *resumed* CUDA run is not bit-identical, but say nothing about two fresh runs, and the reproducibility test runs on CPU only. | The expectation that seed 42 fixes the result was wrong on the GPU | Document it. For the remaining seeds, keep the stack and settings unchanged; enabling deterministic algorithms now would make seed 42's procedure differ from the others'. |
| G3 | The outputs do not identify the execution: run type, Kaggle version, start time. `run.json` is byte-identical across executions by design. | Two executions' records cannot be told apart except by their values and file times | Record the run type, the UTC start time and the driver in `environment.json` and `pins.json` (notebook only; the trainer is unchanged) |

**Observation, not a bug:** every one of the 55 steps was clipped. Gradient norms ran from 2.11 to 16.90 (median
4.88) against `max_grad_norm` 1.0. This is the approved configuration, but it shapes the training dynamics, and the
seed 42 report should discuss it.

**The rest of the training behaviour:**
- loss 1.213 → 0.151 (minimum 0.052);
- loss scale 65,536 throughout; no non-finite value;
- learning rate as planned;
- 3.63 s per step;
- peak 9.22 GiB allocated, and 10,335 MiB on `nvidia-smi`.

## 8. Recommendation

1. **Designate the verified Save & Run All execution as seed 42's run of record.** Its selected checkpoint is
   `checkpoint-step-000055`, digest `24c2aaae977ce319110a7dd979cb365fccfdac6cc89d63b1ba566e31a678beb9`.
   - **It is the execution the approved plan designates:** *Save & Run All*, outputs downloaded from the version.
   - **It is the only one with a complete, hash-verified record:** all 12 checkpoints, the selection record, and the
     resume and re-evaluation checks. The decision rule's input requires exactly that.
   - **The choice does not depend on which checkpoint either execution selected.**
2. **Do not substitute step 35.**
   - The earlier execution's step-35 checkpoint is a different model, and it was not preserved.
   - Choosing the latest run's step 35 would override the pre-registered rule after seeing its values. That
     selection optimism is exactly what the K1 test is built to account for, and only if the rule is followed.
3. **Disclose the earlier execution** in the seed 42 report as a non-designated replicate, together with its
   described selection.
   - If its console output or archive can be retrieved, keep it unchanged as a historical record. Use it only to
     measure the run-to-run variation (§6), never to choose.
4. **Before seeds 1337 and 2026,** decide on G1 and G3, both notebook-only changes, and acknowledge G2. Each seed
   should then be trained exactly once, in the saved version.
