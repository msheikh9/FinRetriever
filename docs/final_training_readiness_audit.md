# Final training-readiness audit: P0 and P1 batch 1

Status: 2026-11-05. An independent, read-only audit of:
- the P0 implementation ([training_full_v1.md](training_full_v1.md));
- the P1 plan `train_v3_p1_batch1` and its 39 generation records;
- the semantic leakage screen;
- the proposed A0 versus A1 experiment ([training_plan_v1.md](training_plan_v1.md) §7 and the full-training plan of
  2026-11-04).

**What this audit did not do:**
- change any code, config, candidate, generation record, validation set or result;
- send any API request;
- train any model;
- commit anything.

**What it computed:**
- The pinned zero-shot BGE model ran offline, on CPU, over question texts only.
- A test bundle was built into a scratch directory outside the repository.
- No evaluation qrels, relevance labels, pools, judgments or audits were opened. Evaluation questions were read only
  through the screen's own loader (`query_id` and `text`).
- The working tree was clean before and after.

**Labels.** Every training and validation question is synthetic and was accepted by an AI review: none is
human-validated or a gold label. The development labels are AI labels, except 4 human grades. The gold set is empty.

## 1. Overall readiness verdict

| Gate | Verdict | Basis |
|---|---|---|
| P0 code, CPU path | **Ready** | The logic is correct on reading (§A). The suites pass. The real-BGE GradCache check matches to 5.4e-8. The smoke run's `run.json` is reproduced byte-for-byte. |
| P0 code, CUDA path | **Not verified** | No CUDA code path has ever run: fp16 GradCache, replay, memory with eager attention, the validation pass (§A9). |
| P1 batch 1: resume generation | **GO, with the conditions in §6** | The 39 records and the plan are intact and bound to the same prompt, parameters and preparation; resume skips them. |
| Three-seed A0 versus A1 experiment | **NO-GO** | Four blockers (§2): the decision rule is selection-biased and not written down; batch 1 alone is too small; the CUDA path is unverified; the batch gate is pending and not written down. |

## 2. Critical blockers

**For resuming P1 generation:** none in code or data. The operational conditions are in §6.

**For the three-seed experiment:**

| # | Blocker | Why it blocks | What resolves it |
|---|---|---|---|
| B1 | **The A0 versus A1 decision rule is selection-biased and exists only in chat.** | Of its three conditions, the first is nearly automatic, because a selected checkpoint can never score below step 0. The second is computed on the same 93 questions that chose the checkpoint. Only the third is largely free of selection (§D1–D2). | A written, selection-aware rule, fixed before any training run (decision D1). |
| B2 | **Batch 1 alone is far below the planned size.** | 66 + about 105 = about 170 pairs gives about 3 steps per epoch at batch 64: about 15 steps per seed and at most 3 candidate checkpoints after step 0. The smallest defensible experiment was planned at 600–730 pairs and about 55 steps. | The Groq budget for batches 2–6, or an explicit decision to run a labelled pilot (D3). |
| B3 | **The CUDA path has never run.** | GradCache's fp16 replay, GradScaler behaviour, eager-attention memory and the validation pass on a GPU are all unexecuted (§A9). | The Kaggle preflight with every CUDA check passing, then one GPU rehearsal run (D8). |
| B4 | **The batch-1 quality gate is pending and not written in the repository.** | It needs the other 61 passages, `build-pairs`, the AI review, the semantic screen and `train_candidates_v3`. Its thresholds exist only in the chat plan. | Write the gate down before it is evaluated (§B8), then evaluate it. |

## 3. Non-critical improvements

None was made. Each would need your approval as a code change.

| # | Improvement | Why |
|---|---|---|
| N1 | Persist each `generate` invocation's request accounting (requests sent, HTTP 429s, waits, stop reason) to a write-once file next to the records. | Today it is printed only, so the HTTP attempts of the earlier run(s) are not on disk. Tokens are on disk (per-record `usage`). |
| N2 | A cumulative request budget across reruns. | `--max-requests` caps one invocation only; every rerun starts again at 0. |
| N3 | Check `selection.json` for completeness: the evaluated steps must be a prefix of `evaluation_steps(planned)`, and each value must equal the checkpoint's metadata and `validation_log.jsonl`. | An evaluation deleted from a record is not detected today, because the re-derivation stays consistent with what remains. Accidental damage is caught by the file hashes. |
| N4 | Log the GradScaler's scale and every skipped (overflowing) step explicitly. | A skipped fp16 step shows only as an infinite or NaN `grad_norm` in `log.jsonl`. With 15–55 steps, a few skipped early steps matter. |
| N5 | Label `MEASURED_T4`, the 5.0 GiB figure the dry run quotes, as measured with SDPA attention and the plain loss. | GradCache runs use eager attention, which keeps the full attention matrices, so that figure understates their memory. The preflight's measurement is the one to trust. |
| N6 | Time and measure one validation slice (for example 512 passages) in the CUDA preflight. | The full-corpus validation pass dominates run time and is unmeasured with eager attention. |
| N7 | Semantic screen: store each question's nearest few similarities, and add the same check to benchmark-time isolation once gold questions exist. | §C5. |
| N8 | Rename the `smoke_subset` preflight check for training configs. | Cosmetic: it reports "66 of 66 candidates selected (subset seed 0)". |
| N9 | Kaggle guide: one seed per session for the full experiment, or download between seeds. | Up to 12 checkpoints × 438 MB per seed, plus 871 MB of optimizer state: about 6 GB per seed, about 18 GB for three seeds in one session. |
| N10 | Two consistency checks at analysis: step 0 must give the same validation value for every seed; and score `checkpoint-step-000000` on the development set beside the stored `bge_zero_shot` run. | Together they catch environment drift between A0 and A1. |

## 4. Tests

### Passed in this audit (2026-11-05)

| Check | Result |
|---|---|
| `python3 -m pytest` (global Python) | 1,264 passed, 28 skipped, 0 failed. The skips are the BERT tests needing the venv's scipy, the missing local Apple 10-K fixture and `pytrec_eval`. |
| The same suite in the dense venv (`HF_HUB_OFFLINE=1`) | 1,267 passed, 25 skipped, 0 failed: 24 for the missing local Apple 10-K fixture, 1 for `pytrec_eval`. |
| `ruff check src tests scripts` | All checks passed |
| `git diff --check` | Clean. The working tree had no changes before or after the audit. |
| Rehearsal bundle rebuilt from the committed code (scratch directory) | 95 files, 9.0 MB (SHA-256 `078fb874…`); 0 entries under `data/eval` |
| Preflight on the extracted bundle (CPU, offline, `--skip-cuda`) | Every non-CUDA check passes:<br>- the 66 candidates re-verified;<br>- `validation_pairs_v4` isolated from them;<br>- 3 seeds × 10 steps;<br>- none of 38 files read is evaluation data;<br>- GradCache relative difference 5.4e-8.<br>Reported "not ready" only because the CUDA checks were skipped. |
| Smoke run record rebuilt in memory from the committed code | Byte-identical to `models/bge_base_ft_v2_smoke/…-seed42/run.json` |
| Smoke step-15 checkpoint as `role: final` | Refused without a record, and refused with a record path that does not exist |
| Prompt, parameter and validator identity | Recomputed from code and config: prompt `f42bdc54…`, parameters `42d5fe54…`, `sec-qcheck-v3`, preparation manifest `21c2293a…`. All equal the train run, and the validation run used the same settings. `check_run`: `bound`. |
| Plan `train_v3_p1_batch1` | SHA-256 `3b9210c3…cec0`, unchanged. 100 train passages, none blocked, from 100 distinct overlap groups, none shared with the 66 candidates (equivalents included) or with `validation_pairs_v4`. |
| The 39 records | Plan positions 1–39 in order:<br>- all `done` at the first attempt, with no error;<br>- all the same model, prompt, validator and parameters, record format 3. |
| Duplicates and lexical overlap of the 63 new questions | No exact repeat and no Jaccard ≥ 0.8 against the 66 candidates or the 93 validation questions (2 at ≥ 0.6 against validation) |
| Semantic distribution against the permitted evaluation questions | No training question at ≥ 0.85; the maximum is 0.80 (§C3) |
| Train against validation, close templates (cosine ≥ 0.85) | Of 22 close pairs, none has the validation answer in the training positive. One shared figure is a coincidence: a different date in another table. |
| Module isolation | Importing the trainer, the evaluator and the selection module does not load `semantic_screen`. |

### Still needed

| # | Test | When |
|---|---|---|
| T1 | Kaggle preflight with CUDA: `cuda_fp16`, `gradcache` on the GPU in fp16, and `gpu_memory` from the measured cached step | Before any training |
| T2 | One GPU rehearsal run (`biencoder_bge_base_ft_v3_rehearsal.yaml`, seed 42), to measure:<br>- step time and validation time;<br>- peak memory with eager attention;<br>- replay differences (expected 0);<br>- skipped fp16 steps;<br>- disk use.<br>It also yields a written `selection.json`, collected by `smoke_results.py`. | After T1, with your approval |
| T3 | GPU interruption and resume: stop at a step between evaluations, then resume. The selection state and evaluations must continue (CPU-tested only). | During T2 |
| T4 | After generation:<br>- `build-pairs` and the AI review;<br>- the first real `screen-questions` run;<br>- `train_candidates_v3` with `semantic_screening`;<br>- a rebuild of `train_candidates_v2` reported as unchanged (the 66 untouched). | Before the gate |
| T5 | The experiment config's bundle and preflight (`train_candidates_v3`) | Before the experiment |
| T6 | End-to-end `role: final` scoring of one real validated run (`check_final_seeds` and isolation), with the step-0 control (N10) | After the experiment |
| T7 | If D1 is approved: test the selection-aware analysis on synthetic selection records before any real run | Before the experiment |

## 5. Risks needing your explicit decision

| # | Decision | Options and recommendation |
|---|---|---|
| D1 | **The A0 versus A1 decision rule** | Replace the 2026-11-04 rule with a selection-aware one, written in the repository before any training (§D7). **Recommended.** Without it, a positive result is weak evidence. |
| D2 | **The semantic screen's 0.9 threshold** | The options (§C5):<br>(a) keep 0.9 and accept that reworded paraphrases pass;<br>(b) keep automatic exclusion at 0.9 and send 0.80–0.90 to a recorded review (1 of 129 current questions);<br>(c) lower automatic exclusion to 0.80 (1 excluded today, 10 at 0.75).<br>**Recommended: (b).** The threshold is not changed without your decision. |
| D3 | **Data size** | Approve the Groq budget for batches 2–6, or run batch 1 only as a pilot labelled underpowered. Batches 2–6 are 500 more passages: about 710k tokens at batch 1's 1,445 per request, and about 8–9 h at the configured pacing, in batches of 100 with stop records. The plan's 855k tokens and 10 h were for all 600 passages. **Recommended:** batches 2–6. |
| D4 | **The resume request cap** | `--max-requests 61` allows at most the 61 remaining replies, so batch 1 stays within 100 completed requests. HTTP 429 retries count against the cap without spending tokens, so a rerun may be needed. The earlier invocations' HTTP attempts were not saved (N1). |
| D5 | **A generator change** | If Groq withdraws `qwen/qwen3.8-27b` or support changes the limits:<br>- pacing or token-limit changes continue the same run (`--continue-run`, recorded);<br>- a model change is a new run, and its questions would no longer come from the validation set's generator.<br>Decide before resuming whether a model change is acceptable at all. |
| D6 | **Company mix in later batches** | Apple and Amazon have only 55 and 40 overlap groups in the train source pool, so later batches cannot keep batch 1's 14% each, and JPMorgan's share will rise towards its 25% cap. The choice: accept the drift, or stop at a smaller, balanced set. |
| D7 | **The pooling rule** (still open from the plan) | Keep it: development pools are never deepened with a fine-tuned model's output, so its development scores stay lower bounds. |
| D8 | **Training approval for the GPU rehearsal** | It resolves B3 without depending on the gate or on new data. Mechanics only; no retrieval result is read from it. |
| – | Batch 64 for the experiment (open from the plan) | Unchanged by this audit |

## 6. Can P1 batch 1 safely resume?

**Yes, under these conditions:**

1. **Approval and limits.**
   - You confirm the remaining 61 requests under the approved batch.
   - Groq's limit has reset, or support has answered.
   - Groq's daily limits are unknown. The run used 56,353 tokens over about one hour before stopping.
2. **Dry run first.** Expect `100 approved passages, 39 done, 0 failed; 61 to request`. A run can leave a failed
   record: if one reply is unusable and the retry is then rate-limited, the passage is recorded as failed. A rerun skips
   failed passages unless `--retry-failed` is given. If a later dry run reports any failed passage, add
   `--retry-failed` so the plan completes.

   ```bash
   python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v3.yaml --split train \
       --plan train_v3_p1_batch1 --max-requests 61
   ```
3. **Then the real run, with its console output kept outside `data/`.** Example path:

   ```bash
   python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v3.yaml --split train \
       --plan train_v3_p1_batch1 --max-requests 61 --confirm-api-use 2>&1 | tee ~/finretriever_logs/p1_batch1_resume.log
   ```
4. **Settings.** Change no setting silently:
   - Pacing or token-limit changes need `--continue-run`, which writes a recorded continuation.
   - A model change is refused by the run binding (D5).
5. **Finish the plan.** Do not close it early (`stop-plan`) unless the gate fails. The first 39 positions hold only 2 of fiscal 2019's 13 passages and 4 of fiscal 2025's 14, so stopping now would skew the years.

**Why resume is safe:**
- Each record is appended and flushed (`fsync`) before the next request.
- A rerun skips every `done` passage, in plan order.
- `<split>.run.json` refuses other generation settings.
- A rate-limit or budget stop records nothing for a passage that has not yet had a reply (see condition 2 for the
  exception).
- A resume only appends, so the 39 lines stay as written. Today's file has SHA-256 `30f83dd7…`.

**Cost projection** (measured on the 39, then extrapolated):

| | Measured (39 passages) | Projected for the other 61 | Batch total |
|---|---|---|---|
| Tokens | 56,353 (46,444 prompt, 9,909 completion; 1,445 per request) | about 88,000 | about 144,000 (the plan estimated about 143,000) |
| Requests | 39, each at the first attempt | 61 if no reply is unusable | 100 |
| Questions passing the validator | 63 of 78 (81%; `pilot_v3` 78%) | about 98 | about 160 |

## 7. Can the three-seed experiment proceed after the quality gate?

**Not directly.** Passing the gate opens the next generation batches, not the experiment. The sequence:

1. Finish batch 1 and run T4, then evaluate the written gate (§B8).
2. If it passes: batches 2–6 under D3, each with the same gate, then build `train_candidates_v3`.
3. Decide D1 and D2, and write the rule down.
4. Pass T1 to T3 on Kaggle; the rehearsal can run earlier, on your approval (D8).
5. Then approve the experiment: three seeds, every seed reported.

If you choose to run the experiment on batch 1 only, the code allows it. But the result would be a pilot: about 15 steps per seed and at most 3 candidates after step 0. It should be labelled as such, and a null result would be uninformative.

---

## Findings

### A. P0

**A1. GradCache: exact on reading and on CPU.**
- **The math.** The surrogate `(embeddings * cached_grad).sum().backward()` gives each parameter exactly the chain-rule
  gradient of the full-batch loss, as long as the replay reproduces the cached embeddings.
- **Dropout replay.**
  - The CPU random state, plus the device's on CUDA, is saved before each cached chunk and restored before its replay.
  - After the last replay, the state is set to where the uncached passes would leave it.
  - This is the mechanism `torch.utils.checkpoint` uses.
  - Eager attention is forced for GradCache runs, so attention dropout comes from the same generator.
- **fp16 scaling.**
  - The loss on the cached embeddings is scaled by the GradScaler, so the embedding gradients carry the scale.
  - The replay backpropagates them into fp16 activations exactly as a scaled plain backward would.
  - `unscale_`, clipping and `step` then behave as in plain AMP.
  - Each autocast block exits before the next, so no weight cast from the no-grad pass is reused with gradients.
- **The replay check.**
  - Each replayed chunk is compared with its cached pass, and a difference above the tolerance aborts the run before
    the optimizer step.
  - The fp16 tolerance (1e-2 relative to the largest value) is loose. A GPU replay should be bit-identical, so T2
    should report 0.
- **Evidence:**
  - the CPU tests (`tests/unit/test_gradcache.py`);
  - today's preflight on real BGE with dropout: relative difference 5.4e-8.

**A2. Validation evaluator isolation: holds.**
- `validation_eval` opens no file. It gets the pairs, the corpus and the `NegativeGuard` in memory.
- `prepare_inputs` loads `validation_pairs_v4` with `training=<the pairs trained on>`. It fails closed on:
  - a shared positive or equivalent;
  - a repeated question;
  - a question with Jaccard ≥ the lexical threshold.
- **Relevance:** the positive plus exact equivalents (50 exact equivalents across the set). The 29 uncertain
  equivalents, the overlap groups and the reuse links are removed from the ranking.
- **The trainer reads no evaluation file:** the preflight's file trace shows none of the 38 files read is one.

**A3. Selection and early stopping: matches the approved rule.**
- **The rule:** candidates at step 0 and every 5 steps; whole-corpus MRR@10; ties at 12 decimals keep the earlier step;
  patience 3 counts candidate evaluations only.
- **The last step:** a final step that is not a multiple of 5 is evaluated, reported and never selected.
- **Failure cases traced in the code:**
  - **a crash after the last checkpoint but before `selection.json`:** the resume writes the record without training;
  - **a crash after an early-stop checkpoint:** the resume writes the record at that step;
  - **an ended run:** it returns before any write.
- **Gap (N3):** an evaluation removed from a record is not detected, though any edited value or file hash is.

**A4. Three seeds: independent runs.**
- **Run directories:** `<output_dir>/<name>-seed<seed>`, each bound by its own `run.json` (the seed is part of the
  config record).
- **Without `--seed`:** a multi-seed config is refused.
- **What a seed changes:** the batch order (`seed + epoch`) and dropout. The data, schedule length, validation set and
  rule are shared.
- **What a seed cannot change:** all seeds start from the same pretrained weights, so step 0 should give an identical
  validation value for every seed (N10).

**A5. Checkpoint hashing and resume integrity: holds.**
- **Hashing:** an evaluated checkpoint is hashed in its temporary directory before `trainer_state.pt` is added, then
  renamed atomically.
- **The optimizer state:** older checkpoints lose only `trainer_state.pt`, which is never hashed.
- **Never written over:** an existing checkpoint is refused.
- **Resume:** it restores the optimizer, scheduler, scaler, random and selection state. Equality with an uninterrupted
  run is tested on CPU; on CUDA a resumed run is close, not bit-identical.

**A6. Final registration and provenance: holds.**
- `role: final` needs a `selection_record` that `verify_selected_checkpoint` accepts:
  - the path is the selected one;
  - `run.json` is unchanged;
  - every file hash matches;
  - the metadata's run and step match.
- A checkpoint whose metadata says it was saved at a fixed step is refused even with a forged record.
- `check_final_seeds` requires every seed, exactly once.
- Isolation now also covers the selecting validation set.
- Verified today on the real smoke checkpoint.

**A7. Kaggle bundle and preflight: complete for the rehearsal.**
- **Contents:** the rebuilt bundle traces what the run reads: the preparation, the 66 candidates with their sources
  and decisions, `validation_pairs_v4` with its carried set, records, decisions and plan.
- **Absent:** no evaluation data, and none of the 39 new records (the rehearsal does not use them).
- **The experiment's bundle:** it must be built from the experiment's config once `train_candidates_v3` exists (T5).

**A8. Compatibility with the smoke checkpoints: holds.**
- Today's code reproduces the smoke `run.json` byte-for-byte.
- The smoke checkpoints remain diagnostic-only.
- Their directory is intact: 3 checkpoints, the log and `run.json`.

**A9. GPU-specific behaviour that has never run:**

| Behaviour | Risk | Covered by |
|---|---|---|
| Dropout replay with the CUDA generator under fp16 autocast and eager attention | A mismatch would abort the run (it is detected, not silent) | T1 (4 pairs), T2 (every step) |
| GradScaler with GradCache on CUDA: overflow and skipped steps | Lost early updates in a short schedule; not logged explicitly (N4) | T2 |
| Peak memory at 16 × 512-token passages with eager attention | The 5.0 GiB smoke figure was measured with SDPA and the plain loss. Eager attention keeps full attention matrices, so the peak is likely several GiB higher (unmeasured). | T1 measures one cached step |
| The validation pass on CUDA: `fork_rng` on a CUDA device, fp16, batch 64, eager attention | Time and memory unmeasured; the 1–1.5 min estimate predates eager attention | T2 (N6) |
| Step time with eager attention | The 2.3 s per step estimate is unmeasured | T2 |
| Determinism | CUDA runs are not bit-identical: a rerun of a seed may select another step. The records keep what happened. | Documented |
| Disk in `/kaggle/working`, and atomic renames there | About 6 GB per seed for the full experiment (N9) | T2 |

### B. P1 batch 1

**B1. Plan integrity: holds.**
- **The file:** SHA-256 `3b9210c3…cec0` as written; seed 20261029; `balanced_years`; at most 25% per company; one
  passage per overlap group.
- **Passages:** all 100 are approved train passages, none blocked, from 100 distinct groups.
- **Exclusions:** 139 groups, those of the 66 candidates and of the earlier v1 and v2 generation. Independently, no
  plan group is used by a candidate passage (positive or equivalent) or by `validation_pairs_v4`.

**B2. The 39 records:**

| Item | Value |
|---|---|
| Positions | Plan positions 1–39, in order; no other plan has records in this file |
| Outcome | 39 `done` at the first attempt; no error and no failed record |
| Identity | `qwen/qwen3.8-27b`, `sec-qgen-v2`, `sec-qcheck-v3`, parameters `42d5fe54…`, record format 3 |
| Candidates | 78 proposed; 63 pass the validator (81%) |
| Rejected (15) | `missing_company` 6, `evidence_length` 5, `copied_from_passage` 3, `evidence_not_in_visible_text` 1 |
| Time | 2026-11-04 21:15:35 to 22:19:00 UTC |
| Pacing | Gaps mostly of 59–62 s; otherwise 0–4 s, when a reply left room under the 1,000 output-tokens-per-minute pacing |
| The long gap | 2,232 s between positions 35 and 36. `generated_at` is stamped before the wait, so this fits a stop and rerun, or a suspended process. The cause is not recorded on disk (N1). |

**B3. Resume and request-budget accounting.**
- **Per invocation:** `--max-requests` counts every HTTP attempt in one invocation, retries and 429s included, and
  stops cleanly at the cap.
- **Not stored:** the count is printed, not stored, and a rerun starts again at 0. So the HTTP attempts already spent
  are unknown, while tokens are known from the records (§6, D4).
- **A failed passage:** it would be skipped unless `--retry-failed`; there are none (§6, condition 2).

**B4. Prompt and validator hashes.**
- The code's `sec-qgen-v2` prompt hashes to `f42bdc54…` and the config's parameters to `42d5fe54…`, both as recorded.
- All 93 questions of `validation_pairs_v4` carry that prompt hash and parameter hash in their provenance, so the
  validation questions and batch 1 come from one generator (§D3).
- Of the 66 candidates, 51 come from `sec-qgen-v2`. The other 15 (`pilot_v2`) come from the earlier `sec-qgen-v1`
  prompt.

**B5. Provenance and candidate inclusion.** Nothing has been built from the 39 records yet. Inclusion will follow the
existing path:
- `build-pairs` (the validator, duplicate and evaluation-overlap checks);
- an AI review decision per pair (`ai_review`, `gold: false`; only `accept` enters);
- `build-candidates` (contract, evidence leakage, recomputed equivalents and, if configured, the semantic screen),
  written once.

The 66 candidates are unchanged:
- `train_candidates_v2/manifest.json` `fbd30a59…`;
- `train_candidates.jsonl` `908d4c79…`.

**B6. Evidence leakage and false-negative protections: unchanged and active.**
- **Blocklist:** the blocklist and split exclude evaluation pools and their overlap groups from every plan passage.
- **Evidence:** the evidence rule and the equivalents run at `build-pairs` and `build-candidates`.
- **In-batch negatives:** they are re-checked before every forward pass.
- **The 63 new questions:** no duplicate or lexical overlap with the candidates or the validation set.
- **The near-template probe (§C3):** no training positive answers a validation question.
- **Known gaps remain as documented:** paraphrased evidence, rounded figures and changed row labels.

**B7. Company and fiscal-year balance.**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Passages done (39) | 6 | 6 | 6 | 6 | 5 | 5 | 5 |
| Passages remaining (61) | 8 | 8 | 8 | 9 | 9 | 10 | 9 |
| Validator pass, done | 11/12 | 11/12 | 10/12 | **7/12** | **7/10** | 9/10 | 8/10 |

| Fiscal year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Done | 2 | 5 | 7 | 7 | 7 | 7 | 4 | 0 |
| Remaining | 11 | 9 | 8 | 7 | 7 | 7 | 10 | 2 |

- **Companies:** balanced so far.
- **Years:** balanced only over the whole plan.
- **Pfizer and Tesla:** they pass the validator least often so far, so the gate should check included pairs per
  company.

**B8. The 100-passage quality gate.** It is written here so it is fixed before evaluation. Thresholds 1 and 2 are from
the plan of 2026-11-04; 3 to 6 are this audit's proposal for your approval.

| # | Criterion | Interim reading |
|---|---|---|
| 1 | At least 0.8 included candidates per planned passage (80 or more of 100) | Not measurable yet. If `pilot_v3`'s later rates hold, the projection is about 1.05. |
| 2 | The AI review accepts at least 75% of the built pairs | Not measurable yet (`pilot_v3`: 81%) |
| 3 | Every company contributes at least 8 included candidates | Not measurable yet. Pfizer is the one to watch. |
| 4 | Every flagged or reviewed screen result is recorded, and no pair at or above the automatic threshold enters | Pending D2 |
| 5 | `train_candidates_v2` rebuilds unchanged, and `validation_pairs_v4` still loads | T4 |
| 6 | The run used one model, prompt, validator and parameter set, or a recorded continuation | Holds so far |

### C. The semantic leakage screen

**C1. Permitted text only: holds.**
- **Files read:** only `leakage.evaluation_question_files`:
  - `data/eval/annotation/questions.jsonl`, 10 questions;
  - `data/eval/gold_queries.jsonl`, empty.
- **What is used:** each row is parsed, but only `query_id` and `text` are used or recorded.
- **What is refused:** a file named like qrels, or a row with a passage, grade, judgment, label or pool field, before
  anything is encoded.
- **Coverage:** the 10 question texts are identical, compared by SHA-256, to the development queries the benchmark
  scores (`dev_ai/pilot_v2` and `pilot_v3`), with the same IDs. So the screen covers today's whole evaluation set.

**C2. Isolation from labels and from training: holds.**
- **Who reads the record:** only the candidate builder, and only when a candidates config names a screening.
- **What it holds:** question hashes, subject IDs, the highest similarity and the nearest evaluation question's ID,
  never its text.
- **Who never loads it:** the trainer, the validation evaluator and selection (verified by import) and the candidate
  loader (tested).
- **Fails closed:** the record binds the question files' SHA-256, so a later gold export makes every screening that
  names the old files fail closed.

**C3. The real distribution** (pinned BGE, no instruction, CPU): each question's highest cosine with the 10 permitted
evaluation questions.

| Questions | n | Median | 90th percentile | Max | ≥ 0.85 | ≥ 0.90 |
|---|---:|---:|---:|---:|---:|---:|
| The 66 candidates | 66 | 0.659 | 0.728 | 0.774 | 0 | 0 |
| Batch 1 so far (validator-passing) | 63 | 0.657 | 0.755 | 0.802 | 0 | 0 |
| `validation_pairs_v4` | 93 | 0.648 | 0.727 | 0.773 | 0 | 0 |
| All v3 validation questions passing the validator | 220 | 0.661 | 0.735 | 0.795 | 0 | 0 |
| Earlier train generation (v1, v2) | 115 | 0.665 | 0.728 | 0.774 | 0 | 0 |

- **The closest pairs:** each shares the company, and sometimes the year, with an evaluation question, but asks a
  different fact. None is a paraphrase.
- **Training questions flagged at lower thresholds** (the 66 plus the 63): 0 at 0.85, 1 at 0.80, 10 at 0.75 and 32 at
  0.70, of 129.
- **A background measure:** questions about different passages still reach high similarity when they share a template.
  - 9 of the 93 validation questions have a training question at ≥ 0.90 (max 0.947).
  - Each asks the same metric of the same company for another year.

**C4. Paraphrase probe.**
- **Design:** the audit wrote 72 variants of 12 synthetic training and validation questions (no evaluation question
  was used). They are small and unvalidated, so they are indicative only.
- **What they measure:** cosine with the original question.

| Variant | Same question? | ≥ 0.90 | ≥ 0.85 | ≥ 0.80 | Median |
|---|---|---:|---:|---:|---:|
| Light paraphrase (synonyms, reordering) | Yes | 12/12 | 12/12 | 12/12 | 0.975 |
| Heavy paraphrase (new vocabulary and structure) | Yes | 1/12 | 4/12 | 7/12 | 0.828 |
| Paraphrase with an alias, ticker or "FY19" | Yes | 0/12 | 4/12 | 9/12 | 0.832 |
| Same template, other fiscal year | No | 12/12 | 12/12 | 12/12 | 0.971 |
| Same template, other company | No | 1/12 | 7/12 | 11/12 | 0.857 |
| Same company and period, other metric | No | 3/12 | 7/12 | 10/12 | 0.862 |

**C5. Assessment.**
- **Not calibrated.**
  - BGE's cosine does not separate a paraphrase from a near-template question: the distributions overlap at every
    threshold tried.
  - At 0.9 the screen detects light rewording and same-template questions, including ones about a different year.
- **False negatives.** Reworded paraphrases of an evaluation question would pass: 11 of 12 heavy and 12 of 12 alias
  paraphrases in the probe.
- **False positives.**
  - Same-template questions for other years or metrics would be excluded. They are not paraphrases, though a
    cautious reading could still exclude them.
  - In today's data neither error occurs: nothing reaches 0.85, and the closest pairs are different facts.
- **Its weight.** The screen is a second line of defence. Evaluation passages and their overlap groups are already
  blocklisted, so a paraphrased question could only be paired with a non-evaluation passage.
- **The options (D2).**
  - (a) Keep 0.9.
  - (b) Keep automatic exclusion at 0.9 and add a recorded review tier at 0.80–0.90 (recommended).
  - (c) Lower automatic exclusion.
  - Today, (b) and (c) cost 1 question.
- **Gold questions.** When gold questions exist, re-screen every training set, and add the semantic check to
  benchmark isolation (N7). Today's 10 questions are the only ones screened against.

### D. Experiment methodology

**D1. Validation checkpoint-selection bias.**
- **Why the selected value is optimistic:** it is the maximum over up to 12 evaluations, against which step 0, also a
  candidate, competes.
- **Rule 1 is near-automatic:** a selected checkpoint never scores below step 0, so "the selected checkpoint beats
  step 0" only says that some later candidate exceeded it.
  - Under no true effect and independent noise, that happens with probability at least 0.875 per seed (3 candidates
    before the earliest possible stop) and about 0.67 for all three seeds.
  - Correlated checkpoints lower these numbers, and more candidates raise them.
- **Rule 2 is biased:** an interval for selected-minus-step-0 computed on the same 93 questions inherits the
  selection (the winner's curse).
- **Rule 3 is the useful one:** the last-checkpoint check is the only arm not chosen by the maximum. Early stopping
  makes its position data-dependent, but it is biased against A1, not for it.

**D2. The three-seed decision rule.**
- **What three seeds give:** a consistency check, not an estimate of seed variance. A seed-level interval would have 2
  degrees of freedom (t multiplier 4.30).
- **What a question-level interval covers:** only the sampling of synthetic questions, for these seeds.
- **The rule must say:** how the seeds combine (per-question differences averaged over seeds), and that all three
  seeds are reported (as `check_final_seeds` enforces).

**D3. Training and validation independence.**
- **Holds at the passage and evidence level:**
  - the split is by overlap group;
  - evidence is checked against validation passages;
  - every run is re-checked against its exact training pairs;
  - the template probe found no leak.
- **Two dependencies remain, and they affect interpretation:**
  - **One generator:** the validation questions and 51 of the 66 candidates share the prompt and model, and so will
    every P1 batch. A gain may partly be learning that style.
  - **The same filings:** both sides draw on the same filings and companies, so validation measures generalisation
    within the filings, not to new filings.

**D4. JPMorgan's concentration.**

| | Corpus | Train source pool (after source quality) | The 66 candidates | `validation_pairs_v4` | Batch-1 plan |
|---|---:|---:|---:|---:|---:|
| JPMorgan's share | 36% (3,629 of 10,137) | 54% (1,742 of 3,224) | 20% | 16% | 14% |

- **Not over-weighted in questions:** the plans cap it, and the validation split gives it a low share. Validation
  MRR@10 averages over questions, so JPMorgan does not drive selection.
- **The effect is on the corpus side.**
  - A JPMorgan question in global mode competes with 3,629 same-company passages, many repeated across years.
  - In-batch negatives from a company-balanced batch are mostly other companies' passages, which teach little
    within-company discrimination.
  - So read company-mode results per company beside global ones.
- **Later batches** will raise JPMorgan's share as Apple and Amazon run out of groups (D6).

**D5. AI-reviewed versus human-gold labels.** Every layer the comparison can use is AI-made:
- the training questions (Qwen) and their acceptance (AI review);
- the validation questions and their acceptance (AI review);
- the development labels: AI, plus 4 human grades. Their audit was AI-assisted, and the `pilot_v3` audit awaits a
  human grader.

The reviewer and the development labeller come from one model family, so their errors may be correlated: agreement
between validation and development results is not independent confirmation. No result can be described as
human-validated, and no retrieval improvement can be claimed before gold labels exist.

**D6. Reading the uncertainty.** An interval on 93 validation questions:
- **Describes:** this synthetic, AI-reviewed question distribution, conditional on the labels. Each question's relevant
  set is its positive and exact equivalents; any other relevant passage counts as non-relevant unless removed.
- **Leaves out:** seed variance (beyond 3 runs), selection bias (unless the test accounts for it), label error, and
  generalisation to real analyst questions or the gold set.
- **Its size:** the plan estimated about ±0.04 MRR@10 for a paired difference, so smaller effects will not show.
- **The development set:** with 10 questions and pools that cannot see a fine-tuned model's new passages, its intervals
  are wide and its point scores are lower bounds. It cannot confirm or refute the validation result.

**D7. A selection-aware rule, for your decision (D1).** To be written in the repository before any training run. Every
input it needs is already in each `selection.json`: per-question validation metrics at every evaluated step, for every
seed.

1. **Primary test, selection-aware:**
   - **The statistic:** for each seed, the rule-selected value minus the step-0 value, averaged over the three seeds.
   - **Its null distribution:** 10,000 random sign flips with a fixed seed. Each flip reverses all of one question's
     differences from step 0 at once, across steps and seeds. The fixed selection rule is then re-applied to the
     recorded evaluations, ties and patience included.
   - **The test:** A1 beats A0 only if the observed statistic exceeds the null's 95th percentile.
   - **The approximation:** a re-applied stop cannot evaluate past the recorded evaluations.
2. **Consistency, all required:**
   - every seed's selected and last evaluated checkpoint exceed step 0;
   - step 0 is identical across seeds (N10).
3. **Reported, not deciding:**
   - a bootstrap interval of the effect over questions and seeds, described as covering synthetic-question sampling
     only;
   - company-mode and per-company results;
   - development diagnostics, with judged@10.
4. **An alternative:** fix a random half of the 93 questions for selection and the other half for the comparison. It is
   simpler, but the interval on 46 questions would be about ±0.06.

## GO / NO-GO

- **Resume P1 batch 1 generation: GO**, under §6:
  - confirm the remaining 61 requests;
  - wait for Groq's limit;
  - dry run first, then `--max-requests 61`;
  - keep the console log;
  - make no silent setting change;
  - finish the plan.
- **The GPU rehearsal (mechanics only, the 66 candidates): no code blocker.** It can run on your training approval
  (D8), and it resolves B3.
- **The three-seed A0 versus A1 experiment: NO-GO** until all four blockers are resolved:
  - B1, a written selection-aware decision rule;
  - B2, the planned data size, or an explicitly labelled pilot;
  - B3, the CUDA preflight and the GPU rehearsal passing;
  - B4, the batch gate written down and passed.
