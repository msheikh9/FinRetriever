# `train_candidates_v3`: build report

**Date:** 2026-11-09. **Preparation:** `bge_base_ft_v3`. **Approval:** the user's final approval of 2026-11-09 (J1 to J4
of [train_v3_p1_batch6_generation_audit.md](train_v3_p1_batch6_generation_audit.md) §14). **Procedure:** §14,
conditions 1–5.

The candidates are synthetic questions written by Groq Qwen and accepted by an AI review (`reviewer: ai_review`,
`gold: false`). They are not human-validated, not gold labels and never evaluation data.

## 1. Result

**`train_candidates_v3` is built and verified: 678 training candidates.**
- **Sources:** the 66 of `train_candidates_v2`, unchanged, and 612 P1 pairs.
- **Expected counts:** every one was met at every stage (table below).
- **Checks:** 44 of 44 independent checks pass. So do the trainer's dry run, the full test suite and Ruff.
- **One finding, explained, nothing written:** a rebuild of `validation_pairs_v4` is no longer byte-identical (§8.2).
  The stored set is unchanged, and a rebuild would keep exactly the same 93 questions. Only the training side the set
  records has grown, now that P1 exists.
- **Not done:** no Groq request, no training, no change to historical records, decisions, validation data or code, no
  commit or push.

| Stage | Expected | Result |
|---|---|---|
| Train records (input) | 560 lines, `3d0b07ef…` | 560 lines, `3d0b07ef…` |
| 1. `build-pairs`: raw train pairs | 699, equal to the audit's replay | **699**, equal by pair ID, passage and question SHA-256 |
| 2. Merged decisions | 699 rows: 612 accept, 70 revise, 17 reject | **699: 612 / 70 / 17**, all `ai_review`, `gold: false` |
| 3. `screen-questions` | 765 questions, 0 at ≥ 0.90, 2 in the 0.80–0.90 band | **765, 0 flagged**, highest 0.8017; **2 in the band** |
| 4. `build-candidates` | 678, largest company share 17.55% | **678** (51 + 15 + 612), **17.55%** |

## 2. Inputs, verified before the build

All 354 files under `data`, `configs`, `src` and `scripts` were byte-identical to the batch 6 audit's after-state, so
the code and data were those that produced the audited counts.

| Input | SHA-256 |
|---|---|
| `generation/train.records.jsonl` (560 records) | `3d0b07ef…` |
| `generation/train.run.json` | `72c9e7bf…` |
| Plans `train_v3_p1_batch1` to `batch5`, `train_v3_p1_batch6_n60` | `3b9210c3…`, `156856f5…`, `fdddb936…`, `b5bc4567…`, `bfcb220d…`, `596920a8…` |
| Batch decisions 1–6 (`data/train_review/train_v3_p1_batch*_quality_decisions.jsonl`) | `52cc6ff1…`, `dc3caa90…`, `04f53d6f…`, `653897d6…`, `f79763a4…`, `4d0445c7…` |
| `configs/train/data_prep_v3.yaml` | `aa273702…` |
| Preparation manifest `bge_base_ft_v3/manifest.json` | `21c2293a…` |
| `train_candidates_v2` manifest | `fbd30a59…` |

The validator is `sec-qcheck-v3`, the equivalence method `evidence-reuse-v2` and the evidence index `d3321b40…`.

## 3. Procedure

Each real stage is write-once. So every stage was first rehearsed in a fresh scratch copy of the preparation, using the
same commands and a prep config that differs only in `output_dir`. Each real stage ran only after the rehearsal had
reproduced every expected count. Its output was then compared with the rehearsal's before the next stage.

```bash
export HF_HUB_OFFLINE=1     # the analysis environment (docs/environment.md); the encoder from the local cache only
python scripts/02_generate_queries.py build-pairs --config configs/train/data_prep_v3.yaml
python merge_decisions.py data/processed/train/bge_base_ft_v3/pairs/train_pairs.jsonl \
    data/train_review/train_v3_p1_quality_decisions.jsonl                 # the re-keying below; scratch script
python scripts/02_generate_queries.py screen-questions --config configs/train/data_prep_v3.yaml \
    --name train_candidates_v3 --pairs --candidates train_candidates_v2
python scripts/02_generate_queries.py build-candidates --config configs/train/data_prep_v3.yaml \
    --candidates configs/train/candidates_v3.yaml
```

The rehearsal also ran `revalidate --split train --dry-run`: all 559 done records are up to date under
`sec-qcheck-v3` (0 revalidated, 0 changed; the failed one is position 87 of batch 1, left failed as approved).

**Rehearsal against real:**

| Output | Real against rehearsal |
|---|---|
| `pairs/train_pairs.jsonl`, `pairs/validation_pairs.jsonl` | byte-identical |
| `pairs/pairs.meta.json` | differs only in the config path and hash it records |
| `train_v3_p1_quality_decisions.jsonl` | byte-identical, and byte-identical to the batch 6 audit's trial file |
| `screening/train_candidates_v3/screening.jsonl` | byte-identical |
| `screening/train_candidates_v3/screening.json` | differs only in the pairs-manifest hash it binds |
| `candidates/train_candidates_v3/` candidate files and log | byte-identical |
| `candidates/train_candidates_v3/manifest.json` | differs only in recorded paths and the hashes of path-bearing files |

## 4. Stage results

### 4.1 `build-pairs`

- **Train:** 560 planned passages, 559 done and 1 failed, 1,117 proposals, 884 passing the validator, **699 pairs**.
- **Pair IDs:** contiguous, `train_000001` to `train_000699`. They are identical to the batch 6 audit's replay by pair
  ID, passage and question SHA-256.
- **By batch:** 112, 124, 126, 133, 120 and 84 pairs.
- **Validation split:** `build-pairs` also built its 162 validation-split pairs from the validation records, as every
  replay did. They have no P1 decision, so `build-candidates` logs and excludes all 162. `validation_pairs_v4` is a
  separate, earlier set and was not touched.
- **Quality checks:** passed.
- **Files:** `train_pairs.jsonl` `a4e4638d…`, `validation_pairs.jsonl` `92f9a78e…`, `pairs.meta.json` `fd295a04…`.
  The pairs manifest binds the 560-record file (`3d0b07ef…`) and the run (`72c9e7bf…`).

### 4.2 The merged decisions file

- **File:** `data/train_review/train_v3_p1_quality_decisions.jsonl` (`211b5c3a…`), 699 rows.
- **Re-keying:**
  - Each batch file was keyed to its own scratch replay's pair IDs.
  - Every row was re-keyed by (`passage_id`, `question_sha256`) to the built pair ID. Every other field was kept as
    written: the decision, issues, note or reason, the reviewer, `gold` and the source audit.
  - The script refuses if a row has no pair, a pair has no decision or two, or any per-batch count differs from its
    audit.
- **The six originals:** only read; their hashes are unchanged.

| Batch file | Rows | Accept | Revise | Reject | Pair ID changed |
|---|---:|---:|---:|---:|---:|
| batch 1 | 112 | 101 | 6 | 5 | 112 |
| batch 2 | 124 | 115 | 6 | 3 | 121 |
| batch 3 | 126 | 108 | 14 | 4 | 100 |
| batch 4 | 133 | 114 | 17 | 2 | 133 |
| batch 5 | 120 | 106 | 14 | 0 | 110 |
| batch 6 | 84 | 68 | 13 | 3 | 0 |
| **Total** | **699** | **612** | **70** | **17** | 576 |

Every row has `reviewer: ai_review` and `gold: false`, and its `source` names an existing batch audit.

### 4.3 The semantic screen

`screening/train_candidates_v3/`: `screening.json` `b927b9fd…`, `screening.jsonl` `1139f499…`.
- **Questions:** 765 distinct, the 699 built train pairs and the 66 of `train_candidates_v2`.
- **Compared against:** the 10 permitted evaluation question texts (`data/eval/annotation/questions.jsonl`;
  `data/eval/gold_queries.jsonl` holds none), using `query_id` and `text` only. The encoder is BGE-base at
  `a5beb1e3…`, without the query instruction.
- **Exclusion threshold:** unchanged at 0.90. **0 questions flagged**, so the screen excluded no pair.
- **Highest similarity:** 0.801735. Of the 678 included, 26 are at or above 0.75.

**The 0.80–0.90 review band** has two questions, each reviewed in its batch audit. That review records why each
question is not a paraphrase; the evaluation question texts are not repeated here.

| Pair (built ID) | Batch pair | Similarity | Nearest | Recorded review | Decision | In v3 |
|---|---|---:|---|---|---|---|
| `train_000019` | batch 1, 005 | 0.8017 | `q_0002` | Batch 1 audit, decision E4: same company and year, a different fact; kept | accept | yes |
| `train_000186` | batch 4, 154 | 0.8014 | `q_0004` | Batch 4 audit, semantic-band row: same company only, a different measure and year; not a paraphrase | revise (scope) | no |

No pair was excluded only for being unscreened. The 41 unscreened pilot pairs were already outside
`train_candidates_v2` for other reasons, and the 162 unscreened P1 pairs are the validation-split pairs.

### 4.4 `build-candidates`

Config: `configs/train/candidates_v3.yaml` (`21ab57c2…`), which names `semantic_screening: train_candidates_v3` and
three sources.

| Source | Pairs | Included | Excluded by reason (a pair can have several) |
|---|---:|---:|---|
| `pilot_v3` (bge_base_ft_v2) | 70 | 51 | quality revise 7, quality reject 6, cross-split evidence 4, evaluation evidence 2; not screened 19 |
| `pilot_v2` (bge_base_ft_v1) | 37 | 15 | evaluation evidence 17, positive now blocked 15, quality revise 3, quality reject 1; not screened 22 |
| `p1` (this preparation) | 861 | 612 | quality revise 70, quality reject 17; no decision 162 and not screened 162 (the validation-split pairs) |
| **Total** | 968 | **678** | |

`candidates/train_candidates_v3/`:
- `train_candidates.jsonl` `7ee3a255…`, 678 rows;
- `validation_candidates.jsonl` empty;
- `candidates_log.jsonl` `7384e454…`, 968 rows;
- `manifest.json` `0586c7ca…`.

**Exclusions in P1:**
- all 612 accepted pairs are included;
- all 70 revised pairs are excluded with `quality_revise`, and all 17 rejected pairs with `quality_reject`;
- no accepted pair failed any other check.

Per batch, 101, 115, 108, 114, 106 and 68 pairs are included.

## 5. The dataset

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_candidates_v2` (the 66) | 10 | 8 | 13 | 8 | 11 | 7 | 9 | 66 |
| P1 | 53 | 35 | 91 | 111 | 106 | 108 | 108 | 612 |
| **`train_candidates_v3`** | **63** | **43** | **104** | **119** | **117** | **115** | **117** | **678** |
| Share | 9.3% | 6.3% | 15.3% | 17.55% | 17.3% | 17.0% | 17.3% | |

The largest share is 17.55% (PFE), under the 25% cap.

| Fiscal year | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 9 | 11 | 18 | 19 | 24 | – | 9 | 90 |
| 2020 | 7 | 6 | 13 | 8 | 14 | 19 | 19 | 86 |
| 2021 | 11 | 5 | 15 | 16 | 13 | 18 | 20 | 98 |
| 2022 | 7 | 5 | 18 | 21 | 10 | 16 | 19 | 96 |
| 2023 | 10 | 5 | 13 | 17 | 12 | 10 | 18 | 85 |
| 2024 | 9 | 6 | 13 | 21 | 16 | 16 | 17 | 98 |
| 2025 | 10 | 5 | 14 | 17 | 28 | 19 | 15 | 108 |
| 2026 | – | – | – | – | – | 17 | – | 17 |

(Walmart's fiscal years end in January, so its filings run from FY2020 to FY2026.)

**Composition:**
- **Passage kinds:** 350 table, 328 narrative.
- **Question types:** financial table 226, risk 139, other 98, accounting policy 73, debt and liquidity 57, MD&A 30, tax
  23, revenue 18, profitability 14.
- **Generators:** 663 candidates come from `qwen/qwen3.8-27b` with prompt `sec-qgen-v2` and 15 (`pilot_v2`) with
  `sec-qgen-v1`, all through Groq.

**Concentrations, as the batch 6 audit reported:**
- Tesla FY2019 has 24 and FY2025 has 28 of Tesla's 117.
- Walmart FY2025–FY2026 has 36 of Walmart's 115.
- Amazon FY2019 has 11 of Amazon's 43.

## 6. Verification

| # | Check | Result |
|---|---|---|
| 1 | Raw pair count | **699** train pairs, contiguous IDs, equal to the audit's replay; per batch as audited |
| 2 | Accepted, revised, rejected | **612 / 70 / 17**, each decision bound to its pair's passage and question SHA-256. In the build, 612 accepted pairs are included, and 70 revised and 17 rejected are excluded. |
| 3 | Semantic screen | **0 exclusions** at ≥ 0.90. Of the 2 band questions, both reviewed, 1 is included and 1 excluded by its revision. Every included question was screened and is not flagged. The screen is bound in the manifest (`b927b9fd…`). |
| 4 | Final count | **678** = 51 + 15 + 612 (manifest, files and log agree) |
| 5 | Company and fiscal-year distribution | §5; largest share 17.55% |
| 6 | Provenance | `verify_candidate_provenance`: **678 traced** (`p1` 612, `pilot_v3` 51, `pilot_v2` 15). Every record names its provider, model, prompt version and hash, generation parameters, corpus, validator `sec-qcheck-v3`, the preparation manifest `21c2293a…` and its origin (`build-pairs` or `carryover:<pilot>`). |
| 7 | Candidate loader and trainer | `load_candidates`: **678** re-verified pairs. Trainer dry run: passes (below). |
| 8 | Validation isolation | `load_validation_set(..., "validation_pairs_v4", training=<the 678>)`: **93 returned, 0 problems**. Independent checks: §8. |
| 9 | `train_candidates_v1` and `v2` unchanged | Every file's SHA-256 is unchanged, and both rebuild as `unchanged (identical copy present)` (§9) |

**Safeguards:**

| Safeguard | Result |
|---|---|
| The original 66 preserved | Every field of all 66 records in v3 equals `train_candidates_v2`'s, including equivalents and provenance. Their log reasons equal v2's, apart from the added "not screened" on the 41 pairs v2 had already excluded. |
| 70 revised and 17 rejected excluded | Yes (check 2) |
| `reviewer: ai_review`, `gold: false` | All 699 merged rows, and every log row that carries a decision (all 806 train-split source pairs) |
| `sec-qcheck-v3` not modified | `src` is unchanged (§9). `VALIDATOR_VERSION` is `sec-qcheck-v3`, and every record and the manifest name it. |
| Full provenance | Check 6 |
| Equivalents and false-negative protections recomputed | §7 |
| Separation from `validation_pairs_v4` | Check 8, §8 |
| No evaluation labels or judged passages | §8.3 |
| Batches 1–6 records and decisions byte-for-byte | §9 |

**The trainer's dry run** (`scripts/05_train_biencoder.py`, without `--confirm-training`). It used a scratch copy of
`biencoder_bge_base_ft_v3_rehearsal.yaml` that differs only in its name, `candidates: train_candidates_v3` and a
scratch `output_dir`. It loads no model and writes nothing; the output directory was never created.
- It loads **678 training candidates** and plans every seed:

  | Seed | Optimizer steps | Warmup | Single-pair batches dropped |
  |---:|---:|---:|---:|
  | 42 | 64 | 7 | 7 |
  | 1337 | 59 | 6 | 5 |
  | 2026 | 62 | 7 | 1 |

  The settings are batch 64, GradCache mini-batches of 16 and 5 epochs.
- It verifies that no batch holds a known equivalent, overlap sibling, reuse link, blocked or other-split passage as an
  in-batch negative.
- It confirms that `validation_pairs_v4` (93) is isolated from the 678, with 10,137 passages ranked per evaluation.

## 7. Equivalents and false-negative protections

- **Recomputed:** `build-candidates` recomputed every pair's equivalents under `evidence-reuse-v2`. A second,
  independent recomputation with the preparation's locator gives the same exact and uncertain sets for all 678, under
  the same index digest (`d3321b40…`).
  - 348 candidates have exact equivalents (799 passages).
  - 316 have uncertain equivalents (982 passages).
- **Never negatives:** for every candidate, `NegativeGuard` excludes as a negative every one of these passages:
  - its exact and uncertain equivalents;
  - every member of its overlap group;
  - every reuse link.

  There are 0 exceptions.
- **Conflicts:** 913 candidate pairs may not share a batch, as in the batch 6 audit. `plan_batches` (seed 42) places
  every pair with no conflict:

  | Batch size | Batches | Full | Smallest | Mean fill |
  |---:|---:|---:|---:|---:|
  | 16 | 43 | 42 | 6 | 0.986 |
  | 32 | 24 | 20 | 1 | 0.883 |
  | 64 | 14 | 10 | 1 | 0.757 |

- **Cross-passage figure probe** over all 678 candidates. It looks for another candidate's positive whose visible text
  contains a 4-digit-or-longer answer figure (not a year) where the guard would allow that positive as a negative.
  - 44 hits on 23 candidates, **all across companies (0 within a company)**, such as "1934", the Securities Exchange Act
    year.
  - Every question names its company, so none of these passages can answer another company's question.
  - The within-company repeats the batch audits found are all in conflict or were revised (batch 6's pair 621).

## 8. Validation and evaluation isolation

### 8.1 `train_candidates_v3` against `validation_pairs_v4`

- `load_validation_set` with the 678 as the training side returns all 93 validation pairs with no problem.
- **Passages:** no passage is both a training-side exclusion (a positive, an equivalent, an overlap sibling or a reuse
  link of any of the 678) and a validation positive or equivalent.
- **Groups:** no overlap group is shared.
- **Splits:** none of the candidates' passages is in the validation split.
- **Questions:** no normalised question is shared.
- **Highest word-set Jaccard** between a candidate and a validation question: 0.765, below the 0.80 threshold.
  - The pair is P1 `train_000052`, Apple corporate debt securities' fair value in FY2025, against v4
    `val_pilot_v2f:validation_000001`, the Level 2 figure in FY2022.
  - They share a template, but the year, the passage and the answer differ.

### 8.2 Finding: a `validation_pairs_v4` rebuild is refused as different; the set is unchanged

`build-validation --validation configs/train/validation_v4.yaml` refused with "already holds a different validation
set … nothing was written". Its files keep their SHA-256 (`40238cb0…`, `2298e1e5…`, `f72e6370…`).

**Cause.** The validation builder's training side includes the preparation's train generation records and, once
`pairs/pairs.meta.json` exists, its built train pairs (`validation_set.training_side`). `validation_pairs_v4` was built
on 2026-11-04, before P1's records and before any `build-pairs` in this preparation, so its training side was the 66 of
`train_candidates_v2` alone. Since today's real `build-pairs`, the same config sees all of P1.

**What a rebuild would change.** I captured the rebuild in memory, with the writer replaced, so nothing could be
written, and compared it with the stored files.
- `validation_pairs.jsonl`: **byte-identical, the same 93 questions.**
- Log: no row's inclusion changes. 195 rows differ only in their recorded `max_training_similarity`, and 5 already
  excluded questions gain `training_query_overlap` as an extra reason. They are `val_pilot_v2f` 120, 151, 158, 198 and
  233.
- Manifest: the recorded training side.

  | | `validation_pairs_v4` (stored) | Rebuild today |
  |---|---|---|
  | Training questions | 66 | 1,180 |
  | Training passages | 188 | 1,706 |
  | Records file hash | empty | `3d0b07ef…` |
  | Pairs manifest hash | empty | `fd295a04…` |
  | Highest training similarity | 0.739 | 0.765 |

**Meaning.**
- `validation_pairs_v4` is unchanged and still valid for checkpoint selection.
- Its isolation holds against the whole of P1, which is stronger than the check against the 678:
  - every question generated for P1 (accepted, rejected or revised);
  - all 699 built pairs' passages.
- What no longer holds is the earlier statement that v4 "rebuilds unchanged". Its manifest records the training side
  it was built against, by design.

**Not acted on:** no new validation version was built, and none is needed for the selection rule. If a byte-identical
rebuild of the validation set matters for the experiment's records, the choice is yours:
- keep v4 and record this note (recommended);
- build a new version, `validation_pairs_v5`, that binds the P1 training side. It would hold the same 93 questions.

### 8.3 Evaluation data

- **Positives:** every candidate's positive is in the train split.
- **Blocked passages:** no positive, equivalent, overlap sibling or reuse link of any candidate is blocked. Blocked
  passages are the evaluation pools, audit samples and their overlap and reuse expansions.
- **Leaks:** no included pair has a recorded evidence leak.
- **Lexical overlap:** the highest word-set Jaccard with an evaluation question is 0.346, below the 0.80 threshold.
- **Records:** no candidate record has a label-bearing field (relevance, grade, judgment, qrels, label, pool), and the
  candidate file contains no `data/eval` path.

**What was read:**
- **Through the pipeline's own loader:** `build-pairs`, `screen-questions`, `build-candidates` and the rebuild checks
  read the 10 permitted evaluation question texts, `query_id` and `text` only. No evaluation question text was printed;
  only the two nearest IDs, `q_0002` and `q_0004`, which the screen records.
- **Never:** no qrels, relevance labels, judged passages, pools or audits were opened.
- **One disclosure:** while checking file formats I printed:
  - the first three rows of the preparation's own `blocklist.jsonl`: passage IDs with block reasons such as
    `annotation_pool`, and no relevance information;
  - the counts and hashes in the manifest's blocklist section.

  Nothing else from them was used. The checks above use only each passage's split in the preparation.

## 9. Unchanged files

**Protected files** (`find data configs src scripts -type f`, SHA-256): **354 before and 365 after, with 0 modified and
0 removed.** The 11 added files are the ten of §11 under `data/processed` and `data/train_review`, plus
`configs/train/candidates_v3.yaml`. So these are byte-for-byte unchanged:
- the six batch decision files;
- the 560-record train records file and its run file;
- the six P1 plans;
- the validation records and validation sets;
- `train_candidates_v1` and `train_candidates_v2`;
- every source file, the validator included;
- every config.

**Rebuild checks** (write-once; an identical rebuild writes nothing):
- `build-candidates --config configs/train/data_prep_v3.yaml --candidates configs/train/candidates_v2.yaml`:
  `unchanged (identical copy present)`;
- `build-candidates --config configs/train/data_prep_v2.yaml --candidates configs/train/candidates_v1.yaml`:
  `unchanged (identical copy present)`;
- `validation_pairs_v4`: §8.2.

| Dataset | Files (SHA-256, unchanged) |
|---|---|
| `train_candidates_v1` (bge_base_ft_v2) | log `0ad4450c…`, manifest `6fb3aa72…`, train `14f491ec…` |
| `train_candidates_v2` (bge_base_ft_v3) | log `c8d38f96…`, manifest `fbd30a59…`, train `908d4c79…` |
| `validation_pairs_v4` | manifest `40238cb0…`, log `2298e1e5…`, pairs `f72e6370…` |

## 10. Tests and lint

| Run | Result |
|---|---|
| `pytest` (the dense venv, `HF_HUB_OFFLINE=1`) | **1,267 passed, 25 skipped**, 0 failed (79 s) |
| `pytest` (global `python3`) | **1,264 passed, 28 skipped**, 0 failed (75 s) |
| `ruff check src tests scripts` | All checks passed |
| `git diff --check` | Clean: no tracked file changed, and the new config and this report have no trailing whitespace |

**The skips are environmental:**
- 24 need the Apple 10-K fixture (`FINRETRIEVER_AAPL_10K`), which is absent here;
- 1 needs `pytrec_eval`, which is not installed;
- 3 under the global Python cannot load BERT, because of this Mac's scipy dlopen error.

The venv's two warnings are third-party deprecation notices (SWIG).

## 11. Files written

| File | SHA-256 | Note |
|---|---|---|
| `data/processed/train/bge_base_ft_v3/pairs/train_pairs.jsonl` | `a4e4638d…` | 699 pairs; gitignored |
| `data/processed/train/bge_base_ft_v3/pairs/validation_pairs.jsonl` | `92f9a78e…` | 162 validation-split pairs, unused; gitignored |
| `data/processed/train/bge_base_ft_v3/pairs/pairs.meta.json` | `fd295a04…` | gitignored |
| `data/processed/train/bge_base_ft_v3/screening/train_candidates_v3/screening.json` | `b927b9fd…` | gitignored |
| `data/processed/train/bge_base_ft_v3/screening/train_candidates_v3/screening.jsonl` | `1139f499…` | gitignored |
| `data/processed/train/bge_base_ft_v3/candidates/train_candidates_v3/` (4 files) | manifest `0586c7ca…`, train `7ee3a255…`, log `7384e454…` | gitignored |
| `data/train_review/train_v3_p1_quality_decisions.jsonl` | `211b5c3a…` | merged decisions; new, uncommitted |
| `configs/train/candidates_v3.yaml` | `21ab57c2…` | new, uncommitted |
| `docs/train_candidates_v3_build_report.md` | | this report; uncommitted |

The scratch tools are `merge_decisions.py` and `verify_v3.py`. They live under the session scratchpad, at
`<session scratch directory>/v3build/`,
with `verify.json`.

## 12. What follows

- **P1 is closed** at 678 candidates (J2), which settles blocker B2 of the readiness audit. 68,510 of the approved
  tokens were not spent.
- **The three-seed experiment stays NO-GO.** Blockers B1 (the selection-aware decision rule) and B3 (CUDA preflight
  and the GPU rehearsal) remain. The next steps, each needing approval:
  - write the experiment's training config on `train_candidates_v3`;
  - run its bundle and preflight (readiness T5).
- **For the experiment config:**
  - At batch size 64 the 913 conflicts leave single-pair batches, which the trainer drops: 7, 5 and 1 over five epochs
    for seeds 42, 1337 and 2026. Each epoch has 11–14 batches of 64.
  - Whether to keep 64 is a choice to record with the config.
- **Data-quality limitations:** unchanged from the batch 6 audit §13. In short:
  - single-pass AI review, not gold;
  - error-prone JPMorgan segment and ExxonMobil supplemental sources, filtered by the review;
  - 101 boilerplate pairs;
  - company and year imbalance (Apple and Amazon underweight; Tesla FY2019/FY2025 and Walmart FY2025–FY2026 heavy);
  - topic and template concentration;
  - false-negative protection limited to recorded equivalents;
  - the frozen validator's known gaps;
  - the uncalibrated screen;
  - the revised pairs excluded rather than repaired.
