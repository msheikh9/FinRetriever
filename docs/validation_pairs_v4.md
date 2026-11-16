# The final validation set: `validation_pairs_v4` (`bge_base_ft_v3`)

Status: 2026-11-04. Validation-question generation is stopped, and this set is frozen.

**What it is:**
- 93 synthetic questions, each with one positive passage from the validation split.
- An LLM (`qwen/qwen3.8-27b` through Groq) wrote the questions.
- An AI quality review (Claude) accepted every one.

**What it is not:**
- not human-validated;
- not gold labels;
- never training data and never evaluation data.

**What it is for:** choosing among checkpoints of a future training run.

**What it was not used for:** the three smoke-test checkpoints predate it, and nothing was selected with it.

| Item | Value |
|---|---|
| Directory | `data/processed/train/bge_base_ft_v3/validation_sets/validation_pairs_v4/` (gitignored, written once) |
| `manifest.json` SHA-256 | `40238cb06d73287821115dbe1b5c33a2dbbd294c7dd5b3c9fdc569c475e24b0e` |
| `validation_pairs.jsonl` SHA-256 | `f72e63706498fe606a0ec12556dab6b0e0ae4ade35fe9ee34d8746eb68e6d98f` |
| `validation_log.jsonl` SHA-256 | `2298e1e595308f60c5452ae8e07f9ebaf30a2011432227450fc6abf31c35bb44` |
| Config | [configs/train/validation_v4.yaml](../configs/train/validation_v4.yaml) |
| Corpus | `corpus_v1@1.2`, SHA-256 `ff9dadf3…2004`, unchanged |
| Preparation | `bge_base_ft_v3`, manifest `21c2293a…`, unchanged |

## 1. How it was built

Each version carries the one before it and re-checks every carried question against `bge_base_ft_v3`. That
is the design the earlier sets set out: one plan per version, with earlier sets carried.

| Version | Carries | Adds | Questions |
|---|---|---|---:|
| `validation_pairs_v2` (existing) | `validation_pairs_v1` (`bge_base_ft_v2`) | none | 16 |
| `validation_pairs_v3` (new, [config](../configs/train/validation_v3.yaml)) | `validation_pairs_v2` | `val_pilot_v2e` positions 1–50 (batch 1) | 31 |
| `validation_pairs_v4` (new, final) | `validation_pairs_v3` | `val_pilot_v2f` positions 1–234 | **93** |

```bash
python3 scripts/02_generate_queries.py stop-plan --config configs/train/data_prep_v3.yaml --split validation \
    --plan val_pilot_v2f --reason "..."          # done once: closes the plan at position 234
python3 scripts/02_generate_queries.py build-validation --config configs/train/data_prep_v3.yaml \
    --validation configs/train/validation_v3.yaml
python3 scripts/02_generate_queries.py build-validation --config configs/train/data_prep_v3.yaml \
    --validation configs/train/validation_v4.yaml
```

**Decisions:** every question of positions 1–234 that the generator accepted has one AI-review decision.
They are in three files under `data/validation_review/`, and the set binds each file by its SHA-256.

| File | Positions | Decisions | Of them accepted |
|---|---|---:|---:|
| `val_pilot_v2e_batch1_quality_decisions.jsonl` | `val_pilot_v2e` 1–50 | 39 | 36 |
| `val_pilot_v2f_positions1-147_quality_decisions.jsonl` (existing) | `val_pilot_v2f` 1–147 | 115 | 101 |
| `val_pilot_v2f_positions148-234_quality_decisions.jsonl` (new) | `val_pilot_v2f` 148–234 | 66 | 63 |

The 66 new decisions come from two audits:
- [batch 4](validation_plan_v2f_batch4_audit.md), positions 148–195;
- [positions 196–234](validation_plan_v2f_positions_196_234_audit.md), reviewed for this freeze.

Each decision row names its audit (`source`) and carries `reviewer: ai_review` and `gold: false`.

**Generation is stopped:** `plans/validation.val_pilot_v2f.stopped.json` (SHA-256 `c8168d44…`) records the stop.
- **Position:** 234 of 361.
- **Records bound:** the 284 record lines existing then (SHA-256 `5fd18cda…`).
- **Reason and date:** recorded in the file.
- **Effect:** `generate --plan val_pilot_v2f` refuses before reading any credential, so no request can be sent
  under it.
- **Binding:** `validation_pairs_v4` must end at or before the stop position, and its manifest binds the stop
  record.
- **Other plans:** `val_pilot_v2e` (stopped in practice after batch 1) and `val_pilot_v2g` (never run) have no
  stop record. The same command can close them.

## 2. What is in it

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `val_pilot_v1` (the original 16) | 3 | 3 | 2 | 1 | 2 | 2 | 3 | 16 |
| `val_pilot_v2e` batch 1 | 2 | 1 | 2 | 1 | 4 | 2 | 3 | 15 |
| `val_pilot_v2f` 1–234 | 6 | 11 | 11 | 8 | 7 | 11 | 8 | 62 |
| **Total** | 11 | 15 | 15 | 10 | 13 | 15 | 14 | **93** |

- **By fiscal year:** 2019 18, 2020 14, 2021 11, 2022 11, 2023 11, 2024 8, 2025 19, 2026 1. FY2024 is the
  thinnest year.
- **By passage kind and question type:**
  - 50 table and 43 narrative passages;
  - 50 financial-table questions, 12 risk, 10 accounting policy, 9 other, and 12 of other types.
- **Imprecision:** the review passed every included question on every aspect, with some marked minor.
  - **By aspect:** units 14, financial relevance 12, period 9, factual accuracy 9 and figures 2.
  - **Issues recorded:** `inaccurate_wording` 8 and `boilerplate` 5.
- **Equivalents:**
  - 25 questions have an exact and 15 an uncertain equivalent passage;
  - all of them lie in the validation split, and the pairs record them.
  - Correlated questions should be read with them.

## 3. Checks

Every carried and every new question was checked again when the set was built. The set reloads with all
of these checks, failing closed if any input changed.

| Check | Result |
|---|---|
| The positive is a validation passage whose overlap group and split unit lie wholly in validation | 93 of 93 |
| No train or blocked passage holds the evidence, exactly or possibly (the strict evidence rule) | 93 of 93 |
| Positive and equivalents: passages on the evaluation blocklist | 0 of 170 |
| Positive and equivalents: passages judged in the `dev_pilot_v3` evaluation labels | 0 of 170 |
| A passage shared with the 66 training candidates (`train_candidates_v2`, 188 passages) | none |
| A question shared or lexically overlapping with a training question (Jaccard ≥ 0.8) | none; highest 0.739 |
| Lexical overlap with an evaluation question | highest 0.31 (threshold 0.8) |
| Duplicate questions | none |
| Provenance: each planned question is its record's accepted candidate, and each carried one is the pair of its source set | 93 of 93 |
| Every decision is an AI review with `gold: false` | yes |

**Unchanged:**
- the corpus, both preparations, the 66 training candidates and every earlier training pair;
- the generation records: the first 284 lines are those bound by the stop, and the file was only read;
- the plans and the earlier validation sets and decisions;
- the evaluation data and labels.

## 4. Code changes

- `validation_set`:
  - **Plan prefix:** a set can take a plan's first N positions (`plan_positions`), where its generation
    ended. Later positions are neither checked nor logged.
  - **Several decision files:** they are read together, and a question decided in two of them is refused.
  - **Stop binding:** a stopped plan must be cut at or before its stop, and the manifest binds the stop
    record. Every decision file is bound like the first.
  - A set built from one decisions file has the same manifest as before. `validation_pairs_v1` and
    `validation_pairs_v2` still load unchanged.
- `sampling_plan`: stop records (`make_stop_record`, `write_stop_record`, `read_stop_record`), written once
  beside the plan.
- `data_cli`:
  - `stop-plan` writes the stop record;
  - `generate --plan` refuses a stopped plan, whether or not `--confirm-api-use` is given.
- **Tests** (`tests/unit/test_validation_set.py`):
  - the prefix and several decision files;
  - every refusal;
  - the stop: no request is sent and no record is written, the stop record is bound, and a tampered stop
    record is refused.

## 5. Limitations

- **Below target:** 93 questions, short of the 120 the split was designed for. They are enough to compare
  checkpoints of one run by paired differences, if those differ about as much as measured in
  [validation_split_v3_design.md](validation_split_v3_design.md). They are not enough to prove small gains.
- **AI-reviewed synthetic questions:** an LLM wrote them from single passages, and an AI review accepted them.
  - Their wording follows a few templates, especially for balance-sheet figures.
  - Some answers omit a qualifier that the passage's table gives (marked minor).
- **The evidence rule is lexical.** It matches quotes and figures, not paraphrases.
  - A fact restated in other words in a training passage can pass it.
  - Batch 4's audit records one known case, position 149.
- **FY2024 has 8 questions** and 2026 has 1. Per-year comparisons on this set are not meaningful.
