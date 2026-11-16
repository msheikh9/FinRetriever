# A1 seed 2026: readiness report

**Date:** 2026-11-11.

**Status (updated the same day, after the interactive dataset check):**
- Seed 2026 is **not trained**. At your request, the **official notebook is generated with the training switch
  open**: `finretriever_a1_train_seed2026.ipynb`, SHA-256
  `c6ca9e5bfb670587c4a5a6deb1bf442939e4968346a9928ea688e7dff171ed1a`. It differs from the closed notebook, which you
  checked on Kaggle, in exactly one line: `START_OPTIMIZER = True`.
- **The interactive dataset check on Kaggle passed** (§7). On Kaggle's real mount, the dataset was identified,
  confirmed against the pinned manifest and recorded, and training was refused.
- **The final local checks pass** on the official notebook (§8).
- **Launch instructions:** §9. The training run is yours to start, once, as a saved version.
- No hyperparameter, dataset, trainer code or historical record changed. No API request, commit or push.

**How to read this report:** sections 1–6 are the readiness review of the closed notebook, as first submitted.
Sections 7–9 record what followed: the Kaggle check, the official notebook and the launch.

**Where everything is:** `dist/kaggle_a1_seed_v3/` (version 3; gitignored, like versions 1 and 2). The version 2
files in `dist/kaggle_a1_seed_v2/`, which trained seed 1337, are unchanged.

## 1. Summary

| Item | Result |
|---|---|
| Dataset provenance in `execution.json` | Implemented. It is recorded in the first code cell, before anything reads the inputs. |
| Expected training dataset confirmed, fail closed | `finretriever-train-v3-a1-r2`, by directory and by the pinned archive and manifest hashes. Anything else stops the notebook before the optimizer can start. |
| Bound to the run identity and the output verification | The gate decision records the provenance digest. A sixth verification check, `dataset_provenance`, ties the record to the bundle the verification ran from and to every input `run.json` records. |
| No secrets recorded | No file content, environment-variable value or credential is recorded. Tested with secrets planted in the environment, in the inputs and in Kaggle-style notebook metadata. |
| Seed 2026 notebook | `finretriever_a1_train_seed2026.ipynb`, `START_OPTIMIZER = False` (`f97cfc4a…`). Every A1 setting is unchanged, on the same T4 stack. **Now the official notebook, switch open: `c6ca9e5b…` (§8).** |
| Tests | Repository suite: 1,309 passed, 25 skipped. Version 3: 150 passed (151 with the official notebook, §8). Version 2: 25 passed. Ruff: clean. |
| Notebook, cell by cell on this Mac | 13 of 13 scenarios behave as required (§4.2). On the official notebook, 15 of 15, including Kaggle's exact mount (§8). |
| Seeds 42 and 1337 | Byte-identical: archives, extractions, selection records, checkpoints and notebooks (§5) |

## 2. The dataset provenance

### 2.1 What Kaggle exposes, and what the notebook relies on

- **Mount layout:** Kaggle mounts each attached dataset under `/kaggle/input`, conventionally in a directory named
  after the dataset's slug. Seeds 42 and 1337 found the bundle by searching `/kaggle/input`, but no record kept the
  path (L5 in the seed 1337 audit).
- **Kaggle's own identifiers:** I found no official documentation of how a running notebook can learn the numeric
  identifiers of its inputs. A saved version's notebook has a copy at `/kaggle/src/script.ipynb`, as the execution
  probe showed. Exported Kaggle notebooks may carry `metadata.kaggle.dataSources`.
- **The design follows from that:**
  - **Decides:** only what the notebook can observe and hash, namely the directory holding the bundle and the
    bundle's bytes.
  - **Recorded only, where available:** Kaggle's numeric identifiers. They never decide.

### 2.2 What is recorded: `execution.json` → `datasets`

The execution identity becomes format 2 and gains `datasets`. Everything else in it is as in version 2.

| Field | Content |
|---|---|
| `attached` | One entry per input directly under `/kaggle/input`: its name, kind, file count, total bytes, and a digest of its listing (relative paths and sizes). No file is opened. |
| `kaggle_data_sources` | From a saved version's `/kaggle/src/script.ipynb`, only these keys: `sourceType`, `sourceId`, `datasetId`, `databundleVersionId` and `isSourceIdPinned` for each data source, plus `dockerImageVersionId`, `accelerator`, `isInternetEnabled` and `isGpuEnabled`. Only booleans, integers and short identifier-like strings are kept. Otherwise `available: false` with the reason. |
| `training` | The expected name; every copy of the bundle found; the dataset's path, layout (`top-level` or `nested`) and the bundle's form (`archive` or `unpacked`); the archive's bytes and SHA-256, its `.sha256` file and the pinned value; the bundle manifest's SHA-256, the pinned value and the name it gives; `identified`, `confirmed` and `problems`; and the times identified and confirmed. |

The gate record adds `training_dataset_confirmed` and `datasets_sha256`, the digest of the whole `datasets` record as
canonical JSON. `pins.json` adds `dataset`. `pretraining_checks.json` adds `0_training_dataset`.

**What the record looks like:** the record the first code cell writes, built by the notebook's own code from the
real r2 archive in Kaggle's usual layout:

```
"training": {"expected_name": "finretriever-train-v3-a1-r2", "bundle": "finretriever_train_v3_a1_r2",
  "identified": true, "name": "finretriever-train-v3-a1-r2", "path": "finretriever-train-v3-a1-r2",
  "layout": "top-level", "form": "archive",
  "source": "finretriever-train-v3-a1-r2/finretriever_train_v3_a1_r2.tar.gz", "archive_bytes": 9550920,
  "archive_sha256": "efc377d9…", "archive_sha256_listed": "efc377d9…", "archive_sha256_expected": "efc377d9…",
  "manifest_sha256": "a6615850…", "manifest_sha256_expected": "a6615850…",
  "manifest_name": "finretriever_train_v3_a1_r2", "confirmed": true, "problems": [], …}
"attached": [{"name": "finretriever-train-v3-a1-r2", "kind": "directory", "files": 2, "bytes": 9551021, …}]
```

### 2.3 When the notebook refuses: fail closed

**Identification.** The training dataset is identified only if every copy of the bundle under the input root lies
in exactly one directory named `finretriever-train-v3-a1-r2`. A copy is either the archive
`finretriever_train_v3_a1_r2.tar.gz` or an unpacked `BUNDLE_MANIFEST.json` that names the bundle. Each copy counts
for the nearest directory of that name, so a nested layout such as `datasets/<owner>/<slug>/` is also identified.

**Confirmation.** The dataset is confirmed only if it was identified, and:
- in archive form, the archive hashes to its `.sha256` file and to the pinned build `efc377d9…`. Only then is it
  unpacked;
- the bundle's `BUNDLE_MANIFEST.json` hashes to the pinned manifest `a6615850…` and names the bundle.

**Refused, with the record written and the outputs packaged:**
- no copy of the bundle;
- the bundle under another dataset's name;
- a second copy anywhere else;
- another build under the expected name.

Each stops the notebook in its first code cell, before the environment, the preflight and the gate.

**Other inputs.** Inputs without a copy of the bundle are recorded by name and do not block. That includes other
bundles, such as r1 (`finretriever_train_v3_a1`), whose manifests name another bundle.

**Training needs three conditions,** read afresh at the gate and again in the training cell:
1. `START_OPTIMIZER is True`. It must now be exactly `True`; version 2 accepted any truthy value.
2. `KAGGLE_KERNEL_RUN_TYPE == "Batch"` exactly, through the unchanged, probe-verified code `3cd0bc44…`.
3. A confirmed training dataset.

### 2.4 How it is bound to the run and its verification

1. **Run identity:** the provenance is in `execution.json`, which is copied into the run directory, hashed in
   `OUTPUT_SHA256SUMS`, and binds `run.json` and `selection.json` by hash, as before.
2. **Gate:** the gate records `datasets_sha256`. Any later change to the provenance breaks that binding.
3. **Run verification:** the notebook's `verify_seed_run.py` (version 3, `679ff007…`) gains a sixth check,
   `dataset_provenance`. It requires:
   - a format 2 identity;
   - a dataset that was identified and confirmed with no problem, under the expected name, and listed among the
     attached inputs;
   - in archive form, the archive equal to its `.sha256` file, the pin, and the identity's
     `bundle_archive_sha256`; in unpacked form, no archive hash;
   - the manifest equal to the pin, to the identity's `manifest_sha256`, and **to the `BUNDLE_MANIFEST.json` the
     verification itself runs from**. That manifest must name the bundle;
   - **every input `run.json` records** in that manifest, with the same SHA-256 and role: the config, the
     data-preparation config, the corpus, and the preparation, candidates and validation manifests;
   - a gate that saw the confirmed dataset, with a digest equal to the record's;
   - times in order: identified, confirmed, gate, training.

   The chain therefore runs: dataset directory → archive → manifest → `run.json` inputs → checkpoints.
4. **Locally:** `check_outputs.py` (version 3) prints the provenance. It checks the provenance against the pins in
   `execution.json` and `pins.json`, and requires the run verification's `dataset_provenance` check to have passed.
   If a run that trained fails any of these, it exits with 1.

### 2.5 What is never recorded

- **Environment variables:** none is read except `KAGGLE_KERNEL_RUN_TYPE` (the execution-mode code, unchanged) and
  the notebook's own path overrides for local testing.
- **Files:** no file content is read from any input except the bundle's own `BUNDLE_MANIFEST.json`, for its `name`.
- **Kaggle's notebook copy:** only the listed keys, and only plain values, are kept.
- **Tests:**
  - the module tests plant a secret in two `KAGGLE_` variables, in a file of another dataset, and in Kaggle-style
    metadata (in a whitelisted key, in a non-whitelisted key and in a cell);
  - every harness scenario plants it in the environment, in an attached file and in the notebook copy. It then
    searches every output file, the run directory and the zip.
- **Result:** the secret was never found.

## 3. The seed 2026 notebook

**File:** `dist/kaggle_a1_seed_v3/finretriever_a1_train_seed2026.ipynb`, SHA-256
`f97cfc4afc662ed6bfe55426545e2978f5a6210089c4fe5c65be5c25dfd0d955`, with `START_OPTIMIZER = False`.
- **Generator:** `make_seed_notebook.py` (`e95ba648…`).
- **What it embeds:** the execution-mode code `3cd0bc44…` (byte-identical to the code that gated seed 1337 and that
  the Kaggle probe verified), the provenance code `dataset_provenance.py` (`7e91ec33…`) and the verification script
  (`679ff007…`). Each is pinned by SHA-256 and checked at run time.
- **Generation check:** the generator confirms that every code cell compiles and that each embedded file round-trips
  byte for byte.

**What changed from the notebook that trained seed 1337:**
- **The seed:** seed 2026, its schedule `49b2b153…`, and the output names. These are exactly the changes between
  version 2's seed 1337 and seed 2026 notebooks.
- **The switch:** closed.
- **The provenance:** it changes only four code cells: setup, dry run and gate, training refusal, and verification
  arguments.
- **Unchanged:** the other five code cells (bundle hashes, install, environment, preflight, packaging) are
  byte-identical to version 2's for the same seed. Every constant version 2 defines keeps its value; version 3 adds
  only `DATASET` and `PROVENANCE_CODE`. The training command is unchanged.

**The approved A1 configuration, unchanged:**

| Setting | Value | Where it is pinned |
|---|---|---|
| Learning rate | 2e-5 | `configs/train/biencoder_bge_base_ft_v3_a1.yaml`, SHA-256 `7a4c4618…`, equal in the repository and the bundle |
| Batch size | 64 | the same config |
| GradCache mini-batch | 16 | the same config |
| FP16 and GradScaler | fp16 on; the default GradScaler, initial scale 65,536 | the same config, and the trainer (unchanged, inside the hash-pinned bundle) |
| Schedule | 55 steps, 6 of warmup; seed 2026's schedule `49b2b15328d978904c017bb699979a1ed62e20a937e8c5ecde5b152f1b4ccede` | the notebook's pin, the dry run and the identity check |
| Batching | balanced, 11 × 5, none dropped, at least 60 negatives | the same checks |
| Checkpoint selection | K1: step 0 and every 5 steps, whole-corpus MRR@10, ties to the earlier step, patience 3 | the trainer (unchanged) |
| Training data | `train_candidates_v3`: 678 candidates, manifest `0586c7ca…` | `EXPECTED_INPUTS`, through the trainer's own loaders |
| Validation data | `validation_pairs_v4`: 93 questions, manifest `40238cb0…`, IDs `b83abf1a…` | the same |
| Environment | Tesla T4 (7.5), CUDA 13.0, cuDNN 92700, Python 3.13.16, torch 2.11.0+cu130, transformers 5.3.0, tokenizers 0.22.2, plus seed 42's 13 training packages | `REHEARSAL_ENVIRONMENT` and `SEED42_STACK`, unchanged |

**Seed 2026's schedule is already on record from both T4 runs.** The preflights of seed 42 (`57c290ff…`) and seed
1337 (`460e80f0…`) plan every seed. For seed 2026 they record `49b2b153…`, 55 steps, no dropped batch, and at least
60 negatives per query, all passing.

## 4. Verification (your items 1–9)

| # | Requirement | Result |
|---|---|---|
| 1 | Full local test suite | Repository: **1,309 passed, 25 skipped** in 90 s, the same counts as before (no repository code changed). Version 3: **150 passed** (§4.1). Version 2: **25 passed**. |
| 2 | Ruff | Repository (`ruff check .`): clean. Version 3 sources, at the 120 columns of the `dist/` tools: clean. |
| 3 | Seeds 42 and 1337 byte-identical | Yes (§5) |
| 4 | Their selection records and checkpoints unchanged | Yes (§5) |
| 5 | The provenance field, tested | 18 module tests, 24 verifier tests of the `dataset_provenance` check (3 passing layouts, 21 faults), the training condition's truth table, and 13 notebook scenarios (§4.1, §4.2) |
| 6 | The five pre-training gates for seed 2026 | All pass where they can run here; the T4-only parts use stand-ins (§4.3) |
| 7 | The execution mode still requires `Batch` | Yes. The code is byte-identical (`3cd0bc44…`); 11 tests; `Interactive`, unset and `batch` are refused in the notebook |
| 8 | Interactive optimizer execution refused | Yes. The gate and the training cell both refuse, and no run directory is created (§4.2) |
| 9 | The switch opens only when explicitly enabled | Yes. The generator writes `False` unless `--start-optimizer` is given; only exactly `True` opens; the delivered notebook equals the generator's output with the switch closed |

**Ruff on the notebooks.** It reports the same rule classes in both the seed 1337 notebook and the seed 2026
notebook:
- f-strings whose placeholder became a literal;
- import style;
- names the notebook defines by running its embedded code.

The seed 2026 notebook has three more names of the last kind: the provenance functions.

### 4.1 Tests (`pytest dist/kaggle_a1_seed_v3`: 150 passed)

| File | Tests | What they establish |
|---|---|---|
| `test_dataset_provenance.py` | 18 | Identification: top-level, nested and unpacked layouts; archive preferred over its unpacked copy; other inputs and other bundles recorded but not counted; 7 refused layouts. Confirmation needs the pinned archive and manifest; an unidentified dataset reports one root cause. Kaggle identifiers come only from whitelisted plain fields. No file content or environment value is recorded. The gate digest is canonical. |
| `test_verify_seed_run.py` | 36 | On a finished CPU fixture run, with provenance built by the notebook's own module: the six checks pass for top-level, nested and unpacked layouts. 11 run-identity faults each fail check 5 only. 18 provenance faults each fail check 6 only: unconfirmed, unidentified, with problems, another name, not among the inputs, another archive or `.sha256`, unpacked with an archive hash, another manifest or pin, a manifest naming another bundle, edited after the gate, a gate bound to other provenance or without a confirmed dataset, out of order, and a version 2 identity. Also: another bundle on disk, another expected name or archive pin, a run input missing from the manifest, and a missing identity. |
| `test_seed_notebook.py` | 85 | The switch is closed unless explicitly enabled. The embedded code is the pinned code, and the execution code is the probe-verified one. The pins are the approved A1 configuration for seed 2026. Untouched cells are byte-identical to version 2's, constants are equal, and the command is unchanged. The training condition's truth table: 80 combinations of switch, execution mode and confirmation, of which only `True` × `Batch` × confirmed allows training. The delivered notebook is the generator's closed output. |
| `test_execution_mode.py` | 11 | Unchanged from version 2 |

### 4.2 The notebook, cell by cell on this Mac (`run_nb_local.py`: 13 of 13 as required)

The training command never runs.
- **Copies used:** the closed scenario uses the delivered notebook. The others use a switch-open copy generated in
  scratch for testing only; it was deleted afterwards.
- **What is real:** the bundle is the real r2 archive. The dry run and the pre-training identity check run for real
  on it.
- **What is stood in:** the environment record and the preflight are T4-only, so a passing T4 record and seed 42's
  real preflight stand in for them.

| Scenario | Run type | Inputs | Outcome |
|---|---|---|---|
| closed | unset | the dataset | Provenance confirmed. Stops at the gate: "START_OPTIMIZER is False" |
| unset, interactive, lowercase | none, `"Interactive"`, `"batch"` | the dataset | Checks 0–4 pass and check 5 fails. The gate and the training cell refuse, and no run directory is created. |
| batch | `"Batch"` | the dataset, plus a Kaggle-style notebook copy | All six checks pass and the gates open. The stub receives exactly the documented command on GPU 0. The record is complete, and its digest is bound to the gate. `check_outputs.py` refuses the stand-in run, because its verification cannot pass, while every provenance line passes. |
| batch_nested | `"Batch"` | `datasets/someone/<slug>/` | Confirmed, `nested` |
| batch_unpacked | `"Batch"` | the bundle already unpacked | Confirmed, `unpacked`, no archive hash |
| batch_extra | `"Batch"` | the dataset, plus the r1 bundle (unpacked) and an unrelated dataset | Confirmed; all three inputs recorded |
| mismatch | `"Batch"` | the dataset; 94 questions expected | Stops at the identity check |
| no_dataset | `"Batch"` | an unrelated dataset only | Stops in the first cell: not identified |
| wrong_name | `"Batch"` | the r2 archive under `finretriever-train-v3-a1` | Stops in the first cell: outside any directory of the expected name |
| two_copies | `"Batch"` | the dataset and a copy in `…-r2-old` | Stops in the first cell |
| other_build | `"Batch"` | another archive under the expected name, with a consistent `.sha256` | Stops in the first cell: not the pinned build, and not unpacked |

**In every stopping scenario:**
- the record is complete, with `confirmed: false` and the reason;
- no gate is recorded, nothing is unpacked, and no run directory exists;
- the outputs are packaged;
- `check_outputs.py` reports `confirmed False` and passes nothing.

The training cell, which Run All would not reach, refuses as well. In every scenario, the planted secret appears in
no output.

### 4.3 The five pre-training gates for seed 2026

| # | Gate | Verified here | On the T4 |
|---|---|---|---|
| 0 | Training dataset (new) | Real archive: identified, confirmed, recorded before anything else | The same code, on Kaggle's mount |
| 1 | Python and packages | The pins equal seed 42's pip freeze (`5432fefb…`); this Mac's stack is correctly refused | Enforced in the environment cell |
| 2 | GPU and CUDA | Stand-in T4 record and seed 42's real preflight | Enforced: T4 7.5, CUDA 13.0, cuDNN 92700, and the preflight's `cuda_fp16`, `gradcache` and `gpu_memory` |
| 3 | Config and schedule | Real: config `7a4c4618…`, schedule `49b2b153…`, 55 steps, 6 of warmup, balanced 11 × 5, none dropped | The same, and already recorded by both T4 preflights |
| 4 | 678 candidates and 93 questions | Real, through the trainer's loaders: 678 (`0586c7ca…`), 93 (`40238cb0…`, IDs `b83abf1a…`), 10,137 passages | The same |
| 5 | Execution mode is Batch | Real code: only `"Batch"` opens | Enforced at the gate and in the training cell |

## 5. Seeds 42 and 1337: unchanged

These were checked before and after the work.

**Archives:**
- seed 42's four downloads still hash to `4a3b9d4a2ba9e54e95a2f0598d7005ad4342b149f90148d37c8c8c1701ebb8c3`;
- seed 1337's archive still hashes to `7f0317afea45065fb0712fea2fe9490ea899f97b3a18fc8e598c145047742048`.

The archives hold every checkpoint and resume state. An unchanged archive hash means every checkpoint is unchanged.

**Byte-identical (282 files):**
- the extracted records: `dist/kaggle_a1_seed42_results/` and `dist/kaggle_a1_seed1337_results/`;
- the selection records: seed 42 `19cca1cb…` and seed 1337 `165064f5…`;
- the notebook directories that trained them: `dist/kaggle_a1_seed42_notebook/` and `dist/kaggle_a1_seed_v2/`;
- the r2 bundle and its rehearsal notebook;
- the probe records;
- `configs/train/`.

**Data:** `train_candidates_v3` and `validation_pairs_v4` are byte-identical (7 files).

**Repository:** no tracked file changed except the documents listed in §10.

## 6. Limitations and open points

| # | Point | Effect | Handling |
|---|---|---|---|
| P1 | Kaggle's numeric identifiers come from an undocumented source: `metadata.kaggle.dataSources` in a saved version's `/kaggle/src/script.ipynb` | They may be absent, and are always absent in an interactive session | Recorded only, never deciding. The name, archive and manifest hashes are the binding identity. The seed 2026 record will show what Kaggle provides. The interactive check recorded them as absent, as expected: `no /kaggle/src/script.ipynb` (§7). |
| P2 | Identification relies on the mount directory carrying the dataset's slug, `/kaggle/input/finretriever-train-v3-a1-r2` | If Kaggle names it otherwise, the notebook stops in its first cell. That costs a session start, with no training. | Optional check before training, needing your OK: an **interactive** Run All of the closed notebook. A CPU session is enough. It records the provenance, then stops at the environment check. It trains nothing, and an interactive run is not a saved version. **Closed by that check (§7):** Kaggle mounted the dataset at `/kaggle/input/datasets/yousefkndesh/finretriever-train-v3-a1-r2/`. It was identified there (nested) and confirmed. |
| P3 | In a nested layout, `attached` gives only top-level names | Other inputs' slugs could be hidden behind a shared parent | The training dataset's full path is always recorded. In the interactive check, `attached` held one entry, `datasets`. Its listing digest equals the r2 bundle's plus its `.sha256`, and nothing else (§7). |
| P4 | A provenance failure packages the outputs | Version 2 stopped without outputs when no bundle was found | Intended: the record of the refusal can be reported |
| P5 | The `dist/` tools and their tests are gitignored | They are not in the repository suite; run them with `pytest dist/kaggle_a1_seed_v3` | As for versions 1 and 2. Putting them under version control is a separate decision. |
| P6 | The environment must equal seed 42's: Python 3.13.16 | Kaggle's current default image ran 3.13.18 in the probe | Run seed 2026 in the same pinned-environment Kaggle notebook as seed 1337. The interactive check's CPU session reported 3.13.18. That session stopped earlier, at the GPU check, so it says nothing about the T4 session's stack. The batch run must report 3.13.16, or the environment cell stops it before training (§9). |
| P7 | GPU training is not bit-reproducible | Any second execution could select differently | Train once, in a single saved version, as for seed 1337 |

## 7. The interactive dataset check on Kaggle (2026-11-11)

You ran the closed notebook (`f97cfc4a…`) with *Run All* in an interactive CPU session. I read the downloaded
`execution.json` (SHA-256 `2904cef3…`) as data, and it agrees with your reading.

| Field | Recorded |
|---|---|
| Execution | `95bee077-156a-4410-ad64-8972cb5e3cb4`, `interactive` (`KAGGLE_KERNEL_RUN_TYPE = "Interactive"`), started 2026-11-11 13:05:55 UTC |
| Seed and switch | 2026; `start_optimizer: false` |
| Pins | config `7a4c4618…`, schedule `49b2b153…`, archive `efc377d9…`, manifest `a6615850…`, execution-mode code `3cd0bc44…` |
| Training dataset | Identified and confirmed at `datasets/yousefkndesh/finretriever-train-v3-a1-r2`: `nested`, `unpacked`, with one copy of the bundle. Its manifest hashes to the pin, `a6615850…`, and names the bundle. No problems. |
| Attached inputs | One entry, `datasets`: 107 files, 56,908,199 bytes, listing digest `55628bac…` |
| Kaggle's identifiers | Absent: `no /kaggle/src/script.ipynb`. That is expected in an interactive session (P1). |
| Environment | CPU only: torch 2.11.0+cpu, Python 3.13.18 |
| Gate and training | None. The session stopped at the environment check, which needs a T4, and nothing was trained. |

**What it settles:**
- **The mount (P2).** Kaggle mounts the dataset under its slug, inside `datasets/<owner>/`, and the notebook
  identified it there.
- **Kaggle unpacks the archive.** It extracted `finretriever_train_v3_a1_r2.tar.gz` into a directory of the same
  name, so the bundle arrives `unpacked`.
  - The archive's own hash therefore cannot be checked on Kaggle, and is `null` by design.
  - The binding is the manifest: its hash must equal the pin.
  - The bundle-verification cell then checks every file against `SHA256SUMS` and against the manifest, and refuses
    any unlisted file. That cell is byte-identical to the one in seed 1337's notebook.
- **Nothing else is mounted.**
  - I rebuilt the layout from the local r2 archive and recomputed the listing digest with the pinned provenance
    module. The bundle unpacked, plus its `finretriever_train_v3_a1_r2.tar.gz.sha256`, gives exactly `55628bac…`:
    107 files, 56,908,199 bytes.
  - The bundle alone does not match, and neither does the bundle with `dataset-metadata.json`.
  - So every file name and size under `/kaggle/input` is the r2 dataset's. Identification on the rebuilt layout
    equals Kaggle's record, field for field.
- **Names only, no values.** The record lists Kaggle's environment-variable names, including three ending in
  `_TOKEN`, as seed 1337's identity did. No value is recorded.

**What it does not settle:** the T4 session's software stack. The CPU session reported Python 3.13.18, but it
stopped at the GPU check, before the stack comparison. See §9, step 2.

## 8. The official notebook and the final local checks

**The notebook:** `dist/kaggle_a1_seed_v3/finretriever_a1_train_seed2026.ipynb`, SHA-256
`c6ca9e5bfb670587c4a5a6deb1bf442939e4968346a9928ea688e7dff171ed1a`. The unchanged generator (`e95ba648…`) wrote it
with `--seed 2026 --start-optimizer`.
- **One line differs** from the closed notebook the interactive check ran: `START_OPTIMIZER = False` became `True`.
  The closed notebook still regenerates byte for byte (`f97cfc4a…`).
- **Every pinned and embedded file is unchanged:**
  - the execution-mode code `3cd0bc44…`, the provenance code `7e91ec33…` and the verification script `679ff007…`;
  - the config `7a4c4618…` and the schedule `49b2b153…`;
  - the archive `efc377d9…` and the manifest `a6615850…`.
- **The switch is one of three conditions.** In an interactive session, or with an unconfirmed dataset, the official
  notebook still refuses to train.

| Check | Result |
|---|---|
| Repository suite | 1,309 passed, 25 skipped, unchanged |
| Version 3 tests | **151 passed.** The test of the delivered notebook now requires the generator's `--start-optimizer` output, with the hash on record. A new test requires that it differs from the checked closed notebook in the switch line only. |
| Version 2 tests | 25 passed |
| Ruff | Clean: the repository and the version 3 sources |
| The notebook, cell by cell | **15 of 15 as required** (below) |
| Seeds 42 and 1337 | The five archives re-hashed: `4a3b9d4a…` ×4 and `7f0317af…`. The 282 protected files (records, selection records, notebooks, the r2 bundle, the probe records, `configs/train/`) and both frozen data sets (7 files) are byte-identical to the snapshot taken before any seed 2026 work. Among tracked files, only the documents in §10 differ. |

**The scenarios** (`run_nb_local.py`, now 15). The closed scenario runs the regenerated closed notebook; every other
scenario runs the official notebook itself.

| Scenario | Outcome |
|---|---|
| closed | Stops at the gate: "START_OPTIMIZER is False" |
| unset, interactive, lowercase | Refused at the gate and in the training cell; no run directory |
| batch, batch_nested, batch_unpacked, batch_extra | The gates open, and the stub receives exactly the documented command on GPU 0 |
| **batch_kaggle** (new) | Kaggle's exact mount, in a batch execution. The provenance equals Kaggle's record of the interactive check, field for field: the attached listing (`55628bac…`) and the identification. All six pre-training checks pass, the gates open, and the stub receives exactly the documented command. |
| **interactive_kaggle** (new) | The same mount, interactive: the dataset is confirmed and checks 0–4 pass, but training is refused |
| mismatch | Stops at the identity check |
| no_dataset, wrong_name, two_copies, other_build | Stop in the first cell |

The planted secret appears in no output, in any scenario.

## 9. Launch instructions: seed 2026, once

1. **Check the file on this Mac, then upload it unedited.** The switch is already open; no cell needs a change.

   ```bash
   shasum -a 256 dist/kaggle_a1_seed_v3/finretriever_a1_train_seed2026.ipynb
   # c6ca9e5bfb670587c4a5a6deb1bf442939e4968346a9928ea688e7dff171ed1a
   ```

2. **Use the same Kaggle notebook as seeds 42 and 1337,** so the environment stays pinned to seed 42's (Python
   3.13.16). Import the file into it, as you did for seed 1337. Then check the settings:
   - Accelerator: **GPU T4**;
   - Internet: **on**;
   - Environment: **pin to original environment**;
   - Input: **`finretriever-train-v3-a1-r2` only**, as in the interactive check. Don't attach a previous version's
     output or any other copy of the bundle: a second copy stops the notebook in its first cell.
3. **Run it once:** *Save Version* → **Save & Run All (Commit)**, not Quick Save. Expect about one GPU hour.
   - Don't run the cells interactively first. They would refuse to train, but the check is already done.
   - Make one saved version only. If it stops, report it rather than running again.
4. **Read the last line.** Success prints `SEED 2026 TRAINED AND VERIFIED. Download kaggle_a1_seed2026_outputs.zip and
   its .sha256; …`. Any other ending is a stop: `TRAINING DID NOT COMPLETE`, `TRAINING COMPLETED, BUT A CHECK FAILED`,
   or an earlier stop. Download what is there and report it.
5. **Download** `kaggle_a1_seed2026_outputs.zip` and `kaggle_a1_seed2026_outputs.zip.sha256` from the version's
   output. Keep the zip: it is the only copy of the 12 checkpoints.
6. **Check it locally:**

   ```bash
   python3 -I dist/kaggle_a1_seed_v3/check_outputs.py ~/Downloads/kaggle_a1_seed2026_outputs.zip \
       --extract dist/kaggle_a1_seed2026_results
   ```

   Expected:
   - the zip verifies, with no mismatched, unlisted or missing file;
   - the execution is `batch` (`"Batch"`) and the gate is open;
   - a T4 with Python 3.13.16;
   - the training dataset is at `datasets/yousefkndesh/finretriever-train-v3-a1-r2`, `nested`, `unpacked`, with
     manifest `a6615850…`;
   - all five provenance lines `PASS`;
   - the run verification passes all six checks;
   - exit code 0.

   Kaggle's numeric identifiers may now appear (a saved version); they are recorded only.
7. **Tell me.** I then do the independent audit, as for seed 1337. The three-seed decision
   (`scripts/07_decide_a1_vs_a0.py`) needs its own approval.

**If it stops:** every stop packages the outputs. Nothing trains before the gate.

| Where | Meaning | What to do |
|---|---|---|
| First code cell | The training dataset is not identified, or is not the verified build. The record says which. | Report it. Nothing trained. |
| Environment | Not a T4, or the stack is not seed 42's. Python 3.13.18 would mean the environment is not pinned. | Report it. Nothing trained. |
| Gate | Not a batch execution, or a pre-training check failed | Report it. Nothing trained. |
| Training | A non-finite gradient (fp16 overflow), or another failure. The checkpoints so far are kept. | Report it. Don't resume, rerun or change the scaler. |
| Verification | Trained, but a check failed | Report it. The outputs are packaged for review. |

## 10. Files

**In `dist/kaggle_a1_seed_v3/`:**

| File | SHA-256 |
|---|---|
| `finretriever_a1_train_seed2026.ipynb`, **official, switch open** | `c6ca9e5bfb670587c4a5a6deb1bf442939e4968346a9928ea688e7dff171ed1a` |
| The closed notebook the interactive check ran (not kept as a file; regenerates byte for byte) | `f97cfc4afc662ed6bfe55426545e2978f5a6210089c4fe5c65be5c25dfd0d955` |
| `make_seed_notebook.py` | `e95ba648cd994ac673a75a3112cdaf6769fd00603b82759cb43f7f7d8bf7e8dc` |
| `dataset_provenance.py` | `7e91ec3321cc63ac84f149bb7c37ad609fedece63f5114ba54e5ee6eb742b130` |
| `execution_mode.py` (unchanged from version 2) | `3cd0bc44ffd081f256c1dbe78c7f26d408894554976c1f7bc65b93f28eddff63` |
| `verify_seed_run.py` | `679ff0079cf18b9d7dcd2e00516a2ca76eef42d875806a9a4205c6c71b64e979` |
| `check_outputs.py` | `c0515fc65d87c6c142c0ccb58dd4eed1d67b16746a6faf96d2433a3244976f83` |
| `run_nb_local.py` (adds the two Kaggle-mount scenarios) | `e3c634b6f634193dc14e6e23fd2cad2cd2827d95e7be6813496ef3bb2d23de91` |
| `test_seed_notebook.py` (requires the official notebook) | `c000239f19ed3ac59d81e57e93cdb9cbc38c22b95b03f0cb6c4a60a496eb7818` |
| `test_dataset_provenance.py`, `test_verify_seed_run.py`, `test_execution_mode.py` | `17661b6a…`, `d1d171e5…`, `2fa9c612…` |

**Documents:**
- new: this report;
- updated: [kaggle_training.md](kaggle_training.md) §7. It says seed 1337 used version 2 and seed 2026 uses version 3,
  and gives the official notebook's hash and Kaggle's mount;
- updated: [seed1337_final_audit.md](seed1337_final_audit.md), a new "Official record" section (accepted; L2 and L5
  closed; the audit itself unchanged);
- updated: [training_a1_execution_safeguards.md](training_a1_execution_safeguards.md), a new §7 pointing here.
