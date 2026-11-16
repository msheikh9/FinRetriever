# A1 execution safeguards: report

**Date:** 2026-11-11.

**Status:**
- Both approved safeguards are implemented, notebook-only, and tested locally.
- **Verified on Kaggle on 2026-11-10:** the probe ran as an interactive Run All and as a saved version, and
  `check_probe.py` reports `SAFEGUARD VERIFIED ON KAGGLE` (§2a).
- Nothing was trained. The seed 42 result and artifacts are unchanged. No hyperparameter, clipping or precision
  setting changed. `train_candidates_v3` and `validation_pairs_v4` are untouched. No API request, no commit.

**Where everything is:** `dist/kaggle_a1_seed_v2/` (gitignored). The seed 42 notebook directory
(`dist/kaggle_a1_seed42_notebook/`) is kept as the record of what ran, unchanged.

## 1. A. Execution-mode protection

**What Kaggle exposes.**
- **The variable:** community posts on Kaggle's own forums report that `KAGGLE_KERNEL_RUN_TYPE` is `"Interactive"` in
  an editor session and `"Batch"` in a saved version (Save & Run All), and absent off Kaggle.
- **The evidence is thin:** I found no official Kaggle documentation of it. No earlier output of this project
  recorded environment variables, so nothing here confirms it either.
- **The consequence:** the safeguard is built to fail closed, and §2 verifies it on Kaggle before it is relied on.

**The rule.**
- **Training runs only if both gates hold:**
  1. `START_OPTIMIZER = True`, your approval;
  2. `KAGGLE_KERNEL_RUN_TYPE` equals `"Batch"` exactly.
- **Everything else refuses:** `"Interactive"`, a missing variable, `"batch"`, `"Batch "` or any other value.
- **Where it is checked:** at the gate, after the preflight and the dry run, and again inside the training cell.
- **An interactive Run All** still verifies everything up to the gate, then stops without training.

**The code.**
- **One source:** the mode is decided by `execution_mode.py`, SHA-256 `3cd0bc44…`. Both the training notebooks and
  the probe embed it, pinned by hash.
- **Why that matters:** a probe result therefore verifies exactly the code that gates training.
- **Second signals, recorded only:** it also records the names (never the values) of the `KAGGLE_` variables, the
  files under `/kaggle/src`, and the process ancestry's executable names. If the probe shows one of these
  discriminates reliably, it can become a required second condition, but only after your approval.

## 2. Verifying it on Kaggle (needs you; no GPU)

`kaggle_execution_probe.ipynb` (`04b4bbed…`) trains nothing and needs no GPU, internet or dataset.

1. **Import** it on Kaggle with *Accelerator: None*.
2. **Interactive run:** *Run All* in the editor, then download `execution_probe_*.json` from the session's *Output*
   panel.
3. **Saved version:** *Save Version* → *Save & Run All (Commit)*, then download `execution_probe_*.json` from the
   version's *Output*.
4. **Check locally:**
   `python3 -I dist/kaggle_a1_seed_v2/check_probe.py --interactive <file from step 2> --batch <file from step 3>`

**It passes (`SAFEGUARD VERIFIED ON KAGGLE`) only if:**
- both records were made on Kaggle;
- both ran the pinned code, in two different executions;
- the interactive record would be refused;
- the saved version's record would be allowed.

A locally simulated record cannot pass; the "made on Kaggle" check refuses it.

**If it fails:** do not train. Report the two records. The indicator would then be replaced by one the probe shows to
work, after your approval.

## 2a. The Kaggle verification (2026-11-10): passed

**The records,** saved unchanged as you provided them, in `logs/kaggle_execution_probe/`:

| Execution | File | SHA-256 | Probe ID | Recorded (UTC) | `KAGGLE_KERNEL_RUN_TYPE` |
|---|---|---|---|---|---|
| Interactive (Run All) | `execution_probe_interactive_1de2c0a0.json` | `4e1f1fd73842ee1c98948d2056f87b3960df474ceaf42051fd47f5c5739287c3` | `1de2c0a0-7b7c-…` | 20:39:28 | `"Interactive"` |
| Saved version (Save & Run All) | `execution_probe_batch_8e34c7d6.json` | `7b0fd0f8f92bbca9ed60c38e01da2870d44ac9277f4327a0cf6a524b550dd045` | `8e34c7d6-c5f0-…` | 20:41:14 | `"Batch"` |

Both ran execution code `3cd0bc44ffd081f256c1dbe78c7f26d408894554976c1f7bc65b93f28eddff63`. That is the SHA-256 of
`dist/kaggle_a1_seed_v2/execution_mode.py`, the code embedded in the seed 1337 notebook.

**The checker's output** (`python3 -I dist/kaggle_a1_seed_v2/check_probe.py --interactive … --batch …`, exit 0):

```
interactive: KAGGLE_KERNEL_RUN_TYPE = 'Interactive' -> interactive (recorded 2026-11-10T20:39:28+00:00)
batch      : KAGGLE_KERNEL_RUN_TYPE = 'Batch' -> batch (recorded 2026-11-10T20:41:14+00:00)
other indicators that differ: ['kaggle_src_files', 'kaggle_variable_names', 'process_ancestry']
PASS  both records were made on Kaggle
PASS  both ran the pinned execution-mode code
PASS  two different executions
PASS  the interactive run is refused
PASS  the saved version is allowed
SAFEGUARD VERIFIED ON KAGGLE
```

**Your five criteria:**

| Criterion | Result |
|---|---|
| Both records are genuine Kaggle executions | yes: Linux process records, and 11–12 Kaggle variables besides the run type |
| Same execution-code SHA-256 | yes: `3cd0bc44…` in both, equal to the file the notebooks embed |
| Interactive execution refused | yes: `"Interactive"` → `interactive`, training not allowed |
| Batch execution allowed | yes: `"Batch"` → `batch`, training allowed |
| Distinct probe IDs | yes: `1de2c0a0…` and `8e34c7d6…` |

**The other indicators agree with the variable, in both executions.** They were recorded only and never decide:

| Indicator | Interactive | Saved version |
|---|---|---|
| Parent process | `jupyter-server` (the kernel has `ipykernel`) | a `python3` process running `papermill` |
| `/kaggle/src` | absent | `script.ipynb` |
| Variable names | as the saved version's, plus `KAGGLE_DISABLE_GOOGLE_GENERATIVE_AI_INTEGRATION` | 12 names |

**About these signals.**
- **What they show:** two independent signals, the papermill runner and the copied `script.ipynb`, confirm the
  variable here.
- **Why they don't gate:** making them a required second condition would change the pinned code, and the probe
  would have to run again. That is your call; I don't recommend it now. The verified variable is enough, and a
  second condition could refuse a legitimate run if Kaggle changes its runner.

**Hygiene.**
- Only the run-type value is recorded. All the other entries are variable names, including
  `KAGGLE_USER_SECRETS_TOKEN` and `KAGGLE_DATA_PROXY_TOKEN`. No secret value appears in either record.
- Records cannot prove their own origin: these were saved from your paste. If you still have the downloaded
  files, `shasum -a 256` should give the hashes above.

**One finding for seed 1337: Python version.**
- **What the probe showed:** its sessions report Python **3.13.18**.
- **What was rehearsed:** the rehearsal and seed 42 ran **3.13.16**.
- **Why it may not matter:** the probe ran on Kaggle's CPU image (Accelerator: None), which can differ from the GPU
  image.
- **Why it might:** it may also mean Kaggle has updated its images.
- **What the seed 1337 notebook does:** it requires the rehearsal's exact stack (Python 3.13.16, torch 2.11.0+cu130,
  CUDA 13.0, cuDNN 92700, transformers 5.3.0, tokenizers 0.22.2, T4). On any other, it stops in its environment cell,
  before the preflight and long before the gate. Nothing trains.
- **How to avoid a stop:** run seed 1337 in a Kaggle notebook pinned to the original environment, such as the one
  that trained seed 42.
- **If it stops anyway:** a newer image means a decision, as in S4. Re-rehearsing on the new stack would break
  comparability with seed 42, because the decision rule requires one stack for every seed.

## 3. B. Run identity

**What is recorded.** The notebook writes `execution.json` with:
- the execution type and the raw `KAGGLE_KERNEL_RUN_TYPE`;
- a random execution ID;
- the notebook start time and the training start and end times (UTC);
- the GPU model and compute capability;
- CUDA, cuDNN and the **driver version**;
- PyTorch, transformers, tokenizers and Python;
- the config hash, the schedule hash, the seed, and the bundle and manifest hashes;
- the gate decision and the training exit code;
- the SHA-256 of the run's `run.json` and `selection.json`.

**Where.** The file goes into the outputs and, after training, **into the run directory**, so it travels with the
checkpoints.

**Why not inside `run.json` or `selection.json`.**
- `run.json` must stay byte-identical across executions; that is how `--resume` recognises its run.
- `selection.json` is written by the trainer, whose code must stay the rehearsed code that trained seed 42.
- Instead, `execution.json` binds both by hash.

**The verification gains a fifth check, `execution_identity`.** It requires:
- a batch execution, with the gate open and exit 0;
- times in order;
- seed, config and schedule equal to `run.json`'s;
- both run-file hashes matching the files on disk;
- the recorded GPU, CUDA, cuDNN and software equal to the trainer's own record in `selection.json`;
- a driver version present.

## 4. C. Documentation

| Document | Change |
|---|---|
| [seed42_final_audit.md](seed42_final_audit.md) | New "Official record" section: step 55 official, with digest, archive and execution; the earlier step-35 execution disclosed as non-authoritative; how to preserve its console output (not yet available here); GPU non-reproducibility; gradient clipping, 55 of 55 steps clipped, norms 2.11–16.90 against a limit of 1.0 |
| [kaggle_training.md](kaggle_training.md) §7 | Each seed is trained exactly once, in a saved version; GPU executions are not bit-reproducible; the version 2 notebooks and `execution.json` |
| [training_a1_seed42_plan.md](training_a1_seed42_plan.md) | An update note: executed twice, the official result, and the plan's gap; the plan itself kept as written |

## 5. Tests and checks

| Check | Result |
|---|---|
| `test_execution_mode.py` | 11 pass: only the exact `"Batch"` is batch; 9 other values, including a missing one, are unknown; a planted secret's value is never recorded, only names and booleans |
| `test_verify_seed_run.py` | 14 pass on finished fixture runs (`TinyEncoder`, CPU): the earlier checks, plus 11 ways the identity can be wrong (interactive, unidentified, `"batch"`, gate closed, another seed or schedule, another execution's selection record, a failed command, times out of order, another environment, no driver), plus a run without an identity. Each fails only the identity check. |
| The seed 1337 notebook, cell by cell on this Mac | **closed:** stops at the gate; **unset**, **interactive**, **lowercase**: refused at the gate and in the training cell, no run directory; **batch:** the gates open, a stub replaces training and receives exactly the documented command on GPU 0, `execution.json` has every field and its copy in the run directory is identical, the post-run tools fail cleanly on the stand-in, packaging and `check_outputs.py` pass. The seed 1337 dry run matched its pinned schedule `795095dc…`. |
| The probe, on this Mac | runs; no secret value in its records; `check_probe.py` refuses local records, an interactive run reporting `"Batch"` and a saved version without the variable; Kaggle-like records pass |
| The repository suite | 1,309 passed, 25 skipped (unchanged: no repository code changed) |
| Ruff | clean |

**Seed 42 is unchanged.**
- **The archive:** all four downloads still hash to `4a3b9d4a…`.
- **The extraction:** its 119 files are unchanged and match the archive's `OUTPUT_SHA256SUMS`.
- **`selection.json`:** `19cca1cb…`, as listed in that file.
- **The notebook directory that ran:** its 6 files are unchanged.
- **The repository:** only the three documents above changed, plus this report.

**Re-checked after the Kaggle probe:**
- the 25 version 2 tests pass;
- the repository suite: 1,309 passed, 25 skipped;
- Ruff is clean;
- the four seed 42 zips still hash to `4a3b9d4a…`, and the extraction and the seed 42 notebook directory are
  unchanged;
- the only new files are the two probe records in `logs/kaggle_execution_probe/`.

## 6. Seed 1337: approved 2026-11-11, notebook ready to run

**Approval:** seed 1337, the approved A1 configuration, `START_OPTIMIZER = True`, Save & Run All only, the seed 42 T4
stack, and the frozen data. Not approved: seed 2026, any hyperparameter change, any dataset change, any commit.

**The notebook:** `dist/kaggle_a1_seed_v2/finretriever_a1_train_seed1337.ipynb`, SHA-256
`19bbf437e414793c54cfdc3bd53531e5cb67e403c72ccbe2354c01c0fe38a510`.
- **Gate:** `START_OPTIMIZER = True`.
- **What it embeds:** the probe-verified execution code (`3cd0bc44…`) and the tested verification script
  (`df3e4038…`).
- **Generator:** `make_seed_notebook.py`, `a181b125…`.

**Your five pre-training checks, all enforced before the optimizer can start.** Any failure halts the run with the
outputs packaged, so the discrepancy can be reported. Nothing is trained.

| # | Check | Where | Required |
|---|---|---|---|
| 1 | Python and package versions | install and environment cells | Python 3.13.16 and the rehearsal's identity keys, plus seed 42's versions of 13 training packages (torch 2.11.0+cu130, numpy 2.1.3, scipy 1.16.3, triton 3.6.0, transformers 5.3.0, tokenizers 0.22.2, safetensors 0.7.0, huggingface-hub 1.7.1, sentence-transformers 5.2.3, pydantic 2.14.1, PyYAML 6.0.3, regex 2025.11.3, accelerate 1.15.0). Two things are recorded but don't decide: whether the whole pip freeze is identical to seed 42's (`5432fefb…`), and whether the driver equals seed 42's (580.178.04). |
| 2 | GPU and CUDA | environment cell, then the preflight | Tesla T4, compute 7.5, CUDA 13.0, cuDNN 92700; the preflight's `cuda_fp16`, `gradcache` and `gpu_memory` pass, with `READY FOR TRAINING` |
| 3 | Config and schedule hashes | dry run and the pre-training identity check | the config file and the trainer's loader both give `7a4c4618…`; the full schedule is `795095dc8898db3193f9334e1007b7d449c5e6155552dc535b28a60b5d87bb45`, with 55 steps, 6 warmup, balanced 11 × 5, none dropped and at least 60 negatives |
| 4 | 678 candidates and 93 validation questions | the preflight's re-verification, and the identity check through the trainer's own loaders | `train_candidates_v3`: 678, manifest `0586c7ca…`; `validation_pairs_v4`: 93, manifest `40238cb0…`, question-ID hash `b83abf1a…`; 10,137 corpus passages. These equal seed 42's `run.json`. |
| 5 | Execution mode is Batch | the gate, then the training cell again | `KAGGLE_KERNEL_RUN_TYPE == "Batch"` exactly; the training cell also requires that the gate passed in this session |

**The record:** checks 1–5 are written to `pretraining_checks.json`.

**Tested on this Mac, with the training command never run.** Six execution scenarios ran through the real cells.
Stand-ins were used for the two T4-only records: a passing T4 environment, and seed 42's real preflight report. Each
scenario was refused where it should be:
- the approval switch off;
- no variable;
- `"Interactive"`;
- `"batch"`;
- a dataset mismatch (93 against 94 questions), refused even in batch with the switch on.

In batch, the stub received exactly the documented seed 1337 command on GPU 0. The real identity check on the real
bundle found the pinned schedule, 678 pairs, 93 questions and 10,137 passages.

**Packages and other tests.**
- The pinned package versions equal seed 42's own pip freeze.
- This Mac's stack is correctly refused (torch, numpy, scipy, triton, pydantic, regex and accelerate differ).
- The 25 version 2 tests pass, and Ruff is clean.

**After training (the notebook, then locally):**
- all 12 checkpoints re-hashed against `selection.json`;
- the K1 selection re-derived, and recomputed independently;
- the optimizer, scheduler, scaler, random and selection state restored on the T4;
- the selected and last checkpoints re-evaluated;
- step 0 against the rehearsal;
- the run identity;
- packaging with every file hashed;
- locally: `check_outputs.py`, then the seed 1337 report.

## 6a. Before the approval (as first prepared; superseded by §6)

**The notebook:** `dist/kaggle_a1_seed_v2/finretriever_a1_train_seed1337.ipynb`, SHA-256 `7a8d5fb2…`; superseded
by `19bbf437…` (§6).
- It is generated with `START_OPTIMIZER = False`.
- It uses the same bundle, config, data and hyperparameters as seed 42, and the stack must equal the rehearsal's.
- Its schedule is `795095dc8898db3193f9334e1007b7d449c5e6155552dc535b28a60b5d87bb45`.

**The order:**
1. ~~Run the probe twice and check it (§2).~~ Done; passed (§2a).
2. Approve seed 1337.
3. Set `START_OPTIMIZER = True`, and train **only** with *Save Version → Save & Run All*.
4. Download and check:
   `python3 -I dist/kaggle_a1_seed_v2/check_outputs.py <zip>`.

## 7. Version 3 for seed 2026: dataset provenance (approved 2026-11-11)

**Seed 1337 is accepted** ([seed1337_final_audit.md](seed1337_final_audit.md)). Its L5 finding, that the attached
Kaggle datasets were not recorded, is addressed for seed 2026 by the version 3 notebook in `dist/kaggle_a1_seed_v3/`.

**What is unchanged:**
- the execution-mode code, `3cd0bc44…`, byte for byte, so the Kaggle probe's verification (§2a) still applies;
- the `run.json` and `selection.json` bindings;
- the five pre-training checks;
- the command and hyperparameters.

**What version 3 adds:** the dataset provenance in `execution.json`, a third training condition (a confirmed training
dataset), and a sixth verification check, `dataset_provenance`.

The design, tests and seed 2026 readiness are in
[training_a1_seed2026_readiness.md](training_a1_seed2026_readiness.md). The version 2 files stay as the record of
what trained seed 1337.
