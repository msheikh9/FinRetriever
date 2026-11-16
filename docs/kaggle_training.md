# Fine-tuning runs on a Kaggle T4

This guide runs a validated fine-tuning config on a Kaggle T4 from a portable bundle: one seed per run, GradCache,
and checkpoints chosen only by the fixed validation rule. The method is in
[training_full_v1.md](training_full_v1.md). The A0-versus-A1 experiment's config is
`configs/train/biencoder_bge_base_ft_v3_a1.yaml`, and its rule, rehearsal and batching are in
[training_a1_implementation_report.md](training_a1_implementation_report.md).

**Each stage needs its own explicit approval.**
- Building the bundle, the preflight (step 5) and the GPU rehearsal (step 6) train nothing; the rehearsal takes no
  optimizer step.
- Only the command in step 7 trains, and only with `--confirm-training`.

**One GPU type for every seed.** The decision rule compares the seeds' step 0 and refuses seeds run on different
hardware or software. Run every seed on a T4, from the same bundle and requirements.

**What a run can and cannot show:**
- The training and validation questions are synthetic and were accepted by an AI review. They are not
  human-validated and not gold labels.
- The validation set chooses among one run's checkpoints and nothing else.
- No evaluation data is in the bundle, and the trainer computes no development or evaluation metric.
- No retrieval improvement is claimed from a run.

## 1. Build the bundle (locally, from the repository root)

```bash
python3 scripts/smoke_bundle.py build --config configs/train/biencoder_bge_base_ft_v3_a1.yaml \
    --name finretriever_train_v3_a1
```

This writes `dist/kaggle/finretriever_train_v3_a1.tar.gz`, its `.sha256` and a `dataset-metadata.json`
template. The bundle (kind `finretriever_training_bundle`) holds only what the run reads:
- the code;
- the config;
- the frozen corpus;
- the `bge_base_ft_v3` preparation;
- the training candidates with their source pairs, generation records and decision files;
- the validation set (`validation_sets/validation_pairs_v4`) with the records, plan, stop record and decision
  files its manifest binds.

The build refuses evaluation data, raw filings, caches, models, results and anything that looks like a credential.

## 2. Upload as a private Kaggle dataset

Upload the archive and its `.sha256` file, and keep the dataset **Private**. Alternatively, set your username in
`dataset-metadata.json` and run `kaggle datasets create -p dist/kaggle`.

## 3. Notebook settings and setup

- **Accelerator:** GPU T4. The run uses one GPU.
- **Internet:** on, for pip and the base-model download.
- **Input:** the private dataset.

Copy the bundle to `/kaggle/working`, check its archive SHA-256 and `SHA256SUMS`, and install
`requirements-kaggle.txt` with the image's torch pinned (`-c /tmp/torch.txt`). This is as in
[kaggle_smoke_test.md](kaggle_smoke_test.md) steps 4 to 6, with the new bundle name.

## 4. Check CUDA and free memory

```
!nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

## 5. Preflight (trains nothing)

```
!python scripts/smoke_preflight.py --config configs/train/biencoder_bge_base_ft_v3_a1.yaml
```

Besides the smoke checks it verifies:
- **validation_set:** the validation set re-verifies, is the bundled one, and shares no passage or question with
  the pairs trained on;
- **schedule:** every seed's schedule matches the bundle (balanced batching: 55 steps, nothing dropped);
- **gradcache:** on the loaded base model, with eager attention and dropout on, GradCache's gradients equal those
  of the same forward passes without caching, on a few real pairs; one cached step over a full mini-batch is
  measured on the GPU;
- **gpu_memory:** the free memory covers the measured step peak plus AdamW's state, with a 15% margin.

It ends with `READY FOR TRAINING` only if every check passes.

## 6. GPU rehearsal (no optimizer step; needs its own approval)

```
!python scripts/gpu_rehearsal.py --config configs/train/biencoder_bge_base_ft_v3_a1.yaml --out rehearsal_outputs \
    2>&1 | tee rehearsal_console.log
```

Ten checks with the trainer's own code (`finretriever.training.gpu_rehearsal`):
1. the environment;
2. real BGE with dropout and eager attention, replayed exactly;
3. attention dropout alone, replayed;
4. GradCache equivalence in fp32;
5. GradCache equivalence in fp16, at the GradScaler's real initial scale and at 1,024;
6. every gradient finite at that initial scale on each seed's first batch and on the 64 longest pairs, and fp16
   against fp32 gradients;
7. the measured peak memory of a full 64-pair step with AdamW's state;
8. the full-corpus validation pass, twice;
9. checkpoint save, re-hash, reload, resume and the sentence-transformers load;
10. no optimizer step and unchanged weights.

**What it guarantees:**
- **No training.** Every optimizer's and the GradScaler's `step` raise for the whole run.
- **An fp16 overflow stops it.** An overflow at the initial scale stops the rehearsal with the overflow reported,
  and smaller scales are probed for information only. No scaler setting is changed; a fixed scale needs a decision.

It ends with `REHEARSAL PASSED` (exit 0) only if every check passes. Download `rehearsal_outputs/`, which holds the
report, the step-0 validation result kept for the identity check across seeds, and check 9's checkpoint, together
with `rehearsal_console.log`. **Stop for review.**

## 7. Train one seed per session (after training approval)

```
%%bash
set -o pipefail
cd /kaggle/working/finretriever_train_v3_a1
mkdir -p smoke_outputs
SEED=42   # then 1337, then 2026: one per session, on the same GPU type; seed 42 is reviewed first
CUDA_VISIBLE_DEVICES=0 python scripts/05_train_biencoder.py \
    --config configs/train/biencoder_bge_base_ft_v3_a1.yaml --seed $SEED --confirm-training \
    2>&1 | tee smoke_outputs/train_seed$SEED.log
```

**Each seed is trained exactly once, in a saved version (approved 2026-11-10).**
- **Why:** GPU training is **not bit-reproducible across separate executions**. The trainer seeds Python, torch and
  CUDA, but does not enable deterministic algorithms, so two executions with the same seed, code and GPU type can
  end with different weights, and can select different checkpoints.
- **What happened with seed 42:** it was trained twice. The two executions selected steps 35 and 55. Step 55, from
  the saved version, is the official result ([seed42_final_audit.md](seed42_final_audit.md)).
- **Seed 1337 used the version 2 notebook** (`dist/kaggle_a1_seed_v2/`;
  [training_a1_execution_safeguards.md](training_a1_execution_safeguards.md)).
  - **It trains only in a positively identified batch execution,** that is, Save & Run All with
    `KAGGLE_KERNEL_RUN_TYPE == "Batch"` exactly. Any other value, or none, refuses (fail closed).
  - **It writes a run identity, `execution.json`,** into the outputs and the run directory. It records the
    execution type, the start times, the GPU, CUDA, the driver, PyTorch, the config and schedule hashes, the seed
    and the hashes of `run.json` and `selection.json`.
  - The command and hyperparameters are unchanged.
- **Seed 2026 uses the version 3 notebook** (`dist/kaggle_a1_seed_v3/`, approved 2026-11-11;
  [training_a1_seed2026_readiness.md](training_a1_seed2026_readiness.md)). It keeps all of version 2 and adds the
  **dataset provenance**:
  - **What is recorded:** `execution.json` gains `datasets`, written in the first code cell, before the bundle is
    read. It holds every attached input by name, file count, size and listing digest, and Kaggle's own dataset
    identifiers where a saved version's notebook copy carries them. It also holds the training dataset: its
    directory, the bundle's form, and the archive and manifest hashes.
  - **A third training condition:** the bundle must sit in exactly one input directory named
    `finretriever-train-v3-a1-r2` and verify against the pinned archive and manifest. Anything else stops the
    notebook before the optimizer can start (fail closed).
  - **What binds it:** the gate decision records the provenance's digest. The run verification gains a sixth
    check, `dataset_provenance`: the recorded manifest must be the bundle the verification runs from, and that
    manifest must list every input `run.json` records.
  - **What is never recorded:** file contents, environment-variable values and credentials.
  - The command, hyperparameters, environment pins and execution-mode code (`3cd0bc44…`) are unchanged.
  - **The official notebook** has the switch open, SHA-256 `c6ca9e5b…`. An interactive check of the closed notebook,
    on 2026-11-11, showed how Kaggle mounts the dataset: at
    `/kaggle/input/datasets/<owner>/finretriever-train-v3-a1-r2/`, with the archive already unpacked. The bundle is
    then bound through its pinned manifest, and every file is checked against it.

Each seed writes `models/<output_dir>/<name>-seed<SEED>/`:
- `run.json`;
- `log.jsonl`, one line per step, with the loss scale it used;
- `validation_log.jsonl`, one line per evaluation with the checkpoint's digest;
- `checkpoint-step-NNNNNN/` at step 0 (the base weights) and every evaluated step, with `metadata.json` holding
  that step's validation metrics;
- `selection.json`, written once when the run reaches its last step. It lists every evaluation with its
  checkpoint's file hashes, the rule, the selected checkpoint, and the GPU and software environment.

**Early stopping fixes the selection; training goes on.** After 3 evaluations without improvement, no later
checkpoint is selectable. Training and evaluation still continue to step 55, recorded as "after the rule's stop,
never selectable", because the decision rule needs every planned evaluation.

**A non-finite gradient stops the run.** An fp16 overflow stops the run at that step, before the optimizer step. The
GradScaler is never left to skip the step and lower its scale, and the checkpoints so far are kept. Report it; a
change of scale is a new decision and a new run.

**Interrupted sessions.**
- **Resuming:** attach the previous session's output as an input, copy the run directory into
  `/kaggle/working/.../models/`, and rerun the same command with `--resume`. The newest checkpoint holds the
  optimizer, scheduler, scaler, random and selection state, so the selection and the after-stop evaluations resume
  where they were.
- **Exactness:** a resumed run repeats the uninterrupted one exactly on CPU. On CUDA it is close, not bit-identical,
  as are two separate executions of the same seed.
- **Finished runs:** a run with `selection.json` has ended, and `--resume` returns without training.
- **Never overwritten:** existing checkpoints and records are never written over, and a different configuration
  in the same directory is refused.

## 8. Collect the results (trains nothing)

```
!python scripts/smoke_results.py --run-dir models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed42 \
    --console-log smoke_outputs/train_seed42.log --out-dir smoke_outputs/seed42
```

This reports:
- the steps and the losses;
- the measured GPU memory;
- every validation evaluation, with those after the rule's stop marked;
- the selected checkpoint, after re-reading `selection.json` and re-hashing every evaluated checkpoint against it;
- the environment;
- checkpoint reloads;
- console lines that need review.

Download each run directory (or at least its `selection.json`, the selected checkpoint and `run.json`).

## 9. The decision (locally, after all three seeds; reads only the selection records)

```bash
python scripts/07_decide_a1_vs_a0.py \
    --selection models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed42/selection.json \
    --selection models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed1337/selection.json \
    --selection models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed2026/selection.json \
    --out results/decisions/bge_base_ft_v3_a1_vs_a0.json
```

**The rule:** a selection-aware sign-flip test with 10,000 flips, PCG64 seed 20261109. A1 beats A0 on the validation
set only if p ≤ 0.05 and every seed selected a checkpoint after step 0. The record is written once.

**Then the benchmark.** Evaluate the selected checkpoints locally as final models (`role: final` with
`selection_record` in `configs/eval/benchmark.yaml`; see [training_full_v1.md](training_full_v1.md) §6). Every
seed must be listed. Development results are diagnostics and never decide.
