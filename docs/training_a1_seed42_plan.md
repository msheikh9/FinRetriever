# A1 seed 42: execution plan, verified configuration and estimates

> **Update, 2026-11-10: executed.** Seed 42 was trained twice, in two separate Kaggle executions. The official result
> is `checkpoint-step-000055` of the saved version (Save & Run All); the earlier execution, which selected step 35,
> is disclosed and non-authoritative ([seed42_final_audit.md](seed42_final_audit.md)).
>
> **A gap in this plan:**
> - nothing stopped the notebook from training in an interactive Run All with the gate open, followed by a saved
>   version, which trains the seed twice. The optional check in step 3 ran with the gate closed and could not
>   train;
> - the plan did not say that two GPU executions of one seed need not be identical.
>
> The version 2 notebooks for the later seeds train only in a saved version
> ([training_a1_execution_safeguards.md](training_a1_execution_safeguards.md)). This plan is kept as written.

**Status, 2026-11-10 (as prepared):** prepared and verified. **Nothing has been trained.** The notebook's optimizer
gate is closed (`START_OPTIMIZER = False`); it opens only after your final confirmation.

**Scope (approved 2026-11-10):**
- seed 42 only, on the rehearsed Kaggle T4 stack;
- then a detailed seed 42 report;
- stop before seeds 1337 and 2026.

**Out of scope:** A2, hard-negative mining, any change of learning rate, schedule, batch size, GradCache or GradScaler,
and any commit or push.

## 1. What runs, from what

| | |
|---|---|
| Bundle | **the r2 bundle, unchanged**: `finretriever_train_v3_a1_r2.tar.gz`, SHA-256 `efc377d93f61b3bb5fbfb6706ecd2b08507d0ed5d30a13f597ba3ca9517e6f65`, manifest `a6615850…`. It is already uploaded as the private dataset `finretriever-train-v3-a1-r2`. |
| Why not a rebuild | It passed the T4 preflight and all ten rehearsal checks. All 104 bundled files are byte-identical to the repository today (re-checked on a fresh extraction). It ships `05_train_biencoder.py` and `smoke_results.py`. |
| Command | the documented one ([kaggle_training.md](kaggle_training.md) §7): `CUDA_VISIBLE_DEVICES=0 python scripts/05_train_biencoder.py --config configs/train/biencoder_bge_base_ft_v3_a1.yaml --seed 42 --confirm-training` |
| Notebook | `dist/kaggle_a1_seed42_notebook/finretriever_a1_train_seed42.ipynb` (SHA-256 `4447559f…`), with `check_outputs.py` for the download. The generator, the verification script and its tests are kept beside it. |

## 2. The verified run configuration

These were verified by a dry run from the extracted bundle. The T4 preflight already verified them for all three
seeds.

| Setting | Value |
|---|---|
| Config | `configs/train/biencoder_bge_base_ft_v3_a1.yaml`, SHA-256 `7a4c4618f9e00ab3ac44cc87aacc5be926753b7b71acd9c9bec47941cc77d4cf` (the rehearsal's) |
| Base model | `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; CLS pooling; the evaluation's query prefix |
| Training data | `train_candidates_v3`: **678** candidates (612 P1, 51 pilot_v3, 15 pilot_v2) |
| Validation | `validation_pairs_v4`: **93** synthetic questions, AI-reviewed and not gold; checkpoint selection only, isolated from the 678 |
| Batches | balanced: 11 per epoch of 61–62 pairs; at least 60 in-batch negatives per query, 60.64 on average; 0 dropped; every batch re-checked against the false-negative guard before its forward pass |
| Schedule | 5 epochs = **55 optimizer steps**; schedule SHA-256 `44b0ed7cd298b859fd1fc088b1ab8bdaf5d5d69a47436a1a078e8ac57b2bd306` |
| Optimizer | AdamW, lr 2.0e-5, weight decay 0.01, gradient clipping at 1.0 |
| Learning rate | warmup over steps 1–6 (3.3e-6 to 2.0e-5), then linear decay: 2.0e-5 at step 7, 1.3e-5 at step 22, 4.5e-6 at step 44, 4.1e-7 at step 55 |
| Loss | in-batch InfoNCE, temperature 0.02 |
| GradCache | mini-batch 16; each mini-batch replayed with its saved random state and checked against its cached pass |
| Precision | fp16 autocast. The GradScaler keeps torch's defaults, initial scale 65,536, unchanged. A non-finite gradient stops the run before the optimizer step. |
| Lengths | queries ≤ 128 tokens (longest 58); passages truncated at 512, each keeping its visible text (605 of the 678 positive pairs reach 512) |
| Environment | must equal the rehearsal's: Tesla T4, compute 7.5, CUDA 13.0, cuDNN 92700, Python 3.13.16, torch 2.11.0+cu130, transformers 5.3.0, tokenizers 0.22.2. The notebook stops before training otherwise. |

## 3. Checkpoint and evaluation schedule

| Steps | What happens |
|---|---|
| 0 | the base model (A0) is evaluated on the validation set and saved as `checkpoint-step-000000`; a selection candidate |
| 5, 10, …, 50 | validation, then a checkpoint; a candidate unless the rule has already stopped |
| 55, the last step | validation, then `checkpoint-step-000055`; then `selection.json` is written once, holding every evaluation, every checkpoint's file hashes, the rule, the selected checkpoint and the environment |
| Epoch boundaries | after steps 11, 22, 33, 44, 55 (no checkpoint of their own) |

**12 evaluations and 12 checkpoints.**

**The rule.**
- Whole-corpus MRR@10 decides, over 10,137 passages; ties go to the earlier step, compared to 12 decimals.
- Selection stops after 3 candidate evaluations in a row without improvement.
- Training and evaluation still continue to step 55. Those later evaluations are recorded as after-stop and are never
  selectable.

**Every checkpoint keeps:**
- `model.safetensors`, under the model's own parameter names (the r2 fix);
- the tokenizer;
- the sentence-transformers files;
- `metadata.json`, with that step's validation result.

Each is hashed when written, and the hashes go into `selection.json`.

**The resume state, `trainer_state.pt`, stays only in the newest checkpoint.** It holds the optimizer, scheduler,
scaler, random and selection state, and the trainer has worked this way since P0. See decision S1.

## 4. Estimated GPU time

| Phase | Basis | Estimate |
|---|---|---|
| Setup: extraction, hashes, pip | r2 session | 3–5 min |
| Environment and preflight | r2: 114.7 s | 2–3 min |
| Dry run and gate | the same code | < 1 min |
| Training: 12 validations | 139.1 s and 141.9 s in r2; about 121 s in r1 | 24–29 min |
| Training: 55 optimizer steps | 3.08 s measured on the T4 for the 64 longest pairs; seed 42's padded token load is 95–98% of that | ≈ 2.7 min (2.9–3.0 s per step) |
| Training: 12 checkpoint writes | 2.0 s for the weights in r2, plus the ≈ 0.87 GB resume state and hashing | 1–1.5 min |
| `smoke_results` | 12 checkpoint reloads on CPU, with the sentence-transformers check, and the comparison with the base model | 3–6 min |
| Run verification | 3 model loads, the restore, and 2 re-evaluations of about 140 s each | ≈ 6 min |
| Packaging | hashing about 6.1 GB twice and writing the zip | 3–5 min |
| **Total** | | **≈ 45–60 min; budget 1 GPU hour** |

**Training command alone:** about 30–34 min.

**Kaggle's limits:** sessions run up to 12 hours, and the weekly GPU quota is about 30 hours. These figures are
secondhand, not from Kaggle's own documentation. Seed 42 uses about 3% of the quota.

**GPU memory:**
- **Expected peak:** close to the rehearsal's worst case, 9.98 GiB reserved of 14.56 (69%), since nearly every
  positive passage fills 512 tokens.
- **Validation:** peaks at about 2.5 GiB of activations.
- **`gpu_monitor.csv`** records `nvidia-smi` every 2 s, CUDA context included.

## 5. Storage

| Item | Size |
|---|---|
| One checkpoint | 438.7 MB (418.4 MiB): `model.safetensors` 437,951,328 B, `tokenizer.json` 0.7 MB, small files |
| Resume state (newest only) | ≈ 0.87 GB: AdamW's two fp32 moments for 108.9 M parameters. The pooler's 0.59 M get no gradient under CLS pooling, so they have no moments. |
| **Run directory** | 12 × 438.7 MB + 0.87 GB + logs ≈ **6.14 GB (5.72 GiB)** |
| Kaggle `/kaggle/working` | bundle 0.06 + run 6.14 + zip 6.14 + logs ≈ **12.4 GB of the 20 GB output limit**. The notebook stops before training with under 14 GiB free. Packaging reads the run in place and never copies it. |
| Download | `kaggle_a1_seed42_outputs.zip` ≈ **6.2 GB** (stored uncompressed: weights do not compress), plus `.sha256` |
| This Mac | **12 GiB free (94% used).** The zip fits. `check_outputs.py` verifies every file inside the zip and extracts only the records (≈ 50 MB), so no second 6 GB copy is needed. See S3 for the later seeds. |

## 6. What the notebook records for steps 3–5 of the approval

| Approval step | Recorded in the outputs |
|---|---|
| 3. Training behaviour | `log.jsonl`: per step, the loss, the learning rate, the gradient norm before clipping, the loss scale used, the GradCache replay difference and the seconds |
| 3. Validation history and selection | `validation_log.jsonl` and `selection.json`: every evaluation (global and company MRR@10, nDCG@10, Recall@100, per question), the rule's stop and the selected checkpoint |
| 3. GPU memory | each checkpoint's `max_allocated`, and `gpu_monitor.csv` |
| 4. Checkpoint integrity | `results/` (`smoke_results`): `selection.json` re-read (the rule re-derived), every checkpoint re-hashed, reloaded through Hugging Face and sentence-transformers, and compared with the base weights |
| 4. Resume behaviour | `run_verification.json`, checked on the T4 with no optimizer step possible (see the list below) |
| Console and environment | the training console, `environment.json`, `pip_freeze.txt`, the preflight, the dry run, `pins.json`, `status.json`; `OUTPUT_SHA256SUMS` over everything |

**What `run_verification.json` checks:**
- every planned evaluation step has its checkpoint, each re-hashes, with no partial write and the resume state only
  in the newest;
- `checkpoint-step-000055` restores through the trainer's own `restore_checkpoint`. The weights equal the file as
  `from_pretrained` reads it. The AdamW moments equal the saved ones, are finite and sit at step 55. The scheduler,
  scaler, random and selection state are as recorded;
- asked to resume the ended run, the trainer recognises it and changes nothing;
- the selected and the last checkpoints, reloaded from disk, reproduce their recorded validation results exactly;
- step 0 ranks the validation set as the rehearsal's base model did (hash `0de690c4…`).

## 7. Execution instructions (after your final confirmation)

1. **Import:** Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* →
   `dist/kaggle_a1_seed42_notebook/finretriever_a1_train_seed42.ipynb`.
2. **Settings:**
   - Accelerator: **GPU T4** (T4 ×2 is fine).
   - Internet: **on**.
   - *Add Input* → *Your Datasets* → `finretriever-train-v3-a1-r2` (no new upload).
   - Environment: **pin to original environment**, and use this same notebook for the later seeds.
3. **The one edit:** in the first code cell, set `START_OPTIMIZER = True`.
   - Optional: first do one *Run All* with `False`. It verifies everything up to the gate in about 8 GPU minutes and
     trains nothing.
4. **Run:** *Save Version* → *Save & Run All (Commit)*, so the run survives a closed browser. Expect about one hour.
   The last cell prints `SEED 42 TRAINED AND VERIFIED`, or names the step that failed. It never raises, so Kaggle keeps
   the output files.
5. **Download** `kaggle_a1_seed42_outputs.zip` and `kaggle_a1_seed42_outputs.zip.sha256` from the version's output.
   Keep the zip: it is the preserved copy of all 12 checkpoints.
6. **Check locally:**

   ```bash
   python3 -I dist/kaggle_a1_seed42_notebook/check_outputs.py ~/Downloads/kaggle_a1_seed42_outputs.zip \
       --extract dist/kaggle_a1_seed42_results
   ```

7. Tell me. I verify the outputs again and write the seed 42 report. Nothing further runs until you approve seeds
   1337 and 2026.

## 8. If it stops

| Where | Meaning | What to do |
|---|---|---|
| Setup, bundle verification, pip | wrong dataset, a changed file, or a package the pins refuse | Report it. Nothing trained. |
| Environment | not a T4, or the stack differs from the rehearsal's (for example, a newer Kaggle image) | Report it. Nothing trained. Training on another stack needs your decision (S4). |
| Preflight or dry run | a check failed | Report it. Nothing trained. |
| Training, exit ≠ 0 | e.g. a non-finite gradient (fp16 overflow): the run stops before that optimizer step, with no scaler change | The checkpoints so far are packaged. **Do not resume or rerun**; report it. A scale change is a new decision. |
| Session lost mid-run | Kaggle ended the session | Nothing is packaged. Report it before any `--resume`: a resumed CUDA run is close to, but not bit-identical with, an uninterrupted one. |
| Results or verification failed | training completed, a check did not | Everything is packaged. Download it for review. |

## 9. Decisions for you

**S1. The resume state is kept only in the newest checkpoint.**
- **What is kept:** every checkpoint's model, tokenizer, sentence-transformers files and metadata, with their hashes,
  for all 12. This is how I read "preserve every checkpoint and its hashes".
- **What is removed:** `trainer_state.pt` from each older checkpoint. It is resume state, it is not hashed, and
  selection never uses it. The trainer has done this since P0.
- **The alternative:** keeping all 12 needs a trainer code change, so the code would no longer be the rehearsed code.
  It also needs ≈ 9.6 GB more, which pushes the Kaggle total past the 20 GB limit.
- **Recommendation:** keep the design.

**S2. How resume is verified.**
- **Uninterrupted by design:** seed 42 runs without an interruption, because an interrupted CUDA run is not
  bit-identical with an uninterrupted one, and the seeds must be comparable.
- **Verified on the T4:** the real step-55 resume state, which includes AdamW moments that the rehearsal's step-0
  check did not have, restores, and the ended-run guard holds.
- **Verified on CPU:** the CPU test shows a resumed run continues bit for bit.
- **Not measured:** a CUDA continuation measurement would need its own decision.
- **Recommendation:** accept.

**S3. Local disk.**
- **Now:** 12 GiB free. Seed 42's 5.7 GiB zip fits.
- **Later seeds:** three seeds need about 17 GiB, so free space or use an external drive before seed 1337.
- **For you to decide:** `~/Downloads` holds two identical copies of the r2 outputs zip (438 MB each) and the 2.2 GB
  smoke backup zip. I have not touched them.

**S4. The Kaggle image.**
- **Now:** the notebook refuses to train on a stack other than the rehearsal's, since the decision rule requires one
  stack for all seeds.
- **If Kaggle's image has changed:** the choice is between re-rehearsing on the new image and running all three seeds
  there, or finding the pinned image.

## 10. Verified locally (no GPU, nothing trained)

- **The r2 bundle:**
  - re-hashed, and SHA256SUMS passes;
  - all 104 files are identical to the repository;
  - the seed 42 dry run from the extracted bundle passes;
  - the dry run writes only bytecode caches.
- **The notebook, run cell by cell on this Mac with the gate closed:**
  - the bundle and hashes verify;
  - the pinned package versions match;
  - the environment cell stops ("needs a T4");
  - the dry run matches the pinned schedule;
  - the gate stops before training, after packaging;
  - the training cell refuses by itself.
- **A gate-open copy (differs by that one line), with a stand-in run directory; the training cell never executed:**
  - the embedded verification script matches its pinned hash;
  - both post-run tools fail cleanly on the stand-in;
  - packaging reads the run in place;
  - the last cell reports the failure without raising.
- **`check_outputs.py`:**
  - verifies the zip and every file inside it;
  - extracts only the records;
  - catches a one-byte change in a checkpoint file;
  - refuses an existing extraction directory.
- **`verify_seed_run.py`,** tested on finished fixture runs (`TinyEncoder` on CPU, as the trainer tests do):
  - all four checks pass on a run whose rule stopped early (after-stop path included), and nothing is written;
  - wrong pins, a changed checkpoint and a missing resume state are each reported, and the other checks still run.
- **The repository:**
  - the full CPU suite: 1,309 passed, 25 skipped;
  - Ruff is clean;
  - no `src/`, `data/` or `configs/` file changed;
  - the disk-space guard refused this Mac's 12.2 GiB, as intended; the local harness lowers it.
