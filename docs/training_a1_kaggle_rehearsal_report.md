# A1 Kaggle preflight and GPU rehearsal: report

**Status, 2026-11-10: the r2 rehearsal PASSED all ten checks on a Kaggle T4, and the preflight is READY FOR
TRAINING. B3 is closed.**
- **The r1 bundle:** it ran first, with 8 of 10 checks passing.
- **Its two failures** are analysed and fixed in [gpu_rehearsal_failure_analysis.md](gpu_rehearsal_failure_analysis.md).
- **What passed** is the rebuilt `finretriever_train_v3_a1_r2` bundle.
- **Where to read it:** §2–§8 report the r2 outputs, verified locally from the downloaded archive (below). §1 and
  the run instructions at the end describe the r1 stage, kept as they were.

**Not done in either rehearsal:** no optimizer step, no GradScaler step, no training, no seed run, no change to data,
hyperparameters or the scaler, no API request, no commit or push. Seed 42's execution plan is in
[training_a1_seed42_plan.md](training_a1_seed42_plan.md).

## 1. Bundle verification (done)

**Built** with `scripts/smoke_bundle.py build --config configs/train/biencoder_bge_base_ft_v3_a1.yaml --name
finretriever_train_v3_a1 --output-dir dist/kaggle_a1`.
- **Location:** a new directory `dist/kaggle_a1/` (gitignored) rather than `dist/kaggle/`. `dist/kaggle/` holds the
  smoke bundle and its `dataset-metadata.json`. The build never overwrites a template, so a shared directory would
  have kept the smoke template, and an upload of that directory would also have carried the smoke archive. The
  smoke bundle's three files keep their SHA-256 (`2c8bead5…`, `20a154f3…`, `b0ed9f8e…`).

| File (`dist/kaggle_a1/`) | SHA-256 | |
|---|---|---|
| `finretriever_train_v3_a1.tar.gz` | `2c3f7a64c77715dd17b895d7b7af05ebe01056919e6162d4b7abe2f66906b3e6` | 9.5 MB; identical to the earlier scratch build (deterministic) |
| `finretriever_train_v3_a1.tar.gz.sha256` | `3446e8d9…` | names the archive's SHA-256 |
| `dataset-metadata.json` | `66348e55…` | template: set your Kaggle username for a CLI upload |
| `BUNDLE_MANIFEST.json`, inside | `6c9ed8f1618d82f4ef328ec5db369536e62262beb4a2ca1f4c09b4992dde17fd` | pinned in the notebook |

**Checked on a fresh extraction:**

| Check | Result |
|---|---|
| Archive against its `.sha256` | OK |
| Every file against `SHA256SUMS` | all 104 match |
| `verify_bundle`: manifest hashes, nothing unlisted | passes: 104 files (code 53, config 2, corpus 2, preparation 7, candidates 4, validation set 5, …) |
| Every bundled file against the repository's | **all 104 byte-identical** |
| `train_candidates_v3` | all 4 files, byte-identical; **678** candidates |
| `validation_pairs_v4` | all 3 files, byte-identical; **93** questions |
| Recorded schedules | seeds 42, 1337, 2026: 55 steps each (schedules `44b0ed7c…`, `795095dc…`, `49b2b153…`, as the dry run) |
| The rehearsal in the bundle | `scripts/gpu_rehearsal.py`, `training/gpu_rehearsal.py` and the checkpoint loader are bundled |
| CPU preflight from the extracted bundle (earlier, `--skip-cuda`) | every non-CUDA check passes |

**Evaluation isolation.**
- **Absent:**
  - no file under `data/eval`;
  - none of the 8 evaluation sources the preparation names (the annotation questions and pool, the dev pool, the
    audits, the gold qrels, queries and metadata);
  - no relevance label, grade, qrels row, judgment or evaluation question text in any bundled file.
- **The only matches** of label-like strings are:
  - file-path strings in a config and in manifests' provenance;
  - the descriptions "AI quality review … not gold".
- **What the bundle does carry, which you should know:** the bundled preparation, which training needs and verifies
  by hash, marks **3,950 passages `blocked`** (`passages.jsonl`), and `blocklist.jsonl` gives each one's reason:
  - 558 in the annotation pool;
  - 667 in the dev pool;
  - 50 in audit samples;
  - 2,122 overlap-group copies and 1,161 evidence-reuse links of those.

  So passage IDs reveal evaluation-pool membership. They carry no relevance judgment: pools hold relevant and
  non-relevant passages alike. This is the mechanism that keeps those passages out of training and out of every
  negative.
- **The corpus.** The full corpus text (10,137 passages) is bundled because validation ranks the whole corpus.
- **Not new.** The approved smoke and rehearsal bundles carried the same.

**The Kaggle notebook,** `dist/kaggle_a1_notebook/finretriever_a1_rehearsal.ipynb` (`6696e325…`, gitignored):
- **What it pins:** the archive and manifest SHA-256 and the package versions.
- **What it requires:** a T4 with compute capability 7.5.
- **What it runs:** the full preflight, then the rehearsal.
- **What it saves:** every output, with SHA-256s, the full check-9 checkpoint included.
- **Stopping:** each step raises on failure, so Run All stops at the first problem.
- **No training:** it contains no training command.
- **Tested locally** as far as this Mac allows:
  - the extraction, pinned-hash and bundle-verification cells pass;
  - the GPU cell stops with "the A1 runs need a T4";
  - the rehearsal cell's refusal path still packages and hashes the outputs, then stops.

  A first version's verification cell called a script the bundle does not ship; the local run caught it, and it was
  fixed.

## 2–6. The r2 results (Kaggle T4, 2026-11-10)

**The outputs.**
- `kaggle_a1_r2_outputs.zip`: SHA-256 `e27c8a0b1fc270e65c530b028d7d70c1d8d9797c0cc89199898421325e3ca19e`, equal to its
  `.sha256`.
- All 20 files match `OUTPUT_SHA256SUMS`; I re-verified them from `dist/kaggle_a1_r2_results/`.
- The rehearsal took 352 s; loading the model took 6.5 s.

**Environment.**
- **Hardware:** Tesla T4 ×2, one used. Compute 7.5, 14.56 GiB, driver 580.178.04.
- **CUDA stack:** CUDA 13.0, cuDNN 92700.
- **Software:** Python 3.13.16, torch 2.11.0+cu130, transformers 5.3.0, tokenizers 0.22.2. The pinned packages
  matched.

**2. Preflight: `READY FOR TRAINING` (114.7 s; 13 of 13 checks pass)**

| Check | Result |
|---|---|
| bundle, config | 104 files match; the config is unchanged (`7a4c4618…`) |
| corpus and preparation | `corpus_v1@1.2`, 10,137 passages; the preparation and evidence index verify |
| candidates, provenance | **678** re-verified; traced to p1 612, pilot_v3 51, pilot_v2 15 |
| schedule | seeds 42, 1337 and 2026: 55 steps each, every batch's negatives allowed |
| validation set | `validation_pairs_v4`, **93** questions, isolated from the 678 pairs; selection only |
| evaluation isolation | `data/eval` and all 8 evaluation sources absent; none of the 44 files read is one |
| model and tokenizer | BGE at `a5beb1e3…`; the tokenizer matches the preparation; the longest query is 58 tokens; 425 of 473 positives are truncated at 512 tokens, each keeping its visible text |
| CUDA fp16, GradCache, GPU memory | fp16 autocast and scaling work; GradCache equals the uncached passes (relative 2.1e-08); 14.5 GiB free against a 9.13 GiB peak |

**3. The rehearsal: `REHEARSAL PASSED` (10 of 10)**

| # | Check | Measurement |
|---|---|---|
| 1 | environment | as above; every seed must run on this GPU type |
| 2 | dropout | 37 active modules (eager attention); another random state changes the embeddings (relative difference 0.25 in fp32, 0.23 in fp16); a restored state replays them exactly (0) |
| 3 | attention dropout replay | 12 modules alone: another state changes the output (0.08 / 0.11); replay is exact (0); probabilities restored |
| 4 | GradCache fp32 | 8 longest pairs, mini-batch 4: relative 5.4e-08, cosine 1.0000000, replay 0 |
| 5 | GradCache fp16 | scales 65,536 and 1,024: relative 2.2e-08, cosine 1.00000, replay 0 |
| 6 | fp16 loss scale | every gradient finite at **65,536** on each seed's first batch (62, 61, 62 pairs) and on the 64 longest pairs; fp16 against fp32 with dropout off: **cosine 0.99998**, relative 5.9e-03 |
| 7 | peak memory | one cached step of the 64 longest pairs with AdamW's state (0.816 GiB): **9.98 GiB reserved of 14.56 (69%)**, 9.21 allocated, 3.08 s |
| 8 | validation | 93 questions over 10,137 passages in 139.1 s and 141.9 s, identical on repeat; peak 2.51 GiB; no weight, mode or random state changed; ranking hash `0de690c4…` |
| 9 | checkpoint resume | `checkpoint-step-000000`, 418.4 MiB, re-hashes; it restores identical weights and the optimizer, scheduler, scaler, random and selection state into a fresh base model; the benchmark's loader agrees to 1.4e-07; a second save is refused |
| 10 | no training | 0 optimizer steps; the weights at the end equal those at load (`acba92a4…`) |

**4. fp16 and equivalence.**
- No overflow at the GradScaler's default initial scale, and the scaler was never changed.
- From one random state, the fp16 and fp32 dropout masks agree on 0.819 of the elements. The dropout-on comparison,
  recorded only, gave cosine 0.5311. This confirms the diagnosis of r1's failure.

**5. Peak memory.** The worst-case step uses 69% of the T4, against an 85% limit. The validation pass peaks at
2.51 GiB.

**6. Checkpoint integrity.** Every flag is true.

**A difference from r1:** validation took about 140 s per pass in r2, against about 121 s in r1, on the same code
path. I have not explained it; the seed estimates use the slower figure.

## 7. Remaining blockers

**None for seed 42.**
- **B1, the decision rule:** implemented (K1).
- **B2, the data size:** 678 accepted (J2).
- **B3, the CUDA path:** closed by this rehearsal.
- **B4, the batch gate:** passed for batches 1–6.

## 8. Ready for the A1 experiment?

**Yes, for seed 42, once you give the final confirmation.**
- The plan, the verified configuration and the estimates are in
  [training_a1_seed42_plan.md](training_a1_seed42_plan.md).
- Seeds 1337 and 2026 need a separate approval after seed 42 is reviewed.

## How to run Stages 2–3 (your Kaggle account)

1. **Upload the dataset** at kaggle.com → *Datasets* → *New Dataset*.
   - Add both `dist/kaggle_a1/finretriever_train_v3_a1.tar.gz` and `dist/kaggle_a1/finretriever_train_v3_a1.tar.gz.sha256`.
   - Title it `finretriever-train-v3-a1` and set visibility to **Private**, then *Create*.
   - Kaggle may unpack the archive; the notebook handles both forms and checks the pinned manifest hash either way.
   - **CLI alternative,** with your own Kaggle token in `~/.kaggle/kaggle.json` (never shared): set your username in
     `dist/kaggle_a1/dataset-metadata.json`, then run `kaggle datasets create -p dist/kaggle_a1`. Datasets are
     private by default.
2. **Import the notebook:** *Code* → *New Notebook* → *File* → *Import Notebook* →
   `dist/kaggle_a1_notebook/finretriever_a1_rehearsal.ipynb`.
3. **Settings:**
   - *Accelerator:* **GPU T4** (T4 ×2 is fine). A P100 is refused by the notebook.
   - *Internet:* **On**, for pip and the BGE download.
   - *Add Input* → *Your Datasets* → `finretriever-train-v3-a1`.
4. **Run All,** or *Save Version* → *Save & Run All* to run it in the background with outputs kept. The notebook
   stops at the first failure and still saves the outputs.
5. **Download** `kaggle_a1_outputs.zip` and `kaggle_a1_outputs.zip.sha256` from the output panel: about 450 MB,
   because it holds the full checkpoint.
6. **Check locally,** then tell me:

   ```bash
   python3 dist/kaggle_a1_notebook/check_outputs.py ~/Downloads/kaggle_a1_outputs.zip --extract dist/kaggle_a1_results
   ```

   It verifies the zip and every file's SHA-256 and prints every check. I will then complete §2–§8 from those files.

**Training stays out of scope.** It needs your separate approval.
