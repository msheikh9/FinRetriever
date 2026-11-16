# Trainer smoke test on a Kaggle T4

This runs the mechanical smoke test of the BGE contrastive trainer
(docs/training_trainer_v1.md, §5) from a portable bundle. It shows that
training runs on a GPU, the loss stays finite, and checkpoints save, reload and
resume.

It measures nothing about retrieval:
- 48 synthetic, AI-reviewed queries (not human-validated, not gold);
- 15 steps;
- no evaluation and no checkpoint selection.

Do not report a retrieval result from it.

## 1. Build the bundle (locally, from the repository root)

```
python3 scripts/smoke_bundle.py build
```

This writes three files to `dist/kaggle/` (gitignored):
- `finretriever_smoke_v1.tar.gz`, the bundle;
- `finretriever_smoke_v1.tar.gz.sha256`, its SHA-256;
- `dataset-metadata.json`, a template for the Kaggle CLI.

The bundle holds only what the run reads. Every file has its repository path
and a SHA-256 in `BUNDLE_MANIFEST.json` and `SHA256SUMS`:
- the code and scripts;
- the two smoke configs, the default and a gradient-checkpointing fallback;
- the data-preparation config, the frozen corpus and its manifest, and the
  `bge_base_ft_v2` preparation;
- the `train_candidates_v1` candidates;
- the pilot pair files and generation records the candidates trace to;
- the two quality-decision files.

It holds no evaluation data, raw SEC files, caches, models, results, `.env`
files or keys. The build refuses any of them.

## 2. Upload as a private Kaggle dataset

**Web interface:**
1. Go to kaggle.com, then Datasets, then New Dataset.
2. Upload `finretriever_smoke_v1.tar.gz` and its `.sha256` file.
3. Keep the visibility **Private**, and give it a title such as
   `finretriever-smoke-v1`.

**CLI**, using your own `~/.kaggle/kaggle.json`, which never goes into the
bundle:
1. Replace `YOUR_KAGGLE_USERNAME` in `dist/kaggle/dataset-metadata.json`.
2. Run `kaggle datasets create -p dist/kaggle`. Datasets are private unless
   `--public` is given.

Kaggle may unpack the archive on upload. Step 4 handles either case.

## 3. Notebook settings

Create a notebook and, in its settings:
- **Accelerator:** GPU T4 (a "T4 x2" machine is fine: the run uses one GPU).
- **Internet:** on. pip and the base model download need it.
- **Add Input:** your private `finretriever-smoke-v1` dataset.

## 4. Copy the bundle to a writable directory

`/kaggle/input` is read-only, and the run writes `models/` and
`smoke_outputs/` inside the bundle directory.

```python
import hashlib, pathlib, shutil, tarfile
src = next(p for p in pathlib.Path("/kaggle/input").rglob("*") if p.name in ("finretriever_smoke_v1.tar.gz", "BUNDLE_MANIFEST.json"))
dst = pathlib.Path("/kaggle/working")
if src.name.endswith(".tar.gz"):
    expected = (src.parent / (src.name + ".sha256")).read_text().split()[0]
    actual = hashlib.sha256(src.read_bytes()).hexdigest()
    assert actual == expected, f"archive SHA-256 {actual} != {expected}"
    with tarfile.open(src) as tar:
        tar.extractall(dst, filter="data")
else:  # Kaggle unpacked it
    shutil.copytree(src.parent, dst / "finretriever_smoke_v1", dirs_exist_ok=False)
print(sorted(p.name for p in (dst / "finretriever_smoke_v1").iterdir()))
```

Every later cell runs in `/kaggle/working/finretriever_smoke_v1`.

## 5. Verify every artifact checksum

```
%cd /kaggle/working/finretriever_smoke_v1
!sha256sum --quiet -c SHA256SUMS && echo "all checksums match"
```

The preflight (step 8) checks them again. It also checks that no unlisted or
evaluation file is present.

## 6. Install only what is missing

Keep the image's CUDA build of torch. Pin it as a constraint, so pip fails
rather than replacing it:

```
!python -c "import torch; print('torch==' + torch.__version__)" > /tmp/torch.txt
!pip install -q -r requirements-kaggle.txt -c /tmp/torch.txt
!python -c "import torch, transformers, tokenizers; print(torch.__version__, torch.version.cuda, transformers.__version__, tokenizers.__version__)"
```

- If pip reports a conflict with torch, stop. Do not upgrade torch; note the
  versions instead.
- `tokenizers` must install at 0.22.2. Another version may serialise the
  tokenizer differently, and then the trainer refuses to run.

## 7. Check CUDA and free GPU memory

```
!nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
!python -c "import torch; f,t=torch.cuda.mem_get_info(); print(torch.cuda.get_device_name(0), f'{f/2**30:.1f} GiB free of {t/2**30:.1f}')"
```

A T4 shows about 15 GiB in total, and less is free:
- the CUDA context and other processes in the session take some;
- restart the session if a previous run left memory allocated.

The preflight compares the measured free memory with the estimate.

## 8. Preflight (checks everything, trains nothing)

```
!python scripts/smoke_preflight.py
```

It writes `smoke_outputs/preflight.json` and ends with `READY FOR TRAINING`
only if every check passes:

| Check | What it verifies |
|---|---|
| `bundle` | Every file matches its SHA-256, and nothing unlisted is present. |
| `config` | The smoke config is the bundled one, unchanged. |
| `corpus_and_preparation` | The corpus (10,137 passages, `corpus_v1@1.2`) and the `bge_base_ft_v2` preparation verify. The evidence index is rebuilt and matches its digest. |
| `candidates`, `provenance` | All 66 candidates re-verify. Each traces to its source pair, its accepting decision and its generation record. |
| `smoke_subset`, `schedule` | The config selects the expected 48 pairs and 15 steps, and every batch holds only allowed negatives. |
| `evaluation_isolation` | `data/eval` and every evaluation source the preparation config names are absent, and none was opened. |
| `model_and_tokenizer` | `BAAI/bge-base-en-v1.5` resolves to `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. The tokenizer's SHA-256 matches the preparation's (`9114256b…`). Every query fits in 128 tokens, and truncating each positive passage at 512 tokens keeps its pair's visible text (28 of 37 positives are truncated). The base weights load in evaluation mode, with no optimizer and no gradients. |
| `cuda_fp16` | CUDA is available, and fp16 autocast with loss scaling gives finite gradients. |
| `gpu_memory` | Free memory, measured now, covers the ESTIMATED peak with a 15% margin. |

The estimates are not measurements:
- 8.6 GiB at batch 16;
- 3.3 GiB with gradient checkpointing.

If `gpu_memory` warns or fails, use the fallback config. It has the same
inputs and schedule, with gradient checkpointing and its own run name:

```
!python scripts/smoke_preflight.py --config configs/train/biencoder_bge_base_smoke_gc.yaml
```

The first preflight downloads the base model (about 440 MB) into the Hugging
Face cache, and training reuses it. If the download stalls, set
`HF_HUB_DISABLE_XET=1` and retry.

transformers prints a "BertModel LOAD REPORT" listing
`embeddings.position_ids` as UNEXPECTED. The checkpoint stores that buffer and
the current model class rebuilds it, so it is harmless. It also appears during
training.

The preflight rebuilds the evidence index on the CPU. That takes minutes, and
longer on a slow CPU.

## 9. Run the smoke test

```
%%bash
set -o pipefail
cd /kaggle/working/finretriever_smoke_v1
mkdir -p smoke_outputs
CUDA_VISIBLE_DEVICES=0 python scripts/05_train_biencoder.py \
    --config configs/train/biencoder_bge_base_smoke.yaml --confirm-training 2>&1 | tee smoke_outputs/train_console.log
```

`--confirm-training` is required. Without it the script only verifies and
prints the plan.

To use the fallback, replace the config with
`configs/train/biencoder_bge_base_smoke_gc.yaml`. Its run directory ends in
`bge_base_ft_v2_smoke_gc-seed42`.

**Expected console output:**
- the plan: 48 of 66 candidates, 15 optimizer steps with 2 warmup steps,
  batches per epoch `[3, 4, 4, 3, 4]`, 15 in-batch negatives per query;
- the lengths: `queries: longest 55 tokens (limit 128, never truncated);
  positive passages: 28 of 37 longer than 512 tokens (longest 1044) are
  truncated to 512, keeping every pair's visible text and evidence`;
- one line per step: `step k/15 epoch e loss … lr …`;
- finally: `done: 15/15 steps; checkpoints ['checkpoint-step-000005',
  'checkpoint-step-000010', 'checkpoint-step-000015']`.

**Outputs:**

```
models/bge_base_ft_v2_smoke/bge_base_ft_v2_smoke-seed42/
  run.json                    configuration, inputs and schedule this directory is bound to
  log.jsonl                   one line per step: loss, learning rate, gradient norm, seconds
  checkpoint-step-000005/     model.safetensors, config.json, tokenizer files, modules.json,
  checkpoint-step-000010/       sentence_bert_config.json, 1_Pooling/, 2_Normalize/, metadata.json
  checkpoint-step-000015/     the same, plus trainer_state.pt (only the newest checkpoint keeps it)
smoke_outputs/preflight.json, smoke_outputs/train_console.log
```

**Interruptions.** If the session stops, rerun the same command with
`--resume` added. It continues from the newest checkpoint.

**Refusals.** The trainer refuses to write over an existing run directory.
Re-running the command after the run completed is refused unless `--resume`
is given, in which case there is nothing left to do.

## 10. Collect the results (no training)

```
!python scripts/smoke_results.py --run-dir models/bge_base_ft_v2_smoke/bge_base_ft_v2_smoke-seed42 \
    --console-log smoke_outputs/train_console.log
```

It writes `smoke_outputs/results.json` and `smoke_outputs/results.md`, and
exits 0 only when the mechanics check out. It reports:
- steps completed;
- the training loss of the first and last step, and its range;
- non-finite losses and gradient norms;
- the measured peak GPU memory, with the device name;
- the training and wall-clock durations;
- for each checkpoint: its files, a reload through Hugging Face and
  sentence-transformers, and probe embeddings that must be finite;
- the newest resumable state;
- whether the final weights differ from the base model's;
- seed and revision;
- every console line that is not the trainer's own report, a progress bar
  or blank, by category (tokenizer, model load, CUDA or memory, numerical,
  error, warning, other). In transformers' LOAD REPORT only
  `embeddings.position_ids` UNEXPECTED counts as known benign; every other
  line is listed under "Console lines needing review".

Report it in this form:

| Item | Value |
|---|---|
| Run, seed | `bge_base_ft_v2_smoke-seed42`, 42 |
| Base model | `BAAI/bge-base-en-v1.5` @ `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| GPU | e.g. Tesla T4; torch, CUDA, transformers and tokenizers versions |
| Steps completed | … of 15 |
| Training loss, first step → last step (range) | … → … (… – …) |
| Non-finite losses / gradient norms | … / … |
| Peak GPU memory (measured, `max_memory_allocated`) | … GiB |
| Duration: training steps / wall clock | … s / … s |
| Checkpoints 5, 10, 15: saved / reloaded / sentence-transformers | … |
| Resumable state (step 15, schedule matches) | … |
| Final weights differ from base | … of … tensors |
| Console lines needing review (tokenizer, model load, CUDA, numerical, other) | none / list |

The statement that goes with it: *Mechanics check only. No retrieval
evaluation was run and no retrieval improvement is claimed.*

The training loss falls partly because 48 pairs repeat over 5 epochs. It is
not evidence of better retrieval.

## 11. Keep the outputs

Save a notebook version, so `/kaggle/working` is kept, or download:
- `smoke_outputs/`, which is small and is what to send back;
- optionally `checkpoint-step-000015/` (about 440 MB).

Checkpoints and outputs never go into the repository (`models/` is gitignored).
