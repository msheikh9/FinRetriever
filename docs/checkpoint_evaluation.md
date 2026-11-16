# Evaluating local fine-tuned checkpoints

Status: 2026-11-04.

**Implemented and verified:**
- the harness loads local checkpoints;
- a pre-flight check runs on the real configuration;
- the three smoke-test checkpoints load and embed correctly.

**Not yet run:** the evaluation of the three checkpoints. It needs about 30 minutes of CPU encoding per
checkpoint. The commands are in §5.

**Unchanged:**
- no model was trained;
- no benchmark result was written or changed;
- the BM25, BGE and E5 baselines;
- the corpus and the evaluation labels.

## 1. What is evaluated

| | |
|---|---|
| Checkpoints | `models/bge_base_ft_v2_smoke/bge_base_ft_v2_smoke-seed42/checkpoint-step-0000{05,10,15}`: the three fixed-step checkpoints of the 15-step Kaggle smoke test (48 synthetic pairs of `train_candidates_v1`, `bge_base_ft_v2`, seed 42) |
| Labels | `dev_pilot_v3`: the frozen development labels, 10 questions and 696 judged pairs (692 AI proposals, 4 human grades). Provisional development labels, not gold. |
| Corpus | `corpus_v1@1.2`, 10,137 passages, SHA-256 `ff9dadf3…2004` |
| Settings | the benchmark config's existing ones: top_k 100, judged@10, both retrieval modes (`global`, `company`), the BGE query instruction, 512 tokens |
| References | the stored zero-shot rankings of `results/dev/pilot_v2` (BM25, BGE, E5 in both modes), scored with the same labels: they are the `results/dev/pilot_v3` scores exactly, and they are never re-run |
| Output | `results/dev/pilot_v3_smoke_checkpoints/` (new, written once) |

**No checkpoint is selected.**
- All three are reported side by side.
- The smoke test's validation set (`validation_pairs_v1`) is not read.
- No metric on any set chooses among them.

**The 15-step run checked the training mechanics only, and its scores are no evidence of a retrieval
improvement.** The report and every `metrics.json` carry this caution (`CHECKPOINT_CAUTION`).

## 2. Loading a local checkpoint safely

`finretriever.retrieval.checkpoint` opens a trainer checkpoint without trusting the directory.

**Before anything loads** (`inspect_checkpoint`):
- **The record:** `metadata.json` is a trainer checkpoint record. Its base model has a built-in input format,
  and it was trained with exactly that format's prefixes (BGE: the query instruction, no passage prefix).
- **Weights:** only `model.safetensors` is accepted.
  - A pickled weight file is refused.
  - `trainer_state.pt` (optimizer state, a pickle) is never opened.
- **No custom code:** `config.json` names none (`auto_map`).
- **Modules:** `modules.json` lists exactly Transformer, Pooling and Normalize. Their settings must agree
  with `config.json` and the metadata: pooling, dimension, a sequence length within the position
  embeddings, no lower-casing and no prompts.
- **No outside files:** no file may be a symbolic link out of the directory.
- **Hashing:** every file that determines the embeddings is hashed (9 files). The digest of those hashes is
  the checkpoint's revision (`sha256:…`), which keys the embedding cache and is recorded with every run.

**Loading** (`load_checkpoint_encoder`):
- **Explicit modules:** the three modules are built explicitly, not chosen by `modules.json`. Loading uses
  local files only (`local_files_only`), no remote code (`trust_remote_code=False`) and safetensors only.
- **Checks on the loaded model:**
  - the tokenizer hashes to the SHA-256 the checkpoint was trained with, which is also the tokenizer its
    training data was measured with;
  - every token id has an embedding row;
  - the sequence limit fits the tokenizer and the position embeddings;
  - the embedding dimension is the model's;
  - the weights did not change while loading.

**Scoring** (`CheckpointRetriever`) is a subclass of the dense retriever, so it shares the baselines'
retrieval interface and code:
- L2-normalised float32 embeddings and cosine scores in float64;
- queries encoded one per forward pass, passages in batches;
- the company restriction, and ties broken by corpus position.

**Verified on the three real checkpoints:**
- each inspects and loads and gives 768-dimensional unit vectors;
- each is identical (maximum difference 0.0) to a plain `SentenceTransformer` load of the same directory;
- the step-15 checkpoint sits at a cosine of about 0.99 from base BGE on sample texts.

The trainer's `2_Normalize` directory is missing from the downloaded checkpoints, so the inspector does not
require it.

## 3. Evaluation isolation

**When it runs:** before any model loads, each checkpoint must show from its own training records that it
never trained on the set's judged passages or questions (`finretriever.evaluation.isolation`). If it
cannot, the run stops (`IsolationError`) and nothing is loaded or written.

**The records it reads**, each checked against the SHA-256 the checkpoint recorded:
- the training corpus;
- the preparation config and manifest;
- the blocklist;
- the training-candidate manifest and file.

**For the three smoke checkpoints:**

| Check | Result |
|---|---|
| Trained on the evaluated corpus (SHA-256 and version) | yes |
| Judged passages of `dev_pilot_v3` on the `bge_base_ft_v2` blocklist | 667 of 667 |
| Passages used by the 66 candidates of `train_candidates_v1` (a superset of the 48 trained on): positives and equivalents that are judged | 0 of 188 |
| Highest similarity of a training question to an evaluation question | 0.28 (threshold 0.8) |

**Two more guards:**
- an evaluation set whose questions or labels lie in a preparation's training or validation data is refused;
- the harness opens no validation set.

## 4. What is reported

**For every run** (checkpoints and stored references alike), computed by the same function as every earlier
set:
- **Metrics:** nDCG@10 (graded), MRR@10, Recall@100 (against the judged positives), and judged@10. Unjudged
  passages count as non-relevant.
- **Sensitivity check:** nDCG@10 and MRR@10 on the judged passages only.
- **Statistical uncertainty:** a 95% percentile bootstrap interval over questions for each metric (10,000
  resamples, fixed seed).
- **Paired comparison** with `bge_zero_shot` in the same mode:
  - the mean per-question difference, with a 95% paired bootstrap interval;
  - wins, ties and losses;
  - the p-value of an exact two-sided sign-flip test (all 2^10 patterns).
- **Per question:**
  - every metric;
  - judged@10, with the top 10 split into relevant, confirmed negative and unjudged (in or out of the
    company asked about);
  - the first relevant rank.
- **Relevant retrieval failures:**
  - every judged-relevant passage ranked below the top 10, with its rank, or "not returned" if it is not in
    the top 100;
  - the questions with no relevant passage in the top 10, listed in the report and counted per run.
- **Provenance:**
  - the checkpoint path, step, file hashes and digest, its `run.json`, and its training inputs;
  - the isolation record;
  - the embeddings' SHA-256 and cache key, and the library versions;
  - the label files' SHA-256 and the benchmark config's SHA-256;
  - each run file's SHA-256;
  - for the references, the stored run files and their hashes.

**How to read it:**
- **Unjudged passages:** `dev_pilot_v3` judged the top 10 of the six zero-shot runs and nothing a fine-tuned
  model ranks differently. A checkpoint's judged@10 will be below 1.0, and its unjudged passages count as
  non-relevant, which biases its scores down. Read every score beside its judged@10, its unjudged counts and
  the judged-only columns.
- **Few questions:** with 10 questions an interval is about ±0.15 nDCG@10 wide. The zero-shot baselines'
  intervals already overlap: BGE global 0.286 [0.145, 0.445], E5 global 0.361 [0.210, 0.514].
- **No improvement can be claimed.** A smoke checkpoint that differs from base BGE by less than this says
  nothing, and one that differs by more may differ because of unjudged passages.

## 5. Commands

Run from the repository root with the dense-retrieval venv, which has working scipy and sentence-transformers.
The models are already cached, so nothing is downloaded.

```bash
export HF_HUB_OFFLINE=1
PY=python   # the environment of requirements/lock-*.txt (docs/environment.md)

# 1. Pre-flight check (seconds). It verifies:
#      - corpus and label provenance;
#      - each checkpoint's inspection and evaluation isolation.
#    It also scores the stored references with their intervals.
#    No model is loaded and nothing is written.
$PY scripts/04_run_baselines.py --eval-set dev_pilot_v3_smoke_checkpoints --check-only

# 2. The evaluation (on mains power). It:
#      - encodes the corpus once per checkpoint, about 30 min each on CPU;
#      - caches the embeddings under data/cache/embeddings/, keyed by the checkpoint digest;
#      - scores both modes;
#      - writes results/dev/pilot_v3_smoke_checkpoints/ once.
$PY scripts/04_run_baselines.py --eval-set dev_pilot_v3_smoke_checkpoints
```

**Exit codes:**
- **0:** all three checkpoints ran.
- **1:** results were written, but a checkpoint could not be loaded (listed as skipped, with the reason).
- **2:** a provenance, isolation or configuration error, or results already present in the output
  directory. Nothing is written.

**Rerunning:**
- With the same results, the rerun changes nothing.
- With different results (other library versions, for example), it is refused. Use a new `output_dir` in
  `configs/eval/benchmark.yaml`.

**On this Mac:**
- `device: auto` uses the CPU, as the stored zero-shot runs did.
- `device: mps` in the three retriever entries would be faster. It gives slightly different numbers and its
  own cache entries.

## 6. Changes

| File | What |
|---|---|
| `src/finretriever/retrieval/checkpoint.py` (new) | Inspection, safe loading, `CheckpointRetriever` |
| `src/finretriever/evaluation/isolation.py` (new) | Evaluation isolation from a checkpoint's training records |
| `src/finretriever/evaluation/uncertainty.py` (new) | Bootstrap intervals, paired differences, the sign-flip test |
| `src/finretriever/retrieval/biencoder.py` | The shared settings check moved into `_configure`, and the model loader into `_load_encoder`, so a subclass can supply its own; behaviour unchanged |
| `src/finretriever/evaluation/benchmark.py` | The changes listed below |
| `scripts/04_run_baselines.py` | Prints intervals, paired differences and isolation; `--check-only` |
| `configs/eval/benchmark.yaml` | The new set `dev_pilot_v3_smoke_checkpoints`; the baselines and the earlier sets are unchanged |
| `tests/unit/test_checkpoint_evaluation.py`, `tests/unit/checkpoint_fixtures.py` (new), `tests/integration/test_run_baselines_script.py` | Loading, refusals, dimensions, ranking, cache identity, provenance, isolation, pre-flight, write-once, uncertainty |

The changes to `benchmark.py`:
- the `dense_checkpoint` retriever type;
- per-set `retrievers`, `reference_runs_from` and `paired_reference`;
- the isolation gate before any model loads;
- bootstrap intervals and paired comparisons;
- relevant-failure lists;
- write-once outputs;
- the evaluation-data path guard;
- the pre-flight mode (`load_models=False`).
