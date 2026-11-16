# First domain-adapted bi-encoder experiment: readiness review and plan

Status: 2026-10-29.

- **No training has run.**
- **No synthetic query exists.** Nothing in `data/eval/` is used for
  training.
- **This plan is preparation only.** It covers the readiness findings, the
  proposed configuration ([`configs/train/biencoder_bge_base.yaml`](../configs/train/biencoder_bge_base.yaml)),
  the first ablations, the compute estimate, and the blockers that must
  close before a run.

## 1. What exists today

| Component | State | Notes |
|---|---|---|
| `scripts/02_generate_queries.py` | Stub (prints, exits 0) | No generator, prompt, rate-limit handling or leakage check. |
| `scripts/05_train_biencoder.py` | Implemented (docs/training_trainer_v1.md) | Dry run by default; `--confirm-training` trains. |
| `scripts/06_run_benchmark.py` | Stub | Development evaluation runs through `04_run_baselines.py --eval-set`. |
| `finretriever.training.biencoder_trainer` | Implemented: in-batch loss, seeding, checkpoints, resume | No GradCache and no hard negatives yet. |
| `finretriever.training.negative_mining` | TODO comments | It plans to filter near-duplicates by cosine threshold. That misses shifted copies (§2). |
| `finretriever.utils.seed` | TODO | |
| `configs/train/biencoder_bge_base.yaml` | Was an unused stub; now the proposed plan (§4) | Problems with the old stub listed below. |
| Evaluation harness | Works for the hub models | `BiEncoderRetriever` requires a Hugging Face model ID and a 40-character commit, so it **cannot load a local fine-tuned checkpoint** (§7). |
| Library support | Present | `CachedMultipleNegativesRankingLoss`, `InformationRetrievalEvaluator` and the `no_duplicates` batch sampler exist in sentence-transformers 5.2.3, the version used for the zero-shot runs. |

**Problems in the old config stub:**
- `num_hard_negatives: 4` with `batch_size: 32` at 512 tokens means 160
  passages per step. That is unlikely to fit on a T4.
- Plain batch 32 gives only 31 in-batch negatives.
- `temperature: 0.05` differs from BGE's own fine-tuning setting of 0.02.
- No model revision pin.
- No split, leakage or validation settings.
- 3 epochs with no checkpoint selection.
- An unversioned output directory.

## 2. Corpus findings that shape training

The analysis is read-only: word 5-gram shingles, MinHash candidates within
each company, and exact confirmation.

| Finding | Value | Consequence |
|---|---|---|
| Size | 10,137 passages, 49 filings (7 companies × 7 years; fiscal 2019–2025, Walmart fiscal 2020–2026) | About 15k pairs at 2 queries per usable passage (§4). |
| Passage length | Chunks of 512 words with 80-word overlap. The median passage has about 683 BGE word pieces, and 90.8% exceed the 512-token limit. | The encoder sees roughly the first 380 words. Generate queries only from that part, or pairs will depend on text the model never reads. |
| Financial tables | 4,437 passages (44%) contain a table | Prompt for figure questions with period and unit; tables are where BM25 and the dense models disagree most. |
| Exact duplicates | 66 texts covering 146 passages | Remove before sampling. |
| Near-duplicates (5-gram Jaccard ≥ 0.8) | 1,432 passages (14%) in 497 clusters, all but one pair across filings | Year-to-year copies. |
| Shifted copies | 6,984 passages (69%) share at least half their 5-grams with a passage from another filing | Risk factors and policies repeat yearly, and chunk boundaries shift. A random passage-level split would leak, and in-batch "negatives" would often be copies of the positive. |
| Overlap groups (≥ 50% of 5-grams shared) | 4,326 components, 3,086 of them single passages; the largest has 89 | Small enough to split by group (§3). |
| Adjacent chunks | 9,692 passages share an 80-word overlap with a neighbour in their section | Exclude same-section neighbours as negatives. |
| Development pools | 667 passages pooled for `q_0001`–`q_0010`; 1,895 (18.7%) including their overlap groups | The source blocklist (§3). |
| Development positives | 54 of the 100 have a passage in another filing sharing at least half their text | Blocking the positives alone is not enough; their copies must go too. |
| Development target filings | 14 company-years: Apple 2021 and 2024; Amazon 2023; JPMorgan 2021; Pfizer 2022–23; Tesla 2019–23; Walmart 2024; ExxonMobil 2020 and 2022 | The company and year overlap with training is inherent to a fixed-corpus setup (§3). |

## 3. Data, splits and leakage

### Rules

- **No evaluation question is ever a training example.** That covers
  development questions, future gold questions, their paraphrases, audit
  judgments and development relevance labels. Such material is never a
  training pair, a negative, or a few-shot example in a generation prompt
  (methodology §9). Only two readers may open `data/eval/`:
  - the blocklist builder, which reads only the pool's passage IDs, never
    grades;
  - the leakage checker, which reads only question texts.
- **The source blocklist.** No synthetic query comes from these passages,
  and none is used as a negative:
  - every passage in the `pilot_v3` pools;
  - the overlap group of each of those passages.

  That is 1,895 passages, leaving 8,242. The rule needs no relevance labels.
  It stops a generated query that paraphrases a development question from
  being paired with that question's answer, or with a copy of it from another
  year.
- **Query checks.** Every synthetic query is checked against every evaluation
  question, in `annotation/questions.jsonl` and any later gold questions:
  - lexically, with `check_no_gold_leakage` (word overlap ≥ 0.8);
  - semantically, with zero-shot cosine ≥ 0.9, which is dropped.

  The lexical check alone misses paraphrases.

### Splits

- **Unit of the split: the overlap group.** Every passage sharing at least
  half its 5-grams with another stays on the same side, so year-to-year
  copies never straddle train and validation.
- **Validation share:** 10% of groups, stratified by company, with seed
  20261029. That gives about 820 validation passages (1 query each) and about
  7,400 training passages (2 queries each, about 14,800 pairs).

### Batching

- **The default sampler is not enough.** `no_duplicates` only prevents
  identical texts in a batch.
- **A group-aware sampler is needed.** No two pairs from the same overlap
  group should share a batch; otherwise copies act as false in-batch
  negatives.
- **Fallback.** If the sampler is not built, keep one source passage per
  group, at the cost of about half the data.

### Company and year overlap

The 14 development target filings remain training sources apart from the
blocked passages.

- **Why this is legitimate.** A fine-tuned model is meant to adapt to this
  corpus, so development diagnostics measure in-corpus adaptation, not
  transfer. A gold set over the same corpus has the same property.
- **How to measure it.** An optional check (A1-H) excludes those 14 filings
  from the training sources. This is a measurement only; no model is chosen
  with it.

### Year sensitivity

Filings repeat text across years, and the rubric treats another year's copy
of qualitative text as 0 when a question names a year.

- **What queries should name.** The company and fiscal year, wherever the
  passage states them.
- **Year-swapped copies as negatives.** A later ablation could use another
  year's copy of a passage as the negative for a query that names a year. It
  is valid only where the copies' figures or statements differ, so it is not
  in the first ablations.

## 4. Proposed configuration (first experiment)

See [`configs/train/biencoder_bge_base.yaml`](../configs/train/biencoder_bge_base.yaml).

| Setting | Value | Why |
|---|---|---|
| Base model | `BAAI/bge-base-en-v1.5` at `a5beb1e3…` | The pinned zero-shot baseline |
| Prefixes | BGE query instruction; no passage prefix | Same as evaluation |
| Loss | `CachedMultipleNegativesRankingLoss` | GradCache gives many in-batch negatives within T4 memory |
| Temperature | 0.02 (scale 50) | BGE's published fine-tuning setting |
| Batch | 128 queries per step; mini-batch 16 | 127 negatives per query; memory bounded by the mini-batch |
| Gradient accumulation | 1 | Accumulation does not add in-batch negatives; GradCache does |
| Sequence length | 512 | Same as evaluation |
| Optimiser | AdamW, learning rate 2e-5, warmup 10%, linear decay, weight decay 0.01 | Standard for a BERT-base fine-tune |
| Epochs | 1 (2 only if validation still improves) | About 116 steps per epoch at 14.8k pairs, so overfitting is the main risk |
| Precision | fp16 automatic mixed precision; gradient checkpointing off unless out of memory | The T4 has no bf16 |
| Seeds | 42 and 1337 | Two runs per ablation, to see seed spread |
| Hard negatives (A2) | 1 per pair | See below |

**Hard-negative mining (A2).**
- **What is mined.** The zero-shot model, company mode, ranks 10–50.
  Candidates scoring at least 95% of the positive's score are dropped.
- **What is excluded.** The positive's overlap group, same-section neighbours
  and the blocklist.
- **Step size.** Batch 64, so 128 passages per step, the same as A1.

## 5. GPU memory, batch sizes and compute (T4, 16 GB)

These are estimates. Run a 50-step smoke test first and record
`torch.cuda.max_memory_allocated` and the throughput.

- **Fixed memory.** BGE-base has 110M parameters. Weights, gradients and
  AdamW states take about 2 GB.
- **Activation memory.** It is dominated by 512-token passages. Expect a
  mini-batch of 16 passages to need about 8–10 GB with fp16, against roughly
  15 GB usable.
  - Plain multiple-negatives loss at batch 32 would probably not fit without
    gradient checkpointing.
  - The cached loss keeps memory at the mini-batch size, whatever the batch.
  - Fallbacks: mini-batch 8, or gradient checkpointing (about 20–30% slower).
- **Throughput.** Inference is about 150–250 passages/s. Training with the
  cached loss is about 40–70 passages/s; it adds one extra forward pass, about
  30–40% more time.

| Step | Estimate |
|---|---|
| A0: re-encode the zero-shot baseline in the training environment | 1–2 min |
| A1, 1 epoch (about 14.8k pairs) | 4–7 min training plus about 5 validations (about 1 min each, full-corpus encode) |
| A2, 1 epoch (twice the passages) | 8–14 min plus validations |
| Hard-negative mining | Local CPU is enough: reuse the cached zero-shot embeddings, about 15k short queries, a few minutes |
| Diagnostics on `dev_pilot_v3` (both modes) | About 2 min per model |
| A1 + A2 at 2 seeds, plus A0 | About 1–1.5 GPU hours |
| Optional A3 (E5, 2 seeds) | About 0.5 GPU hours more |

**Where to run.**
- **Kaggle.** A weekly GPU quota and longer sessions, so more predictable.
  Recommended.
- **Colab free tier.** Can lose the GPU or disconnect. Use it as a backup.
- **Either way.** Pin sentence-transformers, transformers and torch, and
  record them with every run.

**Fair comparison.** Re-encode A0 in the same environment as the fine-tuned
models. The existing zero-shot runs were encoded on CPU with different
library versions.

## 6. Checkpoint selection and validation

- **Selection.** Synthetic validation MRR@10, over the full 10,137-passage
  corpus in both modes, every 25 steps. Early stopping after 3 evaluations
  without improvement. Keep the best checkpoint and the last.
- **Scoring.** The source passage is the only positive. Other members of its
  overlap group are removed from the ranking before scoring; they are
  neither confirmed relevant nor non-relevant. Step 0 (zero-shot) is
  evaluated the same way.
- **Development diagnostics.** After selection only, run `dev_pilot_v3` (10
  questions, AI labels) with judged@10 and unjudged counts. No checkpoint,
  hyperparameter or ablation is chosen on them. A fine-tuned model will rank
  passages the pools never saw, so its judged@10 will fall:
  - report the judged-only scores beside it;
  - never deepen the gold pools with its output (methodology §4, §7).
- **Final claims.** These wait for human gold labels.
- **Provenance per run.** Base revision, training-config SHA-256, train and
  validation file SHA-256, seed, library versions, GPU, steps, the validation
  curve, and the SHA-256 of `model.safetensors`.
- **Storage.** Checkpoints (about 440 MB) live in `models/`, which is
  gitignored, or in the platform's storage. They are never committed and
  never overwritten.

## 7. First ablations

Held constant across ablations: data, splits, steps, learning rate,
sequence length, evaluation and seeds.

| ID | Model | Training data | Negatives | Question it answers |
|---|---|---|---|---|
| A0 | BGE-base zero-shot | none | — | Baseline |
| A1 | BGE-base fine-tuned | Synthetic pairs | In-batch only (127 per query) | Does domain fine-tuning help? |
| A2 | BGE-base fine-tuned | Same pairs | In-batch plus 1 mined hard negative | Do hard negatives add to A1? |
| A3 (optional) | E5-base fine-tuned (`f52bf8ec…`, `query: `/`passage: `, temperature 0.01 to be confirmed) | Same pairs | As A1 (A2 only if A2 helped) | Does the effect carry over to another base model? |
| A1-H (optional) | As A1 | Without the 14 development target filings | As A1 | How much of the diagnostic gain comes from training on those filings? |

**Decision rule, fixed before any run.**
- **A1 beats A0** if synthetic validation MRR@10 improves by more than the
  spread between A1's two seeds.
- **A2 is kept** only if it beats A1 by the same rule.
- **Development diagnostics** are reported for every model in both modes.
  They do not decide.

## 8. Remaining blockers (in order)

1. **Synthetic queries: pipeline built, real run pending.** The pipeline is
   in [training_data_v1.md](training_data_v1.md): overlap groups, blocklist,
   group split, visible text, a resumable Groq generator, quality checks and
   manifests. `prepare` has run: 2,110 passages blocked, 7,200 train and 827
   validation.
   - **Still needed:** real generation with an API key and explicit
     confirmation, a pilot run read by a person, and a semantic leakage check.
   - **Corrected estimate:** the blocklist is larger than §2 estimated (2,110,
     not 1,895), because the overlap grouping now also links cross-company
     boilerplate.
2. **Hard-negative mining.** Not built yet (needed for A2).
3. **Training code.** The trainer, miner, seed utility and
   `05_train_biencoder.py`: config loading, the cached loss, the group-aware
   sampler, the validation evaluator and provenance.
4. **Evaluating a local checkpoint.** The benchmark harness needs a local
   model path with the weights' SHA-256 in place of the hub revision, both in
   the retriever and in the embedding-cache key.
5. **The human audit of `pilot_v3`** (`data/eval/audit/pilot_v3`, 20 items)
   and decisions on the flagged conventions
   ([flag_review_pilot_v3.md](flag_review_pilot_v3.md)), before development
   diagnostics are trusted.
6. **No human gold set.** Results stay diagnostic until one exists.
