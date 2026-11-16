# Figure matching, training candidates and the bi-encoder trainer (v1)

This step fixes the figure leak found by the second pilot audit
(docs/training_pilot_v3_quality_report.md, §6) and builds a versioned set of
training candidates from the AI quality decisions. It also adds a testable
contrastive trainer for BGE-base.

- No model was trained and no API request was sent.
- The tests download nothing from Hugging Face.
- The frozen corpus, the evaluation data and the historical generation records
  are unchanged.

## 1. Figure matching in the evidence check (`evidence-reuse-v2`)

**The bug.**
- The per-pair evidence locator wrote figures as text values with the scale
  attached. An answer "$57,411 million" became `57411m`, but the table cell
  "57,411" stayed `57411`.
- So the answer's figure was never recognised in its own evidence.
- A blocked or other-split passage holding the same labelled cell counted only
  as an uncertain match, which never rejects a pair.

**The rules now** (`finretriever.training.evidence_reuse`: `Figure`,
`figures`, `same_amount`, `refers_to`, `declared_scale`):
- **Normalisation.** Currency signs, separators and spacing are dropped.
  "57,411", "$57,411", "$ 57,411", "$57,411 million", "57,411 millions",
  "$57.411 billion" and "$57,411M" state one amount.
- **Units.** A written unit (thousand, million, billion, trillion, plural,
  "bn", or a letter glued to a dollar figure) scales the digits. So does the
  one unit a passage declares: "(In millions)", "(dollars in thousands)" or
  "(MILLIONS OF DOLLARS)".
  - A declaration that qualifies counts ("shares … in thousands"), or a
    passage declaring several units, leaves the unit unknown.
  - A figure of unknown unit takes whichever unit makes it equal. Two figures
    of unknown unit need the same digits.
- **No near-equality.** 57,411 thousand is not 57,411 million, and 57.4
  billion is not 57,411 million: a rounding is a different amount.
- **Kept apart:**
  - signs: a minus or accounting parentheses, "(16,117)" against "16,117";
  - percentages, including "percent" and basis points, which never equal an
    amount;
  - dates and bare years, which are never figures.
- **Exact matches need the label.** A passage holds a pair's evidence exactly
  ("fact") only if it shows the evidence figure the answer states, as the same
  amount with the same sign, under the evidence's label.
- **Uncertain matches.** These keep a passage out of negatives but never reject
  a pair or add a positive:
  - the same digits under a coincidental label ("value");
  - the same label with another sign, or a conflicting declared unit
    ("other_fact").
- **Answer to evidence.** An answer refers to the evidence cell as written. An
  unsigned answer may state a negative cell in words ("a loss of $1,403
  million"); a negative answer needs a negative cell.

**Unchanged:**
- The reuse index's fact units and digest. The `bge_base_ft_v2` preparation
  still opens, and its recorded digest is re-verified on every load.
- The preparation's reuse links and blocklist. They still key figures as
  written, so "$14.6 billion" in one passage and "14,600" in another do not
  link them for the split. Only the per-pair check compares amounts.

**Versioning.** The equivalence method is now `evidence-reuse-v2`
(`EvidenceLocator.version`, `pair_contract.EQUIVALENCE_METHOD`).
- A pair whose equivalents were found by v1 is stale (`contract_problems`), so
  `check` and `load_training_pairs` refuse the existing `pilot_v3` pairs file.
- That file is not rebuilt or edited. The candidate dataset (§2) re-checks its
  pairs instead.

**Regression tests** (`tests/unit/test_figure_matching.py`, 13 tests):
- Each of the six audit cases is rebuilt from short excerpts of the filings.
  The test asserts a `fact` match in the other passage and the expected
  rejection, with and without the scale word in the answer:
  - 002 and 052: `evaluation_evidence`;
  - 006, 007, 034 and 035: `cross_split_evidence`.
- Counter-cases:
  - the same digits under another label give an uncertain match;
  - a "(dollars in thousands)" cell, a rounded "$57.4 billion" or a negative
    cell is not exact;
  - "$57.411 billion" is exact.
- Unit tests cover normalisation, units, signs, percentages, dates and
  declarations.

**Offline revalidation of every pilot pair** (real corpus, the
`bge_base_ft_v2` preparation):

| Pairs | Outcome under v2 |
|---|---|
| `pilot_v3`, 70 pairs | Exactly the six held pairs now fail. 002 and 052 are found in blocked passages; 006, 007, 034 and 035 in a validation passage's visible text. The other 64 still pass. |
| `pilot_v3`, equivalents | 10 of the 64 change. Four gain exact equivalents: 003 (+2, the same depreciation row in FY2019 and FY2021), 023 (+2), 037 (+1) and 065 (+1). Six gain uncertain ones (012, 021, 029, 041, 058, 059), for example a coincidental "1,403" in a derivatives table. |
| `pilot_v2`, 37 pairs | No eligibility changes: the same 15 are eligible. Pair 032, already ineligible because its positive is blocked, now also shows an evaluation-evidence leak. |

**Still missed:**
- rounded figures;
- a figure under another label;
- three-digit table cells: a figure without a written unit still needs four
  digits to count;
- rows whose label changed between filings;
- paraphrased evidence.

## 2. Training candidates (`train_candidates_v1`)

```
python scripts/02_generate_queries.py build-candidates --config configs/train/data_prep_v2.yaml \
    --candidates configs/train/candidates_v1.yaml
```

`finretriever.training.candidates` reads two sources:
- the 70 pairs `bge_base_ft_v2` built from the second pilot, with
  `data/train_review/pilot_v3_quality_decisions.jsonl`;
- the 37 pairs of `bge_base_ft_v1`, with
  `data/train_review/pilot_v2_quality_decisions.jsonl`.

A pair is included only if every check passes:

| Check | Rule |
|---|---|
| Provenance | The pair file matches its manifest. The generation run, continuations and records still have their recorded SHA-256 (read, never written). The pair's question, type, answer and evidence are an accepted candidate of its passage's latest record, under the parameters and prompt it names. |
| Quality | One decision, bound to the passage and question SHA-256, says `accept`. `revise`, `reject` and a missing decision exclude the pair. |
| Validator | `sec-qcheck-v3` passes, and the positive is still in its split. |
| Evidence | No blocked passage and no other-split visible text holds the evidence (`evidence-reuse-v2`). Equivalents are recomputed. |
| Leakage, duplicates | No lexical overlap with an evaluation question, and no repeated query. |

Output, written once to
`data/processed/train/bge_base_ft_v2/candidates/train_candidates_v1/`:
- `train_candidates.jsonl`: contract records keeping their generation
  provenance;
- `validation_candidates.jsonl`: empty;
- `candidates_log.jsonl`: all 107 source pairs, each included or excluded with
  its reasons, quality decision and leak passages, plus its v1 and v2
  equivalent counts;
- `manifest.json` (SHA-256 `6fb3aa72…`): every input with its SHA-256, and
  the counts below.

| Source | Pairs | Included | Excluded by reason (a pair can have several) |
|---|---|---|---|
| `pilot_v3` | 70 | 51 | revise 7, reject 6, cross-split evidence 4, evaluation evidence 2 |
| `pilot_v2` | 37 | 15 | evaluation evidence 17, positive now blocked 15, revise 3, reject 1 |
| **Total** | 107 | **66** (all train) | |

| Company | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---|---|---|---|---|---|---|
| Candidates | 10 | 8 | 13 | 8 | 11 | 7 | 9 |

JPMorgan has the largest share, at 19.7%.

| Fiscal year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| Candidates | 10 | 10 | 9 | 6 | 9 | 11 | 10 | 1 |

| Type | financial_table | other | risk | debt_liquidity | mdna | accounting_policy | profitability | revenue | tax |
|---|---|---|---|---|---|---|---|---|---|
| Candidates | 24 | 12 | 11 | 6 | 5 | 4 | 2 | 1 | 1 |

- Passage kind: 38 tables and 28 narrative passages.
- 35 candidates have exact equivalents, which are extra positives and never
  negatives. 30 have uncertain ones, which are never negatives either.

These are synthetic queries accepted by an AI review. They are not
human-validated and not gold labels.

**Loading for training** (`load_candidates`). The loader fails closed unless:
- every file matches the manifest;
- the preparation, corpus and evidence index are the ones recorded;
- the method and validator are current;
- the source pair files and decision files are unchanged;
- every pair passes `training_data.verify_pairs` again.

It reads no evaluation file. The lexical check ran at build time and is
bound by the manifest. On the real data, all 66 verify in about 17 s.

## 3. The trainer

`finretriever.training.trainer_setup` holds the configuration, inputs and
schedule, and does not need torch. `finretriever.training.biencoder_trainer`
holds the loss, the loop and the checkpoints. The entry point is
`scripts/05_train_biencoder.py`.

**Model.**
- `BAAI/bge-base-en-v1.5` at revision
  `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, on CPU or CUDA.
- CLS pooling, then L2 normalisation, as BGE's model card specifies.
- Queries carry BGE's instruction, "Represent this sentence for searching
  relevant passages: "; passages take no prefix. The rules come from
  `retrieval.biencoder.resolve_format`, so they cannot drift from evaluation.
- Passages are truncated at the preparation's visible length (512 tokens). A
  query longer than `query_max_length` is an error, never truncated.
- Before the first step, `check_lengths` measures every query and positive
  passage and prints the result (for the smoke subset: longest query 55
  tokens; 28 of 37 positives longer than 512 tokens, the longest 1,044).
  It also checks that truncation keeps each pair's visible text, where the
  validator found its evidence. A batch wider than its limit never reaches the
  model.

**Objective.**
- In-batch InfoNCE: `mean_i -log softmax_j(q_i·p_j / t)_i`. This is
  sentence-transformers' MultipleNegativesRankingLoss with scale `1/t`;
  t = 0.02 gives scale 50, BGE's fine-tuning setting.
- Each query has `batch_size - 1` negatives, all from its own batch.
- `gradient_accumulation_steps` averages the gradients of several batches per
  optimizer step. It does not add negatives, and the metadata says so.

**False negatives.**
- Batches come from `plan_batches`, re-seeded each epoch with `seed + epoch`.
- Every batch is checked again before its forward pass. No in-batch passage
  may be another query's exact or uncertain equivalent, overlap sibling or
  reuse link, or a blocked or other-split passage.
- Single-pair batches have no negative and are dropped; the drop is recorded.

**Optimisation.**
- AdamW, with no weight decay on biases and norms.
- Linear warmup, then linear decay. No step is taken at a learning rate of 0.
- Gradient clipping; fp16 autocast with a GradScaler on CUDA only.
- Optional gradient checkpointing.

**Fails closed** (exit code 1 or 2, nothing trained) when:
- the preparation's visible text was defined for another tokenizer, revision,
  length or passage prefix;
- the loaded tokenizer's SHA-256 differs from the preparation's (checked
  against the cached BGE tokenizer: identical, `9114256b…`);
- truncating a positive passage would cut into its pair's visible text, so the
  tokenizer does not truncate as the preparation did;
- the candidate manifest, files, sources or decisions changed, or the
  candidates are stale;
- fewer than two pairs remain;
- fp16 is set on CPU;
- any configured path lies under `data/eval/`;
- the config has an unknown key, so no evaluation set can be configured for
  checkpoint selection;
- a run directory holds a different run.

**Checkpoints** go to `<output_dir>/<name>-seed<seed>/`.
- `run.json` binds the directory to the configuration, inputs and schedule.
- Each `checkpoint-step-NNNNNN/` holds:
  - the model and tokenizer, in Hugging Face format;
  - the sentence-transformers module files, with CLS pooling and
    normalisation;
  - `metadata.json`, recording the data and objective bindings, the loss
    history, library versions, peak GPU memory (measured, on CUDA only), and
    `selection: none`.
- The newest checkpoint also holds `trainer_state.pt` (optimizer, scheduler,
  scaler and random states). `--resume` continues from it, and the result
  matches an uninterrupted run exactly on CPU.

**No evaluation.**
- The trainer reads no evaluation file and computes no metric.
- Checkpoints are kept at fixed steps. Nothing selects among them, and
  pilot_v3 development metrics are never consulted.

## 4. Tests

`tests/unit/test_biencoder_trainer.py` has 28 tests. All run on CPU, and 27
need only torch:
- **Loss:** it matches a manual InfoNCE, is invariant to embedding scale, and
  is lower when aligned than when shuffled; one pair is refused.
- **Gradients and updates:** gradients are finite and non-zero; the loss falls
  on a fixed batch; a run moves the weights; warmup and decay are recorded.
- **Batches:** planned batches never hold an excluded passage. A batch with a
  blocked passage or a known equivalent is refused before any forward pass.
- **Checkpoints:** names are deterministic; the newest checkpoint alone keeps
  the state; reloaded weights encode identically. A run stopped at step 2 and
  resumed gives the uninterrupted run's losses and weights exactly. Writing
  over a run, or resuming under other settings, is refused.
- **Seeds:** the same seed gives identical weights; another seed gives other
  weights.
- **Fail-closed inputs:**
  - a preparation for another encoder or length;
  - a missing or empty candidate dataset;
  - a tampered manifest, candidate file or decision file;
  - a stale v1 record;
  - another tokenizer, or an over-long query;
  - nine unsafe configurations.
- **No evaluation access:** a spy on `open` sees the corpus, preparation and
  candidates, and no evaluation input.
- **Command line:** a dry run loads no model and writes nothing.
- **Hugging Face path:** a tiny random BERT, built locally with no download, is
  saved, reloaded and loaded by sentence-transformers with identical
  normalised embeddings.
  - It is skipped under this Mac's global Python, because scipy is broken
    there. It passes in `~/venvs/finretriever-dense`.

CI now installs CPU-only PyTorch (`requirements-ci.txt`), so these tests run
there too. transformers is not installed in CI, so the Hugging Face test is
skipped there.

## 5. GPU smoke test

`configs/train/biencoder_bge_base_smoke.yaml`:
- 48 of the 66 candidates (subset seed 13);
- batch 16, so 15 in-batch negatives per query, with no accumulation;
- learning rate 2e-5, warmup 10%, weight decay 0.01, t = 0.02;
- query maximum 128 tokens and passage 512;
- fp16 on CUDA, seed 42, 15 steps, checkpoints at steps 5, 10 and 15.

Dry run on the real data (offline; no model loaded, nothing written):

```
run bge_base_ft_v2_smoke-seed42: 48 training candidates from train_candidates_v1 (66 available)
15 optimizer steps (warmup 2); batches per epoch [3, 4, 4, 3, 4]; 15 in-batch negatives per query; 1 single-pair batches dropped
every batch verified: no known equivalent, overlap sibling, reuse link, blocked or other-split passage is an in-batch negative
ESTIMATED peak GPU memory (not measured): {'weights_gradients_optimizer_gib': 1.63, 'activations_gib': 5.98, 'overhead_gib': 1.0, 'total_gib': 8.61}
```

**Command, on the T4.** The machine needs the repository, plus:
- `data/processed/corpus_v1.jsonl` and its manifest;
- `data/processed/train/bge_base_ft_v2/`;
- `data/processed/train/bge_base_ft_v1/pairs/`;
- the two decision files in `data/train_review/`.

The `data/processed/` files are gitignored, and `data/train_review/` is not
committed yet. The candidate loader re-verifies the source pair files and
the decision files. The base model downloads from Hugging Face at its pinned
revision on that machine.

```
pip install -r requirements.txt
python scripts/05_train_biencoder.py --config configs/train/biencoder_bge_base_smoke.yaml              # dry run
python scripts/05_train_biencoder.py --config configs/train/biencoder_bge_base_smoke.yaml --confirm-training
```

**GPU memory: estimates, not measurements**
(`trainer_setup.estimate_gpu_memory_gib`):
- BERT-base in fp16; AdamW in fp32 at 16 bytes per parameter;
- activations from Korthikanti et al. (2022), plus 1 GiB of overhead;
- memory-efficient attention lowers the real figure, so these are upper
  estimates.

| Batch | Gradient checkpointing | Estimated peak (GiB) | T4, 15 GiB usable |
|---|---|---|---|
| 16 | off | 8.6 | fits |
| 16 | on | 3.3 | fits |
| 32 | off | 14.6 | too tight |
| 32 | on | 4.0 | fits |
| 64 | on | 5.3 | fits |

On CUDA, the first real measurement is written to each checkpoint's
`metadata.json` (`gpu_memory.max_allocated_gib`).

**Running it on Kaggle.** docs/kaggle_smoke_test.md has the portable bundle
(`scripts/smoke_bundle.py`), the preflight (`scripts/smoke_preflight.py`) and
the results collector (`scripts/smoke_results.py`).

**What the smoke test can show:**
- the pipeline runs on GPU;
- the loss is finite and recorded;
- checkpoints are written, and they resume and load.

48 pairs and 15 steps cannot show domain adaptation. No retrieval result
should be read from this run, and no checkpoint should be chosen by any
metric.

**The first T4 run's tokenizer warning.** Its console showed transformers'
"Token indices sequence length is longer than the specified maximum sequence
length for this model (739 > 512)".
- **Source:** the trainer counting tokens of every untruncated positive
  passage, to record how many are truncated. Transformers warns once, at the
  first passage over 512 tokens: the positive of `pilot_v3:train_000003`
  (AAPL FY2020, Item 8), 739 tokens.
- **No model input:** every forward pass truncates at 512 tokens (queries at
  128). A 739-token input would have overflowed BERT's 512 position
  embeddings, and the run completed.
- **Evidence kept:** for all 66 candidates the evidence lies in what the model
  reads.
- **Fix:** the measurement is now silent (`verbose=False`, as
  `retrieval.biencoder` measures), and the trainer prints the lengths itself.
- **Reporting:** `smoke_results.py` had missed the warning, because it searched
  for keywords this message lacks. It now reports every library line in the
  console.

## 6. Blockers and open items

**Update, 2026-11-04.** GradCache, seeds and validation-based checkpoint selection on `validation_pairs_v4` are now
implemented ([training_full_v1.md](training_full_v1.md)). Local checkpoints are evaluated in the benchmark
([checkpoint_evaluation.md](checkpoint_evaluation.md)), and only a run's validation-selected checkpoints can be final
models. Hard-negative mining is still a stub. The list below is kept as it was written.

- **Too few validation pairs to choose checkpoints.**
  - The training plan selects checkpoints on synthetic validation MRR@10,
    so the trainer saves fixed steps only.
  - The validation pilot `val_pilot_v1` gave a separate set,
    `validation_pairs_v1`, of 16 questions
    ([validation_pilot_v1_quality_report.md](validation_pilot_v1_quality_report.md)).
    It is enough to test an evaluator, but not to rank checkpoints.
  - The trainer never reads it: it lives under `validation_sets/`, not
    `candidates/`.
  - A set large enough needed a decision on the validation split (§10 of
    that report).
  - That decision is implemented in `bge_base_ft_v3`
    ([training_data_v3.md](training_data_v3.md)). It holds the same 66
    candidates (`train_candidates_v2`), the 16 questions
    (`validation_pairs_v2`) and a 185-passage plan for about 106 more,
    not yet generated.
  - The trainer configs still point at `bge_base_ft_v2`.
- **Not implemented yet:**
  - GradCache, the cached loss the plan uses for batch 128;
  - hard negatives: `negative_mining` is still a stub, and the A2 ablation
    depends on it.
- **Evaluating a fine-tuned checkpoint.** `04_run_baselines.py` loads models
  by hub name and a 40-character revision. A local checkpoint path is not
  wired in yet. The checkpoint files are sentence-transformers compatible
  (tested).
- **Generator and validator weaknesses** from the audit remain, for prompt
  `sec-qgen-v3`:
  - tables without visible headings;
  - stub and cross-reference passages;
  - non-answers passing the validator.
- **Data size.** 66 AI-reviewed candidates are enough for a mechanics check
  only.
