# Retrieval Performance Analysis v1

> **Status: analysis of existing systems, 2026-11-15. Not a benchmark.** No human relevance label exists for any
> number here. Gold set v1 closed as an incomplete pilot ([closure](gold_set_v1_closure.md)); it was not reopened,
> and none of its grades were read. Validation scores are in-distribution and chose the A1 checkpoints;
> development labels are AI-written and provisional; the real-question results are a metadata diagnostic, not
> relevance. No model was trained, no Kaggle session was run, and no frozen artifact was changed.

## 1. Findings

**No system is best on every measure.** Each evidence source answers a different question, and none of them is
independent, human-labelled evidence of retrieval quality.

| Evidence | Strongest | Weakest | Can it separate the systems? |
|---|---|---|---|
| Validation, 93 synthetic questions (chose A1) | A1 (three seeds), statistically tied with BM25 | BGE zero-shot = A0 | Partly: A1 beats E5 and E5 beats A0/BGE (p ≤ 0.0011); A1 vs BM25 is not separable (p ≥ 0.43) |
| Development, 10 AI-labelled questions | No winner: E5 has the best nDCG@10, A1 seed 2026 the best MRR@10 | BM25 | No: every tested MRR@10 difference has an interval that includes zero |
| 100 real questions, company/year alignment | E5 (and A1 when restricted to the company) | BM25 in global mode; BGE/A0 on fiscal year | Shows *where* results come from, not whether they are relevant |

1. **Fine-tuning (A1) clearly beat its own base model on the questions that selected it.** Validation MRR@10 rose
   from 0.266 (A0) to 0.658–0.669 (global) in all three seeds (p = 0.0001), and Recall@100 reached 0.997–1.000.
   But **BM25 reaches 0.635** on the same questions: the synthetic questions reuse their passage's wording, and A1's
   gain over BM25 (+0.023 to +0.034 in global mode) is not distinguishable from zero.
2. **On the development set A1 is not distinguishable from A0 or E5** (MRR@10 differences −0.042 to +0.142 across
   seeds and modes, every p ≥ 0.31). A1 does bring more judged-relevant passages into the top 10 and top 100 than
   A0 (Recall@10 +0.051 to +0.121), but 15–20% of A1's top 10 is unjudged and counted non-relevant, so its
   development scores are lower bounds.
3. **A1 is more year-aware but less company-aware than its base model.** On the 100 real questions (global mode),
   A1's top 10 holds more passages from the asked fiscal year (20.5–23.1% vs 16.8% for A0) but three times as many
   from other companies (17.0–17.8% vs 5.4%); its top 10 is mostly another company's for 14–15 of the 100 questions
   (A0: 1). The validation set shows the same shift (top-1 wrong company: 12–15 of 93 vs 6).
4. **Fiscal year is the dominant unsolved problem.** Even inside the right company, BGE/A0 and BM25 place passages
   from the asked year at the rate a year-blind ranker would (16.3–17.8% vs 17.6% chance); the best systems (E5,
   A1) reach only 25.5–28.2%. Between 42.1% and 58.5% of each system's company-mode top 10 comes from years outside
   the asked year and its comparative window.
5. **Multi-scope questions are poorly covered.** For the 10 cross-company questions, no system's top 10 holds
   passages for every company-year cell in more than 1 of 10; for multi-passage questions, all cells are covered in
   33–53%.

The most important remaining problems, the evidence for each, and the recommended next step are in sections 7–8.
The single highest-value next improvement is **metadata-aware retrieval** (company and fiscal-year scoping with the
comparative window, plus a company/year header on every passage), evaluated before any further fine-tuning.

## 2. What was compared, and on what

**Seven systems**, each defined exactly as in the frozen `configs/eval/benchmark.yaml` (gold_v1 entries), copied into
`configs/eval/retrieval_analysis_v1.yaml`:

| System | What it is |
|---|---|
| BM25 | Lexical baseline: word tokens (figures such as `1,200` kept whole), pure-Python BM25 |
| BGE zero-shot | `BAAI/bge-base-en-v1.5` (pinned revision), query instruction, CLS pooling, 512-token passages |
| E5 zero-shot | `intfloat/e5-base-v2` (pinned revision), `query:` / `passage:` prefixes, 512-token passages |
| A0 (step 0) | The trainer's step-0 checkpoint: BGE's base weights saved by the trainer, loaded through the checkpoint pipeline (identical in all three seeds, `0d4db737…`) |
| A1 seed 42 / 1337 / 2026 | The three official fine-tuned checkpoints (steps 55, 40 and 40), each with its selection record verified |

**Two retrieval modes:** `global` ranks all 10,137 passages; `company` ranks only the passages of the question's
companies, for every system alike.

**Three evidence sources, none of them a benchmark:**

| Source | Questions | Relevance | What it can and cannot show |
|---|---:|---|---|
| Validation set `validation_pairs_v4` | 93 | One positive passage plus its exact equivalents per question; uncertain equivalents removed from the ranking (as in checkpoint selection) | Synthetic questions written by an LLM from the passages, accepted by an AI review, from the same distribution as A1's training data. **They chose each A1 checkpoint.** A1's scores on them are in-distribution and selection-biased; they are not independent evidence |
| Development set `dev_pilot_v3` | 10 | AI-written graded labels (0/1/2), checked only by an AI-assisted audit | Provisional. The judged pools hold the zero-shot systems' top 10s, not A0's or A1's: their unjudged passages count as non-relevant, which biases them downwards (reported as judged@10). Ten questions give very wide intervals |
| The 100 frozen human-written questions (gold set v1, export v2) | 100 | **None read.** Gold set v1 is closed and incomplete | A metadata alignment diagnostic only: whether each system's top 10 comes from the question's company and fiscal year. It measures a necessary condition, not relevance |

## 3. Integrity checks before any number

- **Archives:** the three Kaggle output archives match their recorded SHA-256 (seed 42 `4a3b9d4a…`, seed 1337
  `7f0317af…`, seed 2026 `7e79fa35…`). Earlier the same day, the three duplicate seed 42 downloads were byte-identical
  to the original; they have since been removed from `~/Downloads` (outside this analysis). No original archive was
  deleted or overwritten.
- **Checkpoints:** only A0 and the three selected checkpoints were unpacked (read-only, `trainer_state.pt` skipped),
  into `models/bge_base_ft_v3_a1/` (git-ignored), the paths the frozen benchmark config names. Every file matches the
  run's `OUTPUT_SHA256SUMS`, and every checkpoint file matches the hashes in the run's `validation_log.jsonl`. The
  selected checkpoints' digests are the official ones (`24c2aaae…`, `8237ec0b…`, `cba9110b…`).
- **Inputs:** corpus `ff9dadf3…`, question export v2 `fbcf2f5a…`, validation manifest `40238cb0…`, training
  candidates `0586c7ca…` (loaded and isolation-checked by the trainer's own setup code).
- **Evaluation isolation (development set):** none of the 667 judged development passages is among A1's training or
  validation passages; the highest training-question similarity to a development question is 0.346 (threshold 0.8).
- **Frozen code and data:** `benchmark.yaml`, the gold-set decision code and every frozen dataset are unchanged; no
  model was trained, no Kaggle session was run.
- **Stage A reproduces on this Mac** (table T3): the recorded per-question first-relevant ranks are identical for
  89–93 of 93 questions in every checkpoint and mode; every difference is a shift of one rank (CPU here vs the
  Kaggle T4 GPU), and the largest aggregate difference is 0.0054 MRR@10 (one question at rank 2 instead of 1).
- **A0 is the base model.** A0's rankings are identical to BGE zero-shot's (top-10 Jaccard 1.000, same top 1 on all
  93 + 100 questions; identical development scores), confirming the step-0 checkpoint is the unmodified base.
- **Development scores reproduce the harness.** Every development number here is recomputed from the stored
  rankings with the harness's own metric code and matches the harness's recorded aggregate to within 2.2e-16
  (column "Max abs Δ vs harness record" in T4).
- **The analysis is deterministic.** It was run twice; the per-question outputs (`validation_per_question.jsonl`,
  `alignment_per_question.jsonl`, `top10.jsonl`) are byte-identical between runs.

**Table T3.** Reproduction of the recorded Stage A validation scores (Kaggle T4 → this Mac, CPU)

| System | Mode | MRR@10 recorded | MRR@10 reproduced | abs Δ MRR@10 | abs Δ nDCG@10 | First relevant rank identical |
|---|---|---|---|---|---|---|
| A0 (step 0) | global | 0.265886 | 0.265694 | 1.92e-04 | 1.92e-04 | 89/93 |
| A0 (step 0) | company | 0.282433 | 0.282241 | 1.92e-04 | 1.92e-04 | 90/93 |
| A1 seed 42 | global | 0.669039 | 0.668501 | 5.38e-04 | 2.89e-04 | 91/93 |
| A1 seed 42 | company | 0.737302 | 0.731925 | 5.38e-03 | 2.43e-03 | 92/93 |
| A1 seed 1337 | global | 0.667273 | 0.668169 | 8.96e-04 | 1.32e-03 | 92/93 |
| A1 seed 1337 | company | 0.737993 | 0.737993 | 0.00e+00 | 2.22e-16 | 93/93 |
| A1 seed 2026 | global | 0.658001 | 0.658150 | 1.49e-04 | 1.55e-04 | 91/93 |
| A1 seed 2026 | company | 0.716735 | 0.716735 | 0.00e+00 | 0.00e+00 | 92/93 |


## 4. Seven-system comparison

### 4.1 Validation: 93 synthetic questions (in-distribution; they chose A1)

Scored exactly as checkpoint selection scored them: relevant = the positive passage plus its exact equivalents;
uncertain equivalents (for example the same table in a later filing) are removed from the ranking. 95% intervals
are question bootstraps.

**Table T1-global.** Validation (93 synthetic questions), global mode

| System | MRR@10 [95% CI] | Recall@10 [95% CI] | nDCG@10 [95% CI] | Recall@100 | Relevant in top 10 | Top 1 relevant |
|---|---|---|---|---|---|---|
| BM25 | 0.635 [0.551, 0.716] | 0.824 [0.746, 0.896] | 0.677 [0.599, 0.752] | 0.970 | 77/93 | 49/93 |
| BGE zero-shot | 0.266 [0.194, 0.340] | 0.460 [0.362, 0.555] | 0.306 [0.233, 0.379] | 0.837 | 46/93 | 16/93 |
| E5 zero-shot | 0.402 [0.321, 0.484] | 0.607 [0.512, 0.699] | 0.442 [0.363, 0.521] | 0.867 | 62/93 | 27/93 |
| A0 (step 0) | 0.266 [0.194, 0.340] | 0.460 [0.362, 0.555] | 0.306 [0.233, 0.379] | 0.837 | 46/93 | 16/93 |
| A1 seed 42 | 0.669 [0.588, 0.747] | 0.901 [0.841, 0.953] | 0.717 [0.646, 0.785] | 1.000 | 86/93 | 52/93 |
| A1 seed 1337 | 0.668 [0.587, 0.748] | 0.874 [0.806, 0.931] | 0.705 [0.634, 0.774] | 0.997 | 84/93 | 52/93 |
| A1 seed 2026 | 0.658 [0.580, 0.737] | 0.906 [0.846, 0.957] | 0.712 [0.644, 0.779] | 1.000 | 86/93 | 50/93 |

**Table T1-company.** Validation (93 synthetic questions), company mode

| System | MRR@10 [95% CI] | Recall@10 [95% CI] | nDCG@10 [95% CI] | Recall@100 | Relevant in top 10 | Top 1 relevant |
|---|---|---|---|---|---|---|
| BM25 | 0.703 [0.624, 0.778] | 0.878 [0.806, 0.939] | 0.741 [0.669, 0.809] | 0.973 | 82/93 | 55/93 |
| BGE zero-shot | 0.282 [0.210, 0.356] | 0.514 [0.418, 0.608] | 0.329 [0.258, 0.401] | 0.869 | 52/93 | 16/93 |
| E5 zero-shot | 0.433 [0.350, 0.517] | 0.623 [0.530, 0.713] | 0.469 [0.388, 0.550] | 0.921 | 64/93 | 30/93 |
| A0 (step 0) | 0.282 [0.210, 0.356] | 0.514 [0.418, 0.608] | 0.329 [0.258, 0.401] | 0.869 | 52/93 | 16/93 |
| A1 seed 42 | 0.732 [0.655, 0.806] | 0.920 [0.865, 0.968] | 0.773 [0.705, 0.834] | 1.000 | 87/93 | 59/93 |
| A1 seed 1337 | 0.738 [0.662, 0.811] | 0.909 [0.851, 0.960] | 0.769 [0.703, 0.832] | 0.997 | 86/93 | 59/93 |
| A1 seed 2026 | 0.717 [0.642, 0.789] | 0.947 [0.901, 0.987] | 0.768 [0.706, 0.827] | 1.000 | 89/93 | 56/93 |

**Table T2.** Validation, paired MRR@10 differences (question bootstrap; sign-flip test)

| Comparison | Mode | Mean difference [95% CI] | Wins/ties/losses | p |
|---|---|---|---|---|
| A1 seed 42 − A0 (step 0) | global | +0.403 [+0.323, +0.485] | 66/23/4 | 0.0001 |
| A1 seed 42 − A0 (step 0) | company | +0.450 [+0.367, +0.534] | 67/22/4 | 0.0001 |
| A1 seed 1337 − A0 (step 0) | global | +0.402 [+0.323, +0.484] | 65/25/3 | 0.0001 |
| A1 seed 1337 − A0 (step 0) | company | +0.456 [+0.375, +0.538] | 67/23/3 | 0.0001 |
| A1 seed 2026 − A0 (step 0) | global | +0.392 [+0.307, +0.480] | 66/21/6 | 0.0001 |
| A1 seed 2026 − A0 (step 0) | company | +0.434 [+0.351, +0.520] | 69/20/4 | 0.0001 |
| BM25 − A0 (step 0) | global | +0.369 [+0.270, +0.467] | 55/26/12 | 0.0001 |
| BM25 − A0 (step 0) | company | +0.421 [+0.325, +0.517] | 62/22/9 | 0.0001 |
| BGE zero-shot − A0 (step 0) | global | +0.000 [+0.000, +0.000] | 0/93/0 | 1.0000 |
| BGE zero-shot − A0 (step 0) | company | +0.000 [+0.000, +0.000] | 0/93/0 | 1.0000 |
| E5 zero-shot − A0 (step 0) | global | +0.136 [+0.059, +0.213] | 41/40/12 | 0.0011 |
| E5 zero-shot − A0 (step 0) | company | +0.151 [+0.067, +0.234] | 44/34/15 | 0.0008 |
| A1 seed 42 − E5 zero-shot | global | +0.266 [+0.182, +0.351] | 55/31/7 | 0.0001 |
| A1 seed 42 − E5 zero-shot | company | +0.299 [+0.215, +0.383] | 54/34/5 | 0.0001 |
| A1 seed 1337 − E5 zero-shot | global | +0.266 [+0.178, +0.354] | 49/35/9 | 0.0001 |
| A1 seed 1337 − E5 zero-shot | company | +0.305 [+0.217, +0.393] | 50/36/7 | 0.0001 |
| A1 seed 2026 − E5 zero-shot | global | +0.256 [+0.172, +0.341] | 53/32/8 | 0.0001 |
| A1 seed 2026 − E5 zero-shot | company | +0.284 [+0.202, +0.367] | 53/33/7 | 0.0001 |
| A1 seed 42 − BM25 | global | +0.034 [-0.064, +0.132] | 32/37/24 | 0.5093 |
| A1 seed 42 − BM25 | company | +0.029 [-0.058, +0.116] | 27/45/21 | 0.5194 |
| A1 seed 1337 − BM25 | global | +0.033 [-0.067, +0.132] | 30/38/25 | 0.5175 |
| A1 seed 1337 − BM25 | company | +0.035 [-0.053, +0.122] | 25/48/20 | 0.4382 |
| A1 seed 2026 − BM25 | global | +0.023 [-0.073, +0.120] | 31/35/27 | 0.6443 |
| A1 seed 2026 − BM25 | company | +0.014 [-0.072, +0.100] | 26/44/23 | 0.7608 |


How to read it:

- **A1 vs A0 (+0.39 to +0.46 MRR@10, p = 0.0001 in every seed and mode)** is the Stage A result, reproduced. It
  is real on this distribution, but the questions were written by an LLM from these passages, accepted by an AI
  review and used to choose each checkpoint, so it is an upper bound, not a generalisation estimate.
- **BM25 nearly matches A1.** BM25 has no training and no selection advantage, so its 0.635 says the synthetic
  questions share much of their wording with their passages. A1's margin over BM25 is +0.014 to +0.035 with
  intervals spanning zero.
- **Restricting to the question's company helps every system** (A1 seed 42: 0.669 → 0.732; BM25 0.635 → 0.703).

**Table T8.** Validation MRR@10 by evidence type (global mode)

| Positive passage | Questions | System | MRR@10 |
|---|---|---|---|
| table | 65 | BM25 | 0.603 |
| table | 65 | BGE zero-shot | 0.197 |
| table | 65 | E5 zero-shot | 0.368 |
| table | 65 | A0 (step 0) | 0.197 |
| table | 65 | A1 seed 42 | 0.693 |
| table | 65 | A1 seed 1337 | 0.697 |
| table | 65 | A1 seed 2026 | 0.683 |
| narrative | 28 | BM25 | 0.709 |
| narrative | 28 | BGE zero-shot | 0.425 |
| narrative | 28 | E5 zero-shot | 0.482 |
| narrative | 28 | A0 (step 0) | 0.425 |
| narrative | 28 | A1 seed 42 | 0.611 |
| narrative | 28 | A1 seed 1337 | 0.600 |
| narrative | 28 | A1 seed 2026 | 0.600 |


A1's largest gain over A0 is on table evidence (0.197 → 0.683–0.697), the bulk of the validation set (65 of 93).
On narrative evidence BM25 (0.709) stays ahead of A1 (0.600–0.611).

### 4.2 Development: 10 questions, AI-written provisional labels

Recomputed from the stored rankings (zero-shot runs from `results/dev/pilot_v2`, A0/A1 from
`results/dev/pilot_v3_a1`). The judged pools contain the zero-shot systems' top 10s but not A0's or A1's: A1's
unjudged passages (15–20% of its top 10) count as non-relevant, so A1's scores are lower bounds.

**Table T4-global.** Development set (10 questions, AI-written provisional labels), global mode

| System | MRR@10 [95% CI] | Recall@10 [95% CI] | nDCG@10 [95% CI] | Recall@100 | judged@10 | Max abs Δ vs harness record |
|---|---|---|---|---|---|---|
| BM25 | 0.323 [0.108, 0.564] | 0.146 [0.057, 0.250] | 0.170 [0.061, 0.305] | 0.578 | 1.000 | 5.6e-17 |
| BGE zero-shot | 0.550 [0.300, 0.800] | 0.206 [0.122, 0.289] | 0.286 [0.145, 0.445] | 0.627 | 1.000 | 1.1e-16 |
| E5 zero-shot | 0.600 [0.340, 0.850] | 0.279 [0.181, 0.379] | 0.361 [0.210, 0.514] | 0.666 | 1.000 | 1.1e-16 |
| A0 (step 0) | 0.550 [0.300, 0.800] | 0.206 [0.122, 0.289] | 0.286 [0.145, 0.445] | 0.627 | 1.000 | 1.1e-16 |
| A1 seed 42 | 0.558 [0.308, 0.800] | 0.327 [0.175, 0.485] | 0.320 [0.176, 0.473] | 0.769 | 0.820 | 5.6e-17 |
| A1 seed 1337 | 0.583 [0.333, 0.800] | 0.327 [0.175, 0.485] | 0.324 [0.179, 0.479] | 0.750 | 0.850 | 2.2e-16 |
| A1 seed 2026 | 0.667 [0.400, 0.933] | 0.257 [0.143, 0.372] | 0.307 [0.167, 0.463] | 0.806 | 0.800 | 1.1e-16 |

**Table T4-company.** Development set (10 questions, AI-written provisional labels), company mode

| System | MRR@10 [95% CI] | Recall@10 [95% CI] | nDCG@10 [95% CI] | Recall@100 | judged@10 | Max abs Δ vs harness record |
|---|---|---|---|---|---|---|
| BM25 | 0.423 [0.183, 0.679] | 0.184 [0.090, 0.287] | 0.194 [0.085, 0.329] | 0.674 | 1.000 | 1.1e-16 |
| BGE zero-shot | 0.575 [0.350, 0.800] | 0.239 [0.161, 0.311] | 0.297 [0.162, 0.450] | 0.643 | 1.000 | 5.6e-17 |
| E5 zero-shot | 0.625 [0.375, 0.864] | 0.296 [0.199, 0.390] | 0.376 [0.225, 0.524] | 0.673 | 1.000 | 1.1e-16 |
| A0 (step 0) | 0.575 [0.350, 0.800] | 0.239 [0.161, 0.311] | 0.297 [0.162, 0.450] | 0.643 | 1.000 | 5.6e-17 |
| A1 seed 42 | 0.592 [0.367, 0.808] | 0.344 [0.206, 0.491] | 0.328 [0.193, 0.477] | 0.839 | 0.810 | 1.1e-16 |
| A1 seed 1337 | 0.617 [0.400, 0.833] | 0.344 [0.206, 0.491] | 0.332 [0.194, 0.482] | 0.833 | 0.830 | 1.1e-16 |
| A1 seed 2026 | 0.717 [0.483, 0.933] | 0.307 [0.187, 0.435] | 0.323 [0.187, 0.476] | 0.839 | 0.800 | 1.1e-16 |

**Table T4-paired.** Development set, paired differences (10 questions; question bootstrap; sign-flip test)

| Comparison | Mode | Metric | Mean difference [95% CI] | Wins/ties/losses | p |
|---|---|---|---|---|---|
| A1 seed 42 − A0 (step 0) | global | mrr@10 | +0.008 [-0.167, +0.208] | 2/5/3 | 1.0000 |
| A1 seed 42 − A0 (step 0) | global | recall@10 | +0.121 [+0.037, +0.213] | 5/5/0 | 0.0625 |
| A1 seed 42 − A0 (step 0) | company | mrr@10 | +0.017 [-0.167, +0.217] | 3/4/3 | 0.9062 |
| A1 seed 42 − A0 (step 0) | company | recall@10 | +0.105 [+0.019, +0.201] | 4/6/0 | 0.1250 |
| A1 seed 1337 − A0 (step 0) | global | mrr@10 | +0.033 [-0.133, +0.225] | 2/6/2 | 0.7500 |
| A1 seed 1337 − A0 (step 0) | global | recall@10 | +0.121 [+0.037, +0.213] | 5/5/0 | 0.0625 |
| A1 seed 1337 − A0 (step 0) | company | mrr@10 | +0.042 [-0.125, +0.233] | 3/5/2 | 0.6875 |
| A1 seed 1337 − A0 (step 0) | company | recall@10 | +0.105 [+0.019, +0.201] | 4/6/0 | 0.1250 |
| A1 seed 2026 − A0 (step 0) | global | mrr@10 | +0.117 [-0.067, +0.342] | 2/6/2 | 0.5000 |
| A1 seed 2026 − A0 (step 0) | global | recall@10 | +0.051 [+0.001, +0.107] | 4/5/1 | 0.1875 |
| A1 seed 2026 − A0 (step 0) | company | mrr@10 | +0.142 [-0.042, +0.358] | 3/5/2 | 0.3125 |
| A1 seed 2026 − A0 (step 0) | company | recall@10 | +0.068 [+0.002, +0.144] | 4/5/1 | 0.1875 |
| E5 zero-shot − A0 (step 0) | global | mrr@10 | +0.050 [-0.243, +0.322] | 4/4/2 | 0.7500 |
| E5 zero-shot − A0 (step 0) | global | recall@10 | +0.073 [+0.011, +0.134] | 5/4/1 | 0.0938 |
| E5 zero-shot − A0 (step 0) | company | mrr@10 | +0.050 [-0.246, +0.325] | 5/2/3 | 0.7422 |
| E5 zero-shot − A0 (step 0) | company | recall@10 | +0.056 [-0.022, +0.129] | 5/3/2 | 0.2188 |
| BM25 − A0 (step 0) | global | mrr@10 | -0.227 [-0.517, +0.075] | 2/3/5 | 0.2188 |
| BM25 − A0 (step 0) | global | recall@10 | -0.060 [-0.190, +0.078] | 2/2/6 | 0.4219 |
| BM25 − A0 (step 0) | company | mrr@10 | -0.152 [-0.492, +0.200] | 3/2/5 | 0.4531 |
| BM25 − A0 (step 0) | company | recall@10 | -0.056 [-0.199, +0.098] | 3/1/6 | 0.4961 |
| A1 seed 42 − E5 zero-shot | global | mrr@10 | -0.042 [-0.333, +0.250] | 3/3/4 | 0.8125 |
| A1 seed 42 − E5 zero-shot | company | mrr@10 | -0.034 [-0.328, +0.265] | 3/3/4 | 0.8438 |
| A1 seed 1337 − E5 zero-shot | global | mrr@10 | -0.017 [-0.283, +0.260] | 3/3/4 | 0.9219 |
| A1 seed 1337 − E5 zero-shot | company | mrr@10 | -0.009 [-0.278, +0.274] | 3/3/4 | 1.0000 |
| A1 seed 2026 − E5 zero-shot | global | mrr@10 | +0.066 [-0.228, +0.369] | 3/4/3 | 0.7188 |
| A1 seed 2026 − E5 zero-shot | company | mrr@10 | +0.091 [-0.203, +0.385] | 4/3/3 | 0.5781 |


Ten questions cannot rank these systems: every MRR@10 interval overlaps every other. The only consistent signal is
that A1 brings more judged-relevant passages into its top 10 than A0 (Recall@10 +0.051 to +0.121; for seeds 42
and 1337 the interval excludes zero but p = 0.0625 with 10 questions) and into its top 100 (0.750–0.806 vs 0.627;
see `results/dev/pilot_v3_a1/report.md` for the paired Recall@100 tests). BM25 is weakest here (MRR@10 0.323),
the reverse of validation: these questions were written with less wording overlap.

## 5. Real-question diagnostic: 100 frozen human-written questions

For each question's top 10, the diagnostic reads only the question's own scope (its companies and fiscal years) and
each retrieved passage's filing metadata. A passage is *exact year* when its fiscal year is one the question asks
about; *later filing* when it is a later 10-K within two years (which may carry the asked year as a comparative
column, so it is not counted as an error); *cross-year* otherwise. *Year-blind chance* is the share of the asked
companies' passages that come from the asked years: what a ranker ignoring the year would get in company mode.
**None of this is relevance**: an in-scope passage can be useless and a later filing can hold the answer.

**Table T5-global.** Real-question alignment diagnostic (100 human-written questions, top 10), global mode

| System | Correct company | Exact year | Exact year among correct company | Year-blind chance | Later filing (comparative window) | Cross-year | Cross-company | Top 1 in exact scope | Any in exact scope | All scope cells covered | Table passages | A1 training/validation passages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 78.7% | 12.4% | 15.8% | 17.6% | 20.3% | 46.0% | 21.3% | 15.0% | 70.0% | 56.0% | 52.0% | 17.0% |
| BGE zero-shot | 94.6% | 16.8% | 17.8% | 17.6% | 24.2% | 53.6% | 5.4% | 18.0% | 82.0% | 68.0% | 59.7% | 15.9% |
| E5 zero-shot | 94.1% | 26.4% | 28.1% | 17.6% | 28.6% | 39.1% | 5.9% | 22.0% | 91.0% | 79.0% | 67.8% | 14.9% |
| A0 (step 0) | 94.6% | 16.8% | 17.8% | 17.6% | 24.2% | 53.6% | 5.4% | 18.0% | 82.0% | 68.0% | 59.7% | 15.9% |
| A1 seed 42 | 82.9% | 23.1% | 27.9% | 17.6% | 21.2% | 38.6% | 17.1% | 19.0% | 90.0% | 78.0% | 59.5% | 13.1% |
| A1 seed 1337 | 83.0% | 20.5% | 24.7% | 17.6% | 22.1% | 40.4% | 17.0% | 21.0% | 87.0% | 74.0% | 54.6% | 13.4% |
| A1 seed 2026 | 82.2% | 21.1% | 25.7% | 17.6% | 22.4% | 38.7% | 17.8% | 23.0% | 85.0% | 73.0% | 58.6% | 14.2% |

**Table T5-company.** Real-question alignment diagnostic (100 human-written questions, top 10), company mode

| System | Correct company | Exact year | Exact year among correct company | Year-blind chance | Later filing (comparative window) | Cross-year | Cross-company | Top 1 in exact scope | Any in exact scope | All scope cells covered | Table passages | A1 training/validation passages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 100.0% | 16.3% | 16.3% | 17.6% | 25.2% | 58.5% | 0.0% | 17.0% | 85.0% | 71.0% | 51.4% | 15.2% |
| BGE zero-shot | 100.0% | 17.8% | 17.8% | 17.6% | 24.9% | 57.3% | 0.0% | 19.0% | 84.0% | 71.0% | 59.9% | 15.0% |
| E5 zero-shot | 100.0% | 28.2% | 28.2% | 17.6% | 29.7% | 42.1% | 0.0% | 22.0% | 94.0% | 82.0% | 67.3% | 14.3% |
| A0 (step 0) | 100.0% | 17.8% | 17.8% | 17.6% | 24.9% | 57.3% | 0.0% | 19.0% | 84.0% | 71.0% | 59.9% | 15.0% |
| A1 seed 42 | 100.0% | 27.5% | 27.5% | 17.6% | 24.9% | 47.6% | 0.0% | 21.0% | 96.0% | 83.0% | 59.9% | 13.6% |
| A1 seed 1337 | 100.0% | 25.5% | 25.5% | 17.6% | 26.0% | 48.5% | 0.0% | 21.0% | 95.0% | 81.0% | 54.4% | 13.8% |
| A1 seed 2026 | 100.0% | 26.0% | 26.0% | 17.6% | 26.3% | 47.7% | 0.0% | 24.0% | 92.0% | 80.0% | 58.0% | 14.5% |


- **Company.** BGE/A0 and E5 keep 94.1–94.6% of their top 10 in the asked company; A1 drops to 82.2–83.0%, near
  BM25 (78.7%). Company mode removes this by construction.
- **Fiscal year.** In company mode BM25 (16.3%) and BGE/A0 (17.8%) sit at the year-blind chance rate (17.6%): they
  do not use the year in the question. E5 (28.2%) and A1 (25.5–27.5%) use it, but weakly.
- **Coverage.** In company mode A1 seed 42 has at least one exact-scope passage for 96 questions and covers every
  scope cell for 83 (E5: 94 and 82).
- **Own training data.** 13.1–14.2% of A1's real-question top 10 comes from its own training or validation
  passages, below their 17.2% share of the corpus: no sign that A1 favours what it was trained on.

**Table T6-by_question_type.** Alignment by question type (global mode, top 10)

| Question type | Questions | System | Correct company | Exact year | Cross-year | Top 1 in exact scope | All scope cells covered | Companies covered | Table passages |
|---|---|---|---|---|---|---|---|---|---|
| exact_fact | 28 | BM25 | 80.0% | 5.7% | 56.4% | 0.0% | 50.0% | 100.0% | 62.5% |
| exact_fact | 28 | BGE zero-shot | 95.0% | 12.9% | 59.6% | 7.1% | 71.4% | 100.0% | 75.7% |
| exact_fact | 28 | E5 zero-shot | 96.8% | 22.1% | 44.3% | 7.1% | 85.7% | 100.0% | 78.6% |
| exact_fact | 28 | A0 (step 0) | 95.0% | 12.9% | 59.6% | 7.1% | 71.4% | 100.0% | 75.7% |
| exact_fact | 28 | A1 seed 42 | 83.6% | 23.6% | 41.1% | 17.9% | 92.9% | 96.4% | 75.0% |
| exact_fact | 28 | A1 seed 1337 | 81.8% | 19.3% | 43.2% | 21.4% | 92.9% | 96.4% | 69.3% |
| exact_fact | 28 | A1 seed 2026 | 83.2% | 21.1% | 41.1% | 25.0% | 89.3% | 96.4% | 75.7% |
| multi_passage | 15 | BM25 | 74.0% | 16.7% | 36.0% | 33.3% | 40.0% | 80.0% | 48.0% |
| multi_passage | 15 | BGE zero-shot | 98.0% | 27.3% | 50.0% | 33.3% | 46.7% | 93.3% | 66.7% |
| multi_passage | 15 | E5 zero-shot | 94.7% | 32.7% | 39.3% | 20.0% | 33.3% | 93.3% | 84.7% |
| multi_passage | 15 | A0 (step 0) | 98.0% | 27.3% | 50.0% | 33.3% | 46.7% | 93.3% | 66.7% |
| multi_passage | 15 | A1 seed 42 | 84.0% | 30.0% | 30.7% | 33.3% | 53.3% | 90.0% | 68.0% |
| multi_passage | 15 | A1 seed 1337 | 83.3% | 27.3% | 32.7% | 33.3% | 46.7% | 86.7% | 58.0% |
| multi_passage | 15 | A1 seed 2026 | 82.7% | 25.3% | 30.7% | 33.3% | 46.7% | 86.7% | 62.0% |
| semantic | 32 | BM25 | 80.6% | 15.3% | 48.8% | 15.6% | 68.8% | 92.2% | 26.2% |
| semantic | 32 | BGE zero-shot | 90.6% | 14.1% | 52.8% | 15.6% | 71.9% | 95.3% | 25.6% |
| semantic | 32 | E5 zero-shot | 91.2% | 24.7% | 40.9% | 28.1% | 84.4% | 95.3% | 38.1% |
| semantic | 32 | A0 (step 0) | 90.6% | 14.1% | 52.8% | 15.6% | 71.9% | 95.3% | 25.6% |
| semantic | 32 | A1 seed 42 | 80.9% | 16.9% | 43.7% | 12.5% | 75.0% | 93.8% | 20.9% |
| semantic | 32 | A1 seed 1337 | 82.2% | 14.4% | 45.6% | 12.5% | 71.9% | 95.3% | 17.8% |
| semantic | 32 | A1 seed 2026 | 80.3% | 15.6% | 44.4% | 15.6% | 71.9% | 93.8% | 20.3% |
| table | 25 | BM25 | 77.6% | 13.6% | 36.8% | 20.0% | 56.0% | 92.0% | 75.6% |
| table | 25 | BGE zero-shot | 97.2% | 18.4% | 50.0% | 24.0% | 72.0% | 100.0% | 81.2% |
| table | 25 | E5 zero-shot | 94.4% | 29.6% | 30.8% | 32.0% | 92.0% | 100.0% | 83.6% |
| table | 25 | A0 (step 0) | 97.2% | 18.4% | 50.0% | 24.0% | 72.0% | 100.0% | 81.2% |
| table | 25 | A1 seed 42 | 84.0% | 26.4% | 34.0% | 20.0% | 80.0% | 100.0% | 86.4% |
| table | 25 | A1 seed 1337 | 85.2% | 25.6% | 35.2% | 24.0% | 72.0% | 100.0% | 83.2% |
| table | 25 | A1 seed 2026 | 83.2% | 25.6% | 33.6% | 24.0% | 72.0% | 96.0% | 86.4% |

**Table T6-by_comparison.** Alignment by comparison (global mode, top 10)

| Comparison | Questions | System | Correct company | Exact year | Cross-year | Top 1 in exact scope | All scope cells covered | Companies covered | Table passages |
|---|---|---|---|---|---|---|---|---|---|
| cross_company | 10 | BM25 | 66.0% | 12.0% | 44.0% | 20.0% | 10.0% | 65.0% | 30.0% |
| cross_company | 10 | BGE zero-shot | 93.0% | 15.0% | 61.0% | 30.0% | 10.0% | 75.0% | 39.0% |
| cross_company | 10 | E5 zero-shot | 91.0% | 24.0% | 50.0% | 10.0% | 10.0% | 75.0% | 58.0% |
| cross_company | 10 | A0 (step 0) | 93.0% | 15.0% | 61.0% | 30.0% | 10.0% | 75.0% | 39.0% |
| cross_company | 10 | A1 seed 42 | 75.0% | 20.0% | 32.0% | 10.0% | 10.0% | 75.0% | 48.0% |
| cross_company | 10 | A1 seed 1337 | 80.0% | 16.0% | 42.0% | 10.0% | 0.0% | 75.0% | 37.0% |
| cross_company | 10 | A1 seed 2026 | 77.0% | 14.0% | 38.0% | 10.0% | 0.0% | 70.0% | 43.0% |
| cross_year | 21 | BM25 | 87.6% | 19.0% | 35.7% | 28.6% | 52.4% | 95.2% | 62.4% |
| cross_year | 21 | BGE zero-shot | 98.1% | 27.1% | 42.4% | 23.8% | 61.9% | 100.0% | 71.0% |
| cross_year | 21 | E5 zero-shot | 96.2% | 35.7% | 28.6% | 42.9% | 76.2% | 100.0% | 82.4% |
| cross_year | 21 | A0 (step 0) | 98.1% | 27.1% | 42.4% | 23.8% | 61.9% | 100.0% | 71.0% |
| cross_year | 21 | A1 seed 42 | 83.3% | 31.9% | 26.2% | 33.3% | 71.4% | 95.2% | 73.8% |
| cross_year | 21 | A1 seed 1337 | 83.3% | 31.4% | 26.2% | 38.1% | 61.9% | 95.2% | 71.0% |
| cross_year | 21 | A1 seed 2026 | 81.9% | 31.0% | 24.3% | 33.3% | 66.7% | 90.5% | 71.9% |
| none | 69 | BM25 | 77.8% | 10.4% | 49.4% | 10.1% | 63.8% | 95.7% | 52.0% |
| none | 69 | BGE zero-shot | 93.8% | 13.9% | 55.9% | 14.5% | 78.3% | 100.0% | 59.3% |
| none | 69 | E5 zero-shot | 93.9% | 23.9% | 40.7% | 17.4% | 89.9% | 100.0% | 64.8% |
| none | 69 | A0 (step 0) | 93.8% | 13.9% | 55.9% | 14.5% | 78.3% | 100.0% | 59.3% |
| none | 69 | A1 seed 42 | 83.9% | 20.9% | 43.3% | 15.9% | 89.9% | 98.6% | 56.8% |
| none | 69 | A1 seed 1337 | 83.3% | 17.8% | 44.5% | 17.4% | 88.4% | 98.6% | 52.2% |
| none | 69 | A1 seed 2026 | 83.0% | 19.1% | 43.2% | 21.7% | 85.5% | 98.6% | 56.8% |


- **Exact-fact questions** (28) are where a wrong-year passage hurts most: BM25 never puts an exact-scope passage
  first, BGE/A0 and E5 do so for 7.1% of them, and A1 for 17.9–25.0%.
- **Cross-company questions** (10): all company-year cells are covered for at most 1 of 10 questions by any system
  (A1 seeds 1337 and 2026: none), and on average only 65–75% of the asked companies appear in the top 10 at all.
- **Cross-year questions** (21): all years covered for 52.4% (BM25) to 76.2% (E5); A1 61.9–71.4%.
- **Multi-passage questions** (15): all cells covered for 33.3% (E5) to 53.3% (A1 seed 42).

## 6. Error analysis

### 6.1 Wrong-company retrievals

**Table T7.** Validation error counts (global mode, 93 questions)

| System | Top 1 wrong company (of them, relevant still in top 10) | Top 1 right company, wrong year (not relevant) | Years off (wrong-year top 1) | Top 1 = same text from another filing (not relevant) | Relevant not in top 10 | Relevant not in top 100 | Answer figure in top 1 | Answer figure in top 10 |
|---|---|---|---|---|---|---|---|---|
| BM25 | 11 (6) | 16 | 1y: 11, 2y: 3, 3y: 2 | 0 | 16 | 2 | 38/64 | 54/64 |
| BGE zero-shot | 6 (2) | 56 | 1y: 19, 2y: 13, 3y: 13, 4y: 8, 5y: 1, 6y: 2 | 0 | 47 | 13 | 13/64 | 33/64 |
| E5 zero-shot | 13 (5) | 36 | 1y: 25, 2y: 3, 3y: 1, 4y: 4, 5y: 2, 6y: 1 | 0 | 31 | 11 | 19/64 | 43/64 |
| A0 (step 0) | 6 (2) | 56 | 1y: 19, 2y: 13, 3y: 13, 4y: 8, 5y: 1, 6y: 2 | 0 | 47 | 13 | 13/64 | 33/64 |
| A1 seed 42 | 14 (12) | 16 | 1y: 11, 2y: 3, 3y: 1, 5y: 1 | 0 | 7 | 0 | 39/64 | 61/64 |
| A1 seed 1337 | 15 (12) | 12 | 1y: 8, 2y: 2, 3y: 1, 5y: 1 | 0 | 9 | 0 | 41/64 | 61/64 |
| A1 seed 2026 | 12 (10) | 16 | 1y: 11, 2y: 2, 3y: 1, 4y: 1, 5y: 1 | 0 | 7 | 0 | 39/64 | 61/64 |


**Table S4.** Real questions whose top 10 is mostly (> 50%) from other companies (of 100)

| System | Global | Company |
|---|---|---|
| BM25 | 15 | 0 |
| BGE zero-shot | 1 | 0 |
| E5 zero-shot | 2 | 0 |
| A0 (step 0) | 1 | 0 |
| A1 seed 42 | 15 | 0 |
| A1 seed 1337 | 14 | 0 |
| A1 seed 2026 | 14 | 0 |

**Table S5.** Real single-company questions: cross-company share of the top 10 by company asked (global)

| System | AAPL (13 q) | AMZN (13 q) | JPM (13 q) | PFE (13 q) | TSLA (13 q) | WMT (13 q) | XOM (12 q) |
|---|---|---|---|---|---|---|---|
| BM25 | 33.8% | 43.1% | 0.0% | 14.6% | 20.8% | 22.3% | 3.3% |
| BGE zero-shot | 19.2% | 5.4% | 0.0% | 7.7% | 3.8% | 0.0% | 0.0% |
| E5 zero-shot | 28.5% | 3.8% | 0.0% | 3.1% | 1.5% | 1.5% | 0.0% |
| A0 (step 0) | 19.2% | 5.4% | 0.0% | 7.7% | 3.8% | 0.0% | 0.0% |
| A1 seed 42 | 38.5% | 29.2% | 0.8% | 6.2% | 3.8% | 31.5% | 2.5% |
| A1 seed 1337 | 40.0% | 28.5% | 0.0% | 8.5% | 6.2% | 28.5% | 4.2% |
| A1 seed 2026 | 40.8% | 30.8% | 0.8% | 8.5% | 4.6% | 30.8% | 3.3% |


A1's wrong-company problem is concentrated in Apple, Amazon and Walmart questions (28.5–40.8% of the top 10 from
other companies, vs 19.2%, 5.4% and 0.0% for A0). On validation, A1's wrong-company top results are mostly
financial tables (7–8 of its 12–15) and in 10–12 of them the relevant passage is still in the top 10.

**Probable mechanism (a hypothesis, supported by the three checks below).** Company identity is often absent from
the passage text, and fine-tuning taught A1 to rely less on it:

**Table S1.** Passages whose text names their own company

| Company | Passages | Naming it |
|---|---|---|
| AAPL | 618 | 23.1% |
| AMZN | 776 | 37.2% |
| JPM | 3629 | 34.5% |
| PFE | 1505 | 60.7% |
| TSLA | 1241 | 27.4% |
| WMT | 1010 | 55.4% |
| XOM | 1358 | 47.3% |
| All | 10137 | 40.8% |
| All table passages | 5668 | 36.2% |

**Table S2.** Real-question top 10: share of passages naming their company

| System | Mode | Correct-company passages naming it | Correct-company passages | Wrong-company passages naming it | Wrong-company passages |
|---|---|---|---|---|---|
| BM25 | global | 84.5% | 787 | 55.9% | 213 |
| BM25 | company | 80.3% | 1000 | n/a | 0 |
| BGE zero-shot | global | 85.9% | 946 | 22.2% | 54 |
| BGE zero-shot | company | 85.6% | 1000 | n/a | 0 |
| E5 zero-shot | global | 80.3% | 941 | 22.0% | 59 |
| E5 zero-shot | company | 79.5% | 1000 | n/a | 0 |
| A0 (step 0) | global | 85.9% | 946 | 22.2% | 54 |
| A0 (step 0) | company | 85.6% | 1000 | n/a | 0 |
| A1 seed 42 | global | 63.4% | 829 | 41.5% | 171 |
| A1 seed 42 | company | 61.6% | 1000 | n/a | 0 |
| A1 seed 1337 | global | 64.8% | 830 | 40.0% | 170 |
| A1 seed 1337 | company | 62.2% | 1000 | n/a | 0 |
| A1 seed 2026 | global | 65.1% | 822 | 39.3% | 178 |
| A1 seed 2026 | company | 62.8% | 1000 | n/a | 0 |


59.2% of passages (and 63.8% of table passages) never name their company; the company exists only as metadata. Of
the asked-company passages in BGE/A0's real-question top 10 (global mode), 85.9% name the company; for A1, 63.4–65.1%,
closer to the corpus rate (40.8%). A1's training data explains why:

**Table S6.** A1's training pairs (train_candidates_v3): company names in question and positive passage

| Training pairs | Count |
|---|---|
| Questions naming the company | 677/678 |
| Positive passages naming the company | 273/678 |
| Question names it, positive passage does not | 405/678 |


In 405 of A1's 678 training pairs the question names the company and the positive passage does not, so
contrastive training rewards matching the topic, the line items and the year rather than the name. This is
consistent with every observation above, but it was not tested by an intervention.

### 6.2 Wrong-year retrievals

- **Frequency.** In company mode, 42.1–58.5% of each system's top 10 is from a year outside the asked year and
  its comparative window (T5-company). On validation, BGE/A0 put a right-company, wrong-year, non-relevant passage
  first for 56 of 93 questions; A1 for 12–16, BM25 16, E5 36 (T7).
- **Nature.** A0's wrong-year top results are spread across 1–6 years (19 of 56 one year off); A1's are mostly one
  year off (8–11 of 12–16), earlier and later filings alike (`top1_wrong_year_offsets` in `results.json` counts
  absolute offsets; `validation_per_question.jsonl` has the signed years). An adjacent filing shares most of its
  wording and line items with the asked one, and a later one repeats the asked year as a comparative column.
- **Labels interact with this.** A later filing that repeats a figure can hold the answer. The validation labels
  treat some of these as uncertain (removed from the ranking) and miss others (counted non-relevant); see example
  E3. Year-aware retrieval and evaluation must both allow the comparative window.

### 6.3 Relevant evidence ranked too low

- **Validation.** A1 misses the top 10 for 7–9 of 93 questions (A0: 47), and all three seeds miss 5 of the same
  questions (`validation_a1_consensus_misses` in `results.json`): their relevant passages sit at ranks 11–79. For
  every question, every A1 seed has a relevant passage in its top 100 (Recall@100 0.997–1.000).
- **Development.** Every system has no judged-relevant passage in its top 10 for at least one of the 10 questions
  (BM25 for four in global mode; see `results/dev/pilot_v3_a1/report.md`).
- **Evidence position.** Dense retrievers find the evidence less often when it sits later in its passage:

**Table S3.** Validation MRR@10 (global) by where the evidence starts in its passage (BGE tokens; 92/93 located; median start 273; 0 ending beyond 510)

| System | Starts at 0-127 (n) | Starts at 128-255 (n) | Starts at 256+ (n) |
|---|---|---|---|
| BM25 | 0.530 (21) | 0.634 (24) | 0.674 (47) |
| BGE zero-shot | 0.444 (21) | 0.251 (24) | 0.189 (47) |
| E5 zero-shot | 0.476 (21) | 0.387 (24) | 0.364 (47) |
| A0 (step 0) | 0.444 (21) | 0.251 (24) | 0.189 (47) |
| A1 seed 42 | 0.709 (21) | 0.798 (24) | 0.577 (47) |
| A1 seed 1337 | 0.713 (21) | 0.787 (24) | 0.580 (47) |
| A1 seed 2026 | 0.659 (21) | 0.800 (24) | 0.578 (47) |


Every validation evidence span lies inside the 510-token encoder window, so truncation does not explain the misses;
but in every dense system MRR@10 is lowest when the evidence starts after token 256 (A0 0.444 → 0.189; E5 0.476 →
0.364; A1 0.659–0.713 → 0.577–0.580), while BM25 is not hurt (0.530 → 0.674). One vector per long passage dilutes
a late detail. This is observational (position correlates with section and table type). It also exposes a blind
spot: the validation set never tests evidence beyond token 510, yet 90.8% of passages are longer than the
encoder window (27.6% of passage tokens are never encoded; development report, retriever section).

### 6.4 Missing financial figures

Of the 64 validation questions whose answer has a distinctive figure (three or more digits, not a year), the
figure appears somewhere in the top 10 for 61 with every A1 seed, 54 with BM25, 43 with E5 and 33 with A0; in the
top 1 for 39–41 (A1), 38 (BM25), 19 (E5) and 13 (A0) (T7). The three figures A1 seed 42 misses are statement or
schedule figures (Amazon's 2020 comprehensive income, E3; Amazon's long-term debt commitments due "thereafter";
Pfizer's 2023 accumulated depreciation); two of the three also appear in a later filing's comparative column.
Figures are matched as whole numbers; a formatted variant (for example "$22.1 billion") would not match, so these
counts are lower bounds.

### 6.5 Difficult multi-passage and multi-scope questions

Questions that need several passages (multi-passage, cross-company, cross-year) are where every system is weakest
(T6). The failure is structural: a single top 10 ranked by similarity to the whole question concentrates on one
company or one year (examples E7, E8). No system decomposes a question into its company-year cells.

### 6.6 Base model vs fine-tuned seeds

**Table T9.** Agreement between dense systems (global mode, top 10)

| Questions | Pair | Mean top-10 Jaccard | Same top 1 |
|---|---|---|---|
| validation | A0 (step 0) vs BGE zero-shot | 1.000 | 93/93 |
| validation | A0 (step 0) vs A1 seed 42 | 0.237 | 25/93 |
| validation | A0 (step 0) vs A1 seed 1337 | 0.241 | 25/93 |
| validation | A0 (step 0) vs A1 seed 2026 | 0.235 | 24/93 |
| validation | BGE zero-shot vs A1 seed 42 | 0.237 | 25/93 |
| validation | BGE zero-shot vs A1 seed 1337 | 0.241 | 25/93 |
| validation | BGE zero-shot vs A1 seed 2026 | 0.235 | 24/93 |
| validation | A1 seed 42 vs A1 seed 1337 | 0.793 | 81/93 |
| validation | A1 seed 42 vs A1 seed 2026 | 0.819 | 74/93 |
| validation | A1 seed 1337 vs A1 seed 2026 | 0.772 | 71/93 |
| gold | A0 (step 0) vs BGE zero-shot | 1.000 | 100/100 |
| gold | A0 (step 0) vs A1 seed 42 | 0.255 | 20/100 |
| gold | A0 (step 0) vs A1 seed 1337 | 0.255 | 16/100 |
| gold | A0 (step 0) vs A1 seed 2026 | 0.257 | 24/100 |
| gold | BGE zero-shot vs A1 seed 42 | 0.255 | 20/100 |
| gold | BGE zero-shot vs A1 seed 1337 | 0.255 | 16/100 |
| gold | BGE zero-shot vs A1 seed 2026 | 0.257 | 24/100 |
| gold | A1 seed 42 vs A1 seed 1337 | 0.756 | 72/100 |
| gold | A1 seed 42 vs A1 seed 2026 | 0.759 | 71/100 |
| gold | A1 seed 1337 vs A1 seed 2026 | 0.725 | 71/100 |

**Table T10.** Validation complementarity: relevant passage in the top 10 of a dense system and of BM25

| Dense system | Mode | Both | Dense only | BM25 only | Neither | Either (union) |
|---|---|---|---|---|---|---|
| A1 seed 42 | global | 72 | 14 | 5 | 2 | 91/93 |
| A1 seed 42 | company | 78 | 9 | 4 | 2 | 91/93 |
| A1 seed 1337 | global | 71 | 13 | 6 | 3 | 90/93 |
| A1 seed 1337 | company | 78 | 8 | 4 | 3 | 90/93 |
| A1 seed 2026 | global | 72 | 14 | 5 | 2 | 91/93 |
| A1 seed 2026 | company | 80 | 9 | 2 | 2 | 91/93 |
| E5 zero-shot | global | 59 | 3 | 18 | 13 | 80/93 |
| E5 zero-shot | company | 62 | 2 | 20 | 9 | 84/93 |
| A0 (step 0) | global | 43 | 3 | 34 | 13 | 80/93 |
| A0 (step 0) | company | 49 | 3 | 33 | 8 | 85/93 |


- **Fine-tuning changed most of the ranking.** A1's top 10 shares only 0.235–0.257 (Jaccard) with A0's, and the
  same top 1 for 24–25 of 93 validation and 16–24 of 100 real questions.
- **The three seeds agree with one another** (Jaccard 0.725–0.819; same top 1 for 71–81 questions): the change is a
  property of the training data and recipe, not of a seed. Seed differences in the aggregates (≤ 0.021 MRR@10 on
  validation) are smaller than their intervals.
- **BM25 and A1 are complementary.** On validation, BM25 finds the relevant passage in its top 10 for 2–6
  questions A1 misses, and A1 for 8–14 that BM25 misses; together they cover 90–91 of 93 (synthetic,
  lexical-overlap questions: an upper bound for what fusion could add).

### 6.7 Manually inspected examples

Inspected by Claude (an AI assistant) on 2026-11-15 from `top10.jsonl` and the corpus. **These are not human
judgments.** For validation questions, "relevant" means the validation set's own label; for real questions, only
company, fiscal year and section are described, and nothing is said about relevance.

| # | Question (set) | What the systems retrieved | What it shows |
|---|---|---|---|
| E1 | "What were Amazon's total gross deferred tax assets reported for the fiscal year ended December 31, 2022, in millions?" (validation `v2f:000009`) | A1 seeds 42 and 1337 rank a Tesla FY2022 income-tax table first and Amazon's relevant table second; seed 2026 ranks it first. The Tesla table never names Tesla. A0 ranks Amazon passages first but has the relevant one nowhere in its top 100 | Wrong company: a table without the company's name, matched on line items and year |
| E2 | Apple's total long-term debt at the end of FY2022 (real, `q_0012`, global; paraphrased: the human-written questions are not published) | All three A1 seeds: 0 Apple passages in the top 10 (ExxonMobil, Walmart and JPMorgan long-term-debt notes). A0: 7 Apple passages (FY2019–2022); E5: 7 (FY2020–2025) | Wrong company at its most extreme; company mode removes it |
| E3 | "What was Amazon's comprehensive income for the fiscal year ended December 31, 2020?" (validation `v2f:000079`) | Answer $22,137 million appears in three Amazon statements: FY2020 (the labelled positive), FY2021 (unlabelled, so counted non-relevant) and FY2022 (labelled uncertain, removed). No A1 seed has any of them in its top 10 (positive at ranks 22, 32, 11); seed 42's top 1 is an FY2021 segment table | Wrong year, missing figure, and a label gap for comparative columns |
| E4 | "For JPMorgan Chase & Co as of December 31, 2021, what is the Firm's stated method for managing the credit risk associated with contracts to sell protection?" (validation `v2f:000017`) | Seven relevant passages (the same sentence in seven years' derivatives notes). A1 seed 42 ranks the general "Credit and investment risk management" sections of other years first; first relevant at 35, 45, 41. A0 and E5: none in the top 100; BM25: 12 | Relevant evidence ranked too low: a one-sentence detail inside a long note loses to a topical section |
| E5 | "What was the combined percentage of annual net sales from fuel and financial services offerings for Walmart U.S. in fiscal 2020?" (validation `v2f:000097`) | Relevant passage (Item 1, FY2020) at 47, 28, 79 for A1; 23 for BM25; outside the top 100 for A0 and E5. A1 seed 42 ranks FY2021 MD&A passages first | A narrow narrative detail, wrong year at the top |
| E6 | "What was the fair value of Apple's Level 2 corporate debt securities in fiscal 2022?" (validation `v2f:000001`) | All A1 seeds rank the relevant FY2022 investments table first; A0 at 12, BM25 22, E5 29 | What fine-tuning fixes: exact table rows |
| E7 | JPMorgan Chase versus Pfizer: net income and operating cash flow in FY2024 (real, `q_0109`, global; paraphrased) | A1 seed 42: both companies, but years 2019–2025 and no FY2024 passage. E5: only Pfizer (three FY2024 passages), no JPMorgan | Multi-scope: no system covers the JPM-2024 cell |
| E8 | How Apple's share repurchases and dividends changed from FY2023 to FY2025 (real, `q_0022`; paraphrased) | Global: A1 seed 42's top 10 is entirely JPMorgan (FY2021–2025); A0 has 8 Apple passages. Company mode: A1 covers FY2023 and FY2025 but mixes in FY2020–2024 | Cross-year question; topic matching across companies |

## 7. Most important remaining retrieval problems

1. **Fiscal-year grounding.** Every system ignores or underuses the asked year (section 5, 6.2). Most damaging for
   exact-fact questions, where the right company's wrong year gives a confidently wrong number.
2. **Company grounding after fine-tuning.** A1 traded company awareness for year and table awareness (6.1). In
   global mode it is the only error type that grew with fine-tuning (top-1 wrong company: 6 → 12–15 of 93).
3. **Multi-scope questions.** Cross-company, cross-year and multi-passage questions rarely get every company-year
   cell into one top 10 (5, 6.5).
4. **Long passages and buried details.** One vector per passage of up to 512 tokens (91% truncated) dilutes
   specific sentences and table rows (6.3, E4, E5).
5. **Evaluation itself.** There is still no independent, human-labelled measure: validation is in-distribution
   and lexical, development has 10 AI-labelled questions with pools that omit A1, and comparative-column passages
   are inconsistently labelled (E3).

## 8. Recommendations for the next model improvement

In order of expected value per unit of effort. None requires reopening gold set v1.

1. **Metadata-aware retrieval (no training).** Resolve the question's companies and fiscal years (a deterministic
   parser over the seven companies' names and year expressions is enough here), then retrieve per company-year
   cell: restrict to the company (company mode already gains +0.059 to +0.070 MRR@10 for A1 on validation and
   removes every cross-company passage), and filter or boost by fiscal year, keeping later filings within the
   two-year comparative window. Merge the per-cell lists for multi-scope questions so each cell gets a share of
   the top 10.
2. **Contextual passage headers (re-encode only).** Prepend "Company | 10-K FY | Section" to each passage before
   encoding. 59% of passages do not name their company; the header puts the missing identity into the vector.
   Test with zero-shot E5 and BGE first; A1 would need retraining to benefit fully.
3. **Hybrid lexical + dense fusion.** Reciprocal-rank fusion of BM25 with A1 or E5. Validation suggests the two
   are complementary (90–91 of 93 covered together); check on development that BM25's weakness there does not
   cancel the gain.
4. **Next fine-tuning round, only after 1–3.** Add hard negatives from the same company's other years and from
   other companies' same-topic tables, so that the model must use the year and the company; train with the
   contextual headers; keep checkpoint selection on a question set that BM25 does not already solve.
5. **Shorter passages for dense retrieval.** Split long passages (for example 256 tokens with overlap, tables by
   row groups) or add a reranker that reads the full passage; evaluate on questions whose evidence lies late in
   its passage.
6. **A small human-labelled evaluation (when annotators are available).** 20–30 real questions, pooled from E5,
   A1, BM25 and the improved pipeline, with comparative-column passages labelled explicitly. Until then, every
   claim of improvement remains provisional.

## 9. Limitations

- **No human relevance labels.** Validation labels are one LLM-written passage per question plus automatically
  found equivalents; development labels are AI-written; the 100 real questions were scored for metadata only.
- **Validation is selection-biased and lexical.** It chose the A1 checkpoints and BM25 reaches 0.635 on it.
- **Development is tiny and pool-biased.** Ten questions; A1's top 10 is 15–20% unjudged.
- **The real-question diagnostic measures a necessary condition only.** Exact-scope passages can be useless and
  later filings can hold the answer; the comparative window (two years) is a heuristic.
- **Company-name and figure checks are string matches.** They miss variants ("the Firm", "$22.1 billion").
- **Mechanisms are hypotheses.** The company-name and evidence-position explanations are supported by
  correlations, not by an intervention.
- **The manual inspection is by an AI assistant**, covers eight examples, and is illustrative only.

## 10. Reproducing this analysis

```bash
PY=python   # the analysis used Python 3.11.5, torch 2.10.0, transformers 5.3.0, sentence-transformers 5.2.3
# 1. development set: A0 and A1 beside the stored zero-shot rankings (encodes each checkpoint once, ~25 min on CPU)
$PY scripts/04_run_baselines.py --config configs/eval/retrieval_analysis_v1.yaml --eval-set dev_pilot_v3_a1
# 2. validation, development re-scoring, the real-question diagnostic, the error analysis and the tables
#    (reuses the cached embeddings: about 3 minutes once every system is cached)
$PY scripts/09_retrieval_analysis.py
# 3. supplementary checks: company names (corpus, top 10s, A1 training pairs), evidence position, cross-company top 10s (about 1 minute)
HF_HUB_OFFLINE=1 $PY scripts/10_retrieval_analysis_supplement.py
```

Outputs (`results/analysis/retrieval_analysis_v1/`):

- `results.json`: every aggregate, interval, paired comparison, reproduction check and input hash;
- `validation_per_question.jsonl`, `alignment_per_question.jsonl`, and `top10.jsonl` (every system's top 10 per
  question and mode);
- `tables.md`: tables T1–T10, generated from `results.json`;
- `supplement.json` and `supplement_tables.md`: tables S1–S6.

Every table in this document is copied unchanged from those two table files. Development rankings are in
`results/dev/pilot_v3_a1/`, with the harness report in `report.md`. The diagnostic's logic is in
`src/finretriever/evaluation/alignment.py`, with tests in `tests/unit/test_alignment.py`.
