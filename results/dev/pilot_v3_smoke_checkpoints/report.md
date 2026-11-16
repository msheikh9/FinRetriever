# dev_pilot_v3_smoke_checkpoints: bm25, bge_zero_shot, e5_zero_shot, bge_ft_v2_smoke_step05, bge_ft_v2_smoke_step10, bge_ft_v2_smoke_step15

> **PROVISIONAL DEVELOPMENT METRICS. Labels are AI-generated development labels (see the label provenance), checked only by an AI-assisted audit; these numbers are not benchmark results.**

- Corpus: `corpus_v1@1.2`, 10,137 passages, SHA-256 `ff9dadf37bb9820e2c87df70a411fd189bb02776609e5ec06173dd03f2344004`
- Labels: `dev_pilot_v3`: 696 judged pairs, by source {'ai_proposal': 692, 'human_judgment': 4}, open flags {'P1': 11, 'P2': 2, 'P3': 1, 'P5': 2, 'low_confidence': 8}, audit ai_assisted
- Settings: top_k 100, judged@10, coverage thresholds {'min_positives': 1, 'min_judged_at_10': 0.7}, retrieval modes global, company

Unjudged passages: Qrels hold judged passages only (grades 0, 1, 2). An unjudged passage is not a confirmed negative: the standard metrics count it as non-relevant (an assumption), so each query reports how much of its top 10 is judged, how much is a confirmed negative, and how much is unjudged, split into passages from the companies the question asks about (in scope) and from other companies (out of scope).

Candidate pools: Only passages in a question's candidate pool were judged. The original pools were built mainly with BM25, plus seeded random and per-filing samples and a few passages found by hand. This set's pools add every passage the stored runs company/bge_zero_shot, company/bm25, company/e5_zero_shot, global/bge_zero_shot, global/bm25, global/e5_zero_shot ranked in their top 10, labelled by AI proposals, so the top 10 of those runs is judged. Below rank 10, passages BM25 did not pool are still unjudged and count as non-relevant, so Recall@100 still favours BM25-like rankings.
In this set, 523 of 696 (75.1%) judged passages and 80 of 100 (80.0%) judged-relevant passages were proposed by BM25 (pooling or the BM25 runs). Judged passages by source: {'bge_zero_shot': 44, 'bge_zero_shot+bm25': 31, 'bge_zero_shot+bm25+e5_zero_shot': 24, 'bge_zero_shot+e5_zero_shot': 11, 'bge_zero_shot+e5_zero_shot+manual': 1, 'bm25': 442, 'bm25+e5_zero_shot': 25, 'bm25+stratified': 1, 'e5_zero_shot': 50, 'e5_zero_shot+manual': 1, 'e5_zero_shot+random': 1, 'manual': 3, 'random': 49, 'stratified': 13}; relevant passages by source: {'bge_zero_shot': 6, 'bge_zero_shot+bm25': 4, 'bge_zero_shot+bm25+e5_zero_shot': 12, 'bge_zero_shot+e5_zero_shot': 2, 'bge_zero_shot+e5_zero_shot+manual': 1, 'bm25': 57, 'bm25+e5_zero_shot': 7, 'e5_zero_shot': 7, 'e5_zero_shot+manual': 1, 'manual': 3}.

Retrieval modes:

- `global`: Each question ranks every passage in the corpus; no company restriction.
- `company`: Each question ranks only the passages whose ticker is one of the question's tickers (query metadata 'tickers'). Other passages are removed before ranking, never filtered out of a global top-k, and every retriever gets the same restriction.

> **Do not conclude that one retriever is better from these numbers. Labels are AI-generated development labels on 10 questions, the judged pools come mostly from BM25, and unjudged passages count as non-relevant. Read every score with its judged@10 and unjudged counts; a comparison needs pools that include the dense retrievers' results and human-judged labels.**

> **Every checkpoint is reported; none was selected, by this set or by any validation set. Checkpoints of a short run (the 15-step smoke test on 48 synthetic pairs) check the training mechanics: their scores are not evidence of a retrieval improvement, and none is claimed. Passages a checkpoint ranks in its top 10 that no stored run ranked there are unjudged and count as non-relevant, so read every score with its judged@10, its interval and the paired comparison.**

Stored references: the runs marked (stored reference) are the rankings in results/dev/pilot_v2 (each run file's SHA-256 checked), scored with this set's labels; they were not run, re-encoded or re-ranked, and are not written again.

## Comparison

| Retriever | Mode | nDCG@10 | MRR@10 | Recall@100 | judged@10 | nDCG@10 judged-only | MRR@10 judged-only | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | Insufficient coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 (stored reference) | global | 0.170 | 0.323 | 0.578 | 1.000 | 0.170 | 0.323 | 0 (0 / 0) | 82 | — |
| bm25 (stored reference) | company | 0.194 | 0.423 | 0.674 | 1.000 | 0.194 | 0.423 | 0 (0 / 0) | 79 | — |
| bge_zero_shot (stored reference) | global | 0.286 | 0.550 | 0.627 | 1.000 | 0.286 | 0.550 | 0 (0 / 0) | 77 | — |
| bge_zero_shot (stored reference) | company | 0.297 | 0.575 | 0.643 | 1.000 | 0.297 | 0.575 | 0 (0 / 0) | 75 | — |
| e5_zero_shot (stored reference) | global | 0.361 | 0.600 | 0.666 | 1.000 | 0.361 | 0.600 | 0 (0 / 0) | 71 | — |
| e5_zero_shot (stored reference) | company | 0.376 | 0.625 | 0.673 | 1.000 | 0.376 | 0.625 | 0 (0 / 0) | 70 | — |
| bge_ft_v2_smoke_step05 | global | 0.292 | 0.567 | 0.628 | 0.960 | 0.292 | 0.567 | 4 (4 / 0) | 72 | — |
| bge_ft_v2_smoke_step05 | company | 0.292 | 0.567 | 0.681 | 0.930 | 0.296 | 0.577 | 7 (7 / 0) | 69 | — |
| bge_ft_v2_smoke_step10 | global | 0.297 | 0.664 | 0.642 | 0.910 | 0.320 | 0.664 | 9 (9 / 0) | 69 | — |
| bge_ft_v2_smoke_step10 | company | 0.297 | 0.664 | 0.692 | 0.880 | 0.320 | 0.664 | 12 (12 / 0) | 66 | — |
| bge_ft_v2_smoke_step15 | global | 0.303 | 0.664 | 0.646 | 0.930 | 0.322 | 0.664 | 7 (7 / 0) | 70 | — |
| bge_ft_v2_smoke_step15 | company | 0.303 | 0.664 | 0.696 | 0.900 | 0.322 | 0.664 | 10 (10 / 0) | 67 | — |

Unjudged and confirmed-negative counts are summed over the top 10 of every question.

## Statistical uncertainty

Mean over 10 questions with a 95% interval: percentile bootstrap over questions (10000 resamples, seed 20261104); paired runs resample the same questions together; sign-flip randomization test of no difference (exact for up to 16 questions, else seeded Monte Carlo). With this few questions the intervals are wide; a difference whose interval includes zero is not evidence of a change.

| Retriever | Mode | ndcg@10 | mrr@10 | recall@100 | judged@10 |
| --- | --- | --- | --- | --- | --- |
| bm25 (stored reference) | global | 0.170 [0.061, 0.305] | 0.323 [0.108, 0.564] | 0.578 [0.385, 0.758] | 1.000 [1.000, 1.000] |
| bm25 (stored reference) | company | 0.194 [0.085, 0.329] | 0.423 [0.183, 0.679] | 0.674 [0.495, 0.843] | 1.000 [1.000, 1.000] |
| bge_zero_shot (stored reference) | global | 0.286 [0.145, 0.445] | 0.550 [0.300, 0.800] | 0.627 [0.455, 0.789] | 1.000 [1.000, 1.000] |
| bge_zero_shot (stored reference) | company | 0.297 [0.162, 0.450] | 0.575 [0.350, 0.800] | 0.643 [0.469, 0.800] | 1.000 [1.000, 1.000] |
| e5_zero_shot (stored reference) | global | 0.361 [0.210, 0.514] | 0.600 [0.340, 0.850] | 0.666 [0.501, 0.818] | 1.000 [1.000, 1.000] |
| e5_zero_shot (stored reference) | company | 0.376 [0.225, 0.524] | 0.625 [0.375, 0.864] | 0.673 [0.505, 0.823] | 1.000 [1.000, 1.000] |
| bge_ft_v2_smoke_step05 | global | 0.292 [0.156, 0.444] | 0.567 [0.317, 0.800] | 0.628 [0.487, 0.752] | 0.960 [0.920, 1.000] |
| bge_ft_v2_smoke_step05 | company | 0.292 [0.156, 0.444] | 0.567 [0.317, 0.800] | 0.681 [0.520, 0.821] | 0.930 [0.860, 0.990] |
| bge_ft_v2_smoke_step10 | global | 0.297 [0.163, 0.438] | 0.664 [0.379, 0.914] | 0.642 [0.498, 0.770] | 0.910 [0.840, 0.970] |
| bge_ft_v2_smoke_step10 | company | 0.297 [0.163, 0.438] | 0.664 [0.379, 0.914] | 0.692 [0.527, 0.833] | 0.880 [0.800, 0.950] |
| bge_ft_v2_smoke_step15 | global | 0.303 [0.167, 0.444] | 0.664 [0.379, 0.914] | 0.646 [0.500, 0.776] | 0.930 [0.860, 0.980] |
| bge_ft_v2_smoke_step15 | company | 0.303 [0.167, 0.444] | 0.664 [0.379, 0.914] | 0.696 [0.530, 0.839] | 0.900 [0.820, 0.970] |

### Paired comparison with `bge_zero_shot`

Each run minus `bge_zero_shot` in the same mode, question by question: the mean difference with a 95% paired bootstrap interval, wins / ties / losses, and the p-value of a two-sided sign-flip test of no difference.

| Retriever | Mode | ndcg@10 | mrr@10 | recall@100 | judged@10 |
| --- | --- | --- | --- | --- | --- |
| bm25 (stored reference) | global | -0.116 [-0.255, +0.004]; 3/1/6; p 0.160 | -0.227 [-0.517, +0.075]; 2/3/5; p 0.219 | -0.048 [-0.257, +0.173]; 5/0/5; p 0.676 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| bm25 (stored reference) | company | -0.103 [-0.249, +0.028]; 4/0/6; p 0.234 | -0.152 [-0.492, +0.200]; 3/2/5; p 0.453 | +0.031 [-0.184, +0.249]; 5/1/4; p 0.785 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| e5_zero_shot (stored reference) | global | +0.074 [-0.053, +0.196]; 5/3/2; p 0.312 | +0.050 [-0.243, +0.322]; 4/4/2; p 0.750 | +0.040 [+0.000, +0.083]; 3/7/0; p 0.250 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| e5_zero_shot (stored reference) | company | +0.078 [-0.051, +0.201]; 6/1/3; p 0.270 | +0.050 [-0.246, +0.325]; 5/2/3; p 0.742 | +0.029 [-0.033, +0.096]; 3/6/1; p 0.500 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| bge_ft_v2_smoke_step05 | global | +0.006 [-0.026, +0.033]; 4/4/2; p 0.688 | +0.017 [-0.075, +0.100]; 2/7/1; p 1.000 | +0.001 [-0.092, +0.083]; 3/4/3; p 1.000 | -0.040 [-0.080, +0.000]; 0/7/3; p 0.250 |
| bge_ft_v2_smoke_step05 | company | -0.005 [-0.036, +0.020]; 4/3/3; p 0.750 | -0.008 [-0.117, +0.100]; 2/6/2; p 1.000 | +0.038 [-0.027, +0.130]; 2/6/2; p 0.625 | -0.070 [-0.140, -0.010]; 0/6/4; p 0.125 |
| bge_ft_v2_smoke_step10 | global | +0.011 [-0.054, +0.081]; 3/4/3; p 0.781 | +0.114 [-0.071, +0.339]; 2/7/1; p 0.500 | +0.015 [-0.096, +0.128]; 3/4/3; p 0.844 | -0.090 [-0.160, -0.030]; 0/5/5; p 0.062 |
| bge_ft_v2_smoke_step10 | company | -0.000 [-0.063, +0.065]; 3/3/4; p 1.000 | +0.089 [-0.111, +0.314]; 2/6/2; p 0.500 | +0.048 [-0.029, +0.145]; 3/5/2; p 0.438 | -0.120 [-0.200, -0.050]; 0/4/6; p 0.031 |
| bge_ft_v2_smoke_step15 | global | +0.017 [-0.048, +0.087]; 4/4/2; p 0.656 | +0.114 [-0.071, +0.339]; 2/7/1; p 0.500 | +0.019 [-0.091, +0.133]; 4/3/3; p 0.750 | -0.070 [-0.140, -0.020]; 0/6/4; p 0.125 |
| bge_ft_v2_smoke_step15 | company | +0.006 [-0.058, +0.070]; 4/3/3; p 0.844 | +0.089 [-0.111, +0.314]; 2/6/2; p 0.500 | +0.053 [-0.029, +0.149]; 4/4/2; p 0.344 | -0.100 [-0.180, -0.030]; 0/5/5; p 0.062 |

## Relevant retrieval failures

A relevant retrieval failure is a passage judged relevant (grade >= 1) that a run ranks below its top 10, or does not return at all in its top 100; a question with no judged-relevant passage in the top 10 has failed.

| Retriever | Mode | Questions without a relevant passage in the top 10 | Relevant passages outside the top 10 | Of them, not in the top 100 |
| --- | --- | --- | --- | --- |
| bm25 (stored reference) | global | q_0001, q_0002, q_0003, q_0006 | 82 | 40 |
| bm25 (stored reference) | company | q_0001, q_0002, q_0003 | 79 | 33 |
| bge_zero_shot (stored reference) | global | q_0006, q_0010 | 77 | 35 |
| bge_zero_shot (stored reference) | company | q_0010 | 75 | 34 |
| e5_zero_shot (stored reference) | global | q_0006 | 71 | 30 |
| e5_zero_shot (stored reference) | company | q_0006 | 70 | 29 |
| bge_ft_v2_smoke_step05 | global | q_0006, q_0010 | 76 | 35 |
| bge_ft_v2_smoke_step05 | company | q_0006, q_0010 | 76 | 33 |
| bge_ft_v2_smoke_step10 | global | q_0006, q_0010 | 78 | 35 |
| bge_ft_v2_smoke_step10 | company | q_0006, q_0010 | 78 | 33 |
| bge_ft_v2_smoke_step15 | global | q_0006, q_0010 | 77 | 34 |
| bge_ft_v2_smoke_step15 | company | q_0006, q_0010 | 77 | 32 |

### judged@10 per question

| Question | bm25 global | bm25 company | bge_zero_shot global | bge_zero_shot company | e5_zero_shot global | e5_zero_shot company | bge_ft_v2_smoke_step05 global | bge_ft_v2_smoke_step05 company | bge_ft_v2_smoke_step10 global | bge_ft_v2_smoke_step10 company | bge_ft_v2_smoke_step15 global | bge_ft_v2_smoke_step15 company |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0002 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 0.800 | 0.800 | 0.800 | 0.800 |
| q_0003 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0004 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.800 | 0.700 | 0.700 | 0.700 | 0.700 |
| q_0005 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 | 0.900 | 1.000 | 1.000 |
| q_0006 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.700 | 1.000 | 0.700 | 1.000 | 0.700 |
| q_0007 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 | 0.900 |
| q_0008 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0009 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.800 | 0.900 | 0.900 |
| q_0010 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Retrievers

- `bm25`: BM25: tokenizer `word`, backend `python`, k1 1.5, b 0.75.
- `bge_zero_shot`: `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `622f038aa3e3358f81d8466c911d02e452746fbc66a3cfff37747dda62f7800a`; cache key `f04c62fe0721bcbb`.
- `e5_zero_shot`: `intfloat/e5-base-v2` at revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`; query prefix `'query: '`, passage prefix `'passage: '`; 768 dimensions, mean pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9208 of 10137 (90.8%) passages truncated, 27.7% of passage tokens beyond the limit; embeddings SHA-256 `73cf951ea6c09a038e8d4062a8ba4c56fd4c7d238c2ee71f5132fea0f69d3d25`; cache key `5ac696a9075b5dfc`.
- `bge_ft_v2_smoke_step05`: local checkpoint `models/bge_base_ft_v2_smoke/bge_base_ft_v2_smoke-seed42/checkpoint-step-000005` (run `bge_base_ft_v2_smoke-seed42`, step 5 of 15, files digest `18e9c34fca6d5b80`), fine-tuned from `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` on 48 synthetic pairs of `train_candidates_v1`; selection: none: fixed-step checkpoint; no development or evaluation metric was computed; query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `c608c3ad176bf7a15890dd4e382dd863a01439a13cfb2cc2d50274775c8c7d24`; cache key `30fe1a0c7339ce74`; evaluation isolation: 667 of 667 judged passages on the training blocklist; 0 of 188 passages of the 66 training candidates (train_candidates_v1) judged; highest training-question similarity to an evaluation question 0.28 (threshold 0.8).
- `bge_ft_v2_smoke_step10`: local checkpoint `models/bge_base_ft_v2_smoke/bge_base_ft_v2_smoke-seed42/checkpoint-step-000010` (run `bge_base_ft_v2_smoke-seed42`, step 10 of 15, files digest `0e23bde33a98cefd`), fine-tuned from `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` on 48 synthetic pairs of `train_candidates_v1`; selection: none: fixed-step checkpoint; no development or evaluation metric was computed; query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `1fbd640df6d8c05bb04fde6ec0694241f7cb46e82526a5c9434043974d0e2958`; cache key `9c5a7c60f9d9310d`; evaluation isolation: 667 of 667 judged passages on the training blocklist; 0 of 188 passages of the 66 training candidates (train_candidates_v1) judged; highest training-question similarity to an evaluation question 0.28 (threshold 0.8).
- `bge_ft_v2_smoke_step15`: local checkpoint `models/bge_base_ft_v2_smoke/bge_base_ft_v2_smoke-seed42/checkpoint-step-000015` (run `bge_base_ft_v2_smoke-seed42`, step 15 of 15, files digest `007a8ef1c996d516`), fine-tuned from `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` on 48 synthetic pairs of `train_candidates_v1`; selection: none: fixed-step checkpoint; no development or evaluation metric was computed; query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `bcfcb9430a24d467624d61147a6bde45520bc1c1d92e93e8d82a9cfb838ff4f1`; cache key `f9547e7d8994e9ce`; evaluation isolation: 667 of 667 judged passages on the training blocklist; 0 of 188 passages of the 66 training candidates (train_candidates_v1) judged; highest training-question similarity to an evaluation question 0.28 (threshold 0.8).

## bm25, global

Top 10 over 10 questions: 18 judged relevant, 82 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.000 | 0.000 | 0.000 | 1.000 | 0 (0 / 0) | 10 | — | 6 (6) | — |
| q_0002 | 3 | 0.000 | 0.000 | 0.667 | 1.000 | 0 (0 / 0) | 10 | 11 | 3 (1) | — |
| q_0003 | 5 | 0.000 | 0.000 | 0.600 | 1.000 | 0 (0 / 0) | 10 | 11 | 5 (2) | — |
| q_0004 | 8 | 0.199 | 0.250 | 0.750 | 1.000 | 0 (0 / 0) | 6 | 4 | 4 (2) | — |
| q_0005 | 16 | 0.114 | 1.000 | 0.438 | 1.000 | 0 (0 / 0) | 9 | 1 | 15 (9) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.333 | 1.000 | 0 (0 / 0) | 10 | 21 | 6 (4) | — |
| q_0007 | 23 | 0.672 | 1.000 | 0.913 | 1.000 | 0 (0 / 0) | 4 | 1 | 17 (2) | — |
| q_0008 | 12 | 0.268 | 0.333 | 0.833 | 1.000 | 0 (0 / 0) | 7 | 3 | 9 (2) | — |
| q_0009 | 16 | 0.313 | 0.500 | 0.250 | 1.000 | 0 (0 / 0) | 7 | 2 | 13 (12) | — |
| q_0010 | 5 | 0.139 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 7 | 4 (0) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bm25, company

Top 10 over 10 questions: 21 judged relevant, 79 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 64 | 6 (5) | — |
| q_0002 | 3 | 0.000 | 0.000 | 1.000 | 1.000 | 0 (0 / 0) | 10 | 11 | 3 (0) | — |
| q_0003 | 5 | 0.000 | 0.000 | 0.600 | 1.000 | 0 (0 / 0) | 10 | 11 | 5 (2) | — |
| q_0004 | 8 | 0.199 | 0.250 | 0.875 | 1.000 | 0 (0 / 0) | 6 | 4 | 4 (1) | — |
| q_0005 | 16 | 0.114 | 1.000 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 1 | 15 (8) | — |
| q_0006 | 6 | 0.211 | 1.000 | 0.500 | 1.000 | 0 (0 / 0) | 8 | 1 | 4 (3) | — |
| q_0007 | 23 | 0.700 | 1.000 | 0.957 | 1.000 | 0 (0 / 0) | 3 | 1 | 16 (1) | — |
| q_0008 | 12 | 0.268 | 0.333 | 0.833 | 1.000 | 0 (0 / 0) | 7 | 3 | 9 (2) | — |
| q_0009 | 16 | 0.313 | 0.500 | 0.312 | 1.000 | 0 (0 / 0) | 7 | 2 | 13 (11) | — |
| q_0010 | 5 | 0.139 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 7 | 4 (0) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_zero_shot, global

Top 10 over 10 questions: 23 judged relevant, 77 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.197 | 0.250 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 4 | 5 (3) | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 1 | 2 (0) | — |
| q_0003 | 5 | 0.441 | 0.500 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 2 | 3 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.375 | 1.000 | 0 (0 / 0) | 9 | 1 | 7 (5) | — |
| q_0005 | 16 | 0.087 | 0.250 | 0.812 | 1.000 | 0 (0 / 0) | 8 | 4 | 14 (3) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 18 | 6 (5) | — |
| q_0007 | 23 | 0.809 | 1.000 | 0.826 | 1.000 | 0 (0 / 0) | 2 | 1 | 15 (4) | — |
| q_0008 | 12 | 0.299 | 1.000 | 0.750 | 1.000 | 0 (0 / 0) | 7 | 1 | 9 (3) | — |
| q_0009 | 16 | 0.337 | 0.500 | 0.438 | 1.000 | 0 (0 / 0) | 5 | 2 | 11 (9) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.400 | 1.000 | 0 (0 / 0) | 10 | 49 | 5 (3) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_zero_shot, company

Top 10 over 10 questions: 25 judged relevant, 75 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.241 | 0.250 | 0.667 | 1.000 | 0 (0 / 0) | 8 | 4 | 4 (2) | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 1 | 2 (0) | — |
| q_0003 | 5 | 0.441 | 0.500 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 2 | 3 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.375 | 1.000 | 0 (0 / 0) | 9 | 1 | 7 (5) | — |
| q_0005 | 16 | 0.087 | 0.250 | 0.812 | 1.000 | 0 (0 / 0) | 8 | 4 | 14 (3) | — |
| q_0006 | 6 | 0.066 | 0.250 | 0.167 | 1.000 | 0 (0 / 0) | 9 | 4 | 5 (5) | — |
| q_0007 | 23 | 0.809 | 1.000 | 0.826 | 1.000 | 0 (0 / 0) | 2 | 1 | 15 (4) | — |
| q_0008 | 12 | 0.299 | 1.000 | 0.750 | 1.000 | 0 (0 / 0) | 7 | 1 | 9 (3) | — |
| q_0009 | 16 | 0.337 | 0.500 | 0.438 | 1.000 | 0 (0 / 0) | 5 | 2 | 11 (9) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.400 | 1.000 | 0 (0 / 0) | 10 | 49 | 5 (3) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## e5_zero_shot, global

Top 10 over 10 questions: 29 judged relevant, 71 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.197 | 0.250 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 4 | 5 (3) | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 1 | 2 (0) | — |
| q_0003 | 5 | 0.737 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 7 | 1 | 2 (0) | — |
| q_0004 | 8 | 0.046 | 0.143 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 7 | 7 (4) | — |
| q_0005 | 16 | 0.425 | 1.000 | 0.812 | 1.000 | 0 (0 / 0) | 5 | 1 | 11 (3) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 30 | 6 (5) | — |
| q_0007 | 23 | 0.466 | 0.500 | 0.826 | 1.000 | 0 (0 / 0) | 4 | 2 | 17 (4) | — |
| q_0008 | 12 | 0.374 | 1.000 | 0.833 | 1.000 | 0 (0 / 0) | 5 | 1 | 7 (2) | — |
| q_0009 | 16 | 0.681 | 1.000 | 0.625 | 1.000 | 0 (0 / 0) | 4 | 1 | 10 (6) | — |
| q_0010 | 5 | 0.125 | 0.111 | 0.400 | 1.000 | 0 (0 / 0) | 9 | 9 | 4 (3) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## e5_zero_shot, company

Top 10 over 10 questions: 30 judged relevant, 70 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.347 | 0.500 | 0.500 | 1.000 | 0 (0 / 0) | 8 | 2 | 4 (3) | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 1 | 2 (0) | — |
| q_0003 | 5 | 0.737 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 7 | 1 | 2 (0) | — |
| q_0004 | 8 | 0.046 | 0.143 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 7 | 7 (4) | — |
| q_0005 | 16 | 0.425 | 1.000 | 0.812 | 1.000 | 0 (0 / 0) | 5 | 1 | 11 (3) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 12 | 6 (5) | — |
| q_0007 | 23 | 0.466 | 0.500 | 0.826 | 1.000 | 0 (0 / 0) | 4 | 2 | 17 (4) | — |
| q_0008 | 12 | 0.374 | 1.000 | 0.833 | 1.000 | 0 (0 / 0) | 5 | 1 | 7 (2) | — |
| q_0009 | 16 | 0.681 | 1.000 | 0.688 | 1.000 | 0 (0 / 0) | 4 | 1 | 10 (5) | — |
| q_0010 | 5 | 0.125 | 0.111 | 0.400 | 1.000 | 0 (0 / 0) | 9 | 9 | 4 (3) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_ft_v2_smoke_step05, global

Top 10 over 10 questions: 24 judged relevant, 72 confirmed negatives, 4 unjudged (4 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.288 | 0.500 | 0.667 | 1.000 | 0 (0 / 0) | 9 | 2 | 5 (2) | — |
| q_0002 | 3 | 0.556 | 1.000 | 0.667 | 0.900 | 1 (1 / 0) | 8 | 1 | 2 (1) | — |
| q_0003 | 5 | 0.332 | 0.167 | 1.000 | 1.000 | 0 (0 / 0) | 7 | 6 | 2 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.500 | 0.800 | 2 (2 / 0) | 7 | 1 | 7 (4) | — |
| q_0005 | 16 | 0.116 | 0.500 | 0.750 | 0.900 | 1 (1 / 0) | 7 | 2 | 14 (4) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 34 | 6 (5) | — |
| q_0007 | 23 | 0.792 | 1.000 | 0.826 | 1.000 | 0 (0 / 0) | 2 | 1 | 15 (4) | — |
| q_0008 | 12 | 0.334 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 7 | 1 | 9 (4) | — |
| q_0009 | 16 | 0.363 | 0.500 | 0.438 | 1.000 | 0 (0 / 0) | 5 | 2 | 11 (9) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.600 | 1.000 | 0 (0 / 0) | 10 | 39 | 5 (2) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_ft_v2_smoke_step05, company

Top 10 over 10 questions: 24 judged relevant, 69 confirmed negatives, 7 unjudged (7 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.288 | 0.500 | 0.667 | 1.000 | 0 (0 / 0) | 9 | 2 | 5 (2) | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 0.900 | 1 (1 / 0) | 8 | 1 | 2 (0) | — |
| q_0003 | 5 | 0.332 | 0.167 | 1.000 | 1.000 | 0 (0 / 0) | 7 | 6 | 2 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.500 | 0.800 | 2 (2 / 0) | 7 | 1 | 7 (4) | — |
| q_0005 | 16 | 0.116 | 0.500 | 0.750 | 0.900 | 1 (1 / 0) | 7 | 2 | 14 (4) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 0.700 | 3 (3 / 0) | 7 | 14 | 6 (5) | — |
| q_0007 | 23 | 0.792 | 1.000 | 0.826 | 1.000 | 0 (0 / 0) | 2 | 1 | 15 (4) | — |
| q_0008 | 12 | 0.334 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 7 | 1 | 9 (4) | — |
| q_0009 | 16 | 0.363 | 0.500 | 0.438 | 1.000 | 0 (0 / 0) | 5 | 2 | 11 (9) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.800 | 1.000 | 0 (0 / 0) | 10 | 39 | 5 (1) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_ft_v2_smoke_step10, global

Top 10 over 10 questions: 22 judged relevant, 69 confirmed negatives, 9 unjudged (9 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.457 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 9 | 1 | 5 (2) | — |
| q_0002 | 3 | 0.556 | 1.000 | 0.667 | 0.800 | 2 (2 / 0) | 7 | 1 | 2 (1) | — |
| q_0003 | 5 | 0.259 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 7 | 3 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.500 | 0.700 | 3 (3 / 0) | 6 | 1 | 7 (4) | — |
| q_0005 | 16 | 0.162 | 1.000 | 0.688 | 0.900 | 1 (1 / 0) | 7 | 1 | 14 (5) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 56 | 6 (5) | — |
| q_0007 | 23 | 0.725 | 1.000 | 0.826 | 0.900 | 1 (1 / 0) | 2 | 1 | 16 (4) | — |
| q_0008 | 12 | 0.385 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 5 | 1 | 7 (4) | — |
| q_0009 | 16 | 0.289 | 0.500 | 0.438 | 0.800 | 2 (2 / 0) | 5 | 2 | 13 (9) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.800 | 1.000 | 0 (0 / 0) | 10 | 32 | 5 (1) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_ft_v2_smoke_step10, company

Top 10 over 10 questions: 22 judged relevant, 66 confirmed negatives, 12 unjudged (12 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.457 | 1.000 | 0.833 | 1.000 | 0 (0 / 0) | 9 | 1 | 5 (1) | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 0.800 | 2 (2 / 0) | 7 | 1 | 2 (0) | — |
| q_0003 | 5 | 0.259 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 7 | 3 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.500 | 0.700 | 3 (3 / 0) | 6 | 1 | 7 (4) | — |
| q_0005 | 16 | 0.162 | 1.000 | 0.688 | 0.900 | 1 (1 / 0) | 7 | 1 | 14 (5) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 0.700 | 3 (3 / 0) | 7 | 31 | 6 (5) | — |
| q_0007 | 23 | 0.725 | 1.000 | 0.826 | 0.900 | 1 (1 / 0) | 2 | 1 | 16 (4) | — |
| q_0008 | 12 | 0.385 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 5 | 1 | 7 (4) | — |
| q_0009 | 16 | 0.289 | 0.500 | 0.438 | 0.800 | 2 (2 / 0) | 5 | 2 | 13 (9) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.800 | 1.000 | 0 (0 / 0) | 10 | 32 | 5 (1) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_ft_v2_smoke_step15, global

Top 10 over 10 questions: 23 judged relevant, 70 confirmed negatives, 7 unjudged (7 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.457 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 9 | 1 | 5 (2) | — |
| q_0002 | 3 | 0.556 | 1.000 | 0.667 | 0.800 | 2 (2 / 0) | 7 | 1 | 2 (1) | — |
| q_0003 | 5 | 0.250 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 7 | 3 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.500 | 0.700 | 3 (3 / 0) | 6 | 1 | 7 (4) | — |
| q_0005 | 16 | 0.162 | 1.000 | 0.688 | 1.000 | 0 (0 / 0) | 8 | 1 | 14 (5) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 71 | 6 (5) | — |
| q_0007 | 23 | 0.725 | 1.000 | 0.870 | 0.900 | 1 (1 / 0) | 2 | 1 | 16 (3) | — |
| q_0008 | 12 | 0.390 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 5 | 1 | 7 (4) | — |
| q_0009 | 16 | 0.354 | 0.500 | 0.438 | 0.900 | 1 (1 / 0) | 5 | 2 | 12 (9) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.800 | 1.000 | 0 (0 / 0) | 10 | 30 | 5 (1) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_ft_v2_smoke_step15, company

Top 10 over 10 questions: 23 judged relevant, 67 confirmed negatives, 10 unjudged (10 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.457 | 1.000 | 0.833 | 1.000 | 0 (0 / 0) | 9 | 1 | 5 (1) | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 0.800 | 2 (2 / 0) | 7 | 1 | 2 (0) | — |
| q_0003 | 5 | 0.250 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 7 | 3 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.500 | 0.700 | 3 (3 / 0) | 6 | 1 | 7 (4) | — |
| q_0005 | 16 | 0.162 | 1.000 | 0.688 | 1.000 | 0 (0 / 0) | 8 | 1 | 14 (5) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 0.700 | 3 (3 / 0) | 7 | 45 | 6 (5) | — |
| q_0007 | 23 | 0.725 | 1.000 | 0.870 | 0.900 | 1 (1 / 0) | 2 | 1 | 16 (3) | — |
| q_0008 | 12 | 0.390 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 5 | 1 | 7 (4) | — |
| q_0009 | 16 | 0.354 | 0.500 | 0.438 | 0.900 | 1 (1 / 0) | 5 | 2 | 12 (9) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.800 | 1.000 | 0 (0 / 0) | 10 | 30 | 5 (1) | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## Questions

- `q_0001` What risks does Apple's fiscal 2024 annual report associate with having most of its hardware built by a few third-party manufacturers abroad?
- `q_0002` Which offerings did Apple credit for the growth in its services revenue in fiscal 2021?
- `q_0003` What reasons did Amazon give for the change in profit from its AWS cloud segment in 2023?
- `q_0004` What average liquidity coverage ratio did JPMorgan Chase report for the firm in the fourth quarter of 2021?
- `q_0005` How much did weaker demand for Pfizer's COVID-19 vaccine and oral antiviral pill reduce its revenues in 2023 versus 2022?
- `q_0006` How much of Tesla's revolving credit facility was undrawn at the end of 2023?
- `q_0007` How did Tesla's revenue from selling emissions credits to other automakers change each year from 2019 through 2023?
- `q_0008` How were Walmart's net sales for the fiscal year ended January 31, 2024 split among its three reportable segments?
- `q_0009` How did ExxonMobil's 2022 capital spending break down by business segment and between U.S. and non-U.S. operations?
- `q_0010` How does ExxonMobil's 2020 annual report say that output decisions by oil-producing nations, such as OPEC members, can affect its business?
