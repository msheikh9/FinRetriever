# dev_pilot_v3_a1: bm25, bge_zero_shot, e5_zero_shot, a0_step0, a1_seed42, a1_seed1337, a1_seed2026

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

> **Each validation-selected checkpoint was chosen by its run's fixed rule on synthetic validation questions accepted by an AI review (not gold), never by this set, and every seed of its run is reported. Passages it ranks in its top 10 that no stored run ranked there are unjudged and count as non-relevant (pools are never deepened with a fine-tuned model's output), so its scores are lower bounds: read each with its judged@10, its judged-only scores, its interval and the paired comparison. Development labels support no claim of a retrieval improvement.**

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
| a0_step0 | global | 0.286 | 0.550 | 0.627 | 1.000 | 0.286 | 0.550 | 0 (0 / 0) | 77 | — |
| a0_step0 | company | 0.297 | 0.575 | 0.643 | 1.000 | 0.297 | 0.575 | 0 (0 / 0) | 75 | — |
| a1_seed42 | global | 0.320 | 0.558 | 0.769 | 0.820 | 0.357 | 0.577 | 18 (16 / 2) | 51 | q_0004 |
| a1_seed42 | company | 0.328 | 0.592 | 0.839 | 0.810 | 0.368 | 0.667 | 19 (19 / 0) | 49 | q_0004, q_0006 |
| a1_seed1337 | global | 0.324 | 0.583 | 0.750 | 0.850 | 0.365 | 0.593 | 15 (11 / 4) | 54 | q_0004 |
| a1_seed1337 | company | 0.332 | 0.617 | 0.833 | 0.830 | 0.376 | 0.683 | 17 (17 / 0) | 51 | q_0004, q_0006 |
| a1_seed2026 | global | 0.307 | 0.667 | 0.806 | 0.800 | 0.360 | 0.667 | 20 (16 / 4) | 53 | q_0002, q_0004 |
| a1_seed2026 | company | 0.323 | 0.717 | 0.839 | 0.800 | 0.375 | 0.767 | 20 (20 / 0) | 51 | q_0004, q_0006 |

Unjudged and confirmed-negative counts are summed over the top 10 of every question.

## Statistical uncertainty

Mean over 10 questions with a 95% interval: percentile bootstrap over questions (10000 resamples, seed 20261104); paired runs resample the same questions together; sign-flip randomization test of no difference (exact for up to 16 questions, else seeded Monte Carlo). With this few questions the intervals are wide; a difference whose interval includes zero is not evidence of a change.

| Retriever | Mode | ndcg@10 | mrr@10 | recall@10 | recall@100 | judged@10 |
| --- | --- | --- | --- | --- | --- | --- |
| bm25 (stored reference) | global | 0.170 [0.061, 0.305] | 0.323 [0.108, 0.564] | 0.146 [0.057, 0.250] | 0.578 [0.385, 0.758] | 1.000 [1.000, 1.000] |
| bm25 (stored reference) | company | 0.194 [0.085, 0.329] | 0.423 [0.183, 0.679] | 0.184 [0.090, 0.287] | 0.674 [0.495, 0.843] | 1.000 [1.000, 1.000] |
| bge_zero_shot (stored reference) | global | 0.286 [0.145, 0.445] | 0.550 [0.300, 0.800] | 0.206 [0.122, 0.289] | 0.627 [0.455, 0.789] | 1.000 [1.000, 1.000] |
| bge_zero_shot (stored reference) | company | 0.297 [0.162, 0.450] | 0.575 [0.350, 0.800] | 0.239 [0.161, 0.311] | 0.643 [0.469, 0.800] | 1.000 [1.000, 1.000] |
| e5_zero_shot (stored reference) | global | 0.361 [0.210, 0.514] | 0.600 [0.340, 0.850] | 0.279 [0.181, 0.379] | 0.666 [0.501, 0.818] | 1.000 [1.000, 1.000] |
| e5_zero_shot (stored reference) | company | 0.376 [0.225, 0.524] | 0.625 [0.375, 0.864] | 0.296 [0.199, 0.390] | 0.673 [0.505, 0.823] | 1.000 [1.000, 1.000] |
| a0_step0 | global | 0.286 [0.145, 0.445] | 0.550 [0.300, 0.800] | 0.206 [0.122, 0.289] | 0.627 [0.455, 0.789] | 1.000 [1.000, 1.000] |
| a0_step0 | company | 0.297 [0.162, 0.450] | 0.575 [0.350, 0.800] | 0.239 [0.161, 0.311] | 0.643 [0.469, 0.800] | 1.000 [1.000, 1.000] |
| a1_seed42 | global | 0.320 [0.176, 0.473] | 0.558 [0.308, 0.800] | 0.327 [0.175, 0.485] | 0.769 [0.638, 0.879] | 0.820 [0.690, 0.910] |
| a1_seed42 | company | 0.328 [0.193, 0.477] | 0.592 [0.367, 0.808] | 0.344 [0.206, 0.491] | 0.839 [0.747, 0.924] | 0.810 [0.660, 0.930] |
| a1_seed1337 | global | 0.324 [0.179, 0.479] | 0.583 [0.333, 0.800] | 0.327 [0.175, 0.485] | 0.750 [0.621, 0.868] | 0.850 [0.730, 0.940] |
| a1_seed1337 | company | 0.332 [0.194, 0.482] | 0.617 [0.400, 0.833] | 0.344 [0.206, 0.491] | 0.833 [0.739, 0.920] | 0.830 [0.700, 0.940] |
| a1_seed2026 | global | 0.307 [0.167, 0.463] | 0.667 [0.400, 0.933] | 0.257 [0.143, 0.372] | 0.806 [0.695, 0.908] | 0.800 [0.700, 0.890] |
| a1_seed2026 | company | 0.323 [0.187, 0.476] | 0.717 [0.483, 0.933] | 0.307 [0.187, 0.435] | 0.839 [0.752, 0.920] | 0.800 [0.690, 0.900] |

### Paired comparison with `bge_zero_shot`

Each run minus `bge_zero_shot` in the same mode, question by question: the mean difference with a 95% paired bootstrap interval, wins / ties / losses, and the p-value of a two-sided sign-flip test of no difference.

| Retriever | Mode | ndcg@10 | mrr@10 | recall@10 | recall@100 | judged@10 |
| --- | --- | --- | --- | --- | --- | --- |
| bm25 (stored reference) | global | -0.116 [-0.255, +0.004]; 3/1/6; p 0.160 | -0.227 [-0.517, +0.075]; 2/3/5; p 0.219 | -0.060 [-0.190, +0.078]; 2/2/6; p 0.422 | -0.048 [-0.257, +0.173]; 5/0/5; p 0.676 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| bm25 (stored reference) | company | -0.103 [-0.249, +0.028]; 4/0/6; p 0.234 | -0.152 [-0.492, +0.200]; 3/2/5; p 0.453 | -0.056 [-0.199, +0.098]; 3/1/6; p 0.496 | +0.031 [-0.184, +0.249]; 5/1/4; p 0.785 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| e5_zero_shot (stored reference) | global | +0.074 [-0.053, +0.196]; 5/3/2; p 0.312 | +0.050 [-0.243, +0.322]; 4/4/2; p 0.750 | +0.073 [+0.011, +0.134]; 5/4/1; p 0.094 | +0.040 [+0.000, +0.083]; 3/7/0; p 0.250 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| e5_zero_shot (stored reference) | company | +0.078 [-0.051, +0.201]; 6/1/3; p 0.270 | +0.050 [-0.246, +0.325]; 5/2/3; p 0.742 | +0.056 [-0.022, +0.129]; 5/3/2; p 0.219 | +0.029 [-0.033, +0.096]; 3/6/1; p 0.500 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| a0_step0 | global | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| a0_step0 | company | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 | +0.000 [+0.000, +0.000]; 0/10/0; p 1.000 |
| a1_seed42 | global | +0.034 [-0.025, +0.091]; 4/3/3; p 0.328 | +0.008 [-0.167, +0.208]; 2/5/3; p 1.000 | +0.121 [+0.037, +0.213]; 5/5/0; p 0.062 | +0.142 [+0.057, +0.234]; 7/2/1; p 0.016 | -0.180 [-0.310, -0.090]; 0/2/8; p 0.008 |
| a1_seed42 | company | +0.031 [-0.025, +0.086]; 5/2/3; p 0.305 | +0.017 [-0.167, +0.217]; 3/4/3; p 0.906 | +0.105 [+0.019, +0.201]; 4/6/0; p 0.125 | +0.196 [+0.074, +0.335]; 7/2/1; p 0.016 | -0.190 [-0.340, -0.070]; 0/3/7; p 0.016 |
| a1_seed1337 | global | +0.038 [-0.014, +0.086]; 6/3/1; p 0.172 | +0.033 [-0.133, +0.225]; 2/6/2; p 0.750 | +0.121 [+0.037, +0.213]; 5/5/0; p 0.062 | +0.123 [-0.020, +0.266]; 7/1/2; p 0.148 | -0.150 [-0.270, -0.060]; 0/3/7; p 0.016 |
| a1_seed1337 | company | +0.035 [-0.015, +0.084]; 7/2/1; p 0.258 | +0.042 [-0.125, +0.233]; 3/5/2; p 0.688 | +0.105 [+0.019, +0.201]; 4/6/0; p 0.125 | +0.190 [+0.071, +0.327]; 7/2/1; p 0.016 | -0.170 [-0.300, -0.060]; 0/3/7; p 0.016 |
| a1_seed2026 | global | +0.020 [-0.005, +0.049]; 4/4/2; p 0.219 | +0.117 [-0.067, +0.342]; 2/6/2; p 0.500 | +0.051 [+0.001, +0.107]; 4/5/1; p 0.188 | +0.179 [+0.065, +0.310]; 7/2/1; p 0.023 | -0.200 [-0.300, -0.110]; 0/2/8; p 0.008 |
| a1_seed2026 | company | +0.026 [+0.005, +0.048]; 6/2/2; p 0.055 | +0.142 [-0.042, +0.358]; 3/5/2; p 0.312 | +0.068 [+0.002, +0.144]; 4/5/1; p 0.188 | +0.196 [+0.069, +0.339]; 7/2/1; p 0.023 | -0.200 [-0.310, -0.100]; 0/2/8; p 0.008 |

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
| a0_step0 | global | q_0006, q_0010 | 77 | 35 |
| a0_step0 | company | q_0010 | 75 | 34 |
| a1_seed42 | global | q_0006, q_0010 | 69 | 21 |
| a1_seed42 | company | q_0010 | 68 | 17 |
| a1_seed1337 | global | q_0006, q_0010 | 69 | 22 |
| a1_seed1337 | company | q_0010 | 68 | 18 |
| a1_seed2026 | global | q_0006, q_0010 | 73 | 19 |
| a1_seed2026 | company | q_0010 | 71 | 17 |

### judged@10 per question

| Question | bm25 global | bm25 company | bge_zero_shot global | bge_zero_shot company | e5_zero_shot global | e5_zero_shot company | a0_step0 global | a0_step0 company | a1_seed42 global | a1_seed42 company | a1_seed1337 global | a1_seed1337 company | a1_seed2026 global | a1_seed2026 company |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0002 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 | 0.700 | 0.900 | 0.600 | 0.900 |
| q_0003 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 |
| q_0004 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.400 | 0.400 | 0.500 | 0.500 |
| q_0005 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 |
| q_0006 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.500 | 0.900 | 0.500 | 0.900 | 0.500 |
| q_0007 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 | 0.900 | 1.000 | 1.000 |
| q_0008 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 1.000 | 1.000 | 0.800 | 0.800 |
| q_0009 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 | 0.900 | 0.700 | 0.800 |
| q_0010 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.800 | 0.900 | 0.900 | 0.800 | 0.800 |

## Retrievers

- `bm25`: BM25: tokenizer `word`, backend `python`, k1 1.5, b 0.75.
- `bge_zero_shot`: `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `622f038aa3e3358f81d8466c911d02e452746fbc66a3cfff37747dda62f7800a`; cache key `f04c62fe0721bcbb`.
- `e5_zero_shot`: `intfloat/e5-base-v2` at revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`; query prefix `'query: '`, passage prefix `'passage: '`; 768 dimensions, mean pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9208 of 10137 (90.8%) passages truncated, 27.7% of passage tokens beyond the limit; embeddings SHA-256 `73cf951ea6c09a038e8d4062a8ba4c56fd4c7d238c2ee71f5132fea0f69d3d25`; cache key `5ac696a9075b5dfc`.
- `a0_step0`: local checkpoint `models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed42/checkpoint-step-000000` (run `bge_base_ft_v3_a1-seed42`, step 0 of 55, files digest `38633262d28ded75`), fine-tuned from `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` on 678 synthetic pairs of `train_candidates_v3`; selection: a candidate of the fixed validation rule (the base model, evaluated at step 0); which checkpoint the rule selects is recorded in ../selection.json when the run ends; query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `622f038aa3e3358f81d8466c911d02e452746fbc66a3cfff37747dda62f7800a`; cache key `f9eca0979562bb6a`; evaluation isolation: 667 of 667 judged passages on the training blocklist; 0 of 1575 passages of the 678 training candidates (train_candidates_v3) judged; highest training-question similarity to an evaluation question 0.346 (threshold 0.8); selecting validation set validation_pairs_v4: 0 of 170 passages of its 93 questions judged, highest question similarity 0.31.
- `a1_seed42`: local checkpoint `models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed42/checkpoint-step-000055` (run `bge_base_ft_v3_a1-seed42`, step 55 of 55, files digest `24c2aaae977ce319`), fine-tuned from `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` on 678 synthetic pairs of `train_candidates_v3`; selection: a candidate of the fixed validation rule (evaluated at step 55); which checkpoint the rule selects is recorded in ../selection.json when the run ends; final model: selected by its run's validation rule at step 55, seed 42 of [42, 1337, 2026], whole-corpus validation MRR@10 0.6690 on `validation_pairs_v4` (record SHA-256 `19cca1cbaebe15ed`); query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `1f3ee0b701bbec89fc1b0ad96bb6a916e03d34ce0a9b14e3df444d5dbe3fd3ff`; cache key `7d6f16da40c01401`; evaluation isolation: 667 of 667 judged passages on the training blocklist; 0 of 1575 passages of the 678 training candidates (train_candidates_v3) judged; highest training-question similarity to an evaluation question 0.346 (threshold 0.8); selecting validation set validation_pairs_v4: 0 of 170 passages of its 93 questions judged, highest question similarity 0.31.
- `a1_seed1337`: local checkpoint `models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed1337/checkpoint-step-000040` (run `bge_base_ft_v3_a1-seed1337`, step 40 of 55, files digest `8237ec0b82a12830`), fine-tuned from `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` on 678 synthetic pairs of `train_candidates_v3`; selection: a candidate of the fixed validation rule (evaluated at step 40); which checkpoint the rule selects is recorded in ../selection.json when the run ends; final model: selected by its run's validation rule at step 40, seed 1337 of [42, 1337, 2026], whole-corpus validation MRR@10 0.6673 on `validation_pairs_v4` (record SHA-256 `165064f5ecbb97df`); query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `791ba55c9576bfa16bb968498be50afff268135ecaaced9539c90c5e9453832a`; cache key `c631ef862652e148`; evaluation isolation: 667 of 667 judged passages on the training blocklist; 0 of 1575 passages of the 678 training candidates (train_candidates_v3) judged; highest training-question similarity to an evaluation question 0.346 (threshold 0.8); selecting validation set validation_pairs_v4: 0 of 170 passages of its 93 questions judged, highest question similarity 0.31.
- `a1_seed2026`: local checkpoint `models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed2026/checkpoint-step-000040` (run `bge_base_ft_v3_a1-seed2026`, step 40 of 55, files digest `cba9110b7f744b0c`), fine-tuned from `BAAI/bge-base-en-v1.5` at `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` on 678 synthetic pairs of `train_candidates_v3`; selection: a candidate of the fixed validation rule (evaluated at step 40); which checkpoint the rule selects is recorded in ../selection.json when the run ends; final model: selected by its run's validation rule at step 40, seed 2026 of [42, 1337, 2026], whole-corpus validation MRR@10 0.6580 on `validation_pairs_v4` (record SHA-256 `2afbdae2f797fcff`); query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `6d4886a67458404964f2a03e5d50120288ffffa4d29fb7fd37293149d1b6ca05`; cache key `966cb3d2df33a04c`; evaluation isolation: 667 of 667 judged passages on the training blocklist; 0 of 1575 passages of the 678 training candidates (train_candidates_v3) judged; highest training-question similarity to an evaluation question 0.346 (threshold 0.8); selecting validation set validation_pairs_v4: 0 of 170 passages of its 93 questions judged, highest question similarity 0.31.

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

## a0_step0, global

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

## a0_step0, company

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

## a1_seed42, global

Top 10 over 10 questions: 31 judged relevant, 51 confirmed negatives, 18 unjudged (16 from the company asked about, 2 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.325 | 0.500 | 0.667 | 1.000 | 0 (0 / 0) | 8 | 2 | 4 (2) | — |
| q_0002 | 3 | 0.405 | 0.500 | 1.000 | 0.800 | 2 (0 / 2) | 6 | 2 | 1 (0) | — |
| q_0003 | 5 | 0.540 | 0.333 | 1.000 | 1.000 | 0 (0 / 0) | 6 | 3 | 1 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.625 | 0.300 | 7 (7 / 0) | 2 | 1 | 7 (3) | low_judged_at_10 |
| q_0005 | 16 | 0.197 | 1.000 | 0.938 | 0.800 | 2 (2 / 0) | 5 | 1 | 13 (1) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.333 | 0.800 | 2 (2 / 0) | 8 | 12 | 6 (4) | — |
| q_0007 | 23 | 0.804 | 1.000 | 0.913 | 0.900 | 1 (1 / 0) | 1 | 1 | 15 (2) | — |
| q_0008 | 12 | 0.490 | 1.000 | 0.667 | 0.900 | 1 (1 / 0) | 3 | 1 | 6 (4) | — |
| q_0009 | 16 | 0.303 | 0.250 | 0.750 | 0.900 | 1 (1 / 0) | 4 | 4 | 11 (4) | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.800 | 0.800 | 2 (2 / 0) | 8 | 22 | 5 (1) | — |

Questions with insufficient relevance coverage: q_0004.

## a1_seed42, company

Top 10 over 10 questions: 32 judged relevant, 49 confirmed negatives, 19 unjudged (19 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.325 | 0.500 | 0.833 | 1.000 | 0 (0 / 0) | 8 | 2 | 4 (1) | — |
| q_0002 | 3 | 0.409 | 0.500 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 2 | 1 (0) | — |
| q_0003 | 5 | 0.540 | 0.333 | 1.000 | 1.000 | 0 (0 / 0) | 6 | 3 | 1 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.625 | 0.300 | 7 (7 / 0) | 2 | 1 | 7 (3) | low_judged_at_10 |
| q_0005 | 16 | 0.197 | 1.000 | 0.938 | 0.800 | 2 (2 / 0) | 5 | 1 | 13 (1) | — |
| q_0006 | 6 | 0.076 | 0.333 | 0.667 | 0.500 | 5 (5 / 0) | 4 | 3 | 5 (2) | low_judged_at_10 |
| q_0007 | 23 | 0.804 | 1.000 | 0.913 | 0.900 | 1 (1 / 0) | 1 | 1 | 15 (2) | — |
| q_0008 | 12 | 0.490 | 1.000 | 0.667 | 0.900 | 1 (1 / 0) | 3 | 1 | 6 (4) | — |
| q_0009 | 16 | 0.303 | 0.250 | 0.750 | 0.900 | 1 (1 / 0) | 4 | 4 | 11 (4) | — |
| q_0010 | 5 | 0.000 | 0.000 | 1.000 | 0.800 | 2 (2 / 0) | 8 | 21 | 5 (0) | — |

Questions with insufficient relevance coverage: q_0004, q_0006.

## a1_seed1337, global

Top 10 over 10 questions: 31 judged relevant, 54 confirmed negatives, 15 unjudged (11 from the company asked about, 4 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.248 | 0.500 | 0.667 | 1.000 | 0 (0 / 0) | 8 | 2 | 4 (2) | — |
| q_0002 | 3 | 0.413 | 0.500 | 0.667 | 0.700 | 3 (0 / 3) | 5 | 2 | 1 (1) | — |
| q_0003 | 5 | 0.536 | 0.333 | 1.000 | 1.000 | 0 (0 / 0) | 6 | 3 | 1 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.625 | 0.400 | 6 (6 / 0) | 3 | 1 | 7 (3) | low_judged_at_10 |
| q_0005 | 16 | 0.197 | 1.000 | 0.938 | 0.800 | 2 (2 / 0) | 5 | 1 | 13 (1) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.333 | 0.900 | 1 (1 / 0) | 9 | 12 | 6 (4) | — |
| q_0007 | 23 | 0.815 | 1.000 | 0.913 | 0.900 | 1 (1 / 0) | 1 | 1 | 15 (2) | — |
| q_0008 | 12 | 0.466 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 4 | 1 | 6 (4) | — |
| q_0009 | 16 | 0.430 | 0.500 | 0.688 | 0.900 | 1 (0 / 1) | 4 | 2 | 11 (5) | — |
| q_0010 | 5 | 0.000 | 0.000 | 1.000 | 0.900 | 1 (1 / 0) | 9 | 25 | 5 (0) | — |

Questions with insufficient relevance coverage: q_0004.

## a1_seed1337, company

Top 10 over 10 questions: 32 judged relevant, 51 confirmed negatives, 17 unjudged (17 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.248 | 0.500 | 0.833 | 1.000 | 0 (0 / 0) | 8 | 2 | 4 (1) | — |
| q_0002 | 3 | 0.417 | 0.500 | 1.000 | 0.900 | 1 (1 / 0) | 7 | 2 | 1 (0) | — |
| q_0003 | 5 | 0.536 | 0.333 | 1.000 | 1.000 | 0 (0 / 0) | 6 | 3 | 1 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.625 | 0.400 | 6 (6 / 0) | 3 | 1 | 7 (3) | low_judged_at_10 |
| q_0005 | 16 | 0.197 | 1.000 | 0.938 | 0.800 | 2 (2 / 0) | 5 | 1 | 13 (1) | — |
| q_0006 | 6 | 0.076 | 0.333 | 0.667 | 0.500 | 5 (5 / 0) | 4 | 3 | 5 (2) | low_judged_at_10 |
| q_0007 | 23 | 0.815 | 1.000 | 0.913 | 0.900 | 1 (1 / 0) | 1 | 1 | 15 (2) | — |
| q_0008 | 12 | 0.466 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 4 | 1 | 6 (4) | — |
| q_0009 | 16 | 0.431 | 0.500 | 0.688 | 0.900 | 1 (1 / 0) | 4 | 2 | 11 (5) | — |
| q_0010 | 5 | 0.000 | 0.000 | 1.000 | 0.900 | 1 (1 / 0) | 9 | 25 | 5 (0) | — |

Questions with insufficient relevance coverage: q_0004, q_0006.

## a1_seed2026, global

Top 10 over 10 questions: 27 judged relevant, 53 confirmed negatives, 20 unjudged (16 from the company asked about, 4 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.315 | 1.000 | 0.667 | 1.000 | 0 (0 / 0) | 8 | 1 | 4 (2) | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 0.600 | 4 (1 / 3) | 5 | 1 | 2 (0) | low_judged_at_10 |
| q_0003 | 5 | 0.434 | 0.333 | 1.000 | 0.900 | 1 (1 / 0) | 6 | 3 | 2 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.625 | 0.500 | 5 (5 / 0) | 4 | 1 | 7 (3) | low_judged_at_10 |
| q_0005 | 16 | 0.162 | 1.000 | 0.875 | 0.800 | 2 (2 / 0) | 6 | 1 | 14 (2) | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.500 | 0.900 | 1 (1 / 0) | 9 | 12 | 6 (3) | — |
| q_0007 | 23 | 0.817 | 1.000 | 0.913 | 1.000 | 0 (0 / 0) | 1 | 1 | 14 (2) | — |
| q_0008 | 12 | 0.349 | 1.000 | 0.667 | 0.800 | 2 (2 / 0) | 3 | 1 | 7 (4) | — |
| q_0009 | 16 | 0.295 | 0.333 | 0.812 | 0.700 | 3 (2 / 1) | 3 | 3 | 12 (3) | — |
| q_0010 | 5 | 0.000 | 0.000 | 1.000 | 0.800 | 2 (2 / 0) | 8 | 24 | 5 (0) | — |

Questions with insufficient relevance coverage: q_0002, q_0004.

## a1_seed2026, company

Top 10 over 10 questions: 29 judged relevant, 51 confirmed negatives, 20 unjudged (20 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Relevant outside top 10 (not returned) | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.315 | 1.000 | 0.833 | 1.000 | 0 (0 / 0) | 8 | 1 | 4 (1) | — |
| q_0002 | 3 | 0.615 | 1.000 | 1.000 | 0.900 | 1 (1 / 0) | 7 | 1 | 1 (0) | — |
| q_0003 | 5 | 0.434 | 0.333 | 1.000 | 0.900 | 1 (1 / 0) | 6 | 3 | 2 (0) | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.625 | 0.500 | 5 (5 / 0) | 4 | 1 | 7 (3) | low_judged_at_10 |
| q_0005 | 16 | 0.162 | 1.000 | 0.875 | 0.800 | 2 (2 / 0) | 6 | 1 | 14 (2) | — |
| q_0006 | 6 | 0.096 | 0.500 | 0.667 | 0.500 | 5 (5 / 0) | 4 | 2 | 5 (2) | low_judged_at_10 |
| q_0007 | 23 | 0.817 | 1.000 | 0.913 | 1.000 | 0 (0 / 0) | 1 | 1 | 14 (2) | — |
| q_0008 | 12 | 0.349 | 1.000 | 0.667 | 0.800 | 2 (2 / 0) | 3 | 1 | 7 (4) | — |
| q_0009 | 16 | 0.308 | 0.333 | 0.812 | 0.800 | 2 (2 / 0) | 4 | 3 | 12 (3) | — |
| q_0010 | 5 | 0.000 | 0.000 | 1.000 | 0.800 | 2 (2 / 0) | 8 | 23 | 5 (0) | — |

Questions with insufficient relevance coverage: q_0004, q_0006.

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
