# dev_pilot_v3: bm25, bge_zero_shot, e5_zero_shot

> **PROVISIONAL DEVELOPMENT METRICS. Labels are AI-generated development labels (see the label provenance), checked only by an AI-assisted audit; these numbers are not benchmark results.**

- Corpus: `corpus_v1@1.2`, 10,137 passages, SHA-256 `ff9dadf37bb9820e2c87df70a411fd189bb02776609e5ec06173dd03f2344004`
- Labels: `dev_pilot_v3`: 696 judged pairs, by source {'ai_proposal': 692, 'human_judgment': 4}, open flags {'P1': 11, 'P2': 2, 'P3': 1, 'P5': 2, 'low_confidence': 8}, audit ai_assisted
- Settings: top_k 100, judged@10, coverage thresholds {'min_positives': 1, 'min_judged_at_10': 0.7}, retrieval modes global, company

Rankings: No retriever was run for this set: it scores the stored rankings in results/dev/pilot_v2 (each run file's SHA-256 checked against its record). Nothing was re-encoded or re-ranked; only the labels differ.

Unjudged passages: Qrels hold judged passages only (grades 0, 1, 2). An unjudged passage is not a confirmed negative: the standard metrics count it as non-relevant (an assumption), so each query reports how much of its top 10 is judged, how much is a confirmed negative, and how much is unjudged, split into passages from the companies the question asks about (in scope) and from other companies (out of scope).

Candidate pools: Only passages in a question's candidate pool were judged. The original pools were built mainly with BM25, plus seeded random and per-filing samples and a few passages found by hand. This set's pools add every passage the stored runs company/bge_zero_shot, company/bm25, company/e5_zero_shot, global/bge_zero_shot, global/bm25, global/e5_zero_shot ranked in their top 10, labelled by AI proposals, so the top 10 of those runs is judged. Below rank 10, passages BM25 did not pool are still unjudged and count as non-relevant, so Recall@100 still favours BM25-like rankings.
In this set, 523 of 696 (75.1%) judged passages and 80 of 100 (80.0%) judged-relevant passages were proposed by BM25 (pooling or the BM25 runs). Judged passages by source: {'bge_zero_shot': 44, 'bge_zero_shot+bm25': 31, 'bge_zero_shot+bm25+e5_zero_shot': 24, 'bge_zero_shot+e5_zero_shot': 11, 'bge_zero_shot+e5_zero_shot+manual': 1, 'bm25': 442, 'bm25+e5_zero_shot': 25, 'bm25+stratified': 1, 'e5_zero_shot': 50, 'e5_zero_shot+manual': 1, 'e5_zero_shot+random': 1, 'manual': 3, 'random': 49, 'stratified': 13}; relevant passages by source: {'bge_zero_shot': 6, 'bge_zero_shot+bm25': 4, 'bge_zero_shot+bm25+e5_zero_shot': 12, 'bge_zero_shot+e5_zero_shot': 2, 'bge_zero_shot+e5_zero_shot+manual': 1, 'bm25': 57, 'bm25+e5_zero_shot': 7, 'e5_zero_shot': 7, 'e5_zero_shot+manual': 1, 'manual': 3}.

Retrieval modes:

- `global`: Each question ranks every passage in the corpus; no company restriction.
- `company`: Each question ranks only the passages whose ticker is one of the question's tickers (query metadata 'tickers'). Other passages are removed before ranking, never filtered out of a global top-k, and every retriever gets the same restriction.

> **Do not conclude that one retriever is better from these numbers. Labels are AI-generated development labels on 10 questions, the judged pools come mostly from BM25, and unjudged passages count as non-relevant. Read every score with its judged@10 and unjudged counts; a comparison needs pools that include the dense retrievers' results and human-judged labels.**

## Comparison

| Retriever | Mode | nDCG@10 | MRR@10 | Recall@100 | judged@10 | nDCG@10 judged-only | MRR@10 judged-only | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | Insufficient coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | global | 0.170 | 0.323 | 0.578 | 1.000 | 0.170 | 0.323 | 0 (0 / 0) | 82 | — |
| bm25 | company | 0.194 | 0.423 | 0.674 | 1.000 | 0.194 | 0.423 | 0 (0 / 0) | 79 | — |
| bge_zero_shot | global | 0.286 | 0.550 | 0.627 | 1.000 | 0.286 | 0.550 | 0 (0 / 0) | 77 | — |
| bge_zero_shot | company | 0.297 | 0.575 | 0.643 | 1.000 | 0.297 | 0.575 | 0 (0 / 0) | 75 | — |
| e5_zero_shot | global | 0.361 | 0.600 | 0.666 | 1.000 | 0.361 | 0.600 | 0 (0 / 0) | 71 | — |
| e5_zero_shot | company | 0.376 | 0.625 | 0.673 | 1.000 | 0.376 | 0.625 | 0 (0 / 0) | 70 | — |

Unjudged and confirmed-negative counts are summed over the top 10 of every question.

## Change from dev_pilot_v2 (labels pilot_v2 → pilot_v3)

Each run below is the same ranking scored twice: with the earlier labels (as recorded by dev_pilot_v2) and with this set's. Only the labels differ.

| Retriever | Mode | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 | Relevant top 10 | Judged positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | global | 0.181 → 0.170 | 0.323 → 0.323 | 0.606 → 0.578 | 0.880 → 1.000 | 12 → 0 | 18 → 18 | 85 → 100 |
| bm25 | company | 0.206 → 0.194 | 0.423 → 0.423 | 0.665 → 0.674 | 1.000 → 1.000 | 0 → 0 | 21 → 21 | 85 → 100 |
| bge_zero_shot | global | 0.247 → 0.286 | 0.450 → 0.550 | 0.572 → 0.627 | 0.490 → 1.000 | 51 → 0 | 17 → 23 | 85 → 100 |
| bge_zero_shot | company | 0.247 → 0.297 | 0.450 → 0.575 | 0.592 → 0.643 | 0.520 → 1.000 | 48 → 0 | 17 → 25 | 85 → 100 |
| e5_zero_shot | global | 0.314 → 0.361 | 0.550 → 0.600 | 0.628 → 0.666 | 0.430 → 1.000 | 57 → 0 | 20 → 29 | 85 → 100 |
| e5_zero_shot | company | 0.330 → 0.376 | 0.575 → 0.625 | 0.637 → 0.673 | 0.480 → 1.000 | 52 → 0 | 21 → 30 | 85 → 100 |

### nDCG@10 per question, earlier → this set

| Question | bm25 global | bm25 company | bge_zero_shot global | bge_zero_shot company | e5_zero_shot global | e5_zero_shot company |
| --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 0.000 → 0.000 | 0.000 → 0.000 | 0.208 → 0.197 | 0.208 → 0.241 | 0.208 → 0.197 | 0.367 → 0.347 |
| q_0002 | 0.000 → 0.000 | 0.000 → 0.000 | 0.556 → 0.556 | 0.556 → 0.556 | 0.556 → 0.556 | 0.556 → 0.556 |
| q_0003 | 0.000 → 0.000 | 0.000 → 0.000 | 0.478 → 0.441 | 0.478 → 0.441 | 0.736 → 0.737 | 0.736 → 0.737 |
| q_0004 | 0.199 → 0.199 | 0.199 → 0.199 | 0.139 → 0.139 | 0.139 → 0.139 | 0.046 → 0.046 | 0.046 → 0.046 |
| q_0005 | 0.114 → 0.114 | 0.114 → 0.114 | 0.049 → 0.087 | 0.049 → 0.087 | 0.317 → 0.425 | 0.317 → 0.425 |
| q_0006 | 0.000 → 0.000 | 0.223 → 0.211 | 0.000 → 0.000 | 0.000 → 0.066 | 0.000 → 0.000 | 0.000 → 0.000 |
| q_0007 | 0.672 → 0.672 | 0.700 → 0.700 | 0.809 → 0.809 | 0.809 → 0.809 | 0.466 → 0.466 | 0.466 → 0.466 |
| q_0008 | 0.371 → 0.268 | 0.371 → 0.268 | 0.000 → 0.299 | 0.000 → 0.299 | 0.254 → 0.374 | 0.254 → 0.374 |
| q_0009 | 0.313 → 0.313 | 0.313 → 0.313 | 0.235 → 0.337 | 0.235 → 0.337 | 0.434 → 0.681 | 0.434 → 0.681 |
| q_0010 | 0.139 → 0.139 | 0.139 → 0.139 | 0.000 → 0.000 | 0.000 → 0.000 | 0.125 → 0.125 | 0.125 → 0.125 |

### judged@10 per question

| Question | bm25 global | bm25 company | bge_zero_shot global | bge_zero_shot company | e5_zero_shot global | e5_zero_shot company |
| --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0002 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0003 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0004 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0005 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0006 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0007 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0008 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0009 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0010 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Retrievers

- `bm25`: BM25: tokenizer `word`, backend `python`, k1 1.5, b 0.75.
- `bge_zero_shot`: `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `622f038aa3e3358f81d8466c911d02e452746fbc66a3cfff37747dda62f7800a`; cache key `f04c62fe0721bcbb`.
- `e5_zero_shot`: `intfloat/e5-base-v2` at revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`; query prefix `'query: '`, passage prefix `'passage: '`; 768 dimensions, mean pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9208 of 10137 (90.8%) passages truncated, 27.7% of passage tokens beyond the limit; embeddings SHA-256 `73cf951ea6c09a038e8d4062a8ba4c56fd4c7d238c2ee71f5132fea0f69d3d25`; cache key `5ac696a9075b5dfc`.

## bm25, global

Top 10 over 10 questions: 18 judged relevant, 82 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.000 | 0.000 | 0.000 | 1.000 | 0 (0 / 0) | 10 | — | — |
| q_0002 | 3 | 0.000 | 0.000 | 0.667 | 1.000 | 0 (0 / 0) | 10 | 11 | — |
| q_0003 | 5 | 0.000 | 0.000 | 0.600 | 1.000 | 0 (0 / 0) | 10 | 11 | — |
| q_0004 | 8 | 0.199 | 0.250 | 0.750 | 1.000 | 0 (0 / 0) | 6 | 4 | — |
| q_0005 | 16 | 0.114 | 1.000 | 0.438 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.333 | 1.000 | 0 (0 / 0) | 10 | 21 | — |
| q_0007 | 23 | 0.672 | 1.000 | 0.913 | 1.000 | 0 (0 / 0) | 4 | 1 | — |
| q_0008 | 12 | 0.268 | 0.333 | 0.833 | 1.000 | 0 (0 / 0) | 7 | 3 | — |
| q_0009 | 16 | 0.313 | 0.500 | 0.250 | 1.000 | 0 (0 / 0) | 7 | 2 | — |
| q_0010 | 5 | 0.139 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 7 | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bm25, company

Top 10 over 10 questions: 21 judged relevant, 79 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 64 | — |
| q_0002 | 3 | 0.000 | 0.000 | 1.000 | 1.000 | 0 (0 / 0) | 10 | 11 | — |
| q_0003 | 5 | 0.000 | 0.000 | 0.600 | 1.000 | 0 (0 / 0) | 10 | 11 | — |
| q_0004 | 8 | 0.199 | 0.250 | 0.875 | 1.000 | 0 (0 / 0) | 6 | 4 | — |
| q_0005 | 16 | 0.114 | 1.000 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0006 | 6 | 0.211 | 1.000 | 0.500 | 1.000 | 0 (0 / 0) | 8 | 1 | — |
| q_0007 | 23 | 0.700 | 1.000 | 0.957 | 1.000 | 0 (0 / 0) | 3 | 1 | — |
| q_0008 | 12 | 0.268 | 0.333 | 0.833 | 1.000 | 0 (0 / 0) | 7 | 3 | — |
| q_0009 | 16 | 0.313 | 0.500 | 0.312 | 1.000 | 0 (0 / 0) | 7 | 2 | — |
| q_0010 | 5 | 0.139 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 7 | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_zero_shot, global

Top 10 over 10 questions: 23 judged relevant, 77 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.197 | 0.250 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 4 | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0003 | 5 | 0.441 | 0.500 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 2 | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.375 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0005 | 16 | 0.087 | 0.250 | 0.812 | 1.000 | 0 (0 / 0) | 8 | 4 | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 18 | — |
| q_0007 | 23 | 0.809 | 1.000 | 0.826 | 1.000 | 0 (0 / 0) | 2 | 1 | — |
| q_0008 | 12 | 0.299 | 1.000 | 0.750 | 1.000 | 0 (0 / 0) | 7 | 1 | — |
| q_0009 | 16 | 0.337 | 0.500 | 0.438 | 1.000 | 0 (0 / 0) | 5 | 2 | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.400 | 1.000 | 0 (0 / 0) | 10 | 49 | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_zero_shot, company

Top 10 over 10 questions: 25 judged relevant, 75 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.241 | 0.250 | 0.667 | 1.000 | 0 (0 / 0) | 8 | 4 | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0003 | 5 | 0.441 | 0.500 | 1.000 | 1.000 | 0 (0 / 0) | 8 | 2 | — |
| q_0004 | 8 | 0.139 | 1.000 | 0.375 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0005 | 16 | 0.087 | 0.250 | 0.812 | 1.000 | 0 (0 / 0) | 8 | 4 | — |
| q_0006 | 6 | 0.066 | 0.250 | 0.167 | 1.000 | 0 (0 / 0) | 9 | 4 | — |
| q_0007 | 23 | 0.809 | 1.000 | 0.826 | 1.000 | 0 (0 / 0) | 2 | 1 | — |
| q_0008 | 12 | 0.299 | 1.000 | 0.750 | 1.000 | 0 (0 / 0) | 7 | 1 | — |
| q_0009 | 16 | 0.337 | 0.500 | 0.438 | 1.000 | 0 (0 / 0) | 5 | 2 | — |
| q_0010 | 5 | 0.000 | 0.000 | 0.400 | 1.000 | 0 (0 / 0) | 10 | 49 | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## e5_zero_shot, global

Top 10 over 10 questions: 29 judged relevant, 71 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.197 | 0.250 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 4 | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0003 | 5 | 0.737 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 7 | 1 | — |
| q_0004 | 8 | 0.046 | 0.143 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 7 | — |
| q_0005 | 16 | 0.425 | 1.000 | 0.812 | 1.000 | 0 (0 / 0) | 5 | 1 | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 30 | — |
| q_0007 | 23 | 0.466 | 0.500 | 0.826 | 1.000 | 0 (0 / 0) | 4 | 2 | — |
| q_0008 | 12 | 0.374 | 1.000 | 0.833 | 1.000 | 0 (0 / 0) | 5 | 1 | — |
| q_0009 | 16 | 0.681 | 1.000 | 0.625 | 1.000 | 0 (0 / 0) | 4 | 1 | — |
| q_0010 | 5 | 0.125 | 0.111 | 0.400 | 1.000 | 0 (0 / 0) | 9 | 9 | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## e5_zero_shot, company

Top 10 over 10 questions: 30 judged relevant, 70 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 6 | 0.347 | 0.500 | 0.500 | 1.000 | 0 (0 / 0) | 8 | 2 | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0003 | 5 | 0.737 | 1.000 | 1.000 | 1.000 | 0 (0 / 0) | 7 | 1 | — |
| q_0004 | 8 | 0.046 | 0.143 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 7 | — |
| q_0005 | 16 | 0.425 | 1.000 | 0.812 | 1.000 | 0 (0 / 0) | 5 | 1 | — |
| q_0006 | 6 | 0.000 | 0.000 | 0.167 | 1.000 | 0 (0 / 0) | 10 | 12 | — |
| q_0007 | 23 | 0.466 | 0.500 | 0.826 | 1.000 | 0 (0 / 0) | 4 | 2 | — |
| q_0008 | 12 | 0.374 | 1.000 | 0.833 | 1.000 | 0 (0 / 0) | 5 | 1 | — |
| q_0009 | 16 | 0.681 | 1.000 | 0.688 | 1.000 | 0 (0 / 0) | 4 | 1 | — |
| q_0010 | 5 | 0.125 | 0.111 | 0.400 | 1.000 | 0 (0 / 0) | 9 | 9 | — |

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
