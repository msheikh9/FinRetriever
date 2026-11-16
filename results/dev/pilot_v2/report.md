# dev_pilot_v2: bm25, bge_zero_shot, e5_zero_shot

> **PROVISIONAL DEVELOPMENT METRICS. Labels are AI-generated development labels (see the label provenance), checked only by an AI-assisted audit; these numbers are not benchmark results.**

- Corpus: `corpus_v1@1.2`, 10,137 passages, SHA-256 `ff9dadf37bb9820e2c87df70a411fd189bb02776609e5ec06173dd03f2344004`
- Labels: `dev_pilot_v2`: 579 judged pairs, by source {'ai_proposal': 575, 'human_judgment': 4}, open flags {'P1': 9, 'P2': 2, 'P5': 1}, audit ai_assisted
- Settings: top_k 100, judged@10, coverage thresholds {'min_positives': 1, 'min_judged_at_10': 0.7}, retrieval modes global, company

Unjudged passages: Qrels hold judged passages only (grades 0, 1, 2). An unjudged passage is not a confirmed negative: the standard metrics count it as non-relevant (an assumption), so each query reports how much of its top 10 is judged, how much is a confirmed negative, and how much is unjudged, split into passages from the companies the question asks about (in scope) and from other companies (out of scope).

Candidate pools: Only passages in a question's candidate pool were judged. The pools were built mainly with BM25, plus seeded random and per-filing samples and a few passages found by hand; no dense retriever proposed candidates. A retriever that ranks passages BM25 did not surface gets more unjudged passages, which count as non-relevant, so the metrics favour BM25-like rankings.
In this set, 511 of 579 (88.3%) judged passages and 80 of 85 (94.1%) judged-relevant passages were proposed by BM25. Judged passages by source: {'bm25': 510, 'bm25+stratified': 1, 'manual': 5, 'random': 50, 'stratified': 13}; relevant passages by source: {'bm25': 80, 'manual': 5}.

Retrieval modes:

- `global`: Each question ranks every passage in the corpus; no company restriction.
- `company`: Each question ranks only the passages whose ticker is one of the question's tickers (query metadata 'tickers'). Other passages are removed before ranking, never filtered out of a global top-k, and every retriever gets the same restriction.

> **Do not conclude that one retriever is better from these numbers. Labels are AI-generated development labels on 10 questions, the judged pools come mostly from BM25, and unjudged passages count as non-relevant. Read every score with its judged@10 and unjudged counts; a comparison needs pools that include the dense retrievers' results and human-judged labels.**

## Comparison

| Retriever | Mode | nDCG@10 | MRR@10 | Recall@100 | judged@10 | nDCG@10 judged-only | MRR@10 judged-only | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | Insufficient coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | global | 0.181 | 0.323 | 0.606 | 0.880 | 0.206 | 0.423 | 12 (0 / 12) | 70 | q_0006 |
| bm25 | company | 0.206 | 0.423 | 0.665 | 1.000 | 0.206 | 0.423 | 0 (0 / 0) | 79 | — |
| bge_zero_shot | global | 0.247 | 0.450 | 0.572 | 0.490 | 0.338 | 0.548 | 51 (40 / 11) | 32 | q_0001, q_0003, q_0004, q_0005, q_0006, q_0008, q_0009, q_0010 |
| bge_zero_shot | company | 0.247 | 0.450 | 0.592 | 0.520 | 0.338 | 0.548 | 48 (48 / 0) | 35 | q_0001, q_0003, q_0004, q_0005, q_0006, q_0008, q_0009, q_0010 |
| e5_zero_shot | global | 0.314 | 0.550 | 0.628 | 0.430 | 0.451 | 0.675 | 57 (44 / 13) | 23 | q_0001, q_0003, q_0004, q_0005, q_0006, q_0008, q_0009, q_0010 |
| e5_zero_shot | company | 0.330 | 0.575 | 0.637 | 0.480 | 0.451 | 0.675 | 52 (52 / 0) | 27 | q_0003, q_0004, q_0005, q_0006, q_0008, q_0009, q_0010 |

Unjudged and confirmed-negative counts are summed over the top 10 of every question.

### judged@10 per question

| Question | bm25 global | bm25 company | bge_zero_shot global | bge_zero_shot company | e5_zero_shot global | e5_zero_shot company |
| --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 0.900 | 1.000 | 0.400 | 0.400 | 0.300 | 0.800 |
| q_0002 | 1.000 | 1.000 | 0.900 | 0.900 | 0.900 | 0.900 |
| q_0003 | 1.000 | 1.000 | 0.600 | 0.600 | 0.500 | 0.500 |
| q_0004 | 1.000 | 1.000 | 0.600 | 0.600 | 0.300 | 0.300 |
| q_0005 | 1.000 | 1.000 | 0.400 | 0.400 | 0.300 | 0.300 |
| q_0006 | 0.000 | 1.000 | 0.000 | 0.300 | 0.000 | 0.000 |
| q_0007 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| q_0008 | 1.000 | 1.000 | 0.100 | 0.100 | 0.200 | 0.200 |
| q_0009 | 1.000 | 1.000 | 0.500 | 0.500 | 0.400 | 0.400 |
| q_0010 | 1.000 | 1.000 | 0.400 | 0.400 | 0.400 | 0.400 |

## Retrievers

- `bm25`: BM25: tokenizer `word`, backend `python`, k1 1.5, b 0.75.
- `bge_zero_shot`: `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; query prefix `'Represent this sentence for searching relevant passages: '`, passage prefix none; 768 dimensions, cls pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9204 of 10137 (90.8%) passages truncated, 27.6% of passage tokens beyond the limit; embeddings SHA-256 `622f038aa3e3358f81d8466c911d02e452746fbc66a3cfff37747dda62f7800a`; cache key `f04c62fe0721bcbb`.
- `e5_zero_shot`: `intfloat/e5-base-v2` at revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`; query prefix `'query: '`, passage prefix `'passage: '`; 768 dimensions, mean pooling, L2-normalised, cosine (dot product); max_seq_length 512, batch size 32, device `cpu`; 9208 of 10137 (90.8%) passages truncated, 27.7% of passage tokens beyond the limit; embeddings SHA-256 `73cf951ea6c09a038e8d4062a8ba4c56fd4c7d238c2ee71f5132fea0f69d3d25`; cache key `5ac696a9075b5dfc`.

## bm25, global

Top 10 over 10 questions: 18 judged relevant, 70 confirmed negatives, 12 unjudged (0 from the company asked about, 12 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 5 | 0.000 | 0.000 | 0.000 | 0.900 | 1 (0 / 1) | 9 | — | — |
| q_0002 | 3 | 0.000 | 0.000 | 0.667 | 1.000 | 0 (0 / 0) | 10 | 11 | — |
| q_0003 | 4 | 0.000 | 0.000 | 0.750 | 1.000 | 0 (0 / 0) | 10 | 11 | — |
| q_0004 | 8 | 0.199 | 0.250 | 0.750 | 1.000 | 0 (0 / 0) | 6 | 4 | — |
| q_0005 | 14 | 0.114 | 1.000 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0006 | 5 | 0.000 | 0.000 | 0.400 | 0.000 | 10 (0 / 10) | 0 | 21 | low_judged_at_10 |
| q_0007 | 23 | 0.672 | 1.000 | 0.913 | 0.900 | 1 (0 / 1) | 3 | 1 | — |
| q_0008 | 7 | 0.371 | 0.333 | 0.714 | 1.000 | 0 (0 / 0) | 7 | 3 | — |
| q_0009 | 11 | 0.313 | 0.500 | 0.364 | 1.000 | 0 (0 / 0) | 7 | 2 | — |
| q_0010 | 5 | 0.139 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 7 | — |

Questions with insufficient relevance coverage: q_0006.

## bm25, company

Top 10 over 10 questions: 21 judged relevant, 79 confirmed negatives, 0 unjudged (0 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 5 | 0.000 | 0.000 | 0.000 | 1.000 | 0 (0 / 0) | 10 | — | — |
| q_0002 | 3 | 0.000 | 0.000 | 1.000 | 1.000 | 0 (0 / 0) | 10 | 11 | — |
| q_0003 | 4 | 0.000 | 0.000 | 0.750 | 1.000 | 0 (0 / 0) | 10 | 11 | — |
| q_0004 | 8 | 0.199 | 0.250 | 0.875 | 1.000 | 0 (0 / 0) | 6 | 4 | — |
| q_0005 | 14 | 0.114 | 1.000 | 0.500 | 1.000 | 0 (0 / 0) | 9 | 1 | — |
| q_0006 | 5 | 0.223 | 1.000 | 0.400 | 1.000 | 0 (0 / 0) | 8 | 1 | — |
| q_0007 | 23 | 0.700 | 1.000 | 0.957 | 1.000 | 0 (0 / 0) | 3 | 1 | — |
| q_0008 | 7 | 0.371 | 0.333 | 0.714 | 1.000 | 0 (0 / 0) | 7 | 3 | — |
| q_0009 | 11 | 0.313 | 0.500 | 0.455 | 1.000 | 0 (0 / 0) | 7 | 2 | — |
| q_0010 | 5 | 0.139 | 0.143 | 1.000 | 1.000 | 0 (0 / 0) | 9 | 7 | — |

Questions with insufficient relevance coverage: none under the configured thresholds.

## bge_zero_shot, global

Top 10 over 10 questions: 17 judged relevant, 32 confirmed negatives, 51 unjudged (40 from the company asked about, 11 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 5 | 0.208 | 0.250 | 0.400 | 0.400 | 6 (5 / 1) | 3 | 4 | low_judged_at_10 |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 0.900 | 1 (1 / 0) | 8 | 1 | — |
| q_0003 | 4 | 0.478 | 0.500 | 1.000 | 0.600 | 4 (4 / 0) | 4 | 2 | low_judged_at_10 |
| q_0004 | 8 | 0.139 | 1.000 | 0.375 | 0.600 | 4 (4 / 0) | 5 | 1 | low_judged_at_10 |
| q_0005 | 14 | 0.049 | 0.250 | 0.786 | 0.400 | 6 (6 / 0) | 3 | 4 | low_judged_at_10 |
| q_0006 | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 10 (0 / 10) | 0 | — | low_judged_at_10 |
| q_0007 | 23 | 0.809 | 1.000 | 0.826 | 1.000 | 0 (0 / 0) | 2 | 1 | — |
| q_0008 | 7 | 0.000 | 0.000 | 0.571 | 0.100 | 9 (9 / 0) | 1 | 27 | low_judged_at_10 |
| q_0009 | 11 | 0.235 | 0.500 | 0.364 | 0.500 | 5 (5 / 0) | 2 | 2 | low_judged_at_10 |
| q_0010 | 5 | 0.000 | 0.000 | 0.400 | 0.400 | 6 (6 / 0) | 4 | 49 | low_judged_at_10 |

Questions with insufficient relevance coverage: q_0001, q_0003, q_0004, q_0005, q_0006, q_0008, q_0009, q_0010.

## bge_zero_shot, company

Top 10 over 10 questions: 17 judged relevant, 35 confirmed negatives, 48 unjudged (48 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 5 | 0.208 | 0.250 | 0.600 | 0.400 | 6 (6 / 0) | 3 | 4 | low_judged_at_10 |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 0.900 | 1 (1 / 0) | 8 | 1 | — |
| q_0003 | 4 | 0.478 | 0.500 | 1.000 | 0.600 | 4 (4 / 0) | 4 | 2 | low_judged_at_10 |
| q_0004 | 8 | 0.139 | 1.000 | 0.375 | 0.600 | 4 (4 / 0) | 5 | 1 | low_judged_at_10 |
| q_0005 | 14 | 0.049 | 0.250 | 0.786 | 0.400 | 6 (6 / 0) | 3 | 4 | low_judged_at_10 |
| q_0006 | 5 | 0.000 | 0.000 | 0.000 | 0.300 | 7 (7 / 0) | 3 | — | low_judged_at_10 |
| q_0007 | 23 | 0.809 | 1.000 | 0.826 | 1.000 | 0 (0 / 0) | 2 | 1 | — |
| q_0008 | 7 | 0.000 | 0.000 | 0.571 | 0.100 | 9 (9 / 0) | 1 | 27 | low_judged_at_10 |
| q_0009 | 11 | 0.235 | 0.500 | 0.364 | 0.500 | 5 (5 / 0) | 2 | 2 | low_judged_at_10 |
| q_0010 | 5 | 0.000 | 0.000 | 0.400 | 0.400 | 6 (6 / 0) | 4 | 49 | low_judged_at_10 |

Questions with insufficient relevance coverage: q_0001, q_0003, q_0004, q_0005, q_0006, q_0008, q_0009, q_0010.

## e5_zero_shot, global

Top 10 over 10 questions: 20 judged relevant, 23 confirmed negatives, 57 unjudged (44 from the company asked about, 13 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 5 | 0.208 | 0.250 | 0.600 | 0.300 | 7 (1 / 6) | 2 | 4 | low_judged_at_10 |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 0.900 | 1 (1 / 0) | 8 | 1 | — |
| q_0003 | 4 | 0.736 | 1.000 | 1.000 | 0.500 | 5 (5 / 0) | 3 | 1 | low_judged_at_10 |
| q_0004 | 8 | 0.046 | 0.143 | 0.500 | 0.300 | 7 (7 / 0) | 2 | 7 | low_judged_at_10 |
| q_0005 | 14 | 0.317 | 1.000 | 0.786 | 0.300 | 7 (7 / 0) | 0 | 1 | low_judged_at_10 |
| q_0006 | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 10 (3 / 7) | 0 | — | low_judged_at_10 |
| q_0007 | 23 | 0.466 | 0.500 | 0.826 | 1.000 | 0 (0 / 0) | 4 | 2 | — |
| q_0008 | 7 | 0.254 | 1.000 | 0.714 | 0.200 | 8 (8 / 0) | 0 | 1 | low_judged_at_10 |
| q_0009 | 11 | 0.434 | 0.500 | 0.455 | 0.400 | 6 (6 / 0) | 1 | 2 | low_judged_at_10 |
| q_0010 | 5 | 0.125 | 0.111 | 0.400 | 0.400 | 6 (6 / 0) | 3 | 9 | low_judged_at_10 |

Questions with insufficient relevance coverage: q_0001, q_0003, q_0004, q_0005, q_0006, q_0008, q_0009, q_0010.

## e5_zero_shot, company

Top 10 over 10 questions: 21 judged relevant, 27 confirmed negatives, 52 unjudged (52 from the company asked about, 0 from other companies).

| Question | Positives | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged top 10 (in / out of scope) | Confirmed negatives top 10 | First relevant rank | Coverage flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_0001 | 5 | 0.367 | 0.500 | 0.600 | 0.800 | 2 (2 / 0) | 6 | 2 | — |
| q_0002 | 3 | 0.556 | 1.000 | 1.000 | 0.900 | 1 (1 / 0) | 8 | 1 | — |
| q_0003 | 4 | 0.736 | 1.000 | 1.000 | 0.500 | 5 (5 / 0) | 3 | 1 | low_judged_at_10 |
| q_0004 | 8 | 0.046 | 0.143 | 0.500 | 0.300 | 7 (7 / 0) | 2 | 7 | low_judged_at_10 |
| q_0005 | 14 | 0.317 | 1.000 | 0.786 | 0.300 | 7 (7 / 0) | 0 | 1 | low_judged_at_10 |
| q_0006 | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 10 (10 / 0) | 0 | — | low_judged_at_10 |
| q_0007 | 23 | 0.466 | 0.500 | 0.826 | 1.000 | 0 (0 / 0) | 4 | 2 | — |
| q_0008 | 7 | 0.254 | 1.000 | 0.714 | 0.200 | 8 (8 / 0) | 0 | 1 | low_judged_at_10 |
| q_0009 | 11 | 0.434 | 0.500 | 0.545 | 0.400 | 6 (6 / 0) | 1 | 2 | low_judged_at_10 |
| q_0010 | 5 | 0.125 | 0.111 | 0.400 | 0.400 | 6 (6 / 0) | 3 | 9 | low_judged_at_10 |

Questions with insufficient relevance coverage: q_0003, q_0004, q_0005, q_0006, q_0008, q_0009, q_0010.

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
