# Retrieval Analysis v1: supplementary tables

## S1. Passages whose text names their own company
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

## S6. A1's training pairs (train_candidates_v3): company names in question and positive passage
| Training pairs | Count |
|---|---|
| Questions naming the company | 677/678 |
| Positive passages naming the company | 273/678 |
| Question names it, positive passage does not | 405/678 |

## S2. Real-question top 10: share of passages naming their company
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

## S3. Validation MRR@10 (global) by where the evidence starts in its passage (BGE tokens; 92/93 located; median start 273; 0 ending beyond 510)
| System | Starts at 0-127 (n) | Starts at 128-255 (n) | Starts at 256+ (n) |
|---|---|---|---|
| BM25 | 0.530 (21) | 0.634 (24) | 0.674 (47) |
| BGE zero-shot | 0.444 (21) | 0.251 (24) | 0.189 (47) |
| E5 zero-shot | 0.476 (21) | 0.387 (24) | 0.364 (47) |
| A0 (step 0) | 0.444 (21) | 0.251 (24) | 0.189 (47) |
| A1 seed 42 | 0.709 (21) | 0.798 (24) | 0.577 (47) |
| A1 seed 1337 | 0.713 (21) | 0.787 (24) | 0.580 (47) |
| A1 seed 2026 | 0.659 (21) | 0.800 (24) | 0.578 (47) |

## S4. Real questions whose top 10 is mostly (> 50%) from other companies (of 100)
| System | Global | Company |
|---|---|---|
| BM25 | 15 | 0 |
| BGE zero-shot | 1 | 0 |
| E5 zero-shot | 2 | 0 |
| A0 (step 0) | 1 | 0 |
| A1 seed 42 | 15 | 0 |
| A1 seed 1337 | 14 | 0 |
| A1 seed 2026 | 14 | 0 |

## S5. Real single-company questions: cross-company share of the top 10 by company asked (global)
| System | AAPL (13 q) | AMZN (13 q) | JPM (13 q) | PFE (13 q) | TSLA (13 q) | WMT (13 q) | XOM (12 q) |
|---|---|---|---|---|---|---|---|
| BM25 | 33.8% | 43.1% | 0.0% | 14.6% | 20.8% | 22.3% | 3.3% |
| BGE zero-shot | 19.2% | 5.4% | 0.0% | 7.7% | 3.8% | 0.0% | 0.0% |
| E5 zero-shot | 28.5% | 3.8% | 0.0% | 3.1% | 1.5% | 1.5% | 0.0% |
| A0 (step 0) | 19.2% | 5.4% | 0.0% | 7.7% | 3.8% | 0.0% | 0.0% |
| A1 seed 42 | 38.5% | 29.2% | 0.8% | 6.2% | 3.8% | 31.5% | 2.5% |
| A1 seed 1337 | 40.0% | 28.5% | 0.0% | 8.5% | 6.2% | 28.5% | 4.2% |
| A1 seed 2026 | 40.8% | 30.8% | 0.8% | 8.5% | 4.6% | 30.8% | 3.3% |
