# Retrieval Analysis v1: generated tables
_Not a benchmark. Validation: 93 synthetic, AI-reviewed questions that chose each A1 checkpoint. Development: 10 questions with AI-written provisional labels. Real questions: a metadata alignment diagnostic on the 100 frozen human-written questions; no relevance label read, no relevance metric computed._

## T1-global. Validation (93 synthetic questions), global mode
| System | MRR@10 [95% CI] | Recall@10 [95% CI] | nDCG@10 [95% CI] | Recall@100 | Relevant in top 10 | Top 1 relevant |
|---|---|---|---|---|---|---|
| BM25 | 0.635 [0.551, 0.716] | 0.824 [0.746, 0.896] | 0.677 [0.599, 0.752] | 0.970 | 77/93 | 49/93 |
| BGE zero-shot | 0.266 [0.194, 0.340] | 0.460 [0.362, 0.555] | 0.306 [0.233, 0.379] | 0.837 | 46/93 | 16/93 |
| E5 zero-shot | 0.402 [0.321, 0.484] | 0.607 [0.512, 0.699] | 0.442 [0.363, 0.521] | 0.867 | 62/93 | 27/93 |
| A0 (step 0) | 0.266 [0.194, 0.340] | 0.460 [0.362, 0.555] | 0.306 [0.233, 0.379] | 0.837 | 46/93 | 16/93 |
| A1 seed 42 | 0.669 [0.588, 0.747] | 0.901 [0.841, 0.953] | 0.717 [0.646, 0.785] | 1.000 | 86/93 | 52/93 |
| A1 seed 1337 | 0.668 [0.587, 0.748] | 0.874 [0.806, 0.931] | 0.705 [0.634, 0.774] | 0.997 | 84/93 | 52/93 |
| A1 seed 2026 | 0.658 [0.580, 0.737] | 0.906 [0.846, 0.957] | 0.712 [0.644, 0.779] | 1.000 | 86/93 | 50/93 |

## T1-company. Validation (93 synthetic questions), company mode
| System | MRR@10 [95% CI] | Recall@10 [95% CI] | nDCG@10 [95% CI] | Recall@100 | Relevant in top 10 | Top 1 relevant |
|---|---|---|---|---|---|---|
| BM25 | 0.703 [0.624, 0.778] | 0.878 [0.806, 0.939] | 0.741 [0.669, 0.809] | 0.973 | 82/93 | 55/93 |
| BGE zero-shot | 0.282 [0.210, 0.356] | 0.514 [0.418, 0.608] | 0.329 [0.258, 0.401] | 0.869 | 52/93 | 16/93 |
| E5 zero-shot | 0.433 [0.350, 0.517] | 0.623 [0.530, 0.713] | 0.469 [0.388, 0.550] | 0.921 | 64/93 | 30/93 |
| A0 (step 0) | 0.282 [0.210, 0.356] | 0.514 [0.418, 0.608] | 0.329 [0.258, 0.401] | 0.869 | 52/93 | 16/93 |
| A1 seed 42 | 0.732 [0.655, 0.806] | 0.920 [0.865, 0.968] | 0.773 [0.705, 0.834] | 1.000 | 87/93 | 59/93 |
| A1 seed 1337 | 0.738 [0.662, 0.811] | 0.909 [0.851, 0.960] | 0.769 [0.703, 0.832] | 0.997 | 86/93 | 59/93 |
| A1 seed 2026 | 0.717 [0.642, 0.789] | 0.947 [0.901, 0.987] | 0.768 [0.706, 0.827] | 1.000 | 89/93 | 56/93 |

## T2. Validation, paired MRR@10 differences (question bootstrap; sign-flip test)
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

## T3. Reproduction of the recorded Stage A validation scores (Kaggle T4 → this Mac, CPU)
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

## T4-global. Development set (10 questions, AI-written provisional labels), global mode
| System | MRR@10 [95% CI] | Recall@10 [95% CI] | nDCG@10 [95% CI] | Recall@100 | judged@10 | Max abs Δ vs harness record |
|---|---|---|---|---|---|---|
| BM25 | 0.323 [0.108, 0.564] | 0.146 [0.057, 0.250] | 0.170 [0.061, 0.305] | 0.578 | 1.000 | 5.6e-17 |
| BGE zero-shot | 0.550 [0.300, 0.800] | 0.206 [0.122, 0.289] | 0.286 [0.145, 0.445] | 0.627 | 1.000 | 1.1e-16 |
| E5 zero-shot | 0.600 [0.340, 0.850] | 0.279 [0.181, 0.379] | 0.361 [0.210, 0.514] | 0.666 | 1.000 | 1.1e-16 |
| A0 (step 0) | 0.550 [0.300, 0.800] | 0.206 [0.122, 0.289] | 0.286 [0.145, 0.445] | 0.627 | 1.000 | 1.1e-16 |
| A1 seed 42 | 0.558 [0.308, 0.800] | 0.327 [0.175, 0.485] | 0.320 [0.176, 0.473] | 0.769 | 0.820 | 5.6e-17 |
| A1 seed 1337 | 0.583 [0.333, 0.800] | 0.327 [0.175, 0.485] | 0.324 [0.179, 0.479] | 0.750 | 0.850 | 2.2e-16 |
| A1 seed 2026 | 0.667 [0.400, 0.933] | 0.257 [0.143, 0.372] | 0.307 [0.167, 0.463] | 0.806 | 0.800 | 1.1e-16 |

## T4-company. Development set (10 questions, AI-written provisional labels), company mode
| System | MRR@10 [95% CI] | Recall@10 [95% CI] | nDCG@10 [95% CI] | Recall@100 | judged@10 | Max abs Δ vs harness record |
|---|---|---|---|---|---|---|
| BM25 | 0.423 [0.183, 0.679] | 0.184 [0.090, 0.287] | 0.194 [0.085, 0.329] | 0.674 | 1.000 | 1.1e-16 |
| BGE zero-shot | 0.575 [0.350, 0.800] | 0.239 [0.161, 0.311] | 0.297 [0.162, 0.450] | 0.643 | 1.000 | 5.6e-17 |
| E5 zero-shot | 0.625 [0.375, 0.864] | 0.296 [0.199, 0.390] | 0.376 [0.225, 0.524] | 0.673 | 1.000 | 1.1e-16 |
| A0 (step 0) | 0.575 [0.350, 0.800] | 0.239 [0.161, 0.311] | 0.297 [0.162, 0.450] | 0.643 | 1.000 | 5.6e-17 |
| A1 seed 42 | 0.592 [0.367, 0.808] | 0.344 [0.206, 0.491] | 0.328 [0.193, 0.477] | 0.839 | 0.810 | 1.1e-16 |
| A1 seed 1337 | 0.617 [0.400, 0.833] | 0.344 [0.206, 0.491] | 0.332 [0.194, 0.482] | 0.833 | 0.830 | 1.1e-16 |
| A1 seed 2026 | 0.717 [0.483, 0.933] | 0.307 [0.187, 0.435] | 0.323 [0.187, 0.476] | 0.839 | 0.800 | 1.1e-16 |

## T4-paired. Development set, paired differences (10 questions; question bootstrap; sign-flip test)
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

## T5-global. Real-question alignment diagnostic (100 human-written questions, top 10), global mode
| System | Correct company | Exact year | Exact year among correct company | Year-blind chance | Later filing (comparative window) | Cross-year | Cross-company | Top 1 in exact scope | Any in exact scope | All scope cells covered | Table passages | A1 training/validation passages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 78.7% | 12.4% | 15.8% | 17.6% | 20.3% | 46.0% | 21.3% | 15.0% | 70.0% | 56.0% | 52.0% | 17.0% |
| BGE zero-shot | 94.6% | 16.8% | 17.8% | 17.6% | 24.2% | 53.6% | 5.4% | 18.0% | 82.0% | 68.0% | 59.7% | 15.9% |
| E5 zero-shot | 94.1% | 26.4% | 28.1% | 17.6% | 28.6% | 39.1% | 5.9% | 22.0% | 91.0% | 79.0% | 67.8% | 14.9% |
| A0 (step 0) | 94.6% | 16.8% | 17.8% | 17.6% | 24.2% | 53.6% | 5.4% | 18.0% | 82.0% | 68.0% | 59.7% | 15.9% |
| A1 seed 42 | 82.9% | 23.1% | 27.9% | 17.6% | 21.2% | 38.6% | 17.1% | 19.0% | 90.0% | 78.0% | 59.5% | 13.1% |
| A1 seed 1337 | 83.0% | 20.5% | 24.7% | 17.6% | 22.1% | 40.4% | 17.0% | 21.0% | 87.0% | 74.0% | 54.6% | 13.4% |
| A1 seed 2026 | 82.2% | 21.1% | 25.7% | 17.6% | 22.4% | 38.7% | 17.8% | 23.0% | 85.0% | 73.0% | 58.6% | 14.2% |

## T5-company. Real-question alignment diagnostic (100 human-written questions, top 10), company mode
| System | Correct company | Exact year | Exact year among correct company | Year-blind chance | Later filing (comparative window) | Cross-year | Cross-company | Top 1 in exact scope | Any in exact scope | All scope cells covered | Table passages | A1 training/validation passages |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 100.0% | 16.3% | 16.3% | 17.6% | 25.2% | 58.5% | 0.0% | 17.0% | 85.0% | 71.0% | 51.4% | 15.2% |
| BGE zero-shot | 100.0% | 17.8% | 17.8% | 17.6% | 24.9% | 57.3% | 0.0% | 19.0% | 84.0% | 71.0% | 59.9% | 15.0% |
| E5 zero-shot | 100.0% | 28.2% | 28.2% | 17.6% | 29.7% | 42.1% | 0.0% | 22.0% | 94.0% | 82.0% | 67.3% | 14.3% |
| A0 (step 0) | 100.0% | 17.8% | 17.8% | 17.6% | 24.9% | 57.3% | 0.0% | 19.0% | 84.0% | 71.0% | 59.9% | 15.0% |
| A1 seed 42 | 100.0% | 27.5% | 27.5% | 17.6% | 24.9% | 47.6% | 0.0% | 21.0% | 96.0% | 83.0% | 59.9% | 13.6% |
| A1 seed 1337 | 100.0% | 25.5% | 25.5% | 17.6% | 26.0% | 48.5% | 0.0% | 21.0% | 95.0% | 81.0% | 54.4% | 13.8% |
| A1 seed 2026 | 100.0% | 26.0% | 26.0% | 17.6% | 26.3% | 47.7% | 0.0% | 24.0% | 92.0% | 80.0% | 58.0% | 14.5% |

## T6-by_question_type. Alignment by question type (global mode, top 10)
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

## T6-by_comparison. Alignment by comparison (global mode, top 10)
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

## T7. Validation error counts (global mode, 93 questions)
| System | Top 1 wrong company (of them, relevant still in top 10) | Top 1 right company, wrong year (not relevant) | Years off (wrong-year top 1) | Top 1 = same text from another filing (not relevant) | Relevant not in top 10 | Relevant not in top 100 | Answer figure in top 1 | Answer figure in top 10 |
|---|---|---|---|---|---|---|---|---|
| BM25 | 11 (6) | 16 | 1y: 11, 2y: 3, 3y: 2 | 0 | 16 | 2 | 38/64 | 54/64 |
| BGE zero-shot | 6 (2) | 56 | 1y: 19, 2y: 13, 3y: 13, 4y: 8, 5y: 1, 6y: 2 | 0 | 47 | 13 | 13/64 | 33/64 |
| E5 zero-shot | 13 (5) | 36 | 1y: 25, 2y: 3, 3y: 1, 4y: 4, 5y: 2, 6y: 1 | 0 | 31 | 11 | 19/64 | 43/64 |
| A0 (step 0) | 6 (2) | 56 | 1y: 19, 2y: 13, 3y: 13, 4y: 8, 5y: 1, 6y: 2 | 0 | 47 | 13 | 13/64 | 33/64 |
| A1 seed 42 | 14 (12) | 16 | 1y: 11, 2y: 3, 3y: 1, 5y: 1 | 0 | 7 | 0 | 39/64 | 61/64 |
| A1 seed 1337 | 15 (12) | 12 | 1y: 8, 2y: 2, 3y: 1, 5y: 1 | 0 | 9 | 0 | 41/64 | 61/64 |
| A1 seed 2026 | 12 (10) | 16 | 1y: 11, 2y: 2, 3y: 1, 4y: 1, 5y: 1 | 0 | 7 | 0 | 39/64 | 61/64 |

## T10. Validation complementarity: relevant passage in the top 10 of a dense system and of BM25
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

## T8. Validation MRR@10 by evidence type (global mode)
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

## T9. Agreement between dense systems (global mode, top 10)
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
