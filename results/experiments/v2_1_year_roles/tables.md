# FinRetriever v2 experiments: generated tables
_Not a benchmark. Validation questions are synthetic and selected the A1 checkpoints; development labels are AI-written; the 100 real questions have no relevance labels and are scored for metadata alignment and parser accuracy only._

## V1. Validation (93 synthetic questions; they selected A1). Mean over seeds (seed range)
| Experiment | Role | Seeds | MRR@10 | Recall@10 | nDCG@10 | Recall@100 | Δ MRR@10 vs E1 (largest p) | Δ Recall@10 vs E1 (largest p) |
|---|---|---|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.665 (0.658–0.669) | 0.893 (0.874–0.906) | 0.712 (0.705–0.717) | 0.999 (0.997–1.000) | — | — |
| E3_a1_year | primary | 3 | 0.679 (0.678–0.681) | 0.888 (0.879–0.901) | 0.707 (0.702–0.712) | 0.962 (0.962–0.962) | +0.014 (p ≤ 0.0005) | -0.005 (p ≤ 0.8637) |
| E3b_a1_company_year | primary | 3 | 0.746 (0.737–0.755) | 0.920 (0.915–0.931) | 0.768 (0.766–0.770) | 0.962 (0.962–0.962) | +0.081 (p ≤ 0.0001) | +0.027 (p ≤ 0.5929) |
| E5_hybrid_rrf_meta | primary | 3 | 0.851 (0.843–0.863) | 0.956 (0.949–0.963) | 0.854 (0.847–0.861) | 0.994 (0.993–0.995) | +0.186 (p ≤ 0.0001) | +0.062 (p ≤ 0.1333) |
| E7_combined | primary | 3 | 0.825 (0.817–0.830) | 0.935 (0.929–0.944) | 0.836 (0.830–0.839) | 0.962 (0.962–0.962) | +0.160 (p ≤ 0.0002) | +0.042 (p ≤ 0.4782) |
| X_combined_boost | exploratory | 3 | 0.836 (0.828–0.841) | 0.961 (0.960–0.962) | 0.835 (0.831–0.838) | 0.992 (0.992–0.992) | +0.171 (p ≤ 0.0001) | +0.067 (p ≤ 0.0680) |

## D1. Development (10 questions, AI-written provisional labels; unjudged = non-relevant)
| Experiment | Role | Seeds | MRR@10 | Recall@10 | nDCG@10 | Recall@100 | judged@10 | MRR@10 judged-only | Δ MRR@10 vs E1 (largest p) |
|---|---|---|---|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.603 (0.558–0.667) | 0.304 (0.257–0.327) | 0.317 (0.307–0.324) | 0.775 (0.750–0.806) | 0.823 (0.800–0.850) | 0.612 (0.577–0.667) | — |
| E3_a1_year | primary | 3 | 0.646 (0.618–0.677) | 0.372 (0.361–0.377) | 0.353 (0.348–0.360) | 0.817 (0.808–0.823) | 0.763 (0.760–0.770) | 0.659 (0.643–0.689) | +0.043 (p ≤ 1.0000) |
| E3b_a1_company_year | primary | 3 | 0.675 (0.642–0.717) | 0.372 (0.361–0.377) | 0.357 (0.354–0.364) | 0.877 (0.870–0.883) | 0.730 (0.710–0.760) | 0.747 (0.731–0.778) | +0.072 (p ≤ 1.0000) |
| E5_hybrid_rrf_meta | primary | 3 | 0.642 (0.625–0.667) | 0.375 (0.369–0.377) | 0.365 (0.359–0.373) | 0.863 (0.852–0.868) | 0.857 (0.840–0.870) | 0.642 (0.625–0.667) | +0.039 (p ≤ 0.8125) |
| E7_combined | primary | 3 | 0.676 (0.637–0.753) | 0.398 (0.390–0.405) | 0.343 (0.341–0.345) | 0.896 (0.889–0.906) | 0.837 (0.830–0.840) | 0.676 (0.637–0.753) | +0.073 (p ≤ 0.6875) |
| X_combined_boost | exploratory | 3 | 0.742 (0.742–0.742) | 0.374 (0.367–0.379) | 0.383 (0.381–0.385) | 0.878 (0.873–0.881) | 0.783 (0.780–0.790) | 0.742 (0.742–0.742) | +0.139 (p ≤ 0.7500) |

## A1. Real-question metadata alignment (100 human-written questions, top 10; mean over seeds). Not relevance; scoped systems align by construction when the parser is right
| Experiment | Role | Seeds | Correct company | Cross-company | Exact year | Cross-year | Top 1 in exact scope | Any in exact scope | All scope cells covered |
|---|---|---|---|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 82.7% | 17.3% | 21.6% | 39.2% | 21.0% | 87.3% | 75.0% |
| E3_a1_year | primary | 3 | 80.7% | 19.3% | 42.6% | 0.0% | 45.3% | 96.3% | 84.7% |
| E3b_a1_company_year | primary | 3 | 100.0% | 0.0% | 53.8% | 0.0% | 52.3% | 100.0% | 92.0% |
| E5_hybrid_rrf_meta | primary | 3 | 100.0% | 0.0% | 48.7% | 25.9% | 92.3% | 99.3% | 92.7% |
| E7_combined | primary | 3 | 100.0% | 0.0% | 55.0% | 0.0% | 64.7% | 100.0% | 95.3% |
| X_combined_boost | exploratory | 3 | 100.0% | 0.0% | 53.0% | 24.0% | 92.3% | 100.0% | 97.0% |

## A2. Real questions whose top 10 covers every company-year cell they ask about, by comparison type (mean over seeds, seed range)
| Experiment | Role | Seeds | Cross-company | Cross-year | Single scope |
|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.3 (0–1) of 10 | 14.0 (13–15) of 21 | 60.7 (59–62) of 69 |
| E3_a1_year | primary | 3 | 4.0 (3–5) of 10 | 15.0 (14–16) of 21 | 65.7 (65–66) of 69 |
| E3b_a1_company_year | primary | 3 | 5.0 (4–6) of 10 | 18.0 (17–19) of 21 | 69 of 69 |
| E5_hybrid_rrf_meta | primary | 3 | 4.3 (4–5) of 10 | 20 of 21 | 68.3 (68–69) of 69 |
| E7_combined | primary | 3 | 7.7 (7–8) of 10 | 18.7 (18–19) of 21 | 69 of 69 |
| X_combined_boost | exploratory | 3 | 8.7 (8–9) of 10 | 19.3 (19–20) of 21 | 69 of 69 |

## S1. Scope decisions per question set: company filtered / year filtered / year fallback / cells
| Experiment | Validation | Development | Real questions |
|---|---|---|---|
| E3_a1_year | 0 / 87 / 0 / 0 of 93 | 0 / 10 / 0 / 0 of 10 | 0 / 100 / 0 / 0 of 100 |
| E3b_a1_company_year | 93 / 87 / 0 / 0 of 93 | 10 / 10 / 0 / 0 of 10 | 99 / 100 / 0 / 0 of 100 |
| E5_hybrid_rrf_meta | 93 / 0 / 0 / 0 of 93 | 10 / 0 / 0 / 0 of 10 | 99 / 0 / 0 / 0 of 100 |
| E7_combined | 93 / 87 / 0 / 0 of 93 | 10 / 10 / 0 / 2 of 10 | 99 / 100 / 0 / 31 of 100 |
| X_combined_boost | 93 / 0 / 0 / 0 of 93 | 10 / 0 / 0 / 0 of 10 | 99 / 0 / 0 / 10 of 100 |

## P1. Query understanding against each set's declared scope (parser qu-v2; validation: the positive passage's company and filing year)
| Set | Questions | Companies exact | Declared company missed | Extra company | Years exact | Declared years inside the year filter | No year parsed | Comparison type agrees |
|---|---|---|---|---|---|---|---|---|
| validation | 93 | 93/93 | 0 | 0 | 75/93 | 87/87 | 6 | — |
| development | 10 | 10/10 | 0 | 0 | 10/10 | 10/10 | 0 | 10/10 |
| real | 100 | 99/100 | 1 | 0 | 100/100 | 100/100 | 0 | 100/100 |

## P2. Year roles (parser qu-v2): questions with a non-reporting year, declared years read as non-reporting, and year mentions by role
| Set | Questions | With a non-reporting year | Declared year given a non-reporting role | fiscal | target | historical | maturity | incidental |
|---|---|---|---|---|---|---|---|---|
| validation | 93 | 3 | 1 | 88 | 1 | 1 | 1 | 1 |
| development | 10 | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| real | 100 | 0 | 0 | 124 | 0 | 0 | 0 | 0 |

## W1. Fiscal-year window candidates on the 771 training and validation questions (the positive passage's filing year kept; never measured on the real questions)
| Window | Filing year kept | Mean share of the company's fiscal years allowed |
|---|---|---|
| exact year + 2 later filings | 681/694 (98.1%) | 36.8% |
| chosen: also 1 earlier filing for years read from dates or plain numbers | 689/694 (99.3%) | 43.2% |
| 1 earlier filing for every year | 690/694 (99.4%) | 48.5% |

## Q1. Baseline and primary experiments per seed
| Experiment | Seed | Validation MRR@10 | Validation Recall@10 | Validation nDCG@10 | Development MRR@10 | Development Recall@10 |
|---|---|---|---|---|---|---|
| E1_a1 | a1_seed42 | 0.669 | 0.901 | 0.717 | 0.558 | 0.327 |
| E1_a1 | a1_seed1337 | 0.668 | 0.874 | 0.705 | 0.583 | 0.327 |
| E1_a1 | a1_seed2026 | 0.658 | 0.906 | 0.712 | 0.667 | 0.257 |
| E3_a1_year | a1_seed42 | 0.679 | 0.884 | 0.707 | 0.618 | 0.377 |
| E3_a1_year | a1_seed1337 | 0.681 | 0.879 | 0.702 | 0.643 | 0.377 |
| E3_a1_year | a1_seed2026 | 0.678 | 0.901 | 0.712 | 0.677 | 0.361 |
| E3b_a1_company_year | a1_seed42 | 0.746 | 0.915 | 0.767 | 0.642 | 0.377 |
| E3b_a1_company_year | a1_seed1337 | 0.755 | 0.915 | 0.770 | 0.667 | 0.377 |
| E3b_a1_company_year | a1_seed2026 | 0.737 | 0.931 | 0.766 | 0.717 | 0.361 |
| E5_hybrid_rrf_meta | a1_seed42 | 0.863 | 0.955 | 0.861 | 0.667 | 0.377 |
| E5_hybrid_rrf_meta | a1_seed1337 | 0.848 | 0.963 | 0.854 | 0.625 | 0.369 |
| E5_hybrid_rrf_meta | a1_seed2026 | 0.843 | 0.949 | 0.847 | 0.633 | 0.377 |
| E7_combined | a1_seed42 | 0.828 | 0.934 | 0.837 | 0.637 | 0.405 |
| E7_combined | a1_seed1337 | 0.830 | 0.944 | 0.839 | 0.637 | 0.398 |
| E7_combined | a1_seed2026 | 0.817 | 0.929 | 0.830 | 0.753 | 0.390 |

## V2. Validation MRR@10 by where the evidence starts in its positive passage (encoder tokens; every validation evidence span ends inside the 510-token window)
| Experiment | Role | Seeds | Evidence starts at token 0-127 | Evidence starts at token 128-255 | Evidence starts at token 256+ |
|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.694 (0.659–0.713) (n 21) | 0.795 (0.787–0.800) (n 24) | 0.578 (0.577–0.580) (n 47) |
| E7_combined | primary | 3 | 0.739 (0.715–0.754) (n 21) | 0.961 (0.951–0.979) (n 24) | 0.790 (0.776–0.797) (n 47) |

## R1. This run minus the reference run `results/experiments/v2_metadata_hybrid` (parser qu-v1), same experiment and seed: mean over seeds (largest p over seeds), and questions whose top 10 changed
| Experiment | Role | Seeds | Δ validation MRR@10 | Δ validation Recall@100 | Δ development MRR@10 | Δ real exact-year share | Validation top 10 changed | Development top 10 changed | Real top 10 changed |
|---|---|---|---|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | 0 | 0 | 0 |
| E3_a1_year | primary | 3 | +0.005 (p ≤ 1.0000) | +0.011 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | 2 | 0 | 0 |
| E3b_a1_company_year | primary | 3 | +0.005 (p ≤ 1.0000) | +0.011 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | 2 | 0 | 0 |
| E5_hybrid_rrf_meta | primary | 3 | +0.009 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | 2 | 0 | 0 |
| E7_combined | primary | 3 | +0.011 (p ≤ 1.0000) | +0.011 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | 2 | 0 | 0 |
| X_combined_boost | exploratory | 3 | +0.009 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | +0.000 (p ≤ 1.0000) | 2 | 0 | 0 |

## B1. Baseline preservation: v2 runs against the frozen v1 analysis (validation)
| Check | Result |
|---|---|
| E1_a1 [a1_seed42] vs v1 a1_seed42/global | 93/93 first-relevant ranks identical |
| E1_a1 [a1_seed1337] vs v1 a1_seed1337/global | 93/93 first-relevant ranks identical |
| E1_a1 [a1_seed2026] vs v1 a1_seed2026/global | 93/93 first-relevant ranks identical |
