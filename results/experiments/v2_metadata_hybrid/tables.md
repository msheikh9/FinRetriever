# FinRetriever v2 experiments: generated tables
_Not a benchmark. Validation questions are synthetic and selected the A1 checkpoints; development labels are AI-written; the 100 real questions have no relevance labels and are scored for metadata alignment and parser accuracy only._

## V1. Validation (93 synthetic questions; they selected A1). Mean over seeds (seed range)
| Experiment | Role | Seeds | MRR@10 | Recall@10 | nDCG@10 | Recall@100 | Δ MRR@10 vs E1 (largest p) | Δ Recall@10 vs E1 (largest p) |
|---|---|---|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.665 (0.658–0.669) | 0.893 (0.874–0.906) | 0.712 (0.705–0.717) | 0.999 (0.997–1.000) | — | — |
| R_bm25 | reference | seedless | 0.635 | 0.824 | 0.677 | 0.970 | -0.030 (p ≤ 0.6443) | -0.069 (p ≤ 0.2824) |
| R_a1_oracle_company | reference | 3 | 0.729 (0.717–0.738) | 0.926 (0.909–0.947) | 0.770 (0.768–0.773) | 0.999 (0.997–1.000) | +0.064 (p ≤ 0.0001) | +0.032 (p ≤ 0.2453) |
| R_a1_oracle_company_year | reference | 3 | 0.766 (0.752–0.783) | 0.904 (0.895–0.911) | 0.769 (0.764–0.776) | 0.941 (0.941–0.941) | +0.101 (p ≤ 0.0001) | +0.011 (p ≤ 0.8635) |
| E2_a1_company | primary | 3 | 0.729 (0.717–0.738) | 0.926 (0.909–0.947) | 0.770 (0.768–0.773) | 0.999 (0.997–1.000) | +0.064 (p ≤ 0.0001) | +0.032 (p ≤ 0.2453) |
| E3_a1_year | primary | 3 | 0.674 (0.673–0.675) | 0.877 (0.868–0.890) | 0.700 (0.695–0.705) | 0.951 (0.951–0.951) | +0.009 (p ≤ 0.5502) | -0.016 (p ≤ 0.8544) |
| E3b_a1_company_year | primary | 3 | 0.740 (0.731–0.750) | 0.909 (0.904–0.920) | 0.761 (0.759–0.763) | 0.951 (0.951–0.951) | +0.076 (p ≤ 0.0001) | +0.016 (p ≤ 0.9346) |
| E4_hybrid_rrf | primary | 3 | 0.718 (0.716–0.719) | 0.913 (0.909–0.916) | 0.763 (0.763–0.764) | 0.998 (0.997–0.998) | +0.053 (p ≤ 0.2197) | +0.019 (p ≤ 0.9725) |
| E5_hybrid_rrf_meta | primary | 3 | 0.842 (0.834–0.854) | 0.956 (0.949–0.963) | 0.847 (0.840–0.854) | 0.994 (0.993–0.995) | +0.177 (p ≤ 0.0001) | +0.062 (p ≤ 0.1333) |
| E6_a1_enriched | primary | 3 | 0.749 (0.737–0.757) | 0.914 (0.908–0.926) | 0.771 (0.761–0.780) | 0.990 (0.989–0.992) | +0.084 (p ≤ 0.0062) | +0.021 (p ≤ 1.0000) |
| E7_combined | primary | 3 | 0.814 (0.807–0.819) | 0.925 (0.918–0.933) | 0.825 (0.820–0.829) | 0.951 (0.951–0.951) | +0.149 (p ≤ 0.0005) | +0.031 (p ≤ 0.7247) |
| E8_a1_tail_windows | primary | 3 | 0.660 (0.655–0.669) | 0.870 (0.864–0.879) | 0.697 (0.691–0.704) | 0.994 (0.986–0.998) | -0.005 (p ≤ 0.9760) | -0.023 (p ≤ 0.8176) |
| X_rrf_k10 | exploratory | 3 | 0.732 (0.730–0.735) | 0.924 (0.915–0.931) | 0.776 (0.772–0.778) | 0.998 (0.997–0.998) | +0.067 (p ≤ 0.0862) | +0.031 (p ≤ 0.4527) |
| X_rrf_k100 | exploratory | 3 | 0.719 (0.718–0.722) | 0.915 (0.909–0.924) | 0.764 (0.762–0.766) | 0.998 (0.997–0.998) | +0.054 (p ≤ 0.2302) | +0.022 (p ≤ 0.9725) |
| X_rrf_dense2 | exploratory | 3 | 0.741 (0.738–0.745) | 0.951 (0.946–0.954) | 0.786 (0.783–0.791) | 0.999 (0.997–1.000) | +0.076 (p ≤ 0.0364) | +0.058 (p ≤ 0.0721) |
| X_rrf_bm25x2 | exploratory | 3 | 0.716 (0.712–0.720) | 0.907 (0.905–0.907) | 0.760 (0.757–0.762) | 0.991 (0.991–0.991) | +0.051 (p ≤ 0.3194) | +0.013 (p ≤ 1.0000) |
| X_hybrid_company_year_filter | exploratory | 3 | 0.807 (0.797–0.813) | 0.922 (0.921–0.925) | 0.823 (0.815–0.829) | 0.951 (0.951–0.951) | +0.142 (p ≤ 0.0005) | +0.029 (p ≤ 0.6540) |
| X_a1_company_year_exact | exploratory | 3 | 0.736 (0.728–0.748) | 0.774 (0.771–0.781) | 0.680 (0.676–0.688) | 0.783 (0.783–0.783) | +0.071 (p ≤ 0.0544) | -0.119 (p ≤ 0.0596) |
| X_a1_company_year_boost | exploratory | 3 | 0.747 (0.740–0.757) | 0.914 (0.914–0.914) | 0.742 (0.739–0.747) | 0.999 (0.997–1.000) | +0.082 (p ≤ 0.0138) | +0.021 (p ≤ 0.7740) |
| X_bm25_enriched | exploratory | seedless | 0.735 | 0.938 | 0.773 | 0.982 | +0.070 (p ≤ 0.1757) | +0.045 (p ≤ 0.3389) |
| X_hybrid_enriched_both | exploratory | 3 | 0.807 (0.793–0.816) | 0.949 (0.939–0.961) | 0.836 (0.823–0.843) | 0.991 (0.991–0.991) | +0.142 (p ≤ 0.0004) | +0.056 (p ≤ 0.3035) |
| X_combined_boost | exploratory | 3 | 0.827 (0.819–0.832) | 0.961 (0.960–0.962) | 0.828 (0.824–0.831) | 0.992 (0.992–0.992) | +0.162 (p ≤ 0.0002) | +0.067 (p ≤ 0.0680) |
| X_combined_no_cells | exploratory | 3 | 0.814 (0.807–0.819) | 0.925 (0.918–0.933) | 0.825 (0.820–0.829) | 0.951 (0.951–0.951) | +0.149 (p ≤ 0.0005) | +0.031 (p ≤ 0.7247) |
| X_combined_plain_a1 | exploratory | 3 | 0.807 (0.797–0.813) | 0.922 (0.921–0.925) | 0.823 (0.815–0.829) | 0.951 (0.951–0.951) | +0.142 (p ≤ 0.0005) | +0.029 (p ≤ 0.6540) |
| X_combined_tail | exploratory | 3 | 0.778 (0.774–0.783) | 0.920 (0.918–0.922) | 0.798 (0.797–0.801) | 0.951 (0.951–0.951) | +0.113 (p ≤ 0.0065) | +0.027 (p ≤ 0.6310) |
| X_a1_sliding256 | exploratory | 1 | 0.720 | 0.923 | 0.760 | 1.000 | +0.052 (p 0.1987) | +0.022 (p 0.3604) |

## D1. Development (10 questions, AI-written provisional labels; unjudged = non-relevant)
| Experiment | Role | Seeds | MRR@10 | Recall@10 | nDCG@10 | Recall@100 | judged@10 | MRR@10 judged-only | Δ MRR@10 vs E1 (largest p) |
|---|---|---|---|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.603 (0.558–0.667) | 0.304 (0.257–0.327) | 0.317 (0.307–0.324) | 0.775 (0.750–0.806) | 0.823 (0.800–0.850) | 0.612 (0.577–0.667) | — |
| R_bm25 | reference | seedless | 0.323 | 0.146 | 0.170 | 0.578 | 1.000 | 0.323 | -0.280 (p ≤ 0.0781) |
| R_a1_oracle_company | reference | 3 | 0.642 (0.592–0.717) | 0.332 (0.307–0.344) | 0.328 (0.323–0.332) | 0.837 (0.833–0.839) | 0.813 (0.800–0.830) | 0.706 (0.667–0.767) | +0.039 (p ≤ 1.0000) |
| R_a1_oracle_company_year | reference | 3 | 0.642 (0.614–0.678) | 0.379 (0.373–0.387) | 0.370 (0.368–0.371) | 0.869 (0.854–0.877) | 0.720 (0.700–0.750) | 0.655 (0.639–0.681) | +0.039 (p ≤ 1.0000) |
| E2_a1_company | primary | 3 | 0.642 (0.592–0.717) | 0.332 (0.307–0.344) | 0.328 (0.323–0.332) | 0.837 (0.833–0.839) | 0.813 (0.800–0.830) | 0.706 (0.667–0.767) | +0.039 (p ≤ 1.0000) |
| E3_a1_year | primary | 3 | 0.646 (0.618–0.677) | 0.372 (0.361–0.377) | 0.353 (0.348–0.360) | 0.817 (0.808–0.823) | 0.763 (0.760–0.770) | 0.659 (0.643–0.689) | +0.043 (p ≤ 1.0000) |
| E3b_a1_company_year | primary | 3 | 0.675 (0.642–0.717) | 0.372 (0.361–0.377) | 0.357 (0.354–0.364) | 0.877 (0.870–0.883) | 0.730 (0.710–0.760) | 0.747 (0.731–0.778) | +0.072 (p ≤ 1.0000) |
| E4_hybrid_rrf | primary | 3 | 0.602 (0.597–0.610) | 0.278 (0.243–0.307) | 0.264 (0.254–0.276) | 0.805 (0.781–0.827) | 0.983 (0.980–0.990) | 0.602 (0.599–0.610) | -0.001 (p ≤ 0.8750) |
| E5_hybrid_rrf_meta | primary | 3 | 0.642 (0.625–0.667) | 0.375 (0.369–0.377) | 0.365 (0.359–0.373) | 0.863 (0.852–0.868) | 0.857 (0.840–0.870) | 0.642 (0.625–0.667) | +0.039 (p ≤ 0.8125) |
| E6_a1_enriched | primary | 3 | 0.635 (0.597–0.656) | 0.382 (0.362–0.404) | 0.367 (0.360–0.378) | 0.811 (0.806–0.814) | 0.723 (0.710–0.740) | 0.725 (0.692–0.742) | +0.032 (p ≤ 0.6094) |
| E7_combined | primary | 3 | 0.676 (0.637–0.753) | 0.398 (0.390–0.405) | 0.343 (0.341–0.345) | 0.896 (0.889–0.906) | 0.837 (0.830–0.840) | 0.676 (0.637–0.753) | +0.073 (p ≤ 0.6875) |
| E8_a1_tail_windows | primary | 3 | 0.601 (0.550–0.670) | 0.369 (0.346–0.395) | 0.333 (0.306–0.348) | 0.793 (0.765–0.818) | 0.803 (0.800–0.810) | 0.642 (0.620–0.670) | -0.002 (p ≤ 1.0000) |
| X_rrf_k10 | exploratory | 3 | 0.607 (0.602–0.613) | 0.272 (0.256–0.286) | 0.277 (0.273–0.280) | 0.805 (0.781–0.827) | 0.963 (0.950–0.970) | 0.607 (0.602–0.613) | +0.004 (p ≤ 1.0000) |
| X_rrf_k100 | exploratory | 3 | 0.597 (0.588–0.607) | 0.262 (0.249–0.282) | 0.257 (0.253–0.259) | 0.805 (0.781–0.827) | 0.977 (0.970–0.980) | 0.601 (0.597–0.608) | -0.005 (p ≤ 1.0000) |
| X_rrf_dense2 | exploratory | 3 | 0.578 (0.553–0.602) | 0.294 (0.285–0.301) | 0.276 (0.270–0.281) | 0.798 (0.768–0.831) | 0.930 (0.930–0.930) | 0.579 (0.553–0.603) | -0.025 (p ≤ 1.0000) |
| X_rrf_bm25x2 | exploratory | 3 | 0.583 (0.583–0.584) | 0.241 (0.241–0.241) | 0.258 (0.258–0.259) | 0.709 (0.700–0.723) | 1.000 (1.000–1.000) | 0.583 (0.583–0.584) | -0.020 (p ≤ 1.0000) |
| X_hybrid_company_year_filter | exploratory | 3 | 0.654 (0.648–0.662) | 0.377 (0.365–0.396) | 0.318 (0.314–0.324) | 0.892 (0.885–0.902) | 0.917 (0.910–0.920) | 0.654 (0.648–0.662) | +0.052 (p ≤ 1.0000) |
| X_a1_company_year_exact | exploratory | 3 | 0.679 (0.645–0.717) | 0.421 (0.413–0.432) | 0.415 (0.403–0.420) | 0.736 (0.736–0.736) | 0.617 (0.610–0.620) | 0.711 (0.692–0.746) | +0.076 (p ≤ 0.5000) |
| X_a1_company_year_boost | exploratory | 3 | 0.675 (0.640–0.719) | 0.409 (0.394–0.428) | 0.402 (0.397–0.406) | 0.837 (0.833–0.839) | 0.693 (0.690–0.700) | 0.699 (0.675–0.737) | +0.072 (p ≤ 0.2500) |
| X_bm25_enriched | exploratory | seedless | 0.402 | 0.154 | 0.169 | 0.595 | 0.880 | 0.403 | -0.201 (p ≤ 0.1875) |
| X_hybrid_enriched_both | exploratory | 3 | 0.546 (0.520–0.582) | 0.267 (0.267–0.267) | 0.260 (0.258–0.261) | 0.854 (0.841–0.866) | 0.883 (0.880–0.890) | 0.560 (0.537–0.590) | -0.057 (p ≤ 0.8750) |
| X_combined_boost | exploratory | 3 | 0.742 (0.742–0.742) | 0.374 (0.367–0.379) | 0.383 (0.381–0.385) | 0.878 (0.873–0.881) | 0.783 (0.780–0.790) | 0.742 (0.742–0.742) | +0.139 (p ≤ 0.7500) |
| X_combined_no_cells | exploratory | 3 | 0.676 (0.637–0.753) | 0.393 (0.386–0.400) | 0.337 (0.336–0.339) | 0.895 (0.883–0.906) | 0.840 (0.830–0.850) | 0.676 (0.637–0.753) | +0.073 (p ≤ 0.6875) |
| X_combined_plain_a1 | exploratory | 3 | 0.654 (0.648–0.662) | 0.377 (0.365–0.396) | 0.318 (0.316–0.323) | 0.892 (0.885–0.902) | 0.917 (0.910–0.920) | 0.654 (0.648–0.662) | +0.052 (p ≤ 1.0000) |
| X_combined_tail | exploratory | 3 | 0.599 (0.593–0.603) | 0.376 (0.376–0.378) | 0.326 (0.320–0.334) | 0.899 (0.894–0.904) | 0.923 (0.910–0.930) | 0.602 (0.595–0.608) | -0.004 (p ≤ 0.8750) |
| X_a1_sliding256 | exploratory | 1 | 0.781 | 0.370 | 0.384 | 0.813 | 0.850 | 0.782 | +0.223 (p 0.0312) |

## A1. Real-question metadata alignment (100 human-written questions, top 10; mean over seeds). Not relevance; scoped systems align by construction when the parser is right
| Experiment | Role | Seeds | Correct company | Cross-company | Exact year | Cross-year | Top 1 in exact scope | Any in exact scope | All scope cells covered |
|---|---|---|---|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 82.7% | 17.3% | 21.6% | 39.2% | 21.0% | 87.3% | 75.0% |
| R_bm25 | reference | seedless | 78.7% | 21.3% | 12.4% | 46.0% | 15.0% | 70.0% | 56.0% |
| R_a1_oracle_company | reference | 3 | 100.0% | 0.0% | 26.3% | 47.9% | 22.0% | 94.3% | 81.3% |
| R_a1_oracle_company_year | reference | 3 | 100.0% | 0.0% | 53.8% | 0.0% | 52.3% | 100.0% | 92.0% |
| E2_a1_company | primary | 3 | 100.0% | 0.0% | 26.3% | 47.9% | 22.0% | 94.3% | 81.3% |
| E3_a1_year | primary | 3 | 80.7% | 19.3% | 42.6% | 0.0% | 45.3% | 96.3% | 84.7% |
| E3b_a1_company_year | primary | 3 | 100.0% | 0.0% | 53.8% | 0.0% | 52.3% | 100.0% | 92.0% |
| E4_hybrid_rrf | primary | 3 | 82.4% | 17.6% | 17.9% | 43.6% | 16.7% | 82.7% | 67.3% |
| E5_hybrid_rrf_meta | primary | 3 | 100.0% | 0.0% | 48.7% | 25.9% | 92.3% | 99.3% | 92.7% |
| E6_a1_enriched | primary | 3 | 93.6% | 6.4% | 31.3% | 40.7% | 38.3% | 95.3% | 86.0% |
| E7_combined | primary | 3 | 100.0% | 0.0% | 55.0% | 0.0% | 64.7% | 100.0% | 95.3% |
| E8_a1_tail_windows | primary | 3 | 83.3% | 16.7% | 22.8% | 38.2% | 21.7% | 90.0% | 74.0% |
| X_rrf_k10 | exploratory | 3 | 83.6% | 16.4% | 18.1% | 43.6% | 18.0% | 84.0% | 71.0% |
| X_rrf_k100 | exploratory | 3 | 82.5% | 17.5% | 17.9% | 43.7% | 16.3% | 83.3% | 67.0% |
| X_rrf_dense2 | exploratory | 3 | 83.3% | 16.7% | 19.2% | 42.3% | 17.7% | 85.7% | 68.0% |
| X_rrf_bm25x2 | exploratory | 3 | 81.6% | 18.4% | 16.7% | 44.4% | 16.7% | 82.7% | 67.7% |
| X_hybrid_company_year_filter | exploratory | 3 | 100.0% | 0.0% | 51.1% | 0.0% | 49.7% | 100.0% | 89.0% |
| X_a1_company_year_exact | exploratory | 3 | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% | 97.7% |
| X_a1_company_year_boost | exploratory | 3 | 100.0% | 0.0% | 66.6% | 10.6% | 100.0% | 100.0% | 97.3% |
| X_bm25_enriched | exploratory | seedless | 82.1% | 17.9% | 13.8% | 47.7% | 18.0% | 74.0% | 64.0% |
| X_hybrid_enriched_both | exploratory | 3 | 92.3% | 7.7% | 22.7% | 46.7% | 28.3% | 97.3% | 84.7% |
| X_combined_boost | exploratory | 3 | 100.0% | 0.0% | 53.0% | 24.0% | 92.3% | 100.0% | 97.0% |
| X_combined_no_cells | exploratory | 3 | 100.0% | 0.0% | 54.5% | 0.0% | 59.7% | 100.0% | 92.3% |
| X_combined_plain_a1 | exploratory | 3 | 100.0% | 0.0% | 52.3% | 0.0% | 55.3% | 100.0% | 93.3% |
| X_combined_tail | exploratory | 3 | 100.0% | 0.0% | 51.9% | 0.0% | 56.0% | 100.0% | 95.0% |
| X_a1_sliding256 | exploratory | 1 | 82.5% | 17.5% | 24.5% | 37.1% | 30.0% | 92.0% | 80.0% |

## A2. Real questions whose top 10 covers every company-year cell they ask about, by comparison type (mean over seeds, seed range)
| Experiment | Role | Seeds | Cross-company | Cross-year | Single scope |
|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.3 (0–1) of 10 | 14.0 (13–15) of 21 | 60.7 (59–62) of 69 |
| R_bm25 | reference | seedless | 1 of 10 | 11 of 21 | 44 of 69 |
| R_a1_oracle_company | reference | 3 | 0.7 (0–1) of 10 | 15.3 (15–16) of 21 | 65.3 (64–66) of 69 |
| R_a1_oracle_company_year | reference | 3 | 5.0 (4–6) of 10 | 18.0 (17–19) of 21 | 69 of 69 |
| E2_a1_company | primary | 3 | 0.7 (0–1) of 10 | 15.3 (15–16) of 21 | 65.3 (64–66) of 69 |
| E3_a1_year | primary | 3 | 4.0 (3–5) of 10 | 15.0 (14–16) of 21 | 65.7 (65–66) of 69 |
| E3b_a1_company_year | primary | 3 | 5.0 (4–6) of 10 | 18.0 (17–19) of 21 | 69 of 69 |
| E4_hybrid_rrf | primary | 3 | 0 of 10 | 11.3 (11–12) of 21 | 56.0 (55–58) of 69 |
| E5_hybrid_rrf_meta | primary | 3 | 4.3 (4–5) of 10 | 20 of 21 | 68.3 (68–69) of 69 |
| E6_a1_enriched | primary | 3 | 3.3 (2–5) of 10 | 17.3 (16–18) of 21 | 65.3 (65–66) of 69 |
| E7_combined | primary | 3 | 7.7 (7–8) of 10 | 18.7 (18–19) of 21 | 69 of 69 |
| E8_a1_tail_windows | primary | 3 | 0 of 10 | 13.7 (12–15) of 21 | 60.3 (59–61) of 69 |
| X_rrf_k10 | exploratory | 3 | 1 of 10 | 12.7 (12–13) of 21 | 57.3 (55–60) of 69 |
| X_rrf_k100 | exploratory | 3 | 0 of 10 | 11 of 21 | 56.0 (55–58) of 69 |
| X_rrf_dense2 | exploratory | 3 | 0 of 10 | 10 of 21 | 58.0 (55–61) of 69 |
| X_rrf_bm25x2 | exploratory | 3 | 0 of 10 | 12.3 (12–13) of 21 | 55.3 (54–57) of 69 |
| X_hybrid_company_year_filter | exploratory | 3 | 4.3 (4–5) of 10 | 15.7 (15–16) of 21 | 69 of 69 |
| X_a1_company_year_exact | exploratory | 3 | 8 of 10 | 20.7 (20–21) of 21 | 69 of 69 |
| X_a1_company_year_boost | exploratory | 3 | 7.7 (7–8) of 10 | 20.7 (20–21) of 21 | 69 of 69 |
| X_bm25_enriched | exploratory | seedless | 0 of 10 | 15 of 21 | 49 of 69 |
| X_hybrid_enriched_both | exploratory | 3 | 2 of 10 | 15.7 (15–16) of 21 | 67.0 (66–68) of 69 |
| X_combined_boost | exploratory | 3 | 8.7 (8–9) of 10 | 19.3 (19–20) of 21 | 69 of 69 |
| X_combined_no_cells | exploratory | 3 | 4.7 (4–5) of 10 | 18.7 (18–19) of 21 | 69 of 69 |
| X_combined_plain_a1 | exploratory | 3 | 7.3 (7–8) of 10 | 17.0 (16–18) of 21 | 69 of 69 |
| X_combined_tail | exploratory | 3 | 7 of 10 | 19 of 21 | 69 of 69 |
| X_a1_sliding256 | exploratory | 1 | 1 of 10 | 17 of 21 | 62 of 69 |

## S1. Scope decisions per question set: company filtered / year filtered / year fallback / cells
| Experiment | Validation | Development | Real questions |
|---|---|---|---|
| R_a1_oracle_company | 93 / 0 / 0 / 0 of 93 | 10 / 0 / 0 / 0 of 10 | 100 / 0 / 0 / 0 of 100 |
| R_a1_oracle_company_year | 93 / 93 / 0 / 0 of 93 | 10 / 10 / 0 / 0 of 10 | 100 / 100 / 0 / 0 of 100 |
| E2_a1_company | 93 / 0 / 0 / 0 of 93 | 10 / 0 / 0 / 0 of 10 | 99 / 0 / 0 / 0 of 100 |
| E3_a1_year | 0 / 89 / 0 / 0 of 93 | 0 / 10 / 0 / 0 of 10 | 0 / 100 / 0 / 0 of 100 |
| E3b_a1_company_year | 93 / 89 / 0 / 0 of 93 | 10 / 10 / 0 / 0 of 10 | 99 / 100 / 0 / 0 of 100 |
| E5_hybrid_rrf_meta | 93 / 0 / 0 / 0 of 93 | 10 / 0 / 0 / 0 of 10 | 99 / 0 / 0 / 0 of 100 |
| E7_combined | 93 / 89 / 0 / 0 of 93 | 10 / 10 / 0 / 2 of 10 | 99 / 100 / 0 / 31 of 100 |
| X_hybrid_company_year_filter | 93 / 89 / 0 / 0 of 93 | 10 / 10 / 0 / 0 of 10 | 99 / 100 / 0 / 0 of 100 |
| X_a1_company_year_exact | 93 / 86 / 3 / 0 of 93 | 10 / 10 / 0 / 0 of 10 | 99 / 100 / 0 / 0 of 100 |
| X_a1_company_year_boost | 93 / 0 / 0 / 0 of 93 | 10 / 0 / 0 / 0 of 10 | 99 / 0 / 0 / 0 of 100 |
| X_combined_boost | 93 / 0 / 0 / 0 of 93 | 10 / 0 / 0 / 0 of 10 | 99 / 0 / 0 / 10 of 100 |
| X_combined_no_cells | 93 / 89 / 0 / 0 of 93 | 10 / 10 / 0 / 0 of 10 | 99 / 100 / 0 / 0 of 100 |
| X_combined_plain_a1 | 93 / 89 / 0 / 0 of 93 | 10 / 10 / 0 / 2 of 10 | 99 / 100 / 0 / 31 of 100 |
| X_combined_tail | 93 / 89 / 0 / 0 of 93 | 10 / 10 / 0 / 2 of 10 | 99 / 100 / 0 / 31 of 100 |

## P1. Query understanding against each set's declared scope (parser qu-v1; validation: the positive passage's company and filing year)
| Set | Questions | Companies exact | Declared company missed | Extra company | Years exact | Declared years inside the year filter | No year parsed | Comparison type agrees |
|---|---|---|---|---|---|---|---|---|
| validation | 93 | 93/93 | 0 | 0 | 75/93 | 88/89 | 4 | — |
| development | 10 | 10/10 | 0 | 0 | 10/10 | 10/10 | 0 | 10/10 |
| real | 100 | 99/100 | 1 | 0 | 100/100 | 100/100 | 0 | 100/100 |

## W1. Fiscal-year window candidates on the 771 training and validation questions (the positive passage's filing year kept; never measured on the real questions)
| Window | Filing year kept | Mean share of the company's fiscal years allowed |
|---|---|---|
| exact year + 2 later filings | 684/699 (97.9%) | 36.8% |
| chosen: also 1 earlier filing for years read from dates or plain numbers | 693/699 (99.1%) | 43.2% |
| 1 earlier filing for every year | 694/699 (99.3%) | 48.5% |

## Q1. Baseline and primary experiments per seed
| Experiment | Seed | Validation MRR@10 | Validation Recall@10 | Validation nDCG@10 | Development MRR@10 | Development Recall@10 |
|---|---|---|---|---|---|---|
| E1_a1 | a1_seed42 | 0.669 | 0.901 | 0.717 | 0.558 | 0.327 |
| E1_a1 | a1_seed1337 | 0.668 | 0.874 | 0.705 | 0.583 | 0.327 |
| E1_a1 | a1_seed2026 | 0.658 | 0.906 | 0.712 | 0.667 | 0.257 |
| E2_a1_company | a1_seed42 | 0.732 | 0.920 | 0.773 | 0.592 | 0.344 |
| E2_a1_company | a1_seed1337 | 0.738 | 0.909 | 0.769 | 0.617 | 0.344 |
| E2_a1_company | a1_seed2026 | 0.717 | 0.947 | 0.768 | 0.717 | 0.307 |
| E3_a1_year | a1_seed42 | 0.673 | 0.874 | 0.700 | 0.618 | 0.377 |
| E3_a1_year | a1_seed1337 | 0.675 | 0.868 | 0.695 | 0.643 | 0.377 |
| E3_a1_year | a1_seed2026 | 0.673 | 0.890 | 0.705 | 0.677 | 0.361 |
| E3b_a1_company_year | a1_seed42 | 0.740 | 0.904 | 0.761 | 0.642 | 0.377 |
| E3b_a1_company_year | a1_seed1337 | 0.750 | 0.904 | 0.763 | 0.667 | 0.377 |
| E3b_a1_company_year | a1_seed2026 | 0.731 | 0.920 | 0.759 | 0.717 | 0.361 |
| E4_hybrid_rrf | a1_seed42 | 0.716 | 0.916 | 0.764 | 0.599 | 0.307 |
| E4_hybrid_rrf | a1_seed1337 | 0.718 | 0.913 | 0.763 | 0.597 | 0.282 |
| E4_hybrid_rrf | a1_seed2026 | 0.719 | 0.909 | 0.764 | 0.610 | 0.243 |
| E5_hybrid_rrf_meta | a1_seed42 | 0.854 | 0.955 | 0.854 | 0.667 | 0.377 |
| E5_hybrid_rrf_meta | a1_seed1337 | 0.839 | 0.963 | 0.847 | 0.625 | 0.369 |
| E5_hybrid_rrf_meta | a1_seed2026 | 0.834 | 0.949 | 0.840 | 0.633 | 0.377 |
| E6_a1_enriched | a1_seed42 | 0.753 | 0.908 | 0.773 | 0.651 | 0.404 |
| E6_a1_enriched | a1_seed1337 | 0.757 | 0.926 | 0.780 | 0.656 | 0.362 |
| E6_a1_enriched | a1_seed2026 | 0.737 | 0.909 | 0.761 | 0.597 | 0.381 |
| E7_combined | a1_seed42 | 0.817 | 0.923 | 0.826 | 0.637 | 0.405 |
| E7_combined | a1_seed1337 | 0.819 | 0.933 | 0.829 | 0.637 | 0.398 |
| E7_combined | a1_seed2026 | 0.807 | 0.918 | 0.820 | 0.753 | 0.390 |
| E8_a1_tail_windows | a1_seed42 | 0.669 | 0.879 | 0.704 | 0.583 | 0.395 |
| E8_a1_tail_windows | a1_seed1337 | 0.657 | 0.868 | 0.696 | 0.550 | 0.346 |
| E8_a1_tail_windows | a1_seed2026 | 0.655 | 0.864 | 0.691 | 0.670 | 0.365 |

## V2. Validation MRR@10 by where the evidence starts in its positive passage (encoder tokens; every validation evidence span ends inside the 510-token window)
| Experiment | Role | Seeds | Evidence starts at token 0-127 | Evidence starts at token 128-255 | Evidence starts at token 256+ |
|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.694 (0.659–0.713) (n 21) | 0.795 (0.787–0.800) (n 24) | 0.578 (0.577–0.580) (n 47) |
| R_bm25 | reference | seedless | 0.530 (n 21) | 0.634 (n 24) | 0.674 (n 47) |
| E4_hybrid_rrf | primary | 3 | 0.665 (0.647–0.675) (n 21) | 0.769 (0.764–0.771) (n 24) | 0.710 (0.706–0.714) (n 47) |
| E6_a1_enriched | primary | 3 | 0.742 (0.737–0.746) (n 21) | 0.903 (0.878–0.927) (n 24) | 0.668 (0.656–0.683) (n 47) |
| E8_a1_tail_windows | primary | 3 | 0.605 (0.591–0.620) (n 21) | 0.701 (0.696–0.710) (n 24) | 0.671 (0.665–0.674) (n 47) |
| X_a1_sliding256 | exploratory | 1 | 0.584 (n 21) | 0.746 (n 24) | 0.773 (n 47) |
| E7_combined | primary | 3 | 0.739 (0.715–0.754) (n 21) | 0.919 (0.910–0.938) (n 24) | 0.790 (0.776–0.797) (n 47) |

## B1. Baseline preservation: v2 runs against the frozen v1 analysis (validation)
| Check | Result |
|---|---|
| E1_a1 [a1_seed42] vs v1 a1_seed42/global | 93/93 first-relevant ranks identical |
| E1_a1 [a1_seed1337] vs v1 a1_seed1337/global | 93/93 first-relevant ranks identical |
| E1_a1 [a1_seed2026] vs v1 a1_seed2026/global | 93/93 first-relevant ranks identical |
| R_bm25 [-] vs v1 bm25/global | 93/93 first-relevant ranks identical |
| R_a1_oracle_company [a1_seed42] vs v1 a1_seed42/company | 93/93 first-relevant ranks identical |
| R_a1_oracle_company [a1_seed1337] vs v1 a1_seed1337/company | 93/93 first-relevant ranks identical |
| R_a1_oracle_company [a1_seed2026] vs v1 a1_seed2026/company | 93/93 first-relevant ranks identical |
