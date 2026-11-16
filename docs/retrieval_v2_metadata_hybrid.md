# FinRetriever v2: metadata-aware hybrid retrieval

> **Status: experiments on existing evaluation inputs, 2026-11-15. Not a benchmark.**
> - **Validation:** 93 synthetic questions that selected the A1 checkpoints; v2's fiscal-year window was also
>   chosen using their filing years.
> - **Development:** 10 questions with AI-written labels.
> - **Real questions:** 100 human-written questions, scored only for metadata alignment and parser accuracy.
>
> None of these is an independent human relevance evaluation, and no improvement on real questions is claimed.
> The A1 checkpoints, the frozen datasets, the v1 analysis and the gold-set closure are unchanged. Nothing was
> retrained.
>
> **Update, 2026-11-16:** [v2.1](#v21-fiscal-year-roles-parser-qu-v2), at the end of this document, adds
> fiscal-year roles to the parser and sets the release default. Sections 1–7 are the v2 record as run on
> 2026-11-15, unchanged.

## 1. Findings

| Component | What happened | Evidence (mean over the three A1 seeds) |
|---|---|---|
| **Query understanding** | Reads the declared scope of the real questions almost perfectly | 100 human-written questions, measured once after the parser was frozen: companies exact 99/100 (the miss, "Sam's Club", fell back to no company constraint), fiscal years exact 100/100, comparison type 100/100 (P1) |
| **Company filter** (E2) | A consistent gain at no recall cost | Validation MRR@10 0.665 → 0.729 (+0.064, p ≤ 0.0001 in every seed; 17–18 questions better, none worse); Recall@100 unchanged; identical to the oracle-company reference (V1) |
| **Fiscal-year filter** (E3, E3b) | Helps ranking but costs recall | Alone: +0.009 (not significant). With the company filter: +0.076 (p ≤ 0.0001), but validation Recall@100 falls from 0.999 to 0.951. The filter cuts relevant passages in years outside the window for 8 questions, and all of them for one, whose question names target years ("by 2025 compared to 2016 levels") rather than its fiscal year |
| **BM25 + A1 fusion** (E4) | Not a clear win on its own | Validation +0.053 (p ≤ 0.22; not significant in any seed); development nDCG@10 0.317 → 0.264 |
| **Fusion + metadata-aware ranking** (E5: company filter, fiscal-year boost) | The largest validation gain, keeping recall | Validation MRR@10 0.842 (+0.177, p ≤ 0.0001 in every seed), Recall@10 0.956, Recall@100 0.994; real questions: top 1 in the exact company and year for 92.3% (E1: 21.0%) |
| **Enriched A1** (E6: company, year and section header) | Restores company grounding without any filter | Validation 0.749 (+0.084, p ≤ 0.0062). On the real questions, without a company filter, the share of the top 10 from other companies falls from 17.3% to 6.4%, and the exact-year share rises from 21.6% to 31.3% |
| **Combined, pre-registered** (E7: fusion with enriched A1, company and fiscal-year filters, cells) | Strong on validation; the best development point estimates; the best multi-scope coverage | Validation 0.814 (+0.149, p ≤ 0.0005), but Recall@100 0.951 (the year filter). Development MRR@10 0.676 vs 0.603, Recall@10 0.398 vs 0.304, Recall@100 0.896 vs 0.775 (none significant on 10 questions; lower bounds). Real questions: every company-year cell covered for 95.3% (E1: 75.0%); cross-company 7–8 of 10 (E1: 0–1) |
| **Long-passage windows** (E8, X_a1_sliding256) | No net gain from tail windows. Shorter windows (exploratory) find late evidence more often but early evidence less often | Tail windows (E8): validation 0.660 against 0.665 (not significant). For evidence starting after token 256, MRR@10 rises from 0.578 to 0.671; for earlier evidence it falls. 256-token sliding windows (seed 42 only): validation 0.720 (+0.052, p 0.20); against seed 42's E1, late evidence 0.773 vs 0.577, early 0.584 vs 0.709 (V2). Development MRR@10 0.781 (+0.223, p 0.031: one seed, 10 questions, uncorrected) |

**Strongest and weakest:**
- **On validation:** the strongest is E5, fusion with metadata-aware ranking (0.842). Next come the exploratory
  X_combined_boost (0.827) and the pre-registered E7 (0.814). The weakest A1-based variants are the tail windows
  (0.660) and the year filter alone (0.674), neither different from the baseline. BM25 alone is the weakest
  overall (0.635).
- **On development:** no primary experiment's MRR@10 difference is significant on 10 questions (smallest p
  0.50). The smallest p on any development metric is 0.035 (E6, seed 2026, Recall@10). The exploratory 256-token
  windows reach p 0.031 for MRR@10, on one seed. Neither is evidence, given the number of comparisons.
- **On the real questions** (alignment, not relevance): E7 is the most complete. Every top 10 comes from the asked
  companies, 55.0% of it from the asked fiscal years, and 95.3% of questions have every company-year cell covered.

**What the evidence supports:** company scoping and passage enrichment, which are consistent in every seed and
also supported by label-free alignment. For fiscal year, a boost rather than a hard filter. What it does not
support: any claim on real-question relevance.

## 2. What v2 adds

The [v1 analysis](retrieval_analysis_v1.md) found six weaknesses:
1. fine-tuning improved fiscal-year alignment;
2. but it increased cross-company retrieval;
3. fiscal-year grounding stayed weak everywhere;
4. BM25 and dense retrieval are complementary;
5. 91% of passages are truncated at 512 tokens;
6. evidence late in a long passage is ranked worse by dense retrievers.

v2 addresses each with one component, built on the existing retrievers.

| v1 weakness | v2 component | Module |
|---|---|---|
| Cross-company retrieval (2); weak fiscal-year grounding (1, 3) | **Query understanding:** companies (ticker, registered name, configured aliases), fiscal years (`FY2024`, `fiscal 2024`, period-end dates, ranges, "versus"), measures, comparison type; explicit fallbacks | `retrieval/query_understanding.py` |
| (2), (3) | **Metadata scope:** a company filter; a fiscal-year filter or boost that keeps later filings repeating the year; a fallback when too few passages remain; company-year cells for multi-scope questions | `retrieval/metadata_scope.py` |
| BM25 and A1 complementary (4) | **Reciprocal Rank Fusion** of BM25 and A1 (k = 60, equal weights) | `retrieval/fusion.py`, `retrieval/hybrid.py` |
| Company identity missing from 59% of passages (2) | **Passage enrichment:** a header (company, ticker, form, fiscal year, section) prepended at encoding time only; the corpus is unchanged | `retrieval/enrichment.py` |
| Truncation (5); late evidence (6) | **Long-passage windows:** tail windows encode the text past the 512-token limit, and sliding 256-token windows cover the whole passage. Each passage is scored by its best window (MaxP), and results map back to the source passage | `retrieval/windows.py`, `retrieval/hybrid.MaxPScorer` |

The design is in [architecture.md](architecture.md) §4.2. The existing retrievers gained two additive features:
- `scores()` methods;
- a `passage_view` setting (default `text`), whose embedding cache key for existing embeddings is unchanged. A
  regression test checks this, and the existing caches were reused.

The same pipeline powers `scripts/12_search_demo.py`.

## 3. Protocol

- **Experiments** (`configs/experiments/v2_metadata_hybrid.yaml`, one entry each, with a role):

  | Role | Experiments | What they do |
  |---|---|---|
  | Baseline | E1 | The existing A1, unchanged |
  | Primary | E2–E8 | One change at a time, then E7, the pre-registered combination |
  | Reference | BM25 alone; oracle scope | The oracle runs use each set's annotated companies and years instead of the parser. They bound what a perfect parser could give and are never a system |
  | Exploratory | X_* | Sensitivity settings, reported in full and never selected |

- **Pre-registration.** The primary settings were fixed in the config before any v2 result was computed:
  - RRF k = 60 with equal weights;
  - the year window and `min_eligible` 20;
  - E7's composition, including the year *filter*.
- **How the year window was chosen.** It was chosen on the 678 training and 93 validation questions, never the
  100 real ones (table W1).
- **Parser freeze.** The parser (`qu-v1`) was frozen before its single measurement on the 100 real questions.
- **Seeds:** every A1-based experiment runs on all three seeds (42, 1337, 2026), except the exploratory 256-token
  windows (seed 42 only, for compute). BM25-only runs have no seed.
- **Scoring.**
  - *Validation* is scored as checkpoint selection scored it. Relevant means the positive plus its exact
    equivalents; uncertain equivalents are never ranked.
  - *Development* is scored with the harness's metric code. Unjudged passages count as non-relevant, and judged@10
    and judged-only MRR are reported.
  - *Real questions* get the v1 alignment diagnostic on the top 10, plus the parser's accuracy against the
    authors' declared scope.
  - Each experiment is compared with E1 on the same seed by a paired bootstrap and a sign-flip test. Tables show
    the mean over seeds, the seed range and the largest p.
- **Multiple comparisons.** With 26 experiments and several metrics, some small p-values are expected by chance.
  Only consistent, all-seed effects are interpreted.

## 4. Results

All tables are generated by `scripts/11_v2_experiments.py` from the run outputs and copied here unchanged.

### 4.1 Validation (93 synthetic questions)

**Table V1.** Validation (93 synthetic questions; they selected A1). Mean over seeds (seed range)

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


### 4.2 Development (10 questions, AI-written labels)

**Table D1.** Development (10 questions, AI-written provisional labels; unjudged = non-relevant)

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


### 4.3 Real questions: alignment and query understanding

**Table A1.** Real-question metadata alignment (100 human-written questions, top 10; mean over seeds). Not relevance; scoped systems align by construction when the parser is right

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


**Table A2.** Real questions whose top 10 covers every company-year cell they ask about, by comparison type (mean over seeds, seed range)

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


**Table P1.** Query understanding against each set's declared scope (parser qu-v1; validation: the positive passage's company and filing year)

| Set | Questions | Companies exact | Declared company missed | Extra company | Years exact | Declared years inside the year filter | No year parsed | Comparison type agrees |
|---|---|---|---|---|---|---|---|---|
| validation | 93 | 93/93 | 0 | 0 | 75/93 | 88/89 | 4 | — |
| development | 10 | 10/10 | 0 | 0 | 10/10 | 10/10 | 0 | 10/10 |
| real | 100 | 99/100 | 1 | 0 | 100/100 | 100/100 | 0 | 100/100 |


**Table S1.** Scope decisions per question set: company filtered / year filtered / year fallback / cells

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


**Table W1.** Fiscal-year window candidates on the 771 training and validation questions (the positive passage's filing year kept; never measured on the real questions)

| Window | Filing year kept | Mean share of the company's fiscal years allowed |
|---|---|---|
| exact year + 2 later filings | 684/699 (97.9%) | 36.8% |
| chosen: also 1 earlier filing for years read from dates or plain numbers | 693/699 (99.1%) | 43.2% |
| 1 earlier filing for every year | 694/699 (99.3%) | 48.5% |


### 4.4 Long passages and evidence position

**Table V2.** Validation MRR@10 by where the evidence starts in its positive passage (encoder tokens; every validation evidence span ends inside the 510-token window)

| Experiment | Role | Seeds | Evidence starts at token 0-127 | Evidence starts at token 128-255 | Evidence starts at token 256+ |
|---|---|---|---|---|---|
| E1_a1 | baseline | 3 | 0.694 (0.659–0.713) (n 21) | 0.795 (0.787–0.800) (n 24) | 0.578 (0.577–0.580) (n 47) |
| R_bm25 | reference | seedless | 0.530 (n 21) | 0.634 (n 24) | 0.674 (n 47) |
| E4_hybrid_rrf | primary | 3 | 0.665 (0.647–0.675) (n 21) | 0.769 (0.764–0.771) (n 24) | 0.710 (0.706–0.714) (n 47) |
| E6_a1_enriched | primary | 3 | 0.742 (0.737–0.746) (n 21) | 0.903 (0.878–0.927) (n 24) | 0.668 (0.656–0.683) (n 47) |
| E8_a1_tail_windows | primary | 3 | 0.605 (0.591–0.620) (n 21) | 0.701 (0.696–0.710) (n 24) | 0.671 (0.665–0.674) (n 47) |
| X_a1_sliding256 | exploratory | 1 | 0.584 (n 21) | 0.746 (n 24) | 0.773 (n 47) |
| E7_combined | primary | 3 | 0.739 (0.715–0.754) (n 21) | 0.919 (0.910–0.938) (n 24) | 0.790 (0.776–0.797) (n 47) |


### 4.5 Per seed

**Table Q1.** Baseline and primary experiments per seed

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


## 5. Interpretation

- **Company scope is the safest gain.** On the evidence available:
  - the parser named the right companies for every validation and development question and 99 of 100 real ones;
  - the filter never made a validation question worse;
  - it costs no recall.

  It also removes A1's main new failure from v1, the drift to other companies.
- **Fiscal year needs care.** A hard filter raises MRR but cuts recall. Two kinds of question break it:
  - a question can name years that are not its fiscal year (targets, maturities, historical references);
  - the same fact can sit in filings outside the window.

  A boost (E5, X_a1_company_year_boost, X_combined_boost) keeps Recall@100 near 0.99 while still putting the asked
  year first. The pre-registered E7 used the filter: its result stands as registered, and the boost variants are
  exploratory evidence for the next version.
- **Fusion helps only together with metadata.** BM25 + A1 alone is mixed: positive on validation, negative on
  development nDCG, where BM25 is weak. Its value appears when metadata decides between the two rankings (E5).
- **Enrichment works without retraining**, and its company-grounding effect is label-free: the real-question
  cross-company share falls from 17.3% to 6.4% (6.4% is close to BGE zero-shot's 5.4% in v1). A1 never saw the
  headers during training, so retraining with them is the natural next step. BM25 with headers also improves
  sharply on validation (X_bm25_enriched: 0.735 against 0.635), because the header supplies the company and year
  words.
- **Long passages:** the v1 dilution hypothesis holds. When a passage is also scored by shorter windows, evidence that starts late in it is found far more often: with 256-token windows, 0.773 against 0.577 for the same seed. Early evidence suffers, though (0.584 against 0.709), because short windows of other passages now outscore the right passage. Tail windows only add text past token 510, which no validation evidence reaches, so on validation they add competitors and nothing else; their development Recall@10 rises (0.369 against 0.304), but not significantly. A plain maximum over windows trades one error for another. The sliding-window development result (MRR@10 0.781) is the highest of any run, but it rests on one seed and 10 AI-labelled questions, and it is one of many comparisons
- **Multi-scope questions:** cells give each company-year its own slots. Cross-company questions with every cell
  covered rise from 0–1 of 10 to 7–8 of 10. This is the clearest structural improvement, though measured by
  metadata only.

### Recommendations for the next version

1. Make the company filter the default (with the parser's fallback). *Done in v2.1: the release default.*
2. Use a fiscal-year *boost* rather than a filter. Teach the parser to tell fiscal-year references from target,
   maturity and historical years. *Done in v2.1 (parser qu-v2); the boost is the default.*
3. Retrain A1 with enriched passages (header_v1), with hard negatives from the same company's other years and from
   other companies' same-topic tables.
4. Fit passages to the encoder instead of max-pooling windows at inference: re-chunk to about 256–384 BGE tokens with overlap and retrain on those chunks (§4.4: plain MaxP helps late evidence and hurts early evidence). Confirm the exploratory sliding-window result on more seeds and on human labels first
5. Evaluate the chosen pipeline on human relevance labels before claiming any improvement. A complete, small
   human-labelled set (for example 30–50 questions with two annotators) is the gating item.

## 6. Limitations

- **Not independent evidence.** The validation questions selected the A1 checkpoints and informed the year
  window. The development set has 10 questions with AI-written labels. The real questions have no relevance
  labels.
- **The development pools predate v2.** v2 systems retrieve unjudged passages (judged@10 between 0.71 and 0.99
  for the primary experiments, per seed), which count as non-relevant, so their development scores are lower bounds.
- **The alignment diagnostic is partly circular for scoped systems.** A company or year filter aligns the top 10
  by construction when the parser is right. The informative numbers there are the parser's accuracy, the
  fallbacks, the cell coverage, and the label-free effect of enrichment without filters.
- **Many comparisons.** 26 experiments over 2 sets and several metrics; there was no multiple-comparison
  correction, only consistency across seeds.
- **Enrichment and windows are applied at inference only.** A1 was trained on plain passages.
- **The parser is rule-based** and English-only. It covers the seven corpus companies' names, tickers and
  configured aliases, not subsidiaries or segments (for example "Sam's Club").

## 7. Reproducibility and integrity

```bash
PY=python   # results used Python 3.11.5, torch 2.10.0, transformers 5.3.0, sentence-transformers 5.2.3, CPU
$PY scripts/11_v2_experiments.py --prepare-only   # builds every embedding cache (enriched passages, windows)
$PY scripts/11_v2_experiments.py                  # runs every experiment; writes results/experiments/v2_metadata_hybrid/
$PY scripts/11_v2_experiments.py --tables-only    # re-renders tables.md from the written outputs
$PY scripts/12_search_demo.py --question "..."    # the same pipeline, interactively
```

- **Outputs:** `manifest.json` records:
  - each experiment's settings and role;
  - the config's SHA-256;
  - every input's hash;
  - the checkpoint digests and selection records;
  - each component's embedding-cache key and embedding hash;
  - the source hashes of the retrieval code;
  - the library versions;
  - the git commit.

  The other outputs are `summary.json` (aggregates, intervals, paired comparisons), `parser_accuracy.json`,
  `year_window_choice.json`, `runs/<experiment>/<seed>/{validation,development,real}.jsonl` (per-question ranks,
  top 10 and the scope trace), and `tables.md`.
- **Baseline preserved:** E1 reproduces the frozen v1 analysis rankings exactly (93 of 93 first-relevant ranks
  for every seed), as do the oracle-company and BM25 references (table B1).

**Table B1.** Baseline preservation: v2 runs against the frozen v1 analysis (validation)

| Check | Result |
|---|---|
| E1_a1 [a1_seed42] vs v1 a1_seed42/global | 93/93 first-relevant ranks identical |
| E1_a1 [a1_seed1337] vs v1 a1_seed1337/global | 93/93 first-relevant ranks identical |
| E1_a1 [a1_seed2026] vs v1 a1_seed2026/global | 93/93 first-relevant ranks identical |
| R_bm25 [-] vs v1 bm25/global | 93/93 first-relevant ranks identical |
| R_a1_oracle_company [a1_seed42] vs v1 a1_seed42/company | 93/93 first-relevant ranks identical |
| R_a1_oracle_company [a1_seed1337] vs v1 a1_seed1337/company | 93/93 first-relevant ranks identical |
| R_a1_oracle_company [a1_seed2026] vs v1 a1_seed2026/company | 93/93 first-relevant ranks identical |


- **Deterministic:** the whole suite (26 experiments, 72 runs over three seeds) was run twice from the same code and embedding caches. All output files are byte-identical, including the manifests.
- **Tests:**
  - `tests/unit/test_query_understanding.py`, `test_fusion.py`, `test_metadata_scope.py`,
    `test_enrichment_and_windows.py` (including the cache-key regression) and `test_v2_experiment_config.py`;
  - `tests/integration/test_hybrid_retrieval.py`: real BM25 and dense retrievers with a fake encoder, the RRF
    order checked against an independent computation, and determinism.

## v2.1: fiscal-year roles (parser qu-v2)

*2026-11-16. Config `configs/experiments/v2_1_year_roles.yaml`, outputs `results/experiments/v2_1_year_roles/`
(its `tables.md` has every table; table R1 compares each run with the frozen v2 run of the same experiment and
seed). The v2 results above are unchanged.*

### What changed

v2 found one failure of the hard year filter: a question naming a target year and its baseline ("by 2025 compared
to 2016 levels") was read as asking about fiscal 2016 and 2025. The filter then removed every relevant passage.
Parser **qu-v2** gives every year in a question a role, read from the words next to it:

| Role | Examples | Constrains retrieval? |
|---|---|---|
| fiscal | "in fiscal 2023", "as of December 31, 2022", both years of "2023 versus 2022" | Yes |
| target | "reduce emissions by 2030", "2030 carbon neutrality goals", "outlook for 2026", "committed to deliver from 2026 through 2028" | No |
| historical | "since 2015", "compared to 2016 levels", the baseline of a target year | No |
| maturity | "notes due 2030", "matures in 2027", "the 2025 Notes", "expire in 2031" | No |
| incidental | "Tax Cuts and Jobs Act of 2017", "the 2019 Stock Incentive Plan", "the 2018 CEO Performance Award" | No |

- **Only fiscal years constrain retrieval.** A year with no cue keeps the qu-v1 reading (fiscal), so roles can
  only remove a year from the constraint, never add one.
- **A question whose years are all non-reporting gets no year constraint,** and the fallback is recorded.
- **Years 2040–2100 are read as targets.** No filing reports them.
- **qu-v1 is kept unchanged.** The v2 config still uses it: re-running v2 with the new code reproduces all 220
  result files byte for byte, and only the manifest's source-file hashes differ.

### How it was developed and tested

- **Development data.** The rules were written after qu-v1's failure on validation_000049 and checked against
  every changed parse on the 771 training and validation questions. Twelve of the 771 change:
  - 10 drop a year that was not the filing's fiscal year, for example "maturing in 2023" in a fiscal 2020
    question, "by 2040 compared to 2018", "the 2018 CEO Performance Award" in a fiscal 2022 question, and
    "committed to deliver … from 2026 through 2028" in a fiscal 2025 filing;
  - 2 drop a year that matched the filing only by coincidence: Tesla's "2025 CEO Performance Award" is an award's
    name, and is described in the FY2025 10-K.
- **Regression tests.** 29 realistic analyst questions written for the tests, covering every role, plus tests
  that roles only ever remove years and that qu-v1 is unchanged.
- **Not a blind test.** The rules were then frozen and run once on every set. qu-v1's errors had been seen first,
  and the real questions' qu-v1 parse had been inspected in v2, so qu-v2's accuracy is not a blind test.

### Results (means over the three A1 seeds; v2 = parser qu-v1)

| Experiment | Validation MRR@10, v2 → v2.1 | Validation Recall@100, v2 → v2.1 | Questions whose top 10 changed (validation / development / real) |
|---|---:|---:|---:|
| E1 · A1 dense (baseline, no scope) | 0.665 → 0.665 | 0.999 → 0.999 | 0 / 0 / 0 |
| E3 · fiscal-year filter | 0.674 → 0.679 | 0.951 → 0.962 | 2 / 0 / 0 |
| E3b · company and fiscal-year filters | 0.740 → 0.746 | 0.951 → 0.962 | 2 / 0 / 0 |
| **E5 · RRF + company filter + fiscal-year boost (release default)** | **0.842 → 0.851** | 0.994 → 0.994 | 2 / 0 / 0 |
| E7 · combined (pre-registered) | 0.814 → 0.825 | 0.951 → 0.962 | 2 / 0 / 0 |
| X_combined_boost (exploratory) | 0.827 → 0.836 | 0.992 → 0.992 | 2 / 0 / 0 |

- **Exactly one validation question changes its metrics.** It is validation_000049, the failure that motivated
  the change. Under the hard filter (E3, E3b, E7) its relevant passage was not retrieved at all; now it ranks 1st
  or 2nd. Under the boost (E5) it moves from rank 5–6 to rank 1.
- **The second changed top 10** (the "2025 CEO Performance Award" question) keeps its first relevant rank.
- **No significance claim.** With one changed question the paired tests cannot be significant (p = 1.0 in every
  seed): this repairs a known failure and is not evidence of a general gain.
- **Development and real questions: identical results.** No development or real question names a year in a
  non-fiscal role (table P2 of the v2.1 tables), so qu-v2 parses all 110 exactly as qu-v1 did. On the real
  questions: companies exact 99/100, years exact 100/100, comparison type 100/100.
- **The hard filter still drops evidence.** Seven validation questions keep only part of their relevant set under
  E3b and E7 (Recall@100 0.962). In every seed, each shortfall is exactly the relevant passages that sit in filings
  outside the allowed fiscal years: the same text, repeated in another year's 10-K. Every one of these questions
  still has a relevant passage in its top 100.
- **The year window.** It keeps the positive's filing year for 689 of the 694 training and validation questions
  that name a reporting year (99.3%; qu-v1: 693 of 699, 99.1%).

### Release default

`E5_hybrid_rrf_meta` with parser qu-v2: BM25 and A1 fused by RRF (k = 60), a company filter, and a fiscal-year
**boost**. It was chosen from the v2 evidence before this run; v2.1 did not change the choice.

| Measure | Value |
|---|---|
| Validation MRR@10 | 0.851, +0.186 over E1 (p ≤ 0.0001 in every seed) |
| Validation Recall@100 | 0.994 |
| Development MRR@10 | 0.642 (not significant) |
| Real questions with the top 1 in the asked company and year | 92.3% (E1: 21.0%) |

- **Why not the hard filter:** its validation MRR is lower in v2.1 too (E3b 0.746, E7 0.825), and it still drops
  evidence.
- **Not chosen on these numbers:** X_combined_boost (validation 0.836, development MRR@10 0.742) is exploratory,
  and is reported, not selected.
- **Where it is used:** the demo ([demo.md](demo.md)), which also shows each hit's BM25 and dense rank before
  fusion.

### Reproduce

```bash
python scripts/11_v2_experiments.py --config configs/experiments/v2_1_year_roles.yaml \
    --output-dir results/experiments/v2_1_year_roles
```

The run refuses to start unless:
- the reference run's manifest matches its recorded SHA-256;
- every experiment's settings are identical to the same experiment's in the v2 run.

E1 reproduces the frozen v1 rankings (93/93 first-relevant ranks in every seed). Two runs gave byte-identical
outputs (59 files, manifests included).
