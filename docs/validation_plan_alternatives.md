# Validation plan alternatives to `val_pilot_v2`: fiscal-year and filing-unit concentration

Status: 2026-11-03.
- **For approval, nothing generated.** Five plans of `bge_base_ft_v3` are
  compared offline.
  - No Groq request was sent and no model was trained.
  - The corpus, evaluation data, pins, candidates and validation sets are
    unchanged, and so is `val_pilot_v2`.
- **Plans:** A is `val_pilot_v2`, unchanged. B–E are new, written once:
  `val_pilot_v2b`, `val_pilot_v2c`, `val_pilot_v2d` and `val_pilot_v2e`.

## 1. Summary

**Terms:**
- **Filing:** one company and fiscal year. The corpus holds 49 filings,
  every one a 10-K, so "per company/fiscal-year" means "per filing".
- **Linked unit (split unit):** overlap groups joined by evidence reuse.
  - In these plans a multi-passage unit is almost always one note or table
    repeated across fiscal years. JPMorgan's 8-passage unit, for example,
    is one statements note over six years.
  - Its passages are in different filings, so a cap per filing does not
    limit a unit, and a cap per unit does not limit a filing.

**Findings:**
- **A cap of 2 per filing (B, D) halves the plan.**
  - At most 2 per filing over 48 filings is a ceiling of 96 passages, and
    the plans reach 89.
  - That is about 52 usable questions, so about 68 with the 16 carried.
  - **No setting reaches 120 under this cap.** Lowering the isolation
    threshold adds 3 passages, and moving training data to validation does
    not help: the ceiling is set by filings, not by the pool.
- **A cap of 3 per unit (C) is the cheapest gain in independence.** It
  costs 3 passages and about 4 questions, and gives the highest effective
  size of the plans with at most 185 requests.
- **Clustering by filing is measured near zero; clustering by unit is not
  measured.**
  - So a filing cap buys representativeness more than independence.
  - A unit cap buys independence if questions of one unit are correlated,
    which their structure suggests.
- **Every plan's yield estimate is probably optimistic.**
  - At isolation scores of 0.5 or more, 11 of 16 `val_pilot_v1` questions
    passed the evidence rule (69%), against 87% predicted.
  - On that band-calibrated estimate, A gives about 92 new questions (108
    with the 16), not 106 (122).
- **The plans' early batches are already balanced across years.** In every
  plan the first 100 passages hold 16–17 from FY2019 (16–17%). The excess
  comes from the last picks, when other years run out.

**Recommendation: E (`val_pilot_v2e`)** (§8):
- **Settings:** at most 8 per filing, at most 3 per unit, isolation share
  at least 0.3, 216 passages.
- **Approval:** in batches of 50, re-estimating the yield after each and
  stopping once about 120 questions are expected.
- **Against A:**
  - the largest unit falls from 8 to 3 and the largest filing from 13 to 8;
  - FY2019 falls from 24% to 18.5%;
  - the effective size rises by about 11–13%;
  - the expected total is about 120 on both yield estimates.
- **Costs:**
  - 31 more requests (216, one more batch);
  - JPMorgan reaches the 25% company cap;
  - 49 passages score only 0.3–0.5;
  - the pool is used to its limit under these caps.

## 2. How the alternatives were made

**New plan options** (`sampling_plan`):
- **`max_per_company_year`:** at most this many passages per company and
  fiscal year.
- **`max_per_unit`:** at most this many passages per split unit. It is used
  with `one_per="group"`.
- **Recording:** both are recorded in the plan's constraints only when
  used, so `pilot_v3`, `val_pilot_v1` and `val_pilot_v2` rebuild
  byte-identically (checked).

**How capped plans keep their quotas:**
- **Quotas:** a company's quota counts the passages it can take under the
  caps, a maximum flow over its units, overlap groups and years.
- **Shared units:** a unit drawn on by two companies shares one cap, so it
  counts for neither.
- **Picks:** each pick is the best-ranked one that still leaves the
  company's quota reachable. Without this check B failed: both of Tesla's
  2023 groups and one of its two 2024 groups also reach other years, and
  the greedy order spent one of them on another year.

**Inputs:** the same as A's.
- Settings: `balanced_years`, at most 25% per company, one passage per
  overlap group, seed 20261029, the same 118 excluded overlap groups (55
  units) and the same three exclusion sources.
- Only the caps and, for E, the isolation threshold differ.

**Sizes are never shortened silently.**
- A plan the caps cannot fill is refused, with the largest size they allow
  (for B, "only 89 passages fit 185 … at most 2 per company and fiscal
  year").
- B, C and D were each requested at 185, refused, and written at the
  largest size accepted.

## 3. The comparison

| | A `val_pilot_v2` | B `…v2b` | C `…v2c` | D `…v2d` | E `…v2e` |
|---|---:|---:|---:|---:|---:|
| Per filing / per unit / min isolated share | – / – / 0.5 | 2 / – / 0.5 | – / 3 / 0.5 | 2 / 3 / 0.5 | 8 / 3 / 0.3 |
| Passages (185 requested for B–D) | 185 | 89 (185 refused) | 182 (185 refused) | 89 (185 refused) | 216 |
| Requests; batches of ≤ 50 | 185; 4 | 89; 2 | 182; 4 | 89; 2 | 216; 5 |
| Time at 2.3/min (worst case 1/min) | 1.3 h (3.1 h) | 0.6 h (1.5 h) | 1.3 h (3.0 h) | 0.6 h (1.5 h) | 1.6 h (3.6 h) |
| Tokens (about 1,344 per request) | 249k | 120k | 245k | 120k | 290k |
| Generator-accepted (× 0.86) | 159 | 77 | 157 | 77 | 186 |
| After leakage rejection: score / band | 126 / 109 | 62 / 53 | 121 / 108 | 61 / 53 | 128 / 123 |
| Usable new questions: score / band | 106 / 92 | 52 / 44 | 102 / 91 | 52 / 44 | 108 / 103 |
| With the 16 carried: score / band | 122 / 108 | 68 / 60 | 118 / 107 | 68 / 60 | 124 / 119 |
| Against 120: score / band | +2 / −12 | −52 / −60 | −2 / −13 | −52 / −60 | +4 / −1 |
| Distinct linked units | 138 | 70 | 148 | 73 | 171 |
| Units with 2+ passages (passages in them) | 19 (66) | 11 (30) | 22 (56) | 11 (27) | 31 (76) |
| Largest unit | 8 | 5 | 3 | 3 | 3 |
| Distinct filings; largest | 48; 13 | 48; 2 | 48; 13 | 48; 2 | 48; 8 |
| FY2019 passages (share) | 45 (24%) | 12 (13%) | 43 (24%) | 12 (13%) | 40 (18.5%) |
| Narrative / table | 112 / 73 | 49 / 40 | 119 / 63 | 50 / 39 | 128 / 88 |
| Effective size: low / central / high (§6) | 113 / 70 / 51 | 66 / 51 / 41 | 114 / 75 / 56 | 66 / 52 / 43 | 121 / 78 / 57 |
| P(better of two checkpoints 0.02 apart is chosen), central; SD 0.11 / 0.15 | 0.94 / 0.87 | 0.90 / 0.83 | 0.94 / 0.88 | 0.91 / 0.83 | 0.95 / 0.88 |

**How to read the yield rows:**
- **"Score":** the planner's estimate. Each passage's isolated share ×
  0.86 (`val_pilot_v1`'s generator acceptance, 43/50), and × 0.84 for
  usable (its review acceptance, 16/19).
- **"Band":** the same, with the share replaced by the evidence-rule pass
  rate `val_pilot_v1` observed in the passage's score band (§7).
- **Both are estimates.**
- **"Leakage rejection":** `val_pilot_v1`'s evidence rules (training,
  blocked and evaluation evidence). Its question-overlap checks removed
  none.

### Company distribution: passages (expected usable questions, score / band)

| Company | A | B | C | D | E | Carried |
|---|---|---|---|---|---|---:|
| AAPL | 13 (7.9 / 6.5) | 10 (6.0 / 5.0) | 13 (7.9 / 6.5) | 10 (6.0 / 5.0) | 16 (8.7 / 7.7) | 3 |
| AMZN | 16 (9.7 / 8.0) | 11 (7.1 / 5.5) | 14 (8.3 / 7.0) | 11 (7.0 / 5.5) | 15 (8.5 / 7.4) | 3 |
| JPM | 37 (22.0 / 18.4) | 14 (7.9 / 7.0) | 45 (25.1 / 22.4) | 14 (7.9 / 7.0) | 54 (27.1 / 25.8) | 2 |
| PFE | 28 (14.7 / 13.9) | 14 (7.9 / 7.0) | 27 (14.3 / 13.4) | 14 (7.9 / 7.0) | 32 (15.5 / 15.3) | 1 |
| TSLA | 28 (14.1 / 13.9) | 13 (7.2 / 6.5) | 28 (14.1 / 13.9) | 13 (7.2 / 6.5) | 29 (13.1 / 13.8) | 2 |
| WMT | 26 (15.5 / 12.9) | 13 (7.2 / 6.5) | 21 (12.1 / 10.5) | 13 (7.1 / 6.5) | 26 (13.4 / 12.4) | 2 |
| XOM | 37 (21.8 / 18.4) | 14 (8.8 / 7.0) | 34 (19.8 / 16.9) | 14 (8.8 / 7.0) | 44 (21.9 / 21.0) | 3 |

- **The 25% company cap:** it allows 54 at 216 passages and 45 at 182;
  JPMorgan reaches it in both E and C.
- **Why JPMorgan grows:** caps shrink the quotas of companies whose
  passages are concentrated (Tesla's FY2019, Walmart's and ExxonMobil's
  long units), and the quota moves to JPMorgan, which has the largest pool.
- **B and D are the most even** across companies, at half the size.

### Fiscal years: passages by company, 2019–2026

| Company | A | C | E | B and D |
|---|---|---|---|---|
| AAPL | 4 3 2 0 1 2 1 0 | 4 3 2 0 1 2 1 0 | 5 4 3 0 1 2 1 0 | 2 2 2 0 1 2 1 0 |
| AMZN | 6 3 1 2 1 1 2 0 | 4 3 1 2 1 1 2 0 | 4 3 1 2 1 1 3 0 | 2 2 1 2 1 1 2 0 |
| JPM | 5 5 5 6 6 5 5 0 | 7 8 6 6 6 6 6 0 | 8 8 7 7 8 8 8 0 | 2 2 2 2 2 2 2 0 |
| PFE | 10 3 4 3 3 2 3 0 | 9 3 4 3 3 2 3 0 | 7 6 5 4 4 3 3 0 | 2 2 2 2 2 2 2 0 |
| TSLA | 13 2 3 3 1 1 5 0 | 13 2 3 3 1 1 5 0 | 8 4 3 2 2 2 8 0 | 2 2 2 2 2 1 2 0 |
| WMT | 0 7 4 1 2 3 3 6 | 0 4 3 1 2 4 3 4 | 0 5 5 2 3 3 4 4 | 0 2 2 1 2 2 2 2 |
| XOM | 7 7 7 4 4 3 5 0 | 6 5 6 4 4 4 5 0 | 8 8 8 5 5 5 5 0 | 2 2 2 2 2 2 2 0 |
| **All** | 45 30 26 19 18 17 24 6 | 43 28 25 19 18 20 25 4 | 40 38 32 22 24 24 32 4 | 12 14 13 11 12 12 13 2 |

- **Coverage:** every plan covers every fiscal year each company still has
  eligible passages in. Apple has none in 2022; Walmart's fiscal years run
  2020–2026.
- **The concentration is Tesla's:**
  - its 2023 and 2024 filings have only 2 eligible groups each, so its
    quota fills from 2019 (15 groups);
  - E cuts Tesla's 2019 from 13 to 8 and takes the rest from its 2025
    filing (7 eligible groups) and from other companies.

### Linked units: how many passages each contributes

| Passages from one unit | A | B | C | D | E |
|---:|---:|---:|---:|---:|---:|
| 1 | 119 | 59 | 126 | 62 | 140 |
| 2 | 9 | 6 | 10 | 6 | 17 |
| 3 | 3 | 3 | 12 | 5 | 14 |
| 4–5 | 5 | 2 | – | – | – |
| 8 | 2 | – | – | – | – |

**A's largest units:**
- JPMorgan `grp_01824`: 8 passages, one statements note over 2019–2024;
- Walmart `grp_07889`: 8, one note over 2020–2026, mostly tables;
- ExxonMobil `grp_08904`: 5;
- Amazon `grp_00687`: 5;
- JPMorgan `grp_01705`: 5 tables.

## 4. Safeguards, checked on every plan

All five plans pass every check:
- validation passages only, kept by the source-quality rules;
- one passage per overlap group;
- no passage in an excluded overlap group or unit (the evidence of
  `train_candidates_v2`, `val_pilot_v1`'s units and the carried questions'
  equivalents);
- isolation share at or above the threshold;
- company, filing and unit caps held;
- made from the `bge_base_ft_v3` manifest (`21c2293a…`).

**Not changed by any plan:**
- **The split:** a plan only chooses validation passages, so pins,
  evidence-reuse-v2 links, the 66 candidates and the 16 questions are as
  they were.
- **Data files:** only the four new plan files were added under `data/`.
  No file in `data/`, `configs/` or `results/` changed.

**Write-once:** re-running each plan command reports "unchanged (identical
plan present)", and `val_pilot_v2`'s file is unchanged.

**Dry run:** `generate --plan val_pilot_v2e` without `--confirm-api-use`
sent nothing and wrote nothing.

**Later checks:** a later `build-validation` applies the evidence,
evaluation-leakage, training-overlap and duplicate rules to every question
again.

## 5. What clustering was measured

**Data:** the 82 synthetic questions with stored per-question results (the
66 candidates and 16 validation questions; base models; no evaluation
labels).
- **Statistic:** the one-way intraclass correlation of per-question MRR@10,
  and of paired differences, by company, filing and v3 unit.
- **Intervals:** 95% cluster-bootstrap intervals.

| Statistic | Company | Filing | Unit |
|---|---|---|---|
| BGE MRR@10 | 0.01 (−0.08 to 0.08) | −0.02 (−0.33 to 0.29) | −0.24 (−0.80 to 0.30) |
| BGE − BGE without instruction (small change) | 0.00 (−0.07 to 0.07) | −0.01 (−0.35 to 0.34) | −0.25 (−1.11 to 0.68) |
| BGE − E5 | 0.04 (−0.05 to 0.10) | 0.04 (−0.31 to 0.33) | 0.37 (−0.19 to 0.70) |
| BGE − BM25 | −0.04 (−0.09 to −0.01) | −0.14 (−0.42 to 0.16) | 0.21 (−0.27 to 0.69) |

**Measured:**
- **By company:** near zero.
- **By filing:** near zero, with intervals of about ±0.3. The 82 questions
  fall in 44 filings, 27 of them holding two or more.

**Not measured:**
- **By unit.** Only 20 units hold two questions, and none more, so the
  intervals span almost everything.
- **What the point estimates show:** different dense models' differences
  (0.37) and dense against lexical (0.21) hint at substantial unit
  correlation. The small-change proxy shows none, but it is too noisy to
  rule anything out.

## 6. Effective size: 120 questions are not 120 observations

**Model (assumed):** additive correlations of per-question checkpoint
differences.
- **The parameters:** ρ_c within a company, ρ_f more within a filing, ρ_u
  within a unit.
- **The design effect:** DEFF = 1 + ρ_c(M_c − 1) + ρ_f(M_f − 1) +
  ρ_u(M_u − 1), and the effective size is n / DEFF.
  - Each M is the size-weighted mean cluster size, Σm²/n, expected over
    which planned questions survive.
  - The 16 carried questions are included.

**Scenarios:**

| Scenario | ρ_c | ρ_f | ρ_u | Grounds |
|---|---:|---:|---:|---|
| Low | 0 | 0 | 0.1 | Measured company and filing values; little unit correlation |
| Central | 0.02 | 0.05 | 0.3 | Middle of the measured company range; filing above its point estimates; unit assumed |
| High | 0.04 | 0.10 | 0.5 | Top of the company range; unit like the BGE − E5 point estimate |

| | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| Expected questions n (score estimate, with the 16) | 122 | 68 | 118 | 68 | 124 |
| M: company / filing / unit | 19.1 / 3.6 / 1.83 | 10.1 / 1.9 / 1.39 | 19.0 / 3.5 / 1.30 | 10.1 / 1.9 / 1.25 | 20.3 / 3.5 / 1.29 |
| Effective size: low | 113 | 66 | 114 | 66 | 121 |
| Effective size: central | 70 | 51 | 75 | 52 | 78 |
| Effective size: high | 51 | 41 | 56 | 43 | 57 |

**Reading:**
- **About 120 questions are worth about 50–80 independent ones** under the
  central and high assumptions, and close to 120 only if units barely
  correlate.
- **The company term is the largest part** (DEFF +0.36 at ρ_c = 0.02 with
  7 companies of about 17 questions each), and it is nearly the same in
  every plan. It stays as long as there are seven companies.
- **The unit term is what the plans change:**
  - A: +0.25 central, +0.42 high;
  - C and E: +0.09 central, +0.15 high.
- **A strict filing cap lowers the effective size,** although each question
  is more independent: B and D halve n and only modestly shrink the filing
  term.
- **Choosing a checkpoint** (δ = 0.02 MRR@10, SD 0.11–0.15 from the design
  measurements, an assumption for real checkpoints):
  - C and E: 0.88–0.95 correct choices (central);
  - A: 0.87–0.94;
  - B and D: 0.83–0.91.

## 7. The yield estimate is probably optimistic

**Isolation score against `val_pilot_v1`'s outcome** (the 43 generated
questions; the score computed in `bge_base_ft_v2`, where they were judged):

| Isolation share | Questions | Predicted pass | Observed pass |
|---|---:|---:|---:|
| 0–0.3 | 20 | 0.05 | 0.20 (4) |
| 0.3–0.5 | 7 | 0.41 | 0.57 (4) |
| 0.5–0.8 | 4 | 0.70 | 0.50 (2) |
| 0.8–1 | 12 | 0.93 | 0.75 (9) |
| All | 43 | 0.41 | 0.44 (19) |

**What this shows (measured):**
- **Above 0.5 the score over-predicts:** 11 of 16 observed (69%) against
  about 87% predicted. Below 0.5 it under-predicts.
- **The aggregate agreement** (41% against 44%) came from errors in
  opposite directions.

**The band estimate** in §3 uses the observed rates: 11/16 at 0.5 or more,
4/7 at 0.3–0.5, 4/20 below 0.3.
- **The bands are small:** the 95% interval of 11/16 is about 44–86%.
- **What follows for the plans:**
  - plans at 0.5 or more (A–D) probably yield 11–15% less than the
    planner prints;
  - a 0.3 threshold costs less than the score implies, so E, with 49
    passages in 0.3–0.5, gains relatively.

## 8. Recommendation

**Recommendation: E, `val_pilot_v2e`, approved one batch of 50 at a time,
re-estimating after each batch and stopping once about 120 questions are
expected.**

**Statistical diversity:**
- **It gives the highest effective size of the five** (78 central, 57
  high, against 70 and 51 for A).
- **It caps units at 3 passages,** in 171 units against 138.
- **The unit cap is the constraint the evidence supports:** unit
  correlation is plausible and unmeasured; filing correlation is measured
  near zero.

**Fiscal years:**
- **FY2019 falls** from 24% to 18.5%, and the largest filing from 13
  passages (Tesla 2019) to 8.
- **A cap of 8 is about the tightest that still reaches 120:**
  - with 3 per unit, caps of 5, 6 or 7 reach at most about 107, 115 and
    120 questions (score estimate), 101, 109 and 115 (band);
  - a cap of 4 reaches about 99;
  - a cap of 2 reaches 68.

**Yield:**
- **About 124 questions (score) or 119 (band), with the 16.** It is the
  only alternative near 120 on both estimates.
- **A reaches 108 on the band estimate,** C 107, and B and D 60.

**Company representation:**
- **The cost:** JPMorgan rises to 25% (54 passages, about 27 questions),
  and Apple and Amazon stay small (7–9 expected questions each).
- **A 20% company cap would limit JPMorgan** to 39, but allows only 196
  passages: about 115 questions (score) or 110 (band) with the 16
  (simulated, not written).

**Training data:** unaffected by any of the five. No plan moves a passage,
and B and D's ceiling cannot be lifted by moving training data.

**Costs and risks of E:**
- **Requests:** 216, 31 more than A: 5 batches (50, 50, 50, 50, 16), about
  1.6 h (up to 3.6 h), about 290,000 tokens.
- **Uncertain passages:** 49 passages score 0.3–0.5. Their pass rate rests
  on 7 pilot questions.
- **No room under these caps:** E is the largest plan they allow. A
  shortfall needs looser caps or a lower threshold, in a new plan, with
  approval.
- **Several questions per unit remain:** 31 units give 2–3 questions each,
  so a later validation set must still record each question's equivalents.

**Staged approval:**
- **Plan order alternates companies** and spreads each company's years
  first, so every prefix is balanced. E's first 150 passages hold 23 from
  FY2019 (15%).
- **Stopping early costs nothing in balance.**
- **The stopping rule must depend only on question counts,** never on
  retrieval results, so it cannot bias checkpoint choice.

**If you prefer to keep 185 requests:** choose C. It has the best effective
size at that budget, but leaves FY2019 at 24% and expects 107–118
questions.

**B and D are not recommended for this target.**
- They give the most even company and year mix.
- But they halve the sample, lower the effective size, and cannot reach
  120 with one question per passage.

## 9. Commands

Common options of all five plans:

```bash
COMMON="--config configs/train/data_prep_v3.yaml --split validation --method balanced_years \
  --max-company-share 0.25 --one-per group --exclude-evidence-of train_candidates_v2 \
  --exclude-validation-set configs/train/data_prep_v2.yaml:validation_pairs_v1 \
  --exclude-validation-set configs/train/data_prep_v3.yaml:validation_pairs_v2"
```

**The plans** (each written once):

```bash
python3 scripts/02_generate_queries.py plan $COMMON --name val_pilot_v2b --size 89 --min-isolated-share 0.5 \
    --max-per-company-year 2                    # 185 refused: "only 89 passages fit 185 …"
python3 scripts/02_generate_queries.py plan $COMMON --name val_pilot_v2c --size 182 --min-isolated-share 0.5 \
    --max-per-unit 3                            # 185 refused: "only 182 passages fit 185 …"
python3 scripts/02_generate_queries.py plan $COMMON --name val_pilot_v2d --size 89 --min-isolated-share 0.5 \
    --max-per-company-year 2 --max-per-unit 3   # 185 refused
python3 scripts/02_generate_queries.py plan $COMMON --name val_pilot_v2e --size 216 --min-isolated-share 0.3 \
    --max-per-company-year 8 --max-per-unit 3
```

**Plan files** (SHA-256), in `data/processed/train/bge_base_ft_v3/plans/`:

| Plan | SHA-256 |
|---|---|
| `validation.val_pilot_v2b.json` | `4a92438b…dfed88` |
| `validation.val_pilot_v2c.json` | `1e077148…a0b6` |
| `validation.val_pilot_v2d.json` | `939d5c9b…81f8d6` |
| `validation.val_pilot_v2e.json` | `63922fc0…e92f5` |

**Dry run** (sends nothing, needs no key):

```bash
python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v3.yaml --split validation \
    --plan val_pilot_v2e --max-requests 50
```

**Each approved batch:** the same command with `--confirm-api-use` and
`GROQ_API_KEY` set.

## 10. Open decisions

1. **Which plan to generate:** E is recommended, C keeps 185 requests, and
   A is unchanged.
2. **Whether to accept JPMorgan at 25%** in E, or tighten the company cap
   and accept fewer questions.
3. **The stopping rule for staged approval:** for example, stop when
   generated, evidence-checked questions are expected to give 104 usable
   new ones.
4. **Whether to use one plan:** generation records stay with their passage,
   so a passage done under one plan is skipped under another. Generating
   from two of these plans would therefore not duplicate requests, but it
   would mix their designs.
