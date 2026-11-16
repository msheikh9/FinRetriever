# Validation plan `val_pilot_v2f`: audit of positions 144–147 and cumulative statistics

Status: 2026-11-04.
- **What this is:** an AI review (Claude) of the 4 generation records for plan positions 144–147, the end
  of batch 3's window, with cumulative statistics for every reviewed batch. It is not human annotation,
  and no decision here is a gold label.
- **The plan is not finished:** `val_pilot_v2f` has 361 positions. Positions 1–147 (the windows of batches
  1–3) are now processed, and 214 (148–361) remain.
- **What was not done:**
  - no generation record, plan, training pair, candidate, calibration, validation set or decision file
    was changed;
  - no validation pair was built, and no decision file was written;
  - no API request was sent and nothing was trained.
- **Checksums:** 272 files under `data/`, `configs/` and `results/` were checksummed before and after the
  audit, and none changed. Since batch 3's audit, only `validation.records.jsonl` has changed.
- **Earlier documents:** the plan [validation_plan_v2f.md](validation_plan_v2f.md), and the audits of
  [batch 1](validation_plan_v2f_batch1_audit.md), [batch 2](validation_plan_v2f_batch2_audit.md) and
  [batch 3](validation_plan_v2f_batch3_audit.md) (positions 98–143).

## 1. Summary

- **Positions 144–147:**
  - the generator accepted 2 of 4 (144, 147);
  - the evidence rule left 1 (144);
  - the review accepts both, so **1 usable question** (144), against 1.3 expected.
- **Batch 3's full window (98–147):** 12 usable from 50 passages, against 13.0 expected.
- **This plan so far (147 positions):** 50 usable, against 52.4 expected.
- **Running total:** **81 usable validation questions:**
  - 16 carried;
  - 15 from `val_pilot_v2e`;
  - 50 from this plan (18, 20 and 12 from batches 1–3).
- **The two passages flagged for checking:**
  - 146 (FY2021 discussing 2020): the generator rejected it for copying. Its year is right, but the
    question is circular, so the review would have rejected it too.
  - 147 (columns 2023, 2024, 2025): Qwen took the right column. The question is correct, but the evidence
    rule excludes it on two possible matches that do not hold the answer (§3).
- **Recommendation:** run batch 4 (positions 148–197) unchanged, with the review kept (§6).
  - **Worth the cost:** batches 4–6 (to position 297), at 5–6 requests per usable question.
  - **Not worth it:** positions 298–361, at 11 or more requests each, almost all from JPMorgan and
    ExxonMobil.

## 2. Provenance

| Check | Result |
|---|---|
| Plan | `validation.val_pilot_v2f.json` (`a7903bc7…`), unchanged. The 4 records are its positions 144–147, in order. Position 148 (JPM FY2022) is next |
| Run | The same bound run as batches 1–3: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, parameters `42d5fe54…`, validator `sec-qcheck-v3` |
| Records | 4 done, 0 failed, each on its first response attempt, no errors. 144–146 at 00:52:27–00:52:31 UTC; 147 at 00:55:29 UTC, in a separate execution |
| Tokens | 5,379: 4,788 prompt and 591 completion (1,345 per passage) |
| Files | `validation.records.jsonl` is now `d7eb9d4b…`, 197 lines. Its first 193 lines are byte-identical to the file batch 3 was audited on (`be52f9d1…`), so the run only appended. `validation.run.json` `34e6b74d…` is unchanged |
| Re-runs | The validator reproduces all 4 outcomes; all 4 plan scores recompute exactly |

## 3. The four passages

| # | Source | Question (verbatim) | Answer (verbatim) | Generator | Evidence rule | Decision |
|---|---|---|---|---|---|---|
| 144 | TSLA FY2019, narrative/risk (Item 1A) | According to Tesla's fiscal 2019 Form 10-K, what specific risk is associated with the potential inability to ramp up Gigafactory Shanghai on time? | Tesla may continue to be exposed to unfavorable tariffs, duties, or costs, which could be detrimental compared to locally-based competitors. | Accepted | **In** | **ACCEPT: usable** |
| 145 | WMT FY2020, narrative/statements (Item 8) | What was the total depreciation and amortization expense reported by Walmart Inc. for property and equipment, property under finance leases and financing obligations, property under capital leases, and intangible assets in fiscal 2020? | $11.0 billion | `copied_from_passage` | – | Rejection correct |
| 146 | XOM FY2021, narrative/business | In which business segment did ExxonMobil report that Downstream margins did not reach the lower end of the 10-year range until late in fiscal 2021? | Downstream margins only reached the lower end of the 10-year range late in 2021 | `copied_from_passage` | – | Rejection correct |
| 147 | AMZN FY2025, table/statements (Item 8) | What was the total lease cost for Amazon in the year ended December 31, 2025, in millions? | $20,296 million | Accepted | Out: possible evaluation evidence, possible training evidence | ACCEPT (excluded by the rule) |

**144, Tesla FY2019 risk factors:**
- **Evidence rule:** in. No passage anywhere else holds the sentence, and it has no equivalents in
  validation. The sentence appears only in this FY2019 filing.
- **Factual accuracy:** pass. The answer paraphrases the quoted sentence. "Such" tariffs refers to tariffs
  on U.S.-built vehicles and duties on Chinese components, named in the visible sentences before it.
- **Financial relevance:** pass. A company-specific risk (tariff and duty costs in China), not generic
  wording.
- **Company and year:** pass; the question names Tesla and the fiscal 2019 Form 10-K.
- **Figures and units:** not applicable.
- **Visibility:** the quote ends at token 161 of 512.
- **Similarity:**

| Pool | Highest Jaccard |
|---|---:|
| Evaluation | 0.105 |
| Training (all 150 ever generated) | 0.297 |
| The 16 carried | 0.216 |
| Batch 3 | 0.231 |
| Other validation pools | ≤ 0.244 |

- **Linked unit:** its first generated passage.

**145, Walmart FY2020 property note:**
- **Generator:** rejected for copying the sentence's long list of asset types; correct by the rule.
- **Answer:** right: $11.0 billion for fiscal 2020, the first of three years in the sentence.

**146, ExxonMobil FY2021 business environment** (flagged: a FY2021 filing that discusses 2020):
- **The year:** the passage covers early 2020 (COVID-19 and the price collapse) and the recovery "through
  2021". The question's "late in fiscal 2021" matches the quoted sentence ("late in 2021"), so there is no
  period error.
- **The question is circular:** it asks "in which business segment" while naming Downstream, and the
  answer restates the sentence instead of naming a segment.
- **Generator:** rejected for copying, which is correct. Had it passed, the review would have rejected it
  as having no answerable content.

**147, Amazon FY2025 lease-cost table** (flagged: columns 2023, 2024, 2025):
- **Factual accuracy, year, figures and units:** pass. $20,296 million is the "Total lease cost" for the
  year ended December 31, 2025, the third column, and the question names that year.
- **Financial relevance and company:** pass.
- **Visibility:** the quote ends at token 164 of 512.
- **Evidence rule:** out, on two possible matches. Neither holds the answer:
  - a blocked passage of Amazon's FY2024 filing, linked to an evaluation pool by overlap group. Its
    commitments table has "Thereafter: 20,296", the same number by coincidence;
  - a training passage of Amazon's FY2023 filing holds the same "Total lease cost" row, with 2021–2023
    values.
- **The rule worked as designed:** it treats any possible match outside validation as a leak, and is
  deliberately conservative. This is a measured cost of that design (one correct question lost), not a
  reason to relax it.
- **Similarity:**

| Pool | Highest Jaccard | Against |
|---|---:|---|
| Evaluation | 0.192 | |
| Training | 0.524 | Amazon's 2021 operating expenses |
| Batch 1 | 0.565 | Amazon's 2025 cash taxes |
| Batch 2 | 0.500 | |

- **Linked unit:** its second generated passage, after an Amazon FY2024 passage.
- **The filing:** the fourth plan passage from Amazon's FY2025 filing. That filing keeps its four usable
  questions.

## 4. Cumulative statistics

**This plan by batch:**

| | Batch 1 (1–49) | Batch 2 (50–97) | Batch 3 (98–147) | All 147 |
|---|---:|---:|---:|---:|
| Passages | 49 | 48 | 50 | 147 |
| Generator accepted | 35 (71%) | 41 (85%) | 39 (78%) | 115 (78%) |
| Evidence pass, of accepted | 24 (69%) | 21 (51%) | 15 (38%) | 60 (52%) |
| Review: accept / revise / reject | 26 / 4 / 5 | 39 / 2 / 0 | 36 / 3 / 0 | 101 / 9 / 5 |
| Review acceptance, of evidence-passing | 0.75 | 0.95 | 0.80 | 0.83 |
| Factual errors, of accepted | 6 | 2 | 2 | 10 (9%) |
| **Usable** | **18** | **20** | **12** | **50** |
| Expected by the plan | 23.4 | 16.0 | 13.0 | 52.4 |
| Tokens | 64,397 | 64,963 | 69,662 | 199,022 |

**Every reviewed pilot:**

| | `val_pilot_v1` | `val_pilot_v2e` batch 1 | `val_pilot_v2f` 1–147 | All |
|---|---:|---:|---:|---:|
| Passages generated | 50 | 50 | 147 | 247 |
| Questions reviewed (generator-accepted) | 43 | 39 | 115 | 197 |
| Evidence pass | 19 | 17 | 60 | 96 |
| Usable | 16 (the carried) | 15 | 50 | **81** |
| Usable per passage | 32% | 30% | 34% | 33% |

**Review findings over this plan's 115 reviewed questions:**
- **Company attribution:** correct in all 115.
- **Factual accuracy:** 10 fail, 4 minor. The failures:
  - the wrong row, year or sign: 2, 14, 23, 26, 99;
  - what a figure is: a nil row with a held-for-sale footnote (32), a pro forma adjustment (74), debt due
    within a year called redeemed (91), a dash read as nil (112);
  - no product named (18).
- **Fiscal year:** 4 fail, 8 minor. The failures are an earlier year's figure read as the current year's
  (23, 26, 99) and a question with no year whose answer depends on it (98).
- **Figures:** 6 fail, 3 minor. **Units:** 12 minor (a missing "millions"); none wrong.
- **Financial relevance:** 3 fail, all in batch 1: two administrative passages (39, 41) and one trivial
  answer (45). 13 minor.
- **Boilerplate:** 9 questions.
  - The evidence rule excluded 6.
  - 45 passed it but was rejected.
  - 17 and 108 are usable; every copy of their sentences lies in validation, recorded as equivalents.
- **Ten of the 60 evidence-passing questions** (17%) were revised or rejected by the review alone: 14, 18,
  32, 39, 41, 45, 74, 98, 99 and 112. Without the review, the set would hold 60 questions, about one in six
  of them wrong or unsuitable.
- **Generator rejections:** 32 of 147 passages: copied wording 12, long quotes 10, no company 8, quote not
  verbatim 1, answer not visible 1. Every rejection checked in the four audits was correct by the rule.

**The predictor over all 115 reviewed questions** (out of sample):
- **Accuracy:** AUC 0.794, against 0.697 for the old sentence share. Brier 0.186, against 0.256 for the
  constant rate.
- **Calibration:** 53.6 passes predicted, 60 observed:

| Predicted pass | Questions | Predicted | Observed |
|---|---:|---:|---:|
| Below 0.2 | 20 | 2.2 | **0** |
| 0.2–0.4 | 25 | 7.5 | 12 |
| 0.4–0.6 | 33 | 16.2 | 20 |
| 0.6–0.8 | 26 | 17.6 | 18 |
| 0.8 and above | 11 | 10.0 | 10 |

- **Against the unranked pilots:** evidence pass 52% against 44% (p = 0.16); usable per passage 34%
  against 31% (p = 0.36). The ranking front-loaded the yield, and the total gain is not established.

**The 81 usable questions:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Usable | 11 | 15 | 13 | 8 | 10 | 12 | 12 | 81 |
| Remaining plan passages | 0 | 7 | 69 | 37 | 32 | 22 | 47 | 214 |
| Expected from them (projection) | 0 | 0.7 | 11.7 | 5.6 | 5.8 | 2.6 | 6.0 | 32.5 |

- **Fiscal years:** 2019 16, 2020 12, 2021 11, 2022 10, 2023 11, **2024 6**, 2025 15.
- **Kinds:** this plan's 50 are 26 tables and 24 narrative passages.
- **Filings:** 44 filings. 18 give one question, 17 two, 7 three and 2 four (Amazon FY2019 and FY2025).
- **Linked units:** no unit holds more than 3 generated passages.
- **Isolation:** every positive's unit and overlap group lie wholly in validation.
- **Leakage:** none. No usable question's evidence is held by any train or blocked passage. Across the 81,
  the highest similarity to an evaluation question is 0.28, and to any of the 150 training questions 0.74,
  both under the 0.8 threshold.

## 5. The remaining supply and its cost

**By block of 50** (projections at this plan's observed rates: generator 0.78, review 0.83). "Low end
zero" assumes passages scoring below 0.2 keep giving nothing, as all 20 so far did.

| Positions | Mean score | Below 0.2 | Expected usable | Low end zero | Requests per usable | Cumulative total |
|---|---:|---:|---:|---:|---:|---:|
| 148–197 (batch 4) | 0.31 | 20 | 9.9 | 8.2 | 5.0–6.1 | 91 (89) |
| 198–247 | 0.28 | 17 | 9.0 | 8.2 | 5.6–6.1 | 100 (97) |
| 248–297 | 0.24 | 21 | 7.9 | 6.0 | 6.3–8.3 | 108 (103) |
| 298–347 | 0.14 | 38 | 4.6 | 2.1 | 11–23 | 112 (106) |
| 348–361 | 0.12 | 12 | 1.1 | 0.3 | 13–50 | 114 (106) |

- **The whole remainder:** 32.5 usable questions (bootstrap 30–36), or 24.8 with the low end at zero. The
  final total would be about 106–114.
- **Cost per batch of 50:** about 70,000 tokens, 22–25 minutes of generation, and a full review. The review
  is the larger effort, since it removes about one in six evidence-passing questions.
- **Batch 4's composition:** Amazon 7, JPMorgan 9, Pfizer 9, Tesla 9, Walmart 8, ExxonMobil 8.
  - It has 7 FY2024 passages, the year the set is shortest of.
  - 20 of its 50 passages score below 0.2, so it also tests the predictor's low end directly.
- **The tail is unbalanced:**
  - positions 298–361 are 38 JPMorgan, 17 ExxonMobil, 7 Pfizer and 2 Tesla passages;
  - 50 of their 64 passages score below 0.2.
- **What more questions buy:** the set's precision grows slowly. For a rate near 50%, the standard error
  of a validation metric is about 0.056 at 81 questions, 0.052 at 91, 0.048 at 108 and 0.047 at 114.

## 6. Recommendation

**Batch 4 (positions 148–197) should proceed on `val_pilot_v2f` unchanged,** with every question
AI-reviewed as before.
- **The safeguards held:** across 147 positions there was no leakage, no question overlap, no unit
  crossing the split, and no cap breached.
- **The cost per question is unchanged:** about 10 usable questions for 50 requests, the same rate as
  batch 3.
- **It improves balance:** six companies nearly evenly, and 7 FY2024 passages.
- **It answers the open question:** its 20 low-scoring passages show whether the predictor's low end yields
  anything. That decides whether batches 5 and 6 are worth running.

**Is the remaining supply worth its API cost?**
- **Through position 297 (batches 4–6): yes.**
  - About 27 usable questions at 5–8 requests each, for a total of about 103–108.
  - That narrows a metric's standard error by about 14% (0.056 to 0.048), for about 210,000 tokens and
    three reviews.
- **Positions 298–361: no.**
  - About 6 usable questions at best, and about 2 if the low end stays at zero.
  - Each costs 11–50 requests, and nearly all come from JPMorgan and ExxonMobil, which would worsen the
    company balance.
  - I recommend not running them under this plan.
- **Decide again after batch 4:**
  - if its low-scoring passages again give nothing, batches 5 and 6 are worth about 14 questions
    together, at 6–8 requests each;
  - if the set's purpose is satisfied near 90–100 questions, stopping after batch 4 or 5 is reasonable.
- **120 questions is out of reach of this plan.** Reaching it would need a supply change, either a unit cap
  of 4–5 or a larger validation split, in a new plan version.

**Other decisions for you, none blocking batch 4:**
- **Build-time caps:** whether to cap questions per company, filing or template. Two Amazon filings give
  four questions each, and Apple's Level 2 securities template gives five.
- **Recording:** the decisions of batches 1–3 (positions 1–147) can go into versioned files in
  `data/validation_review/`, in `val_pilot_v2e`'s format, once you approve them.
