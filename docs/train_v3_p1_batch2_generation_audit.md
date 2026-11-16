# P1 batch 2 (`train_v3_p1_batch2`): generation audit and quality gate

Status: 2026-11-06.
- **What this is:**
  - an audit of the 100 generation records of plan `train_v3_p1_batch2` in `bge_base_ft_v3`;
  - an AI review (Claude) of the 124 pairs they build;
  - the batch quality gate, and a comparison with batch 1.

  It uses the method of the [batch 1 audit](train_v3_p1_batch1_generation_audit.md). It is not human annotation, and
  no decision here is a gold label.
- **Where the decisions are:** `data/train_review/train_v3_p1_batch2_quality_decisions.jsonl`.
  - 124 rows (115 accept, 6 revise, 3 reject), each `reviewer: ai_review` and `gold: false`.
  - Each row is bound to its `pair_id`, `passage_id` and the SHA-256 of its exact question. The pair IDs are those of
    the pair set that the current 200 train records build (§12).
  - SHA-256 `dc3caa90…f830`.
- **What was not done:**
  - no question was generated and no API request was sent;
  - no generation record, plan, candidate dataset, validation set or earlier decision was changed. Batch 1's records
    and decisions are byte-identical (§2);
  - `build-pairs`, `screen-questions` and `build-candidates` were not run in `bge_base_ft_v3`. They ran only in a
    scratch copy of the preparation, which was then discarded. No final candidate dataset was built;
  - nothing was trained, committed or pushed.
- **Protected files:** 345 files under `data/`, `configs/`, `src/` and `scripts/` were checksummed before and after
  the audit. They are unchanged; the decisions file is the only file added.
- **Earlier documents:** [the batch 1 audit](train_v3_p1_batch1_generation_audit.md),
  [the readiness audit](final_training_readiness_audit.md) and [the full-training plan](training_full_v1.md), §9.

## 1. Verdict

**The batch passes the approved quality gate on all four criteria.**

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | At least 0.8 included pairs per planned passage | 115 included for 100 planned passages: **1.15** | **Pass** |
| 2 | At least 75% AI-review acceptance | 115 of 124 built pairs: **92.7%** | **Pass** |
| 3 | Zero accepted leakage violations | **0**: no included pair fails the evidence, cross-split, lexical, semantic or duplicate checks (§7, §8) | **Pass** |
| 4 | Complete provenance for every included pair | **115 of 115** carry the full provenance block, and the fail-closed provenance check passes (§10) | **Pass** |

The further checks also hold:
- **Companies:** every company contributes at least 13 included pairs (Amazon is lowest).
- **Semantic screen:** no question is flagged at 0.90, and no batch 2 question falls in the 0.80–0.90 review band
  (highest 0.7929).
- **Existing data:** `train_candidates_v2` rebuilds byte-identical, and `validation_pairs_v4` loads apart from all
  282 trial training pairs.
- **One generator:** all 200 records use one model, prompt, validator and parameter set.

**GO for batch 3**, under the conditions in §13:
1. **Same generator:** the same model, prompt, validator, parameters and plan options, with the validator unchanged.
2. **Option A:** no real `build-pairs` until all six batches are generated.
3. **The budget:** about 1% margin is left for batches 3–6 at batch 2's rate (§12).
4. **Company balance:** a decision is needed before batch 4, not batch 3.
   - Batch 3's plan is the same under every option: it takes the last 16 Apple and 5 Amazon groups.
   - Batches 4–6 would then have neither company (§12).

**Unchanged:** the three-seed A0 versus A1 experiment stays **NO-GO**. Blockers B1 (a selection-aware decision rule),
B2 (the planned data size) and B3 (the GPU rehearsal) still stand.

## 2. Records and provenance

| Check | Result |
|---|---|
| Plan | `train.train_v3_p1_batch2.json`, SHA-256 `156856f5…7a8b`, unchanged; 100 train passages |
| Records file | `generation/train.records.jsonl`, 200 lines, SHA-256 `79853cdb…8587` |
| Batch 1's records | Lines 1–100 hash to `91116337…56d1`, as the batch 1 audit recorded them. The run only appended. |
| Batch 2's records | Lines 101–200: one record per plan position, in plan order; no other plan's records; 200 distinct passages in the file |
| Outcome | **100 `done`, 0 `failed`**. Every record took 1 attempt, with no error. |
| Identity | All 200 records: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…`, record format 3 |
| Run | `train.run.json` is unchanged (`72c9e7bf…`). It binds prompt SHA-256 `f42bdc54…`, the same parameters and preparation `21c2293a…`. There is no continuation file. |
| Replies | 100 distinct response hashes (199 across the file; batch 1's failed record has none) |
| Validator re-run | `revalidate --dry-run`: 199 up to date, 0 changed, 1 failed record (batch 1's position 87) |
| Tokens | **146,678** (121,656 prompt, 25,022 completion), equal in the records and the six run summaries. Batches 1 and 2 total 294,962. |
| Completion tokens per reply | 169–424, median 244.5. No reply came near the 768 limit. |

**The runs:** six invocations, from 2026-11-05 20:13 to 2026-11-06 16:07 UTC (record timestamps), from
`logs/train_v3_p1_batch2.log`. The summary you gave (6 requests, 9,080 tokens, no HTTP 429) is the sixth.

| Run | Request budget | HTTP requests | HTTP 429 | Done | Tokens | Ended by |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 100 | 67 | 1 | 66 | 96,559 | a rate-limit wait of 265 s |
| 2 | 33 | 10 | 2 | 8 | 11,716 | a wait of 204 s, after a 125 s wait was taken |
| 3 | 23 | 12 | 1 | 11 | 15,908 | a wait of 247 s |
| 4 | 11 | 9 | 2 | 7 | 10,476 | a wait of 627 s, after a 174 s wait was taken |
| 5 | 2 | 2 | 0 | 2 | 2,939 | its request budget |
| 6 | 10 | 6 | 0 | 6 | 9,080 | the end of the plan |
| **All** | | **106** | **6** | **100** | **146,678** | |

**How Groq's limits were handled:**
- **The generator worked as designed.** 106 HTTP requests gave 100 usable replies and 6 HTTP 429 responses.
  - It waited out the two waits of at most 180 s.
  - It stopped cleanly at the four longer ones, without recording the passage.
  - Each rerun skipped the completed passages.
  - It spent 4,613 s in pacing and rate-limit waits.
- **No passage was lost or retried** because of a rate limit.
- **What the headers show:** the last response of each run showed 1,000 requests per day (934 or more remaining) and
  8,000 tokens per minute. The 429 waits of 2–10 minutes are longer than the per-minute window, so some limit the
  headers do not report is binding. Expect several resumes per batch.

**Against batch 1 and the plan's per-batch estimate** ([training_full_v1.md](training_full_v1.md), §9):

| | Estimate | Batch 1 | Batch 2 |
|---|---|---:|---:|
| Tokens | about 143,000 | 148,284 | 146,678 |
| Questions passing the validator | about 155 | 151 | 159 |
| Pairs built | about 140 | 112 | 124 |
| AI review accepts | about 115 | 101 | 115 |
| Candidates | about 100 (range about 80–115) | 101 | 115 |

## 3. The funnel

200 proposed (100 × 2) → **159** pass the generator's validator (79.5%) → **124** pairs built → **115** accepted by
the review → **115** included in the trial candidate set.

| | Plan | Done | Proposed | Pass validator | Pairs built | Accepted (included) |
|---|---:|---:|---:|---:|---:|---:|
| Batch 1 | 100 | 99 | 198 | 151 (76%) | 112 | 101 |
| Batch 2 | 100 | 100 | 200 | 159 (80%) | 124 | 115 |
| **Both** | **200** | **199** | **398** | **310 (78%)** | **236** | **216** |

- **Exclusions at each step:**
  - `build-pairs` excluded 35 questions, every one for evidence leakage: `evaluation_evidence` 27 and
    `cross_split_evidence` 8 (§5);
  - nothing was excluded for a validator re-check, a duplicate or evaluation overlap;
  - the trial candidate build excluded nothing beyond the 9 that the review did not accept.
- **Per passage:** 20 passages contribute no included pair, 45 contribute one and 35 contribute two. Batch 1: 30, 39
  and 31.
- **Batch 1 in this replay:** the scratch `build-pairs` ran over all 200 records.
  - Batch 1's outcome is unchanged: the same 112 pairs and the same 39 exclusions.
  - Every pair reappears with the same passage and question hash, but all 112 pair IDs are renumbered, as batch 1's §12
    expected.
  - The trace of every candidate through `build_pairs`' own functions matches the written pairs, ID for ID.

## 4. Generator acceptance and rejection (validator `sec-qcheck-v3`)

| Reason | Count | Verdict |
|---|---:|---|
| `missing_company` | 16 | All correct: the question does not name the company. One names Sam's Club instead of Walmart. |
| `copied_from_passage` | 14 | 6 copy a run of prose (9–11 words). 7 are borderline: 8-word runs that are mostly dates, line items or names, and one 11-word run that is mostly an Act's name. They are correct by the strict rule. 1 is false (position 5): a cash-flow row label whose `[TABLE]` marker lies before the window, so the label exemption does not apply. |
| `evidence_length` | 5 | All correct by the rule: four quotes of 308–431 characters (limit 300) and one of 14 (minimum 15). |
| `answer_not_supported` | 4 | Positions 27 and 95 are false: the answers are right, but their years are not in the quotes. Position 55 is correct by the rule: the answer adds "4-5-4" from outside the quote. Position 24 is correct in outcome only: the rule fired on the date, but the answer is wrong anyway (retained loans given as total wholesale exposure). |
| `number_not_in_passage` | 2 | Position 78 is correct: a figure cut off by the window. Position 63 is false: the answer is right, but the question's "December 31" is not in the visible text. |

**Totals:**
- 30 correct, one of them (position 24) for the wrong reason;
- 7 correct by the strict rule;
- **4 false rejections**, positions 5, 27, 63 and 95: 2.0% of proposals. Batch 1 had 3 (1.5%).

The four false rejections repeat batch 1's three patterns:
- years missing from the quote (batch 1, position 49);
- a date not in the visible text (position 57);
- a row label outside a `[TABLE]` block (position 60).

There is no new validator weakness. Fixing these mid-P1 would change the generator's identity, so the validator stays
as it is.

| Pos | Source | Reason | Question (verbatim) | Verdict |
|---:|---|---|---|---|
| 5 | TSLA FY2021, narrative, Item 8 | `copied_from_passage` | How much did Tesla, Inc. spend on purchases of property and equipment excluding finance leases, net of sales, in fiscal 2020? | False: the 11-word run is a cash-flow row label, but the table's [TABLE] marker lies before the visible window, so the label exemption does not apply. |
| 10 | JPM FY2025, table, Business Segment & Corporate Results | `missing_company` | What was the pre-provision profit for the Commercial & Investment Bank segment in the year ended December 31, 2023? | Correct: the question does not name the company. |
| 12 | TSLA FY2023, table, Item 8 | `copied_from_passage` | How many shares were reserved and available for issuance under Tesla's 2019 Equity Incentive Plan as of December 31, 2023? | Correct by the rule, borderline: an 8-word run, mostly the plan's wording (the same question as batch 1, position 12). |
| 13 | WMT FY2022, table, Item 7A | `evidence_length` | According to the financial instrument table, what were the expected maturity date principal cash flows for Walmart's short-term borrowings due in fiscal 2023, and what was the associated weighted-average interest rate? | Correct by the rule: the quote has 431 characters (limit 300). |
| 15 | AAPL FY2021, table, Item 8 | `missing_company` | According to the rollforward table, what was the amount of the increase in gross unrecognized tax benefits related to tax positions taken during the current year for fiscal 2021? | Correct: the question does not name the company. |
| 16 | AMZN FY2025, narrative, Item 9B | `missing_company` | What was the expiration date of the trading plan that Brian T. Olsavsky terminated in November 2025? | Correct: the question does not name the company. |
| 17 | JPM FY2024, narrative, Credit And Investment Risk Management | `evidence_length` | What specific activities expose JPMorgan Chase to credit risk in its wholesale business operations during fiscal 2024? | Correct by the rule: the quote has 318 characters (limit 300). |
| 18 | PFE FY2024, narrative, Item 1 | `copied_from_passage` | What specific legislative review in the EU is cited as a potential factor that could reduce the basic period of regulatory data protection for Pfizer in fiscal 2024? | Correct by the rule, borderline: an 8-word run naming the legislative effect. |
| 21 | XOM FY2025, table, Business Results | `missing_company` | What was the record annual production level achieved in Guyana in fiscal 2025? | Correct: the question does not name the company. |
| 21 | XOM FY2025, table, Business Results | `missing_company` | According to the Upstream Financial Results table, what is the first data column for the Earnings (loss) line item? | Correct: the question does not name the company. |
| 24 | JPM FY2023, table, Credit And Investment Risk Management | `answer_not_supported` | What was the total wholesale credit exposure for JPMorgan Chase & Co. as of December 31, 2023, and how much of that total was rated as investment grade? | Correct outcome, wrong reason: the rule fired on the date, but $672,472 million is the retained-loans row, not total wholesale credit exposure. |
| 24 | JPM FY2023, table, Credit And Investment Risk Management | `copied_from_passage` | What specific credit exposure amount did JPMorgan Chase & Co. include associated with First Republic in its wholesale credit portfolio as of December 31, 2023? | Correct by the rule, borderline: an 8-word run, half of it the date. |
| 26 | TSLA FY2020, table, Item 8 | `copied_from_passage` | What was the gross cost of Tesla's operating lease vehicles as of December 31, 2020? | Correct by the rule, borderline: an 8-word run, a line item and the date. |
| 27 | WMT FY2020, table, Item 7 | `answer_not_supported` | What were the calendar comparable sales and fuel impact percentages for Walmart U.S. in fiscal 2020 and fiscal 2019? | False: the answer is right (2.9%, 3.7%, 0.0%, 0.1%); its years are not in the quote. |
| 27 | WMT FY2020, table, Item 7 | `missing_company` | How much did eCommerce sales contribute to Sam's Club comparable sales in fiscal 2020 and fiscal 2019? | Correct: the question names Sam's Club, not Walmart. |
| 31 | JPM FY2021, table, Capital Risk Management | `copied_from_passage` | How much tentative net capital is J.P. Morgan Securities required to hold in excess of, following its registration as a security-based swap dealer effective November 1, 2021? | Correct: copies a 10-word run of the sentence. |
| 32 | PFE FY2025, narrative, Item 7 | `missing_company` | What was the status of the U.S. accelerated approval for Braftovi in combination with cetuximab and fluorouracil-based chemotherapy as of February 2026? | Correct: the question does not name the company. |
| 37 | AMZN FY2019, narrative, Item 9B | `copied_from_passage` | Did Amazon state that it could accurately calculate the net profit for the orders processed for individuals and entities covered by the Iran Threat Reduction and Syria Human Rights Act? | Correct by the rule, borderline: an 11-word run, mostly the Act's name. |
| 41 | WMT FY2025, narrative, Item 1A | `copied_from_passage` | How does Walmart describe the potential impact of the use of remote work infrastructure in recent years on its cybersecurity posture in fiscal 2025? | Correct: copies a 9-word run of the sentence. |
| 48 | WMT FY2026, narrative, Item 7 | `evidence_length` | What specific pricing strategies did Walmart identify as influencing its net sales and gross profit margin in response to cost increases? | Correct by the rule: the quote has 344 characters (limit 300). |
| 49 | XOM FY2019, narrative, Item 2 | `missing_company` | What is the maximum term for OMLs granted under the 1969 Petroleum Act in Nigeria, and how can they be renewed? | Correct: the question does not name the company. |
| 49 | XOM FY2019, narrative, Item 2 | `missing_company` | What is the extended end date for the production sharing agreement regarding the Azeri-Chirag-Gunashli field in Azerbaijan? | Correct: the question does not name the company. |
| 52 | JPM FY2022, narrative, Operational Risk Management | `copied_from_passage` | What specific environmental condition did JPMorgan Chase & Co. experience during the early stages of the COVID-19 pandemic that required greater judgment in adjusting model outputs? | Correct: copies a 9-word run of the sentence. |
| 53 | PFE FY2019, table, Item 1 | `missing_company` | What were the product revenues for Lyrica in the markets impacted by patent expiration for the year ended December 31, 2019? | Correct: the question does not name the company. |
| 53 | PFE FY2019, table, Item 1 | `copied_from_passage` | According to Pfizer's disclosure, when is the basic product patent for Chantix in the U.S. expected to expire? | Correct: copies a 10-word run of the footnote. |
| 55 | WMT FY2021, table, Item 7 | `answer_not_supported` | How does Walmart define comparable sales in its fiscal calendar versus the retail calendar used in quarterly releases? | Correct by the rule: the answer adds '4-5-4', which is in the passage but not in the quote. |
| 57 | AAPL FY2022, table, Item 8 | `evidence_length` | What was the total amount of principal payments scheduled for Apple's term debt in fiscal 2023? | Correct by the rule: the quote has 14 characters (minimum 15). |
| 63 | XOM FY2024, table, Consolidated Statement Of Comprehensive | `number_not_in_passage` | How much was the foreign exchange translation adjustment for Exxon Mobil in the fiscal year ended December 31, 2023? | False: the answer is right (1,241); the question's 'December 31' is not in the visible text, whose columns read 2023. |
| 70 | XOM FY2022, narrative, Item 2 | `missing_company` | What was the initial term duration for Petroleum Prospecting Licenses under the Petroleum Industry Act enacted in 2021? | Correct: the question does not name the company. |
| 73 | JPM FY2023, narrative, Business Segment Results | `copied_from_passage` | What impact did the higher interest rate environment have on JPMorgan Chase & Co's cost of funds for loans and Markets activities during the period ended December 31, 2023? | Correct by the rule, borderline: an 8-word run naming the activities. |
| 75 | TSLA FY2024, narrative, Item 8 | `missing_company` | What is the effective date for annual periods for the new crypto assets standard issued by the FASB in December 2023? | Correct: the question does not name the company. |
| 76 | WMT FY2020, table, Item 7 | `copied_from_passage` | How much of the cash and cash equivalents held by Walmart as of January 31, 2020, could only be accessed through dividends or intercompany financing arrangements subject to Flipkart minority shareholder approval? | Correct: copies an 11-word run of the sentence. |
| 78 | AAPL FY2024, table, Item 8 | `number_not_in_passage` | How much were Apple's total deferred tax assets, net, as of September 28, 2024? | Correct: $26,007 million is not in the visible text (the row is cut off; it is the difference of two figures). |
| 79 | AMZN FY2023, narrative, Item 9B | `missing_company` | What was the end date of the trading plan adopted by Andrew Jassy in November 2023? | Correct: the question does not name the company. |
| 82 | TSLA FY2020, table, Item 8 | `missing_company` | How many shares were reserved and available for issuance under the 2019 Equity Incentive Plan at the end of fiscal 2020? | Correct: the question does not name the company. |
| 93 | AMZN FY2025, table, Item 8 | `evidence_length` | According to Amazon's financial statements, what was the primary driver of the reclassification adjustments for gains on available-for-sale debt securities in fiscal 2025? | Correct by the rule: the quote has 308 characters (limit 300). |
| 95 | PFE FY2020, table, Item 7 | `answer_not_supported` | What was Pfizer's research and development expense in fiscal 2020, and how did it change compared to fiscal 2019? | False: the answer is right ($9,405 million, up 12% from $8,394 million); its years are not in the quote. |
| 97 | WMT FY2026, narrative, Item 1A | `copied_from_passage` | According to Walmart's fiscal 2026 risk factors, who bears the majority of the costs associated with adaptation and losses from physical damage or business interruption? | Correct: copies a 9-word run of the sentence. |
| 98 | XOM FY2019, narrative, Supplemental Information On Oil And Gas | `copied_from_passage` | What was Exxon Mobil's standardized measure of discounted future net cash flows for the United States region as of December 31, 2019? | Correct by the rule, borderline: 'standardized measure of discounted future net cash flows' is a standard term not on the exempt list. |
| 99 | PFE FY2019, narrative, Item 1A | `missing_company` | How many molecules of drugs were included in the pilot project for centralized volume-based procurement that China's government initiated in 2018? | Correct: the question does not name the company. |
| 99 | PFE FY2019, narrative, Item 1A | `missing_company` | In which month and year did the Chinese government begin the nationwide expansion of the volume-based procurement pilot? | Correct: the question does not name the company. |

## 5. Pair building: evidence-leakage exclusions (35)

`build-pairs` excludes a question whose evidence, quote or labelled figure is held exactly by:
- a blocked passage: `evaluation_evidence`;
- the visible text of a passage in another split: `cross_split_evidence`.

These rules fail closed by design, and every exclusion below follows them.

| Where the evidence is | Batch 2 | Batch 1 |
|---|---:|---:|
| An evaluation-pool or audit passage | 6 | 6 |
| An overlap-group copy of an evaluation passage | 14 | 11 |
| Only passages blocked for reusing evaluation text | 7 | 16 |
| The visible text of a validation passage | 8 | 6 |
| **All** | **35** (22.0% of 159) | **39** (25.8% of 151) |

- **Amazon:** the rule took 11 of Amazon's 24 questions. Its balance-sheet, cash-flow and lease tables repeat across
  filings, and copies sit in evaluation pools and in validation passages.
- **Walmart:** the rule took 8 of 22; the other companies lost 1–5.
- **The conservative evidence-reuse exclusions** fell from 16 to 7. The rule is unchanged, as in batch 1 (decision E5).
- **Cross-split:** the 8 validation-passage holders keep the training set and `validation_pairs_v4` apart.

| Pos | Source | Question (verbatim) | Answer (verbatim) | Rule | Where the evidence is |
|---:|---|---|---|---|---|
| 5 | TSLA FY2021, narrative, Item 8 | What was the net cash provided by operating activities for Tesla, Inc. for the fiscal year ended December 31, 2021? | 11,497 | `evaluation_evidence` | an evaluation-pool or audit passage (3 blocked) |
| 9 | AMZN FY2024, narrative, Item 8 | How much long-term debt did Amazon report at the end of fiscal 2023? | $58,314 | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 10 | JPM FY2025, table, Business Segment & Corporate Results | What was JPMorgan Chase & Co's total net revenue for Consumer & Community Banking in fiscal 2025? | $76,029 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 20 | WMT FY2023, narrative, Item 7 | What specific risks does Walmart identify regarding its access to commercial paper markets in fiscal 2023, and how might this affect its financing costs? | The company may not be able to access commercial paper markets with historical flexibility, potentially requiring reliance on more expensive types of debt financing. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (7 blocked) |
| 20 | WMT FY2023, narrative, Item 7 | Which specific matters are disclosed in the Consolidated Financial Statements for Walmart's fiscal 2023 regarding legal contingencies and indemnification obligations? | The company discusses Prescription Opiate Litigation, the Settlement Framework, Asda Equal Value Claims indemnification, Money Transfer Agent Services Matters, and related government investigations. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 22 | AAPL FY2023, table, Item 8 | What were the carrying amounts of Apple's current and non-current term debt included in fair value hedges for fiscal 2023 and fiscal 2022? | $18,247 million for fiscal 2023 and $18,739 million for fiscal 2022 | `cross_split_evidence` | the visible text of 1 validation passage |
| 23 | AMZN FY2020, table, Item 8 | What was Amazon.com, Inc.'s total stockholders' equity at the end of fiscal 2020? | $93,404 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 23 | AMZN FY2020, table, Item 8 | How much long-term debt did Amazon.com, Inc. report as of December 31, 2020? | $31,816 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 26 | TSLA FY2020, table, Item 8 | What is the estimated useful life for solar energy systems in service according to Tesla's depreciation policy? | 30 to 35 years | `evaluation_evidence` | an overlap-group copy of an evaluation passage (3 blocked) |
| 30 | AMZN FY2023, table, Item 8 | What was the total lease cost for Amazon.com Inc for the fiscal year ended December 31, 2023? | $18,918 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 33 | TSLA FY2024, narrative, Item 1 | What was Tesla's total worldwide employee headcount as of December 31, 2024? | 125,665 | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 34 | WMT FY2024, table, Item 7 | How much net cash did Walmart use in financing activities during the fiscal year ended January 31, 2024? | Walmart used $13,414 million in financing activities in fiscal 2024. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 42 | XOM FY2020, narrative, Market Risks, Inflation And Other Uncert | How does ExxonMobil manage its exposure to currency exchange rate risk on cash flows from sales, expenses, financing, and investment transactions? | The Corporation makes limited use of currency exchange contracts to mitigate the impact of changes in currency values, and exposures related to the Corporation's use of these contracts are not material. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (7 blocked) |
| 43 | AAPL FY2019, table, Item 8 | What was the total fair value of Apple's restricted cash, cash equivalents, and marketable securities as of September 28, 2019? | $18.9 billion | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 45 | JPM FY2019, table, Consolidated Results Of Operations | What was JPMorgan Chase's total net revenue for the fiscal year ended December 31, 2019, in millions? | $115,627 million | `evaluation_evidence` | an evaluation-pool or audit passage (2 blocked) |
| 50 | AAPL FY2020, table, Item 8 | What were Apple's total net sales in the U.S. and China respectively for fiscal 2020, in millions? | U.S.: $109,197 million; China: $40,308 million | `evaluation_evidence` | an evaluation-pool or audit passage (3 blocked; also 1 validation) |
| 51 | AMZN FY2025, table, Item 8 | What was Amazon's total stockholders' equity as of December 31, 2025, in millions? | $411,065 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 51 | AMZN FY2025, table, Item 8 | How much did Amazon report in long-term debt for the fiscal year ended December 31, 2025? | $65,648 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 58 | AMZN FY2020, table, Item 8 | What was the net cash provided by or used in financing activities for Amazon.com, Inc. for the fiscal year ended December 31, 2020, in millions? | ($1,104) million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (8 blocked) |
| 58 | AMZN FY2020, table, Item 8 | How much did Amazon.com, Inc. report in total net sales for the fiscal year ended December 31, 2020, in millions? | $386,064 million | `evaluation_evidence` | an evaluation-pool or audit passage (3 blocked; also 1 validation) |
| 61 | TSLA FY2023, narrative, Item 8 | What was the net cash used in investing activities for Tesla in the year ended December 31, 2023? | (15,584) | `evaluation_evidence` | only passages blocked for reusing evaluation text (3 blocked) |
| 62 | WMT FY2022, table, Item 7 | What was the net cash used in investing activities for Walmart Inc. in fiscal 2022, and how did it compare to fiscal 2021? | Net cash used in investing activities was $6,015 million in fiscal 2022 and $10,071 million in fiscal 2021. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 62 | WMT FY2022, table, Item 7 | How much cash held by Walmart outside the U.S. was not freely transferable to the U.S. as of January 31, 2022, due to local laws or restrictions? | $4.3 billion | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 63 | XOM FY2024, table, Consolidated Statement Of Comprehensive | What was the net income including noncontrolling interests reported by Exxon Mobil in fiscal 2024? | 35,063 millions | `cross_split_evidence` | the visible text of 3 validation passages |
| 65 | AMZN FY2023, narrative, Item 8 | How much long-term debt did Amazon have outstanding at the end of fiscal 2023 compared to the prior year? | $58,314 million | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 68 | TSLA FY2022, table, Item 8 | How many years had PricewaterhouseCoopers LLP served as Tesla's auditor as of the report date? | Since 2005 | `cross_split_evidence` | the visible text of 1 validation passage |
| 74 | PFE FY2024, narrative, Item 8 | What was the total amount of legally binding and enforceable commitments Pfizer had as of December 31, 2024? | $4.1 billion | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 80 | JPM FY2021, table, Business Segment Results | What was JPMorgan Chase's total Commercial Banking net revenue for the year ended December 31, 2021? | $10,008 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 80 | JPM FY2021, table, Business Segment Results | How much did JPMorgan Chase's Corporate Client Banking revenue change in fiscal 2021 compared to fiscal 2020? | Increased from $3,203 million to $3,508 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 84 | XOM FY2021, narrative, Item 2 | What was Exxon Mobil's total consolidated crude oil production in thousands of barrels daily for fiscal 2021? | 1,304 thousands of barrels daily | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 86 | AMZN FY2020, narrative, Item 8 | How does Amazon account for the fair value of restricted stock units under its stock-based compensation policy? | Recognized as expense over the service period, net of estimated forfeitures, using the accelerated method, based on the number of shares granted and the quoted price of common stock. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (7 blocked) |
| 90 | WMT FY2025, table, Item 7 | According to Walmart Inc.'s MD&A, what was the net consideration paid for the acquisition of VIZIO in fiscal 2025? | $1.9 billion | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 93 | AMZN FY2025, table, Item 8 | What was the total other income, net reported by Amazon for the fiscal year ended December 31, 2025? | $15,229 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 100 | WMT FY2021, table, Item 7 | What were Walmart's total recorded long-term debt obligations as of January 31, 2021, and how much was scheduled to be paid during fiscal 2022? | Total long-term debt was $44,309 million, with $3,115 million due in fiscal 2022. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (3 blocked; also 1 validation) |
| 100 | WMT FY2021, table, Item 7 | How might a downgrade in Walmart's short-term credit ratings affect its access to capital markets? | It could impair the ability to access commercial paper markets with historical flexibility, potentially requiring reliance on more expensive debt financing. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (7 blocked) |

## 6. AI review of the 124 pairs

**How each pair was judged:** as in batch 1, each pair was read against its positive passage's full visible text.
- **Accept:** correct company, period, figure and unit; supported by the quote; a useful financial retrieval
  query. Minor issues are noted but do not exclude.
- **Revise:** a specific error that a reworded question would fix.
- **Reject:** no financial content, or no real answer.

**Precedents, applied here:**
- **From the validation audits and batch 1:**
  - a wrong unit fails;
  - a part given as the whole is revised;
  - a question naming no year is revised when the fact changes between filings, and accepted with a note when it does
    not.
- **New in batch 1:**
  - an executive's age and an executive's previous employer are rejected (pairs 098 and 099);
  - a yes/no question answered by a stub's "None" is rejected (pair 035).

**Corpus look-ups were used for:**
- pair 111: Pfizer's FY2020 income statement;
- pair 121: whether other Pfizer filings name these products;
- pair 208: the header of the block, in the preceding chunk.

**Result:** 115 accept (92.7%), 6 revise, 3 reject.

| Aspect (124 pairs) | Pass | Minor | Fail | n/a |
|---|---:|---:|---:|---:|
| Factual accuracy | 111 | 7 | 6 (002, 032, 077, 111, 208, 232) | – |
| Financial relevance | 112 | 9 | 3 (040, 225, 226) | – |
| Company | 124 | – | – | – |
| Period | 115 | 6 (names no year) | 3 (002, 077, 208) | – |
| Figures (answer figures in the quote) | 91 | – | – | 33 |
| Units | 75 | 16 (no unit) | 0 | 33 |

**Issues noted on accepted pairs** (one pair can carry two):
- **Boilerplate:** 21, mostly risk factors and governance statements repeated yearly.
- **Weak financial relevance:** 9. These are two insider trading plans, a sanctions disclosure, a privacy-law risk
  statement, a collaboration's rights, licence terms, a workforce statistic, an officer appointment and an acquired
  subsidiary's data.
- **Inaccurate wording:** 7.
- **No issue:** 81.

Batch 1: 19, 10, 5 and 67.

**Not accepted (9):**

| Pair | Pos | Decision | Reason |
|---|---:|---|---|
| 002 | 92 | Revise | Wrong period: $78,869 million remains as of September 28, 2019, the end of Apple's fiscal 2019 fourth quarter, not its third. |
| 077 | 87 | Revise | Wrong column: $11,718 million is 2019; 2018 is $12,260 million. The table is also the CIB segment's international metrics, not the Firm's. |
| 111 | 46 | Revise | Wrong sign: (5,435) is the non-GAAP reconciliation's deduction. Pfizer reported $5,435 million of income from discontinued operations for 2019 (its FY2020 income statement, in the corpus). |
| 208 | 98 | Revise | Wrong period: the block is "As of December 31, 2018". Its header is in the preceding chunk, and the 2019 block lies past the window. |
| 225 | 56 | Revise | Asks ExxonMobil's CFO's age with her identity; the age has no financial content. |
| 232 | 77 | Revise | Asks for "gross net" productive wells: 15,760 is the net count; the gross count is 25,816. |
| 032 | 8 | Reject | The "answer" is the model's reasoning: it says the total is cut off, then proposes a different question. |
| 040 | 85 | Reject | A yes/no question answered by the boilerplate "None" of a 569-character Item 9B stub. |
| 226 | 56 | Reject | An executive's previous employer and role. |

**Patterns:**
- **Tables:** 3 misreadings and 1 ambiguous measure.
  - 077 reads the wrong column.
  - 111 takes a reconciling deduction for the reported figure.
  - 208 takes a block for the filing year when its header lies before the window.
  - 232 merges two headers of a flattened table ("gross" and "net").
- **The window's edge:** 2 errors come from text the generator could not see.
  - 032 asks for a total past the window. Batch 1's position 82 gave a similar reply, which the validator rejected by
    chance for its quote length.
  - 208's header lies before the window.
- **Reasoning in the answer field:** pair 032 is the first such answer to pass the validator. Its only number, 2024,
  is in the quote, so no rule fires.
  - A scan of all 398 proposals of both batches for this language found only these 2.
  - The review catches it.
  - A validator rule for it would change the generator's identity, so it belongs after P1.
- **Period:** 3 errors (002, 077, 208), as in batch 1.
- **Low-value sources:** 3 non-accepts, against 5 in batch 1.
  - ExxonMobil's executive-officer table (position 56, like batch 1's 97) gave one revise and one reject.
  - Apple's Item 9B stub (85) gave one reject.

Every pair, verbatim (pair numbers are those of the 200-record pair set; positions are batch 2's plan positions):

| Pair | Pos | Source | Type | Question (verbatim) | Answer (verbatim) | Decision | Note |
|---:|---:|---|---|---|---|---|---|
| 001 | 92 | AAPL FY2019, table, Item 5 | financial_table | How many shares of Apple Inc. common stock were purchased under publicly announced plans or programs during the period from June 30, 2019 to August 3, 2019? | 23,860 thousand shares | Accept | Correct; shares in thousands, as the header says. |
| 002 | 92 | AAPL FY2019, table, Item 5 | financial_table | What was the approximate dollar value of shares that could still be purchased under Apple's plans or programs as of the end of the fiscal 2019 third quarter? | $78,869 million | **Revise** | Wrong period: $78,869 million remains as of September 28, 2019, the end of fiscal 2019's fourth quarter (the fiscal year-end), not its third. |
| 003 | 43 | AAPL FY2019, table, Item 8 | financial_table | In fiscal 2018, what was the adjusted cost of Apple's mortgage- and asset-backed securities? | 18,105 | Accept | Correct prior-year column (2018); no unit (millions). |
| 006 | 1 | AAPL FY2020, table, Item 5 | other | What was the total value of shares delivered under Apple's accelerated share repurchase arrangement that ended in August 2020? | $6.0 billion | Accept | Correct: the $6.0 billion ASR; 64 million shares at an average $94.14 confirm the total. |
| 007 | 1 | AAPL FY2020, table, Item 5 | financial_table | According to the cumulative total shareholder return data, what was the value of an initial $100 investment in Apple Inc. common stock as of September 2020? | $424 | Accept | Correct; the cumulative-return table's September 2020 value. |
| 010 | 50 | AAPL FY2020, table, Item 8 | financial_table | What was the total amount of long-lived assets Apple held in the U.S. as of September 26, 2020, in millions? | $25,890 million | Accept | Correct. |
| 014 | 15 | AAPL FY2021, table, Item 8 | tax | What was Apple's total net deferred tax asset as of September 25, 2021, and how did it compare to the prior year? | $13,073 million in fiscal 2021, compared to $8,157 million in fiscal 2020 | Accept | Correct; both years. |
| 015 | 64 | AAPL FY2021, table, Item 8 | debt_liquidity | What was the total term debt of Apple Inc. as of September 25, 2021, in millions? | $118,063 million | Accept | Correct. |
| 016 | 64 | AAPL FY2021, table, Item 8 | financial_table | How many shares of Apple Inc. common stock were repurchased during fiscal 2021? | 656 million shares | Accept | Correct. |
| 017 | 29 | AAPL FY2022, table, Item 5 | financial_table | What was the total number of shares purchased by Apple Inc. during the period from August 28, 2022 to September 24, 2022, and what was the average price paid per share for those purchases? | 63,813 thousand shares at an average price of $155.59 per share | Accept | Correct; shares in thousands, as the header says. |
| 018 | 29 | AAPL FY2022, table, Item 5 | debt_liquidity | How much of the $405 billion share repurchase program authorized by Apple's Board of Directors had been utilized as of September 24, 2022? | $344.3 billion | Accept | Correct; repurchase authorization used. |
| 020 | 57 | AAPL FY2022, table, Item 8 | financial_table | How many shares of Apple common stock were repurchased during fiscal 2022 under the share repurchase program? | 568,589 thousand shares | Accept | Correct; shares in thousands, as the header says (the text gives 569 million). |
| 023 | 71 | AAPL FY2023, table, Item 5 | other | What was the total value of the up-front payments committed under the new accelerated share repurchase agreements Apple entered into in August 2023? | $5.0 billion | Accept | Correct. |
| 024 | 71 | AAPL FY2023, table, Item 5 | financial_table | According to the stock performance comparison, what was the cumulative total shareholder return for Apple Inc. common stock as of September 2023, assuming a $100 investment in September 2018? | $317 | Accept | Correct; the cumulative-return table's September 2023 value. |
| 027 | 22 | AAPL FY2023, table, Item 8 | mdna | What percentage of Apple's total trade receivables was attributed to third-party cellular network carriers as of September 30, 2023? | 41% | Accept | Correct. |
| 032 | 8 | AAPL FY2024, narrative, Item 8 | financial_table | What was Apple's total shareholders' equity reported for the fiscal year ended September 28, 2024? | The passage lists components of shareholders' equity but the total value is cut off in the provided text; however, the line item 'Total shareholders'' is present. Based strictly on the visible data for the last complete line before the cut, the question cannot be fully answered with a specific total figure from the text provided as it is truncated. However, looking at the instruction 'Ask only what the passage states', the total line is incomplete. Let's select a complete line instead. What was the balance of Common stock and additional paid-in capital for Apple at the end of fiscal 2024? | **Reject** | The answer is the model's reasoning, not an answer: it says the total is cut off, then proposes a different question; the total lies past the window. |
| 033 | 8 | AAPL FY2024, narrative, Item 8 | debt_liquidity | How much term debt did Apple report as a non-current liability as of September 28, 2024? | $85,750 | Accept | Correct; no unit (millions). |
| 034 | 78 | AAPL FY2024, table, Item 8 | tax | What was the impact of the State Aid Decision on Apple's provision for income taxes in fiscal 2024? | $10,246 million | Accept | Correct; the tax reconciliation is in millions. |
| 037 | 36 | AAPL FY2025, narrative, Item 8 | financial_table | What was the fair value of Apple's corporate debt securities in fiscal 2025? | 46,560 | Accept | Correct; no unit (millions). |
| 038 | 36 | AAPL FY2025, narrative, Item 8 | debt_liquidity | How much of Apple's cash and cash equivalents were held in certificates of deposit and time deposits as of fiscal 2025? | 904 | Accept | Correct; no unit (millions). |
| 039 | 85 | AAPL FY2025, narrative, Item 9B | other | Who is assumed the role of Principal Accounting Officer for Apple Inc. following the transition effective January 1, 2026? | Ben Borders | Accept | Correct; an officer appointment, from a 569-character Item 9B stub; ungrammatical ('Who is assumed'). |
| 040 | 85 | AAPL FY2025, narrative, Item 9B | other | Did Apple Inc. disclose any insider trading arrangements in its fiscal 2025 Form 10-K? | None | **Reject** | A yes/no question answered by the boilerplate 'None' of a 569-character Item 9B stub (batch 1 precedent, pair 035). |
| 043 | 72 | AMZN FY2019, table, Item 7A | financial_table | What was the estimated fair value of Amazon's asset-backed securities as of December 31, 2019? | $2,360 million | Accept | Correct. |
| 044 | 72 | AMZN FY2019, table, Item 7A | debt_liquidity | How much debt did Amazon report having as of December 31, 2019, including the current portion? | $24.8 billion | Accept | Correct. |
| 046 | 37 | AMZN FY2019, narrative, Item 9B | risk | What was the approximate value of consumer products delivered by Amazon for individuals potentially acting for Iranian embassies and diplomatic organizations located outside Iran during the period from January 2012 to December 2019? | approximately $13,700 | Accept | Correct; a sanctions disclosure of immaterial value. |
| 047 | 86 | AMZN FY2020, narrative, Item 8 | other | What were the amounts of valuations and adjustments of equity securities for Amazon in fiscal 2019 and fiscal 2020? | $231 million in 2019 and $833 million in 2020 | Accept | Correct; both years. |
| 056 | 2 | AMZN FY2022, table, Item 8 | financial_table | What was the total lease cost for Amazon in fiscal 2022? | $17,157 million | Accept | Correct. |
| 057 | 2 | AMZN FY2022, table, Item 8 | financial_table | What was the weighted-average remaining lease term for Amazon's finance leases as of December 31, 2022? | 10.3 years | Accept | Correct reading of the flattened table (finance leases, December 31, 2022). |
| 059 | 65 | AMZN FY2023, narrative, Item 8 | financial_table | What was Amazon's total assets reported on the balance sheet as of December 31, 2023? | $527,854 million | Accept | Correct. |
| 060 | 30 | AMZN FY2023, table, Item 8 | debt_liquidity | How much was the present value of Amazon's total lease liabilities as of December 31, 2022? | $84,823 million | Accept | Correct prior-year column (December 31, 2022); names its date. |
| 063 | 79 | AMZN FY2023, narrative, Item 9B | other | How many shares of Amazon.com, Inc. common stock did Jeffrey Bezos plan to sell under his Rule 10b5-1(c) trading plan adopted in November 2023? | 50,000,000 shares | Accept | Correct; an executive's Rule 10b5-1 plan (up to 50,000,000 shares). |
| 064 | 9 | AMZN FY2024, narrative, Item 8 | financial_table | What was Amazon's total assets as of December 31, 2024? | $624,894 | Accept | Correct; no unit (millions). |
| 065 | 44 | AMZN FY2024, narrative, Item 8 | other | What was the estimated fair value of Amazon's convertible notes to Anthropic as of December 31, 2024? | Approximately $13.8 billion | Accept | Correct. |
| 066 | 44 | AMZN FY2024, narrative, Item 8 | other | How much did Amazon invest in the third convertible note to Anthropic in Q4 2024? | $1.3 billion | Accept | Correct. |
| 072 | 16 | AMZN FY2025, narrative, Item 9B | other | How many shares of Amazon.com, Inc. common stock did Jeffrey P. Bezos agree to sell under the trading plan adopted in November 2025? | 15,000,000 shares | Accept | Correct; an executive's Rule 10b5-1 plan (up to 15,000,000 shares). |
| 073 | 45 | JPM FY2019, table, Consolidated Results Of Operations | revenue | How much operating lease income was included in JPMorgan Chase's other income for the year ended December 31, 2019? | $5.5 billion | Accept | Correct. |
| 076 | 87 | JPM FY2019, table, Business Segment Results | financial_table | What was JPMorgan Chase & Co's total assets under custody for the year ended December 31, 2019? | $26,831 billion | Accept | Correct; AUC is stated in billions (about $26.8 trillion). |
| 077 | 87 | JPM FY2019, table, Business Segment Results | revenue | How much net revenue did JPMorgan Chase & Co generate from the Europe/Middle East/Africa region in fiscal 2018? | $11,718 million | **Revise** | Wrong column: $11,718 million is 2019; the 2018 figure is $12,260 million. The table is also the CIB segment's international metrics, not the Firm's. |
| 081 | 94 | JPM FY2020, narrative, Firmwide Risk Management | risk | According to JPMorgan Chase & Co's fiscal 2020 Form 10-K, what are the potential quantitative risks the Firm may face, such as effects on earnings or capital? | Reduction in earnings and capital, liquidity outflows, and fines or penalties | Accept | Correct; a risk-governance statement repeated yearly. |
| 082 | 94 | JPM FY2020, narrative, Firmwide Risk Management | other | Who is responsible for overseeing and delegating authorities to LOB CROs and Firmwide Risk Executives at JPMorgan Chase & Co for fiscal 2020? | The Firm's CRO | Accept | Correct; risk governance repeated yearly. |
| 083 | 38 | JPM FY2020, narrative, Firmwide Risk Management | risk | Who is responsible for setting and approving JPMorgan Chase & Co's risk appetite framework for fiscal 2020? | The Firm's CEO, Chief Financial Officer (CFO) and CRO | Accept | Correct; risk governance repeated yearly. |
| 084 | 38 | JPM FY2020, narrative, Firmwide Risk Management | risk | Which board committee serves as the principal oversight body for risk matters at JPMorgan Chase & Co in its fiscal 2020 10-K? | The Board Risk Committee | Accept | Correct; risk governance repeated yearly. |
| 086 | 31 | JPM FY2021, table, Capital Risk Management | financial_table | What was the actual net capital of J.P. Morgan Securities in millions as of December 31, 2021? | $24,581 million | Accept | Correct. |
| 090 | 3 | JPM FY2022, narrative, Capital Risk Management | risk | What is the primary objective of the Capital Risk Management function at JPMorgan Chase & Co for fiscal 2022? | To provide independent oversight of capital risk across the Firm | Accept | Correct; a capital-risk governance statement repeated yearly. |
| 091 | 3 | JPM FY2022, narrative, Capital Risk Management | other | Which department is responsible for capital management at JPMorgan Chase & Co according to its fiscal 2022 filing? | Treasury and CIO | Accept | Correct; capital governance. |
| 092 | 52 | JPM FY2022, narrative, Operational Risk Management | accounting_policy | How does increased uncertainty in model environments affect the process of adjusting model outputs for JPMorgan Chase & Co. in fiscal 2022? | It may necessitate a greater degree of judgment and analytics to inform any adjustments. | Accept | Correct; a model-risk statement from a 559-character stub, repeated yearly. |
| 095 | 73 | JPM FY2023, narrative, Business Segment Results | accounting_policy | How does JPMorgan Chase & Co allocate expense costs for services provided by corporate support units to its business segments during fiscal 2023? | Generally based on the actual cost and use of services provided. | Accept | Correct; an expense-allocation method repeated yearly. |
| 096 | 17 | JPM FY2024, narrative, Credit And Investment Risk Management | risk | What three factors form the basis of JPMorgan Chase's credit risk measurement methodology? | The probability of default of an obligor or counterparty, the loss severity given a default event, and the exposure at default. | Accept | Correct; names no period; a methodology statement repeated yearly. |
| 097 | 66 | JPM FY2024, table, Credit And Investment Risk Management | financial_table | What was the total credit portfolio exposure for JPMorgan Chase & Co at December 31, 2024? | $3,038,506 million | Accept | Correct. |
| 098 | 66 | JPM FY2024, table, Credit And Investment Risk Management | accounting_policy | How does JPMorgan Chase & Co treat mortgage loans that are past due but insured by U.S. government agencies in its nonperforming loan reporting? | They are excluded from nonperforming loans based upon the government guarantee. | Accept | Correct; names no period; a reporting convention repeated yearly. |
| 102 | 59 | JPM FY2025, narrative, Country Risk Management | accounting_policy | How does JPMorgan Chase & CO determine which country to attribute exposure to in its fiscal 2025 country risk management approach? | Based on the country where the largest proportion of the counterparty's assets are located or where the largest proportion of its revenue is derived. | Accept | Correct; a country-risk methodology repeated yearly. |
| 103 | 59 | JPM FY2025, narrative, Country Risk Management | risk | Which types of activities are excluded from JPMorgan Chase & CO's measurement of country exposure in fiscal 2025? | Activities resulting in contingent or indirect exposure, such as providing clearing services or secondary exposure to collateral on securities financing receivables. | Accept | Correct; a country-risk methodology repeated yearly. |
| 106 | 4 | PFE FY2019, narrative, Item 2 | other | How many owned and leased properties did Pfizer hold as of December 31, 2019, and what was their total square footage? | 453 properties amounting to approximately 47 million square feet | Accept | Correct. |
| 107 | 4 | PFE FY2019, narrative, Item 2 | other | What were the primary drivers for the reduction of 45 sites and 6 million square feet in Pfizer's property portfolio during 2019? | Divestment of properties connected to the GSK Consumer Healthcare joint venture and additions from the Array acquisition | Accept | Correct as the filing words it; the Array additions offset the reduction rather than drive it. |
| 109 | 95 | PFE FY2020, table, Item 7 | mdna | According to the MD&A, what were the primary drivers for the increase in Pfizer's cost of sales in 2020 compared to 2019? | The increase was primarily due to increased sales volumes and an increase in royalty expenses resulting from higher sales of related products. | Accept | Correct as far as the window goes; the list of drivers continues past it. |
| 110 | 46 | PFE FY2020, narrative, Item 7 | financial_table | What were the total acquisition-related items, net of tax, reported by Pfizer for the year ended December 31, 2020? | 35 (in millions) | Accept | Correct cell; a non-GAAP reconciling amount whose direction the question does not state; the unit is given as '(in millions)'. |
| 111 | 46 | PFE FY2020, narrative, Item 7 | profitability | How much did Pfizer report as income from discontinued operations, net of tax, for the year ended December 31, 2019? | (5,435) (in millions) | **Revise** | Wrong sign: (5,435) is the deduction in the non-GAAP reconciliation; Pfizer reported income from discontinued operations of $5,435 million for 2019 (its FY2020 income statement). |
| 116 | 25 | PFE FY2021, table, Item 8 | financial_table | What was the reported net income before allocation to noncontrolling interests for Pfizer Inc. for the year ended December 31, 2021? | $22,025 million | Accept | Correct. |
| 117 | 25 | PFE FY2021, table, Item 8 | financial_table | How much depreciation and amortization expense did Pfizer Inc. record in its consolidated statements of cash flows for fiscal year 2020? | $4,681 million | Accept | Correct prior-year column; names its year. |
| 118 | 67 | PFE FY2021, table, Item 8 | financial_table | What was the net cash provided by or used in financing activities for Pfizer Inc. in the year ended December 31, 2021? | (9,816) | Accept | Correct; no unit (millions). |
| 119 | 67 | PFE FY2021, table, Item 8 | tax | How much cash did Pfizer Inc. pay for income taxes during the period ended December 31, 2020? | $3,153 | Accept | Correct prior-year column; names its year; no unit (millions). |
| 120 | 60 | PFE FY2022, narrative, Item 1 | risk | How does Pfizer describe the impact of China's centralized Volume-Based Procurement (VBP) program on the pricing of its off-patent medicines in fiscal 2022? | The program has resulted in further lowering the price of medicines, especially off-patent medicines, and this trend is expected to continue. | Accept | Correct; a China pricing risk statement. |
| 121 | 60 | PFE FY2022, narrative, Item 1 | risk | Which specific Pfizer products are identified as likely to be included in future rounds of China's Volume-Based Procurement bidding process? | Sulperazon and Vfend injectables. | Accept | Correct; names no period, but only the FY2022 filing calls them likely (FY2023 reports them included). |
| 125 | 11 | PFE FY2022, table, Item 7 | debt_liquidity | What were Pfizer's commercial paper and long-term debt ratings and outlooks assigned by Moody's and S&P as of fiscal 2022? | Moody's: P-1 (Short-Term), A1 (Long-Term), Stable Outlook; S&P: A-1+ (Short-Term), A+ (Long-Term), Stable Outlook. | Accept | Correct; the ratings table. |
| 126 | 11 | PFE FY2022, table, Item 7 | mdna | How did Moody's change Pfizer's long-term debt rating and outlook in November 2022? | Moody's increased the long-term debt rating from A2 to A1 and set the outlook to Stable. | Accept | Correct. |
| 129 | 39 | PFE FY2023, table, Item 7 | financial_table | What amount of cash did Pfizer use in investing activities from continuing operations for the year ended December 31, 2023? | $(32,278) million | Accept | Correct. |
| 130 | 39 | PFE FY2023, table, Item 7 | mdna | According to Pfizer's MD&A, what was the primary driver for the change in cash from investing activities in 2023 compared to 2022? | The change was driven mainly by $43.4 billion cash paid in 2023 for the acquisition of Seagen, net of cash acquired. | Accept | Correct; the Seagen payment. |
| 132 | 88 | PFE FY2023, narrative, Item 8 | accounting_policy | How much did Pfizer record for upfront payments to Beam in acquired in-process research and development expenses during the fourth quarter of 2021? | $300 million | Accept | Correct; the Beam upfront payment. |
| 133 | 88 | PFE FY2023, narrative, Item 8 | other | What is the maximum total deal consideration Pfizer could pay Beam if it exercises its opt-in rights for all three targets under their collaboration agreement? | up to $1.35 billion | Accept | Correct. |
| 134 | 18 | PFE FY2024, narrative, Item 1 | risk | How does Pfizer describe the impact of increased enforcement of privacy and data protection laws on potential penalties, as noted in its fiscal 2024 business overview? | Fines and penalties are increasing | Accept | Correct; a privacy-law risk statement rather than Pfizer's finances. |
| 137 | 74 | PFE FY2024, narrative, Item 8 | financial_table | What was the estimated fair value of Pfizer's contingent consideration for acquisitions as of December 31, 2024, and how was it allocated between current and noncurrent liabilities? | $517 million, with $39 million in Other current liabilities and $477 million in Other noncurrent liabilities | Accept | Correct; the total and its split. |
| 138 | 32 | PFE FY2025, narrative, Item 7 | other | Which company holds exclusive commercialization rights for Emblaveo in the U.S. and Canada according to Pfizer's fiscal 2025 10-K? | AbbVie | Accept | Correct; a collaboration's commercialization rights. |
| 140 | 81 | PFE FY2025, table, Item 8 | financial_table | What was the notional amount of outstanding foreign exchange contracts hedging intercompany forecasted inventory sales for Pfizer Inc. as of December 31, 2025? | $5.0 billion | Accept | Correct. |
| 141 | 81 | PFE FY2025, table, Item 8 | financial_table | How much fair value liability did Pfizer Inc. report for interest rate contracts designated as hedging instruments as of December 31, 2024? | $425 million | Accept | Correct prior-year column of the flattened table (liability, December 31, 2024). |
| 142 | 47 | TSLA FY2019, narrative, Item 7 | mdna | How many total vehicles did Tesla deliver in fiscal 2019? | 367,656 | Accept | Correct. |
| 143 | 47 | TSLA FY2019, narrative, Item 7 | mdna | What was the quarter-over-quarter growth rate for Tesla's solar deployments in the second half of 2019? | approximately 48% and 26% | Accept | Correct; the two quarter-over-quarter rates. |
| 146 | 96 | TSLA FY2019, table, Item 8 | financial_table | What was the carrying value of Tesla's Convertible Senior Notes as of December 31, 2019, and how did it compare to their fair value at the same date? | The carrying value was $3,686 million and the fair value was $6,067 million as of December 31, 2019. | Accept | Correct; carrying and fair value at the same date. |
| 147 | 96 | TSLA FY2019, table, Item 8 | financial_table | According to Tesla's fiscal 2019 inventory breakdown, what was the total amount of finished goods inventory reported for the year ended December 31, 2019? | Finished goods inventory was $1,356 million for the year ended December 31, 2019. | Accept | Correct; a year-end balance asked 'for the year'. |
| 152 | 82 | TSLA FY2020, table, Item 8 | financial_table | What was the net investment in sales-type leases for Tesla, Inc. as of December 31, 2020? | $91 million | Accept | Correct. |
| 155 | 54 | TSLA FY2021, table, Item 8 | financial_table | What was the grant date fair value per share for Tesla stock options in fiscal 2021? | $384.07 | Accept | Correct; the Black-Scholes table's 2021 column. |
| 156 | 54 | TSLA FY2021, table, Item 8 | other | How many shares were available for issuance under Tesla's Employee Stock Purchase Program as of December 31, 2021? | 33.8 million shares | Accept | Correct. |
| 160 | 19 | TSLA FY2022, narrative, Item 7 | risk | According to Tesla's fiscal 2022 MDNA, what impact did the rapid and substantial interest rate hikes by central banks in developed countries have on vehicle financing? | It impacted the affordability of vehicle lease and finance arrangements. | Accept | Correct; a macro trend in the fiscal 2022 MD&A. |
| 161 | 19 | TSLA FY2022, narrative, Item 7 | mdna | What operational strategy did Tesla implement in the second half of 2022 to address challenges with vehicle transportation capacity during peak delivery periods? | Tesla transitioned to a more even regional mix of vehicle builds each week. | Accept | Correct. |
| 162 | 68 | TSLA FY2022, table, Item 8 | financial_table | What was Tesla's total current assets in millions of dollars as of December 31, 2022? | $40,917 million | Accept | Correct. |
| 165 | 61 | TSLA FY2023, narrative, Item 8 | debt_liquidity | How much cash did Tesla receive from proceeds of issuances of debt during the year ended December 31, 2023? | 3,931 | Accept | Correct; no unit (millions). |
| 166 | 12 | TSLA FY2023, table, Item 8 | financial_table | What was the weighted average exercise price of Tesla's stock options at the end of the period for the year ended December 31, 2023? | $35.11 | Accept | Correct. |
| 168 | 33 | TSLA FY2024, narrative, Item 1 | other | What percentage of Tesla's managers were promoted from internal non-managerial positions as of the fiscal 2024 report? | 68% | Accept | Correct; a workforce statistic. |
| 169 | 75 | TSLA FY2024, narrative, Item 8 | accounting_policy | When did Tesla adopt ASU No. 2023-07 regarding reportable segment disclosures, and what method of adoption was used? | retrospectively on December 31, 2024 | Accept | Correct. |
| 173 | 89 | TSLA FY2025, narrative, Item 7 | mdna | How many consumer vehicles did Tesla produce and deliver in fiscal 2025? | Produced approximately 1.66 million and delivered approximately 1.64 million consumer vehicles. | Accept | Correct; both figures. |
| 174 | 89 | TSLA FY2025, narrative, Item 7 | risk | Which Tesla business segment is expected to face a relatively larger impact from the current tariff regime in fiscal 2025? | The energy generation and storage business. | Accept | Correct; tariff exposure by business. |
| 175 | 40 | TSLA FY2025, table, Item 8 | financial_table | What was the fair value of Tesla's total digital assets held as of December 31, 2025? | $1,008 million | Accept | Correct. |
| 176 | 40 | TSLA FY2025, table, Item 8 | financial_table | How did the fair value of Tesla's Bitcoin holdings change between December 31, 2024 and December 31, 2025? | It decreased from $1,074 million to $1,007 million | Accept | Correct; both dates. |
| 179 | 76 | WMT FY2020, table, Item 7 | financial_table | What was the total net cash used in investing activities for Walmart Inc. for the fiscal year ended January 31, 2020? | $(9,128) million | Accept | Correct. |
| 181 | 6 | WMT FY2021, narrative, Item 7 | other | How many new associates did Walmart hire during the period described in its fiscal 2021 MD&A? | more than 500,000 | Accept | Correct figure; 'the period described in its fiscal 2021 MD&A' is awkward but names the filing year. |
| 182 | 6 | WMT FY2021, narrative, Item 7 | mdna | What amount of incremental costs did Walmart incur in fiscal 2021 related to operating through the pandemic? | $4.0 billion | Accept | Correct. |
| 183 | 55 | WMT FY2021, table, Item 7 | financial_table | What was the calendar comparable sales percentage for Walmart U.S. with fuel impact in fiscal 2021? | 8.7% | Accept | Correct with-fuel figure (8.7%); 'with fuel impact' could be read as the fuel impact itself, (0.2)%. |
| 189 | 13 | WMT FY2022, table, Item 7A | mdna | What was the net change in the fair value of Walmart's interest rate swaps for fiscal 2022, and what was the primary driver of this change? | The net fair value decreased $0.2 billion, primarily due to fluctuations in market interest rates. | Accept | Correct. |
| 190 | 69 | WMT FY2023, narrative, Item 1A | risk | What specific types of investments does Walmart Inc. cite as part of its strategy that may not adequately allow it to grow its eCommerce business or maintain its market position in fiscal 2023? | eCommerce, technology, talent, supply chain automation, acquisitions, joint ventures, store remodels and other customer initiatives | Accept | Correct; an eCommerce strategy risk factor. |
| 191 | 69 | WMT FY2023, narrative, Item 1A | risk | How does Walmart Inc. describe the potential impact of a greater concentration of eCommerce and online grocery sales on its physical store traffic and subsequent sales in fiscal 2023? | It could reduce store and club traffic, reducing opportunities for cross-store or cross-club sales and materially adversely affecting financial performance | Accept | Correct; an eCommerce strategy risk factor. |
| 194 | 83 | WMT FY2024, narrative, Item 1A | risk | What specific operational failures or security incidents does Walmart identify as potential causes for interference with meeting customer expectations in its fiscal 2024 filing? | Inherent flaws in technology, faulty implementation, or cybersecurity incidents affecting third-party systems or services. | Accept | Correct; a technology risk factor. |
| 195 | 83 | WMT FY2024, narrative, Item 1A | risk | According to Walmart's risk factors for fiscal 2024, what are the potential consequences if the company fails to maintain a seamless, user-friendly, and secure digital shopping experience across its channels? | Competitive disadvantage, loss of eCommerce and other sales, harm to reputation, and material adverse impact on global eCommerce growth and results of operations. | Accept | Correct; a digital-platform risk factor. |
| 197 | 34 | WMT FY2024, table, Item 7 | financial_table | What was Walmart's free cash flow for the fiscal year ended January 31, 2024, and how does it compare to the amount reported for the fiscal year ended January 31, 2023? | Free cash flow was $15,120 million for fiscal 2024 and $11,984 million for fiscal 2023. | Accept | Correct; both years. |
| 198 | 41 | WMT FY2025, narrative, Item 1A | risk | What specific consumer data does Walmart state that VIZIO Holding Corp. collects, as mentioned in the fiscal 2025 form? | Certain television viewing data. | Accept | Correct; an acquired subsidiary's data, from a cybersecurity risk factor. |
| 199 | 90 | WMT FY2025, table, Item 7 | financial_table | What was the net cash used in investing activities for Walmart Inc. in fiscal 2025, as reported in millions? | $(21,379) million | Accept | Correct. |
| 206 | 97 | WMT FY2026, narrative, Item 1A | risk | What specific operational disruptions does Walmart identify as potential consequences of weather or natural events in its fiscal 2026 Form 10-K? | Physical damage or loss of properties, closure of stores, clubs and distribution or fulfillment centers, limited operating hours, workforce shortages, and the inability of customers and associates to reach locations. | Accept | Correct list; the events also include terrorist attacks and violence, named before the window, not only weather. |
| 207 | 48 | WMT FY2026, narrative, Item 7 | tax | How did Walmart characterize the impact of the One Big Beautiful Bill Act on its income tax expense and effective tax rate for fiscal 2026? | The impacts were not material to income tax expense or effective tax rate, though certain provisions decreased cash taxes paid in fiscal 2026. | Accept | Correct; the OBBB Act's tax effect. |
| 208 | 98 | XOM FY2019, narrative, Supplemental Information On Oil And Gas | tax | How much were the future income tax expenses included in the standardized measure of discounted net cash flows for Exxon Mobil's Australia region in fiscal 2019? | 10,499 | **Revise** | Wrong period: this block is 'As of December 31, 2018' (its header is in the preceding chunk); the December 31, 2019 block lies past the window. |
| 209 | 91 | XOM FY2020, table, Business Profile | profitability | What was Exxon Mobil's total upstream earnings after income taxes for fiscal 2020, and how did that compare to the figure for fiscal 2019? | Total upstream earnings after income taxes were negative $20,030 million for fiscal 2020, compared to positive $14,442 million for fiscal 2019. | Accept | Correct; both years, with the sign. |
| 210 | 91 | XOM FY2020, table, Business Profile | financial_table | According to the business profile, what was the return on average capital employed for Exxon Mobil's U.S. downstream segment in 2020? | The return on average capital employed for the U.S. downstream segment was negative 7.4 percent in 2020. | Accept | Correct. |
| 212 | 42 | XOM FY2020, narrative, Market Risks, Inflation And Other Uncert | mdna | What factors does ExxonMobil cite for mitigating the impact of inflation on non-energy costs in recent years? | Cost reductions from efficiency and productivity improvements. | Accept | Correct; names no year; an inflation statement repeated yearly. |
| 215 | 84 | XOM FY2021, narrative, Item 2 | financial_table | How much natural gas liquids (NGL) production did Exxon Mobil report for its equity companies in Asia in fiscal 2020? | 62 thousands of barrels daily | Accept | Correct prior-year column; names its year. |
| 218 | 35 | XOM FY2021, table, 8. Investments, Advances And Long-Term R | financial_table | What was the total of Exxon Mobil's equity method company investments and advances reported for the period ended December 31, 2021, in millions of dollars? | $39,551 million | Accept | Correct. |
| 219 | 35 | XOM FY2021, table, 8. Investments, Advances And Long-Term R | financial_table | What were the net long-term receivables and miscellaneous amounts for Exxon Mobil as of December 31, 2020, after deducting reserves of $6,115 million? | $4,788 million | Accept | Correct prior-year column with its own reserve figure. |
| 220 | 28 | XOM FY2022, narrative, Item 2 | other | What was ExxonMobil's net offshore acreage in Qatar at the end of fiscal 2022? | 80 thousand offshore acres | Accept | Correct. |
| 221 | 28 | XOM FY2022, narrative, Item 2 | other | How many net development wells did ExxonMobil complete in Kazakhstan during the year ended December 31, 2022? | 2.3 net development wells | Accept | Correct. |
| 224 | 70 | XOM FY2022, narrative, Item 2 | other | How long is the production period for Oil Mining Leases granted under the 1969 Petroleum Act, as described in ExxonMobil's 2022 filing? | 20 years | Accept | Correct; licence terms (the quote's maximum term; the passage also gives a 20-year production period). |
| 225 | 56 | XOM FY2023, table, Item 4 | other | Who served as Exxon Mobil Corporation's Chief Financial Officer as of February 28, 2024, and what was their age? | Kathryn A. Mikells, age 58 | **Revise** | The CFO's identity is a governance fact, but the question also asks her age, which has no financial content (batch 1 precedent, pair 098). |
| 226 | 56 | XOM FY2023, table, Item 4 | other | What role did James R. Chapman hold at Dominion Energy, Inc. prior to joining Exxon Mobil in late 2022? | Executive Vice President, Chief Financial Officer and Treasurer | **Reject** | An executive's previous employer and role: no financial content (batch 1 precedent, pair 099). |
| 227 | 14 | XOM FY2023, table, 19. Income And Other Taxes | financial_table | What was the balance of unrecognized tax benefits for Exxon Mobil Corp at December 31, 2023, in millions of dollars? | 3,935 | Accept | Correct; the question states the unit (millions). |
| 228 | 14 | XOM FY2023, table, 19. Income And Other Taxes | tax | What was the net interest expense on income tax reserves for Exxon Mobil Corp for the year 2023? | $60 million | Accept | Correct. |
| 230 | 7 | XOM FY2024, narrative, Note 1. Summary Of Accounting Policies | accounting_policy | How does Exxon Mobil Corp account for the initial reporting of gains and losses on derivatives designated as cash flow hedges in fiscal 2024? | The gain or loss is initially reported as a component of other comprehensive income and subsequently reclassified into earnings in the period that the forecasted transaction affects earnings. | Accept | Correct; a hedge-accounting policy repeated yearly. |
| 231 | 7 | XOM FY2024, narrative, Note 1. Summary Of Accounting Policies | accounting_policy | What valuation method does Exxon Mobil Corp use for crude oil and products inventories, and how are these inventories carried on the balance sheet? | Inventories are carried at the lower of current market value or cost, with cost generally determined under the last-in, first-out method (LIFO). | Accept | Correct; names no period; a stable inventory policy. |
| 232 | 77 | XOM FY2025, table, Item 2 | financial_table | What was ExxonMobil's total gross net productive oil wells in the United States at year-end 2025? | 15,760 | **Revise** | Ambiguous measure: the question asks for 'gross net' wells; 15,760 is the net count, and the gross count is 25,816. |
| 233 | 77 | XOM FY2025, table, Item 2 | other | How much natural gas is ExxonMobil contractually committed to deliver worldwide from 2026 through 2028? | 2.9 trillion cubic feet | Accept | Correct. |

## 7. Evidence leakage and false-negative protections

**Leakage, on the 115 included pairs (all from the trial build, §10):**

| Check | Result |
|---|---|
| Evidence rule, re-checked at the candidate build | 0 leak passages; every included pair passes |
| Lexical overlap with the permitted evaluation questions (word Jaccard; reject at 0.8) | Highest 0.333 (batch 1: 0.346) |
| Semantic screen (`screen-questions --pairs --candidates train_candidates_v2`, in scratch) | 302 questions (the 236 pairs and the 66) against the 10 permitted question texts, `query_id` and text only. **0 flagged at 0.90.** The highest of all, 0.8017, is batch 1's pair 005. Of batch 2's 115: highest 0.7929 (pair 097), 0 at or above 0.80, 2 at or above 0.75, 14 at or above 0.70; median 0.655. |
| The 0.80–0.90 review band (approved) | **No batch 2 question.** Batch 1's pair 005 is still the only one, and batch 1's review of it stands. The threshold was not changed. |
| Cross-split | 8 excluded at `build-pairs`. `load_validation_set(validation_pairs_v4, training=…)` passes against all 282 trial training pairs. |
| Duplicates | None (§8) |

**Evidence within the model's window:**
- **The generator** sees only the tokenizer-visible part of each passage (512 tokens for BGE, special tokens
  included).
- **The validator** located every accepted quote inside that part.
- **The independent re-check:** all 115 included quotes lie verbatim in the positive's visible text (batch 1: 99
  verbatim, 2 by segments or cells).
  - Counted with the special tokens, as the window is, the latest quote ends at the window's last token, 512.
  - 3 end after token 500; batch 1 has 2 on the same count, ending by token 511.
- **The window's limits** still produce questions about text the generator could not see: pairs 032 and 208, and
  rejection 78. The review catches them (§6).

**False negatives:**
- **Equivalents:** 54 of the 115 record exact equivalents (134 passages), and 57 record uncertain ones (159
  passages). The trainer never uses any of them as a negative. Batch 1: 50 (114) and 53 (139).
- **Conflicts:** `NegativeGuard` finds 205 conflicting pairs among the 282 trial pairs, and the planner keeps them in
  different batches.
  - By source: batch 2 with batch 2, 54; batch 1 with batch 2, 39; batch 2 with the 66, 23.
  - 89 of the 115 are in at least one conflict.
  - Two examples: Apple's term-debt lines in the FY2024 and FY2025 balance sheets (pair 033 and `pilot_v3`'s 012), and
    Pfizer's FY2022 rating questions (pairs 125 and 126) with its 2021 sustainability notes (`pilot_v3`'s 033).
- **Batches:** with 282 pairs, conflict-free plans fill better than with 167.

 | Batch size | Batches | Full | Mean fill (batch 1 trial) |
 |---:|---:|---:|---:|
 | 16 | 18 | 17 | 0.979 (0.949) |
 | 32 | 10 | 8 | 0.881 (0.746) |
 | 64 | 7 | 4 | 0.630 (0.522) |

- **An independent probe:** for each included pair, any other included pair's positive was checked for the answer's
  figure (4 or more digits, not a year) wherever the guard would still allow it as a negative.
  - 2 hits, both Tesla's 1,356 (finished goods, pair 147) found in two Apple passages: chance matches in another
    company's passage.
  - No unrecorded equivalent was found.
- **Known gaps remain:** paraphrased evidence, rounded figures and changed row labels.

## 8. Duplicate questions

| Against | Same normalised question | Highest word Jaccard | Cosine ≥ 0.90 (BGE, zero-shot) |
|---|---:|---:|---|
| Within the 115 | 0 | 0.667 | 4 (highest 0.941): Apple's shares repurchased in fiscal 2021 and 2022; Bezos's trading plans in FY2023 and FY2025; Amazon's total assets at 2023 and 2024; Apple's cumulative return at 2020 and 2023 |
| Batch 1's 101 | 0 | 0.667 | 8 (highest 0.924): Amazon lease, Apple repurchase, Pfizer cash-flow and rating, and Walmart cash-flow templates for other years or measures |
| The 66 of `train_candidates_v2` | 0 | 0.652 | 2 (highest 0.907): other Tesla and Amazon balance-sheet lines |
| `validation_pairs_v4` (93) | 0 | 0.765 | 5 (highest 0.953): Tesla's total current assets at 2022 against 2025; Apple's corporate debt securities, fiscal 2025 against 2022 |
| Earlier train generation (v1, v2) | 0 | 0.783 | 4 (highest 0.917): Walmart cash-flow, Amazon lease and Tesla balance-sheet templates; the Jaccard maximum is Pfizer's 2021 financing cash flow against Amazon's 2019 |

- **No question repeats another.** Every cosine and Jaccard match is the same template asked about a different year,
  measure or company, with a different positive.
- **A trend to watch:** there are more same-template neighbours than in batch 1 (4 within the batch against 2; up to
  0.765 Jaccard against validation, against 0.60). They are mostly Apple and Amazon statement lines. As those
  companies' remaining groups run out (§12), the plan draws them from the same sections.

## 9. Company and fiscal-year accuracy and distribution

**Accuracy:**
- **Company:** every question names its positive's company (115 of 115 included, and all 124).
- **Period:**
  - 106 included questions name the filing's fiscal year or the prior year;
  - 3 name only another year, each legitimately: a January 1, 2026 officer transition in the FY2025 filing, the Q4
    2021 Beam payment described in the FY2023 filing, and 2026–2028 delivery commitments;
  - 6 name no year, all for facts that do not change between filings. Pair 121's products appear only in the FY2022
    filing.

  The period errors are the 3 revised pairs (§6).

**Distribution by company:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Planned | 14 | 14 | 14 | 15 | 14 | 15 | 14 |
| Done | 14 | 14 | 14 | 15 | 14 | 15 | 14 |
| Pass validator | 25 | 24 | 21 | 23 | 23 | 22 | 21 |
| Evidence exclusions | 3 | 11 | 4 | 1 | 5 | 8 | 3 |
| Pairs built | 22 | 13 | 17 | 22 | 18 | 14 | 18 |
| **Included** | **19** | **13** | **16** | **21** | **18** | **14** | **14** |
| Batch 1 included | 17 | 17 | 11 | 16 | 15 | 15 | 10 |
| With batch 1 and the 66 (282) | 46 | 38 | 40 | 45 | 44 | 36 | 33 |

**Distribution by fiscal year:**

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Planned | 13 | 15 | 13 | 13 | 15 | 14 | 15 | 2 |
| Included | 13 | 15 | 16 | 19 | 17 | 17 | 16 | 2 |
| Batch 1 included | 15 | 13 | 17 | 15 | 12 | 14 | 13 | 2 |
| With batch 1 and the 66 (282) | 38 | 38 | 42 | 40 | 38 | 42 | 39 | 5 |

- **Companies:** balanced, at 13–21 included each (batch 1: 10–17).
  - Amazon is lowest: the evidence rule took 11 of its 24 questions (§5).
  - Walmart and ExxonMobil are next. Walmart lost 8 questions to the evidence rule; ExxonMobil lost 4 of its 18 pairs
    to the review.
  - Pfizer is highest (21), losing only 2 questions after the validator.
- **The 282 together:** Apple has the largest share, 16.3%, and ExxonMobil the smallest, 11.7%.
- **Years:** balanced: 13–19 per full year in the batch, 38–42 in the 282.
- **Passage kind:** 57 table and 58 narrative pairs (54 and 46 planned passages; batch 1: 43 and 58).
- **Question type:**
  - `financial_table` 42 (batch 1: 29), `other` 21, `risk` 18;
  - `mdna` 11, `accounting_policy` 8, `debt_liquidity` 8;
  - `tax` 5, `revenue` 1, `profitability` 1.

## 10. Provenance and consistency of the included pairs

**The trial build (scratch copy only):**
- **Sources:**
  - `pilot_v3` and `pilot_v2` with their existing decisions;
  - P1's pair set (236 train pairs from batches 1 and 2), with a scratch decisions file. It holds batch 1's 112
    decisions, re-keyed by `passage_id` and `question_sha256` to the 200-record pair IDs, and this batch's 124;
  - the scratch screening record.
- **Result:** 282 candidates (51 + 15 + 101 + 115).
- **Re-keying batch 1** is mechanical: all 112 decisions found their pair by passage and question hash, and the
  committed file was not touched.
- **The 162 validation pairs** that `build-pairs` also builds have no decision and are excluded
  (`no_quality_decision`), as in batch 1.

**Provenance on all 115 included records:** one identical block of 11 fields:
- corpus `ff9dadf3…4004`;
- preparation `bge_base_ft_v3`, `21c2293a…`;
- provider `groq`, model `qwen/qwen3.8-27b`;
- prompt `sec-qgen-v2`, `f42bdc54…`;
- parameters `42d5fe54…`;
- validator `sec-qcheck-v3`;
- origin `build-pairs`.

No field is missing or empty.

**What the manifest binds:**
- the pair files and the pairs manifest;
- the generation run and records;
- the decisions files;
- the screening record (`semantic_overlap`, threshold 0.9).

It is labelled "AI quality review (accept only); not human-validated; not gold".

**Checks run on the trial set:**
- `verify_candidate_provenance` traces all 282 back to their records and decisions.
- `load_candidates`, the trainer's fail-closed loader, loads all 282.
- **The 66 are unchanged:** a rebuild of `train_candidates_v2`'s sources in scratch gives byte-identical files,
  `train_candidates.jsonl` `908d4c79…` and `candidates_log.jsonl` `c8d38f96…`.

## 11. Batch 2 against batch 1

| | Batch 1 | Batch 2 |
|---|---:|---:|
| Done / failed | 99 / 1 | 100 / 0 |
| Tokens per planned passage | 1,483 | 1,467 |
| Pass the validator | 151 of 198 (76.3%) | 159 of 200 (79.5%) |
| False validator rejections | 3 (1.5%) | 4 (2.0%) |
| Evidence exclusions | 39 (25.8%) | 35 (22.0%) |
| Pairs built | 112 | 124 |
| Accept / revise / reject | 101 / 6 / 5 | 115 / 6 / 3 |
| **Acceptance** | **90.2%** | **92.7%** |
| **Included per planned passage** | **1.01** | **1.15** |
| Passages with no included pair | 30 | 20 |
| Factual errors (fail) | 4 | 6 |
| Period errors | 3 | 3 |
| Table misreadings | 3 | 3 (and 1 ambiguous measure) |
| Low-value-source exclusions | 5 | 3 |
| Highest semantic similarity | 0.8017 | 0.7929 |
| Cosine ≥ 0.90 within the batch | 2 | 4 |
| Company range of included pairs | 10–17 | 13–21 |
| HTTP 429 responses | 0 (resume) | 6 |

**Systematic differences:**
- **Higher yield:**
  - the validator passed more questions;
  - the evidence rule excluded fewer, mainly fewer evidence-reuse-only exclusions (7 against 16);
  - the review accepted more;
  - no passage failed.
- **The same error profile:** 6 factual errors against 4, the same 3 period errors, and the same kinds of table
  misreading. With these counts the difference is not meaningful.
- **One new failure mode:** reasoning text in the answer field (pair 032). It is rare (2 of 398 proposals), and the
  review catches it (§6).
- **More table questions:** `financial_table` is 37% of the included, against 29%.
- **More same-template neighbours** (§8). These are not duplicates.
- **The validator's weaknesses are stable:** the same three false-rejection patterns.
- **The source filter's weak spots recur:** Item 9B stubs and the ExxonMobil executive-officer table.
- **Rate limits:** batch 2 met 6 HTTP 429 responses and needed six invocations. The generator handled them cleanly
  (§2).

Nothing in batch 2 suggests a drift in the generator or a quality problem that the gate does not catch.

## 12. Sequencing, capacity and budget for batch 3

**Sequencing (option A, approved):**
- `train_v3_p1_batch2_quality_decisions.jsonl` is bound to the pair set that the current 200 records build. The
  scratch replay confirmed that pair set exactly.
- Batch 3 will append records and renumber the pair IDs again.
- At the end of P1:
  1. the real `build-pairs` runs once;
  2. batch 1's and batch 2's decisions are carried into one new decisions file keyed by the final pair IDs;
  3. the mapping by `passage_id` and `question_sha256` is mechanical (verified here for all 112 of batch 1).
- Batch 3's plan must be written only now, after batch 2's records exist, because `--after-generated` excludes
  generated passages, not other plans.

**Capacity:** a scratch plan with batch 1's and batch 2's exact options, now excluding both batches' groups, finds:
- **Apple:** 16 eligible overlap groups left, spread over all seven years (2019 4, 2020 1, 2021 2, 2022 2, 2023 2, 2024
  1, 2025 4);
- **Amazon:** 5 groups left, all FY2019;
- **The other five companies:** 75 or more each.

| Scratch plan | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 passages (batch 3) | 16 | 5 | 16 | 16 | 16 | 16 | 15 |
| 400 passages (batches 3–6) | 16 | 5 | 76 | 76 | 76 | 76 | 75 |

- **Batch 3:** under the frozen options it takes all of Apple's and Amazon's remaining groups. Its fiscal years spread
  wider than batches 1–2: 2019 18, 2020 11, 2021 14, 2022 16, 2023 12, 2024 12, 2025 15, 2026 2.
- **Batches 4–6** would have no Apple or Amazon passages.
- **All 600 planned passages** under the frozen options would hold Apple 44 (7.3%) and Amazon 33 (5.5%), and 103–106
  (17–18%) for each of the others. No company exceeds the 25% cap.
- **The timing of the decision:**
  - Batch 3's plan is the same whatever is decided for batches 4–6, because all batches are pooled at the end.
  - The decision is therefore needed before batch 4, not before batch 3. This corrects my note when batch 2 was
    prepared.

**Budget:**
- About 740,000 tokens were approved for batches 2–6. Batch 2 used 146,678, so about 593,300 remain.
- At batch 2's 1,467 tokens per passage, batches 3–6 (400 passages) need about 586,700, a margin of about 1%.
- At batch 1's rate (1,483), they need about 593,200, exactly the remainder.

## 13. Recommendation

- **Batch 2 quality gate: PASS** on every approved criterion (§1).
- **Batch 3: GO**, under four conditions:
  1. **One generator:** the same model, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…` and
     plan options. Only `--name` changes, and the plan is written now that batch 2's records exist. The validator
     weaknesses in §4 and §6 are not fixed mid-P1, and the review keeps reading every answer.
  2. **Option A:** no real `build-pairs` or `build-candidates` in `bge_base_ft_v3` until batch 6 is generated and
     gated (§12).
  3. **Rate limits:** run with `--max-requests` set to the remaining passages, keep the console output with `tee`, and
     rerun the same command after each clean stop.
  4. **Stop rule:** batch 3 is replayed and gated on a scratch copy, as here, before batch 4 is planned.
- **The three-seed experiment: still NO-GO**, until B1–B3 of the readiness audit are resolved.

**Your decisions:**

| # | Decision | Recommendation |
|---|---|---|
| F1 | Company balance for batches 4–6 (needed before batch 4's plan): (a) continue under the frozen options, with no Apple or Amazon passages in batches 4–6; (b) stop P1 after batch 3 (about 300 passages, roughly balanced); (c) continue with smaller batches 4–6 of the five other companies | (a) if the planned data size matters more than an even mix: the 25% cap still holds, and the shortfall is a corpus limit, not a parameter. (b) if an even mix matters more. Either way, record the choice before batch 4. |
| F2 | Budget margin for batches 3–6 (about 1% at batch 2's rate) | Under (a), allow a small contingency (about 5%, some 30,000 tokens), or accept that batch 6 may stop short. Under (b) or (c), no change. |
| F3 | Validator and source-filter weaknesses (4 false rejections; reasoning in an answer; Item 9B stubs and executive-officer tables) | Keep them unchanged through P1. Record them for a later validator version. |
| F4 | The semantic review band | Keep the approved 0.90 exclusion and the 0.80–0.90 recorded review. Batch 2 put no question in the band. |

## Appendix: how the results were obtained

Everything below ran in a scratch copy of `data/processed/train/bge_base_ft_v3`. It used a copy of
`configs/train/data_prep_v3.yaml` that differs only in `output_dir`, the venv, and `HF_HUB_OFFLINE=1`.

```bash
python3 scripts/02_generate_queries.py revalidate --config SCRATCH/data_prep_v3_scratch.yaml --split train --dry-run
python3 scripts/02_generate_queries.py build-pairs --config SCRATCH/data_prep_v3_scratch.yaml
python3 scripts/02_generate_queries.py screen-questions --config SCRATCH/data_prep_v3_scratch.yaml \
    --name train_v3_p1_batch2 --pairs --candidates train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_v2check.yaml   # byte-identical to train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_trial.yaml     # pilot_v3, pilot_v2, P1 batches 1-2; semantic_screening
python3 scripts/02_generate_queries.py plan --config SCRATCH/data_prep_v3_scratch.yaml --split train \
    --name cap_after_b2_400 --size 400 --method balanced_years --max-company-share 0.25 --one-per group \
    --exclude-candidate-groups train_candidates_v2 \
    --after-generated configs/train/data_prep_v2.yaml configs/train/data_prep_v1.yaml   # capacity only (also --size 100)
```

**Other scripts:**
- A trace of every candidate through `build_pairs`' own functions. It matches the written pairs, ID for ID.
- The trainer's `load_candidates` and `plan_batches`, `verify_candidate_provenance`, and `load_validation_set` with
  the training pairs.
- The duplicate and false-negative probes, the quote and token-position checks, and a scan of all 398 proposals for
  reasoning text.
- Corpus look-ups for pairs 111, 121 and 208.

No evaluation file was read except the 10 permitted question texts, through the screen's own loader and the lexical
check. No evaluation question text was printed or reviewed, because no batch 2 question reached the review band.
