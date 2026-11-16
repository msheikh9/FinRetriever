# Validation pilot `val_pilot_v2e`, batch 1: audit of the first 50 records

Status: 2026-11-03.
- **What this is:** an AI review (Claude) of the first 50 generation records of plan `val_pilot_v2e` in
  `bge_base_ft_v3`. It is not human annotation, and no decision here is a gold label.
- **What was not done:**
  - no generation record, training pair, candidate or validation set was changed;
  - no validation set was built. The decisions were recorded afterwards, in
    `data/validation_review/val_pilot_v2e_batch1_quality_decisions.jsonl` and `…_outcomes.jsonl`
    ([validation_plan_v2f.md](validation_plan_v2f.md) §2);
  - no API request was sent and nothing was trained.
- **Checksums:** 267 files under `data/`, `configs/` and `results/` were checksummed before and after the
  audit, and none changed.
- **The checks:** every validation-set check (validator, evidence rule, question overlap, duplicates, unit
  isolation) was re-run in memory, as `build-validation` would run it, with nothing written.

## 1. Summary

**Quality is better than in `val_pilot_v1`.**
- **Decisions:** of the 39 questions the generator accepted, the review accepts 36, revises 1 and rejects 2.
  `val_pilot_v1`: 36 of 43 accepted.
- **One clear factual error** (45), against five in `val_pilot_v1`. It gives a segment's net income as
  JPMorgan Chase's, below a cut-off segment heading.
- **Company attribution:** every question that names a company names the right one.
- **Figures:** no number is invented.
- **Headless tables:** the context rules removed the headless-table errors of `val_pilot_v1`.

**Yield is much lower than planned.**
- **The funnel:**
  - the generator accepted 39 of 50 questions (78%);
  - the strict evidence rule leaves 17 (44% of 39);
  - the review leaves **15 usable questions from 50 requests**.
- **Against the plan:** it expected about 28 from these 50 (planner's score) or 24 (band rates).
- **At this rate the whole of `val_pilot_v2e` gives about 81 validation questions with the 16 carried**
  (95% range 63–104), not 120 (§8).
- **The cause:** the isolation score that ordered the plan over-predicts.
  - It predicted 75% of these questions would pass the evidence rule; 44% did.
  - It never counts the answer-figure matches the evidence rule applies, and these alone removed 9 of
    the 22 excluded questions.

**Recommendation:** do not run batch 2 unchanged (§9).
- **Generation itself** (prompt, validator, model) can stay.
- **First, offline:**
  - re-score the 166 passages not yet generated with a predictor that applies the evidence rule's
    figure matching, calibrated on the 82 questions observed so far;
  - re-order or re-plan them in a new plan version;
  - decide whether 120 remains the target.

## 2. Provenance

| Check | Result |
|---|---|
| Plan | `validation.val_pilot_v2e.json` (`63922fc0…`), unchanged. The 50 records are its positions 1–50, in order |
| Run | One bound run, no continuation: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2` (`f42bdc54…`), parameters `42d5fe54…`, validator `sec-qcheck-v3`, one question per passage |
| Records | 50 done, 0 failed, each on its first attempt, no errors. Sent 2026-11-03, 16:37–16:57 UTC (2.5 per minute) |
| Tokens | 66,095: 59,176 prompt and 6,919 completion, 1,322 per request (`val_pilot_v1`: 1,344). Completion 93–304, median 127, against the 768 limit |
| Files | `validation.records.jsonl` `cc1b6ce6…`, `validation.run.json` `34e6b74d…`: read only, unchanged |
| Validator re-run | Reproduces all 50 outcomes (39 accepted, 11 rejected, the same reasons) |

## 3. The funnel

| Stage | Kept | Removed | Why |
|---|---:|---:|---|
| Requests | 50 | | |
| Generator (`sec-qcheck-v3`) | 39 | 11 | `missing_company` 4, `copied_from_passage` 3, `evidence_length` 2, `question_format` 1, `answer_not_supported` 1 |
| Evidence rule (validation: any match outside validation) | 17 | 22 | Possible training evidence 10, evaluation evidence 8, possible evaluation evidence 7, training evidence (exact, visible) 2; some have several |
| Question overlap (evaluation, training, duplicates) | 17 | 0 | Highest Jaccard: 0.385 against an evaluation question, 0.739 against a training question (threshold 0.8) |
| AI review | **15** | 2 | Reject 01 and 15. Revise 45 (already out by evidence) |

**Generator rejections** (11; verdict on each in §4):
- **Correct by the rule** (10):
  - `missing_company`: 27, 42, 43, 46;
  - `copied_from_passage`: 07, 09, 32;
  - `evidence_length`: 13, 17;
  - `question_format`: 23.
- **Errors they stopped:** 17 and 23 would have been wrong (a table's scope, a prior-year figure).
- **False rejection** (1): 26. The answer is right, but its years are not in the quoted row, as
  `val_pilot_v1`'s 46.

**Question types accepted:** `financial_table` 20, `other` 5, `risk` 5, `accounting_policy` 3, `tax` 2,
`mdna` 2, `debt_liquidity` 1, `profitability` 1.

## 4. Per-question decisions

- **Decisions:** ACCEPT, REVISE and REJECT apply to the 39 questions the generator accepted. For the 11
  it rejected, the note gives a verdict on that rejection.
- **"Evidence rule":** "In" means no train or blocked passage holds the quote, exactly or possibly, in its
  full text.
- **Would enter a validation set:** 15 questions, accepted and "In": 03, 04, 10, 12, 14, 19, 28, 29, 30,
  34, 35, 40, 47, 48, 50.

| # | Source | Question (verbatim) | Answer (verbatim) | Decision | Evidence rule | Note |
|---|---|---|---|---|---|---|
| 01 | AAPL FY2023, narrative/other | When does the insider trading plan entered into by Apple's Senior Vice President of Retail expire? | October 15, 2024 | **REJECT** | **In** | A 10b5-1 trading-plan expiry date (Item 9B): administrative, no financial content. |
| 02 | AMZN FY2021, narrative/statements | What was Amazon's total assets balance as of December 31, 2021? | $420,549 | ACCEPT | Out: possible evaluation evidence | Correct; no unit (millions). |
| 03 | JPM FY2022, narrative/risk | According to JPMorgan Chase & Co's fiscal 2022 Form 10-K, what are the primary responsibilities of employees in each Line of Business and Corporate regarding reputation risk? | To consider the reputation of the Firm when deciding whether to offer a new product, engage in a transaction or client relationship, enter a new jurisdiction, initiate a business process or consider any other activity. | ACCEPT | **In** | Correct; reputation-risk governance boilerplate repeated yearly. |
| 04 | PFE FY2022, narrative/business | In fiscal 2022, until when does Pfizer expect regulatory exclusivity for Vynmac in Japan to expire? | March 2029 | ACCEPT | **In** | Correct. |
| 05 | TSLA FY2024, narrative/risk | In fiscal 2024, how did Tesla describe the regulatory requirements for vehicle safety and emissions in countries outside the U.S. compared to the U.S.? | Standards relating to vehicle safety, fuel economy and emissions are often materially different from equivalent requirements in the U.S. | ACCEPT | Out: possible training evidence | Correct; generic regulatory risk factor. |
| 06 | WMT FY2022, narrative/statements | What was the effective income tax rate for Walmart Inc. for the fiscal year ended January 31, 2022, and how does it compare to the rate for the fiscal year ended January 31, 2020? | The effective income tax rate was 25.4% for the fiscal year ended January 31, 2022, and 24.4% for the fiscal year ended January 31, 2020. | ACCEPT | Out: evaluation evidence | Correct; compares FY2022 with FY2020, skipping FY2021. |
| 07 | XOM FY2025, narrative/statements | How many of Exxon Mobil's projects with exploratory well costs capitalized for more than one year as of December 31, 2025, had drilling in the preceding year or activity planned in the next two years? | 10 projects | Generator: `copied_from_passage` | – | Correct by the rule: the question repeats an 8-word phrase of the note. A natural question otherwise. |
| 08 | AAPL FY2025, narrative/mdna | How much did Apple Inc. spend on repurchasing its common stock during fiscal 2025? | $89.3 billion | ACCEPT | Out: evaluation evidence, possible training evidence | Correct. |
| 09 | AMZN FY2024, narrative/other | How many years of work experience on the security team at Amazon does Amazon's chief security officer have? | Over 15 years | Generator: `copied_from_passage` | – | Correct by the rule; also weak relevance (the chief security officer's experience). |
| 10 | JPM FY2021, table/statements | What was the total notional amount of credit default swaps for protection sold by JPMorgan Chase & Co as of December 31, 2021? | $(443,481) million | ACCEPT | **In** | Correct; keeps the table's parentheses for protection sold. |
| 11 | PFE FY2025, table/mdna | What was Pfizer's total revenue for fiscal 2025, and how did its net cash flow from operations change compared to the prior year? | Total revenues were $62.6 billion, and net cash flow from operations was $11.7 billion, representing a decrease of 8% compared to 2024. | ACCEPT | Out: possible evaluation evidence | Correct; pairs each change with its metric; two-part question. |
| 12 | TSLA FY2023, narrative/risk | What specific action might be forced upon Elon Musk by banking institutions if Tesla's common stock price declines substantially, as described in the fiscal 2023 risk factors? | Sell shares of Tesla common stock to satisfy his loan obligations | ACCEPT | **In** | Correct; risk factor repeated yearly (6 validation equivalents). |
| 13 | WMT FY2023, table/statements | What were the components of Walmart's deferred tax assets as of January 31, 2023, and what was the total amount before valuation allowances? | Loss and tax credit carryforwards: $7,690 million; Accrued liabilities: $3,312 million; Share-based compensation: $237 million; Lease obligations: $4,653 million; Other: $839 million; Total deferred tax assets: $16,731 million | Generator: `evidence_length` | – | Correct by the rule: the quote spans six table rows; the question asks for every component. |
| 14 | XOM FY2024, table/business | How many total dry exploratory wells did Exxon Mobil Corp drill in fiscal 2024? | 3 | ACCEPT | **In** | Correct row; the table counts net wells, the question omits 'net'. |
| 15 | AAPL FY2024, narrative/other | What is the scope of Apple Inc.'s insider trading policy regarding personnel coverage in fiscal 2024? | It applies to all Company personnel, including directors, officers, employees, and other covered persons. | **REJECT** | **In** | Insider-trading-policy boilerplate (Item 10); the answer follows from the question (as val_pilot_v1 41). |
| 16 | AMZN FY2023, table/statements | What was the expected future amortization expense for acquired finite-lived intangible assets for the year ended December 31, 2024, as reported by Amazon.com Inc? | $715 million | ACCEPT | Out: possible evaluation evidence | Correct. |
| 17 | JPM FY2023, narrative/mdna | What was the reported net income for the Corporate segment of JPMorgan Chase & Co for the year ended December 31, 2023? | 2,015 | Generator: `evidence_length` | – | Correct by the rule, and it stopped an error: the table's heading is cut off and its total net income is 4,110, so it is a sub-portfolio's segment table, not the Corporate segment's results. |
| 18 | PFE FY2024, narrative/mdna | What were Pfizer's four key strategic priorities for fiscal 2025 as outlined in its 2024 Form 10-K? | 1. Improve R&D productivity with sharpened focus; 2. Expand margins and maximize operational efficiency; 3. Achieve commercial excellence in our key categories; 4. Optimize capital allocation. | ACCEPT | Out: evaluation evidence | Correct. |
| 19 | TSLA FY2022, table/statements | What was the total fair value of Tesla's cash, cash equivalents and short-term investments reported in millions as of December 31, 2021? | $17,707 million | ACCEPT | **In** | Correct prior-year figure. |
| 20 | WMT FY2024, table/statements | What was Walmart's effective income tax rate for the fiscal year ended January 31, 2024? | 25.5% | ACCEPT | Out: evaluation evidence | Correct. |
| 21 | XOM FY2022, narrative/business | How much refining capacity was taken offline due to industry rationalization during the period described in ExxonMobil's fiscal 2022 10-K Business Environment section? | more than 3 million barrels per day | ACCEPT | Out: possible training evidence | Correct industry figure; the period ('the period described in … section') is vague. |
| 22 | AAPL FY2021, table/statements | What was Apple's total segment operating income in millions for fiscal 2021? | $137,006 million | ACCEPT | Out: evaluation evidence | Correct. |
| 23 | AMZN FY2022, table/statements | What was the total estimated fair value of Amazon's cash, cash equivalents, and marketable securities as of December 31, 2022, and what was the cost or amortized cost of the U.S. government and agency securities held in that period? | The total estimated fair value was $96,049 million, and the cost or amortized cost of U.S. government and agency securities was $2,301 million. | Generator: `question_format` | – | Correct by the rule (over 40 words), and it stopped an error: $96,049 is the December 31, 2021 total, given as 2022's. |
| 24 | JPM FY2019, table/risk | What was JPMorgan Chase's total credit exposure for the consumer segment, excluding credit card, as of December 31, 2019? | $386,452 million | ACCEPT | Out: training evidence | Correct. |
| 25 | PFE FY2020, table/mdna | What was Pfizer's reported total revenues for fiscal 2020, and how did the net cash flow from operations change compared to 2019? | Total revenues were $41.9 billion, and net cash flow from operations increased by 14% compared to 2019. | ACCEPT | Out: possible evaluation evidence | Correct; two-part question (same template as 11). |
| 26 | TSLA FY2021, table/mdna | What was Tesla's effective tax rate for the year ended December 31, 2021, and how did it compare to the rate in 2020? | 11% in 2021 compared to 25% in 2020 | Generator: `answer_not_supported` | – | **False rejection**: the answer is right, but its years are not in the quoted row (as val_pilot_v1 46). |
| 27 | WMT FY2020, narrative/other | What amount was the California Air Resources Board seeking in reimbursement for unclaimed deposits related to automotive refrigerant sales in fiscal 2020? | approximately $4.2 million | Generator: `missing_company` | – | Correct: names no company. |
| 28 | XOM FY2023, table/other | How many shares of Exxon Mobil's common stock were remaining available for future issuance under equity compensation plans approved by security holders as of fiscal 2023? | 54,253,587 shares | ACCEPT | **In** | Correct; 'as of fiscal 2023' for December 31, 2023. |
| 29 | AAPL FY2019, table/mdna | What amount of lease-related assets and liabilities did Apple Inc. anticipate recording on its Condensed Consolidated Balance Sheet upon adoption of ASU 2016-02? | approximately $8 billion | ACCEPT | **In** | Correct; names no fiscal year (adoption in fiscal 2020). |
| 30 | AMZN FY2025, narrative/statements | What was the weighted average remaining life of Amazon's long-term contracts as of December 31, 2025? | 4.1 years | ACCEPT | **In** | Correct. |
| 31 | JPM FY2020, narrative/risk | How much total capital did JPMorgan Chase & Co exclude from its capital metrics for the year ended December 31, 2020, due to the CECL capital transition provisions, and what were the specific components of this amount? | The Firm excluded $5.7 billion, which consisted of the $2.7 billion day 1 impact to retained earnings and 25% of the $12.2 billion increase in the allowance for credit losses (excluding allowances on PCD loans). | ACCEPT | Out: evaluation evidence, possible training evidence | Correct; 2.7 + 25% of 12.2 = 5.75, reported as 5.7. |
| 32 | PFE FY2021, narrative/mdna | How much impact did actuarial valuation and other pension and postretirement plan gains or losses have on Pfizer Inc.'s net income attributable to common shareholders in fiscal 2020? | 1,092 | Generator: `copied_from_passage` | – | Correct by the rule: repeats the row label; the answer has no unit or sign. |
| 33 | TSLA FY2020, narrative/risk | Which specific supplier did Tesla identify as manufacturing battery cells at its Gigafactory Nevada and experiencing temporary operational suspensions during the first half of fiscal 2020? | Panasonic | ACCEPT | Out: evaluation evidence | Correct. |
| 34 | WMT FY2021, narrative/mdna | What were the total contractual obligations for Walmart Inc. as of the fiscal 2021 period ended January 31, 2021? | $121,401 | ACCEPT | **In** | Correct; no unit (millions). |
| 35 | XOM FY2021, table/business | How many net dry exploratory wells did Exxon Mobil Corp drill in the United States in fiscal 2021? | 1 | ACCEPT | **In** | Correct (consolidated 1, equity companies none). |
| 36 | AAPL FY2020, table/statements | What was the fair value of Apple's total marketable securities, including cash, as of September 26, 2020? | $191,830 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct figure; 'total marketable securities, including cash' for the total of cash, cash equivalents and marketable securities. |
| 37 | AMZN FY2020, table/statements | What was the total estimated fair value of Amazon's corporate debt securities as of December 31, 2020? | $29,988 | ACCEPT | Out: training evidence | Correct; no unit (millions). |
| 38 | JPM FY2025, table/risk | What was JPMorgan Chase & Co's earnings-at-risk sensitivity in billions of dollars for a parallel shift of +200 basis points in rates as of December 31, 2025? | $3.7 billion | ACCEPT | Out: possible training evidence | Correct. |
| 39 | PFE FY2023, table/statements | What was the fair value of corporate debt securities held in Pfizer's U.S. pension plans as of December 31, 2023? | $3,668 million | ACCEPT | Out: possible training evidence | Correct; U.S. pension plans, total column. |
| 40 | TSLA FY2025, table/statements | What was Tesla's total current assets as of December 31, 2025? | $68,642 million | ACCEPT | **In** | Correct. |
| 41 | WMT FY2026, table/statements | What was Walmart Inc.'s comprehensive income attributable to Walmart in millions of dollars for the fiscal year ended January 31, 2026? | $22,728 million | ACCEPT | Out: possible evaluation evidence | Correct. |
| 42 | XOM FY2019, narrative/risk | In fiscal 2019, what was the planned annual capacity of the Vistamaxx performance polymer unit included in the Baytown Chemical Expansion Project? | 400-thousand-tonnes-per-year | Generator: `missing_company` | – | Correct: names no company. |
| 43 | AAPL FY2024, narrative/other | How many shares of common stock is Jeff Williams authorized to sell under the Rule 10b5-1(c) trading plan entered into in August 2024? | 100,000 shares | Generator: `missing_company` | – | Correct: names no company; also an administrative trading-plan detail. |
| 44 | AMZN FY2019, table/statements | What was Amazon's total assets reported on the balance sheet at the end of fiscal 2019, in millions? | $225,248 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct. |
| 45 | JPM FY2024, narrative/mdna | What was JPMorgan Chase's net income for fiscal 2024 and what was the year-over-year percentage change? | $24.8 billion, up 23% | **REVISE** | Out: possible training evidence | $24.8 billion, up 23%, is the Commercial & Investment Bank's net income, not JPMorgan Chase's: the segment heading is above the visible text. |
| 46 | PFE FY2019, narrative/business | What was the estimated impact on industry revenues over the next decade from the proposed changes to the Patented Medicine Prices Review Board regulations in Canada? | An approximately $26 billion reduction | Generator: `missing_company` | – | Correct: names no company; an industry-wide estimate. |
| 47 | TSLA FY2019, narrative/mdna | What specific reliability concern regarding the Model 3 fleet did Tesla identify in its fiscal 2019 MD&A that could negatively impact servicing capabilities? | Unexpected reliability issues in the rapidly growing Model 3 fleet could outpace and overburden servicing capabilities and parts inventory. | ACCEPT | **In** | Correct. |
| 48 | WMT FY2025, narrative/business | How many distribution facilities did Walmart U.S. utilize for its supply chain in fiscal 2025? | 164 distribution facilities | ACCEPT | **In** | Correct; an operational count. |
| 49 | XOM FY2020, narrative/risk | What was ExxonMobil's upstream production volume in fiscal 2020 and how did government and economic curtailments affect that figure? | Upstream business produced 3.8 million oil-equivalent barrels per day, and curtailments negatively impacted 2020 production by approximately 0.2 million oil-equivalent barrels per day. | ACCEPT | Out: evaluation evidence, possible training evidence | Correct; two-part question; the quote joins two adjacent sentences with '...' (verbatim fragments). |
| 50 | AAPL FY2021, narrative/statements | What was the fair value of Apple's corporate debt securities classified as non-current marketable securities in fiscal 2020? | 58,909 | ACCEPT | **In** | Correct prior-year figure; no unit (millions). |

## 5. Aspects reviewed (39 accepted questions)

| Aspect | Pass | Minor | Fail | n/a | Notes |
|---|---:|---:|---:|---:|---|
| Factual correctness | 36 | 2 | 1 | 0 | Fail: 45 (segment as firm). Minor: 14 omits "net" wells; 36 describes the cash-and-securities total loosely |
| Financial relevance | 34 | 3 | 2 | 0 | Fail: 01 (trading-plan date), 15 (insider-trading policy). Minor: 03 and 05 (boilerplate), 48 (operational count) |
| Company attribution | 39 | 0 | 0 | 0 | Every question names the filer correctly |
| Fiscal year | 34 | 5 | 0 | 0 | Minor: 01, 15 and 29 name no year; 21's period is vague; 28 says "as of fiscal 2023" |
| Figures | 30 | 1 | 1 | 7 | Fail: 45. Minor: 10 keeps the table's parentheses (protection sold) |
| Units | 26 | 4 | 0 | 9 | Minor: 02, 34, 37 and 50 give no unit (millions); none gives a wrong unit |

**Evidence support:**
- **Every quote is verbatim** in the visible text. 49 joins two adjacent sentences with "...", which the
  quote matcher accepts as fragments.
- **Every answer is supported** by its quote. 45's quote is verbatim, but it belongs to a segment.

**Visibility within BGE's 512-token limit:**
- **Where the text comes from:** the visible text is the passage truncated to the tokenizer's window, and
  every quote lies inside it.
- **Where the quotes end:** the latest ends at token 495 of 512 (41), and 7 end after token 430 (06, 24,
  37, 39, 40, 41, 44).
- **None is cut off.**

## 6. Passages that start mid-table or mid-section

All 50 passages passed the context rules (`headless_table_start`, `fragment_opening`): a passage that
starts inside a table names a period in its column headings, and a fragment names its company and
period before its first figure.

| Opening | Passages | Accepted | Errors |
|---|---|---:|---|
| Inside a table (rows before any table marker, or a table's end first) | 02, 06, 07, 13, 14, 17, 20, 22, 23, 32, 34, 35, 36, 37, 50 | 10 | 36 minor; rule-rejected 17 (table scope) and 23 (year) would have been wrong |
| Mid-sentence (lowercase or punctuation first) | 03, 05, 08, 10, 12, 18, 21, 26, 27, 30, 31, 38, 39, 40, 42, 46, 47, 48 | 14 | None |
| Mid-section, at a sentence or table boundary below a cut-off heading | 04, 09, 16, 19, 29, 45, 49 | 6 | **45**: the Commercial & Investment Bank heading is cut off, and its results are read as the firm's |
| At an item, note or statement heading | 01, 11, 15, 24, 25, 28, 33, 41, 43, 44 | 9 | 01 and 15 rejected for relevance, not error |

**What the rules leave:**
- **A table whose headings name the period but not the scope:**
  - 17: a sub-portfolio's segment table, whose total net income is 4,110 (millions);
  - 23: the prior-year column.
- **A section below a cut-off segment heading:** 45.
- **Why the rules miss them:** they check for a period and an entity, not for a segment or table title.

## 7. Duplicates, distribution and leakage

**Duplicates and overlap:**
- **None exact or near-exact.** No two questions share a normalised key. The highest Jaccard within the
  batch is 0.679 (11 and 25).
- **Same template, same company, different year** (correlated, not duplicates):

| Questions | Template | Same linked unit? | Evidence rule |
|---|---|---|---|
| 06, 20 (and rejected 13) | Walmart's effective tax rate note | Yes, `grp_07889`: the unit's cap of 3 is used in batch 1 | 06 and 20 out |
| 11, 25 | Pfizer's financial highlights: revenue and the change in operating cash flow | Yes, `grp_05167` | Both out |
| 36, 50 | Apple's September 2020 securities table: the FY2020 report, and the FY2021 report's prior-year column | Yes, `grp_00155`. 50's answer is also in 36's passage | 36 out, 50 in |
| 14, 35 | ExxonMobil's dry exploratory wells | No | Both in. The carried question v1 #14, on ExxonMobil's productive exploratory wells, uses the same table |
| 02, 44 | Amazon's total assets | No | Both out |

**Distribution:**
- **The batch:** AAPL 8 and every other company 7.
  - Fiscal years: 2019 6, 2020 7, 2021 8, 2022 6, 2023 7, 2024 8, 2025 7, 2026 1.
  - 28 narrative, 22 tables.
- **The 15 that would enter a validation set:**
  - by company: TSLA 4, XOM 3, AAPL 2, JPM 2, WMT 2, AMZN 1, PFE 1;
  - by year: 2021 4, 2022 3, 2025 3, 2019 2, 2023 2, 2024 1, and none for 2020 or 2026;
  - 8 narrative, 7 tables.
- **The 166 passages not yet generated are less even:** JPM 47, XOM 37, PFE 25, TSLA 22, WMT 19, AAPL 8,
  AMZN 8; FY2019 34. The plan's alternation front-loads the small companies.

**Evidence overlap with training, blocked passages and evaluation pools:**
- **The 17 "In" questions:**
  - no train or blocked passage holds the evidence, exactly or possibly, in full text;
  - no positive is used by any training pair, and every positive's overlap group and unit lie wholly in
    validation.
- **The 22 excluded questions:**
  - by training holders only: 7;
  - by blocked holders only: 10;
  - by both: 5.
- **Their blocked holders:**
  - in an evaluation pool itself (annotation, development or audit pool): 8 questions;
  - only linked to a pool, by evidence reuse or its overlap group: 7.
- **Evaluation questions:** highest Jaccard 0.385 (20, Walmart's tax rate). None paraphrases an
  evaluation question.
- **Training questions:**
  - against all 150 ever generated (`bge_base_ft_v1` and `v2` train records, `train_candidates_v2`), the
    highest Jaccard is 0.739;
  - that is 50, Apple's non-current corporate debt for FY2020, against the same table for FY2023.
- **Earlier validation questions:**
  - against `val_pilot_v1`'s 43: 0.75 (41 against v1 #20, Walmart's comprehensive income for FY2021, which
    is not in the validation set);
  - against the 16 carried questions: 0.6 (50).

## 8. Why the yield is low

**The mechanism** (measured on the 22 exclusions):
- **Statement figures** recur as comparative columns in the next year's 10-K, and risk-factor and
  governance text repeats every year.
- **When that adjacent filing is in training or blocked,** the evidence rule removes the question.
- **By kind:** tables pass 7 of 19, narratives 10 of 20. By section: risk 2 of 8, statements 5 of 15, MD&A
  3 of 8, business 4 of 5, other 3 of 3.

**The planner's score misses half of it:**
- **What the score does:** `isolated_evidence_share` locates each visible sentence with an empty answer.
- **What the rule does:** it also searches for the answer's figure in other passages of the company (the
  locator's `value` and `other_fact` matches).
- **Excluded only by such figure matches:** 9 of the 22 (02, 11, 16, 25, 36, 38, 39, 41, 44).
- **The other 13 were found by quote, span or fact matches:** the score samples only sentences of six or
  more words, and table rows often do not qualify.

**Calibration (measured):**

| Isolation share | Accepted questions | Score predicted | Band rate predicted (`val_pilot_v1`) | Observed pass |
|---|---:|---:|---:|---:|
| 0.3–0.5 | 8 | 0.40 | 0.57 | 3 (0.38) |
| 0.5–0.8 | 12 | 0.63 | 0.69 | 3 (0.25) |
| 0.8–1 | 19 | 0.97 | 0.69 | 11 (0.58) |
| All | 39 | 0.75 | 0.66 | 17 (0.44) |

**Projection for the whole plan (an estimate):**
- **At batch 1's rate** of 15 usable per 50 requests (95% 19–44% per request), the 166 remaining passages
  give about 50 (32–73). With the 16 carried and these 15, that is about **81 (63–104)**.
- **On band rates pooled with `val_pilot_v1`:** about 91.
- **Reaching 120** would need about 89 more usable questions: about 300 requests at this rate, while the
  plan has 166 left.

## 9. Systematic weaknesses of Qwen's outputs

1. **Scope below a cut-off heading:** a segment's or sub-portfolio's figures read as the firm's (45;
   rule-rejected 17), and a prior-year column read as the current year (23). This is `val_pilot_v1`'s main
   error, now rarer.
2. **The company name is left out** in 4 of 50 (`val_pilot_v1`: 1 of 50): 27, 42, 43 and 46.
3. **Units are dropped** for tables "in millions" (02, 34, 37, 50).
4. **Two-part questions** combine two metrics (06, 11, 25, 31, 49; 23 was rejected for length). They are
   correct, but less like a single search need.
5. **Administrative and governance questions** come from Item 9B, Item 10 and Item 1C passages: 10b5-1
   trading plans, insider-trading policy and the chief security officer's experience (01, 09, 15, 43).
   None of these 4 passages gives a usable question.
6. **Template repetition** across one company's years (§7), which the unit cap limits only within a unit.
7. **Boilerplate risk factors** repeated every year (03, 05, 12): correct questions with many validation
   equivalents (12 has 6).
8. **Copying row labels** (`copied_from_passage`, 3), and one false `answer_not_supported` (26), as in
   `val_pilot_v1`.

## 10. Comparison with `val_pilot_v1`

| | `val_pilot_v1` (`bge_base_ft_v2`) | Batch 1 of `val_pilot_v2e` (`bge_base_ft_v3`) |
|---|---|---|
| Generator acceptance | 43 / 50 (86%) | 39 / 50 (78%) |
| Main generator rejections | `evidence_length` 3 | `missing_company` 4, `copied_from_passage` 3 |
| Evidence rule pass (of accepted) | 19 / 43 (44%) | 17 / 39 (44%) |
| Review: accepted / revise / reject (of accepted) | 36 / 5 / 2 (84%) | 36 / 1 / 2 (92%) |
| Factual errors | 5 (12%): table scope and period | 1 (3%): a segment's results as the firm's |
| Usable per request | 16 / 50 (32%) | 15 / 50 (30%) |
| Company attribution | All correct | All correct |
| Tokens per request | 1,344 | 1,322 |

**What changed and what did not:**
- **Better:** v3's context rules removed the headless-table errors, and quality improved.
- **Unchanged:** yield. v3's larger validation split did not lower the share removed by the evidence rule
  (44% passing in both). The comparative-figure mechanism works through adjacent filings, which the
  company-tuned split often leaves in training or blocked.

## 11. Recommendation for batch 2

**Do not proceed unchanged.**
- **Quality supports continuing:** the prompt, validator and model need no change.
- **Yield does not:**
  - unchanged, batch 2 is expected to give about 15 usable questions for 50 requests;
  - the plan as a whole about 81, not 120;
  - its remaining passages are concentrated in JPMorgan, ExxonMobil and FY2019.

**Before batch 2, offline and with your approval:**
1. **A fact-level isolation predictor:**
   - locate each visible sentence and table row with its figures as the answer, as the evidence rule
     does;
   - calibrate it on the 82 questions observed so far (`val_pilot_v1` 43, batch 1 39).
2. **A new plan version** for the 166 passages not yet generated (and, if needed, the rest of the pool):
   - ordered by that predictor, keeping E's caps;
   - generation records stay, so nothing done in batch 1 is repeated.
3. **Optionally, a source-quality rule** for administrative passages: Item 9B insider-trading
   arrangements, Item 10 insider-trading policy and Item 1C governance. None of batch 1's 4 such passages
   gave a usable question.
4. **The target:**
   - at the observed rate, 120 needs about 300 more requests, beyond what plan E holds;
   - the choice is between more requests from a re-planned pool and a smaller validation set (about
     80–90);
   - relaxing the evidence rule would weaken the leakage protection, and is not recommended.

**If you prefer to proceed unchanged:** batch 2 is safe to run. It adds questions of the same quality and
more calibration data, but it does not change the projection.
