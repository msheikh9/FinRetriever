# Validation plan `val_pilot_v2f`, batch 3: audit of records 98–143

Status: 2026-11-04.
- **What this is:** an AI review (Claude) of the 46 generation records of batch 3 of plan `val_pilot_v2f` in
  `bge_base_ft_v3`. It covers plan positions 98–143 only. It is not human annotation, and no decision here
  is a gold label.
- **What was not done:**
  - no generation record, plan, training pair, candidate, calibration, validation set or decision file
    was changed;
  - no validation set or validation pair was built, and no decision file was written: the decisions
    below exist only in this report until they are recorded;
  - no API request was sent and nothing was trained.
- **Checksums:** 272 files under `data/`, `configs/` and `results/` were checksummed before and after the
  audit, and none changed. Since batch 2's audit, only `validation.records.jsonl` has changed.
- **The checks:** every validation-set check (validator, evidence rule, question overlap, duplicates, unit
  isolation) was re-run in memory, as `build-validation` would run it, with nothing written.
- **Earlier documents:** [validation_plan_v2f.md](validation_plan_v2f.md) (the plan),
  [validation_plan_v2f_batch1_audit.md](validation_plan_v2f_batch1_audit.md) (positions 1–49) and
  [validation_plan_v2f_batch2_audit.md](validation_plan_v2f_batch2_audit.md) (positions 50–97).

## 1. Summary

**Batch 3 met its estimate:**
- **The funnel:**
  - two executions completed 46 passages (positions 98–143); positions 144–147 of the batch's window
    were not reached;
  - the generator accepted 37 (80%);
  - the evidence rule left 14;
  - the review leaves **11 usable questions**.
- **Against the plan:** it expected 11.7 from these 46 passages.
- **All three batches:** 49 usable from 143 passages, against 51.1 expected.
- **Running total:** 80 usable validation questions (16 carried, 15 from `val_pilot_v2e`, and 18, 20 and
  11 from this plan's three batches).

**Quality: the review caught three period and table errors that every automatic check passed:**
- **99** gives fiscal 2018's row of Apple's AOCI roll-forward as fiscal 2019's. The table lists 2018
  first, and fiscal 2019's row lies past the window.
- **112** answers "—" for ExxonMobil's notes due 2024. The dash means the notes left the long-term debt
  table for current liabilities, not that nothing was owed.
- **98** names no year, and ExxonMobil's chemical-demand forecast changes by filing: the FY2019 filing,
  also in validation, gives a different figure.
- **All three passed the evidence rule,** so the review alone keeps them out. Review acceptance of
  evidence-passing questions fell to 0.79 (11 of 14), against 0.95 in batch 2.
- **Elsewhere:** 2 factual errors in 37 accepted questions (5%), as in batch 2. Company and segment
  attribution is right in all 37, and no answer gives a wrong unit.

**The predictor held up, and is conservative in its middle range:**
- **This batch:** 11.5 evidence passes predicted, 14 observed. AUC 0.83, its best batch.
- **All three batches (113 questions):** 52.6 predicted, 59 observed; AUC 0.79.
- **Its low end is optimistic:** the 20 accepted questions predicted below 0.2 were expected to give 2.2
  passes, and gave none. Half of the remaining plan (108 of 218 passages) scores below 0.2.

**Evidence reuse across filing years is now the main cause of exclusion:** 20 of the 23 excluded
questions have their evidence in another year's filing, usually as that filing's comparative column.

**Recommendation:** run the remaining four passages (144–147) on `val_pilot_v2f` unchanged, with the
review kept, and then stop for a decision on batch 4 (§12).
- **Cost:** 4 requests, about 5,600 tokens and 2 minutes, for about 1.2 usable questions.
- **The total:** the whole plan projects about 114–116 usable questions (a range of about 110–118), or
  about 106 if passages below 0.2 keep yielding nothing.

## 2. Provenance

| Check | Result |
|---|---|
| Plan | `validation.val_pilot_v2f.json` (`a7903bc7…`), unchanged. The 46 records are its positions 98–143, in order. Position 144 (TSLA FY2019) is next |
| Run | The same bound run as batches 1 and 2, no continuation: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, parameters `42d5fe54…`, validator `sec-qcheck-v3`, one question per passage |
| Executions | Two: 42 records 23:16–23:34 UTC (positions 98–139, 2.3 per minute), then, after a 36-minute gap, 4 records 00:10–00:11 UTC (positions 140–143) |
| Records | 46 done, 0 failed, each on its first response attempt, no errors. Records do not log HTTP retries, so the reported three 429s and 50 attempts cannot be checked from them; 46 completions and 3 retries account for 49 |
| Tokens | 64,283: 57,964 prompt and 6,319 completion, 1,397 per passage. Completion 82–322, median 131, against the 768 limit |
| Files | `validation.records.jsonl` is now `be52f9d1…`, 193 lines. Its first 147 lines are byte-identical to the file batch 2 was audited on (`45164891…`), so the run only appended. `validation.run.json` `34e6b74d…` is unchanged |
| Passages | None repeats a passage of `val_pilot_v2e` or of batches 1 and 2 |
| Validator re-run | Reproduces all 46 outcomes (37 accepted, 9 rejected, the same reasons) |
| Predictor re-run | All 46 plan scores recompute exactly from `configs/train/evidence_pass_v1.json` |

## 3. The funnel

| Stage | Kept | Removed | Why |
|---|---:|---:|---|
| Passages completed | 46 | | Positions 98–143, in two executions. 144–147 not reached |
| Generator (`sec-qcheck-v3`) | 37 | 9 | `copied_from_passage` 4, `missing_company` 3, `evidence_length` 2 |
| Evidence rule (validation: any match outside validation) | 14 | 23 | Evaluation evidence 11, possible training evidence 9, training evidence (exact, visible) 7, possible evaluation evidence 6; some have several |
| Question overlap (evaluation, training, duplicates) | 14 | 0 | Highest Jaccard: 0.393 against an evaluation question, 0.762 against a training question (threshold 0.8) |
| AI review | **11** | 3 | Revise 98, 99 and 112 |

**Generator rejections** (9; verdict on each in §4):
- **All correct by the rule. No false rejections;** all nine answers are right.
- **Copied wording (105, 110, 118, 122):** four, against one in batch 2. Each repeats a long phrase of its
  sentence or heading. 110 (Tesla's quarterly crypto-adoption table) and 122 (JPMorgan's pro forma
  table) were the two batch 3 positions flagged for known failure patterns, so the generator removed both.
- **No company named (130, 137, 142):** a product (Eliquis), a counterparty (Viatris) and a regulation
  (the Federal Reserve's rule) instead of the filer.
- **Long quotes (136, 140):** each quotes a whole wide table row (356 and 669 characters, limit 300).

**Question types accepted:** `financial_table` 17, `tax` 6, `debt_liquidity` 5, `accounting_policy` 4,
`other` 4, `risk` 1.

## 4. Per-question decisions

- **Decisions:** ACCEPT, REVISE and REJECT apply to the 37 questions the generator accepted. For the 9 it
  rejected, the note gives a verdict on that rejection.
- **"Evidence rule":** "In" means no train or blocked passage holds the evidence, exactly or possibly, in
  its full text. "Out" lists the reasons. Only an "In" question with ACCEPT is usable.
- **REVISE** means the question would be acceptable reworded. Records are never edited, so a revised
  question is excluded like a rejected one.

| # | Source | Question (verbatim) | Answer (verbatim) | Decision | Evidence rule | Note |
|---|---|---|---|---|---|---|
| 98 | XOM FY2020, narrative/risk | What is ExxonMobil's forecast for the growth in worldwide chemical demand by 2030? | Over 40 percent by 2030 | **REVISE** | **In** | Right for the FY2020 filing, but the question names no year and the forecast changes by year: the FY2019 filing (two validation passages) says about 45 percent by 2030. |
| 99 | AAPL FY2019, table/statements | What was the total amount of other comprehensive income or loss before reclassifications for Apple Inc. for the fiscal year ended September 28, 2019, expressed in millions? | $(4,415) million | **REVISE** | **In** | (4,415) is fiscal 2018's row, between the September 30, 2017 and September 29, 2018 balances; fiscal 2019's is 3,484, past the window. |
| 100 | AMZN FY2024, narrative/statements | What was the net carrying value of Amazon's total finite-lived intangible assets as of December 31, 2024? | $7,439 million | ACCEPT | Out: evaluation evidence, possible training evidence | Correct; the quote joins the row label to its cell. Same template as 121. |
| 101 | JPM FY2022, table/risk | What was the total issuance of senior notes by JPMorgan Chase & Co's Parent Company in the U.S. market for the year ended December 31, 2022, in millions? | $32,600 million | ACCEPT | Out: possible training evidence | Correct: the Parent Company's 2022 column. |
| 102 | PFE FY2024, table/statements | What was the net value of Pfizer's property, plant and equipment as of December 31, 2024, in millions? | $18,393 million | ACCEPT | Out: training evidence | Correct. |
| 103 | TSLA FY2021, table/mdna | What was the reported Research and Development expense for Tesla, Inc. for the year ended December 31, 2021, in millions of dollars? | $2,593 million | ACCEPT | Out: evaluation evidence | Correct. |
| 104 | WMT FY2023, table/mdna | What was the operating income for Walmart U.S. Segment in fiscal 2023, and how does it compare to fiscal 2022? | Operating income was $20,620 million for fiscal 2023, compared to $21,587 million for fiscal 2022. | ACCEPT | Out: evaluation evidence | Correct; the Walmart U.S. segment. Same table as 139, a year later. |
| 105 | XOM FY2024, table/statements | What was the amount of Exxon Mobil's Euro-denominated debt designated as a net investment hedge of its European business as of December 31, 2024? | $3.1 billion | Generator: `copied_from_passage` | – | Correct by the rule: copies the sentence (Euro-denominated debt, net investment hedge of its European business). |
| 106 | AAPL FY2020, narrative/statements | How does Apple account for the time value of options designated as cash flow hedges in fiscal 2020? | It is excluded from the assessment of hedge effectiveness and recognized in the financial statement line item to which the hedge relates on a straight-line basis over the life of the hedge. | ACCEPT | Out: possible evaluation evidence | Correct; a hedge-accounting policy sentence repeated in Apple's later filings. |
| 107 | AMZN FY2023, table/statements | What was Amazon's total net property and equipment as of December 31, 2023? | $204,177 million | ACCEPT | Out: evaluation evidence, training evidence | Correct; the table lists 2022 before 2023. |
| 108 | JPM FY2019, narrative/statements | How does JPMorgan Chase account for fees received or paid related to securities financing agreements in its financial statements for fiscal 2019? | Fees are recorded over the life of the agreement in interest income and interest expense on the Consolidated statements of income. | ACCEPT | **In** | Correct; the sentence recurs verbatim in JPMorgan's FY2020-FY2025 filings, all in validation (6 exact equivalents). |
| 109 | PFE FY2022, narrative/mdna | What was the expected total cost for Pfizer to separate from Upjohn, and what portion of that cost had been incurred by the end of fiscal 2022? | Pfizer expected to incur approximately $700 million in costs for the Upjohn separation, of which approximately 85% had been incurred since inception through December 31, 2022. | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct. |
| 110 | TSLA FY2024, narrative/statements | What was the unaudited impact on digital assets, net for Tesla, Inc. for the three months ended December 31, 2024, resulting from the adoption of the new crypto assets standard? | $347 million increase | Generator: `copied_from_passage` | – | Correct by the rule: copies the sentence. The answer is right. |
| 111 | WMT FY2024, table/statements | What was the total assets reported by Walmart Inc. in millions of dollars for the fiscal year ended January 31, 2024? | $252,399 million | ACCEPT | Out: evaluation evidence, training evidence | Correct; a balance as of January 31, asked 'for the fiscal year ended'. |
| 112 | XOM FY2023, narrative/other | What was the balance of Exxon Mobil's Euro-denominated 0.142% notes due 2024 as of December 31, 2023? | — | **REVISE** | **In** | The dash is not a nil balance: the notes mature in 2024, so the long-term debt table leaves them out and they sit in current liabilities (the note's previous chunk says so). |
| 113 | AAPL FY2020, narrative/statements | What was the balance of Apple's non-current term debt as of September 26, 2020? | $98,667 million | ACCEPT | Out: possible training evidence | Correct. Same balance-sheet template as 134. |
| 114 | AMZN FY2021, table/statements | What was the amount of cash Amazon paid for income taxes, net of refunds, for the year ended December 31, 2021? | $3,688 million | ACCEPT | Out: evaluation evidence | Correct; the third of three year columns. |
| 115 | JPM FY2021, narrative/statements | What was the carrying value of JPMorgan Chase's Series DD fixed-rate preferred stock as of December 31, 2021? | 1,696 million | ACCEPT | Out: possible training evidence | Correct. |
| 116 | PFE FY2025, table/statements | What was the gross carrying amount of identified intangible assets for Pfizer Inc. as of December 31, 2025? | $126,227 million | ACCEPT | **In** | Correct (104,006 + 22,221). |
| 117 | TSLA FY2022, table/statements | What was Tesla's total provision for income taxes for the year ended December 31, 2022? | $1,132 million | ACCEPT | Out: evaluation evidence, training evidence | Correct. |
| 118 | WMT FY2026, narrative/business | How many stores did Walmart International operate across 18 countries outside of the U.S. in fiscal 2026? | 5,743 stores | Generator: `copied_from_passage` | – | Correct by the rule: copies the sentence. |
| 119 | XOM FY2022, narrative/other | What was the outstanding balance of Exxon Mobil Corporation's 2.610% notes due 2030 as of December 31, 2022? | 2,000 | ACCEPT | **In** | Correct; no unit (millions). Same note as 112, a year earlier. |
| 120 | AAPL FY2021, table/statements | What was Apple's effective tax rate for fiscal 2021? | 13.3% | ACCEPT | Out: evaluation evidence, training evidence | Correct. |
| 121 | AMZN FY2020, narrative/statements | What was the weighted average life remaining for Amazon's total finite-lived intangible assets as of December 31, 2020? | 14.4 years | ACCEPT | **In** | Correct; the quote joins the row label to its cells. Same template as 100. |
| 122 | JPM FY2024, table/statements | What was the unaudited pro forma net interest income for JPMorgan Chase & Co for the year ended December 31, 2023, assuming the First Republic acquisition had occurred on January 1, 2022? | $90,856 million | Generator: `copied_from_passage` | – | Correct by the rule: copies the pro forma heading. The answer is right, but pro forma, the pattern flagged for this position. |
| 123 | PFE FY2023, table/statements | What was the accumulated depreciation for property, plant and equipment reported by Pfizer Inc. as of December 31, 2023? | $16,045 million | ACCEPT | **In** | Correct; 102's FY2024 passage repeats the figure as its prior-year column and is recorded as an equivalent. |
| 124 | TSLA FY2020, table/statements | What was the amount of deferred tax assets related to net operating loss carry-forwards for Tesla, Inc. as of December 31, 2020? | $2,172 million | ACCEPT | Out: possible training evidence | Correct. |
| 125 | WMT FY2021, narrative/business | How many associates did Walmart employ worldwide as of the end of fiscal 2021? | more than 2.3 million | ACCEPT | Out: evaluation evidence | Correct; a headcount. |
| 126 | XOM FY2025, table/other | What was ExxonMobil's effective income tax rate for fiscal 2025? | 31% | ACCEPT | **In** | Correct. |
| 127 | AAPL FY2019, table/statements | What was Apple's total provision for income taxes in fiscal 2019? | $10,481 million | ACCEPT | Out: evaluation evidence, training evidence | Correct. |
| 128 | AMZN FY2025, table/statements | What was the total estimated fair value of Amazon's cash, cash equivalents, and marketable securities as of December 31, 2024? | $101,202 | ACCEPT | **In** | Correct; no unit (millions). The FY2025 filing's prior-year table, named by its date. |
| 129 | JPM FY2025, narrative/statements | What was the total net amount of U.S. GAAP nettable derivative payables for interest rate contracts at JPMorgan Chase & Co. for the fiscal year ended December 31, 2025? | $6,729 | ACCEPT | **In** | Correct; a balance as of December 31, asked 'for the fiscal year ended'; no unit (millions). |
| 130 | PFE FY2021, narrative/business | In fiscal 2021, what is the permitted launch date for settled generic companies of Eliquis in the U.S. if the formulation patent is held invalid or not infringed in future litigation? | November 21, 2026 | Generator: `missing_company` | – | Correct: names no company (Eliquis is Pfizer's, with BMS). |
| 131 | TSLA FY2023, table/statements | What was Tesla's income before income taxes for the year ended December 31, 2023, and how did it compare to the prior two years? | $9,973 million in 2023, $13,719 million in 2022, and $6,343 million in 2021 | ACCEPT | Out: evaluation evidence, possible training evidence | Correct; all three years, in order. |
| 132 | WMT FY2025, table/mdna | What was Walmart's Return on Assets for the fiscal year ended January 31, 2025? | 7.9% | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct. |
| 133 | XOM FY2019, narrative/risk | According to ExxonMobil's FY2019 Form 10-K, what is the daily capacity of the Wink to Webster pipeline for transporting crude oil to the Houston refining center? | more than 1 million barrels per day | ACCEPT | **In** | Correct; a pipeline's capacity. |
| 134 | AAPL FY2019, narrative/statements | What was the balance of Apple's term debt classified as a current liability as of September 29, 2018? | $8,784 million | ACCEPT | Out: possible training evidence | Correct; the prior-year column, named by its date. |
| 135 | AMZN FY2019, table/statements | What was the amount of amortization expense for acquired intangible assets reported by Amazon in fiscal 2019? | $565 million | ACCEPT | **In** | Correct; the third of three years in the sentence. |
| 136 | JPM FY2023, narrative/statements | What was the earliest redemption date for JPMorgan Chase & Co's Series CC preferred stock as of December 31, 2023? | November 1, 2022 | Generator: `evidence_length` | – | Correct by the rule: the quote is the whole row. The answer is right. |
| 137 | PFE FY2020, narrative/statements | What agreement did Viatris make regarding liabilities from the Viagra and Cialis Multi-District Litigation as part of the transaction completed in November 2020? | Viatris agreed to assume liabilities and indemnify Pfizer for liabilities arising out of the matter. | Generator: `missing_company` | – | Correct: names Viatris but not Pfizer. |
| 138 | TSLA FY2025, narrative/statements | What amount of stock-based compensation expense did Tesla, Inc. record for the 2025 CEO performance award for the year ended December 31, 2025? | $162 million | ACCEPT | Out: possible evaluation evidence | Correct; a specific figure inside the auditor's critical audit matter. |
| 139 | WMT FY2022, table/mdna | What was Walmart U.S.'s net sales in millions of dollars for the fiscal year ended January 31, 2022? | $393,247 million | ACCEPT | Out: possible evaluation evidence | Correct; the Walmart U.S. segment. 104's passage also holds the figure. |
| 140 | XOM FY2020, table/other | What was the net carrying value of Exxon Mobil's long-term obligations to equity companies as of December 31, 2020? | $3,253 million | Generator: `evidence_length` | – | Correct by the rule: the quote is the whole row. The answer is right. |
| 141 | AMZN FY2020, table/statements | What was the amount of excess tax benefits from stock-based compensation included in Amazon's tax provision for the year ended December 31, 2020? | $1.8 billion | ACCEPT | Out: possible evaluation evidence, training evidence | Correct; the third of three years in the sentence. |
| 142 | JPM FY2020, narrative/other | What was the final issuance date for the interim final rule regarding the definition of eligible retained income issued by the Federal Reserve in March 2020? | August 26, 2020 | Generator: `missing_company` | – | Correct: names no company (JPMorgan's regulatory overview). |
| 143 | PFE FY2019, narrative/risk | What specific rights does GSK hold regarding the separation and listing of the Pfizer-GSK consumer healthcare joint venture during the first five years after the transaction closed? | GSK has the sole right to decide whether and when to initiate a separation and listing, and may also sell all or part of its stake in the joint venture in a contemporaneous initial public offering. | ACCEPT | **In** | Correct; the joint venture's terms, unchanged by year. |

## 5. Aspects reviewed (37 accepted questions)

| Aspect | Pass | Minor | Fail | n/a | Notes |
|---|---:|---:|---:|---:|---|
| Factual correctness | 35 | 0 | 2 | 0 | Fail: 99 (fiscal 2018's row given as fiscal 2019's), 112 (a dash read as a nil balance) |
| Financial relevance | 35 | 2 | 0 | 0 | Minor: 125 (a headcount), 133 (a pipeline's capacity) |
| Company and segment attribution | 37 | 0 | 0 | 0 | Every question names the filer; 104 and 139 name the Walmart U.S. segment, and 101 the Parent Company column, correctly |
| Fiscal year | 33 | 2 | 2 | 0 | Fail: 98 (names no year, and the answer changes by year), 99. Minor: 111 and 129 ("for the fiscal year ended" for a balance) |
| Figures | 33 | 0 | 2 | 2 | Fail: 99 and 112 |
| Units | 32 | 3 | 0 | 2 | Minor: 119, 128 and 129 give no unit (millions); none gives a wrong unit |

**Table headings and reporting periods** (the main finding):
- **99, a roll-forward table:** Apple's AOCI table runs from the September 30, 2017 balance through fiscal
  2018 to the September 29, 2018 balance, then through fiscal 2019. The visible text holds fiscal 2018's
  block; Qwen took its "before reclassifications" row, (4,415), as fiscal 2019's. Fiscal 2019's row,
  3,484, starts just past the window. The 2017 balance row is visible, so the error is a misreading.
- **112, a dash in a debt table:** "—" for the 0.142% notes due 2024 at December 31, 2023 means the notes
  were reclassified to current liabilities, as debt maturing within one year. They were still owed. The
  note's preceding chunk says so ("These amounts exclude that portion of long-term debt, totaling $4,009
  million, which matures within one year"), but that text is above the window. The error is the same as
  batch 2's 91.
- **98, a question with no year:** "What is ExxonMobil's forecast for the growth in worldwide chemical
  demand by 2030?" The FY2020 filing says "over 40 percent"; the FY2019 filing, in two validation
  passages, says "approximately 45 percent". As a validation label it would count a correct FY2019
  passage as a miss. Minor was arguable, since the answer is right for its passage. It is revised because
  the question cannot be answered without the year it omits.
- **Prior-year columns were read correctly:** 107 and 128 take the right column where the table lists the
  older year first, and 114, 135 and 141 take the third of three years. 128 (the FY2025 filing's
  December 31, 2024 table) and 134 (the FY2019 balance sheet's September 29, 2018 column) ask about the
  prior year and name its date correctly.

**Numerical accuracy:** every figure in an accepted answer was checked against its cell or sentence.
Apart from 99 and 112, each is the right number for the right row, column and date. 116's total
($126,227 million) is the sum of its finite-lived and indefinite-lived subtotals.

**Evidence support:**
- **The quote matcher accepts every quote.** All are verbatim except 100's and 121's, which join a row
  label to its cells (Amazon's intangible-assets table).
- **Answers:** every answer is supported by its quote. 99's quote is the right row of the wrong year,
  and 112's dash is visible.

**Visibility within BGE's 512-token limit:**
- **Where the text comes from:** 43 of 46 passages are truncated; 98, 122 and 126 are complete (334, 291
  and 388 tokens). Every quote lies inside the window.
- **Where the quotes end:** the latest ends at token 477 (127). Others after token 450: 131 (473), 101
  (460) and 123 (454). The fragment quotes end at tokens 386 (100) and 375 (121).
- **Text past the window:** two questions would have needed it. 99's correct row lies past the window,
  and the question was revised. 112's explanation lies above it.

## 6. Passages that start mid-table or mid-section

All 46 passages passed the context rules (`headless_table_start`, `fragment_opening`).

| Opening | Passages | Accepted | Errors |
|---|---|---:|---|
| Inside a table (rows before any table marker, or a table's end first) | 99, 100, 107, 110, 111, 112, 113, 115, 117, 119, 120, 121, 124, 127, 128, 129, 131, 134, 136, 140, 141 | 18 | 99 (the earlier year's block is the visible one); 112 (the explanation is above the window) |
| Mid-sentence (lowercase or punctuation first) | 98, 101, 102, 103, 104, 106, 114, 123, 125, 130, 132, 133, 138, 139, 142, 143 | 14 | 98, but not an opening error |
| Mid-sentence with a capital, or a footnote | 109, 118, 122, 135, 137 | 2 | None |
| At a note, section or table heading | 105, 108, 116, 126 | 3 | None |

- **Context above the window:** 112 depends on it, like batch 1's 18.
- **The three flagged patterns:**
  - 110, the quarterly table without a year: the generator rejected it for copying;
  - 122, pro forma: rejected for copying;
  - 138, a critical audit matter: gave an accepted question on a specific figure ($162 million for
    Tesla's 2025 CEO award), excluded by the evidence rule on figure matches alone.

## 7. Duplicates, templates, distribution and leakage

**Duplicates:**
- **None exact or near-exact.** No two questions share a normalised key, within the batch or with any
  earlier pool.
- **Highest Jaccard within the batch:** 0.615, between 120 and 126 (Apple's and ExxonMobil's effective tax
  rates).
- **Highest against earlier batches:** 0.812, between 107 (Amazon's net property and equipment, FY2023)
  and batch 2's question on Tesla's for the same date. This is a different company; 107 is excluded by
  the evidence rule anyway.
- **Highest against any pool:** 0.900, between 128 and a `val_pilot_v1` question on the same Amazon table
  two years earlier. That question is not among the 16 carried (highest against those: 0.533) and is in
  no training or validation set, so this is template similarity, not leakage.

**Repeated statement templates** (correlated, not duplicates):

| Questions | Template | Evidence rule | Usable |
|---|---|---|---|
| 100, 121 | Amazon's intangible-assets table (FY2024, FY2020) | 100 out, 121 in | 121 |
| 112, 119 (and 105) | ExxonMobil's long-term debt note (FY2023, FY2022; 105 FY2024) | Both in | 119 (112 revised; 105 generator-rejected) |
| 113, 134 | Apple's balance sheet, term debt (FY2020 non-current, FY2019 current) | Both out | – |
| 104, 139 | The Walmart U.S. segment table (FY2023, FY2022) | Both out | – |
| 102, 107, 123 | Property, plant and equipment notes (Pfizer FY2024 and FY2023, Amazon FY2023) | 123 in | 123 |
| 117, 120, 124, 126, 127, 131 | Income-tax notes (provision, rate, deferred tax assets, pre-tax income) | 126 in | 126 |

- **In the usable set:** no batch 3 usable question reaches 0.6 against any other of the 80 usable
  questions. The set's tightest template is still Apple's Level 2 securities table from earlier batches
  (five questions, pairwise up to 0.789).

**Evidence reuse across filing years:**
- **Among the 23 excluded questions:**
  - 14 have holders both in their own filing and in other years' filings;
  - 6 only in other years' filings: 100, 106, 114, 115, 132 and 138;
  - 3 only in their own filing: 101, 124 and 134.
- **The usual form is a comparative column:** a value reported for year N is repeated as the prior-year
  column of year N+1's filing. 102's FY2024 property total, for example, reappears in the FY2025 filing,
  which is in training. Repeated policy wording is the other form: 106's hedge-accounting sentence
  appears in Apple's FY2021, FY2022 and FY2025 filings.
- **By match:** figure matches alone excluded 6 (101, 109, 115, 124, 138, 139). Some of these may be
  coincidences, such as a "$162 million" elsewhere in Tesla's earlier filings for 138. The rule is
  deliberately conservative and treats them as matches.
- **Within validation, reuse is recorded, not excluded:** same-split holders become equivalents of the
  pair. Among the 11 usable:
  - 108 has 6 exact equivalents, the same policy sentence in JPMorgan's FY2020–FY2025 filings;
  - 123 has 1, batch 3's own 102 passage, whose prior-year column repeats the figure;
  - 121 has 1 uncertain;
  - 128 and 135 have copies in other validation filings, but past those passages' windows, so they are
    not equivalents.
- **108 is a boilerplate question that passes the evidence rule,** like batch 1's 17: every copy lies in
  validation, so it enters with its six equivalents. The question says "fiscal 2019", but any year's copy is labelled
  relevant.

**Linked units and filings:**
- **Isolation:** every positive's linked unit and overlap group lie wholly in validation; checked for all
  46 passages.
- **The cap:** counting `val_pilot_v1`, `val_pilot_v2e` and batches 1 and 2, every unit holds at most 3
  generated passages. Six reach 3 with this batch: `grp_01824` (JPMorgan's preferred stock), `grp_01540`,
  `grp_08915` (ExxonMobil's debt note), `grp_00694`, `grp_07774` and `grp_05036`.
- **Filings revisited:** all 41 filings in the batch already gave a passage to batches 1 or 2. Apple's
  FY2019 filing gave three passages here (99, 127, 134).
- **Across all 80 usable questions:** they come from 44 filings. 19 filings give one question, 16 give two,
  7 give three, and 2 give four: Amazon's FY2019 and FY2025 filings (135 and 128 here). Position 147 is
  from Amazon's FY2025 filing too.

**Distribution:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch passages | 6 | 7 | 7 | 7 | 6 | 6 | 7 | 46 |
| Usable, this batch | 0 | 3 | 2 | 3 | 0 | 0 | 3 | 11 |
| Usable so far (16 carried, 15, 18, 20, 11) | 11 | 15 | 13 | 8 | 9 | 12 | 12 | 80 |
| Remaining plan passages | 0 | 8 | 69 | 37 | 33 | 23 | 48 | 218 |
| Expected usable from them (projection) | 0 | 1.0 | 11.8 | 5.6 | 6.2 | 2.9 | 6.3 | 33.9 |
| Projected final | 11 | 16.0 | 24.8 | 13.6 | 15.2 | 14.9 | 18.3 | 113.9 |

- **Fiscal years:**
  - The batch has 7 passages from 2019, 9 from 2020, 6 from each of 2021 and 2023–2025, 5 from 2022 and 1
    from 2026.
  - The 11 usable questions: 2019 4, 2020 1, 2022 1, 2023 1, 2025 4.
  - All 80 usable so far: 2019 15, 2020 12, 2021 11, 2022 10, 2023 11, 2024 6, 2025 15, 2026 0.
  - Projected final: 2019 about 25, 2020 19, 2021 15, 2022 12, 2023 14, 2024 9, 2025 20. FY2024 is still
    the thinnest, and this batch added none.
- **Apple's supply is exhausted:** position 134 was the plan's last Apple passage. Apple ends at 11, and
  this batch's six Apple passages gave nothing usable: five were excluded by the evidence rule and one
  (99) was revised.
- **Kinds:** 24 tables and 22 narrative passages. The 11 usable are 5 tables and 6 narrative.
- **Concentration at the end:** JPMorgan would hold about 25 of 114 (22%).

**Evidence overlap with training, blocked passages and evaluation pools:**
- **The 14 "In" questions:**
  - no train or blocked passage holds the evidence;
  - no positive is used by any training pair.
- **The 23 excluded questions:**
  - by holder: training only 6, blocked only 7, both 10;
  - by blocked holder: one in an evaluation pool itself for 9, only linked to a pool for 8.
- **Question overlap:**

| Pool | Highest Jaccard | Question |
|---|---:|---|
| Evaluation questions | 0.393 | 139 |
| Training questions (all 150 ever generated) | 0.762 | 100 (against Amazon's contract-based intangibles, FY2022) |
| The 16 carried | 0.533 | 101 |
| `val_pilot_v1` (not carried) | 0.900 | 128 (see Duplicates) |
| `val_pilot_v2e` batch 1 | 0.636 | 128 |
| `val_pilot_v2f` batch 1 | 0.625 | 114 |
| `val_pilot_v2f` batch 2 | 0.812 | 107 |

## 8. Actual yield against the predictor

**By stage** (planned rates from the calibration; observed in batch 3):

| Stage | Planned | Observed |
|---|---:|---:|
| Generator acceptance | 0.82 | 37 / 46 (0.80) |
| Evidence pass, of accepted | 11.5 predicted for these 37 (mean 0.31) | 14 (0.38) |
| Review acceptance, of passing | 0.86 | 11 / 14 (0.79) |
| Usable | 11.7 for these 46 | **11** |
| Usable, three batches | 51.1 for the 143 | **49** |

**The predictor's accuracy** (measured, out of sample):
- **This batch:** AUC 0.834, against 0.703 for the old sentence share. Brier score 0.180, against 0.239
  for the calibration's constant pass rate (0.44) and 0.235 for the batch's own rate.
- **All three batches:** AUC 0.794 (sentence share 0.692). Brier 0.186, against 0.256 and 0.250.
- **Calibration, three batches:** 52.6 predicted and 59 observed, conservative in the middle and optimistic
  at the bottom:

| Predicted pass | Accepted questions | Predicted | Observed |
|---|---:|---:|---:|
| Below 0.2 | 20 | 2.2 | **0** |
| 0.2–0.4 | 25 | 7.5 | 12 |
| 0.4–0.6 | 31 | 15.3 | 19 |
| 0.6–0.8 | 26 | 17.6 | 18 |
| 0.8 and above | 11 | 10.0 | 10 |

**Why batch 3 met its estimate:** the evidence stage beat its prediction (14 against 11.5). Generator
acceptance (0.80) was close to plan, and review acceptance (0.79) was below it.

**Batch 1's figure-count pattern:** low-figure passages gave 0.33 usable questions per passage, and the
others 0.22. Pooled over all 243 generated passages, both give 0.33. No rule is needed.

**Against the earlier pilots** (which had no ranking):
- **Evidence pass:** batch 3 alone 38%, against 44% (p = 0.79). Its passages are ranked lower (mean
  predicted pass 0.31, against 0.48 in batch 2 and 0.62 in batch 1). Over all three batches: 52% (59 /
  113), against 44% (p = 0.16). The gain that was significant after two batches (p = 0.039) no longer is.
- **Usable per passage:** batch 3 24%; all three batches 34% (49 / 143), against 31% (p = 0.35).
- **What this means:** the ranking front-loaded the yield, as designed. The predictor's own calibration,
  not the comparison with unranked pilots, is the measure for the rest of the plan.

**Re-estimate for the 218 remaining passages** (projections):

| Model | Remaining | 144–147 | Next 50 (144–193) | Total with the 80 usable |
|---|---:|---:|---:|---:|
| As planned (calibration rates) | 36.4 | 1.3 | 11.4 | 116 |
| This plan's observed rates (generator 0.79, review 0.83, 143 passages) | 33.9 (bootstrap 31–37) | 1.2 | 10.6 | 114 |
| Generator and review acceptance by figure class, pooled over 243 passages | 35.6 (bootstrap 33–38) | 1.2 | 11.1 | 116 |
| Scaled by three batches (49 / 51.1) | 34.9 | 1.2 | 10.9 | 115 |
| Observed rates, with passages below 0.2 giving nothing | 26.2 | 1.2 | – | 106 |

**Yield per request falls with the ranking:**
- **Mean score by block of 50 positions:** 0.32 (144–193), 0.27, 0.26, 0.14 and 0.11 (344–361).
- **Requests per usable question:** about 4.5 in batch 3 (50 requests, 11 usable); about 4.7 in the next
  block; 5.6–5.9 in the two after; 10.6 and 13.8 in the last two. These figures use the predictor's
  rates, which may be optimistic below 0.2.
- **Passages below 0.2:** 108 of the remaining 218, the first at position 151. They are expected to give
  7.7 usable questions, and the 20 such questions seen so far gave none.

## 9. Did the predictor and the administrative filter help?

**The predictor:**
- **Ranking: yes.** AUC 0.83 in this batch and 0.79 over 113 questions it never saw.
- **Calibration:** good from 0.2 up, optimistic below 0.2 (0 of 2.2 expected passes). Not significant on
  its own (the chance of no pass when 2.2 are expected is about 11%), but it matters for the half of the
  remaining plan that scores below 0.2.
- **Yield against unranked pilots:** not established over three batches (§8). The ranking moved the yield
  forward rather than raising the total by a measurable amount.

**The administrative filter:**
- **No administrative question reached the review:** no trading-plan, insider-trading, cybersecurity,
  Item 6, Section 13(r) or website-services passage was in the batch.
- **The three flagged positions:** 110 and 122 were rejected by the generator, and 138 gave a specific,
  non-boilerplate question (§6).
- **What remains:** no critical audit matter, pro forma passage or yearless quarterly table. Two
  forward-looking-statement notices (265, 348) and two new-standard notices (192, 295) remain. None falls
  in 144–147.

## 10. Systematic weaknesses of Qwen's outputs in this batch

1. **Reporting periods in tables:** an earlier year's row read as the current year's in a roll-forward
   table (99). This is the batch 1 pattern (26), in a different table layout.
2. **Mislabelling what a figure is:** a dash read as a nil balance (112), the same confusion between
   long-term and current debt as batch 2's 91.
3. **Leaving out the year when the answer depends on it:** 98.
4. **Copied wording:** 4 of 46, against 1 in batch 2.
5. **No company named:** 3 of 46, as in batch 2.
6. **Whole-row quotes from wide tables:** 136 and 140, over the 300-character limit.
7. **Units dropped:** 3 answers give no "millions" (119, 128, 129); batch 2 had 1.
8. **Template repetition across years:** Amazon's intangible assets (100, 121), ExxonMobil's debt note
   (105, 112, 119), the Walmart U.S. segment (104, 139) and income-tax notes (six questions).
9. **Better than in earlier batches:** no administrative or trivial question; no question depends on
   text past the window; correct company and segment attribution throughout.

## 11. Comparison with earlier pilots

| | `val_pilot_v1` | `val_pilot_v2e` batch 1 | `val_pilot_v2f` batch 1 | `val_pilot_v2f` batch 2 | `val_pilot_v2f` batch 3 |
|---|---|---|---|---|---|
| Plan | Balanced years, no score | Balanced years, isolation share ≥ 0.3 | Balanced years, ranked by `evidence-pass-v1`; administrative rule | Same plan, positions 50–97 | Same plan, positions 98–143 |
| Passages completed | 50 | 50 | 49 (50 requests) | 48 (50 requests) | 46 (50 requests reported, two executions) |
| Generator acceptance | 43 (86%) | 39 (78%) | 35 (71%) | 41 (85%) | 37 (80%) |
| Mean predicted pass, of accepted | – | – | 0.62 | 0.48 | 0.31 |
| Evidence pass, of accepted | 19 (44%) | 17 (44%) | **24 (69%)** | 21 (51%) | 14 (38%) |
| Review: accept / revise / reject, of accepted | 36 / 5 / 2 | 36 / 1 / 2 | 26 / 4 / 5 | 39 / 2 / 0 | 34 / 3 / 0 |
| Review acceptance, of evidence-passing | – | – | 0.75 | 0.95 | 0.79 |
| Factual errors, of accepted | 5 (12%) | 1 (3%) | 6 (17%) | 2 (5%) | 2 (5%) |
| Usable per passage | 16 (32%) | 15 (30%) | 18 (37%) | **20 (42%)** | 11 (24%) |
| Tokens per passage | 1,344 | 1,322 | 1,314 | 1,353 | 1,397 |

## 12. Recommendation for the remaining four passages

**Run positions 144–147 on `val_pilot_v2f` unchanged,** with every question AI-reviewed as here.
- **Why:**
  - the safeguards held: no leakage, no question overlap, no unit crossing the split, every cap respected;
  - the predictor ranks well (AUC 0.83 here), and the four passages score 0.40–0.56, above the range
    where it is optimistic;
  - none of the four shows a known failure pattern;
  - it completes batch 3's window, so any later batch starts at position 148.
- **What to expect** (projections): about 1.2 usable questions from 4 requests, about 5,600 tokens and 2
  minutes.
- **The four passages:**
  - 144, Tesla FY2019 risk factors (Gigafactory Shanghai);
  - 145, Walmart FY2020 property and equipment note;
  - 146, ExxonMobil FY2021 business environment;
  - 147, Amazon FY2025 lease-cost table. If usable, it would be that filing's fifth question.
- **What to watch:**
  - 147's table lists 2023, 2024 and 2025 in that order, the layout where year confusion is possible;
  - 146 is a FY2021 filing that discusses 2020, so the question's year needs checking.

**Then stop for a decision before batch 4** (positions 148–197). None of these blocks the four passages:
- **How far to run the plan:**
  - The next block expects about 10–11 usable questions from 50 requests. The two after expect about 9
    and 8; the last 68 positions about 6 in all, at 11–14 requests each.
  - Half of what remains scores below 0.2, where no evidence pass has been seen yet.
  - Stopping after position 297 (three more batches of 50) would give about 104–108 usable questions.
    Running the whole plan would give about 114–116, or about 106 if the low end yields nothing.
- **The target:**
  - 120 is now out of reach of this plan without a supply change: a unit cap of 4 or 5, or a larger
    validation split.
  - About 105–110 is the realistic range.
- **Balance in the final set:**
  - Apple is closed at 11 and Amazon nearly so.
  - JPMorgan would be about 22% of the set, and FY2024 the thinnest year (about 9).
  - A per-company, per-filing or per-template cap at build time is worth deciding. Two Amazon filings
    already give four questions each, and Apple's Level 2 securities template gives five.
- **A period check for the review:** all three of this batch's review exclusions concern reporting
  periods, and every automatic check passes them. The review should keep checking roll-forward tables,
  dashes in debt tables, and questions with no year whose answer changes between filings. No rule change
  is proposed.
- **Recalibration:** not needed before 144–147. If the plan continues past position 150, a recalibration
  of the low end (or a cut-off at 0.2) is worth weighing, since it decides most of the remaining cost.
- **Recording:** the decisions of batches 1–3 can go into versioned files in `data/validation_review/`,
  in the format of `val_pilot_v2e`'s batch, once you approve them.
