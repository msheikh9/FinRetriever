# Validation plan `val_pilot_v2f`, batch 1: audit of the first 49 records

Status: 2026-11-04.
- **What this is:** an AI review (Claude) of the first 49 generation records of plan `val_pilot_v2f` in
  `bge_base_ft_v3`. It is not human annotation, and no decision here is a gold label.
- **What was not done:**
  - no generation record, plan, training pair, candidate, calibration or validation set was changed;
  - no validation set was built, and no decision file was written: the decisions below exist only in this
    report until they are recorded;
  - no API request was sent and nothing was trained.
- **Checksums:** 272 files under `data/`, `configs/` and `results/` were checksummed before and after the
  audit, and none changed.
- **The checks:** every validation-set check (validator, evidence rule, question overlap, duplicates, unit
  isolation) was re-run in memory, as `build-validation` would run it, with nothing written.
- **The plan:** [validation_plan_v2f.md](validation_plan_v2f.md).

## 1. Summary

**The new predictor did what it was built for:**
- **Evidence-rule pass rate:** 24 of the 35 accepted questions passed (69%), against 44% in both
  `val_pilot_v1` and batch 1 of `val_pilot_v2e`. Fisher's exact test gives p = 0.016 against the 82
  questions of both pilots together.
- **Out of sample:** the calibration never saw these questions.
  - Ranking: AUC 0.80, against 0.65 for the old sentence score.
  - Accuracy: it predicted 21.6 passes, and 24 were observed.

**Usable yield rose only a little:**
- **The funnel:**
  - 50 HTTP attempts completed 49 passages (one 503 was retried);
  - the generator accepted 35 (71%);
  - the evidence rule left 24;
  - the review leaves **18 usable questions**.
- **Against the plan:** it expected 23.4 from these 49 passages.
- **Against earlier pilots:** 18 per 49 (37%) is above 30% and 32%, but the difference is not established
  (p = 0.58).
- **Where it was lost:** in generator and review rejections, concentrated in passages with two or fewer
  significant figures. The predictor ranks these highest when their sentences are unique (§8).

**Quality is lower than in `val_pilot_v2e`'s batch:**
- **Factual errors:** 6 of the 35 accepted questions (17%), against 1 of 39 and `val_pilot_v1`'s 5 of 43.
- **Errors that passed the evidence rule:** three of the six (14, 18, 32). The review is what keeps them
  out.
- **Company attribution:** every question that names a company names the right one.

**Recommendation:** batch 2 may proceed on `val_pilot_v2f` unchanged, with the review kept (§12).
- **The total:** at these rates, the whole plan gives about 99–113 usable validation questions, not about 119.
- **Reaching 120** now needs the optimistic case, or one of the supply changes already listed in
  [validation_plan_v2f.md](validation_plan_v2f.md).

## 2. Provenance

| Check | Result |
|---|---|
| Plan | `validation.val_pilot_v2f.json` (`a7903bc7…`), unchanged. The 49 records are its positions 1–49, in order. Position 50 (AAPL FY2021, a table) was not reached |
| Run | The same bound run, no continuation: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2` (`f42bdc54…`), parameters `42d5fe54…`, validator `sec-qcheck-v3`, one question per passage |
| Records | 49 done, 0 failed, each on its first response attempt, no errors. The 503 retry spent the 50th request of the budget; records do not log HTTP retries. Sent 2026-11-03, 20:39–21:03 UTC (2.0 per minute) |
| Tokens | 64,397: 57,550 prompt and 6,847 completion, 1,314 per passage. Completion 97–325, median 128, against the 768 limit |
| Files | `validation.records.jsonl` is now `e5c7ffd3…`. Its first 50 lines are byte-identical to `val_pilot_v2e` batch 1 (`cc1b6ce6…`), so the run only appended. `validation.run.json` `34e6b74d…` is unchanged |
| Validator re-run | Reproduces all 49 outcomes (35 accepted, 14 rejected, the same reasons) |
| Predictor re-run | All 49 plan scores recompute exactly from `configs/train/evidence_pass_v1.json` |

## 3. The funnel

| Stage | Kept | Removed | Why |
|---|---:|---:|---|
| HTTP attempts | 50 | | One 503, retried |
| Passages completed | 49 | 1 | Position 50 not reached |
| Generator (`sec-qcheck-v3`) | 35 | 14 | `evidence_length` 7, `copied_from_passage` 5, `missing_company` 2 |
| Evidence rule (validation: any match outside validation) | 24 | 11 | Possible evaluation evidence 7, possible training evidence 6, training evidence (exact, visible) 2, evaluation evidence 2; some have several |
| Question overlap (evaluation, training, duplicates) | 24 | 0 | Highest Jaccard: 0.25 against an evaluation question, 0.696 against a training question (threshold 0.8) |
| AI review | **18** | 6 | Reject 18, 39, 41 and 45. Revise 14 and 32. Also revise 02 and 23 and reject 26, which the evidence rule had already excluded |

**Generator rejections** (14; verdict on each in §4):
- **All correct by the rule.**
- **Long list quotes:** 7 rejections are `evidence_length`.
  - Six are risk-factor or business lists: 03, 11, 25, 31, 46 and 47.
  - One is a whole eight-cell table row (42).
  - All four risk-factor passages of the batch were lost this way.
- **Strict rejections:** two of the `copied_from_passage` rejections, 06 and 40, are natural questions that
  repeat a row label or a sentence.
- **No false rejections.**

**Question types accepted:** `financial_table` 19, `accounting_policy` 4, `other` 4, `risk` 3,
`debt_liquidity` 2, `profitability` 2, `tax` 1.

## 4. Per-question decisions

- **Decisions:** ACCEPT, REVISE and REJECT apply to the 35 questions the generator accepted. For the 14
  it rejected, the note gives a verdict on that rejection.
- **"Evidence rule":** "In" means no train or blocked passage holds the evidence, exactly or possibly, in
  its full text.
- **Would enter a validation set:** 18 questions, accepted and "In": 01, 08, 09, 10, 12, 16, 17, 24, 27,
  28, 30, 33, 34, 36, 43, 44, 48, 49.
- **REVISE:** in this pipeline it excludes the question. Records are never edited, so a corrected question
  would need a new generation.

| # | Source | Question (verbatim) | Answer (verbatim) | Decision | Evidence rule | Note |
|---|---|---|---|---|---|---|
| 01 | AAPL FY2022, narrative/statements | What was the fair value of Apple's Level 2 corporate debt securities in fiscal 2022? | 79,450 | ACCEPT | **In** | Correct; no unit (millions). Apple's securities-table template (also 36, 43). |
| 02 | AMZN FY2023, narrative/statements | What were Amazon's liabilities for payroll related expenses as of December 31, 2022 and December 31, 2023? | $7.7 billion as of December 31, 2022 and $5.4 billion as of December 31, 2023 | **REVISE** | Out: possible evaluation evidence, possible training evidence | $5.4 billion is the gift-card liability at December 31, 2022, not payroll at 2023; payroll is $7.7 billion. |
| 03 | JPM FY2022, narrative/risk | What specific restrictive government policies could JPMorgan Chase face in jurisdictions where the local economy is subject to substantial government influence? | Price, capital or exchange controls, including imposition of punitive transfer and convertibility restrictions or forced currency exchange, and expropriation or nationalization of assets or confiscation of property. | Generator: `evidence_length` | – | Correct by the rule (list quote over 300 characters); a generic risk factor. |
| 04 | PFE FY2022, narrative/statements | What specific assumption did the auditor identify as involving especially challenging judgment in the evaluation of Pfizer's U.S. Medicare, Medicaid, and performance-based contract rebates accrual for fiscal 2022? | The product-specific experience ratio assumption | Generator: `copied_from_passage` | – | Correct by the rule: repeats an auditor's phrase. A critical-audit-matter question. |
| 05 | TSLA FY2021, table/statements | What was the fair value of Tesla's interest rate swap liabilities at December 31, 2021? | 31 million | ACCEPT | Out: possible evaluation evidence | Correct. |
| 06 | WMT FY2026, narrative/statements | What was Walmart's total current liabilities as of January 31, 2026? | 107,469 | Generator: `copied_from_passage` | – | Correct by the rule (the row label); a natural question otherwise. |
| 07 | XOM FY2022, table/other | What was the total dollar amount for the 15 projects with completed exploratory activity progressing toward development as of December 31, 2022? | $2,324 million | Generator: `missing_company` | – | Correct: names no company. |
| 08 | AAPL FY2023, table/statements | What were Apple's total undiscounted lease liabilities for finance leases as of September 30, 2023? | $1,425 million | ACCEPT | **In** | Correct. |
| 09 | AMZN FY2022, table/statements | What were Amazon's total gross deferred tax assets reported for the fiscal year ended December 31, 2022, in millions? | 38,883 million | ACCEPT | **In** | Correct. |
| 10 | JPM FY2019, table/statements | What were the total proceeds received from loan sales for JPMorgan Chase in the year ended December 31, 2019? | $91,495 million | ACCEPT | **In** | Correct row; the table covers loans sold to the GSEs and in Ginnie Mae securitizations, which the question omits. |
| 11 | PFE FY2024, narrative/business | In fiscal 2024, how does Pfizer describe the potential financial impact on revenues if patent protection for its products expires or is lost prior to the expiration date? | Pfizer typically loses market exclusivity, leading to generic and biosimilar competition that can substantially decrease revenues for impacted products, often in a very short period of time. | Generator: `evidence_length` | – | Correct by the rule; the answer also uses a sentence outside the quote. |
| 12 | TSLA FY2024, narrative/business | How many actively enrolled learners had access to funding under Tesla's Education Assistance Program by the end of fiscal 2024? | 5,436 | ACCEPT | **In** | Correct; a workforce-programme count. |
| 13 | WMT FY2023, table/business | How many total retail and wholesale units did Walmart International operate across all geographic markets as of January 31, 2023? | 5,306 units | ACCEPT | Out: possible evaluation evidence, training evidence | Correct; an operational count, and non-Canadian counts are as of December 31, 2022. |
| 14 | XOM FY2024, narrative/statements | What was ExxonMobil's charge to other comprehensive income attributable to the company for total pension and other postretirement benefits in fiscal 2024? | 480 | **REVISE** | **In** | The row is '(Charge)/credit': 480 is a credit, not a charge. No unit (millions). |
| 15 | AAPL FY2025, table/statements | What was Apple's total operating income for fiscal 2025? | $133,050 | ACCEPT | Out: evaluation evidence | Correct; no unit (millions). |
| 16 | AMZN FY2021, table/statements | What was the total amount of Amazon's long-term debt principal and interest commitments expected to be paid thereafter, as of December 31, 2021, in millions? | $52,784 million | ACCEPT | **In** | Correct. |
| 17 | JPM FY2021, narrative/statements | For JPMorgan Chase & Co as of December 31, 2021, what is the Firm's stated method for managing the credit risk associated with contracts to sell protection? | The Firm manages the credit risk on contracts to sell protection by purchasing protection with identical or similar underlying reference entities. | ACCEPT | **In** | Correct; a policy sentence repeated yearly (6 validation equivalents). |
| 18 | PFE FY2025, narrative/business | According to Pfizer's fiscal 2025 disclosure, when is the composition of matter patent scheduled to expire? | November 2026 | **REJECT** | **In** | Names no product: the footnote's product is above the visible text, so any composition-of-matter patent fits. |
| 19 | TSLA FY2022, table/statements | What was the fair value of Tesla's money market funds as of December 31, 2022? | $2,188 million | ACCEPT | Out: possible training evidence | Correct. |
| 20 | WMT FY2024, narrative/statements | What was the total amount, including attorneys' fees and costs, that Walmart agreed to pay to resolve substantially all opioid-related lawsuits with states, political subdivisions, and Native American tribes in fiscal 2024? | approximately $3.1 billion | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct amount ('up to approximately'); agreed in November 2022 (fiscal 2023), reported in the fiscal 2024 10-K. |
| 21 | XOM FY2023, narrative/other | How does Exxon Mobil Corp typically determine when revenue is recognized for customer transactions in fiscal 2023? | Revenue is recognized when the customer has taken control, which is typically when title transfers and the customer has assumed the risks and rewards of ownership. | ACCEPT | Out: possible training evidence | Correct; revenue-recognition policy repeated yearly. |
| 22 | AAPL FY2024, table/statements | What was the notional amount of Apple's derivative instruments designated as accounting hedges for foreign exchange contracts in fiscal 2024? | $64,069 million | ACCEPT | Out: evaluation evidence, training evidence | Correct. |
| 23 | AMZN FY2024, table/statements | What was the gross amount of assets Amazon had acquired under finance leases as of December 31, 2024? | $62.5 billion | **REVISE** | Out: possible evaluation evidence | $62.5 billion is December 31, 2023; the 2024 figure is $56.5 billion ('… and 2024' order). |
| 24 | JPM FY2024, table/statements | What was the total notional amount of investment-grade credit derivatives sold by JPMorganChase with a maturity of less than one year as of December 31, 2024? | $(135,950) million | ACCEPT | **In** | Correct; keeps the table's parentheses for protection sold. |
| 25 | PFE FY2023, narrative/mdna | What changes did Pfizer make to its commercial organization effective January 1, 2024, and what were the purposes of these changes? | Pfizer created three divisions within its Biopharma segment: the Pfizer Oncology Division, the Pfizer U.S. Commercial Division, and the Pfizer International Commercial Division, to incorporate Seagen and improve focus, speed, and execution. | Generator: `evidence_length` | – | Correct by the rule (stitched list quote). |
| 26 | TSLA FY2020, table/statements | What was Tesla's gross profit for the three months ended December 31, 2020? | $1,391 | **REJECT** | Out: possible evaluation evidence, possible training evidence | The table is 2019's quarters (Q4 2020 gross profit was $2,066 million, in the previous chunk); the visible text names no year. |
| 27 | WMT FY2025, narrative/statements | How much did Walmart Inc. receive in settlement proceeds for the opioid-related derivative actions in December 2024, and how was it recorded? | Walmart Inc. received settlement proceeds of approximately $99 million on December 24, 2024, and recorded it as a reduction to operating, selling, general, and administrative expenses. | ACCEPT | **In** | Correct; two-part question; the quote ends at token 510 of 512. |
| 28 | XOM FY2025, table/statements | How many shares of restricted stock and units were issued and outstanding at the end of fiscal 2025 for ExxonMobil? | 40,654 thousand shares | ACCEPT | **In** | Correct. |
| 29 | AAPL FY2021, narrative/other | Did Apple Inc. report any payments, gross revenues, or net profits related to its interactions with the Russian Federal Security Service during the three-month period ended June 26, 2021? | No, the company did not make payments, nor did it receive gross revenues or net profits in connection with such engagement. | Generator: `copied_from_passage` | – | Correct by the rule; also administrative (Item 9B Section 13(r)), which the filter's markers miss. |
| 30 | AMZN FY2020, narrative/statements | How many diluted shares were used in the computation of earnings per share for Amazon.com in the fourth quarter of fiscal 2020? | 513 | ACCEPT | **In** | Correct; no unit (millions). |
| 31 | JPM FY2025, narrative/risk | In fiscal 2025, how does JPMorganChase describe the potential impact of emerging technological advancements like automation and AI on its workforce and operations? | These advances could lead to workforce displacement, requiring investment in employee training, management of morale and retention impacts, and competition for candidates with advanced technological skills, all of which could negatively impact the business and operations. | Generator: `evidence_length` | – | Correct by the rule; a generic risk factor. |
| 32 | PFE FY2021, table/statements | What was the balance of property, plant and equipment included in the components of assets and liabilities of discontinued operations and other assets held for sale for Pfizer as of December 31, 2021? | $— million | **REVISE** | **In** | The PP&E row is nil for 2021, but footnote [a] says the 2021 amounts ($25 million) are PP&E held for sale. |
| 33 | TSLA FY2023, narrative/business | How many graduates did Tesla hire through its Manufacturing Development Program in 2023? | 373 | ACCEPT | **In** | Correct; a workforce-programme count. |
| 34 | WMT FY2022, table/business | What was the total number of retail units owned and leased by Walmart U.S. as of January 31, 2022? | 4,742 units | ACCEPT | **In** | Correct; an operational count. |
| 35 | XOM FY2019, table/other | What was the suspended exploratory well cost for the Neptun Deep project in Romania as of December 31, 2019, in millions of dollars? | 536 million dollars | Generator: `missing_company` | – | Correct: names no company. |
| 36 | AAPL FY2020, narrative/statements | What was the fair value of Apple's corporate debt securities classified as Level 2 in fiscal 2019? | 86,047 | ACCEPT | **In** | Correct prior-year figure; no unit (millions). Same template as 01 (Jaccard 0.789). |
| 37 | AMZN FY2025, table/statements | What was Amazon's total cash taxes paid, net of refunds, in dollars millions for the year ended December 31, 2025? | $8,295 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct. |
| 38 | JPM FY2023, narrative/other | By what date does JPMorgan Chase & Co. expect to remediate the limited number of contracts remaining that reference synthetic U.S. dollar LIBOR? | September 30, 2024 | Generator: `copied_from_passage` | – | Correct by the rule: repeats the sentence. |
| 39 | PFE FY2020, narrative/business | What specific shareholder services information does Pfizer Inc. state is available on its website for fiscal 2020? | Computershare Investment Program, book-entry share ownership and direct deposit of dividends | **REJECT** | **In** | Shareholder-services information on Pfizer's website: administrative, no financial content. |
| 40 | TSLA FY2025, narrative/statements | What was the grant date fair value of the 2025 CEO Interim Award granted to Tesla, Inc.'s CEO? | $26.06 billion | Generator: `copied_from_passage` | – | Correct by the rule; a natural question otherwise. |
| 41 | WMT FY2021, narrative/statements | What reason did Walmart Inc. provide for reserving Item 6 in its fiscal 2021 Form 10-K? | Early adoption of Item 301 of Regulation S-K, as deleted pursuant to SEC Release No. 33-10890; 34-90459 | **REJECT** | **In** | Why Item 6 is reserved: filing mechanics, no financial content. |
| 42 | XOM FY2021, narrative/other | What was the net carrying value of Exxon Mobil's advances to and receivables from equity companies as of December 31, 2020? | $8,812 million | Generator: `evidence_length` | – | Correct by the rule: the quote is a whole 8-cell row. |
| 43 | AAPL FY2019, narrative/statements | What was the fair value of Apple's Level 2 non-U.S. government securities at the end of fiscal 2018? | 21,400 | ACCEPT | **In** | Correct prior-year figure; no unit (millions). Apple's securities-table template. |
| 44 | AMZN FY2019, table/statements | What was the face value of Amazon's long-term debt at December 31, 2019? | $23,513 | ACCEPT | **In** | Correct; no unit (millions). |
| 45 | JPM FY2020, table/other | What accounting principle does JPMorgan Chase & Co follow to determine whether off-balance sheet arrangements are recognized on-balance sheet or disclosed off-balance sheet? | U.S. GAAP | **REJECT** | **In** | The answer, U.S. GAAP, follows from the question (as val_pilot_v2e 15). |
| 46 | PFE FY2019, narrative/risk | What specific tax ruling did Pfizer state was a condition for the combination of Upjohn with Mylan in fiscal 2019? | An Internal Revenue Service ruling and an opinion of tax counsel that certain transactions constitute a tax-free reorganization under Section 368(a)(1)(D) of the Internal Revenue Code. | Generator: `evidence_length` | – | Correct by the rule; a long legal condition. |
| 47 | TSLA FY2019, narrative/risk | According to Tesla's fiscal 2019 Form 10-K, what are the specific risks associated with lithium-ion cells that could adversely affect the company's business? | Risks include increased material costs or decreased supply, supply disruptions due to quality issues or recalls, and foreign currency fluctuations affecting purchases denominated in currencies like the Japanese yen. | Generator: `evidence_length` | – | Correct by the rule: a three-item list. |
| 48 | WMT FY2020, narrative/statements | How much did Walmart pay in June 2019 to resolve the investigations by the DOJ and SEC regarding possible FCPA violations? | $283 million | ACCEPT | **In** | Correct. |
| 49 | XOM FY2020, narrative/other | What was the fair value of ExxonMobil's advances to equity companies classified as Level 3 at December 31, 2019? | $6,729 million | ACCEPT | **In** | Correct prior-year figure; 'advances to equity companies' for 'advances to/receivables from equity companies'. |

## 5. Aspects reviewed (35 accepted questions)

| Aspect | Pass | Minor | Fail | n/a | Notes |
|---|---:|---:|---:|---:|---|
| Factual correctness | 27 | 2 | 6 | 0 | Fail: 02 and 23 (wrong year's figure), 14 (credit called a charge), 18 (no product), 26 (2019 quarters as 2020's), 32 (footnote overrides the row). Minor: 10 omits the table's scope; 49 shortens the row label |
| Financial relevance | 28 | 4 | 3 | 0 | Fail: 39 (website shareholder services), 41 (Item 6 "reserved"), 45 (the answer, U.S. GAAP, follows from the question). Minor: 12 and 33 (workforce programmes), 13 and 34 (unit counts) |
| Company attribution | 35 | 0 | 0 | 0 | Every question names the filer correctly |
| Fiscal year | 30 | 3 | 2 | 0 | Fail: 23, 26. Minor: 13 (most counts are as of December 31), 20 (agreed in fiscal 2023), 39 ("for fiscal 2020") |
| Figures | 24 | 2 | 3 | 6 | Fail: 02, 23, 26. Minor: 24 keeps the parentheses for protection sold; 32 gives a nil as "$— million" |
| Units | 21 | 8 | 0 | 6 | Minor: eight answers give no unit (millions); none gives a wrong unit |

**Evidence support:**
- **The quote matcher accepts every quote.**
  - Verbatim: all but two.
  - 39: verbatim apart from the case of its first letter.
  - 49: fragments that join a row label to its Level 3 cell.
- **Answers:** every answer is supported by its quote. Three questions mislead even so:
  - 14: the quote says "(Charge)/credit";
  - 18: names no product;
  - 32: a footnote changes the meaning of the row.

**Visibility within BGE's 512-token limit:**
- **Where the text comes from:** the visible text is the passage truncated to the tokenizer's window.
  42 of 49 passages are truncated, and every quote lies inside the window.
- **Where the quotes end:** the latest ends at token 510 of 512 (27), and four end after token 430 (09,
  27, 34, 43).
- **The one cut sentence:** 27's sentence continues past the window ("… administrative" is the last visible
  word, followed by "expenses"). Its quote is visible, and its answer completes the phrase.
- **No quote is cut off.**

## 6. Passages that start mid-table or mid-section

All 49 passages passed the context rules (`headless_table_start`, `fragment_opening`).

| Opening | Passages | Accepted | Errors |
|---|---|---:|---|
| Inside a table (rows before any table marker, or a table's end first) | 01, 05, 06, 07, 08, 14, 15, 23, 26, 30, 35, 36, 42, 43, 44, 49 | 12 | **26**: quarter headings with no year; the table is 2019's. 14 and 23 err elsewhere (a sign convention, a sentence's year order) |
| Mid-sentence (lowercase or punctuation first) | 02, 03, 04, 09, 10, 18, 19, 20, 21, 22, 24, 25, 31, 32, 33, 37, 39, 40, 46, 47, 48 | 14 | **18**: the footnote's product is above the visible text. 02 and 32 err within the visible text |
| Mid-sentence, first word capitalised | 11, 12, 16, 17, 27, 28, 38 | 5 | None |
| At an item, note or section heading | 13, 29, 34, 41, 45 | 4 | 41 and 45 rejected for relevance, not error |

**What the rules leave:**
- **A table whose column headings name a quarter but no year (26):**
  - the rule counts "Three Months Ended December 31" as a period;
  - the passage is the second half of Tesla's FY2020 quarterly note, which covers 2019's quarters;
  - one more such passage remains in the plan (position 110).
- **A note below a cut-off heading (18):** as `val_pilot_v2e`'s 45.

## 7. Duplicates, concentration, distribution and leakage

**Duplicates:**
- **None exact or near-exact.** No two questions share a normalised key.
- **Highest Jaccard within the batch:** 0.789, between 01 and 36, which use the same template for two
  different years.

**Same template, different year** (correlated, not duplicates):

| Questions | Template | Linked unit | Evidence rule |
|---|---|---|---|
| 01, 36, 43 | Apple's marketable-securities table (FY2022; FY2020 report, 2019 column; FY2019 report, 2018 column) | Three different units | All in. With the two carried questions and `val_pilot_v2e`'s 50, **6 of Apple's 9 usable questions come from this note** |
| 17, 24 | JPMorgan's credit-derivatives note (FY2021 policy; FY2024 maturity profile) | `grp_01705`, with `val_pilot_v2e`'s 10: three usable questions, the unit's cap | Both in |
| 13, 34 | Walmart's unit counts (Item 2) | `grp_07805` | 13 out, 34 in |
| 05, 19 | Tesla's fair-value table | `grp_06669` (with `val_pilot_v1`'s passage: the cap) | Both out |
| 12, 33 | Tesla's workforce programmes | Different units | Both in |

**Linked units and filings:**
- **Batch units holding two passages:** five (table above, plus `grp_01437` JPMorgan risk factors and
  `grp_08904` ExxonMobil suspended wells, all four of those generator-rejected).
- **The cap:** counting `val_pilot_v1` and `val_pilot_v2e`, every unit holds at most 3 generated passages.
- **Filings:** no filing appears twice in the batch.

**Distribution:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch passages | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 49 |
| Usable, this batch | 4 | 4 | 3 | **0** | 2 | 3 | 2 | 18 |
| Usable so far (16 carried, 15, 18) | 9 | 8 | 7 | **2** | 8 | 7 | 8 | 49 |
| Remaining plan passages | 13 | 22 | 83 | 51 | 46 | 36 | 61 | 312 |
| Expected usable from them (projection) | 2.3 | 4.2 | 18.8 | 9.6 | 9.8 | 6.8 | 11.7 | 63.3 |

- **Fiscal years:**
  - The batch has 6 passages from 2019, 7 from each of 2020 to 2025, and 1 from 2026.
  - The 18 usable questions: 2019 3, 2020 4, 2021 2, 2022 3, 2023 2, 2024 2, 2025 2.
  - All 49 usable so far: 2019 8, 2020 6, 2021 7, 2022 8, 2023 7, 2024 4, 2025 9, 2026 0.
- **Kinds:** 29 narrative passages and 20 tables. The 18 usable are 10 narrative and 8 tables.
- **Pfizer gave nothing:**
  - three generator rejections for long quotes (11, 25, 46) and one for copying (04);
  - the review rejected 18 and 39 and revised 32.

**Evidence overlap with training, blocked passages and evaluation pools:**
- **The 24 "In" questions:**
  - no train or blocked passage holds the evidence;
  - no positive is used by any training pair;
  - every positive's overlap group and linked unit lie wholly in validation.
- **The 11 excluded questions:**
  - by holder: training only 2 (19, 21), blocked only 3 (05, 15, 23), both 6;
  - by blocked holder: one in an evaluation pool itself for 4 (02, 13, 20, 26), only linked to a pool for 5;
  - by match: figure matches alone excluded 3 (02, 05, 26), quote, span or fact matches 8. The figure
    matches the predictor now models no longer dominate (`val_pilot_v2e`: 9 of 22).
- **Question overlap:**

| Pool | Highest Jaccard | Question |
|---|---:|---|
| Evaluation questions | 0.25 | 05 |
| Training questions (all 150 ever generated) | 0.696 | 36 |
| `val_pilot_v1` and the 16 carried | 0.773 | 36 (Apple's Level 2 corporate debt, FY2021) |
| `val_pilot_v2e` batch 1 | 0.682 | 36 |

## 8. Actual yield against the predictor

**By stage** (planned rates from the calibration; observed in batch 1):

| Stage | Planned | Observed |
|---|---:|---:|
| Generator acceptance | 0.82 | 35 / 49 (0.71) |
| Evidence pass, of accepted | 21.6 predicted for these 35 (mean 0.62) | 24 (0.69) |
| Review acceptance, of passing | 0.86 | 18 / 24 (0.75) |
| Usable | 23.4 for these 49 (23.5 for the first 50) | **18** |

**The predictor's accuracy** (measured, out of sample):
- **AUC:** 0.803, against 0.648 for the old sentence share on the same questions.
- **Brier score:** 0.164, against 0.276 for the calibration's constant pass rate (0.44). The batch's own
  pass rate, known only afterwards, gives 0.216.

| Predicted pass | Accepted questions | Predicted | Observed |
|---|---:|---:|---:|
| Below 0.4 | 9 | 2.5 | 3 |
| 0.4–0.6 | 7 | 3.6 | 4 |
| 0.6–0.8 | 9 | 6.3 | 8 |
| 0.8 and above | 10 | 9.2 | 9 |

**Where the shortfall is:** the yield model multiplies the predicted pass by a constant generator and
review acceptance. Neither was constant in this batch.

| | ≤ 2 significant figures | 3 or more |
|---|---:|---:|
| Passages | 25 | 24 |
| Mean predicted pass | 0.78 | 0.56 |
| Generator accepted | 13 | 22 |
| Evidence pass (predicted) | 10 (9.4) | 14 (12.2) |
| Review accepted, of passing | 4 of 10 | 14 of 14 |
| **Usable per passage** | **0.16** | **0.58** |
| Usable per passage in `val_pilot_v1` and `v2e` batch 1 | 0.41 (15 / 37) | 0.25 (16 / 63) |

**The mechanism:**
- **Where they rank:** with no significant figure, a passage's score is its sentence share alone, so a
  passage with unique sentences reaches the cap of 0.95.
- **What they are:** in this batch, those passages were risk-factor lists, administrative text and
  footnotes, and the generator and the review lost most of them:
  - 12 of the 14 generator rejections;
  - all 6 evidence-passing questions the review did not accept.
- **Why there is no rule for it:** the earlier pilots show the opposite pattern, so 25 passages do not
  establish it. It is recorded here to check after batch 2, not used.

**Re-estimate for the 312 remaining passages** (projections):

| Model | Remaining | Batch 2 (positions 50–99) | Total with the 49 usable |
|---|---:|---:|---:|
| As planned (calibration rates) | 64.2 | 16.6 | 113 |
| Generator and review acceptance by figure class, pooled over all 149 passages | 63.3 (bootstrap 58–68) | 16.1 | 112 |
| Scaled by batch 1's shortfall (18 / 23.4) | 49.5 | 12.8 | 99 |

## 9. Did the predictor and the administrative filter help?

**The predictor:**
- **Evidence rule: yes, measured.**
  - 69% of accepted questions passed, against 44% (p = 0.016; against `val_pilot_v2e`'s batch alone,
    p = 0.037).
  - It ranks well and is calibrated on questions it never saw.
- **Usable yield: a small gain, not established.**
  - 18 from 49 passages, against the 15.8 projected for `val_pilot_v2e`'s next 50 and the 15 and 16 of the
    earlier batches.
  - The evidence gain was mostly spent on lower generator and review acceptance.

**The administrative filter:**
- **The targeted types are gone:** no 10b5-1 plan, insider-trading policy or cybersecurity-governance
  passage reached the batch. `val_pilot_v2e`'s batch had four, none usable.
- **Three other administrative passages came through:**
  - 29, a Section 13(r) disclosure in Item 9B (generator-rejected);
  - 39, shareholder services on Pfizer's website;
  - 41, Walmart's Item 6 "reserved".
  - None is usable. 39 and 41 also passed the evidence rule: no train or blocked passage holds their
    text, so the predictor ranked them near the top.
- **Few remain:** a scan of the remaining 312 passages finds one more website-availability passage
  (position 87), two critical audit matters (69, 138) and one quarter-only table (110). A further rule would
  save about four requests, so none is proposed now.

## 10. Systematic weaknesses of Qwen's outputs

1. **Year order in "… were $X and $Y as of December 31, 2023 and 2024":** Amazon lists the earlier year
   first, and the earlier figure is given as the later year's (02, 23).
2. **Prior-period tables with no visible year:** 2019's quarters read as 2020's (26).
3. **Sign conventions:** a "(Charge)/credit" row's positive value is called a charge (14).
4. **Footnotes that change a row's meaning:** 32.
5. **Context above the window:** a footnote's product (18), as `val_pilot_v2e`'s segment heading (45).
6. **Long list quotes:** `evidence_length` 7, against 2 in `val_pilot_v2e`'s batch. The predictor ranks
   risk-factor lists highly.
7. **Administrative and trivial questions:** 39, 41 and 45.
8. **Template repetition across years:** Apple's securities table (01, 36, 43).
9. **Units dropped** for tables "in millions" (8 answers).
10. **Better than in `val_pilot_v2e`:**
    - the company name is left out in 2 of 49 (4 of 50 before);
    - no false generator rejection.

## 11. Comparison with earlier pilots

| | `val_pilot_v1` | `val_pilot_v2e` batch 1 | `val_pilot_v2f` batch 1 |
|---|---|---|---|
| Plan | Balanced years, no score | Balanced years, isolation share ≥ 0.3 | Balanced years, ranked by `evidence-pass-v1`; administrative rule |
| Passages completed | 50 | 50 | 49 (50 requests) |
| Generator acceptance | 43 (86%) | 39 (78%) | 35 (71%) |
| Evidence pass, of accepted | 19 (44%) | 17 (44%) | **24 (69%)** |
| Review: accept / revise / reject, of accepted | 36 / 5 / 2 | 36 / 1 / 2 | 26 / 4 / 5 |
| Factual errors, of accepted | 5 (12%) | 1 (3%) | 6 (17%) |
| Usable per passage | 16 (32%) | 15 (30%) | **18 (37%)** |
| Tokens per passage | 1,344 | 1,322 | 1,314 |

## 12. Recommendation for batch 2

**Batch 2 may proceed on `val_pilot_v2f` unchanged** (positions 50–99, one batch of at most 50 requests),
with every question AI-reviewed as here.
- **The safeguards held:**
  - no leakage, no question overlap, no unit crossing the split;
  - every cap respected;
  - the evidence-rule pass rate is significantly higher.
- **Why no new plan version:**
  - what was lost was lost in generation and review, which the review catches;
  - few remaining passages share batch 1's specific failure patterns;
  - a rule demoting low-figure passages would rest on 25 passages and contradict the earlier pilots.

**What to expect from batch 2** (projections):
- **Usable questions:** about 13–16.
- **Cost:** about 64,000 tokens and 25 minutes at batch 1's pace.
- **Composition:**
  - it starts at position 50;
  - 7 or 8 passages per company, 8 from FY2019;
  - 16 with two or fewer significant figures.

**What changes:**
- **The total:**
  - The whole plan now projects about 99–113 usable questions in all (§8), not about 119.
  - The low end is near the earlier stress case (102).
  - 120 needs either the optimistic case or a supply change: a unit cap of 4 or 5, or a larger validation
    split. This decision does not block batch 2.
- **Recalibration after batch 2,** as planned (about 180 observations):
  - estimate generator and review acceptance by passage class, not as constants;
  - test the figure-count effect on new data.
- **Watch:**
  - Pfizer (2 usable so far; 51 passages remain, about 10 expected);
  - Apple's securities note (6 of Apple's 9).
  - When the final set is built, a cap on questions per note template may be worth deciding.
- **Recording batch 1:** the decisions above can go into versioned files in `data/validation_review/`,
  in the format of `val_pilot_v2e`'s batch, once you approve them.
