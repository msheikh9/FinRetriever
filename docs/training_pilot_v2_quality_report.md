# Training-data pilot `pilot_v2`: quality audit

Status: 2026-10-30. This is an audit of the first real synthetic-query pilot:
`qwen/qwen3.8-27b`, prompt `sec-qgen-v1`, validator `sec-qcheck-v2`.

- Every decision below is an AI review (Claude), not human annotation, and
  none of it is a gold label.
- Nothing was generated, trained, changed or committed for this audit. The
  records, pairs and evaluation data are as `build-pairs` left them, and
  their checksums match before and after.

## 1. Decision

**NO-GO for full-scale generation. GO for a second, larger pilot once the
fixes in §8 are in.**

- **What passed is mostly good.** 33 of the 37 pairs are accepted as they
  stand, 3 need revision and 1 should be rejected. No accepted pair invents a
  figure. One misreads a table column and one misdates an accounting change.
- **The validator errs on the strict side.** Of the 11 `pilot_v2`
  rejections, 8 were false: the answers were right, but a table quote left
  out cells or the question reused an accounting term. Two are judgment
  calls, and one was correct (an invented quote).
- **The main risk is in the pairs' neighbours, not the questions.**
  - 29 of the 37 pairs have their evidence or labelled figure in at least
    one other corpus passage. For 25 of them that passage lies outside the
    pair's overlap group, so the planned group-aware sampler will not stop it
    acting as a false negative.
  - For 6 pairs the same evidence or figure sits in passages pooled for the
    evaluation questions. The passage-level blocklist cannot see that.
- **Groq's limit makes a full run long.** At 1,000 output tokens per minute
  the paced resume managed about 2 requests per minute. Every training and
  validation passage would take about 61 hours; one passage per overlap
  group, about 29.

## 2. Provenance

| Check | Result |
|---|---|
| Frozen corpus | `corpus_v1@1.2`, SHA-256 `ff9dadf3…4004`, 10,137 passages; matches the config |
| Preparation | Manifest `27550ad1…777f`, and every prepared file matches it; config `735cfbf3…5153` |
| Generation run | `train.run.json` `b20daae5…dd40`: prompt `sec-qgen-v1`, `qwen/qwen3.8-27b`, the same manifest |
| Continuation | Continuation 1 (`7933fdb8…6e14`) changes only request settings: `max_completion_tokens` 2048 → 768, 1,000 output tokens per minute, a 180 s wait cap. It counted 16 records with SHA-256 `ac9f934d…`, the file as it stood before the resume, so no earlier record was rewritten |
| Records | 25 passages done, 0 failed. 24 are on validator v2; 1 (`pilot_v1`'s Apple record) cannot be revalidated |
| Pairs | `train_pairs.jsonl` `605e1ab0…0180` matches `pairs.meta.json`, and `check` finds 0 violations. `validation_pairs.jsonl` is empty |
| Every pair (37/37) | Positive is in the corpus and an approved train source; its overlap group lies wholly in train with no member on the blocklist; it is in the `pilot_v2` plan; its record and parameter hash match |
| BGE visibility | Every evidence quote lies within the first 512 tokens: the latest ends at token 489 (real BGE tokenizer, local cache) |
| Sources | The 37 pairs come from 22 passages in 22 overlap groups |
| `pilot_v1` Apple exhibit index | Done with 0 accepted (both questions rejected by the validator v1 date bug). **Contributed no pairs.** It is now excluded by the source filter and cannot be revalidated, because its answers were not stored |

## 3. Per-pair decisions

Each pair was read against its full visible passage.
- **Accept:** clear, correct company and period, answer supported by the
  quote, figures with the right unit and date, useful as a retrieval query.
- **Revise:** a specific defect that a regenerated or edited question could
  fix. No question was changed.
- **Reject:** no value for financial retrieval.

Flags record what the pair's evidence (or its labelled figure, for numeric
answers) shares with *other* passages of the same company:
- **G:** in another passage of the same overlap group (an in-batch false
  negative unless batching is group-aware).
- **X:** in a train passage outside the group (a false negative that
  group-aware batching does not prevent).
- **E:** in a passage pooled for the evaluation questions (blocked).
- **V:** in a validation passage.

| Pair | Source | Type | Question (verbatim) | Answer (verbatim) | Decision | Flags | Note |
|---|---|---|---|---|---|---|---|
| 001 | AAPL FY2019, Item 1A | risk | What specific operational issues regarding Apple Inc.'s service offerings are identified as risks in their 2019 Form 10-K filing? | Quality issues, outages, service slowdowns, or errors. | **Accept** | G E | Supported. |
| 002 | AAPL FY2019, Item 1A | risk | According to Apple Inc.'s 2019 Form 10-K, what potential financial consequences could arise from failure to detect and fix hardware or software defects? | Product liability claims, recalls, write-offs, and significant warranty and litigation costs. | **Accept** | G X E | Supported; mild inference linking unfixed defects to liability costs. |
| 003 | AAPL FY2020, Item 8 | financial_table | What was the reported net income for Apple Inc. for the fiscal year ended September 26, 2020? | 57,411 | **Accept** | G X | Correct ($ millions); the answer omits the unit. |
| 004 | AAPL FY2021, Item 8 | financial_table | What was the total amount of right-of-use assets for Apple Inc. as of the fiscal year ended September 25, 2021? | $10,948 million | **Accept** | X | Correct; strong table question. |
| 005 | AAPL FY2021, Item 8 | other | How much did Apple Inc. report in lease costs associated with variable payments for its leases in 2021? | $12.9 billion | **Accept** | X | Correct; 6-word copy. |
| 006 | AAPL FY2024, Item 1A | risk | According to Apple Inc.'s 10-K for the fiscal year ended 2024-09-28, what proportion of the company's total net sales comes from operations outside the U.S.? | a majority | **Accept** | G | Supported; qualitative answer; 6-word copy. |
| 007 | AAPL FY2024, Item 1A | risk | In its fiscal year 2024-09-28 Form 10-K, Apple Inc. identifies which specific economic conditions as potential drivers of adverse consumer confidence and spending? | slow growth or recession, high unemployment, inflation, tighter credit, higher interest rates, and currency fluctuations | **Accept** | G | Supported; awkward ISO-date phrasing. |
| 008 | AMZN FY2021, Item 7 | financial_table | What was the total operating expenses for Amazon.com Inc in the fiscal year ended December 31, 2021? | $444,943 | **Accept** | X E | Correct; the same figure is in pair 000012's positive (in-batch false negative). |
| 009 | AMZN FY2021, Item 7 | mdna | How did the year-over-year percentage growth for Marketing expenses change for Amazon.com Inc between the fiscal years ended December 31, 2020 and December 31, 2021? | It increased from 17% to 48% | **Accept** | – | Correct reading of the growth rows. |
| 010 | AMZN FY2022, Item 8 | financial_table | What was the net carrying value of Amazon's contract-based finite-lived intangible assets as of December 31, 2022? | 2,848 | **Accept** | X | Correct ($ millions). |
| 011 | AMZN FY2022, Item 8 | other | What was the weighted average life remaining for Amazon's marketing-related intangible assets in fiscal year 2022? | 18.6 | **Accept** | X | Correct (years); typed `other` though tabular. |
| 012 | AMZN FY2023, Item 8 | financial_table | How much did Amazon.com Inc. report as fulfillment costs in its 2022 annual results compared to 2021? | 84,299 in 2022 and 75,111 in 2021 | **Accept** | G X E | Correct; asks only for prior-year comparatives. |
| 013 | AMZN FY2024, Item 8 | other | Which segment of Amazon.com Inc. was primarily impacted by the change in useful life for server and networking equipment in 2024? | AWS segment | **Accept** | X | Correct. |
| 014 | JPM FY2019, MD&A, accounting developments | other | How did JPMorgan Chase & Co plan to phase in the impact of the CECL adoption on retained earnings for regulatory capital purposes during the period from 2020 to 2023? | At 25 percent per year in each of 2020 to 2023 | **Accept** | X | Strong: a specific regulatory-capital mechanism. |
| 015 | JPM FY2022, Note 12, loans | financial_table | What was the total amount of loans sold by JPMorgan Chase & Co in the fiscal year ended December 31, 2022? | $44,818 million | **Accept** | G X | Correct (carrying value of retained loans sold, $ millions). |
| 016 | JPM FY2022, Note 12, loans | other | How much accrued interest receivable did JPMorgan Chase & Co exclude from its loan totals at December 31, 2022? | $5.2 billion | **Accept** | X | Correct footnote figure. |
| 017 | JPM FY2023, Note 12, loans | other | How frequently does JPMorgan Chase & CO update the collateral values for commercial real estate loans in its 2023 fiscal year reporting? | Every six to twelve months | **Accept** | G | Correct; loan-policy text repeated in all seven JPM filings. |
| 018 | JPM FY2023, Note 12, loans | other | What valuation method does JPMorgan Chase & CO generally apply to estimate the fair value of illiquid securities or other financial assets serving as collateral in its 2023 fiscal year? | A discounted cash flow model | **Accept** | G | Correct; same repeated policy text. |
| 019 | JPM FY2024, MD&A, results of operations | other | What were the primary factors that caused JPMorgan Chase & Co to report a higher effective tax rate in fiscal year 2024 compared to 2023? | Adoption of new tax credit accounting guidance and a higher level of pretax income with changes in income mix, including Visa share gains. | **Accept** | – | Strong: causal MD&A question. |
| 020 | JPM FY2024, MD&A, results of operations | financial_table | How did the First Republic acquisition affect JPMorgan Chase & Co's income tax expense in the prior year relative to fiscal year 2024? | It reduced the effective tax rate in the prior year because the impact was reflected in the estimated bargain purchase gain. | **Accept** | – | Supported; typed `financial_table` but there is no table. |
| 021 | PFE FY2023, Item 7 | financial_table | What was the total cost of sales reported by Pfizer Inc for the fiscal year ended December 31, 2022? | $34,344 million | **Accept** | G X E V | Correct GAAP-reported figure; prior-year comparative. |
| 022 | PFE FY2023, Item 7 | financial_table | How much net income did Pfizer Inc attribute to its common shareholders for the year ended December 31, 2022? | $31,372 million | **Accept** | G X V | Correct; prior-year comparative. |
| 023 | PFE FY2025, Item 8 | financial_table | What was the beginning carrying value of Pfizer's equity-method investment in Haleon at the start of the 2024 fiscal year? | $11,451 million | **Accept** | – | Correct; the FY2025 note's 2024 table. |
| 024 | PFE FY2025, Item 8 | other | Why did Pfizer stop applying the equity method to its stake in Haleon during 2024? | Loss of ability to exercise significant influence after the October 2024 sale | **Accept** | G X | Strong: reason for an accounting change. |
| 025 | TSLA FY2020, Item 1A | other | What specific action did Tesla, Inc. take on April 26, 2019, to modify the previously agreed-upon settlement terms regarding disclosure procedures? | Amended the settlement to clarify certain previously-agreed disclosure procedures | **Accept** | G X | Supported; legal/governance fact from a 25-passage group. |
| 026 | TSLA FY2022, Item 8 | financial_table | What is the specified range of estimated useful lives for solar energy systems in service for Tesla, Inc. in fiscal year 2022? | 30 to 35 years | **Accept** | G X | Correct; policy table repeated across years. |
| 027 | TSLA FY2022, Item 8 | other | In which section of the consolidated statements of operations does Tesla, Inc. recognize impairment losses for digital assets for the fiscal year ended 2022-12-31? | Restructuring and other | **Accept** | G X | Correct; policy text repeated across years. |
| 028 | TSLA FY2025, Item 8 | financial_table | What was the adjustment amount applied to Tesla's digital assets, net balance due to adopting ASU 2023-08 in the fiscal year ended 2025-12-31? | $303 million | **Revise** | X | Says ASU 2023-08 was adopted in the fiscal year ended 2025-12-31; the passage says January 1, 2024. The $303 million is right. |
| 029 | TSLA FY2025, Item 8 | other | How did Tesla, Inc. apply the new crypto assets standard effective January 1, 2024, as disclosed in its Form 10-K for the fiscal year ended 2025-12-31? | Modified retrospective approach | **Accept** | X | Correct; the period is anchored properly. |
| 030 | WMT FY2020, Item 7 | debt_liquidity | What were the specific dollar amounts for Walmart Inc.'s free cash flow in fiscal years 2020, 2019, and 2018? | $14.6 billion, $17.4 billion and $18.3 billion | **Accept** | G V | Strong: three years of free cash flow. |
| 031 | WMT FY2020, Item 7 | mdna | According to Walmart Inc.'s MD&A, what factors contributed to the decrease in net cash provided by operating activities in fiscal 2020 compared to the prior year? | Contribution to Asda pension plan, full year of Flipkart operations, and timing of vendor payments | **Accept** | X V | Strong causal question; 7-word copy. |
| 032 | WMT FY2025, Item 8 | financial_table | What were the total net sales for Walmart U.S. in fiscal year 2025, broken down by grocery, general merchandise, health and wellness, and other categories? | Grocery: $276,003 million; General merchandise: $113,921 million; Health and wellness: $62,092 million; Other: $10,399 million; Total: $462,415 million | **Accept** | – | Correct; long enumerating question. |
| 033 | WMT FY2025, Item 8 | revenue | How much of Walmart International's total net sales was attributed to eCommerce in fiscal 2024? | approximately $24.8 billion | **Accept** | X E | Correct; prior-year figure. |
| 034 | WMT FY2026, Item 1 | other | Where can investors access Walmart Inc.'s filings with the SEC, including its Form 10-K for the fiscal year ended 2026-01-31, free of charge? | On or through the corporate website at www.stock.walmart.com | **Reject** | G X | Boilerplate on where SEC filings are posted; in every Walmart 10-K (16-passage group); no financial content. |
| 035 | WMT FY2026, Item 1 | other | To which specific group of executives does Walmart Inc. provide recurring updates regarding workforce representation data for the fiscal year ended 2026-01-31? | Senior leadership, including the President and CEO, and members of the Board of Directors | **Revise** | – | Calls the Board of Directors "executives"; human-capital text with little financial content. |
| 036 | XOM FY2019, Item 5 | other | What did Exxon Mobil Corp state in its February 2, 2016 earnings release regarding the suspension of purchases intended to reduce outstanding shares? | Purchases to reduce shares outstanding were suspended effective beginning the first quarter of 2016 | **Accept** | – | Correct; names the 2016 release, not the filing year. |
| 037 | XOM FY2021, Note 17, pensions | financial_table | How much in contributions did Exxon Mobil Corp expect to make for its non-U.S. other postretirement benefits in 2022? | 405 | **Revise** | – | Misreads the flattened table: 405 is the non-U.S. pension contribution; both other-postretirement columns are "—". |

**Totals:** 33 Accept, 3 Revise, 1 Reject.

Flags:
- G 17, X 23, E 6, V 4.
- 29 pairs have at least one flag.
- 25 have a duplicate outside their overlap group.

## 4. Strong and weak pairs

**Strong.** These questions need the passage's meaning, not only its words,
and name a specific fact.
- **024 (PFE FY2025).** "Why did Pfizer stop applying the equity method to
  its stake in Haleon during 2024?" The answer is the loss of significant
  influence after the October 2024 sale.
- **014 (JPM FY2019).** The 25%-a-year phase-in of the CECL retained-earnings
  impact for regulatory capital, 2020–2023.
- **019 (JPM FY2024).** The drivers of the higher 2024 effective tax rate.
- **031 (WMT FY2020).** Why operating cash flow fell in fiscal 2020: the
  Asda pension contribution, a full year of Flipkart, and vendor payment
  timing.
- **004 (AAPL FY2021).** Total right-of-use assets, $10,948 million.

**Weak.**
- **034 (WMT FY2026).** Where Walmart's SEC filings are posted. Every
  10-K says this, and it is in a 16-passage overlap group.
- **037 (XOM FY2021).** The question asks for "non-U.S. other postretirement
  benefits" contributions and answers 405. The flattened header reads
  `Pension Benefits | Other Postretirement Benefits U.S. | Non-U.S. | Gross |
  Medicare Subsidy Receipt`. So 405 is the non-U.S. *pension* figure, and
  both postretirement cells are "—". This is the one factual error among
  accepted pairs, caused by how the table was flattened.
- **028 (TSLA FY2025).** It attributes adopting ASU 2023-08 to "the fiscal
  year ended 2025-12-31"; the passage says it was adopted on January 1, 2024.
  Prompt rule 2 ("name the fiscal year in every question") pushes the model
  to attach the filing's year to facts from other periods.
- **008 and 012 (AMZN).** 008 asks for Amazon's 2021 total operating
  expenses ($444,943 million). That figure is also in 012's positive, the
  FY2023 income statement, so a batch holding both treats a correct passage
  as a negative.

## 5. Rejection analysis

| # | Source | Reason | Question (verbatim) | Answer | Verdict | Why |
|---|---|---|---|---|---|---|
| 1 | AAPL FY2025, Item 15 exhibit index (`pilot_v1`) | `number_not_in_passage` | What are the specific coupon rates and maturity years listed for the global notes described in Apple Inc.'s Exhibit 4.24 for the fiscal year ended 2025-09-27? | not stored | **False rejection** | Validator v1 date bug (fixed in v2). Answer and evidence were not stored, so it cannot be re-judged in full. |
| 2 | AAPL FY2025, Item 15 exhibit index (`pilot_v1`) | `number_not_in_passage` | Which trustee is named in the indenture dated October 28, 2021, referenced in Apple Inc.'s Form 10-K for the fiscal year ended 2025-09-27? | not stored | **False rejection** | Same date bug. |
| 3 | AMZN FY2023, Item 8 | `evidence_not_in_visible_text` | What was the total operating expense reported by Amazon.com Inc. for the fiscal year ended December 31, 2023? | 537,933 | **False rejection** | 537,933 is right; the quote left out the 2021 and 2022 cells of the row. |
| 4 | TSLA FY2020, Item 1A | `evidence_not_in_visible_text` | According to Tesla, Inc.'s 2020 annual report, how have dealer associations challenged the company's ability to sell vehicles in certain U.S. states? | By challenging regulator decisions on compliance with state laws and proposing laws to prevent dealer licenses | **False rejection** | Both fragments are verbatim; the model joined them with "...". |
| 5 | XOM FY2021, Note 17, pensions | `evidence_not_in_visible_text` | What was the fair value of plan assets for Exxon Mobil Corp's U.S. funded pension plans where the accumulated benefit obligation exceeded plan assets in 2021? | 13,266 | **False rejection** | 13,266 is right; the quote rebuilt a row label that the flattened table shortens. |
| 6 | PFE FY2024, Item 8 | `copied_from_passage` | What was the total cash paid for income taxes, net of refunds, by Pfizer Inc for the fiscal year ended December 31, 2024? | $3,605 million | **False rejection** | The copied run is the line item "cash paid for income taxes, net of refunds". |
| 7 | PFE FY2024, Item 8 | `copied_from_passage` | For Pfizer Inc's fiscal year 2024, what was the impact of the Transition Tax liability on the reconciliation of the U.S. statutory income tax rate to the effective tax rate? | (6.0)% | **Uncertain** | A 10-word copy of the table's title; a standard phrase, but long. |
| 8 | XOM FY2019, Item 5 | `evidence_not_in_visible_text` | According to the issuer purchases table in Exxon Mobil Corp's 2019 Form 10-K, what was the total number of shares purchased by the company during the quarter ended December 31, 2019? | 0 | **Correct** | The quoted row "Total \| - \| - \| - \| -" does not exist. The answer 0 is supported by the prose, so the quote was invented, not the fact. |
| 9 | AMZN FY2024, Item 8 | `evidence_not_in_visible_text` | What amount of cash did Amazon.com Inc. pay for operating leases in the fiscal year ended December 31, 2024? | $12,341 million | **False rejection** | $12,341 million is right; the quote left out the 2022 and 2023 cells. |
| 10 | JPM FY2019, MD&A, accounting developments | `copied_from_passage` | What was the net increase to the allowance for credit losses reported by JPMorgan Chase & Co upon adopting the CECL guidance on January 1, 2020? | $4.3 billion | **False rejection** | The copied run is the accounting term "net increase to the allowance for credit losses". |
| 11 | XOM FY2020, auditor's report | `copied_from_passage` | What was Exxon Mobil Corp's consolidated upstream property, plant and equipment, net balance as of December 31, 2020, and what was the related depreciation and depletion expense for the same fiscal year? | The balance was $167.5 billion and the expense was $41.4 billion. | **False rejection** | The copied run is "the related depreciation and depletion expense for the". |
| 12 | XOM FY2020, auditor's report | `copied_from_passage` | Why did the auditors determine that the impact of proved oil and natural gas reserves on upstream PP&E was a critical audit matter for Exxon Mobil Corp in fiscal year 2020? | Due to significant management judgment in estimating reserves and a high degree of auditor judgment in evaluating the data, methods, and assumptions used. | **Uncertain** | Copies 13 words: the critical audit matter's name. Rejecting is defensible. |
| 13 | AAPL FY2020, Item 8 | `evidence_not_in_visible_text` | How much depreciation and amortization did Apple Inc. record as an adjustment to reconcile net income for the fiscal year ended September 28, 2019? | 12,547 | **False rejection** | 12,547 is right; the quote left out the 2020 cell. |

- **Verdicts:** 10 false rejections, 2 uncertain, 1 correct.
  - The 2 `pilot_v1` rejections came from the date bug that validator v2
    fixed.
  - Of the 11 `pilot_v2` rejections: 8 false, 2 uncertain, 1 correct.
- **No figure was invented.** Every figure in all 48 `pilot_v2` proposals
  appears in its passage with the meaning the question gives it, except
  pair 037's misattributed column.
- **Genuine model errors:** one invented quote (#8, correctly rejected), one
  misread column (037) and one misdated adoption (028). The last two were
  accepted; the validator cannot catch either.
- **Over-strict validation.**
  - `evidence_not_in_visible_text` fired 5 times on correct answers: three
    table quotes that skipped cells, one quote of two fragments joined by
    "...", and one rebuilt row label.
  - `copied_from_passage` fired 3 times on accounting terms or line-item
    names that an analyst would type anyway.
- **What a fix would recover.** Matching quoted fragments in order and table
  cells as a subset of one row would recover 4 of the 13. Not counting
  line-item names and standard terms as copying would recover 3 more, which
  would take the pilot from 37 to 44 of 48 accepted.

## 6. Distribution

**Accepted pairs (37)**

| Company | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---|---|---|---|---|---|---|
| Pairs | 7 | 6 | 7 | 4 | 5 | 6 | 2 |

| Fiscal year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| Pairs | 4 | 4 | 5 | 6 | 5 | 5 | 6 | 2 |

| Type | other | financial_table | risk | mdna | debt_liquidity | revenue |
|---|---|---|---|---|---|---|
| Pairs | 15 | 14 | 4 | 2 | 1 | 1 |

- **Types are noisy.** 41% are `other`, and some labels are wrong: 011 is
  tabular but typed `other`, and 020 is typed `financial_table` with no
  table. Types are not used in training.
- **Question length:** 21.3 words on average (median 21, range 15–30).
- **Copying.**
  - The longest copied run per question is 4 words or fewer for 23
    questions and 5 for 9.
  - 5 questions copy 6–7 words, breaking the prompt's five-word rule but
    under the filter's 8.
- **Periods.** Every question names a year.
  - 7 use ISO dates ("2024-09-28"), 12 written full dates and 14 a "fiscal
    YYYY" form.
  - None of the 10 evaluation questions uses an ISO date.
- **Duplicates:** none. The highest word Jaccard between two questions is
  0.48, from shared templates such as "What was the total … for the fiscal
  year ended December 31, 2022?".
- **Boilerplate.** 9 pairs (006, 007, 017, 018, 025, 026, 027, 034, 035) ask
  about text that recurs across all 7 fiscal years in its overlap group.
  These groups hold 7–25 passages.
- **Zero accepted:** 3 of 25 completed passages.
  - `pilot_v1`'s Apple exhibit index (the date bug).
  - Pfizer's FY2024 tax note: both questions copied a line-item name or a
    table title.
  - ExxonMobil's FY2020 auditor's report: both copied wording from the
    critical audit matter.
- **Overlap groups:** 13 of the 22 positive passages belong to multi-passage
  groups (2–25 passages), and 6 of those groups span 6–7 fiscal years.
- **Answers.** 5 answers give bare numbers without the unit the passage
  states (003, 008, 010, 011, 037). Answers are not used in training.

## 7. Remaining leakage and false-negative risks

1. **Evidence reaches evaluation pools below the grouping threshold (E, 6
   pairs).**
   - The blocklist removes evaluation-pool passages and their overlap
     groups, where a group needs at least 50% of 5-grams shared.
   - Adjacent chunks and year-to-year rewrites share less. The positives of
     001 and 002 share 16–31% of their 5-grams with blocked Apple
     risk-factor chunks (FY2019 chunk 5, FY2020, FY2021) yet repeat whole
     sentences of them. 012's evidence row is in a blocked Amazon FY2022
     passage.
   - None of the 6 asks what an evaluation question asks. They do train the
     model on text inside evaluation pools.
2. **Train–validation fact duplication (V, 4 pairs).** For example, 031's
   evidence sentence is in a validation passage (Walmart FY2021 MD&A) that
   shares only 7% of its 5-grams with the positive. Group-disjoint splits do
   not make the facts disjoint.
3. **False negatives within groups (G, 17 pairs).** The training plan's
   group-aware sampler covers these. It does not exist yet, and without it
   the plan's fallback is one passage per group.
4. **False negatives across groups (X, 23 pairs; 25 with E or V included).**
   - Prior-year comparatives (012, 021, 022, 033), headline figures
     (003, 008) and repeated policy text are all stated in other filings.
   - At 128 queries per batch with in-batch negatives, and with A2 mining
     same-company negatives, these passages will often be scored as
     negatives.
   - Excluding only the positive's group and its same-section neighbours,
     as the plan does, misses most of them.
5. **Year-only distinctions.** For the 9 boilerplate pairs, the visible
   text names no year, so the year in the question comes only from metadata
   that BGE cannot read. Questions for every year's copy would differ only
   in the year.
6. **Semantic leakage against evaluation questions.**
   - The lexical maximum is 0.263 (word Jaccard; 0 pairs above 0.5).
   - Read side by side with the 10 evaluation questions, no pair paraphrases
     one. The nearest topics are different facts: 013 (AWS server
     useful-life change, 2024) vs `q_0003` (AWS profit change, 2023); 032
     (Walmart U.S. sales by category, fiscal 2025) vs `q_0008` (sales by
     segment, fiscal 2024); and 006/007 (Apple FY2024 macroeconomic risks)
     vs `q_0001` (Apple FY2024 manufacturing risks).
   - The automated check stays lexical, and `gold_queries.jsonl` is still
     empty.

## 8. Scaling recommendation

### Generation

Do not scale `sec-qgen-v1` / `sec-qcheck-v2` as they are. Make the changes
below, then run a second pilot. The current `generation/` directory stays as
the record of `sec-qgen-v1`. A new prompt is a new run, so move the
directory aside (as with the llama archive) before generating again.

### Prompt (`sec-qgen-v2`)

- **Periods:** name the period as an analyst would ("fiscal 2024", "in
  2023", "as of December 31, 2022"), never as an ISO date. Ask about the
  filing's own fiscal year unless the passage only covers other periods.
  When asking about an earlier period, do not attach the filing's year to
  it (pair 028).
- **Evidence:** copy one contiguous span exactly. For a table row, copy the
  label and every cell up to the one needed, and never shorten with "...".
- **Tables:** column headers apply to the cells beneath them. If a figure's
  column is unclear, ask about another figure (pair 037).
- **Answers:** give the unit the passage states.
- **Topics:** no questions about where documents are filed or published, or
  about website addresses.
- **Types:** split `other` (for example into accounting policy, tax,
  legal/regulatory and human capital), or drop types from the prompt.

### Validator and filters (`sec-qcheck-v3`, revalidated offline on stored candidates)

- **Evidence matching:** accept fragments joined by "...", each at least 15
  characters, verbatim and in order. Accept table quotes whose label is
  verbatim and whose cells are an in-order subset of one visible row. This
  recovers 4 false rejections; the invented quote (#8) still fails.
- **Copying:** do not count runs that are a table title or row label in the
  passage, or a short list of standard line items ("allowance for credit
  losses", "cash paid for income taxes, net of refunds"). Keep the 8-word
  limit for everything else. This recovers 3.
- **Evidence-level leakage check in `build-pairs`:** drop a pair whose
  evidence, or labelled figure, appears verbatim in any blocked passage. In
  this pilot that removes 6 pairs. When validation pairs exist, drop
  validation pairs whose evidence appears in a train passage.
- **False-negative list per pair:** record every other passage that contains
  the pair's evidence or labelled figure. The trainer should use none of
  them as an in-batch or mined negative, alongside the overlap group.
- **Source filter:** add filing-availability and website text to
  `repetitive_administrative` (pair 034's passage passed).

### Sampling

- **One passage per overlap group.** The 6,822 kept training passages
  form 3,212 groups. This halves the requests, removes year-only copies and
  matches the plan's fallback.
- **Company balance.** JPMorgan holds 1,358 of those groups (42%), while
  Apple and Amazon hold 164 and 157. Cap JPMorgan, for example at 600
  groups, unless the mix is wanted.
- **Seeded plans.** Keep writing versioned, seeded plans balanced by company
  and year.

### How much to generate, at 1,000 output tokens per minute

- **Observed rates.** Replies average 239 completion tokens (maximum 381)
  and 966 prompt tokens.
  - With 768 tokens reserved per request, the resume ran at about 2.1
    requests per minute. A second request fits in the same minute only after
    a reply of 232 tokens or fewer, and half of the replies were longer.
  - At 512 the rate would be about 3 per minute. Keep 768 until the second
    pilot shows the reply-length tail, then move to 512 if the longest
    replies stay under about 400 tokens.
- **Yield:** 1.54 accepted pairs per request in this pilot, and about 1.8
  with validator v3.

| Stage | Requests | Time at ~2.1/min | Expected pairs |
|---|---|---|---|
| Second pilot: 200 train + 100 validation groups, balanced | 300 | ~2.5 h | ~300–360 train, ~80–90 validation |
| First training set: every train group (3,212) + every validation group (416) | 3,628 | ~29 h (~20 h at 512 tokens) | ~4,900–5,800 train, ~350 validation |
| The same with JPMorgan capped at 600 groups | 2,870 | ~23 h | ~3,800–4,400 train |
| Every passage, 2 questions each (the plan's ~14.8k pairs) | 7,641 | ~61 h | ~10,500–12,300 train, ~700 validation |

- **Gate before the first training set.** Review a random 60 train pairs and
  every validation pair from the second pilot. Require:
  - at most 10% Revise or Reject;
  - no wrong figure or column;
  - no pair with evidence in an evaluation-pool passage once the new check
    runs;
  - false rejections at most a quarter of all rejections.
- **Generate from every passage only** after the group-aware sampler and the
  false-negative lists exist, and once the first fine-tune shows gains on
  the development diagnostics.
- **Daily limits are unknown.** At about 1,200 tokens per request, the first
  training set is about 4.4 million tokens. Before a multi-day run, read the
  `x-ratelimit-limit-requests` and `-tokens` headers that `generate` prints,
  or the Groq console.

### Validation set

- **Source:** the existing validation split (819 kept passages in 416
  overlap groups, disjoint from train by construction). Use one question
  per passage and one passage per group, from the same prompt and validator
  version as train.
- **Leakage check:** drop any validation pair whose evidence or labelled
  figure appears in a train passage or a blocked passage. Pair 031 shows the
  overlap is real.
- **Size:** 100 groups in the second pilot, then all 416.
- **Use:** early stopping and checkpoint choice only. It is AI-generated:
  never report it as evaluation and never label it gold. Spot-check it by
  hand before relying on it.

## 9. Limits of this audit

- **One reviewer.** One AI reviewer judged every pair against the filing's
  own text, not external sources. A human should check the 4 non-accepted
  pairs and a sample of the rest.
- **Duplicates are a lower bound.** Detection used exact evidence matches
  plus a label-and-figure heuristic, so repeats in other wording are not
  counted.
- **Semantic leakage was checked by reading,** against the 10 existing
  evaluation questions only.
- **Training uses only the query and the positive passage.** Answer defects
  (missing units, wrong column) matter for auditing and for pairs whose
  question states a false fact (037, 028), not through the answers
  themselves.
