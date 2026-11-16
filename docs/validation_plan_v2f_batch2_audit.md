# Validation plan `val_pilot_v2f`, batch 2: audit of records 50–97

Status: 2026-11-04.
- **What this is:** an AI review (Claude) of the 48 generation records of batch 2 of plan `val_pilot_v2f` in
  `bge_base_ft_v3`. It covers plan positions 50–97 only. It is not human annotation, and no decision here
  is a gold label.
- **What was not done:**
  - no generation record, plan, training pair, candidate, calibration, validation set or decision file
    was changed;
  - no validation set or validation pair was built, and no decision file was written: the decisions
    below exist only in this report until they are recorded;
  - no API request was sent and nothing was trained.
- **Checksums:** 272 files under `data/`, `configs/` and `results/` were checksummed before and after the
  audit, and none changed. Since batch 1's audit, only `validation.records.jsonl` has changed.
- **The checks:** every validation-set check (validator, evidence rule, question overlap, duplicates, unit
  isolation) was re-run in memory, as `build-validation` would run it, with nothing written.
- **Earlier documents:** [validation_plan_v2f.md](validation_plan_v2f.md) (the plan) and
  [validation_plan_v2f_batch1_audit.md](validation_plan_v2f_batch1_audit.md) (batch 1, positions 1–49).

## 1. Summary

**Batch 2 beat its estimate:**
- **The funnel:**
  - 50 HTTP attempts completed 48 passages (one 429 and one 503 were retried);
  - the generator accepted 41 (85%);
  - the evidence rule left 21;
  - the review leaves **20 usable questions**.
- **Against the plan:** it expected 16.0 from these 48 passages.
- **Both batches together:** 38 usable from 97 passages, against 39.4 expected. Batch 1's shortfall and
  batch 2's surplus cancel out.
- **Running total:** 69 usable validation questions (16 carried, 15 from `val_pilot_v2e`, 18 and 20 from
  this plan's two batches).

**Quality is the best of any batch so far:**
- **Factual errors:** 2 of the 41 accepted questions (5%), against 6 of 35 in batch 1.
  - 74 calls a pro forma adjustment an expense Pfizer recorded. It passed the evidence rule, so only the
    review keeps it out.
  - 91 calls ExxonMobil's debt due within a year "redeemed".
- **Rejections:** none. No question in the batch is administrative or trivial.
- **Company attribution:** every question names the right company.
- **Units:** one answer drops "millions", against eight in batch 1.

**The predictor held up out of sample:**
- **This batch:** it predicted 19.5 evidence passes for the 41 accepted questions, and 21 were observed.
  AUC 0.71.
- **Both batches (76 questions):** 41.1 predicted, 45 observed. AUC 0.77, against 0.65 for the old
  sentence score.
- **Batch 1's low-figure pattern did not recur:** passages with two or fewer significant figures gave 0.40
  usable questions per passage, and the others 0.42.

**Recommendation:** batch 3 may proceed on `val_pilot_v2f` unchanged, with the review kept (§12).
- **The total:** at the rates of both batches, the whole plan gives about 114–118 usable questions (a range
  of about 109–121), up from 99–113 after batch 1.
- **Reaching 120** still needs the optimistic case or a supply change.

## 2. Provenance

| Check | Result |
|---|---|
| Plan | `validation.val_pilot_v2f.json` (`a7903bc7…`), unchanged. The 48 records are its positions 50–97, in order. Position 98 (XOM FY2020) is next |
| Run | The same bound run as batch 1, no continuation: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, parameters `42d5fe54…`, validator `sec-qcheck-v3`, one question per passage |
| Records | 48 done, 0 failed, each on its first response attempt, no errors. The 429 and 503 retries spent two of the 50 requests; records do not log HTTP retries. Sent 2026-11-03, 22:32–22:53 UTC (2.3 per minute) |
| Tokens | 64,963: 58,804 prompt and 6,159 completion, 1,353 per passage. Completion 91–183, median 125, against the 768 limit |
| Files | `validation.records.jsonl` is now `45164891…`, 147 lines. Its first 99 lines are byte-identical to the file batch 1 was audited on (`e5c7ffd3…`), so the run only appended. `validation.run.json` `34e6b74d…` is unchanged |
| Passages | None repeats a passage of `val_pilot_v2e` or batch 1 |
| Validator re-run | Reproduces all 48 outcomes (41 accepted, 7 rejected, the same reasons) |
| Predictor re-run | All 48 plan scores recompute exactly from `configs/train/evidence_pass_v1.json` |

## 3. The funnel

| Stage | Kept | Removed | Why |
|---|---:|---:|---|
| HTTP attempts | 50 | | One 429 and one 503, both retried |
| Passages completed | 48 | | The two retries spent two requests; position 98 not reached |
| Generator (`sec-qcheck-v3`) | 41 | 7 | `missing_company` 3, `copied_from_passage` 1, `evidence_not_in_visible_text` 1, `answer_not_supported` 1, `evidence_length` 1 |
| Evidence rule (validation: any match outside validation) | 21 | 20 | Possible training evidence 11, evaluation evidence 6, possible evaluation evidence 6, training evidence (exact, visible) 4; some have several |
| Question overlap (evaluation, training, duplicates) | 21 | 0 | Highest Jaccard: 0.28 against an evaluation question, 0.611 against a training question (threshold 0.8) |
| AI review | **20** | 1 | Revise 74. Also revise 91, which the evidence rule had already excluded |

**Generator rejections** (7; verdict on each in §4):
- **All correct by the rule. No false rejections.**
- **No company named (53, 71, 85):** each names a product or a lawsuit (Vyndaqel, the iOS settlement,
  VirnetX) instead of the filer.
- **Caught by the verbatim check (60):** the quote says "$11.5 million" where the text says
  "$11.5 billion". The answer itself is right.
- **Caught by the window (62):** Qwen chose a table row whose value lies past the 512-token window, and
  answered that the passage does not give it.
- **Strict rejections:** 52 repeats the sentence's wording. 75's quote is 307 characters against the 300
  limit; it is a natural question otherwise.
- **Long list quotes:** one `evidence_length`, against seven in batch 1. The batch has three risk-family
  passages, and all three gave accepted questions.

**Question types accepted:** `financial_table` 18, `debt_liquidity` 6, `risk` 5, `tax` 4,
`accounting_policy` 3, `revenue` 2, `other` 2, `profitability` 1.

## 4. Per-question decisions

- **Decisions:** ACCEPT, REVISE and REJECT apply to the 41 questions the generator accepted. For the 7 it
  rejected, the note gives a verdict on that rejection.
- **"Evidence rule":** "In" means no train or blocked passage holds the evidence, exactly or possibly, in
  its full text.
- **Would enter a validation set:** 20 questions, accepted and "In": 54, 55, 57, 59, 65, 67, 73, 76, 78,
  79, 81, 83, 84, 86, 87, 90, 93, 94, 95, 97.
- **REVISE:** in this pipeline it excludes the question. Records are never edited, so a corrected question
  would need a new generation.

| # | Source | Question (verbatim) | Answer (verbatim) | Decision | Evidence rule | Note |
|---|---|---|---|---|---|---|
| 50 | AAPL FY2021, table/statements | What were the notional amounts of Apple's foreign exchange contracts designated as accounting hedges in millions of dollars for fiscal 2021 and fiscal 2020? | $76,475 million in 2021 and $57,410 million in 2020 | ACCEPT | Out: evaluation evidence, possible training evidence | Correct; both years. Apple's derivative note (also 57, 78; batch 1's 22). |
| 51 | AMZN FY2022, narrative/statements | What was the balance of long-term debt for Amazon.com Inc as of December 31, 2022? | $67,150 | ACCEPT | Out: evaluation evidence, training evidence | Correct; no unit (millions). |
| 52 | JPM FY2022, narrative/other | What proportion of JPMorgan Chase's remaining U.S. dollar LIBOR derivatives exposure is expected to be remediated through central counterparty initiatives in the second quarter of 2023? | approximately 40% | Generator: `copied_from_passage` | – | Correct by the rule: repeats the sentence. |
| 53 | PFE FY2024, narrative/business | What is the scheduled expiry date for the basic patent of Vyndaqel in Japan for the treatment of polyneuropathy? | August 2026 | Generator: `missing_company` | – | Correct: names no company (Vyndaqel is Pfizer's). |
| 54 | TSLA FY2021, table/statements | What was the amount of net operating loss carry-forwards reported as a deferred tax asset by Tesla, Inc. as of December 31, 2021, in millions? | $7,607 million as of December 31, 2021 | ACCEPT | **In** | Correct. |
| 55 | WMT FY2023, narrative/business | What were Walmart International's net sales for fiscal 2023 and what percentage of the company's consolidated net sales did this represent? | $101.0 billion, representing 17% of consolidated net sales | ACCEPT | **In** | Correct; both figures of the sentence. |
| 56 | XOM FY2022, table/other | What was the amount of Exxon Mobil's unused long-term lines of credit as of December 31, 2022? | $1.2 billion | ACCEPT | Out: evaluation evidence, possible training evidence | Correct. |
| 57 | AAPL FY2020, table/statements | What was the total fair value of foreign exchange contract derivative liabilities designated as hedge instruments for Apple in fiscal 2020? | $1,561 million | ACCEPT | **In** | Correct designated-hedge figure; 'total' echoes the Total Fair Value column ($2,046), but the qualifier selects $1,561. |
| 58 | AMZN FY2023, table/statements | What was the amount of foreign currency translation adjustments, net of tax, reported by Amazon.com, Inc. in its consolidated statements of comprehensive income for the year ended December 31, 2023? | $1,027 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct. |
| 59 | JPM FY2021, table/statements | What were the gross derivative receivables for U.S. GAAP nettable interest rate contracts designated as OTC at JPMorgan Chase & Co for the year ended December 31, 2021? | $251,953 million | ACCEPT | **In** | Correct; a balance as of December 31, asked 'for the year ended'. |
| 60 | PFE FY2022, narrative/statements | What was the total cash payment made by Pfizer to acquire the outstanding common shares of Biohaven during fiscal 2022, excluding debt repayment and preference redemptions? | approximately $11.5 billion | Generator: `evidence_not_in_visible_text` | – | Correct: the quote says '$11.5 million', the text '$11.5 billion'. The answer is right. |
| 61 | TSLA FY2024, table/statements | What was the balance of Construction in progress for Tesla, Inc. as of December 31, 2024? | $6,783 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct. |
| 62 | WMT FY2026, table/statements | What was the amount of confirmed supplier financing obligations outstanding at the end of fiscal 2026 for Walmart Inc., in millions? | The passage does not provide the final figure for confirmed obligations outstanding at the end of the year for fiscal 2026 as the text is cut off before completing the row. | Generator: `answer_not_supported` | – | Correct: the row's value lies past the 512-token window, and the answer says so. |
| 63 | XOM FY2024, table/other | What was ExxonMobil's total income tax expense, including both current and deferred portions, for fiscal 2024? | $13.8 billion | ACCEPT | Out: possible evaluation evidence | Correct. |
| 64 | AAPL FY2019, table/statements | What was Apple's effective tax rate for fiscal year 2019? | 15.9% | ACCEPT | Out: evaluation evidence | Correct. |
| 65 | AMZN FY2024, table/statements | What was the total long-term lease liability for Amazon for operating leases at December 31, 2024? | $69,050 million | ACCEPT | **In** | Correct; the quote ends at token 501 of 512. |
| 66 | JPM FY2019, narrative/statements | What factors does JPMorgan Chase consider when assigning internal risk ratings to borrowers or counterparties? | Obligor's debt capacity, financial flexibility, earnings level, repayment sources, contingencies, management strength, and industry/geography. | ACCEPT | Out: possible training evidence | Correct; a credit-policy sentence repeated in every year's filing (7 training holders). |
| 67 | PFE FY2023, narrative/statements | What was the fair value of Pfizer's investment in Haleon based on quoted market prices as of December 31, 2023? | $12.1 billion | ACCEPT | **In** | Correct. |
| 68 | TSLA FY2022, table/statements | What was the aggregate intrinsic value of Tesla's stock options at the end of the year ended December 31, 2022? | $32.79 billion | ACCEPT | Out: possible training evidence | Correct. |
| 69 | WMT FY2024, table/statements | According to the critical audit matter description for Walmart Inc. in fiscal 2024, what specific criteria does management use to record a liability for legal proceedings and regulatory matters? | Management determines it is probable that a loss has been incurred and the amount of the loss can be reasonably estimated. | ACCEPT | Out: possible training evidence | Correct; the standard loss-contingency criterion, quoted in an auditor's critical audit matter. |
| 70 | XOM FY2023, table/other | What was the total balance of ExxonMobil's accumulated other comprehensive income as of December 31, 2022, in millions of dollars? | (13,270) millions of dollars | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct prior-year balance; keeps the table's parentheses for a negative. |
| 71 | AAPL FY2021, narrative/statements | What was the aggregate payment amount for the settlement of the iOS Performance Management class actions approved in March 2021? | $310 million | Generator: `missing_company` | – | Correct: names no company (Apple's iOS settlement). |
| 72 | AMZN FY2021, table/statements | What was the total accumulated depreciation and amortization for Amazon.com, Inc. as of December 31, 2021? | 78,519 (in millions) | ACCEPT | Out: evaluation evidence | Correct. |
| 73 | JPM FY2024, table/statements | What was the fair value of auto and other consumer loans acquired by JPMorgan Chase as of May 1, 2023? | $2,030 million | ACCEPT | **In** | Correct; First Republic's loans at the acquisition date. |
| 74 | PFE FY2025, narrative/statements | What was the estimated additional interest expense Pfizer recorded for the year ended December 31, 2023, related to debt issued and commercial paper borrowings to partially finance the Seagen acquisition? | approximately $984 million | **REVISE** | **In** | $984 million is a pro forma adjustment in the unaudited supplemental results, not interest expense Pfizer recorded. |
| 75 | TSLA FY2020, narrative/statements | What was the total net cash proceeds from the public offering of common stock completed by Tesla in February 2020? | $2.31 billion | Generator: `evidence_length` | – | Correct by the rule (307 characters, limit 300); a natural question otherwise. |
| 76 | WMT FY2025, table/statements | What were Walmart's total deferred tax assets, net of valuation allowances, as of January 31, 2025? | $9,093 million | ACCEPT | **In** | Correct. |
| 77 | XOM FY2025, narrative/other | What specific year did ExxonMobil set as the target for reaching Scope 1 and Scope 2 net zero in its operated assets? | 2050 | ACCEPT | Out: possible evaluation evidence | Correct year; the filing calls it an 'ambition', not a target, in its forward-looking-statements notice. |
| 78 | AAPL FY2020, table/statements | What was the total amount of gains or losses recognized in Other Comprehensive Income included in the effectiveness assessment for Apple's cash flow hedges in fiscal 2020? | $308 million | ACCEPT | **In** | Correct: the table's total (365 and (57)). |
| 79 | AMZN FY2020, table/statements | What was Amazon's comprehensive income for the fiscal year ended December 31, 2020? | $22,137 million | ACCEPT | **In** | Correct. Same template as 86 (Jaccard 0.706). |
| 80 | JPM FY2025, narrative/statements | What was the carrying value of JPMorgan Chase & Co's Series CC preferred stock in millions as of December 31, 2025? | 1,258 million | ACCEPT | Out: possible training evidence | Correct. |
| 81 | PFE FY2021, narrative/risk | What ownership interest in the GSK Consumer Healthcare JV is Pfizer expected to hold following the demerger announced in June 2021? | 32% | ACCEPT | **In** | Correct, as the filing's expectation. |
| 82 | TSLA FY2023, table/statements | What was Tesla's total net property, plant and equipment as of December 31, 2023? | $29,725 million | ACCEPT | Out: possible training evidence | Correct. |
| 83 | WMT FY2022, narrative/business | Which specific data product suite did Walmart U.S. launch in Fiscal 2022 to serve merchants and suppliers? | Walmart Luminate | ACCEPT | **In** | Correct; a product launch. The quote ends at token 508 of 512. |
| 84 | XOM FY2019, table/business | What were the year-end 2019 gross and net operated wells for Exxon Mobil Corporation? | 27,532 gross and 23,857 net operated wells | ACCEPT | **In** | Correct; an operational count. |
| 85 | AAPL FY2019, narrative/statements | What was the total amount of damages awarded to VirnetX in the first lawsuit after post-trial proceedings, and what did the Federal Circuit decide regarding this verdict on January 15, 2019? | The damages increased to $440 million, and the Federal Circuit affirmed the verdict. | Generator: `missing_company` | – | Correct: names no company; the quote is also stitched. |
| 86 | AMZN FY2025, table/statements | What was the comprehensive income for Amazon.com, Inc. for the fiscal year ended December 31, 2025? | $105,934 million | ACCEPT | **In** | Correct. Same template as 79. |
| 87 | JPM FY2023, narrative/business | What were JPMorgan Chase's total assets and stockholders' equity as of December 31, 2023? | $3.9 trillion in assets and $327.9 billion in stockholders' equity | ACCEPT | **In** | Correct. |
| 88 | PFE FY2020, narrative/statements | In what year is Pfizer's crystalline form patent for Ibrance expected to expire? | 2034 | ACCEPT | Out: training evidence | Correct. |
| 89 | TSLA FY2025, narrative/statements | What is the effective date for annual periods for ASU No. 2025-10 regarding government grants, and how does Tesla expect this change to affect its consolidated financial statements in fiscal 2025? | Effective for annual reporting periods beginning after December 15, 2028; Tesla does not expect a material impact on its consolidated financial statements. | ACCEPT | Out: evaluation evidence, training evidence | Correct; two-part. The ASU takes effect after December 15, 2028, so 'in fiscal 2025' misframes it; the second part repeats for every ASU. |
| 90 | WMT FY2021, table/business | What was the total square footage, in thousands, for Walmart International's operations in Mexico as of January 31, 2021? | 101,993 thousand square feet | ACCEPT | **In** | Correct; Mexico's counts are as of December 31, 2020 (footnote 1). |
| 91 | XOM FY2021, table/other | How much of Exxon Mobil's long-term debt maturing in 2022 had been redeemed by December 31, 2021? | $2,392 million | **REVISE** | Out: possible training evidence | $2,392 million is debt maturing within one year, moved to current liabilities, not debt redeemed (the redeemed 2.397% notes were $1,150 million). |
| 92 | AAPL FY2021, narrative/statements | What was the balance of Apple's non-current term debt as of September 25, 2021? | $109,106 million | ACCEPT | Out: training evidence | Correct. |
| 93 | AMZN FY2019, table/statements | What was Amazon's reported net income for the fourth quarter of the year ended December 31, 2019? | $3,268 million | ACCEPT | **In** | Correct; the column headings name 2019. |
| 94 | JPM FY2020, narrative/risk | When did JPMorgan Chase cease its participation in the Primary Dealer Credit Facility (PDCF) in 2020? | May 2020 | ACCEPT | **In** | Correct. |
| 95 | PFE FY2019, narrative/business | What is the maximum fine percentage of global annual revenue that Pfizer faces for noncompliance with the EU's General Data Protection Regulation? | 4 percent | ACCEPT | **In** | Correct; a GDPR rule, not a Pfizer-specific figure. |
| 96 | TSLA FY2019, narrative/risk | What is the financial penalty Tesla must pay to the SUNY Foundation for each year it fails to meet its obligations under the Gigafactory New York agreement? | $41.2 million per year | ACCEPT | Out: possible evaluation evidence | Correct; the filing calls it a 'program payment'. |
| 97 | WMT FY2020, table/business | What was the combined percentage of annual net sales from fuel and financial services offerings for Walmart U.S. in fiscal 2020? | less than 1% | ACCEPT | **In** | Correct. |

## 5. Aspects reviewed (41 accepted questions)

| Aspect | Pass | Minor | Fail | n/a | Notes |
|---|---:|---:|---:|---:|---|
| Factual correctness | 37 | 2 | 2 | 0 | Fail: 74 (a pro forma adjustment called recorded interest expense), 91 (debt due within a year called redeemed). Minor: 57 ("total" echoes another column; the qualifier selects the right cell), 77 ("target" for the filing's "ambition") |
| Financial relevance | 34 | 7 | 0 | 0 | Minor: 69 (the standard loss-contingency criterion), 77 (a forward-looking-statements notice), 83 (a product launch), 84 (well counts), 89 (a new accounting standard), 90 (square footage), 95 (a GDPR rule, not a Pfizer figure) |
| Company attribution | 41 | 0 | 0 | 0 | Every question names the filer correctly |
| Fiscal year | 38 | 3 | 0 | 0 | Minor: 59 ("for the year ended" for a balance), 89 ("in fiscal 2025" for a standard effective after December 15, 2028), 90 (Mexico's counts are as of December 31, 2020) |
| Figures | 32 | 1 | 1 | 7 | Fail: 91. Minor: 70 keeps the table's parentheses for a negative balance |
| Units | 33 | 1 | 0 | 7 | Minor: 51 gives no unit (millions); none gives a wrong unit |

**Numerical accuracy:**
- **Every figure in an accepted answer was checked against its cell or sentence.** Apart from 91, each is
  the right number, for the right row, column and date.
- **Batch 1's failure modes did not recur:**
  - no year-order slip in "… $X and $Y as of 2023 and 2024";
  - no sign-convention error;
  - no prior-period table read as the current year's: 93's quarterly table names 2019 in its headings.

**Evidence support:**
- **The quote matcher accepts every quote.** All are verbatim except 89's, which joins two fragments with
  "…".
- **Answers:** every answer is supported by its quote. Two questions mislabel what the figure is:
  - 74: the passage's heading, "unaudited supplemental pro forma", is visible;
  - 91: the quote itself says "matures within one year".

**Visibility within BGE's 512-token limit:**
- **Where the text comes from:** the visible text is the passage truncated to the tokenizer's window. 47
  of 48 passages are truncated; 63 is complete at 388 tokens. Every quote lies inside the window.
- **Where the quotes end:** the latest ends at token 508 of 512 (83). Six end after token 470: 65 (501),
  81 (496), 93 (493), 95 (481), 86 (476) and 83. 89's fragments end at token 315.
- **No quote is cut off:** 83's sentence ends inside the window; the two words after it are the last
  visible.
- **The one question that needed text past the window was rejected:** 62's row value lies beyond token
  512, and the validator rejected the answer.

## 6. Passages that start mid-table or mid-section

All 48 passages passed the context rules (`headless_table_start`, `fragment_opening`).

| Opening | Passages | Accepted | Errors |
|---|---|---:|---|
| Inside a table (rows before any table marker, or a table's end first) | 51, 54, 58, 61, 64, 65, 72, 74, 76, 79, 80, 84, 92, 93 | 14 | 74, but its pro forma heading is visible: not an opening error |
| Mid-sentence (lowercase or punctuation first) | 50, 52, 55, 59, 60, 62, 66, 67, 69, 71, 73, 75, 81, 83, 88, 89, 94, 95, 96 | 14 | None |
| Mid-sentence with a capital, or a footnote | 53, 57, 68, 78, 82, 85, 97 | 5 | None |
| At an item, note, section or statement heading | 56, 63, 70, 77, 86, 87, 90, 91 | 8 | 91 misreads a visible sentence |

- **Context above the window:** no question depends on text above the window (batch 1's 18).
- **Tables with no year in their headings:** none in this batch. The only other one in the plan (position
  110) falls in batch 3.

## 7. Duplicates, concentration, distribution and leakage

**Duplicates:**
- **None exact or near-exact.** No two questions share a normalised key, within the batch or with any
  earlier pool.
- **Highest Jaccard within the batch:** 0.706, between 79 and 86, which use the same template for two
  different years.
- **Highest against batch 1:** 0.571, between 50 and batch 1's 22 (Apple's hedge notionals, FY2021 and
  FY2024).

**Same template, different year** (correlated, not duplicates):

| Questions | Template | Linked unit | Evidence rule |
|---|---|---|---|
| 79, 86 (and 58) | Amazon's statement of comprehensive income (FY2020 and FY2025 totals; 58 is FY2023's currency row) | `grp_00672`: three generated passages, the cap | 79 and 86 in; 58 out |
| 50, 57, 78 | Apple's derivatives note (FY2021 notionals; FY2020 fair value; FY2020 OCI gains) | 50 shares `grp_00254` with batch 1's 22; 57 and 78 are in separate units | 57 and 78 in; 50 out |
| 83, 97 | Walmart U.S.'s Item 1 description (FY2022, FY2020) | `grp_07772` | Both in |
| 90 | Walmart International's unit counts (FY2021) | `grp_07805`, with batch 1's 13 and 34: the cap | 90 in; with 34, two usable questions |
| 56, 91 | ExxonMobil's long-term debt note (FY2022, FY2021) | `grp_08915` | Both out |
| 61, 82 | Tesla's property, plant and equipment note (FY2024, FY2023) | Separate units | Both out |

**Linked units and filings:**
- **The cap:** counting `val_pilot_v1`, `val_pilot_v2e` and batch 1, every unit holds at most 3 generated
  passages. Three units reach the cap: `grp_00672`, `grp_07805` and `grp_05037` (Pfizer's patent
  table).
- **Apple's supply is running out:** 7 Apple passages came from three filings, FY2021 (50, 71, 92), FY2020
  (57, 78) and FY2019 (64, 85). Six Apple passages remain, all in batch 3.
- **Filings revisited:** every filing in the batch already gave a passage to batch 1 (44 of 44 filings),
  as the year-balanced plan intends.
- **Across all 69 usable questions:**
  - they come from 41 filings: 18 filings give one question, 18 give two and 5 give three (AMZN FY2019,
    AMZN FY2025, JPM FY2021, WMT FY2025, AAPL FY2020);
  - no linked unit holds more than three (`grp_01705`, JPMorgan's credit-derivatives note).

**Distribution:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch passages | 7 | 7 | 7 | 7 | 7 | 7 | 6 | 48 |
| Usable, this batch | 2 | 4 | 4 | 3 | 1 | 5 | 1 | 20 |
| Usable so far (16 carried, 15, 18, 20) | 11 | 12 | 11 | **5** | 9 | 12 | 9 | 69 |
| Remaining plan passages | 6 | 15 | 76 | 44 | 39 | 29 | 55 | 264 |
| Expected usable from them (projection) | 0.6 | 2.0 | 15.0 | 7.7 | 7.5 | 3.7 | 8.5 | 45.1 |
| Projected final | 11.6 | 14.0 | 26.0 | 12.7 | 16.5 | 15.7 | 17.5 | 114.1 |

- **Fiscal years:**
  - The batch has 7 passages from each of 2019 and 2020, 9 from 2021, 6 from each of 2022 to 2025, and
    1 from 2026.
  - The 20 usable questions: 2019 3, 2020 5, 2021 4, 2022 1, 2023 3, 2024 2, 2025 2.
  - All 69 usable so far: 2019 11, 2020 11, 2021 11, 2022 9, 2023 10, 2024 6, 2025 11, 2026 0.
  - Projected final: 2019 about 23, 2020 20, 2021 16, 2022 12, 2023 14, 2024 10, 2025 18. FY2024 stays the
    thinnest.
- **Kinds:** 26 tables and 22 narrative passages. The 20 usable are 13 tables and 7 narrative.
- **Pfizer gave its first usable questions of this plan:** 67, 81 and 95.
  - 74 was revised and 88 failed the evidence rule.
  - The generator rejected 53 (no company) and 60 (quote miscopied).
- **Concentration at the end:** the plan's remaining supply is mostly JPMorgan's. JPMorgan would hold about
  26 of 114 (23%), and Apple about 12.

**Evidence overlap with training, blocked passages and evaluation pools:**
- **The 21 "In" questions:**
  - no train or blocked passage holds the evidence;
  - no positive is used by any training pair;
  - every positive's overlap group and linked unit lie wholly in validation. This was checked for all 48
    passages.
- **The 20 excluded questions:**
  - by holder: training only 8 (66, 68, 69, 80, 82, 88, 91, 92), blocked only 5 (63, 64, 72, 77, 96),
    both 7 (50, 51, 56, 58, 61, 70, 89);
  - by blocked holder: one in an evaluation pool itself for 3 (51, 56, 61), only linked to a pool for 9;
  - by match: figure matches alone excluded 7 (58, 61, 63, 70, 80, 82, 91); quote, span, sentence or
    fact matches excluded 13.
- **Boilerplate is removed by the evidence rule:** all four boilerplate questions (66, 69, 77, 89) are
  excluded. Their sentences recur elsewhere in the corpus, and training or blocked passages hold those
  copies. For 66, 69 and 77 the copies are in other years' filings; for 89 they are in other passages of
  the same filing.
- **Question overlap:**

| Pool | Highest Jaccard | Question |
|---|---:|---|
| Evaluation questions | 0.28 | 68 |
| Training questions (all 150 ever generated) | 0.611 | 87 (against JPMorgan's assets under management, FY2025) |
| `val_pilot_v1` and the 16 carried | 0.600 | 82 (against Apple's net property, plant and equipment) |
| `val_pilot_v2e` batch 1 | 0.50 | 64 |
| `val_pilot_v2f` batch 1 | 0.571 | 50 |

## 8. Actual yield against the predictor

**By stage** (planned rates from the calibration; observed in batch 2):

| Stage | Planned | Observed |
|---|---:|---:|
| Generator acceptance | 0.82 | 41 / 48 (0.85) |
| Evidence pass, of accepted | 19.5 predicted for these 41 (mean 0.48) | 21 (0.51) |
| Review acceptance, of passing | 0.86 | 20 / 21 (0.95) |
| Usable | 16.0 for these 48 | **20** |
| Usable, both batches | 39.4 for the 97 | **38** |

**The predictor's accuracy** (measured, out of sample):
- **This batch:** AUC 0.712, against 0.635 for the old sentence share. Brier score 0.210, against 0.255
  for the calibration's constant pass rate (0.44) and 0.250 for the batch's own rate.
- **Both batches:** AUC 0.769 (sentence share 0.649). Brier 0.189, against 0.265 and 0.242.
- **Calibration, both batches:** slightly conservative, 41.1 predicted and 45 observed.

| Predicted pass | Accepted questions | Predicted | Observed |
|---|---:|---:|---:|
| Below 0.4 | 21 | 5.1 | 6 |
| 0.4–0.6 | 22 | 11.0 | 13 |
| 0.6–0.8 | 22 | 14.9 | 16 |
| 0.8 and above | 11 | 10.0 | 10 |

**Why batch 2 beat its estimate:** the evidence stage was close to its prediction. The two constant
stages were both higher than planned:
- generator acceptance was 0.85: one long-list quote, against seven in batch 1;
- review acceptance was 0.95: one error among the evidence-passing questions, against six.

**Batch 1's figure-count pattern, re-tested:**

| | ≤ 2 significant figures | 3 or more |
|---|---:|---:|
| Passages | 15 | 33 |
| Mean predicted pass | 0.55 | 0.44 |
| Generator accepted | 12 | 29 |
| Evidence pass (predicted) | 6 (6.6) | 15 (12.9) |
| Review accepted, of passing | 6 of 6 | 14 of 15 |
| **Usable per passage, batch 2** | **0.40** | **0.42** |
| Usable per passage, batch 1 | 0.16 | 0.58 |
| Usable per passage, all 197 generated passages | 0.32 (25 / 77) | 0.37 (44 / 120) |

- **The pattern did not recur.** Pooled over all four batches, low-figure passages have lower generator
  acceptance (0.71 against 0.86) and review acceptance (0.71 against 0.96). They also pass the evidence
  rule more often (0.64 against 0.45), so the yields per passage are close.
- **No rule is needed:** batch 1's decision not to demote these passages stands.

**Re-estimate for the 264 remaining passages** (projections):

| Model | Remaining | Batch 3 (positions 98–147) | Total with the 69 usable |
|---|---:|---:|---:|
| As planned (calibration rates) | 48.1 | 13.0 | 117 |
| This plan's observed rates (generator 0.78, review 0.84, 97 passages) | 45.1 (bootstrap 40–50) | 12.2 | 114 |
| Generator and review acceptance by figure class, pooled over 197 passages | 48.8 (bootstrap 46–52) | 13.4 | 118 |
| Scaled by both batches (38 / 39.4) | 46.4 | 12.5 | 115 |

**Yield per request falls with the ranking:** the plan is ordered by predicted pass.
- **Mean score by batch:** 0.68 in batch 1, 0.47 in batch 2 and 0.37 in batch 3.
- **Later blocks of 50 positions:** 0.30, 0.28, 0.24, 0.14 and 0.11.
- **Batch 3:** about 4 requests per usable question.
- **The last 114 passages (positions 248–361):** about 14 usable questions in all, about 8 requests each.

## 9. Did the predictor and the administrative filter help?

**The predictor:**
- **Evidence rule: yes, measured over both batches.**
  - Batch 2 alone: 51% against 44% (p = 0.28). This is expected, since batch 2's passages are ranked
    lower (mean predicted pass 0.48, against 0.62 in batch 1).
  - Both batches: 59% (45 / 76) against 44% (p = 0.039).
  - It is calibrated on questions it never saw.
- **Usable yield: higher in both batches, not established.**
  - Batch 2: 42% per passage, against 30% and 32% in the earlier pilots (p = 0.14).
  - Both batches: 39% (38 / 97), against 31% (p = 0.15).

**The administrative filter:**
- **No administrative question reached the review:** no trading-plan, insider-trading, cybersecurity,
  Item 6, Section 13(r) or website-services passage was in the batch.
- **What became of the two passages batch 1's scan flagged:**
  - 69, a critical audit matter, gave a boilerplate question that the evidence rule excluded;
  - 87, an Item 1 opening that mentions the website, gave a usable question on JPMorgan's total assets
    and equity.
- **What remains:** one critical audit matter (138), two forward-looking-statement notices (265, 348), two
  new-standard notices (192, 295) and one pro forma passage (122). Positions 122 and 138 fall in batch 3.
  None needs a new rule.

## 10. Systematic weaknesses of Qwen's outputs in this batch

1. **Mislabelling what a figure is:** a pro forma adjustment called a recorded expense (74); debt due
   within a year called redeemed (91). The number is right in both, and the meaning wrong.
2. **No company named:** 3 of 48, against 2 of 49 in batch 1. The question names a product or lawsuit
   instead.
3. **Copying errors in quotes:** "$11.5 million" for "$11.5 billion" (60), caught by the verbatim check.
4. **Rows cut by the window:** Qwen chose a row whose value is not visible (62), and said so. It was
   caught.
5. **Boilerplate:** 66, 69, 77 and 89. All four were excluded by the evidence rule.
6. **Template repetition across years:** Amazon's comprehensive income (79, 86) and Apple's derivatives
   note (50, 57, 78).
7. **Better than in batch 1:**
   - long list quotes 1 (batch 1: 7);
   - units dropped 1 (8);
   - no year-order, sign or prior-period errors;
   - no administrative or trivial question.

## 11. Comparison with earlier pilots

| | `val_pilot_v1` | `val_pilot_v2e` batch 1 | `val_pilot_v2f` batch 1 | `val_pilot_v2f` batch 2 |
|---|---|---|---|---|
| Plan | Balanced years, no score | Balanced years, isolation share ≥ 0.3 | Balanced years, ranked by `evidence-pass-v1`; administrative rule | Same plan, positions 50–97 |
| Passages completed | 50 | 50 | 49 (50 requests) | 48 (50 requests) |
| Generator acceptance | 43 (86%) | 39 (78%) | 35 (71%) | 41 (85%) |
| Mean predicted pass, of accepted | – | – | 0.62 | 0.48 |
| Evidence pass, of accepted | 19 (44%) | 17 (44%) | **24 (69%)** | 21 (51%) |
| Review: accept / revise / reject, of accepted | 36 / 5 / 2 | 36 / 1 / 2 | 26 / 4 / 5 | 39 / 2 / 0 |
| Factual errors, of accepted | 5 (12%) | 1 (3%) | 6 (17%) | 2 (5%) |
| Usable per passage | 16 (32%) | 15 (30%) | 18 (37%) | **20 (42%)** |
| Tokens per passage | 1,344 | 1,322 | 1,314 | 1,353 |

## 12. Recommendation for batch 3

**Batch 3 may proceed on `val_pilot_v2f` unchanged** (positions 98–147, one batch of at most 50 requests),
with every question AI-reviewed as here.
- **The safeguards held:**
  - no leakage, no question overlap, no unit crossing the split;
  - every cap respected.
- **The predictor is calibrated** on 76 questions it never saw.
- **Quality improved:** 2 factual errors in 41, both caught by the review.
- **Why no new plan version:**
  - batch 1's low-figure pattern did not recur;
  - three passages in batch 3 share a known failure pattern (110, 122, 138), and the review catches them.

**What to expect from batch 3** (projections):
- **Usable questions:** about 12–13.
- **Cost:** about 65,000 tokens and 22–25 minutes at the pace of both batches.
- **Composition:**
  - it starts at position 98;
  - Apple's last 6 passages, 8 each from Amazon and ExxonMobil, 7 from each other company;
  - 10 from FY2020 and 8 from FY2019;
  - 11 with two or fewer significant figures.

**Decisions for you, none blocking batch 3:**
- **The total:**
  - The whole plan now projects about 114–118 usable questions (§8), a range of about 109–121.
  - 120 needs the optimistic case, or a supply change: a unit cap of 4 or 5, or a larger validation
    split.
- **How far to run the plan:**
  - Later batches cost more per question: about 4 requests per usable question in batch 3, about 8 at the
    end.
  - Whether to stop at a smaller set (about 100) or run all 264 passages is worth deciding after batch 3
    or 4.
- **Balance in the final set:**
  - Apple and Amazon are nearly exhausted, and JPMorgan would be about 23% of the set.
  - A per-company or per-template cap at build time is worth deciding. The template caps would cover
    Apple's securities note (6 of Apple's 11) and its derivatives note (2).
- **Recalibration:** not needed before batch 3.
  - The plan is fixed and write-once, so a new calibration would only change projections and future
    plans.
  - The figure-class model moves the projection by less than one question (48.8 against 48.1).
- **Recording:** the decisions of both batches can go into versioned files in `data/validation_review/`,
  in the format of `val_pilot_v2e`'s batch, once you approve them.
