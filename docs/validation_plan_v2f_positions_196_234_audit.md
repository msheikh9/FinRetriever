# Validation plan `val_pilot_v2f`: audit of positions 196–234, where generation stopped

Status: 2026-11-04.
- **What this is:** an AI review (Claude) of the 39 generation records for plan positions 196–234 of
  `val_pilot_v2f` in `bge_base_ft_v3`. These are the last records before generation was stopped. It is
  not human annotation, and no decision here is a gold label.
- **Where the decisions are:** they are recorded together with batch 4's (positions 148–195):
  - `data/validation_review/val_pilot_v2f_positions148-234_quality_decisions.jsonl`;
  - the outcomes log beside it.
  The final validation set `validation_pairs_v4` reads both files ([validation_pairs_v4.md](validation_pairs_v4.md)).
- **What was not done:**
  - no generation record, plan, training candidate, pair or earlier decision was changed;
  - no API request was sent and nothing was trained.
- **Earlier documents:**
  - [the plan](validation_plan_v2f.md);
  - the audits of [batch 1](validation_plan_v2f_batch1_audit.md), [batch 2](validation_plan_v2f_batch2_audit.md),
    [batch 3](validation_plan_v2f_batch3_audit.md), [positions 144–147](validation_plan_v2f_positions_144_147_audit.md)
    and [batch 4](validation_plan_v2f_batch4_audit.md).

## 1. Summary

**The records:**
- 39 records for positions 196–234 were appended after the batch 4 audit. Each one is `done` on its first
  attempt, and together they used 53,947 tokens.
- Positions 196–198 came from batch 4's last execution (12:43 UTC, 4,211 tokens).
- Positions 199–234 came from three later executions: 13:37, 14:27 and 15:29–15:32 UTC, 49,736 tokens in
  all.
- The first 248 record lines are byte-identical to those the batch 4 audit read. The order follows the
  plan, and the validator re-run agrees with every record.

**The funnel:** 39 passages → 32 accepted by the generator (82%) → 5 pass the evidence rule → 5 accepted
by the review.
- **5 usable questions:** positions 199, 212, 217, 228 and 230.
- **Against the estimate:** 7.5 were expected at the calibration's rates. The predictor expected 7.3
  evidence passes among the 32 accepted; 5 passed.

**The review:** 31 accepted, 1 revised (225), none rejected.
- **225 is revised:**
  - It gives the fair value of the *available-for-sale* mortgage-backed securities ($78,081 million) as
    JPMorgan's total MBS.
  - The AFS heading is in the previous chunk. The next chunk lists held-to-maturity MBS at an amortized cost
    of $134,356 million.
  - This is the same error as batch 4's 178.
- **Minor issues that do not change the answer:**
  - seven questions omit a qualifier the table gives (199, 212, 216, 224, 226, 230, 234);
  - five answers give no unit (203, 205, 225, 226, 228);
  - one keeps the table's parentheses (209);
  - one asks "as of" for a full-year amount (230).

**Low-scoring passages:** 16 positions scored below 0.2. The generator accepted 15 of them, and none
passed the evidence rule.

**FY2024:** two passages, 205 (JPMorgan) and 229 (Exxon Mobil). The evidence rule excluded both, so
FY2024 stays at 8 questions in the final set.

**The predictor:** AUC 0.767 and Brier score 0.124 on the 32 accepted questions.

**Generation is stopped:**
- `plans/validation.val_pilot_v2f.stopped.json` closes the plan after position 234 of 361.
- It binds the 284 record lines that existed at the stop.
- `generate --plan val_pilot_v2f` now refuses before reading any credential.
- Positions 235–361 were never generated.

## 2. Generator rejections

| # | Rejection | Assessment |
|---|---|---|
| 210 | `evidence_length` | Correct by the rule: the quote, a whole table row, is longer than 300 characters (439). |
| 211 | `number_not_in_passage` | Strict: "December 31" is not in the passage, whose columns read 2020. The answer is right. |
| 215 | `evidence_length` | Correct by the rule: a five-item list (416 characters). |
| 218 | `evidence_length` | Correct by the rule: the quote "2026 \| $2,598" is shorter than 15 characters (13). |
| 219 | `evidence_length` | Correct by the rule: a whole table row (656 characters). |
| 221 | `missing_company` | Correct: the question does not name Pfizer. |
| 232 | `copied_from_passage` | Correct by the rule: the answer copies the passage's sentence. |

## 3. Per-question decisions

Questions and answers are verbatim. "In" means the evidence rule passes. The usable questions are those that are
both "In" and ACCEPT.

| # | Source | Question (verbatim) | Answer (verbatim) | Decision | Evidence rule | Note |
|---|---|---|---|---|---|---|
| 196 | PFE FY2022, narrative/mdna | What specific regulatory action did Pfizer anticipate could impact its business, operations, and financial condition following its enactment in August 2022? | The Inflation Reduction Act (IRA) | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; the IRA, signed in August 2022, whose impact Pfizer says it continues to evaluate. |
| 197 | TSLA FY2025, table/statements | What was the total value of Tesla's inventory as of December 31, 2025? | $12,392 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; the inventory note's total. |
| 198 | WMT FY2021, table/mdna | What was Walmart's return on investment (ROI) for the fiscal year ended January 31, 2021? | 14.0% | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; ROI for the fiscal year ended January 31, 2021. |
| 199 | XOM FY2022, table/business | How many total productive development wells did Exxon Mobil Corp drill in fiscal 2022? | 573 | ACCEPT | **In** | Correct row; the table counts net wells and the question omits 'net' (as validation_000014). |
| 200 | JPM FY2019, narrative/statements | Which internal executives review JPMorgan Chase's allowance for credit losses at least quarterly? | The CRO, the CFO and the Controller | ACCEPT | Out: training evidence | Correct; a governance sentence repeated in JPMorgan's other years' filings (exact copies in train passages). |
| 201 | PFE FY2025, table/statements | What were the total Biopharma segment revenues for Pfizer in fiscal 2025? | $61,199 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; the CODM segment table's 2025 column. |
| 202 | TSLA FY2019, table/statements | What was the balance of net operating loss carry-forwards included in Tesla's deferred tax assets as of December 31, 2019? | $1,846 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; the deferred tax asset for net operating loss carry-forwards. |
| 203 | WMT FY2022, narrative/mdna | What was the net sales for Walmart's Sam's Club segment excluding fuel for fiscal 2022? | $64,860 | ACCEPT | Out: evaluation evidence | Correct; Sam's Club net sales excluding fuel (named in the footnote). No unit (millions). |
| 204 | XOM FY2023, table/other | What was Exxon Mobil Corp's net income including noncontrolling interests in millions of dollars for fiscal 2023? | 37,354 million dollars | ACCEPT | Out: possible evaluation evidence, training evidence | Correct; consolidated, before noncontrolling interests. |
| 205 | JPM FY2024, narrative/risk | What was JPMorgan Chase's total consumer credit exposure in fiscal 2024? | 1,671,825 | ACCEPT | Out: possible training evidence | Correct; a balance asked 'in fiscal 2024'. No unit (millions). |
| 206 | PFE FY2023, table/statements | What was the total equity for Pfizer Inc. and Subsidiary Companies as of the beginning of fiscal 2021? | $63,473 million | ACCEPT | Out: training evidence | Correct; the opening balance of the three-year equity statement, named by its date (as batch 4's 155). |
| 207 | TSLA FY2023, table/statements | What was Tesla's total operating lease liability as of December 31, 2023? | $4,343 million | ACCEPT | Out: evaluation evidence, possible training evidence | Correct. |
| 208 | WMT FY2020, narrative/mdna | What was the operating income for Walmart's Sam's Club segment in fiscal 2020? | $1,486 million | ACCEPT | Out: possible training evidence | Correct; the Sam's Club segment table (named in its footnote). |
| 209 | XOM FY2025, narrative/other | What was the net cash used in financing activities by ExxonMobil in fiscal 2025? | (39,081) million | ACCEPT | Out: evaluation evidence | Correct; keeps the table's parentheses for a net outflow (as batch 4's 188). |
| 210 | JPM FY2025, narrative/other | What is the required effective date for the new guidance on purchased loans issued in November 2025 for JPMorgan Chase & Co? | January 1, 2027 | Generator: `evidence_length` | – | Correct by the rule: the quote, a whole table row, is longer than 300 characters. The answer (January 1, 2027) is right. |
| 211 | PFE FY2021, table/mdna | What was Pfizer Inc.'s GAAP reported net income attributable to common shareholders for the year ended December 31, 2020, in millions? | $9,159 million | Generator: `number_not_in_passage` | – | Strict: 'December 31' is not in the passage, whose columns read 2020. The answer, $9,159 million, is right. |
| 212 | TSLA FY2025, narrative/mdna | What next-generation industrial storage product did Tesla introduce in 2025? | Megablock | ACCEPT | **In** | The passage introduces 'Megapack 3 and Megablock, our next-generation industrial storage product'; the answer names Megablock only. A product launch. |
| 213 | WMT FY2021, table/statements | What was Walmart's total long-term debt excluding amounts due within one year as of January 31, 2021? | $41,194 million | ACCEPT | Out: evaluation evidence | Correct; net of the $3,115 million due within one year. |
| 214 | XOM FY2019, narrative/other | How much cash flow did Exxon Mobil Corp generate in fiscal 2019 from the divestment of its non-operated upstream assets in Norway? | $3.1 billion | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct. |
| 215 | JPM FY2023, narrative/risk | What specific factors could cause the actual results of JPMorgan Chase's acquisition of First Republic Bank to differ from anticipated positive outcomes during fiscal 2023? | The settlement of the final purchase price, the total cost of integration, the time required to complete the integration, the overall performance of the assets and liabilities acquired, or an improved price for JPMorgan Chase's common stock. | Generator: `evidence_length` | – | Correct by the rule: a five-item list longer than 300 characters. |
| 216 | PFE FY2020, table/mdna | What is the total amount of long term, including current, that Pfizer reported as of December 31, 2020? | $39,135 millions | ACCEPT | Out: evaluation evidence | Correct row; the question drops 'debt' ('long term, including current'), and the figure is the contractual-obligations total of principal due, which it does not say. |
| 217 | TSLA FY2019, narrative/risk | Which new vehicle model did Tesla begin high-volume manufacturing of at multiple facilities, starting with Gigafactory Shanghai in the fourth quarter of 2019? | Model 3 | ACCEPT | **In** | Correct; a production fact in the risk factors. |
| 218 | WMT FY2025, table/statements | What were the annual maturities of Walmart's long-term debt for fiscal 2026, in millions? | $2,598 million | Generator: `evidence_length` | – | Correct by the rule: the quote '2026 | $2,598' is shorter than 15 characters. The answer is right. |
| 219 | XOM FY2020, table/other | What was the net carrying value of Exxon Mobil's derivative assets as of December 31, 2020? | 153 million dollars | Generator: `evidence_length` | – | Correct by the rule: a whole table row longer than 300 characters. The answer (153) is right. |
| 220 | JPM FY2020, narrative/other | What full-year 2021 net interest income outlook did JPMorgan Chase & Co provide on a managed basis? | approximately $55 billion | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; management's full-year 2021 outlook, named. |
| 221 | PFE FY2019, narrative/business | In late 2019, how many new molecules were covered by the expansion of China's national volume-based procurement program, and which specific Biopharma products were included in this list? | 33 new molecules, including Biopharma's Zithromax tablets and Diflucan tablets | Generator: `missing_company` | – | Correct: the question does not name Pfizer. |
| 222 | TSLA FY2023, narrative/statements | What was Tesla's total liabilities and equity as of December 31, 2023, compared to December 31, 2022? | Total liabilities and equity was $106,618 million in fiscal 2023 and $82,338 million in fiscal 2022. | ACCEPT | Out: evaluation evidence, possible training evidence | Correct; both years (equal to total assets). |
| 223 | WMT FY2022, table/statements | What was Walmart Inc.'s total equity as of January 31, 2022, expressed in millions of dollars? | $91,891 million | ACCEPT | Out: training evidence | Correct; including noncontrolling interest. |
| 224 | XOM FY2021, table/business | What was Exxon Mobil's average production price for natural gas in the Asia region during 2019, expressed per thousand cubic feet? | 2.07 dollars per thousand cubic feet | ACCEPT | Out: possible training evidence | Correct row, consolidated subsidiaries' 2019 price in the FY2021 filing's three-year table; the question omits 'consolidated subsidiaries' (equity companies' prices follow). |
| 225 | JPM FY2022, narrative/statements | What was the fair value of JPMorgan Chase's total mortgage-backed securities as of December 31, 2022? | 78,081 | **REVISE** | Out: possible training evidence | $78,081 million is the fair value of the available-for-sale mortgage-backed securities only: the AFS heading is in the previous chunk, and the next chunk gives held-to-maturity MBS (amortized cost $134,356 million). JPMorgan's total MBS were larger. No unit (millions). |
| 226 | PFE FY2023, narrative/mdna | What was the impact of restructuring charges and implementation costs on Pfizer's net income attributable to common shareholders in the year ended December 31, 2021? | 1,309 | ACCEPT | Out: possible evaluation evidence | The non-GAAP reconciliation's pre-tax adjustment for restructuring charges, implementation costs and additional depreciation (the question drops the last); its direction and tax effect are not stated. No unit (millions). |
| 227 | TSLA FY2025, narrative/statements | How many shares were issued under Tesla's Employee Stock Purchase Plan during the year ended December 31, 2025? | 1.9 million shares | ACCEPT | Out: possible training evidence | Correct. |
| 228 | WMT FY2020, table/mdna | What was the total amount of Walmart's contractual obligations as of fiscal year 2020? | $123,992 | ACCEPT | **In** | Correct; contractual obligations as of January 31, 2020. No unit (millions). |
| 229 | XOM FY2024, table/business | How many total net productive exploratory wells did Exxon Mobil drill through its consolidated subsidiaries in fiscal 2024? | 5 | ACCEPT | Out: training evidence | Correct; consolidated subsidiaries' net productive exploratory wells. |
| 230 | JPM FY2019, table/statements | What were the net liquidation losses for JPMorgan Chase's subprime residential mortgage securitized assets as of December 31, 2019? | $532 million | ACCEPT | **In** | Correct row; a full-year amount asked 'as of December 31, 2019', and the table covers Firm-sponsored private-label securitizations with continuing involvement, which the question omits. |
| 231 | PFE FY2025, narrative/statements | What was Pfizer's total long-term debt reported on the balance sheet as of December 31, 2025? | 61,641 million | ACCEPT | Out: possible evaluation evidence | Correct; the balance-sheet line (the current portion sits in short-term borrowings). |
| 232 | TSLA FY2019, narrative/business | How does Tesla structure the financial relationship between the customer and the third-party lender in its solar loan offering? | Tesla is not a party to the loan agreement, and the third-party lender has no recourse against Tesla with respect to the loan. | Generator: `copied_from_passage` | – | Correct by the rule: the answer copies the passage's sentence. |
| 233 | WMT FY2022, table/statements | What was Walmart's total current tax provision for the fiscal year ended January 31, 2022, in millions? | $5,515 million | ACCEPT | Out: possible training evidence | Correct. |
| 234 | XOM FY2022, table/other | What was the amount of depreciation and depletion reported by Exxon Mobil in its consolidated statement of cash flows for fiscal 2022? | $24,040 million | ACCEPT | Out: possible evaluation evidence | Correct row; the line includes impairments, which the question omits. |

## 4. Aspects of the 32 reviewed questions

| Aspect | pass | minor | fail | n/a |
|---|---:|---:|---:|---:|
| Factual accuracy | 24 | 7 | 1 | – |
| Financial relevance | 30 | 2 | – | – |
| Company | 32 | – | – | – |
| Period | 31 | 1 | – | – |
| Figures | 26 | 1 | 1 | 4 |
| Units | 23 | 5 | – | 4 |

## 5. Leakage, duplicates and visibility

- **Evidence rule:** it excluded 27 questions, each because a train or blocked passage holds its
  evidence, exactly or possibly.
  - 17 are held by a blocked passage, one that holds evaluation evidence or lies in its group. Six of them
    match exactly (203, 207, 209, 213, 216, 222) and eleven possibly.
  - The other ten are held only by train passages: four exactly (200, 206, 223, 229) and six possibly.
- **Overlap with evaluation questions:** the highest is 0.333 (198), far below the 0.8 threshold.
- **Overlap with training questions:** the highest is 0.5 (197 and 207, inventory and lease templates).
- **Duplicates:**
  - no question repeats another;
  - ten pairs in the window share a balance-sheet template with a similarity of 0.42–0.53. All of them
    are below 0.8.
- **Positives:** no positive is used by a training pair, and every split unit lies wholly in validation.
  `build-validation` re-checked both.
- **Visibility within BGE's 512-token limit:** every quote ends by token 504. The latest ends at 504 (222)
  and 501 (217).

## 6. After the stop

- **`val_pilot_v2f`:** positions 1–234 give 62 usable questions.
- **The final validation set** (`validation_pairs_v4`) holds 93 questions:
  - the 16 original questions;
  - 15 from `val_pilot_v2e` batch 1;
  - these 62.
