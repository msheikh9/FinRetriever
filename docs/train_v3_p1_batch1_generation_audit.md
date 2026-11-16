# P1 batch 1 (`train_v3_p1_batch1`): generation audit and quality gate

Status: 2026-11-05.
- **What this is:**
  - an audit of the 99 completed generation records of plan `train_v3_p1_batch1` in `bge_base_ft_v3`, and of its one
    failed record;
  - an AI review (Claude) of the 112 pairs they build;
  - the batch quality gate.

  It is not human annotation, and no decision here is a gold label.
- **Where the decisions are:** `data/train_review/train_v3_p1_batch1_quality_decisions.jsonl`.
  - 112 rows (101 accept, 6 revise, 5 reject), each `reviewer: ai_review` and `gold: false`.
  - Each row is bound to its `pair_id`, `passage_id` and the SHA-256 of its exact question.
  - SHA-256 `52cc6ff1…6d06`.
- **What was not done:**
  - no question was generated and no API request was sent;
  - no generation record, plan, candidate dataset, validation set or earlier decision was changed;
  - `build-pairs`, `screen-questions` and `build-candidates` were not run in `bge_base_ft_v3`. They ran only in a
    scratch copy of the preparation, which was then discarded;
  - nothing was trained, committed or pushed.
- **Protected files:** 343 files under `data/`, `configs/`, `src/` and `scripts/` were checksummed before and after
  the audit. They are unchanged; the decisions file is the only file added.
- **Earlier documents:** [the readiness audit](final_training_readiness_audit.md), with the gate proposal in §B8, and
  [the full-training plan](training_full_v1.md), §9.

## 1. Verdict

**The batch passes the approved quality gate on all four criteria.**

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | At least 0.8 included pairs per passage | 101 included for 100 planned passages: **1.01** (1.02 per completed passage) | **Pass** |
| 2 | At least 75% AI-review acceptance | 101 of 112 built pairs: **90.2%** | **Pass** |
| 3 | Zero accepted leakage violations | **0**: no included pair fails the evidence, cross-split, lexical, semantic or duplicate checks (§7) | **Pass** |
| 4 | Complete provenance for every included pair | **101 of 101** carry the full provenance block, and the fail-closed provenance check passes (§10) | **Pass** |

The readiness audit's further checks (§B8, items 3–6) also hold:
- **Companies:** every company contributes at least 8 included pairs. The fewest are ExxonMobil's 10.
- **Semantic screen:** no question is flagged. The one question in the proposed 0.80–0.90 review band is reviewed in
  §7.
- **Existing data:** `train_candidates_v2` rebuilds byte-identical, and `validation_pairs_v4` still loads.
- **One generator:** all records use one model, prompt, validator and parameter set.

**GO for continuing P1 generation, under four conditions:**
1. **Budget:** you approve the Groq budget for the next batches (readiness audit, D3). At batch 1's 1,483 tokens per
   passage, batches 2–6 (500 passages) need about 740,000 tokens.
2. **Sequencing:** the real `build-pairs` and `build-candidates` are not run in `bge_base_ft_v3` until P1 generation
   is finished, or you approve a change for batch-wise pair sets (§12). Under the current code, building batch 1 now
   would block the pairs of batches 2–6 and break the new candidate set's provenance once batch 2 starts.
3. **Position 87:** it stays failed (§11), or it is retried before any real `build-pairs`, through a recorded
   continuation.
4. **One generator:** the same model, prompt `sec-qgen-v2`, validator `sec-qcheck-v3` and parameters `42d5fe54…`.
   The validator weaknesses in §4 are not fixed in the middle of P1.

**Unchanged:** the three-seed A0 versus A1 experiment stays **NO-GO**. Of the readiness audit's four blockers,
batch 1's gate (B4) is now resolved. B1 (a selection-aware decision rule), B2 (the planned data size) and B3 (the GPU
rehearsal) still stand.

## 2. Records and provenance

| Check | Result |
|---|---|
| Plan | `train.train_v3_p1_batch1.json`, SHA-256 `3b9210c3…cec0`, unchanged; 100 train passages |
| Records file | `generation/train.records.jsonl`, 100 lines, SHA-256 `91116337…56d1`. One record per plan position, in plan order; no other plan's records |
| The 39 earlier records | Lines 1–39 hash to `30f83dd7…`, as the readiness audit recorded them. The resume only appended. |
| Outcome | **99 `done`, 1 `failed` (position 87)**; every record took 1 attempt |
| Identity | All 100 records: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…`, record format 3 |
| Run | `train.run.json` binds prompt SHA-256 `f42bdc54…`, the same parameters and preparation `21c2293a…`; there is no continuation file |
| Replies | 99 distinct response hashes; the failed record has none, because no usable reply came back |
| Validator re-run | `revalidate --dry-run`: 99 up to date, 0 changed, 1 failed record |
| This run (positions 40–100) | 2026-11-05, 16:06:46–16:52:49 UTC. 61 HTTP requests against a budget of 61, 0 HTTP 429 responses, 2,703 s of pacing waits. No stop by budget or rate limit. |
| Tokens | Positions 1–39: 56,353 (46,444 prompt, 9,909 completion). Positions 40–100: **91,931** (74,903 prompt, 17,028 completion), equal to the run log. Batch total **148,284**. |
| Completion tokens per reply | Done records: 163–504, median 261. The failed reply: 768, the limit. |

The HTTP attempts of the earlier invocations (positions 1–39) were not recorded. Their tokens are known from the
records. As the readiness audit noted (§B3), `--max-requests` counts only within one invocation.

**Against the plan's estimates** ([training_full_v1.md](training_full_v1.md), §9):

| | Estimate | Actual |
|---|---|---|
| Tokens | about 143,000 | 148,284 |
| Questions passing the validator | about 155 | 151 |
| Pairs built | about 140 | 112 |
| AI review accepts | about 115 | 101 |
| Candidates | about 100 (range about 80–115) | 101 |

The evidence rule excluded more than in `pilot_v3` (39 of 151 against 8 of 78), as the plan expected for these less
isolated passages. The review accepted more (90% against 81%), so the final count matches the estimate.

## 3. The funnel

198 proposed (99 × 2) → **151** pass the generator's validator (76.3%) → **112** pairs built → **101** accepted by
the review → **101** included in the trial candidate set.

| | Plan | Done | Proposed | Pass validator | Pairs built | Accepted (included) |
|---|---:|---:|---:|---:|---:|---:|
| Positions 1–39 | 39 | 39 | 78 | 63 (81%) | 49 | 42 |
| Positions 40–100 | 61 | 60 | 120 | 88 (73%) | 63 | 59 |
| **All** | **100** | **99** | **198** | **151 (76%)** | **112** | **101** |

- **Exclusions at each step:**
  - `build-pairs` excluded 39 questions, every one for evidence leakage: `evaluation_evidence` 33 and
    `cross_split_evidence` 6 (§5);
  - nothing was excluded for a validator re-check, a duplicate or evaluation overlap;
  - the trial candidate build excluded nothing beyond the 11 that the review did not accept.
- **Per passage:** 30 passages contribute no included pair, 39 contribute one and 31 contribute two.

## 4. Generator acceptance and rejection (validator `sec-qcheck-v3`)

| Reason | Count | Verdict |
|---|---:|---|
| `evidence_length` | 18 | All correct by the rule: the quotes have 301–539 characters, against the 300 limit. Mostly whole table rows or lists. |
| `missing_company` | 12 | All correct: the question does not name the company. |
| `copied_from_passage` | 11 | 4 copy a run of prose (9–16 words). 6 are borderline 8-word runs, mostly entity or term names: correct by the strict rule. 1 is false (position 60): the run is the flattened table's row label, which the label exemption missed. |
| `evidence_not_in_visible_text` | 3 | All correct: two quotes join pieces that are not adjacent or are reversed, and one row lies past the window. |
| `number_not_in_passage` | 2 | Position 79 correct (a figure computed from two others). Position 57 is false: the answer is right, but the date the question names is not in the visible text. |
| `answer_not_supported` | 1 | False (position 49): the answer is right, but its years are not in the quote. |

**Totals:**
- 38 clearly correct;
- 6 correct by the strict rule;
- **3 false rejections**, positions 49, 57 and 60: 2% of proposals.

Fixing the validator mid-P1 would change the generator's identity, so it is left as is (condition 4).

| Pos | Source | Reason | Question (verbatim) | Verdict |
|---:|---|---|---|---|
| 4 | PFE FY2019, narrative, Item 1 | `missing_company` | Which regulatory body is identified as the point of entry for businesses seeking to sell drugs in Japan, and what specific approvals must it grant before marketing? | Correct: the question does not name the company. |
| 4 | PFE FY2019, narrative, Item 1 | `missing_company` | What specific responsibility does the EMA's Pharmacovigilance Risk Assessment Committee hold regarding product safety for EU authorities? | Correct: the question does not name the company. |
| 12 | TSLA FY2021, table, Item 8 | `copied_from_passage` | What was the total amount of future minimum master lease payments to be received from investors by Tesla as of December 31, 2021? | Correct by the rule: copies a 10-word run ('future minimum master lease payments to be received from investors'). |
| 12 | TSLA FY2021, table, Item 8 | `copied_from_passage` | How many shares were reserved and available for issuance under Tesla's 2019 Equity Incentive Plan as of December 31, 2021? | Correct by the rule, borderline: an 8-word run, mostly the plan's wording. |
| 14 | XOM FY2023, table, Business Results | `missing_company` | According to the 2023 Upstream Earnings Factor Analysis, how much did lower realizations decrease earnings, and what were the primary drivers? | Correct: the question does not name the company. |
| 17 | JPM FY2024, table, Country Risk Management | `missing_company` | How does the Firm characterize its remaining exposure to Russia regarding materiality? | Correct: the question does not name the company. |
| 18 | PFE FY2024, narrative, Item 7 | `copied_from_passage` | What were Pfizer's effective tax rates for non-GAAP adjusted income in fiscal 2024, 2023, and 2022? | Correct by the rule, borderline: an 8-word run naming the measure. |
| 21 | XOM FY2024, table, Note 21. Mergers And Acquisitions | `missing_company` | According to the provisional fair value summary for the Pioneer acquisition in fiscal 2024, what amount was recorded for Property, plant & equipment? | Correct: the question does not name the company. |
| 26 | TSLA FY2024, table, Item 8 | `evidence_length` | What was the fair value of Tesla's certificates of deposit and time deposits classified as short-term investments as of December 31, 2024, in millions? | Correct by the rule: the quote has 315 characters (limit 300). |
| 30 | AMZN FY2024, narrative, Item 9B | `missing_company` | What was the end date of the trading plan adopted by CFO Brian Olsavsky in November 2024? | Correct: the question does not name the company. |
| 31 | JPM FY2021, narrative, Note 4 – Credit Risk Concentrations | `evidence_length` | How much off-balance sheet credit exposure did JPMorgan Chase report for the Automotive sector in fiscal 2020? | Correct by the rule: the quote has 314 characters (limit 300). |
| 32 | PFE FY2025, narrative, Item 7 | `evidence_length` | What was the impact of acquisition-related items on Pfizer's net income attributable to common shareholders for the fiscal year ended December 31, 2025? | Correct by the rule: the quote has 354 characters (limit 300). |
| 34 | WMT FY2024, narrative, Item 1A | `evidence_not_in_visible_text` | What specific challenges regarding third-party payers could impact Walmart's clinical operations and reimbursement rates in fiscal 2024? | Correct: the quote joins two pieces in the reverse of their order in the passage. |
| 36 | AAPL FY2019, narrative, Item 8 | `evidence_length` | What was the potential reduction to derivative assets and liabilities due to rights of set-off associated with Apple's derivative contracts as of September 29, 2018? | Correct by the rule: the quote has 307 characters (limit 300). |
| 39 | PFE FY2023, table, Item 7 | `evidence_length` | How did Moody's and S&P change their outlooks on Pfizer's long-term debt in March and October 2023? | Correct by the rule: the quote has 394 characters (limit 300). |
| 40 | TSLA FY2025, narrative, Item 1A | `evidence_length` | What specific covenant restrictions does Tesla state may limit its ability to operate its business under certain debt facilities in fiscal 2025? | Correct by the rule: the quote has 409 characters (limit 300). |
| 46 | PFE FY2020, narrative, Item 1 | `evidence_length` | What are the four specific diversity and inclusion initiatives Pfizer outlined for fiscal 2020? | Correct by the rule: the quote has 535 characters (limit 300). |
| 46 | PFE FY2020, narrative, Item 1 | `evidence_length` | How frequently does Pfizer conduct its Performance and Leadership Insights process, and what three dimensions are evaluated during these reviews? | Correct by the rule: the quote has 506 characters (limit 300). |
| 49 | XOM FY2019, table, Consolidated Balance Sheet | `answer_not_supported` | What was ExxonMobil's total assets as of December 31, 2019, and December 31, 2018, in millions of dollars? | False: the answer is right; its years are not in the quote. |
| 52 | JPM FY2022, narrative, Consolidated Results Of Operations | `evidence_length` | What drove the decrease in asset management, administration, and commissions revenue for JPMorgan Chase in fiscal 2022? | Correct by the rule: the quote has 459 characters (limit 300). |
| 52 | JPM FY2022, narrative, Consolidated Results Of Operations | `evidence_length` | Why did mortgage fees and related income decrease for JPMorgan Chase in fiscal 2022? | Correct by the rule: the quote has 340 characters (limit 300). |
| 55 | WMT FY2021, table, Item 7 | `copied_from_passage` | How much of Walmart's cash and cash equivalents as of January 31, 2021, may not be freely transferable to the U.S. due to local laws or other restrictions? | Correct: copies a 16-word run of the sentence. |
| 56 | XOM FY2023, narrative, 13. Financial Instruments And Derivative | `evidence_length` | What was the net carrying value of Exxon Mobil's derivative liabilities as of December 31, 2022? | Correct by the rule: the quote has 457 characters (limit 300). |
| 56 | XOM FY2023, narrative, 13. Financial Instruments And Derivative | `evidence_length` | How much was the difference in carrying value and fair value for Exxon Mobil's long-term debt at December 31, 2022? | Correct by the rule: the quote has 402 characters (limit 300). |
| 57 | AAPL FY2020, table, Item 6 | `number_not_in_passage` | What was Apple's total net sales for the fiscal year ended September 26, 2020, and how does this compare to the fiscal year ended September 28, 2019? | False: the answer is right; the question's date September 28, 2019 is not in the visible text, whose columns read 2019. |
| 59 | JPM FY2025, narrative, Item 1A | `evidence_length` | What specific types of events or factors beyond JPMorganChase's control could seriously disrupt its business and operations in fiscal 2025? | Correct by the rule: the quote has 301 characters (limit 300). |
| 59 | JPM FY2025, narrative, Item 1A | `evidence_length` | What are the potential consequences for JPMorganChase if it fails to implement effective countermeasures against unauthorized transactions or if government scrutiny of personal information handling leads to stricter laws? | Correct by the rule: the quote has 539 characters (limit 300). |
| 60 | PFE FY2022, narrative, Item 7 | `copied_from_passage` | How did actuarial valuation and other pension and postretirement plan gains or losses impact Pfizer's net income attributable to common shareholders in fiscal 2020? | False: the 9-word run is the flattened table's row label, which the label exemption missed. |
| 64 | AAPL FY2022, narrative, Item 8 | `evidence_not_in_visible_text` | What was Apple's total debt position, combining current and non-current term debt, at the end of fiscal 2022? | Correct: the quote joins two rows that are not adjacent, for a summed figure ($110,087 million). |
| 66 | JPM FY2024, table, Credit And Investment Risk Management | `missing_company` | What were the net charge-offs or recoveries for the Individuals and Individual Entities industry segment for the year ended December 31, 2024? | Correct: the question does not name the company. |
| 70 | XOM FY2025, narrative, Business Environment | `missing_company` | What does the filing state about the expected trend in light-duty vehicle demand for liquid fuels between the current decade and 2050? | Correct: the question does not name the company. |
| 72 | AMZN FY2024, table, Item 8 | `copied_from_passage` | How many shares of Rivian's Class A common stock did Amazon hold as of December 31, 2024, and what approximate ownership interest did this represent? | Correct by the rule, borderline: an 8-word run naming the security. |
| 74 | PFE FY2021, narrative, Item 1A | `copied_from_passage` | How has the consolidation of MCOs and other private third-party payers affected their negotiating power in Pfizer's fiscal 2021 disclosure? | Correct by the rule, borderline: an 8-word run naming the payers. |
| 79 | AMZN FY2020, narrative, Item 8 | `number_not_in_passage` | What was the total amount of unrecorded liabilities and other commitments for Amazon in fiscal 2020, and how was that total broken down between unconditional purchase obligations and other commitments? | Correct: $177,912 million is not in the visible text (a difference of two figures). |
| 80 | JPM FY2021, narrative, Item 1A | `evidence_length` | What specific negative consequences does JPMorgan Chase identify in its fiscal 2021 risk factors regarding a significant failure or disruption of its operations and systems? | Correct by the rule: the quote has 524 characters (limit 300). |
| 81 | PFE FY2025, table, Item 1 | `evidence_not_in_visible_text` | What is the description provided for the Pfizer U.S. division within the commercial structure table effective January 1, 2026? | Correct: the quoted table row lies past the visible window. |
| 82 | TSLA FY2024, table, Item 8 | `evidence_length` | What was Tesla's adjusted net income for the three months ended September 30, 2024? | Correct by the rule: the quote has 474 characters (limit 300). |
| 82 | TSLA FY2024, table, Item 8 | `missing_company` | How much were the adjustments to deferred tax assets for the quarter ended March 31, 2024? | Correct: the question does not name the company. |
| 84 | XOM FY2021, narrative, 1. Summary Of Accounting Policies | `copied_from_passage` | What depreciation method and useful life does Exxon Mobil Corp use for refinery, chemical process, and lubes basestock manufacturing equipment in fiscal 2021? | Correct by the rule, borderline: an 8-word run naming the equipment. |
| 90 | WMT FY2025, narrative, Item 7 | `missing_company` | According to the financial data, what was the percentage of net sales for operating expenses in the fiscal year ended January 31, 2023? | Correct: the question does not name the company. |
| 91 | XOM FY2020, narrative, Business Environment And Risk Assessment | `copied_from_passage` | How is the non-OECD share of global liquid fuels demand expected to change by 2040 according to Exxon Mobil Corp's business environment assessment? | Correct: copies a 9-word run of the sentence. |
| 92 | AAPL FY2019, table, Item 8 | `missing_company` | According to the term debt summary for fiscal 2019, what was the effective interest rate range for the 2013–2018 fixed-rate notes maturing in 2047? | Correct: the question does not name the company. |
| 93 | AMZN FY2019, narrative, Item 8 | `missing_company` | What was the primary driver for the reclassification of Prime membership fees in the year ended December 31, 2018? | Correct: the question does not name the company. |
| 95 | PFE FY2020, table, Item 7 | `copied_from_passage` | How did the completion of the Upjohn separation in November 2020 affect Pfizer's long-term debt ratings from Moody's and S&P? | Correct by the rule, borderline: an 8-word run naming the event. |
| 98 | XOM FY2019, table, Liquidity And Capital Resources | `evidence_length` | What was the total amount of long-term debt excluding finance lease obligations reported by Exxon Mobil Corporation as of December 31, 2019? | Correct by the rule: the quote has 319 characters (limit 300). |
| 99 | PFE FY2019, narrative, Item 7A | `copied_from_passage` | What effect would a one hundred basis point increase in interest rates as of December 31, 2019, have on Pfizer's net income according to its sensitivity analysis? | Correct: copies a 14-word run of the sentence. |
| 100 | WMT FY2021, narrative, Item 1A | `evidence_length` | What potential consequences for noncompliance with privacy laws like the TCPA does Walmart identify in its fiscal 2021 filing, including impacts on its business operations? | Correct by the rule: the quote has 305 characters (limit 300). |

## 5. Pair building: evidence-leakage exclusions (39)

`build-pairs` excludes a question whose evidence, quote or labelled figure is held exactly by:
- a blocked passage: `evaluation_evidence`;
- the visible text of a passage in another split: `cross_split_evidence`.

These rules fail closed by design, and every exclusion below follows them.

| Where the evidence is | Exclusions |
|---|---:|
| An evaluation-pool or audit passage | 6 |
| An overlap-group copy of an evaluation passage | 11 |
| Only passages blocked for reusing evaluation text | 16 |
| The visible text of a validation passage | 6 |

**The 16 exclusions held only by evidence-reuse blocks are conservative, as in `pilot_v3`** (its quality report, §5).
Narrowing the rule would be a methodology change for you to decide, and it is not recommended mid-P1.

**Cross-split:** the 6 validation-passage holders keep the training set and `validation_pairs_v4` apart.

| Pos | Source | Question (verbatim) | Answer (verbatim) | Rule | Where the evidence is |
|---:|---|---|---|---|---|
| 1 | AAPL FY2021, table, Item 7 | What was the effective tax rate for Apple in fiscal 2020? | 14.4% | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked; also 2 validation) |
| 9 | AMZN FY2022, narrative, Item 7A | What was the estimated fair value of Amazon's asset-backed securities as of December 31, 2022? | $2,572 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 10 | JPM FY2025, narrative, Business Segment & Corporate Results | What was the operating lease income reported by JPMorgan Chase & Co for the year ended December 31, 2025? | $3.8 billion | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 13 | WMT FY2020, table, Item 7 | What was the Return on Investment (ROI) for Walmart Inc. for the fiscal year ended January 31, 2020? | 13.4% | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked; also 1 validation) |
| 13 | WMT FY2020, table, Item 7 | What were the Total assets for Walmart Inc. as of January 31, 2019? | $219,295 | `cross_split_evidence` | the visible text of 2 validation passages |
| 14 | XOM FY2023, table, Business Results | What were ExxonMobil's total Upstream earnings excluding Identified Items (Non-GAAP) for the year ended December 31, 2023? | $23,609 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 18 | PFE FY2024, narrative, Item 7 | What was Pfizer's non-GAAP adjusted net income attributable to common shareholders for the year ended December 31, 2022? | $37,717 | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 20 | WMT FY2023, table, Item 7 | What was the amount of net cash used in investing activities for Walmart Inc. for the fiscal year ended January 31, 2023? | $(17,722) million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 20 | WMT FY2023, table, Item 7 | How much cash and cash equivalents did Walmart Inc. hold that were not freely transferable to the U.S. as of January 31, 2023? | $2.9 billion | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 21 | XOM FY2024, table, Note 21. Mergers And Acquisitions | What was the fair value of the ExxonMobil common stock issued to acquire Pioneer Natural Resources in May 2024? | $63 billion | `evaluation_evidence` | an evaluation-pool or audit passage (5 blocked; also 1 validation) |
| 29 | AAPL FY2022, table, Item 7 | What was Apple's provision for income taxes in fiscal 2022, and how did it compare to fiscal 2021? | Apple's provision for income taxes was $19,300 million in fiscal 2022, compared to $14,527 million in fiscal 2021. | `evaluation_evidence` | an evaluation-pool or audit passage (6 blocked; also 1 validation) |
| 29 | AAPL FY2022, table, Item 7 | What were the primary drivers for the increase in Apple's research and development expenses in fiscal 2022? | The increase was driven primarily by increases in headcount-related expenses and engineering program costs. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 33 | TSLA FY2020, table, Item 7 | What was Tesla's automotive sales revenue in dollars for the year ended December 31, 2020, and how does that compare to the prior year in terms of dollar change? | Automotive sales were $26,184 million in 2020, representing a $6,232 million increase compared to 2019. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 38 | JPM FY2020, table, Consolidated Results Of Operations | What was JPMorgan Chase & Co's total provision for credit losses in millions of dollars for the year ended December 31, 2020? | $17,480 million | `evaluation_evidence` | an evaluation-pool or audit passage (5 blocked) |
| 43 | AAPL FY2025, table, Item 8 | What was the total amount of other non-current assets reported by Apple Inc. for the fiscal year ended September 27, 2025? | $83,727 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 43 | AAPL FY2025, table, Item 8 | How much one-time income tax charge did Apple record in the fourth quarter of 2024 as a result of the European Court of Justice's decision on the State Aid case? | $10.2 billion, net | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 49 | XOM FY2019, table, Consolidated Balance Sheet | How much long-term debt did ExxonMobil report in its consolidated balance sheet for the year ended December 31, 2019? | Long-term debt was $26,342 million. | `evaluation_evidence` | an evaluation-pool or audit passage (7 blocked; also 2 validation) |
| 55 | WMT FY2021, table, Item 7 | What was the net cash used in investing activities for Walmart Inc. for the fiscal year ended January 31, 2021, in millions? | $(10,071) million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 57 | AAPL FY2020, table, Item 6 | How many diluted shares did Apple use in computing earnings per share for the fiscal year 2020? | 17,528,214 thousand shares. | `evaluation_evidence` | an evaluation-pool or audit passage (3 blocked) |
| 58 | AMZN FY2022, table, Item 8 | If Amazon exercises the option to extend the maturity of the Term Loan to January 2025, what will the interest rate spread become? | 1.05% | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 61 | TSLA FY2022, narrative, Item 1A | What was Tesla's aggregate outstanding debt as of December 31, 2022? | $2.06 billion | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 63 | XOM FY2024, table, Business Results | How much did ExxonMobil plan to reach in global recycling capacity by 2027 for its Advanced Recycling operations? | 1 billion pounds per year | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 65 | AMZN FY2023, narrative, Item 8 | What was the net cash provided by or used in financing activities for Amazon.com in the year ended December 31, 2023? | $(15,879) million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (7 blocked) |
| 65 | AMZN FY2023, narrative, Item 8 | How much did Amazon.com pay in principal repayments of finance leases during the fiscal year 2022? | $7,941 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 68 | TSLA FY2023, table, Item 8 | How much did excess tax benefits related to stock-based compensation reduce Tesla's tax provision in the year ended December 31, 2021? | $7,123 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 69 | WMT FY2023, table, Item 7 | What was Walmart's return on investment for the fiscal year ended January 31, 2023? | 12.7% | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 69 | WMT FY2023, table, Item 7 | How much were Walmart's total assets as of January 31, 2022, according to the balance sheet data? | $244,860 | `cross_split_evidence` | the visible text of 1 validation passage |
| 72 | AMZN FY2024, table, Item 8 | What was the total other income or expense, net, reported by Amazon for the year ended December 31, 2024, in millions? | $(2,250) million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked; also 1 validation) |
| 73 | JPM FY2023, table, Business Segment Results | What was JPMorgan Chase & Co's average investment securities portfolio balance in millions for the year ended December 31, 2023? | $602,718 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 73 | JPM FY2023, table, Business Segment Results | What amount of held-to-maturity securities did JPMorgan Chase & Co transfer to available-for-sale in connection with the adoption of new hedge accounting guidance effective January 1, 2023? | $7.1 billion | `cross_split_evidence` | the visible text of 1 validation passage |
| 75 | TSLA FY2020, narrative, Item 7 | What were the net proceeds to Tesla, Inc. from the at-the-market offering of common stock completed in December 2020? | approximately $4.99 billion | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 76 | WMT FY2022, table, Item 7 | How did Walmart's total revenues change in percentage terms in fiscal 2022 compared to the prior year? | 2.4% | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 77 | XOM FY2022, table, 18. Disclosures About Segments And Relat | What was ExxonMobil's total revenue in millions of dollars for the year ended December 31, 2022? | 398,675 million dollars | `evaluation_evidence` | only passages blocked for reusing evaluation text (3 blocked) |
| 77 | XOM FY2022, table, 18. Disclosures About Segments And Relat | How much in long-lived assets did ExxonMobil report for Canada as of December 31, 2022? | 31,106 million dollars | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked; also 1 validation) |
| 81 | PFE FY2025, table, Item 1 | Which operating segment of Pfizer in fiscal 2025 is identified as the only reportable segment? | Biopharma | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked; also 2 validation) |
| 84 | XOM FY2021, narrative, 1. Summary Of Accounting Policies | Under what specific circumstance does Exxon Mobil Corp apply the straight-line depreciation method to upstream assets instead of the unit-of-production method? | When the expected life of the asset does not reasonably correlate with that of the underlying reserves | `evaluation_evidence` | only passages blocked for reusing evaluation text (5 blocked) |
| 89 | TSLA FY2025, narrative, Item 1A | What potential operational consequences does Tesla, Inc. identify in its fiscal 2025 risk factors regarding work stoppages by unions at its suppliers and logistics partners? | Delays in the manufacture and sale of products, which may harm the business and operating results. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (3 blocked) |
| 90 | WMT FY2025, narrative, Item 7 | What was the gross profit for Walmart Inc. in fiscal 2025, and how did it compare to the amount reported in fiscal 2024? | Gross profit was 10,203 in fiscal 2025, compared to 9,431 in fiscal 2024. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 98 | XOM FY2019, table, Liquidity And Capital Resources | How many shares of common stock did Exxon Mobil Corporation acquire for the treasury during 2018? | 8 million shares | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |

## 6. AI review of the 112 pairs

**How each pair was judged:** each pair was read against its positive passage's full visible text.
- **Accept:** correct company, period, figure and unit; supported by the quote; a useful financial retrieval
  query. Minor issues are noted but do not exclude.
- **Revise:** a specific error that a reworded question would fix.
- **Reject:** no financial content, or no real answer.

**Precedents from the earlier audits, applied here:**
- **Wrong unit:** an answer that states a wrong unit fails (the validation audits report "none gives a wrong unit");
  a missing unit is minor.
- **Part presented as the whole:** a part given as a total when a larger total exists is revised (validation
  `val_pilot_v2f` 178 and 225).
- **Period:** a question naming no year is revised when the fact changes between filings (validation 98), and
  accepted with a note when it does not.

**Result:** 101 accept (90.2%), 6 revise, 5 reject.

| Aspect (112 pairs) | Pass | Minor | Fail | n/a |
|---|---:|---:|---:|---:|
| Factual accuracy | 103 | 5 | 4 (012, 031, 051, 070) | – |
| Financial relevance | 97 | 10 | 5 (035, 046, 098, 099, 106) | – |
| Company | 112 | – | – | – |
| Period | 96 | 13 | 3 (031, 042, 075) | – |
| Figures (answer figures in the quote) | 83 | – | – | 29 |
| Units | 60 | 22 (no unit) | 1 (012) | 29 |

**Issues noted on accepted pairs:**
- **Boilerplate:** 19, mostly risk factors repeated yearly.
- **Weak financial relevance:** 10. These are holders of record, an insider trading plan, contract terms, employee
  benefits, a COVID-leave date, privacy-law rights and cyber-governance expertise.
- **Inaccurate wording:** 5.
- **No issue:** 67.

**Not accepted (11):**

| Pair | Pos | Decision | Reason |
|---|---:|---|---|
| 012 | 22 | Revise | Wrong unit: Apple's balance sheet counts shares in thousands (every Apple 10-K in the corpus says so), so 15,550,061 is about 15.55 billion shares. |
| 031 | 16 | Revise | Wrong date: $4,002 million is the row "Gross tax contingencies – January 1" for 2023, that is the December 31, 2022 balance. |
| 042 | 38 | Revise | Names no year and points to "the period discussed in the Consolidated Results of Operations". |
| 051 | 10 | Revise | Wrong scope and form: one segment's results (Card Services, BWM) attributed to the Firm, and "how much" answered with a reason. |
| 070 | 96 | Revise | $745 million is only the non-current warranty reserve. Tesla's FY2019 filing gives the reserve as $1,089 million, with a $344 million current portion. |
| 075 | 54 | Revise | Names no filing year, but the split-adjusted IPO price changes between filings: $17.00 in FY2019, $3.40 in FY2020–FY2021, about $1.13 from FY2022 (corpus). |
| 035 | 23 | Reject | A yes/no question answered by the boilerplate "None" of a 471-character Item 5 stub. |
| 046 | 3 | Reject | A page-number lookup in the index of risk functions: navigation, not content. |
| 098 | 97 | Reject | An executive's age. |
| 099 | 97 | Reject | An executive's previous employer and dates. |
| 106 | 28 | Reject | The boilerplate reason the schedules are omitted, from a 730-character cross-reference stub. |

**Patterns:**
- **Tables:** 3 table misreadings.
  - 031 reads an opening-balance row as a year-end balance.
  - 070 gives a part as the whole.
  - 012 states the wrong scale.
- **Scope and context:** 2 errors (042, 051) and 1 ambiguous period (075).
- **Low-value sources:** 5 questions came from passages the source filter let through: an Item 5 stub, an Item 8
  pointer, a risk-function index and an executive-officer table.

Every pair, verbatim:

| Pair | Pos | Source | Type | Question (verbatim) | Answer (verbatim) | Decision | Note |
|---:|---:|---|---|---|---|---|---|
| 001 | 36 | AAPL FY2019, narrative, Item 8 | debt_liquidity | What was the net cash collateral received by Apple related to derivative instruments as of September 28, 2019, and where was it recorded on the balance sheet? | $1.6 billion, recorded as other current liabilities | Accept | Correct: $1.6 billion of collateral received, recorded in other current liabilities. |
| 002 | 92 | AAPL FY2019, table, Item 8 | debt_liquidity | What was the aggregate principal amount of Apple's outstanding floating- and fixed-rate notes as of September 28, 2019? | $101.7 billion | Accept | Correct. |
| 003 | 8 | AAPL FY2020, narrative, Item 8 | debt_liquidity | What was the total amount of Apple's non-current portion of term debt in millions as of fiscal 2020? | $98,667 million | Accept | Correct. |
| 004 | 8 | AAPL FY2020, narrative, Item 8 | accounting_policy | How much net investment in foreign operations did Apple hedge using Japanese yen-denominated notes as of September 28, 2019? | $1.0 billion | Accept | Correct figure; $1.0 billion is the carrying value of the notes designated as the hedge, which the question calls the investment hedged. |
| 005 | 1 | AAPL FY2021, table, Item 7 | financial_table | What was Apple's total other income or expense, net, for fiscal 2021, and what was the year-over-year change percentage? | $258 million, with a (68)% change | Accept | Correct; the total and its change. |
| 006 | 50 | AAPL FY2021, narrative, Item 8 | financial_table | What was the amount of share-based compensation recorded in Apple's statement of stockholders' equity for the fiscal year ended September 25, 2021? | 8,108 | Accept | Correct; no unit (millions). |
| 007 | 50 | AAPL FY2021, narrative, Item 8 | financial_table | How much common stock did Apple repurchase during the fiscal year ended September 25, 2021, as reflected in retained earnings? | (85,502) | Accept | Correct as scoped (the retained-earnings portion); no unit (millions). |
| 008 | 64 | AAPL FY2022, narrative, Item 8 | profitability | How did Apple's retained earnings change from fiscal 2021 to fiscal 2022? | From $5,562 million to $(3,068) million | Accept | Correct; typed profitability. |
| 009 | 78 | AAPL FY2023, table, Item 5 | other | How many shareholders of record did Apple Inc. have as of October 20, 2023? | 23,763 | Accept | Correct; holders of record. |
| 010 | 78 | AAPL FY2023, table, Item 5 | financial_table | What was the approximate dollar value of shares that could still be bought under Apple's repurchase program at the end of the quarter ended September 30, 2023? | $74,069 million | Accept | Correct; remaining repurchase authorization. |
| 011 | 22 | AAPL FY2023, narrative, Item 8 | debt_liquidity | What was the balance of Apple's commercial paper as of September 30, 2023? | $5,985 | Accept | Correct; no unit (millions). |
| 012 | 22 | AAPL FY2023, narrative, Item 8 | financial_table | How many shares of Apple common stock were issued and outstanding at the end of fiscal 2023? | 15,550,061 shares | **Revise** | Wrong unit: Apple's balance sheet counts shares in thousands, so 15,550,061 is about 15.55 billion shares, not 15,550,061 shares. |
| 013 | 71 | AAPL FY2024, narrative, Item 1C | risk | How long has Apple's Head of Corporate Information Security held their current leadership role within the company? | Since 2016 | Accept | Correct; cyber-governance expertise; names no period, stable across the Item 1C filings. |
| 014 | 71 | AAPL FY2024, narrative, Item 1C | risk | Which specific committee of Apple's Board receives updates regarding cybersecurity incidents and reviews related risks with management? | The Audit and Finance Committee | Accept | Correct; names no period, stable across filings. |
| 015 | 15 | AAPL FY2024, table, Item 5 | financial_table | How many shares of Apple Inc. common stock were purchased during the period from June 30, 2024, to August 3, 2024, under the share repurchase plan? | 35,697 thousand shares | Accept | Correct; shares in thousands, as the header says. |
| 016 | 15 | AAPL FY2024, table, Item 5 | financial_table | What was the approximate remaining dollar value of shares that Apple Inc. was authorized to purchase under its repurchase programs as of September 28, 2024? | $89,074 million | Accept | Correct. |
| 017 | 85 | AAPL FY2025, narrative, Item 8 | financial_table | What was Apple's total depreciation and amortization expense for the fiscal year ended September 27, 2025? | 11,698 | Accept | Correct; no unit (millions). |
| 018 | 85 | AAPL FY2025, narrative, Item 8 | debt_liquidity | How much did changes in inventory impact Apple's cash flow from operating activities in fiscal 2024? | (1,046) | Accept | Correct prior-year figure; names its year; no unit (millions). |
| 019 | 44 | AMZN FY2019, narrative, Item 7 | mdna | What was the total sum of shares outstanding plus outstanding stock awards for Amazon in the year ended December 31, 2019? | 512 million | Accept | Correct; a year-end count asked 'in the year'. |
| 020 | 44 | AMZN FY2019, narrative, Item 7 | accounting_policy | Which compensation instrument does Amazon primarily use for equity compensation, and what is the rationale provided for this choice? | Restricted stock units, because this model aligns the long-term interests of shareholders and employees | Accept | Correct; names no period; repeated yearly. |
| 021 | 93 | AMZN FY2019, narrative, Item 8 | accounting_policy | How did Amazon's adoption of the revenue recognition standard affect its retained earnings in fiscal 2018? | increased retained earnings by approximately $650 million | Accept | Correct; the January 1, 2018 adoption. |
| 022 | 79 | AMZN FY2020, narrative, Item 8 | debt_liquidity | How much cash, cash equivalents, marketable securities, and property and equipment had Amazon pledged or restricted as collateral as of December 31, 2020? | $875 million | Accept | Correct. |
| 023 | 37 | AMZN FY2020, table, Item 8 | tax | What was the total provision for income taxes, net reported by Amazon for the fiscal year ended December 31, 2020? | $2,863 million | Accept | Correct. |
| 024 | 37 | AMZN FY2020, table, Item 8 | tax | According to the reconciliation of income taxes, what was the tax impact of foreign earnings and losses for Amazon in the year ended December 31, 2019? | $453 million | Accept | Correct prior-year figure; names its year. |
| 025 | 51 | AMZN FY2021, table, Item 8 | financial_table | What was Amazon's total long-term lease liabilities for finance leases as of December 31, 2021? | $15,670 million | Accept | Correct. |
| 026 | 51 | AMZN FY2021, table, Item 8 | debt_liquidity | How much was the imputed interest deduction for operating leases included in Amazon's lease liability calculation at the end of fiscal 2020? | $5,734 million | Accept | Correct prior-year figure (December 31, 2020). |
| 027 | 2 | AMZN FY2021, table, Item 8 | tax | What was the dollar amount of the tax impact from foreign earnings and losses for Amazon in the year ended December 31, 2021? | $(1,349) million | Accept | Correct. |
| 028 | 2 | AMZN FY2021, table, Item 8 | accounting_policy | How much did Amazon report in excess tax benefits from stock-based compensation for fiscal 2021? | $1.9 billion | Accept | Correct; a tax fact typed accounting_policy. |
| 029 | 9 | AMZN FY2022, narrative, Item 7A | debt_liquidity | What was the total face value of Amazon's long-term debt as of December 31, 2022, and what type of securities did it primarily consist of? | $70.5 billion, primarily consisting of fixed rate unsecured senior notes | Accept | Correct. |
| 030 | 58 | AMZN FY2022, table, Item 8 | debt_liquidity | What was the total amount of long-term debt principal and interest commitments for Amazon in fiscal 2022? | $102,396 million | Accept | Correct; commitments as of December 31, 2022. |
| 031 | 16 | AMZN FY2023, table, Item 8 | financial_table | What was Amazon's gross tax contingencies balance as of December 31, 2023? | $4,002 million | **Revise** | Wrong date: $4,002 million is the row 'Gross tax contingencies – January 1' for 2023, the balance at December 31, 2022; the December 31, 2023 balance lies past the window. |
| 032 | 16 | AMZN FY2023, table, Item 8 | tax | What was the approximate amount of Amazon's foreign net operating loss carryforwards before tax effects as of December 31, 2023? | $10.2 billion | Accept | Correct. |
| 033 | 30 | AMZN FY2024, narrative, Item 9B | other | How many shares of Amazon common stock did CEO Andrew Jassy plan to sell under his Rule 10b5-1 trading plan adopted in November 2024? | 80,400 shares | Accept | Correct; an executive's Rule 10b5-1 plan. |
| 034 | 23 | AMZN FY2025, narrative, Item 5 | other | How many shareholders of record did Amazon have as of January 28, 2026? | 12,165 | Accept | Correct; holders of record. |
| 035 | 23 | AMZN FY2025, narrative, Item 5 | other | Did Amazon report any unregistered securities sales in fiscal 2025? | None | **Reject** | A yes/no question answered by the boilerplate 'None' of a 471-character Item 5 stub. |
| 036 | 86 | AMZN FY2025, table, Item 8 | tax | What was Amazon's income before income taxes attributable to U.S. operations for the fiscal year ended December 31, 2025? | $89,537 million | Accept | Correct. |
| 037 | 86 | AMZN FY2025, table, Item 8 | tax | How much did research and development tax credits reduce Amazon's effective tax rate for the year ended December 31, 2024? | (3.9) percent | Accept | Correct; the 2024 rate column. |
| 038 | 94 | JPM FY2019, narrative, Business Segment Results | revenue | What was the total net revenue for JPMorgan Chase & Co's Total Markets segment in fiscal 2019? | $20,912 | Accept | Correct figure; Markets is a CIB business, not a segment; no unit (millions). |
| 039 | 94 | JPM FY2019, narrative, Business Segment Results | financial_table | How much did JPMorgan Chase & Co report in Noninterest revenue for its Fixed Income Markets division in fiscal 2018? | $9,119 | Accept | Correct prior-year column; no unit (millions). |
| 040 | 45 | JPM FY2019, table, Credit And Investment Risk Management | financial_table | What was the total balance of JPMorgan Chase's government guaranteed residential mortgage loans as of December 31, 2019? | $2,936 million | Accept | Correct. |
| 041 | 45 | JPM FY2019, table, Credit And Investment Risk Management | debt_liquidity | What was the carrying value of JPMorgan Chase's outstanding HELOCs at December 31, 2019? | $22 billion | Accept | Correct. |
| 042 | 38 | JPM FY2020, table, Consolidated Results Of Operations | mdna | How much did JPMorgan Chase & Co's average interest-earning assets increase in the period discussed in the Consolidated Results of Operations? | $434 billion | **Revise** | Names no year and points to 'the period discussed in the Consolidated Results of Operations'; the $434 billion is 2020's increase. |
| 043 | 80 | JPM FY2021, narrative, Item 1A | risk | According to JPMorgan Chase's fiscal 2021 disclosure, what outcomes could result from a failure to appropriately assess and manage relationships with internal and external service providers? | Potential harm to clients and customers with associated liability, regulatory fines or penalties, lower revenues and opportunity costs, increased operational costs, or harm to reputation. | Accept | Correct; vendor-oversight risk factor. |
| 044 | 31 | JPM FY2021, narrative, Note 4 – Credit Risk Concentrations | financial_table | What was JPMorgan Chase's total credit exposure to the Oil & Gas sector in fiscal 2021? | 42,606 | Accept | Correct; no unit (millions). |
| 045 | 3 | JPM FY2022, table, Firmwide Risk Management | risk | In JPMorgan Chase & Co's fiscal 2022 Form 10-K, what is the senior executive forum responsible for overseeing valuation activities across the firm called, and who composes it? | The Firmwide Valuation Governance Forum (VGF), composed of senior finance and risk executives | Accept | Correct; risk governance. |
| 046 | 3 | JPM FY2022, table, Firmwide Risk Management | financial_table | According to the risk governance table in JPMorgan Chase & Co's fiscal 2022 Form 10-K, which specific risk function is assigned to pages 130? | Investment Portfolio Risk | **Reject** | A page-number lookup in the index of risk functions ('assigned to pages 130'): navigation, no financial content. |
| 047 | 24 | JPM FY2023, narrative, Item 1A | risk | In fiscal 2023, what specific types of security flaws does JPMorgan Chase identify as potential vulnerabilities in both internally developed and externally provided software? | zero-day vulnerabilities | Accept | Correct; cybersecurity risk factor. |
| 048 | 24 | JPM FY2023, narrative, Item 1A | risk | According to JPMorgan Chase's fiscal 2023 risk factors, what is a potential negative consequence of a successful penetration of its systems or those of a vendor involving a malicious actor? | demands that JPMorgan Chase pay a ransom to a malicious actor that has perpetrated a cybersecurity breach | Accept | Correct; one of several listed consequences. |
| 049 | 66 | JPM FY2024, table, Credit And Investment Risk Management | financial_table | What was JPMorgan Chase's total wholesale credit exposure in the Real Estate industry as of December 31, 2024, and how much of that was categorized as criticized performing? | $207,050 million total exposure; $10,858 million criticized performing | Accept | Correct columns: credit exposure and criticized performing. |
| 050 | 17 | JPM FY2024, table, Country Risk Management | financial_table | What was JPMorgan Chase & Co's total country exposure to Germany as of December 31, 2024? | $103.9 billion | Accept | Correct. |
| 051 | 10 | JPM FY2025, narrative, Business Segment & Corporate Results | mdna | How much of JPMorgan Chase & Co's noninterest expense increase in fiscal 2025 was attributed to auto lease depreciation? | Higher auto lease depreciation on higher auto operating lease assets | **Revise** | Wrong scope and form: the passage gives one segment's results (Card Services, BWM, the Apple Card transaction), not the Firm's, and 'how much' is answered with a reason, not an amount. |
| 052 | 53 | PFE FY2019, table, Item 1 | financial_table | Which Upjohn products generated direct product revenues exceeding $1 billion for Pfizer in fiscal 2019? | Lyrica and Lipitor | Accept | Correct reading of the flattened table (two products in 2019). |
| 053 | 53 | PFE FY2019, table, Item 1 | other | Which therapeutic areas does Pfizer identify as being served by Upjohn products, listing one specific drug for each area mentioned? | Cardiovascular (Lipitor, Norvasc and Revatio); Pain and neurology (Lyrica and Celebrex); Psychiatry (Effexor, Zoloft and Xanax); Urology (Viagra); and Ophthalmology (Xalatan/Xalacom) | Accept | Correct; asks for one drug per area but the answer lists all; names no period. |
| 054 | 99 | PFE FY2019, narrative, Item 7A | mdna | How does Pfizer manage its interest rate risk regarding its investment portfolio? | Pfizer strives to maintain a predominantly floating-rate basis position, though this strategy may change based on prevailing market conditions. | Accept | Correct; names no period; an investment-rate policy repeated yearly. |
| 055 | 95 | PFE FY2020, table, Item 7 | debt_liquidity | What were Pfizer's short-term and long-term credit ratings from Moody's as of fiscal 2020, and what was the assigned outlook? | Short-term rating P-1, long-term rating A2, with a Stable outlook | Accept | Correct. |
| 056 | 74 | PFE FY2021, narrative, Item 1A | risk | What actions do private payers and managed care entities, such as PBMs, take to manage drug utilization and costs according to Pfizer's fiscal 2021 risk factors? | They employ formularies to control costs and encourage utilization of certain drugs, including through formulary inclusion or favorable formulary placement. | Accept | Correct; managed-care risk factor. |
| 057 | 25 | PFE FY2021, table, Item 7 | financial_table | What were Pfizer's short-term and long-term debt ratings according to S&P as of fiscal 2021? | A-1+ short-term and A+ long-term | Accept | Correct. |
| 058 | 25 | PFE FY2021, table, Item 7 | debt_liquidity | What was the amount of Pfizer's remaining share-purchase authorization at December 31, 2021? | approximately $5.3 billion | Accept | Correct. |
| 059 | 60 | PFE FY2022, narrative, Item 7 | financial_table | What was the amount of non-GAAP adjusted cost of sales for Pfizer in the year ended December 31, 2020? | $8,386 | Accept | Correct 2020 column of the reconciliation; names its year; no unit (millions). |
| 060 | 11 | PFE FY2022, table, Item 7 | financial_table | What was the total amount of cash provided by operating activities from continuing operations for Pfizer for the year ended December 31, 2022? | $29,267 million | Accept | Correct. |
| 061 | 11 | PFE FY2022, table, Item 7 | mdna | According to Pfizer's MD&A, what was the total amount of charges recorded in 2022 that primarily represented the equity-method pro rata share of restructuring and separation costs for Haleon/the Consumer Healthcare JV? | $307 million | Accept | Correct. |
| 062 | 88 | PFE FY2023, narrative, Item 7 | tax | What were Pfizer's effective tax rates for GAAP Reported income from continuing operations in fiscal 2023, 2022, and 2021? | (105.4)% in 2023, 9.6% in 2022 and 7.6% in 2021 | Accept | Correct. |
| 063 | 88 | PFE FY2023, narrative, Item 7 | accounting_policy | What components made up the $1.2 billion in Restructuring charges and certain acquisition-related costs for Pfizer in 2023? | $785 million of integration costs and other charges, $190 million of transaction costs and $125 million of employee termination-related charges | Accept | Correct; typed accounting_policy. |
| 064 | 39 | PFE FY2023, table, Item 7 | financial_table | What were Pfizer's long-term and short-term debt ratings assigned by Moody's and S&P as of the filing date for fiscal 2023? | Moody's assigned a P-1 short-term rating and A2 long-term rating; S&P assigned an A-1 short-term rating and A long-term rating. | Accept | Correct; the ratings as of the filing date. |
| 065 | 67 | PFE FY2024, table, Item 8 | accounting_policy | What was the healthcare cost trend rate assumed for the year following fiscal 2024 for Pfizer's U.S. postretirement benefit plans? | 7.5% | Accept | Correct. |
| 066 | 67 | PFE FY2024, table, Item 8 | financial_table | What was the beginning benefit obligation for Pfizer's U.S. pension plans in millions of dollars for the year ended December 31, 2024? | $10,756 million | Accept | Correct column (U.S. pension plans, 2024). |
| 067 | 32 | PFE FY2025, narrative, Item 7 | financial_table | How much did certain asset impairments affect Pfizer's other income or deductions for the year ended December 31, 2025? | (4,940) | Accept | Correct cell; a non-GAAP reconciling amount whose direction the question does not state; no unit (millions). |
| 068 | 47 | TSLA FY2019, table, Item 7 | debt_liquidity | What was the total amount of finance lease obligations, including imputed interest, for Tesla, Inc. as of December 31, 2019? | $1,795 million | Accept | Correct. |
| 069 | 47 | TSLA FY2019, table, Item 7 | financial_table | How much of Tesla's total purchase obligations as of December 31, 2019, was attributable to purchase orders issued under binding and enforceable agreements? | $2.50 billion | Accept | Correct. |
| 070 | 96 | TSLA FY2019, table, Item 8 | financial_table | What was the balance of Tesla's accrued warranty reserve as of December 31, 2019, in millions? | $745 million | **Revise** | $745 million is only the non-current warranty reserve (the table lists other long-term liabilities); the reserve's balance was $1,089 million with the $344 million current portion. |
| 071 | 96 | TSLA FY2019, table, Item 8 | accounting_policy | Why did Tesla de-recognize its build-to-suit lease liabilities and reclassify them as operating lease right-of-use liabilities at the start of fiscal 2019? | As a result of the adoption of the new lease standard on January 1, 2019 | Accept | Correct. |
| 072 | 75 | TSLA FY2020, narrative, Item 7 | accounting_policy | How many vesting tranches comprised the 2018 CEO Performance Award approved by Tesla, Inc. stockholders in March 2018? | 12 vesting tranches | Accept | Correct. |
| 073 | 33 | TSLA FY2020, table, Item 7 | mdna | According to Tesla's management discussion, how were payroll-related benefits received during the first half of 2020 treated in the financial results relative to idle capacity charges? | Payroll-related benefits were primarily netted against disclosed idle capacity charges and marginally reduced operating expenses. | Accept | Correct. |
| 074 | 54 | TSLA FY2021, narrative, Item 5 | other | How many holders of record did Tesla, Inc. have for its common stock as of January 31, 2022? | 7,051 | Accept | Correct; holders of record. |
| 075 | 54 | TSLA FY2021, narrative, Item 5 | other | What was the initial public offering price per share for Tesla, Inc. stock on June 28, 2010, after adjusting for the stock split? | $3.40 per share | **Revise** | Names no filing year, and the split-adjusted IPO price changes between filings: $17.00 (FY2019), $3.40 (FY2020-FY2021), about $1.13 (FY2022 on). |
| 076 | 61 | TSLA FY2022, narrative, Item 1A | risk | How might the early retirement of manufacturing equipment affect Tesla's results of operations? | It would shorten the useful life, accelerate depreciation, and harm results of operations. | Accept | Correct; names no period; repeated risk factor. |
| 077 | 5 | TSLA FY2022, narrative, Item 3 | other | What was the final settled fine amount imposed on Tesla's German subsidiary for battery take-back non-compliance in fiscal 2022? | 600,000 euro | Accept | Correct; the November 2022 settlement. |
| 078 | 5 | TSLA FY2022, narrative, Item 3 | risk | What is the current expected impact on Tesla's business from the California district attorney investigation into waste segregation practices? | Not currently expected to have a material adverse impact | Accept | Correct; names no period. |
| 079 | 19 | TSLA FY2023, narrative, Item 1A | risk | Which specific manufacturing facilities and future sites did Tesla identify in its fiscal 2023 10-K as requiring successful implementation of efficient and cost-effective manufacturing capabilities? | California, Nevada, Texas, China, Germany and any future sites such as Mexico | Accept | Correct. |
| 080 | 19 | TSLA FY2023, narrative, Item 1A | risk | What specific past supplier and automation challenges did Tesla cite as examples of production ramp delays for its initial Model 3 and Model X launches? | unanticipated supplier issues that led to delays during the initial ramp of our first Model X and experienced challenges with a supplier and with ramping full automation for certain of our initial Model 3 manufacturing processes | Accept | Correct; the answer copies the sentence. |
| 081 | 68 | TSLA FY2023, table, Item 8 | tax | What was Tesla's total current income tax provision for fiscal 2023? | $1,348 million | Accept | Correct. |
| 082 | 26 | TSLA FY2024, table, Item 8 | accounting_policy | How did Tesla classify its money market funds and digital assets within the fair value hierarchy for fiscal 2024, and what method was used to value them? | Level I, using quoted prices in active markets | Accept | Correct. |
| 083 | 89 | TSLA FY2025, narrative, Item 1A | risk | Why might Tesla, Inc. be compelled to initiate product recalls for its fiscal 2025 reporting period even if it disagrees with the defect determination? | Testing or investigations by government regulators or industry groups may compel recalls or lead to negative public perceptions about product safety. | Accept | Correct; recall risk factor. |
| 084 | 40 | TSLA FY2025, narrative, Item 1A | accounting_policy | How does Tesla describe the basis for its warranty reserve estimates in fiscal 2025? | Based on actual claims incurred to date and an estimate of the nature, frequency and costs of future claims. | Accept | Correct. |
| 085 | 62 | WMT FY2020, narrative, Item 7 | accounting_policy | What threshold for change in retail square feet causes a store with a changed format to be excluded from Walmart's comparable sales calculation? | more than five percent | Accept | Correct; comparable-sales definition; names no period. |
| 086 | 62 | WMT FY2020, narrative, Item 7 | accounting_policy | How long must Walmart own an acquisition before its sales are included in the comparable sales metric? | 12 months | Accept | Correct; names no period. |
| 087 | 100 | WMT FY2021, narrative, Item 1A | risk | What specific rights do California residents have under the CCPA and CPRA regarding their personal information as described in Walmart's fiscal 2021 risk factors? | Rights to access, correct, and require deletion of personal information, opt out of certain sharing, and receive detailed usage information. | Accept | Correct; describes the privacy laws rather than Walmart's finances. |
| 088 | 6 | WMT FY2021, narrative, Item 8 | financial_table | What was the amount of losses on disposal of business operations reported by Walmart Inc. for the fiscal year ended January 31, 2021? | 8,401 | Accept | Correct; no unit (millions). |
| 089 | 6 | WMT FY2021, narrative, Item 8 | financial_table | How did the change in accounts payable for Walmart Inc. compare between the fiscal year ended January 31, 2021 and the fiscal year ended January 31, 2020? | 6,966 in fiscal 2021 and (274) in fiscal 2020 | Accept | Correct; no unit (millions). |
| 090 | 27 | WMT FY2022, narrative, Item 1 | other | What specific company-paid benefits does Walmart Inc. provide to its full-time and eligible part-time associates in the U.S. during fiscal 2022? | 401(k) match, maternity leave, a paid parental leave program, paid time off, Associate Stock Purchase Plan match, life insurance, behavioral health services, and a store discount card or Sam's Club membership | Accept | Correct; employee benefits. |
| 091 | 27 | WMT FY2022, narrative, Item 1 | other | Until what date did Walmart Inc. extend its COVID-leave policy for associates in fiscal 2022? | March 31, 2022 | Accept | Correct; a COVID-leave policy date. |
| 092 | 76 | WMT FY2022, table, Item 7 | financial_table | What was the amount of net cash provided by operating activities for Walmart in the fiscal year ended January 31, 2022? | $24,181 million | Accept | Correct. |
| 093 | 83 | WMT FY2024, narrative, Item 1A | risk | According to Walmart's fiscal 2024 risk factors, what specific operational areas could be impacted by a global or regional health pandemic? | Business operations, demand for products and services, in-stock positions, costs of doing business, access to inventory, supply chain operations, and the extent and duration of containment measures. | Accept | Correct; pandemic risk factor. |
| 094 | 83 | WMT FY2024, narrative, Item 1A | risk | What potential changes in customer behavior and economic conditions does Walmart identify as factors that could challenge its ability to adjust inventory levels during a health crisis? | Customer demand for certain products may fluctuate, customer behaviors may change, and consumer disposable income could be negatively impacted. | Accept | Correct; names no period. |
| 095 | 34 | WMT FY2024, narrative, Item 1A | risk | How does Walmart describe the potential financial impact of its failure to retain qualified associates on its operations? | It could materially adversely affect the company's financial performance and its ability to conduct and expand operations. | Accept | Correct; names no period; a risk-factor heading. |
| 096 | 41 | WMT FY2025, table, Item 8 | financial_table | What was Walmart's operating lease cost for the fiscal year ended January 31, 2025? | $2,347 million | Accept | Correct. |
| 097 | 41 | WMT FY2025, table, Item 8 | financial_table | How much cash did Walmart pay for operating cash flows from finance leases in the fiscal year ended January 31, 2024? | $315 million | Accept | Correct prior-year column; names its year. |
| 098 | 97 | WMT FY2026, table, Item 1 | other | What was the age of Walmart's Chief Executive Officer as of the filing of the fiscal 2026 Form 10-K? | 51 years old | **Reject** | An executive's age: no financial content. |
| 099 | 97 | WMT FY2026, table, Item 1 | other | How long did Daniel J. Bartlett serve in his role prior to becoming Executive Vice President, Corporate Affairs at Walmart in June 2013? | From November 2007 to June 2013 | **Reject** | An executive's previous employer and dates: no financial content; 'how long' is answered with a date range. |
| 100 | 48 | WMT FY2026, narrative, Item 1A | risk | What specific negative incidents does Walmart Inc. identify as potential drivers of customer dissatisfaction and consumer boycotts in its fiscal 2026 risk factors? | Loss of merchandise due to shrink or theft, ineffective or misused AI technologies, flawed AI search and discovery results, and data breaches resulting from cyberattacks. | Accept | Correct. |
| 101 | 48 | WMT FY2026, narrative, Item 1A | risk | How might a failure to adequately predict customer demand or optimize distribution centers impact Walmart Inc.'s business operations in fiscal 2026? | It could result in excess or insufficient inventory, service interruptions, and increased costs, which could significantly harm the business. | Accept | Correct. |
| 102 | 91 | XOM FY2020, narrative, Business Environment And Risk Assessment | other | What is the projected growth in global demand for liquid fuels by 2040 compared to 2018 for Exxon Mobil Corp? | approximately 110 million barrels of oil equivalent per day, an increase of about 9 percent | Accept | Correct; global demand in ExxonMobil's outlook, which the question attributes to ExxonMobil. |
| 103 | 42 | XOM FY2020, table, 12. Earnings Per Share | profitability | What was the net income or loss attributable to ExxonMobil for fiscal 2020? | (22,440) millions of dollars | Accept | Correct. |
| 104 | 42 | XOM FY2020, table, 12. Earnings Per Share | financial_table | How many weighted average common shares did ExxonMobil have outstanding in 2019? | 4,270 millions of shares | Accept | Correct prior-year figure; names its year. |
| 105 | 28 | XOM FY2021, narrative, Item 8 | other | Which independent registered public accounting firm audited Exxon Mobil's consolidated financial statements for the fiscal year ended December 31, 2021? | PricewaterhouseCoopers LLP | Accept | Correct; the auditor, from a 730-character Item 8 pointer. |
| 106 | 28 | XOM FY2021, narrative, Item 8 | other | Why were Financial Statement Schedules omitted from Exxon Mobil's Form 10-K for the period ended December 31, 2021? | They are not applicable or the required information is shown in the consolidated financial statements or notes thereto. | **Reject** | The boilerplate reason schedules are omitted, from a cross-reference stub. |
| 107 | 35 | XOM FY2022, narrative, Item 2 | other | What is the range of production periods for exploration and production activities in Angola under Exxon Mobil's 2022 contract terms? | 20 to 30 years | Accept | Correct; contract terms. |
| 108 | 35 | XOM FY2022, narrative, Item 2 | other | How long is the production period for crude oil in Equatorial Guinea according to the production sharing contracts mentioned in Exxon Mobil's 2022 Form 10-K? | 30 years | Accept | Correct; contract terms. |
| 109 | 63 | XOM FY2024, table, Business Results | financial_table | What was ExxonMobil's total Chemical Products earnings (U.S. GAAP) in fiscal 2024? | $2,577 million | Accept | Correct. |
| 110 | 70 | XOM FY2025, narrative, Business Environment | other | According to Exxon Mobil's fiscal 2025 Form 10-K, what is the projected global demand for liquid fuels by 2050 in terms of oil-equivalent barrels per day and the percentage change from 2024? | Nearly 115 million oil-equivalent barrels per day, an increase of about 10 percent from 2024. | Accept | Correct; an energy-outlook projection. |
| 111 | 7 | XOM FY2025, narrative, Note 18. Incentive Program | accounting_policy | How many shares remained available for award under ExxonMobil's 2003 Incentive Program at the end of fiscal 2025? | 41 million shares | Accept | Correct. |
| 112 | 7 | XOM FY2025, narrative, Note 18. Incentive Program | financial_table | What were the total grants of restricted stock and units under the 2003 Incentive Program for ExxonMobil in fiscal 2024? | 10,393 thousand | Accept | Correct. |

## 7. Evidence leakage and false-negative protections

**Leakage, on the 101 included pairs (all from the trial build, §10):**

| Check | Result |
|---|---|
| Evidence rule, re-checked at the candidate build | 0 leak passages; every included pair passes |
| Lexical overlap with the permitted evaluation questions (word Jaccard; reject at 0.8) | Highest 0.346; none above 0.5 |
| Semantic screen (`screen-questions --pairs --candidates train_candidates_v2`, in scratch) | 178 questions (the 112 and the 66) against the 10 permitted question texts, `query_id` and text only. **0 flagged at 0.9**; highest 0.8017. Of the 101: 1 at or above 0.80, 7 at or above 0.75, 20 at or above 0.70; median 0.645. |
| The one question in the proposed 0.80–0.90 review band | Pair 005, Apple's total other income for fiscal 2021, against `q_0002`. They share the company and fiscal year but ask different facts. Not a paraphrase; it may stay. The threshold was not changed (D2 is still yours). |
| Cross-split | 6 excluded at `build-pairs`. `load_validation_set(validation_pairs_v4, training=…)` passes against all 167 trial training pairs. |
| Quotes | All 101 lie in the positive's visible text, within BGE's 512-token window: 99 verbatim, 1 as segments joined by an ellipsis, 1 with skipped cells. The latest ends at token 509. |

**False negatives:**
- **Equivalents:** 50 of the 101 record exact equivalents (114 passages) and 53 record uncertain ones (139 passages).
  The trainer never uses any of them as a negative.
- **Conflicts:** `NegativeGuard` finds 89 conflicting pairs among the 167 trial pairs, and the planner keeps them in
  different batches. Two examples:
  - Amazon's tax tables in the FY2020 and FY2021 filings (pairs 023, 024, 027 and 028);
  - Apple's repurchase tables for FY2023 and FY2024 (pairs 009, 010, 015 and 016).
- **Batches:** conflict-free plans of the 167 pairs fill well only at small sizes.

  | Batch size | Batches | Full | Mean fill |
  |---:|---:|---:|---:|
  | 16 | 11 | 10 | 0.949 |
  | 32 | 7 | 5 | 0.746 |
  | 64 | 5 | 2 | 0.522 |

- **An independent probe:** for each included pair, any other included pair's positive was checked for the answer's
  figure (4 or more digits, not a year) wherever the guard would still allow it as a negative. 2 hits, both chance
  matches in another company's passage:
  - Amazon's 1,349 in an ExxonMobil reserves passage;
  - Walmart's 2,347 in a JPMorgan credit-exposure passage.

  No unrecorded equivalent was found.
- **Known gaps remain:** paraphrased evidence, rounded figures and changed row labels.

## 8. Duplicate questions

| Against | Same normalised question | Highest word Jaccard | Cosine ≥ 0.90 (BGE, zero-shot) |
|---|---:|---:|---|
| Within the 101 | 0 | 0.63 | 2: Apple's remaining repurchase authorization, FY2023 against FY2024 (0.927); Pfizer's ratings, FY2021 against FY2023 (0.908) |
| The 66 of `train_candidates_v2` | 0 | 0.565 | 1 (0.911): Amazon finance-lease templates, different facts |
| `validation_pairs_v4` (93) | 0 | 0.60 | 2: Amazon's long-term finance lease liabilities at December 31, 2021 against operating leases at 2024 (0.937); Amazon's long-term debt face value at 2022 against 2019 (0.908) |
| Earlier train generation (v1, v2) | 0 | 0.565 | 4 (highest 0.912): the same templates for other years or measures |

Every cosine match is the same template asked about a different fact, year or measure, with a different positive. No
question repeats another.

## 9. Company and fiscal-year accuracy and distribution

**Accuracy:**
- **Company:** every question names its positive's company (101 of 101 included, and all 112).
- **Period:**
  - 84 included questions name the filing's fiscal year or the prior year;
  - 5 name only another year, each legitimately: two holders-of-record dates after year-end, a three-year
    reconciliation's first year, the 2018 award, and an outlook's base year;
  - 12 name no year, all for facts that do not change between filings.

  The period errors are the 3 revised pairs (§6).

**Distribution by company:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Planned | 14 | 14 | 14 | 15 | 14 | 15 | 14 |
| Done | 14 | 14 | 13 | 15 | 14 | 15 | 14 |
| Pass validator | 24 | 24 | 18 | 18 | 22 | 26 | 19 |
| Pairs built | 18 | 19 | 14 | 16 | 17 | 17 | 11 |
| **Included** | **17** | **17** | **11** | **16** | **15** | **15** | **10** |
| With the 66 (167) | 27 | 25 | 24 | 24 | 26 | 22 | 19 |

**Distribution by fiscal year:**

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Planned | 13 | 14 | 15 | 14 | 14 | 14 | 14 | 2 |
| Included | 15 | 13 | 17 | 15 | 12 | 14 | 13 | 2 |
| With the 66 (167) | 25 | 23 | 26 | 21 | 21 | 25 | 23 | 3 |

- **Companies:** balanced.
  - ExxonMobil is lowest: the evidence rule took 8 of its 19 questions. Walmart lost 9 of 26 and the others 2–6.
  - JPMorgan is next: one failed passage, 4 evidence exclusions and 3 review exclusions.
  - Pfizer, the weakest at the validator (18 of 30), lost only 2 afterwards.
- **JPMorgan's share:** 10.9% of the 101 and 14.4% of the 167; the largest company share in the 167 is Apple's 16.2%.
- **Years:** balanced over the batch, as the readiness audit expected once the plan finished.
- **Passage kind:** 43 table and 58 narrative pairs (49 and 51 planned passages).
- **Question type:**
  - `financial_table` 29, `risk` 18;
  - `debt_liquidity` 13, `accounting_policy` 13, `other` 13;
  - `tax` 8, `mdna` 4, `profitability` 2, `revenue` 1.

## 10. Provenance and consistency of the included pairs

**The trial build (scratch copy only):**
- **Sources:**
  - `pilot_v3` and `pilot_v2` with their existing decisions;
  - this batch's pairs with the new decisions;
  - the scratch screening record.
- **Result:** 167 candidates (51 + 15 + 101).
- **Batch 1's pair source:** 274 rows, because `build-pairs` also builds 162 validation pairs from the validation
  generation records. Those have no decision and are excluded (`no_quality_decision`). They are unused, since
  validation comes from `build-validation`.

**Provenance on all 101 included records:** one identical block of 11 fields:
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
- the decisions file;
- the screening record (`semantic_overlap`, threshold 0.9).

It is labelled "AI quality review (accept only); not human-validated; not gold".

**Checks run on the trial set:**
- `verify_candidate_provenance` traces all 167 back to their records and decisions.
- `load_candidates`, the trainer's fail-closed loader, loads all 167.
- **The 66 are unchanged:** a rebuild of `train_candidates_v2`'s sources in scratch gives byte-identical files,
  `train_candidates.jsonl` `908d4c79…` and `candidates_log.jsonl` `c8d38f96…`.

## 11. The failed passage: position 87

**The passage:** `passage_filing_jpm_10-k_9047c6aa2394_000008_ac519db3411f`, JPMorgan FY2020, "Business Segment
Results", a table passage of 2,063 visible characters. It holds:
- footnotes on COVID-19 payment deferrals and PPP loans;
- the start of the CCB "Selected metrics" table: households, branches, digital customers, card volume, deposits.

**The record:**
- `status: failed`, 1 attempt, HTTP 200;
- `error_code: length`, `request_failed: true`;
- not rate-limited, not stopped by the budget;
- usage 1,231 prompt and **768 completion** tokens (1,999);
- no candidates and no response hash.

**Why it failed:**
- **The cause:** the reply reached `max_completion_tokens` (768) before it finished (`finish_reason: length`).
- **Why it was not retried:** the generator never accepts a truncated reply (`_reply_text`). It also does not retry
  one, since the same limit would usually cut it again (`_generate_one`). The passage was therefore recorded as
  failed after one request.
- **What is not known:** the reply text is not kept by design, so its content is not on disk.
  - The 99 successful replies used 163–504 completion tokens (median 261).
  - Two questions with quotes of at most 300 characters fit in that range.
  - This reply was therefore abnormally long. Overlong table-row quotes or repetition are possible causes. So is
    reasoning text: `reasoning_effort: none` should suppress it, but any that was produced can count against the
    limit.
- **The passage itself is not the problem:** it is an approved train passage that passed the source filter.

**Consequences:**
- **Data:** 1 of 100 positions is lost. JPMorgan keeps 13 of its 14 passages and FY2020 13 of 14, so the balance is
  unaffected.
- **Resume:** `generate --plan train_v3_p1_batch1` now skips it (`skipped_failed`) unless `--retry-failed` is given,
  so it is failed but not unrecoverable.
- **A retry:**
  - At 768 tokens it would probably fail again.
  - A higher limit needs `--continue-run`, because `max_completion_tokens` is a request setting. That is a recorded
    continuation; it does not change what an accepted reply says.

**Recommendation:** leave it failed. The gate passes without it. If you want it, retry once with a higher limit,
before any real `build-pairs` (§12), at a cost of about 2,000–2,500 tokens.

## 12. Sequencing: pair sets are write-once and bound to the records file

This is a pipeline constraint for continuing P1. It does not reflect on batch 1's quality.

- **`build-pairs` is write-once per preparation:**
  - `write_pairs` refuses a pair set that differs from the one present;
  - pairs from new generation records belong in a new pair set, which the code does not yet provide.
- **Records hash:**
  - The pair set records the train records file's SHA-256.
  - `build-candidates` and `verify_candidate_provenance`, which the Kaggle bundle builder and the preflight call,
    refuse once that file changes.
  - A trainer's `load_candidates` does not re-check it.
- **Pair IDs:** they are numbered in corpus order over all train records. With more records, batch 1's IDs change.
  A decisions file bound to the old IDs then fails closed, because a decision may not move to another question.
- **Tested in scratch:**
  - after a one-byte change to the records file, `build-candidates` refused;
  - `verify_candidate_provenance` raised a `CandidateError`;
  - `load_candidates` still loaded the set.
- **The consequence:** batches 2–6 append to the same records file. So running the real `build-pairs` and
  `train_candidates_v3` now would:
  1. leave batches 2–6 unable to get a pair set in `bge_base_ft_v3`;
  2. make `train_candidates_v3` fail the bundle and preflight check as soon as batch 2's first record is written.

**Options (your decision):**
- **A, recommended, no code change:**
  1. Generate all of P1 first, and gate each batch by an exact scratch replay, as here.
  2. Run the real `build-pairs` once, at the end.
  3. Carry this review over into a new decisions file keyed by the final pair IDs. Its rows already carry
     `passage_id` and `question_sha256`, so the mapping is mechanical.
- **B:** a code change for batch-wise pair sets, each bound to a frozen snapshot of the records. It needs your
  approval and new tests.

`data/train_review/train_v3_p1_batch1_quality_decisions.jsonl` is bound to the pair set that the current 100 records
produce. The scratch replay confirmed that pair set exactly: its pairs match `build-pairs`' output, ID for ID.

## 13. Recommendation

- **Batch 1 quality gate: PASS** on every approved criterion (§1).
- **Continuing P1 generation: GO**, under the four conditions of §1:
  1. the Groq budget approval;
  2. the sequencing in §12 (option A, unless you choose B);
  3. position 87 left failed, or retried before the final pair build;
  4. one generator.
- **Batch 2:** each later batch needs its own write-once plan, which excludes the overlap groups already used. The
  readiness audit's D6, on JPMorgan's share rising in later batches, stays open.
- **The three-seed experiment: still NO-GO**, until B1–B3 of the readiness audit are resolved.

**Your decisions:**

| # | Decision | Recommendation |
|---|---|---|
| E1 | Groq budget for batches 2–6 (about 740,000 tokens at batch 1's rate), or stop at batch 1 as a labelled pilot | Approve if the experiment is to reach its planned size |
| E2 | Sequencing (§12): A, generate everything first, or B, a code change for batch-wise pair sets | A |
| E3 | Position 87: leave failed, or retry once with a higher token limit through a recorded continuation | Leave failed |
| E4 | The semantic review band (readiness D2): pair 005 at 0.8017 is the only case. It is reviewed here as a different fact. | Record the band and keep pair 005 |
| E5 | The 16 conservative `evaluation_evidence` exclusions and the 3 false validator rejections | Keep the rules unchanged through P1 |

## Appendix: how the results were obtained

Everything below ran in a scratch copy of `data/processed/train/bge_base_ft_v3`. It used a copy of
`configs/train/data_prep_v3.yaml` that differs only in `output_dir`, the venv, and `HF_HUB_OFFLINE=1`.

```bash
python3 scripts/02_generate_queries.py revalidate --config SCRATCH/data_prep_v3_scratch.yaml --split train --dry-run
python3 scripts/02_generate_queries.py build-pairs --config SCRATCH/data_prep_v3_scratch.yaml
python3 scripts/02_generate_queries.py screen-questions --config SCRATCH/data_prep_v3_scratch.yaml \
    --name train_v3_p1_batch1 --pairs --candidates train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_v2check.yaml   # byte-identical to train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_trial.yaml     # pilot_v3, pilot_v2, this batch; semantic_screening
```

**Other scripts:**
- A trace of every candidate through `build_pairs`' own functions. It matches the written pairs, ID for ID.
- The trainer's `load_candidates` and `plan_batches`, `verify_candidate_provenance`, and `load_validation_set` with
  the training pairs.
- The duplicate and false-negative probes.
- Corpus look-ups for pairs 012, 070 and 075.

No evaluation file was read except the 10 permitted question texts, through the screen's own loader and the lexical
check.
