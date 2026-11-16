# P1 batch 3 (`train_v3_p1_batch3`): generation audit and quality gate

Status: 2026-11-07.
- **What this is:**
  - an audit of the 100 generation records of plan `train_v3_p1_batch3` in `bge_base_ft_v3`;
  - an AI review (Claude) of the 126 pairs they build;
  - an investigation of answers built from the model's reasoning rather than the passage, over all three batches;
  - the batch quality gate, a comparison with batches 1 and 2, and a reassessment of batches 4–6.

  It uses the method of the [batch 1](train_v3_p1_batch1_generation_audit.md) and
  [batch 2](train_v3_p1_batch2_generation_audit.md) audits. It is not human annotation, and no decision here is a gold
  label.
- **Where the decisions are:** `data/train_review/train_v3_p1_batch3_quality_decisions.jsonl`.
  - 126 rows (108 accept, 14 revise, 4 reject), each `reviewer: ai_review` and `gold: false`.
  - Each row is bound to its `pair_id`, `passage_id` and the SHA-256 of its exact question. The pair IDs are those of
    the pair set that the current 300 train records build (§13).
  - SHA-256 `04f53d6f…36cc`.
- **What was not done:**
  - no question was generated and no API request was sent;
  - no generation record, plan, candidate dataset, validation set or earlier decision was changed. Batch 1's and
    batch 2's records and decisions are byte-identical (§2);
  - `build-pairs`, `screen-questions`, `build-candidates` and `plan` were not run in `bge_base_ft_v3`. They ran only in a
    scratch copy of the preparation, which was then discarded. No final candidate dataset was built;
  - the validator is unchanged;
  - nothing was trained, committed or pushed.
- **Protected files:** 347 files under `data/`, `configs/`, `src/` and `scripts/` were checksummed before and after
  the audit. They are unchanged; the decisions file is the only file added.
- **Earlier documents:** [the batch 2 audit](train_v3_p1_batch2_generation_audit.md),
  [the batch 1 audit](train_v3_p1_batch1_generation_audit.md), [the readiness audit](final_training_readiness_audit.md)
  and [the full-training plan](training_full_v1.md), §9.

## 1. Verdict

**The batch passes the approved quality gate on all four criteria.**

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | At least 0.8 included pairs per planned passage | 108 included for 100 planned passages: **1.08** | **Pass** |
| 2 | At least 75% AI-review acceptance | 108 of 126 built pairs: **85.7%** | **Pass** |
| 3 | Zero accepted leakage violations | **0**: no included pair fails the evidence, cross-split, lexical, semantic or duplicate checks (§8, §9) | **Pass** |
| 4 | Complete provenance for every included pair | **108 of 108** carry the full provenance block, and the fail-closed provenance check passes (§11) | **Pass** |

The further checks also hold:
- **Reasoning answers:** none in batch 3. The detector, a looser variant and the review of every answer found no answer
  built from reasoning; batch 2's pair 032 remains one of only 2 such proposals in 597 across three batches (§7).
- **Semantic screen:** no question is flagged at 0.90, and no batch 3 question falls in the 0.80–0.90 review band
  (highest 0.7922).
- **Existing data:** `train_candidates_v2` rebuilds byte-identical, and `validation_pairs_v4` loads apart from all
  390 trial training pairs.
- **One generator:** all 300 records use one model, prompt, validator and parameter set.

**Lower acceptance than batches 1–2 (85.7% against 90.2% and 92.7%), for an identified reason.** Batch 3 drew 12 of its
16 JPMorgan passages from segment-results sections (4 of 14 in each earlier batch). Four of their 15 pairs attribute a
segment's figure to the wrong business, mostly because the segment's heading lies before the window (§6). The review
catches these errors, and the batch is 10.7 points above the gate.

**GO for batch 4**, under the conditions in §14:
1. **Same generator:** the same model, prompt, validator, parameters and plan options, with the validator unchanged.
2. **Option A:** no real `build-pairs` until all six batches are generated.
3. **Company balance:** no Apple or Amazon group is left (§13).
   - Under the frozen options, batch 4 holds JPMorgan, Pfizer, Tesla, Walmart and ExxonMobil at 20 passages each, 2–3
     per fiscal year.
   - The 25% cap holds.
4. **Budget:** batch 4 needs about 147,700 of the remaining 447,066 tokens. Batches 4–6 together leave about a 1%
   margin (§13).

**Unchanged:** the three-seed A0 versus A1 experiment stays **NO-GO**. Blockers B1 (a selection-aware decision rule),
B2 (the planned data size) and B3 (the GPU rehearsal) still stand.

## 2. Records and provenance

| Check | Result |
|---|---|
| Plan | `train.train_v3_p1_batch3.json`, SHA-256 `fdddb936…f2a5`, unchanged; 100 train passages |
| Records file | `generation/train.records.jsonl`, 300 lines, SHA-256 `3a8952fe…217c` |
| Batches 1 and 2 | Lines 1–100 hash to `91116337…56d1` and lines 1–200 to `79853cdb…8587`, as the batch 1 and 2 audits recorded them. The runs only appended. |
| Batch 3's records | Lines 201–300: one record per plan position, in plan order; no other plan's records; 300 distinct passages in the file |
| Outcome | **100 `done`, 0 `failed`**. Every record took 1 attempt, with no error. |
| Identity | All 300 records: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…`, record format 3 |
| Run | `train.run.json` is unchanged (`72c9e7bf…`). It binds prompt SHA-256 `f42bdc54…`, the same parameters and preparation `21c2293a…`. There is no continuation file. |
| Replies | 100 distinct response hashes (299 across the file; batch 1's failed record has none) |
| Proposals | 199: position 25's reply held one question instead of two |
| Validator re-run | `revalidate --dry-run`: 299 up to date, 0 changed, 1 failed record (batch 1's position 87) |
| Tokens | **146,256** (120,687 prompt, 25,569 completion), equal in the records and the four run summaries. Batches 1–3 total 441,218. |
| Completion tokens per reply | 129–500, median 242. No reply came near the 768 limit. |
| Against the dry-run estimate | About 146,800 tokens and about 106 HTTP requests were estimated; the run used 146,256 (0.3% less) and 106 |

**The runs:** four invocations, from 2026-11-06 17:22 to 21:25 UTC (record timestamps), from
`logs/train_v3_p1_batch3.log`. The log holds no credential.

| Run | Request budget | HTTP requests | HTTP 429 | HTTP 503 | Done | Tokens | Ended by |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 100 | 91 | 2 | 2 | 87 | 126,862 | a rate-limit wait of 632 s, after a 60 s wait was taken |
| 2 | 9 | 4 | 1 | 0 | 3 | 4,808 | a wait of 463 s |
| 3 | 5 | 5 | 0 | 1 | 4 | 5,308 | its request budget |
| 4 | 10 | 6 | 0 | 0 | 6 | 9,278 | the end of the plan |
| **All** | | **106** | **3** | **3** | **100** | **146,256** | |

**How Groq's limits were handled:**
- **The generator worked as designed.** 106 HTTP requests gave 100 usable replies, 3 HTTP 429 responses and 3 HTTP 503
  responses.
  - It waited out the one 429 wait of at most 180 s and stopped cleanly at the two longer ones, without recording the
    passage.
  - **New in batch 3:** the 503s were Groq's "over capacity" errors, the first in the kept logs. Each was retried
    after 2 s and the retry succeeded.
  - Each rerun skipped the completed passages, and the generator spent 4,545 s in pacing and rate-limit waits.
- **No passage was lost or retried** because of a rate limit or a server error.
- **What the headers show:** the last response of each run showed 1,000 requests per day (914 or more remaining) and
  8,000 tokens per minute. As in batch 2, the 429 waits of 7–11 minutes are longer than any per-minute window.

**Against batches 1–2 and the plan's per-batch estimate** ([training_full_v1.md](training_full_v1.md), §9):

| | Estimate | Batch 1 | Batch 2 | Batch 3 |
|---|---|---:|---:|---:|
| Tokens | about 143,000 | 148,284 | 146,678 | 146,256 |
| Questions passing the validator | about 155 | 151 | 159 | 154 |
| Pairs built | about 140 | 112 | 124 | 126 |
| AI review accepts | about 115 | 101 | 115 | 108 |
| Candidates | about 100 (range about 80–115) | 101 | 115 | 108 |

## 3. The funnel

199 proposed → **154** pass the generator's validator (77.4%) → **126** pairs built → **108** accepted by the review →
**108** included in the trial candidate set.

| | Plan | Done | Proposed | Pass validator | Pairs built | Accepted (included) |
|---|---:|---:|---:|---:|---:|---:|
| Batch 1 | 100 | 99 | 198 | 151 (76%) | 112 | 101 |
| Batch 2 | 100 | 100 | 200 | 159 (80%) | 124 | 115 |
| Batch 3 | 100 | 100 | 199 | 154 (77%) | 126 | 108 |
| **All** | **300** | **299** | **597** | **464 (78%)** | **362** | **324** |

- **Exclusions at each step:**
  - `build-pairs` excluded 28 questions, every one for evidence leakage: `evaluation_evidence` 27 and
    `cross_split_evidence` 1 (§5). They are not training data, so they get no review decision.
  - Nothing was excluded for a validator re-check, a duplicate or evaluation overlap.
  - The trial candidate build excluded nothing beyond the 18 pairs the review did not accept.
- **Per passage:** 23 passages contribute no included pair, 46 contribute one and 31 contribute two (batch 2: 20, 45 and
  35; batch 1: 30, 39 and 31).
- **Batches 1–2 in this replay:** the scratch `build-pairs` ran over all 300 records.
  - Batches 1 and 2 keep the same 112 and 124 pairs and the same 39 and 35 exclusions.
  - Every pair reappears with the same passage and question hash, but 232 of the 236 pair IDs are renumbered, as
    expected (§13).
  - The trace of every candidate through `build_pairs`' own functions matches the written pairs, ID for ID.

## 4. Generator acceptance and rejection (validator `sec-qcheck-v3`)

| Reason | Count | Verdict |
|---|---:|---|
| `missing_company` | 19 | All correct: the question does not name the company. One names only JPMorgan's "Treasury and CIO unit". |
| `copied_from_passage` | 10 | 3 copy a run of prose (9–13 words). 5 are borderline: runs of 8–12 words that are mostly a court's or pension scheme's name, dates, or an 8-word fragment; they are correct by the strict rule. 2 are false (positions 41 and 70): row labels of tables whose `[TABLE]` marker lies before the window, so the label exemption does not apply. |
| `evidence_length` | 10 | All correct by the rule: quotes of 301–470 characters (limit 300). |
| `answer_not_supported` | 2 | Both false (positions 67 and 86): the answers are right, but their dates or years are not in the quotes. |
| `malformed` | 2 | Correct: the reply gave no question for the entry (positions 51 and 65). |
| `refers_to_context` | 1 | Correct: "According to the table provided" (position 44). |
| `evidence_not_in_visible_text` | 1 | Correct by the rule: the quote splices the Total row onto the January row out of order (position 76); the answer itself is right. |

**Totals:**
- 36 correct;
- 5 correct by the strict rule;
- **4 false rejections**, positions 41, 67, 70 and 86: 2.0% of proposals. Batch 2 had 4 (2.0%) and batch 1 had 3
  (1.5%).

The four false rejections repeat two known patterns:
- a row label outside a `[TABLE]` block (batch 2's position 5, batch 1's 60);
- years or dates missing from the quote (batch 2's positions 27 and 95, batch 1's 49).

There is no new validator weakness. The two reasons not seen in batch 2, `malformed` and `refers_to_context`, are the
validator working as designed. Fixing the weaknesses mid-P1 would change the generator's identity, so the validator stays
as it is.

| Pos | Source | Reason | Question (verbatim) | Verdict |
|---:|---|---|---|---|
| 3 | JPM FY2022, narrative, Item 1 | `copied_from_passage` | When are JPMorgan's banking entities in the EU and U.K. required to comply with the Basel III final rules? | Correct: copies a 9-word run of the sentence. |
| 3 | JPM FY2022, narrative, Item 1 | `missing_company` | What is the primary objective of the U.K. government's Edinburgh Reforms package announced in December 2022? | Correct: the question does not name the company. |
| 4 | PFE FY2019, narrative, Item 1A | `missing_company` | What specific ruling by the U.S. Circuit Court of Appeals for the Fifth Circuit in December 2019 created uncertainty regarding the individual mandate of the ACA? | Correct: the question does not name the company. |
| 6 | WMT FY2021, narrative, Item 1A | `copied_from_passage` | What specific federal multi-district litigation case is Walmart currently defending in the U.S. District Court for the Northern District of Ohio regarding opioid distribution? | Correct by the rule, borderline: a 12-word run, mostly the court's name. |
| 12 | TSLA FY2023, table, Item 8 | `evidence_length` | How much of Tesla's U.S. government securities were classified as short-term investments as of December 31, 2023? | Correct by the rule: the quote has 301 characters (limit 300). |
| 14 | XOM FY2023, narrative, Business Results | `missing_company` | How much higher were the average production volumes in the Permian in 2023 compared to the previous year? | Correct: the question does not name the company. |
| 15 | AAPL FY2024, table, Item 8 | `missing_company` | According to the cash flow summary, what were the total proceeds from or repayments of commercial paper, net, for fiscal 2023 in millions? | Correct: the question does not name the company. |
| 16 | AMZN FY2019, table, Item 8 | `missing_company` | How many millions of common stock shares were issued at the beginning of the period shown in the consolidated statements of stockholders' equity for January 1, 2017? | Correct: the question does not name the company. |
| 18 | PFE FY2021, narrative, Item 7 | `missing_company` | What was the result of the Phase 3 JAVELIN Lung 100 trial for avelumab in the high PD-L1+ population during FY2021? | Correct: the question does not name the company. |
| 18 | PFE FY2021, narrative, Item 7 | `missing_company` | Why was enrollment stopped for the C4591015 study evaluating BNT162b2 in pregnant women in the fourth quarter of 2021? | Correct: the question does not name the company. |
| 26 | TSLA FY2024, table, Item 8 | `missing_company` | How many shares were reserved and available for issuance under the 2019 Equity Incentive Plan as of December 31, 2024? | Correct: the question does not name the company. |
| 28 | XOM FY2025, narrative, Note 6. Financial Instruments And Deriva | `evidence_length` | What was the net carrying value of Exxon Mobil's derivative liabilities as of December 31, 2024? | Correct by the rule: the quote has 470 characters (limit 300). |
| 28 | XOM FY2025, narrative, Note 6. Financial Instruments And Deriva | `evidence_length` | What was the difference between the carrying value and fair value of Exxon Mobil's long-term debt at December 31, 2024? | Correct by the rule: the quote has 402 characters (limit 300). |
| 29 | AAPL FY2022, table, Item 5 | `missing_company` | How did the S&P 500 Index perform relative to the initial $100 investment benchmark as of September 2022? | Correct: the question does not name the company. |
| 31 | JPM FY2021, table, Business Segment Results | `missing_company` | How much was the addition to the allowance for credit losses in the prior year compared to 2021? | Correct: the question does not name the company. |
| 32 | PFE FY2025, narrative, Item 1A | `missing_company` | What change in the U.S. uninsured population does the Congressional Budget Office estimate the OBBBA will cause? | Correct: the question does not name the company. |
| 32 | PFE FY2025, narrative, Item 1A | `missing_company` | When are most provisions of the EU's landmark reform on innovative medicines expected to apply? | Correct: the question does not name the company. |
| 37 | JPM FY2019, narrative, Business Segment Results | `evidence_length` | What specific factors drove the increase in noninterest revenue for JPMorgan Chase's Treasury and CIO segment in fiscal 2019? | Correct by the rule: the quote has 379 characters (limit 300). |
| 41 | XOM FY2020, narrative, Item 2 | `copied_from_passage` | What was Exxon Mobil's total crude oil and natural gas liquids production in 2020? | False: the 8-word run is a row label ('Total crude oil and natural gas liquids production'), but the table's [TABLE] marker lies before the visible window, so the label exemption does not apply; the answer is right (1,663 and 276 kbd). |
| 44 | PFE FY2020, table, Item 7 | `copied_from_passage` | What was the status of accruals for future benefits under the PCPP and the defined benefit section of the Pfizer Group Pension Scheme effective January 1, 2018? | Correct by the rule, borderline: an 11-word run, mostly the pension scheme's name. |
| 44 | PFE FY2020, table, Item 7 | `refers_to_context` | According to the table provided for Pfizer's U.S. plans, what data points are reported at the end of each year? | Correct: 'the table provided' refers to the passage rather than the filing. |
| 47 | XOM FY2019, narrative, 16. Litigation And Other Contingencies | `evidence_length` | What was Exxon Mobil's assessment regarding the potential financial impact of the Nigeria litigation on its consolidated results for fiscal 2019? | Correct by the rule: the quote has 322 characters (limit 300). |
| 47 | XOM FY2019, narrative, 16. Litigation And Other Contingencies | `missing_company` | In what year did the Contractors appeal the dismissal of their Nigerian federal high court lawsuit to the Nigerian Court of Appeal? | Correct: the question does not name the company. |
| 49 | JPM FY2022, narrative, Business Segment Results | `evidence_length` | Which specific international consumer growth initiatives does JPMorgan Chase's Other Corporate segment include, and when did the C6 Bank transaction close? | Correct by the rule: the quote has 306 characters (limit 300). |
| 50 | PFE FY2019, narrative, Item 1A | `missing_company` | What specific proposals regarding federal Medicaid payments and Medicare Part D rebates were included in the Congressional Budget Office release for December 2018? | Correct: the question does not name the company. |
| 50 | PFE FY2019, narrative, Item 1A | `evidence_length` | According to Pfizer's risk factors for fiscal 2019, what are some potential consequences of failing to comply with applicable regulatory requirements? | Correct by the rule: the quote has 461 characters (limit 300). |
| 51 | TSLA FY2021, table, Item 8 | `malformed` | (none) | Correct: the reply gave no question for this entry. |
| 53 | XOM FY2024, table, Item 2 | `copied_from_passage` | What were ExxonMobil's contractually committed delivery volumes for oil and natural gas for the period from 2025 through 2027 as reported in fiscal 2024? | Correct by the rule, borderline: a 9-word run, mostly the date range. |
| 54 | AAPL FY2023, table, Item 8 | `copied_from_passage` | What was the status of the authority to grant new awards under the 2014 Plan for Apple Inc. in the third quarter of 2022? | Correct: copies a 10-word run of the sentence. |
| 55 | JPM FY2025, table, Business Segment & Corporate Results | `missing_company` | How much net income did the Treasury and CIO unit report for the year ended December 31, 2023? | Correct: the question does not name the company. |
| 57 | TSLA FY2023, narrative, Item 9B | `missing_company` | What was the expiration date for the Rule 10b5-1 trading arrangement adopted by Andrew Baglino in November 2023? | Correct: the question does not name the company. |
| 58 | WMT FY2022, narrative, Item 1A | `evidence_length` | How might the completion of the pending acquisitions by Walmart's fintech joint venture affect the company's relationship with financial regulatory authorities? | Correct by the rule: the quote has 340 characters (limit 300). |
| 61 | JPM FY2024, narrative, Business Segment & Corporate Results | `copied_from_passage` | What is the general basis for allocating costs to business segments when they use services provided by Corporate support units or other segments, according to JPMorgan Chase & Co's fiscal 2024 10-K? | Correct by the rule, borderline: an 8-word run of the sentence. |
| 65 | XOM FY2022, narrative, 1. Summary Of Accounting Policies | `malformed` | (none) | Correct: the reply gave no question for this entry. |
| 67 | JPM FY2023, table, Business Segment Results | `answer_not_supported` | What was JPMorgan Chase & Co's total net revenue for the Corporate & Investment Bank segment in fiscal 2023, and how much of that was derived from noninterest revenue? | False: the answer is right ($48,807 million and $40,315 million); its date is not in the quote. |
| 67 | JPM FY2023, table, Business Segment Results | `evidence_length` | Which specific products and services does the Banking division of JPMorgan Chase & Co's Corporate & Investment Bank offer to clients regarding capital markets and payments? | Correct by the rule: the quote has 448 characters (limit 300). |
| 70 | WMT FY2020, narrative, Item 7 | `copied_from_passage` | What was the operating income for the Walmart U.S. segment in fiscal 2020 and what was the corresponding operating income as a percentage of net sales? | False: the 8-word run is a row label ('Operating income as a percentage of net sales'), but the table's [TABLE] marker lies before the visible window, so the label exemption does not apply. |
| 74 | PFE FY2023, table, Item 1 | `evidence_length` | According to the commercial operations table for fiscal 2023, what is the specific description provided for the Pfizer Oncology Division? | Correct by the rule: the quote has 431 characters (limit 300). |
| 76 | WMT FY2024, table, Item 5 | `evidence_not_in_visible_text` | How many shares did Walmart repurchase in total during the fiscal 2024 fourth quarter, and what was the approximate dollar value of shares that could still be bought under the plan at the end of January 2024? | Correct by the rule: the quote joins the Total row to the January row out of order; the answer itself is right (29,465,379 shares; $16.5 billion). |
| 78 | AAPL FY2025, table, Item 8 | `copied_from_passage` | What were the weighted-average interest rates on Apple's commercial paper as of September 27, 2025, and September 28, 2024? | Correct by the rule, borderline: a 9-word run, mostly the two dates. |
| 83 | XOM FY2021, table, Business Results | `missing_company` | According to the business results, how much did identified items impact earnings in 2020, and what was the primary driver of that impact? | Correct: the question does not name the company. |
| 85 | JPM FY2020, narrative, Business Segment Results | `copied_from_passage` | How does JPMorgan Chase & Co allocate the cost of unsecured long-term debt and preferred stock dividends to its reportable business segments in fiscal 2020? | Correct: copies a 13-word run of the sentence. |
| 86 | PFE FY2020, table, Item 7 | `answer_not_supported` | What were Pfizer's total worldwide revenues for the year ended December 31, 2020, and what was the percentage change compared to 2019? | False: the answer is right ($41,908 million, up 2%); its years are not in the quote. |
| 86 | PFE FY2020, table, Item 7 | `missing_company` | According to the analysis of the change in worldwide revenues, what was the dollar amount of the impact from the completion of the Consumer Healthcare JV transaction for the year ended December 31, 2020? | Correct: the question does not name the company. |
| 99 | TSLA FY2022, table, Item 8 | `missing_company` | What was the balance of additional paid-in capital at January 1, 2021, after the adoption of the accounting standard update? | Correct: the question does not name the company. |

## 5. Pair building: evidence-leakage exclusions (28)

`build-pairs` excludes a question whose evidence, quote or labelled figure is held exactly by:
- a blocked passage: `evaluation_evidence`;
- the visible text of a passage in another split: `cross_split_evidence`.

These rules fail closed by design, and every exclusion below follows them.

| Where the evidence is | Batch 3 | Batch 2 | Batch 1 |
|---|---:|---:|---:|
| An evaluation-pool or audit passage | 4 | 6 | 6 |
| An overlap-group copy of an evaluation passage | 5 | 14 | 11 |
| Only passages blocked for reusing evaluation text | 18 | 7 | 16 |
| The visible text of a validation passage | 1 | 8 | 6 |
| **All** | **28** (18.2% of 154) | **35** (22.0% of 159) | **39** (25.8% of 151) |

- **Apple:** the rule took 9 of Apple's 28 questions. They ask for net sales, diluted shares, comprehensive income,
  PP&E, tax rates, a hedge notional and lease payments, figures that recur in blocked passages.
- **Tesla and Walmart:** the rule took 6 of 27 and 5 of 28. The other companies lost 1–3.
- **The conservative evidence-reuse exclusions** rose from 7 to 18 (batch 1: 16). The rule is unchanged (decision E5).
- **Cross-split:** the one validation-passage holder keeps the training set and `validation_pairs_v4` apart. Five
  evaluation exclusions also have a validation copy.

| Pos | Source | Question (verbatim) | Answer (verbatim) | Rule | Where the evidence is |
|---:|---|---|---|---|---|
| 2 | AMZN FY2019, table, Item 8 | How much did Amazon have outstanding under its Credit Agreement as of December 31, 2019? | No borrowings | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 8 | AAPL FY2023, table, Item 8 | What was Apple's effective tax rate for fiscal 2023? | 14.7% | `evaluation_evidence` | only passages blocked for reusing evaluation text (3 blocked) |
| 13 | WMT FY2022, table, Item 8 | How did Walmart measure the fair value of its derivatives, and what level of inputs was used? | Using the income approach and Level 2 inputs, including relevant interest rate and foreign currency forward curves | `evaluation_evidence` | only passages blocked for reusing evaluation text (3 blocked) |
| 16 | AMZN FY2019, table, Item 8 | What was Amazon's total stockholders' equity in millions of dollars as of December 31, 2019? | $62,060 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 19 | TSLA FY2022, narrative, Item 8 | How much net cash did Tesla provide from operating activities in fiscal 2021? | 11,497 | `evaluation_evidence` | an evaluation-pool or audit passage (3 blocked) |
| 34 | WMT FY2024, table, Item 7 | How much depreciation and amortization did Walmart report for the fiscal year ended January 31, 2023? | 10,945 | `evaluation_evidence` | only passages blocked for reusing evaluation text (3 blocked) |
| 36 | AAPL FY2025, table, Item 8 | What was the notional amount of Apple's foreign exchange contracts designated as accounting hedges in fiscal 2025? | $62,647 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 39 | TSLA FY2025, table, Item 8 | How much of Tesla's commercial paper was classified as short-term investments at the end of fiscal 2025? | $5,617 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 60 | AAPL FY2021, table, Item 8 | What was the net value of Apple's property, plant, and equipment as of September 25, 2021? | $39,440 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 61 | JPM FY2024, narrative, Business Segment & Corporate Results | How many reportable business segments did JPMorgan Chase & Co have for fiscal 2024 following the reorganization in the second quarter of 2024? | Three | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 63 | TSLA FY2022, table, Item 8 | What was the comprehensive income attributable to common stockholders for Tesla, Inc. for the year ended December 31, 2022? | $12,141 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 64 | WMT FY2023, table, Item 8 | What was Walmart's total comprehensive income attributable to the company for the fiscal year ended January 31, 2023? | $10,028 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked; also 1 validation) |
| 64 | WMT FY2023, table, Item 8 | How much did Walmart report in cash and cash equivalents as of January 31, 2023? | $8,625 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked; also 1 validation) |
| 65 | XOM FY2022, narrative, 1. Summary Of Accounting Policies | What factors does Exxon Mobil Corp. consider when determining if the carrying value of an equity method investment may not be recoverable? | Examples include a history of operating losses, negative earnings and cash flow outlook, significant downward revisions to oil and gas reserves, and the financial condition and prospects for the investee's business segment or geographic region | `evaluation_evidence` | an overlap-group copy of an evaluation passage (5 blocked) |
| 66 | AAPL FY2025, table, Item 8 | How much did Apple Inc. pay in fixed cash for operating leases during fiscal 2025? | $2.1 billion | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 72 | AAPL FY2019, table, Item 6 | What were Apple's total net sales in millions for fiscal 2019? | $260,174 million | `evaluation_evidence` | an evaluation-pool or audit passage (15 blocked) |
| 72 | AAPL FY2019, table, Item 6 | How many diluted shares, in thousands, did Apple use to compute earnings per share for the year ended September 28, 2019? | 4,648,913 thousand shares | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 74 | PFE FY2023, table, Item 1 | Which four in-line oncology medicines did Pfizer add to its portfolio through the acquisition of Seagen completed in December 2023? | Padcev, Adcetris, Tukysa and Tivdak | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 75 | TSLA FY2020, table, Item 8 | Which tax years was Tesla, Inc. under examination by the IRS for during the fiscal 2020 period? | 2015 to 2018 | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 80 | PFE FY2025, narrative, Item 1A | How does Pfizer describe the potential sufficiency of its cyber liability insurance in the event of a system interruption or breach? | The insurance may not be sufficient to cover the financial, legal, business, or reputational losses that may result from an interruption or breach of systems. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (5 blocked) |
| 90 | AAPL FY2025, table, Item 8 | What was Apple's effective tax rate for fiscal 2024? | 24.1% | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 93 | TSLA FY2021, narrative, Item 1 | How does Tesla's used vehicle business support new vehicle sales in fiscal 2021? | It supports new vehicle sales by integrating the trade-in of a customer's existing Tesla or non-Tesla vehicle with the sale of a new or used Tesla vehicle. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 96 | AAPL FY2019, table, Item 8 | What was Apple's total comprehensive income for the fiscal year ended September 28, 2019, in millions? | $58,037 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 96 | AAPL FY2019, table, Item 8 | How did Apple's total other comprehensive loss change in the year ended September 29, 2018 compared to the year ended September 28, 2019, in millions? | The total other comprehensive loss was $(3,026) million in the year ended September 29, 2018, compared to $2,781 million in the year ended September 28, 2019. | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 97 | JPM FY2025, table, Consolidated Results Of Operations | What was JPMorgan Chase's total net revenue for the year ended December 31, 2025? | $182,447 million | `evaluation_evidence` | an evaluation-pool or audit passage (2 blocked) |
| 97 | JPM FY2025, table, Consolidated Results Of Operations | How much operating lease income was included in JPMorgan Chase's other income for fiscal 2024? | $2.8 billion | `evaluation_evidence` | an evaluation-pool or audit passage (4 blocked; also 1 validation) |
| 99 | TSLA FY2022, table, Item 8 | How much impairment loss did Tesla record on digital assets during the year ended December 31, 2022? | $204 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (4 blocked; also 1 validation) |
| 100 | WMT FY2022, table, Item 8 | Did Walmart's long-term debt issuances in fiscal 2022 contain financial covenants restricting dividend payments or stock repurchases? | No, the issuances do not contain any financial covenants which restrict the Company's ability to pay dividends or repurchase company stock. | `evaluation_evidence` | only passages blocked for reusing evaluation text (3 blocked; also 1 validation) |

## 6. AI review of the 126 pairs

**How each pair was judged:** as in batches 1 and 2, each pair was read against its positive passage's full visible
text.
- **Accept:** correct company, period, figure and unit; supported by the quote; a useful financial retrieval
  query. Minor issues are noted but do not exclude.
- **Revise:** a specific error that a reworded question would fix.
- **Reject:** no financial content, or no real answer.

**Precedents, applied here:**
- a wrong unit or measure fails, and a part given as the whole is revised (validation audits, batch 1's pair 070);
- a question naming no year is revised when the fact changes between filings, and accepted with a note when it does not
  (batch 1's pair 075);
- an executive's age is rejected (batch 1's pair 098), and revised when asked together with a governance fact (batch 2's
  pair 225);
- boilerplate with no financial content is rejected (batch 1's pair 106).

**Corpus look-ups were used for:**
- pairs 106 and 123: JPMorgan's Corporate segment heading and table, in the preceding chunks;
- pair 118: JPMorgan's Firm-wide deposits at December 31, 2021;
- pair 190: Pfizer's definition of TSRUs;
- pair 229: Tesla's unrecognized compensation for non-performance awards, in a later chunk;
- pair 236: Tesla's December 31, 2019 share count in its FY2019–FY2021 filings.

**Result:** 108 accept (85.7%), 14 revise, 4 reject.

| Aspect (126 pairs) | Pass | Minor | Fail | n/a |
|---|---:|---:|---:|---:|
| Factual accuracy | 106 | 7 | 13 (010, 018, 106, 112, 118, 123, 181, 190, 215, 229, 236, 325, 361) | – |
| Financial relevance | 110 | 11 | 5 (060, 061, 152, 169, 174) | – |
| Company | 126 | – | – | – |
| Period | 115 | 8 (names no year) | 3 (215, 236, 325) | – |
| Figures (answer figures in the quote) | 85 | – | – | 41 |
| Units | 71 | 14 (no unit) | 0 | 41 |

**Issues noted on accepted pairs** (one pair can carry two):
- **Boilerplate:** 22, mostly risk factors and governance or methodology statements repeated yearly.
- **Weak financial relevance:** 11. These are three insider trading plan facts, four human-capital statements (values,
  diversity, performance management, leadership), an officer appointment, two fund-rating methodology footnotes and a
  planned pension contribution named without its amount.
- **Inaccurate wording:** 7. These include "segment" for JPMorgan's Markets and Merchant Services businesses, and a
  repayment row without its maturity qualifier.
- **No issue:** 72.

Batch 2: 21, 9, 7 and 81. Batch 1: 19, 10, 5 and 67.

**Not accepted (18):**

| Pair | Pos | Decision | Reason |
|---|---:|---|---|
| 010 | 1 | Revise | Wrong measure named: it asks how many shares of record; 22,797 is the number of shareholders of record. |
| 018 | 22 | Revise | The same error: 23,502 is the number of shareholders of record. |
| 060 | 9 | Revise | Asks Amazon's CEO's age with his identity; the age has no financial content. |
| 106 | 37 | Revise | Wrong scope: the Corporate segment's net income ($1.1 billion against a $1.2 billion loss) is given as Treasury and CIO's. Corporate's heading is in the preceding chunk. |
| 112 | 43 | Revise | Wrong scope: loss days are reported for CIB Markets as a whole; 1 is 2019's figure for all of Markets, not Fixed Income Markets'. |
| 118 | 31 | Revise | Part as the whole: $1,148,110 million is Consumer & Community Banking's deposits; the Firm's were $2,462,303 million. |
| 123 | 49 | Revise | Wrong scope: the Corporate segment's noninterest expense ($1,034 million, down $768 million) is given as Other Corporate's. |
| 181 | 98 | Revise | Ambiguous measure: "total operating lease cost" fits neither row; $714 million is the operating lease cost line, and the total lease cost is $1,218 million. |
| 190 | 38 | Revise | The figure is right, but the question expands TSRU as "Time-Deferred Stock Unit". That is not in the passage, and it is wrong: Pfizer defines TSRUs as Total Shareholder Return Units (§7). |
| 215 | 87 | Revise | Names no year, and the figure changes between filings: $151 million of foreign deferred tax assets is FY2019's. |
| 229 | 5 | Revise | Part as the whole: $65 million is the 2018 CEO Performance Award's unrecognized expense; Tesla also had $3.43 billion for other awards. |
| 236 | 63 | Revise | A split-adjusted count without the filing: 2,716 million is the FY2022 filing's figure; the December 31, 2019 balance is 181 million in the FY2019 filing and 905 million in FY2020–FY2021. |
| 325 | 77 | Revise | Wrong column: 49 is the non-U.S. figure for 2020; 2019's lies past the window. |
| 361 | 71 | Revise | The answer does not answer the question: it asks for total capitalized costs, and "$9.1 billion less than Upstream PP&E" is a reconciling gap. The table's total is $419,340 million. |
| 061 | 9 | Reject | A board member's age. |
| 152 | 92 | Reject | The safe-harbour disclaimer that forward-looking statements are not updated: boilerplate with no financial content. |
| 169 | 62 | Reject | A table-header lookup: the expected cash flows lie past the window, so the answer is only two column headings. |
| 174 | 11 | Reject | An executive's age. |

**Patterns** (one pair can fall in two):
- **Scope:** 5 (106, 112, 118, 123, 229), against 1 in batch 2 (077).
  - Four are JPMorgan segment passages. Their heading lies before the window (Corporate, CCB), or their table mixes
    businesses (Markets).
  - Batch 3 drew 12 of its 16 JPMorgan passages from segment-results sections, against 4 of 14 in each earlier batch.
    Those 12 gave 15 pairs, 4 of them revised.
  - Tesla's 229 is the fifth: a single award's figure in the award's own section, given as the company total.
- **Measure:** 3. Pairs 010 and 018 say "shares of record" for shareholders of record. Batch 1's three correctly worded
  shareholder counts were accepted. Pair 181 is the lease-cost case.
- **Period:** 3 (215, 236, 325), as in batches 1 and 2.
- **The window's edge:** 5 errors come from text the generator could not see, against 2 in batch 2.
  - 106, 118 and 123 lack the heading that lies before the window.
  - 325 and 169 ask for figures past it.
- **The model's knowledge in a question:** 1 (190), §7.
- **An answer that does not answer:** 1 (361).
- **Low-value sources:** 5 non-accepts, against 3 in batch 2.
  - Amazon's board-of-directors table (position 9) gave one revise and one reject.
  - Pfizer's executive-officer table (11) gave one reject. Such tables recur in every batch: Walmart's in batch 1
    and ExxonMobil's in batch 2.
  - The forward-looking-statements preamble of Pfizer's FY2019 risk factors (92) and a table header at a window's end
    (62) gave one reject each.

Every pair, verbatim (pair numbers are those of the 300-record pair set; positions are batch 3's plan positions):

| Pair | Pos | Source | Type | Question (verbatim) | Answer (verbatim) | Decision | Note |
|---:|---:|---|---|---|---|---|---|
| 005 | 42 | AAPL FY2019, table, Item 8 | financial_table | What was the gross amount of property, plant and equipment for Apple Inc. as of September 28, 2019? | $95,957 million | Accept | Correct. |
| 006 | 42 | AAPL FY2019, table, Item 8 | other | What percentage of Apple's total trade receivables was accounted for by cellular network carriers as of September 28, 2019? | 51% | Accept | Correct. |
| 008 | 84 | AAPL FY2019, table, Item 8 | debt_liquidity | What was the total term debt of Apple Inc. as of September 28, 2019? | $101,679 million | Accept | Correct. |
| 009 | 84 | AAPL FY2019, table, Item 8 | financial_table | How many shares of Apple common stock were repurchased by the company during fiscal 2019? | 345,205 thousand shares | Accept | Correct; thousands of shares, as the header says. |
| 010 | 1 | AAPL FY2020, table, Item 5 | other | How many shares of record did Apple Inc. have as of October 16, 2020? | 22,797 shareholders of record | **Revise** | Wrong measure named: the question asks how many shares of record; 22,797 is the number of shareholders of record. |
| 011 | 1 | AAPL FY2020, table, Item 5 | financial_table | What was the average price paid per share for Apple's open market and privately negotiated share repurchases during the period from August 2, 2020 to August 29, 2020? | $115.99 per share | Accept | Correct. |
| 017 | 22 | AAPL FY2021, table, Item 5 | financial_table | What was the approximate dollar value of shares remaining to be purchased under Apple's repurchase program as of the end of fiscal 2021? | $60,851 million | Accept | Correct; the remaining authorization as of September 25, 2021, in millions. |
| 018 | 22 | AAPL FY2021, table, Item 5 | other | How many shares of record were held by Apple shareholders as of October 15, 2021? | 23,502 | **Revise** | Wrong measure named: the question asks how many shares of record were held; 23,502 is the number of shareholders of record (as pair 010). |
| 022 | 60 | AAPL FY2021, table, Item 8 | debt_liquidity | What percentage of Apple's total trade receivables was attributed to cellular network carriers at the end of fiscal 2021? | 42% | Accept | Correct. |
| 028 | 29 | AAPL FY2022, table, Item 5 | financial_table | What was the value of $100 invested in Apple Inc. common stock, including reinvestment of dividends, as of the market close on September 2022? | $411 | Accept | Correct; the cumulative-return table's September 2022 value. |
| 030 | 48 | AAPL FY2022, table, Item 8 | debt_liquidity | What was the total amount of Apple's term debt as of September 24, 2022? | $111,824 million | Accept | Correct. |
| 031 | 48 | AAPL FY2022, table, Item 8 | financial_table | How much commercial paper did Apple repay in fiscal 2021? | $6,567 million | Accept | Correct row; the question omits that these are repayments of paper with maturities over 90 days (shorter paper is reported net). |
| 040 | 8 | AAPL FY2023, table, Item 8 | financial_table | How much did Apple capitalize for research and development in its deferred tax assets as of September 30, 2023? | $6,294 million | Accept | Correct figure; it is the deferred tax asset for capitalized research and development, not the amount capitalized. |
| 041 | 54 | AAPL FY2023, table, Item 8 | financial_table | How many Restricted Stock Units did Apple Inc. grant during fiscal 2023, and what was the weighted-average grant date fair value per RSU for those awards? | 88,768 thousand RSUs with a weighted-average grant date fair value of $150.87 per RSU | Accept | Correct; fiscal 2023's grants (thousands of RSUs, as the header says). |
| 049 | 15 | AAPL FY2024, table, Item 8 | debt_liquidity | What was the weighted-average interest rate on Apple's commercial paper as of September 28, 2024? | 5.00% | Accept | Correct. |
| 054 | 36 | AAPL FY2025, table, Item 8 | other | As of September 27, 2025, what percentage of Apple's total trade receivables was attributable to its third-party cellular network carriers? | 34% | Accept | Correct. |
| 055 | 90 | AAPL FY2025, table, Item 8 | tax | How much was the impact of the State Aid Decision on Apple's provision for income taxes in fiscal 2024? | $10,246 million | Accept | Correct; the same fact as batch 2's pair 034, from the next year's filing; NegativeGuard keeps the two apart. |
| 056 | 66 | AAPL FY2025, table, Item 8 | financial_table | What was the total amount of right-of-use assets for Apple Inc. as of September 27, 2025, in millions? | $12,238 million | Accept | Correct. |
| 057 | 78 | AAPL FY2025, table, Item 8 | debt_liquidity | What was the net cash flow from commercial paper for Apple in fiscal 2025, specifically for maturities of 90 days or less? | $(5,820) million | Accept | Correct; the 90-days-or-less row for 2025. |
| 060 | 9 | AMZN FY2019, table, Item 1 | other | Who was the Chief Executive Officer and Chairman of the Board for Amazon.com in its fiscal 2019 Form 10-K, and what was their age at that time? | Jeffrey P. Bezos, Age 56 | **Revise** | The CEO's identity is a governance fact, but the question also asks his age, which has no financial content (batch 2 precedent, pair 225). |
| 061 | 9 | AMZN FY2019, table, Item 1 | other | What was the age of the board member listed as Former Chief Executive Officer of PepsiCo, Inc. in Amazon.com's fiscal 2019 board of directors listing? | 64 | **Reject** | A board member's age: no financial content (batch 1 precedent, pair 098). |
| 067 | 30 | AMZN FY2019, table, Item 8 | financial_table | What was the present value of Amazon's total lease liabilities as of December 31, 2019? | $52,814 million | Accept | Correct. |
| 068 | 30 | AMZN FY2019, table, Item 8 | other | For how much did Amazon acquire Whole Foods Market in 2017, net of cash acquired? | approximately $13.2 billion | Accept | Correct; the Whole Foods acquisition. |
| 069 | 2 | AMZN FY2019, table, Item 8 | financial_table | What was the total amount of debt principal and interest contractual commitments reported by Amazon for fiscal 2019? | $39,190 million | Accept | Correct; the commitments table's total as of December 31, 2019, in millions. |
| 070 | 23 | AMZN FY2019, table, Item 8 | tax | What was Amazon's provision for income taxes, net for the year ended December 31, 2019, and how did it compare to the year ended December 31, 2018? | The provision for income taxes, net was $2,374 million for the year ended December 31, 2019, compared to $1,197 million for the year ended December 31, 2018. | Accept | Correct; both years. |
| 071 | 23 | AMZN FY2019, table, Item 8 | financial_table | How much was Amazon's income before income taxes attributable to international operations for the year ended December 31, 2019? | Amazon's income before income taxes attributable to international operations was $691 million for the year ended December 31, 2019. | Accept | Correct. |
| 104 | 79 | JPM FY2019, table, Business Segment Results | accounting_policy | For fiscal 2019, how does JPMorgan Chase determine the 'primary share class' used in its performance analysis for non-U.S. domiciled funds? | It is the share class recommended by Morningstar as the best proxy for the portfolio, often the most retail version based on annual management charge, minimum investment, currency, and other factors. | Accept | Correct; a fund-rating methodology footnote repeated yearly. |
| 105 | 79 | JPM FY2019, table, Business Segment Results | other | What specific time periods are used for the star ratings assigned by Nomura in the analysis of JPMorgan Chase's funds? | Three-year risk-adjusted performance only. | Accept | Correct; a fund-rating methodology footnote repeated yearly. |
| 106 | 37 | JPM FY2019, narrative, Business Segment Results | profitability | What was the net income reported by JPMorgan Chase's Treasury and CIO segment in fiscal 2019, and how did it compare to the prior year? | Net income was $1.1 billion in 2019, compared with a net loss of $1.2 billion in 2018. | **Revise** | Wrong scope: the net income of $1.1 billion against a $1.2 billion loss is the Corporate segment's (its heading is in the preceding chunk); Treasury and CIO is part of Corporate, not a segment, and its overview starts after this paragraph. |
| 110 | 85 | JPM FY2020, narrative, Business Segment Results | other | What is the definition of payment processing-only clients in JPMorgan Chase & Co's Merchant Services segment for fiscal 2020? | Clients that only use payment services offered by Merchant Services and in general do not currently utilize other services offered by the Firm. | Accept | Correct; a client definition; Merchant Services is a business, not a segment. |
| 111 | 43 | JPM FY2020, table, Business Segment Results | revenue | What was the total net revenue for JPMorgan Chase's Total Markets segment in fiscal 2020? | $29,483 million | Accept | Correct; Markets is a business within the CIB segment, not a segment. |
| 112 | 43 | JPM FY2020, table, Business Segment Results | risk | How many loss days did JPMorgan Chase's Fixed Income Markets segment report for the year ended December 31, 2019? | 1 | **Revise** | Wrong scope: the loss-days row has one value per year for CIB Markets as a whole (its footnote: Fixed Income and Equity Markets together); 1 is 2019's figure for all of Markets, not Fixed Income Markets. |
| 118 | 31 | JPM FY2021, table, Business Segment Results | financial_table | What was JPMorgan Chase's total deposits balance at the end of fiscal 2021? | $1,148,110 million | **Revise** | Part given as the whole: the table is the Consumer & Community Banking segment's metrics; $1,148,110 million is CCB's deposits, while the Firm's were $2,462,303 million at December 31, 2021 (its FY2021 balance sheet). |
| 120 | 73 | JPM FY2021, narrative, Market Risk Management | risk | How does JPMorgan Chase & Co define a profit and loss drawdown in the context of its fiscal 2021 market risk management? | A decline in revenue from its year-to-date peak level | Accept | Correct; a market-risk definition repeated yearly. |
| 121 | 73 | JPM FY2021, narrative, Market Risk Management | accounting_policy | Which specific risk committee establishes the structural interest rate risk policy and limits for JPMorgan Chase & Co? | The CTC Risk Committee | Accept | Correct; names no period; interest-rate risk governance repeated yearly. |
| 123 | 49 | JPM FY2022, narrative, Business Segment Results | profitability | What was the noninterest expense and the year-over-year change for JPMorgan Chase's Other Corporate segment in fiscal 2022? | Noninterest expense was $1.0 billion, down $768 million. | **Revise** | Wrong scope: $1.0 billion, down $768 million, is the Corporate segment's noninterest expense ($1,034 million against $1,802 million in its table, the preceding chunk); Other Corporate is part of Corporate, not a segment. |
| 128 | 91 | JPM FY2022, narrative, Liquidity Risk Management | risk | What is the primary objective of JPMorgan Chase & Co's Liquidity Risk Management function in fiscal 2022? | To provide independent oversight of liquidity risk across the Firm. | Accept | Correct; liquidity-risk governance repeated yearly. |
| 129 | 91 | JPM FY2022, narrative, Liquidity Risk Management | debt_liquidity | Which departments are responsible for liquidity management at JPMorgan Chase & Co in fiscal 2022? | Treasury and CIO. | Accept | Correct; liquidity governance repeated yearly. |
| 134 | 24 | JPM FY2023, narrative, Business Segment Results | revenue | What was the year-over-year change in Payments revenue for JPMorgan Chase & Co in fiscal 2023, and what was the primary driver? | Payments revenue was up 22%, driven by deposit margin expansion on higher rates and fees. | Accept | Correct; the CIB's Payments revenue. |
| 135 | 24 | JPM FY2023, narrative, Business Segment Results | revenue | How did Lending revenue for JPMorgan Chase & Co change in fiscal 2023 compared to the prior year, and what specific factor contributed to this decline? | Lending revenue was down 27%, driven by $494 million of fair value losses on hedges of retained loans. | Accept | Correct. |
| 136 | 17 | JPM FY2024, table, Business Segment & Corporate Results | financial_table | What was the principal transactions revenue for JPMorgan Chase's Total Markets segment in fiscal 2024? | $24,129 million | Accept | Correct; Markets is a business within the CIB segment, not a segment. |
| 137 | 17 | JPM FY2024, table, Business Segment & Corporate Results | revenue | How much lending- and deposit-related fees did JPMorgan Chase's Equity Markets generate in the year ended December 31, 2023? | $40 million | Accept | Correct; the 2023 column. |
| 144 | 10 | JPM FY2025, table, Business Segment & Corporate Results | accounting_policy | What types of securities are excluded from the long-term debt rankings reported by JPMorgan Chase & Co for fiscal 2025? | Money market, short-term debt and U.S. municipal securities | Accept | Correct; a footnote defining the league-table rankings, repeated yearly. |
| 145 | 10 | JPM FY2025, table, Business Segment & Corporate Results | revenue | What components make up the Markets revenue for JPMorgan Chase & Co as described in the fiscal 2025 Form 10-K? | Principal transactions, fees, commissions and other income, as well as net interest income | Accept | Correct; the definition of Markets revenue, repeated yearly. |
| 146 | 55 | JPM FY2025, table, Business Segment & Corporate Results | revenue | What was JPMorgan Chase & Co's total net revenue for the Corporate segment in fiscal 2024? | $17,394 million | Accept | Correct; the 2024 column. |
| 151 | 92 | PFE FY2019, narrative, Item 1A | debt_liquidity | What type of future financial commitments does Pfizer expect to make regarding its U.S. qualified plans during fiscal 2020? | Voluntary contribution | Accept | Correct; a planned voluntary pension contribution, from the forward-looking-statements list (no amount). |
| 152 | 92 | PFE FY2019, narrative, Item 1A | other | What is Pfizer's stated obligation regarding the public updating of forward-looking statements under SEC rules? | No obligation to publicly update except as required by law or SEC rules | **Reject** | The safe-harbour disclaimer that forward-looking statements are not updated: boilerplate with no financial content (batch 1 precedent, pair 106). |
| 153 | 4 | PFE FY2019, narrative, Item 1A | risk | How does Pfizer expect the full invalidation of the ACA to impact its business, given that revenues from health insurance exchanges and Medicaid expansion are not material? | The impact is expected to be limited | Accept | Correct; names no year, but only the FY2019 filing makes this statement (after the Texas v. Azar ruling). |
| 168 | 62 | PFE FY2021, table, Item 8 | accounting_policy | In Pfizer's fiscal 2021 10-K, what types of securities are included in Level 2 investments for the company's benefit plans? | Corporate bonds, government and government agency obligations, and other fixed income securities valued using bid evaluation pricing models or quoted prices of securities with similar characteristics. | Accept | Correct for fixed income securities; other asset classes list further Level 2 holdings (commingled funds, insurance contracts); a valuation policy repeated yearly. |
| 169 | 62 | PFE FY2021, table, Item 8 | financial_table | According to the cash flow data in Pfizer's fiscal 2021 10-K, what are the specific categories of benefit plans listed for expected future cash flow information? | Pension Plans and Postretirement Plans | **Reject** | A table-header lookup: the expected cash flows lie past the window, so the answer is only the names of two column headings. |
| 172 | 56 | PFE FY2022, narrative, Item 1 | other | What are the four specific core values that Pfizer established in fiscal 2022 to define its culture and approach to achieving its purpose? | Courage, Excellence, Equity and Joy | Accept | Correct; corporate values from the human-capital section, not Pfizer's finances. |
| 173 | 56 | PFE FY2022, narrative, Item 1 | risk | According to Pfizer's fiscal 2022 disclosure, what specific actions does the company take to foster an inclusive environment regarding clinical trials? | Working to help ensure demographics of clinical trials correlate to those of the countries where trials are taking place | Accept | Correct; a diversity action on clinical-trial demographics. |
| 174 | 11 | PFE FY2022, table, Item 3 | financial_table | What is the age of Albert Bourla, Pfizer's Chief Executive Officer, as reported in the 2022 Form 10-K? | 61 | **Reject** | An executive's age: no financial content (batch 1 precedent, pair 098). |
| 175 | 11 | PFE FY2022, table, Item 3 | other | Who served as Pfizer's Chief Financial Officer and Executive Vice President starting in May 2022? | David M. Denton | Accept | Correct; an officer appointment, from the executive-officer table. |
| 181 | 98 | PFE FY2022, table, Item 8 | financial_table | What was Pfizer's total operating lease cost reported for the fiscal year ended December 31, 2022? | $714 million | **Revise** | Ambiguous measure: 'total operating lease cost' matches neither row exactly; $714 million is the operating lease cost line, and the total lease cost of these operating leases is $1,218 million. |
| 182 | 98 | PFE FY2022, table, Item 8 | accounting_policy | What was the weighted-average discount rate for Pfizer's operating leases as of December 31, 2022? | 3.0% | Accept | Correct. |
| 190 | 38 | PFE FY2023, table, Item 8 | financial_table | What was the total compensation cost recognized pre-tax for Pfizer's Time-Deferred Stock Unit (TSRU) awards during the year ended December 31, 2023? | $244 million | **Revise** | Correct figure ($244 million, the TSRU column for 2023), but the question expands TSRU as 'Time-Deferred Stock Unit', which is not in the passage and is wrong: Pfizer's filings define TSRUs as Total Shareholder Return Units. |
| 191 | 38 | PFE FY2023, table, Item 8 | accounting_policy | How are stock option awards amortized for Pfizer Inc. in its financial statements? | Amortized on a straight-line basis over the vesting term into Cost of sales, Selling, informational and administrative expenses, and/or Research and development expenses, as appropriate. | Accept | Correct; names no period; an amortization policy repeated yearly. |
| 192 | 25 | PFE FY2024, table, Item 1 | other | In fiscal 2024, which specific commercial divisions made up the Pfizer Biopharma reportable segment? | Pfizer Oncology Division, Pfizer U.S. Commercial Division, and Pfizer International Commercial Division | Accept | Correct; the 2024 commercial structure. |
| 194 | 68 | PFE FY2024, narrative, Item 1 | other | What is the name of the performance management approach used by Pfizer in fiscal 2024, and how often are goals set and feedback provided under this system? | The approach is called Performance and Leadership Insights, and it operates on six-month semesters where goals are set and feedback is received. | Accept | Correct; a performance-management practice from the human-capital section. |
| 195 | 68 | PFE FY2024, narrative, Item 1 | other | What specific leadership mindset did Pfizer introduce in 2024, and what are its core components as described in the company's business overview? | Pfizer introduced the Actionable Attitudes leadership mindset, which includes anchoring in trust, challenging convention, embracing uncertainty, engaging with compassion, prioritizing progress over perfection, and rising above. | Accept | Correct; a leadership programme from the human-capital section. |
| 199 | 80 | PFE FY2025, narrative, Item 1A | risk | What specific operational risks does Pfizer identify regarding the probabilistic nature of generative AI outputs? | Generative AI output is probabilistic, may not be reproducible or generate consistent results over time, and can generate outputs that are false, misleading, incomplete, or inconsistent. | Accept | Correct; names no year; an AI risk factor of the FY2025 filing. |
| 204 | 45 | TSLA FY2019, narrative, Item 1A | risk | What are the potential consequences for Tesla, Inc. in fiscal 2019 if the company fails to efficiently manage manufacturing ramps and cost-down efforts for Model 3 and Model Y? | Delays in establishing or sustaining Model 3 and Model Y ramps, or inability to meet cost and profitability targets. | Accept | Correct; a production-ramp risk factor. |
| 205 | 45 | TSLA FY2019, narrative, Item 1A | risk | Which specific future manufacturing facility in Germany did Tesla, Inc. identify in fiscal 2019 as a potential source of delays due to design, construction, or regulatory approval issues? | Gigafactory Berlin. | Accept | Correct; the planned Gigafactory Berlin. |
| 214 | 87 | TSLA FY2019, table, Item 8 | financial_table | What was Tesla's tax at the statutory federal rate for the year ended December 31, 2019, in millions? | $(139) million | Accept | Correct; the 2019 column (the first; later rows label it). |
| 215 | 87 | TSLA FY2019, table, Item 8 | tax | How much net deferred tax assets did Tesla report in foreign jurisdictions, and what was management's assessment regarding their realization? | Net $151 million, more-likely-than-not to be fully realized | **Revise** | Names no year, and the figure changes between filings: net $151 million of foreign deferred tax assets is the FY2019 filing's. |
| 216 | 33 | TSLA FY2020, narrative, Item 1A | accounting_policy | How does Tesla classify digital assets under applicable accounting rules in its fiscal 2020 10-K? | Indefinite-lived intangible assets | Accept | Correct; the accounting classification of digital assets. |
| 217 | 33 | TSLA FY2020, narrative, Item 1A | risk | Which currencies does Tesla identify as the primary sources of foreign currency risk for its fiscal 2020 operations? | Chinese yuan, euro, Canadian dollar and British pound | Accept | Correct; the main currencies named in the fiscal 2020 filing. |
| 221 | 75 | TSLA FY2020, table, Item 8 | financial_table | What was the balance of gross unrecognized tax benefits for Tesla, Inc. as of December 31, 2020? | $380 million | Accept | Correct. |
| 222 | 93 | TSLA FY2021, narrative, Item 1 | revenue | What is Tesla's primary driver of sales leads in fiscal 2021, and how does it affect marketing costs? | Media coverage and word of mouth are the primary drivers of sales leads, allowing sales at relatively low marketing costs without traditional advertising. | Accept | Correct; Tesla's sales-lead strategy, repeated yearly. |
| 225 | 51 | TSLA FY2021, table, Item 8 | financial_table | What was Tesla's total current assets in millions for the fiscal year ended December 31, 2021? | $27,100 million | Accept | Correct; a year-end balance asked 'for the fiscal year'. |
| 228 | 5 | TSLA FY2021, narrative, Item 8 | accounting_policy | How much aggregate catch-up expense did Tesla recognize in the year ended December 31, 2021, due to operational milestones becoming probable of achievement? | $571 million | Accept | Correct; the 2018 CEO Performance Award's catch-up expense. |
| 229 | 5 | TSLA FY2021, narrative, Item 8 | accounting_policy | What was the total unrecognized stock-based compensation expense remaining for Tesla as of December 31, 2021? | $65 million | **Revise** | Part given as the whole: $65 million is the unrecognized expense of the 2018 CEO Performance Award only; Tesla also had $3.43 billion for non-performance awards (a later chunk of the same filing). |
| 236 | 63 | TSLA FY2022, table, Item 8 | financial_table | How many shares of common stock were outstanding for Tesla, Inc. as of the balance date of December 31, 2019? | 2,716 million shares | **Revise** | Split-adjusted count without the filing: 2,716 million is the FY2022 filing's figure after the 2020 and 2022 splits; the December 31, 2019 balance is 181 million in the FY2019 filing and 905 million in FY2020-FY2021 (batch 1 precedent, pair 075). |
| 237 | 19 | TSLA FY2022, narrative, Item 8 | financial_table | What was the change in inventory reported in Tesla's cash flow from operating activities for the year ended December 31, 2022? | (6,465) | Accept | Correct; the 2022 column (the first; later rows label it); no unit (millions). |
| 241 | 12 | TSLA FY2023, table, Item 8 | financial_table | What was the fair value of Tesla's money market funds classified as cash and cash equivalents as of December 31, 2023? | $109 million | Accept | Correct; fair value and the cash-equivalent amount are both $109 million. |
| 244 | 57 | TSLA FY2023, narrative, Item 9B | other | How many shares of Tesla common stock was Robyn Denholm's Rule 10b5-1 trading arrangement, adopted in October 2023, designed to cover for potential sale? | 281,116 shares | Accept | Correct; a director's Rule 10b5-1 plan (up to 281,116 shares). |
| 248 | 26 | TSLA FY2024, table, Item 8 | financial_table | What was the net investment in sales-type leases reported by Tesla, Inc. as of December 31, 2024, and how much was classified as other non-current assets? | The net investment in sales-type leases was $440 million as of December 31, 2024, with $288 million reported as other non-current assets. | Accept | Correct; the total and its non-current part. |
| 249 | 69 | TSLA FY2024, narrative, Item 9B | other | What was the target net proceeds amount for the Rule 10b5-1 trading arrangement adopted by Tesla director Ira Ehrenpreis in December 2024? | $15 million | Accept | Correct; a director's Rule 10b5-1 plan (net proceeds of $15 million), from a 798-character Item 9B stub. |
| 250 | 69 | TSLA FY2024, narrative, Item 9B | other | What is the expiration date for the trading plan established by Tesla board member Ira Ehrenpreis on December 6, 2024? | June 3, 2025 | Accept | Correct; the same plan's expiration date. |
| 255 | 81 | TSLA FY2025, narrative, Item 7 | profitability | What was Tesla, Inc.'s net income attributable to common stockholders for fiscal 2025, and how did it compare to the prior year? | $3.79 billion, a decrease of $3.30 billion compared to the prior year | Accept | Correct; the figure and its change. |
| 256 | 81 | TSLA FY2025, narrative, Item 7 | mdna | How much did Tesla, Inc.'s capital expenditures amount to in 2025, and what was the change from 2024? | $8.53 billion in 2025, a decrease of $2.82 billion compared to 2024 | Accept | Correct; the figure and its change. |
| 259 | 39 | TSLA FY2025, table, Item 8 | financial_table | What was the fair value of Tesla's Level II investments as of December 31, 2025? | $27,546 million | Accept | Correct. |
| 260 | 27 | WMT FY2020, table, Item 6 | accounting_policy | What threshold of change in retail square feet causes a store conversion to be excluded from Walmart's comparable sales calculation? | More than five percent | Accept | Correct; names no year; the comparable-sales definition, from a 776-character footnote stub, repeated yearly (the same fact as batch 1's pair 085; NegativeGuard keeps the two apart). |
| 261 | 27 | WMT FY2020, table, Item 6 | accounting_policy | On what date did Walmart adopt ASU 2016-02, Leases (Topic 842), and was the prior fiscal year's financial data recast? | February 1, 2019; no, prior financial statements were not recast | Accept | Correct; the lease standard's adoption date and method. |
| 264 | 70 | WMT FY2020, narrative, Item 7 | mdna | What drove the 14 basis point decrease in gross profit rate for the Walmart U.S. segment in fiscal 2020? | The decrease was primarily the result of continued price investments, which were partially offset by better merchandise mix, including strength in private brands, and less pressure from transportation costs. | Accept | Correct; the fiscal 2020 drivers, nearly verbatim (the reasoning detector's R2 flag on 'which were' is a false positive). |
| 266 | 94 | WMT FY2021, narrative, Item 1A | risk | What specific operational and financial areas does Walmart identify as being subject to unpredictable and volatile impacts from the COVID-19 pandemic in fiscal 2021? | Business operations, demand for products and services, in-stock positions, costs of doing business, access to inventory, supply chain operations, ability to predict future performance, exposure to litigation, and financial performance. | Accept | Correct; the pandemic risk factor's list. |
| 267 | 94 | WMT FY2021, narrative, Item 1A | risk | How did the fluctuation in customer demand during the pandemic affect Walmart's inventory management and product delivery in fiscal 2021? | It resulted in higher out-of-stock positions in certain products and delays in delivering those products. | Accept | Correct. |
| 268 | 52 | WMT FY2021, narrative, Item 1A | risk | What specific liability consequence does Walmart face if payment card transaction data is compromised, according to its fiscal 2021 risk disclosures? | Liabilities to payment card issuers and others, which could be substantial | Accept | Correct; a payment-card risk statement. |
| 269 | 52 | WMT FY2021, narrative, Item 1A | risk | In fiscal 2021, what potential outcomes are cited for failing to comply with anti-money laundering, sanctions, and consumer financial regulations associated with Walmart's financial products and services? | Fines, sanctions, penalties, and harm to reputation | Accept | Correct; a compliance risk statement repeated yearly (pair 300 asks the same of fiscal 2025). |
| 271 | 6 | WMT FY2021, narrative, Item 1A | risk | In which federal district court did Walmart file a declaratory judgment action against the Department of Justice and the Drug Enforcement Administration on October 22, 2020? | U.S. District Court for the Eastern District of Texas | Accept | Correct; opioid litigation. |
| 279 | 58 | WMT FY2022, narrative, Item 1A | risk | What specific regulatory powers do the DEA, FDA, and state authorities hold over Walmart's operations regarding pharmaceuticals and controlled substances in the U.S.? | They have broad enforcement powers, including the ability to seize or recall products and impose significant criminal, civil and administrative sanctions for violations. | Accept | Correct; a regulatory risk statement. |
| 282 | 100 | WMT FY2022, table, Item 8 | financial_table | What was the total repayment of matured debt for Walmart in fiscal 2022, in millions? | $3,010 million | Accept | Correct. |
| 283 | 13 | WMT FY2022, table, Item 8 | financial_table | What was the fair value of Walmart's receive fixed-rate, pay variable-rate interest rate swaps designated as fair value hedges as of January 31, 2022? | $(47) million | Accept | Correct. |
| 286 | 20 | WMT FY2023, narrative, Item 1A | risk | In fiscal 2023, what specific types of third-party goods sold on Walmart's digital platforms could expose the company to liability or reputational risk? | Controversial, counterfeit, pirated, or stolen goods, or goods that fail to comply with applicable law or proprietary rights. | Accept | Correct; a marketplace risk factor. |
| 287 | 20 | WMT FY2023, narrative, Item 1A | risk | According to Walmart's fiscal 2023 risk factors, what is the current legal status of laws regarding intellectual property and product liability for marketplace transactions on its platforms? | It is currently unsettled whether these laws apply to the company. | Accept | Correct; a marketplace risk factor. |
| 293 | 76 | WMT FY2024, table, Item 5 | other | What was the average price paid per share for Walmart's stock repurchases in December 2023, adjusted for the 3-for-1 stock split? | $51.41 | Accept | Correct; split-adjusted, as the footnote says. |
| 294 | 34 | WMT FY2024, table, Item 7 | financial_table | What was Walmart's return on investment (ROI) for fiscal 2024? | 15.0% | Accept | Correct. |
| 297 | 82 | WMT FY2025, narrative, Item 1A | risk | What specific liability does Walmart Inc. identify in its fiscal 2025 risk factors regarding the compromise of health information? | Liability under the Health Insurance Portability and Accountability Act of 1996, as amended by the American Recovery and Reinvestment Act of 2009 | Accept | Correct; a data-breach liability statement. |
| 298 | 82 | WMT FY2025, narrative, Item 1A | risk | How does the increasing sophistication of AI technologies impact the risk of identity fraud for Walmart Inc. according to its fiscal 2025 disclosure? | Malicious actors may exploit AI to create convincing false identities or manipulate verification processes | Accept | Correct; an AI-enabled fraud risk. |
| 299 | 40 | WMT FY2025, narrative, Item 1A | risk | What specific financial services does Walmart Inc. offer through its ONE fintech venture and partners during fiscal 2025? | money transfers, digital payment platforms, bill payment, money orders, check cashing, prepaid access, co-branded credits cards, installment lending and earned wage access | Accept | Correct; the financial services offered, from a regulatory risk factor. |
| 300 | 40 | WMT FY2025, narrative, Item 1A | risk | What potential consequences does Walmart Inc. identify for failing to comply with laws and regulations related to its financial services in fiscal 2025? | fines, sanctions, penalties and harm to our reputation | Accept | Correct; a compliance risk statement. |
| 310 | 88 | WMT FY2026, table, Item 7 | financial_table | What was Walmart's total capital expenditure for fiscal 2026, and how much of that was allocated to new stores and clubs including expansions and relocations? | Total capital expenditures were $26,642 million, with $1,406 million allocated to new stores and clubs, including expansions and relocations. | Accept | Correct figures; $1,406 million is the U.S. new-store line (International is not broken down). |
| 311 | 88 | WMT FY2026, table, Item 7 | profitability | How did Walmart's return on assets compare between fiscal 2026 and fiscal 2025, and what was the primary driver of the change? | ROA was 8.2% in fiscal 2026 and 7.9% in fiscal 2025. The increase was primarily due to an increase in net income resulting from higher fair value of equity and other investments and higher operating income. | Accept | Correct; both years and the driver. |
| 312 | 46 | WMT FY2026, table, Item 7 | financial_table | What was the net cash provided by operating activities for Walmart Inc. in fiscal 2026? | $41,565 million | Accept | Correct. |
| 313 | 46 | WMT FY2026, table, Item 7 | profitability | What were the primary factors driving the increase in gross profit rate for Walmart's Sam's Club segment in fiscal 2026? | Operational efficiencies and higher margins in fuel, partially offset by higher eCommerce fulfillment costs and reorganization charges | Accept | Correct; the paragraph is Sam's Club's (membership, Sam's Cash). |
| 314 | 89 | XOM FY2019, narrative, Quarterly Information | profitability | What was the net income attributable to ExxonMobil for the fourth quarter of fiscal 2019? | 5,690 million dollars | Accept | Correct. |
| 315 | 89 | XOM FY2019, narrative, Quarterly Information | other | How many registered shareholders of ExxonMobil common stock were there as of December 31, 2019? | 354,828 | Accept | Correct; registered shareholders at December 31, 2019. |
| 317 | 41 | XOM FY2020, narrative, Item 2 | financial_table | How much synthetic oil did Exxon Mobil's consolidated subsidiaries in Canada/Other Americas produce in 2019? | 65 thousands of barrels daily | Accept | Correct prior-year column; names its year. |
| 324 | 77 | XOM FY2020, narrative, 17. Pension And Other Postretirement Ben | financial_table | What was the interest cost for Exxon Mobil's U.S. pension benefits in fiscal 2020? | 708 | Accept | Correct; no unit (millions). |
| 325 | 77 | XOM FY2020, narrative, 17. Pension And Other Postretirement Ben | financial_table | How much was the net pension enhancement and curtailment/settlement cost for Exxon Mobil's non-U.S. pension benefits in 2019? | 49 | **Revise** | Wrong column: 49 is the non-U.S. figure for 2020; the 2019 figure lies past the window. |
| 326 | 35 | XOM FY2021, table, Item 2 | financial_table | What was the total oil-equivalent proved developed reserves for Exxon Mobil's consolidated subsidiaries in the United States at the end of fiscal 2021? | 3,544 million barrels | Accept | Correct; oil-equivalent, consolidated subsidiaries, year-end 2021. |
| 327 | 35 | XOM FY2021, table, Item 2 | accounting_policy | How does Exxon Mobil convert natural gas to an oil-equivalent basis for its reserve reporting? | At six billion cubic feet per one million barrels | Accept | Correct; names no period; a conversion convention repeated yearly. |
| 331 | 83 | XOM FY2021, table, Business Results | financial_table | What was Exxon Mobil's worldwide net production of crude oil, natural gas liquids, bitumen, and synthetic oil in thousands of barrels daily for fiscal 2021? | 2,289 thousands of barrels daily | Accept | Correct. |
| 339 | 21 | XOM FY2022, table, 7. Equity Company Information | financial_table | What was ExxonMobil's share of total revenues from its equity companies in fiscal 2022, and how did this compare to the total revenues reported for these entities in the same year? | ExxonMobil's share was $57,528 million, while total revenues were $183,812 million. | Accept | Correct; the total and ExxonMobil's share. |
| 340 | 21 | XOM FY2022, table, 7. Equity Company Information | accounting_policy | What were the impairment charges related to upstream equity investments recorded by ExxonMobil in fiscal 2020, and where are these amounts included in the Consolidated Statement of Income? | Impairments were $0.6 billion and are included in "Income from equity affiliates" or "Other income". | Accept | Correct; the 2020 figure and its income-statement lines. |
| 343 | 14 | XOM FY2023, narrative, Business Results | other | What was the value of the all-stock transaction ExxonMobil agreed to for the acquisition of Pioneer in October 2023? | $59.5 billion | Accept | Correct; the Pioneer transaction's value. |
| 346 | 59 | XOM FY2023, table, Supplemental Information On Oil And Gas | financial_table | What were the total sales to third parties for ExxonMobil's Consolidated Subsidiaries in 2023, broken down by region? | United States: 5,098 million; Canada/Other Americas: 4,027 million; Europe: 1,345 million; Africa: 298 million; Asia: 2,490 million; Australia/OceANIA: 4,588 million; Total: 17,846 million | Accept | Correct; all seven figures, in millions (the answer misspells 'Oceania'). |
| 347 | 59 | XOM FY2023, table, Supplemental Information On Oil And Gas | accounting_policy | How much did ExxonMobil exclude from the results of operations for producing activities in 2022 due to other Upstream functions? | $4,802 million | Accept | Correct; the 2022 figure. |
| 348 | 53 | XOM FY2024, table, Item 2 | financial_table | According to the Gross and Net Productive Wells table, how many gross oil wells did ExxonMobil report for the United States at year-end 2024? | 30,333 gross oil wells | Accept | Correct reading of the flattened header (year-end 2024, oil, gross). |
| 352 | 95 | XOM FY2024, table, Note 3. Miscellaneous Financial Informat | financial_table | What was Exxon Mobil's total inventory value for crude oil, products, and merchandise as of December 31, 2024? | $19,444 million | Accept | Correct. |
| 353 | 95 | XOM FY2024, table, Note 3. Miscellaneous Financial Informat | other | How much did Exxon Mobil spend on research and development expenses in fiscal 2023? | $879 million | Accept | Correct prior-year figure; names its year. |
| 354 | 7 | XOM FY2024, table, Note 8. Investments, Advances And Long-T | financial_table | What was the total amount of equity method company investments and advances reported by Exxon Mobil for fiscal 2024? | 41,094 million dollars | Accept | Correct. |
| 355 | 7 | XOM FY2024, table, Note 8. Investments, Advances And Long-T | financial_table | By how much did Exxon Mobil's long-term receivables and miscellaneous, net of reserves, change between December 31, 2023, and December 31, 2024? | Decreased from 5,846 million to 5,763 million dollars | Accept | Correct; both dates. |
| 361 | 71 | XOM FY2025, table, Supplemental Information On Oil And Gas | accounting_policy | What were the total consolidated subsidiary capitalized costs for Exxon Mobil Corp as of the end of fiscal 2025, in billions of US$, excluding transportation, research, and LNG operations? | $9.1 billion less than Upstream PP&E | **Revise** | The answer does not answer the question: it asks for total capitalized costs, and '$9.1 billion less than Upstream PP&E' is the gap between net capitalized costs and Note 9's Upstream PP&E (the table's total capitalized costs are $419,340 million). |
| 362 | 71 | XOM FY2025, table, Supplemental Information On Oil And Gas | financial_table | According to the Capitalized Costs table for Exxon Mobil Corp, what was the total amount of unproved property costs reported in Australia/Oceania as of December 31, 2025, in millions of dollars? | $2,659 million | Accept | Correct. |

## 7. Special investigation: answers from reasoning instead of the passage

**What was looked for:** an answer built from the model's own reasoning or knowledge rather than from the visible
passage. Batch 2's pair 032 is the known case. Its answer field holds reasoning: it says the total is cut off, then
proposes a different question. Its only number is in the quote, so the validator passed it.

**Four checks were run:**
1. **The detector,** over every recorded proposal of batches 1–3, accepted or rejected: 596 answers (597 proposals,
   less one `malformed` reply with no answer).
   - **R1:** meta-language, where the answer talks about the text, the question or its own reasoning ("the provided
     text", "cut off", "not explicitly stated", "let's", "instead").
   - **R2:** an embedded question.
   - **R3:** a discursive answer, over 200 characters with three or more sentences.
   - **R4:** weak grounding: four or more of the answer's content words, and over 40% of them, absent from the visible
     text.
2. **A looser grounding rule:** two or more content words, and over 25%, absent from the visible text.
3. **The two batch 3 passages at risk,** read in full. They end on a cut-off total row, the setting of pair 032.
4. **The review of all 126 batch 3 pairs** against the visible text (§6), which reads every answer.

**The detector's flags, each inspected by hand** (batch 1 and 2 pair numbers are those of their own audits):

| Batch, position | Rules | Validator | Verdict |
|---|---|---|---|
| 1, 82 | R1, R3, R4 (13 of 21 words absent) | rejected (`evidence_length`) | **Genuine.** "The adjusted net income is not explicitly stated in the provided text segment, but…": reasoning about a cut-off row. The validator rejected it by chance. |
| 2, 8 (pair 032) | R1, R2, R3, R4 (41 of 54) | accepted | **Genuine.** The known failure, rejected by batch 2's review. |
| 1, 46 | R3 | rejected (`evidence_length`) | False positive: a list of four initiatives, quoted from the passage. |
| 2, 91 (pair 209) | R4 (5 of 12) | accepted | False positive: "negative" and "positive" paraphrase the signs; the figures are in the quote. |
| 3, 70 (pair 264) | R2 | accepted | False positive: "which were" matches the question pattern; the answer is nearly verbatim. |
| 3, 74 | R3 | rejected (`evidence_length`) | False positive: the table's 431-character description of a division, copied verbatim. |

**Results:**
- **Batch 3 has no reasoning answer** among its 199 proposals.
- **The looser rule** flags 14 answers in all: the same 2 genuine cases and 12 grounded paraphrases. The words it finds
  missing are "fiscal", "compared", "representing", and inflections such as "anchoring" for "anchor in". Its three
  batch 3 flags (pairs 339, 071 and 195) are grounded.
- **The cut-off totals:** the scan finds 10 planned passages whose visible text ends on a label with no value. Six are
  true cut-off totals: batch 1's positions 22, 49 and 82, batch 2's 8, and batch 3's 15 and 38. The other four are prose.
  - In batches 1–2, two of the four true cases produced a reasoning answer (batch 1's 82 and batch 2's 8).
  - In batch 3, neither did. **Position 15** (Apple FY2024, ending "Total term debt") gave pair 049, the commercial
    paper rate of 5.00%, and a rejection for a missing company: the fiscal 2023 commercial paper total, $(3,978)
    million, which is correct.
  - **Position 38** (Pfizer FY2023, ending "Total compensation cost related to nonvested awards not yet recognized,
    pre…") gave pairs 190 and 191. Both answers are in the visible text, and neither asks about the cut-off row.
- **Knowledge in place of the passage:** the review found one case in batch 3, in a question rather than an answer.
  - Pair 190 expands TSRU as "Time-Deferred Stock Unit". The passage has only "TSRU". Pfizer's filings define it as
    Total Shareholder Return Units, and "Time-Deferred" appears nowhere in Pfizer's corpus.
  - The answer ($244 million, the TSRU column for 2023) is right. The pair is revised.
  - The detector cannot see this: it reads answers only, and the expansion's words are ordinary.
- **Related, not reasoning:** pair 361's answer is in the passage but does not answer its question (revised). Five errors
  come from context beyond the window (§6).

**Conclusion:**
- **The failure did not recur.** It remains rare: 2 genuine cases in 597 proposals across three batches. The validator
  caught one by chance; the review caught the other.
- **No change to the generator is needed for P1.** The review reads every answer, so the detector stays a scratch audit
  tool, run on every batch, and the validator is unchanged.
- **After P1:** an R1/R2 rejection and a check for question words absent from the passage (pair 190's kind) are
  candidates for the next validator version.

## 8. Evidence leakage and false-negative protections

**Leakage, on the 108 included pairs (all from the trial build, §11):**

| Check | Result |
|---|---|
| Evidence rule, re-checked at the candidate build | 0 leak passages; every included pair passes |
| Lexical overlap with the permitted evaluation questions (word Jaccard; reject at 0.8) | Highest 0.263 (batch 2: 0.333; batch 1: 0.346) |
| Semantic screen (`screen-questions --pairs --candidates train_candidates_v2`, in scratch) | 428 questions (the 362 pairs and the 66) against the 10 permitted question texts, `query_id` and text only. **0 flagged at 0.90.** The highest of all, 0.8017, is still batch 1's pair 005 (019 in the 300-record numbering). Of batch 3's 108: highest 0.7922 (pair 146), 0 at or above 0.80, 4 at or above 0.75, 17 at or above 0.70; median 0.656. |
| The 0.80–0.90 review band (approved) | **No batch 3 question.** Batch 1's pair 005 is still the only one, and batch 1's review of it stands. The threshold was not changed. |
| Cross-split | 1 excluded at `build-pairs`. `load_validation_set(validation_pairs_v4, training=…)` passes against all 390 trial training pairs. |
| Duplicates | None (§9) |

**Evidence within the model's window:**
- **The generator** sees only the tokenizer-visible part of each passage (512 tokens for BGE, special tokens
  included).
- **The validator** located every accepted quote inside that part.
- **The independent re-check:** 105 of the 108 included quotes lie verbatim in the positive's visible text.
  - The other 3 are found by segments or cells. Pairs 204 and 250 elide text with "…", and pair 225 joins two cells of
    the balance sheet with an elision.
  - Batch 2 had 115 of 115 verbatim.
  - Counted with the special tokens, as the window is, the latest quote ends at token 506. 2 end after token 500 (batch
    2: 3, the latest at 512).
- **The window's limits** still produce questions about text the generator could not see: pairs 106, 118 and 123
  (headings before the window), and 325 and 169 (figures past it). The review catches them (§6).

**False negatives:**
- **Equivalents:** 58 of the 108 record exact equivalents (151 passages), and 41 record uncertain ones (102
  passages). The trainer never uses any of them as a negative. Batch 2: 54 (134) and 57 (159).
- **Conflicts:** `NegativeGuard` finds 347 conflicting pairs among the 390 trial pairs, and the planner keeps them in
  different batches.
  - Involving batch 3: batch 3 with batch 3, 57; batch 1 with batch 3, 43; batch 2 with batch 3, 32; batch 3 with the
    66, 10.
  - 86 of the 108 are in at least one conflict.
- **Batches:** conflicts grew faster than pairs (347 among 390, against 205 among 282). Later batches revisit the same
  statements in adjacent filings, so the mean fill at batch sizes 16 and 32 fell slightly.

 | Batch size | Batches | Full | Mean fill (batch 2 trial) |
 |---:|---:|---:|---:|
 | 16 | 26 | 24 | 0.938 (0.979) |
 | 32 | 15 | 12 | 0.813 (0.881) |
 | 64 | 9 | 5 | 0.677 (0.630) |

- **An independent probe:** for each included pair, any other included pair's positive was checked for the answer's
  figure (4 or more digits, not a year) wherever the guard would still allow it as a negative.
  - 3 hits, all chance matches in another company's passage: Walmart's 3,010 (pair 282) in JPMorgan's Corporate table,
    and the "2009" of the HIPAA amendment's name (pair 297) in Amazon's commitments table and Pfizer's executive table.
  - No unrecorded equivalent was found.
- **Repeated facts:** §9.
- **Known gaps remain:** paraphrased evidence, rounded figures and changed row labels.

## 9. Duplicate questions and repeated financial facts

| Against | Same normalised question | Highest word Jaccard | Cosine ≥ 0.90 (BGE, zero-shot) |
|---|---:|---:|---|
| Within the 108 | 0 | 0.722 | 3 (highest 0.916): JPMorgan's Markets and Corporate revenue lines for different years |
| Batch 1's 101 | 0 | 0.824 | 10 (highest 0.963): Walmart's comparable-sales threshold, the same definition (below); otherwise JPMorgan's Markets revenue in 2019 against 2020, Apple's repurchase authorization, and Amazon's lease, commitment and tax lines for other years |
| Batch 2's 115 | 0 | 0.850 | 11 (highest 0.989): Apple's fiscal 2024 State Aid impact, the same fact (below); otherwise Amazon's lease liabilities at 2019 against 2022, Tesla's current assets at 2021 against 2022, Apple's repurchases, receivables and term debt, and JPMorgan's capital and liquidity risk functions |
| The 66 of `train_candidates_v2` | 0 | 0.609 | 4 (highest 0.930): Tesla's net income and VIE assets, Apple's right-of-use assets and Amazon's finance-lease liabilities, for other years or measures |
| `validation_pairs_v4` (93) | 0 | 0.548 | 3 (highest 0.934): Tesla's total current assets at 2021 against 2025 |
| Earlier train generation (v1, v2) | 0 | 0.609 | 5 (highest 0.930): the same questions as the 66 (they come from these generations), and one more Amazon lease line |

**No question repeats another.**

**Repeated financial facts:** every batch 3 pair was checked against every other trial candidate of the same company. A
neighbour shares a 4-or-more-digit answer figure (not a year), or has a question cosine of at least 0.90, or a word
Jaccard of at least 0.70. There are 28 such neighbours.
- **Two ask the same fact twice, from different passages.**
  - **Pair 055** (Apple's FY2025 filing) and **batch 2's pair 034** (FY2024 filing; 048 in the 300-record numbering)
    both ask for fiscal 2024's State Aid impact on the tax provision, $10,246 million. Cosine 0.989.
  - **Pair 260** (Walmart FY2020, an Item 6 footnote) and **batch 1's pair 085** (another FY2020 passage; now 262) both
    ask for the comparable-sales threshold, "more than five percent". Cosine 0.963.
  - Both are in `NegativeGuard` conflict, so they are never in one batch and never each other's negative. Each positive
    holds the fact, so both are valid pairs, and they are kept.
- **The other 26** ask the same template for a different year, measure or line, with a different answer. 12 of them are
  in conflict anyway (adjacent filings); the 14 that are not are different facts. None is an unrecorded equivalent.

**A trend to watch:** same-template neighbours keep growing (cosine ≥ 0.90: 10 against batch 1 and 11 against batch 2;
batch 2 had 8 against batch 1). They are mostly Apple and Amazon statement lines and JPMorgan segment tables. With
Apple's and Amazon's groups exhausted, batches 4–6 will draw these templates from five companies.

## 10. Company, fiscal-year and numerical accuracy, and distribution

**Accuracy:**
- **Company:** every question names its positive's company (108 of 108 included; all 126 passed the validator's
  company rule).
- **Period:**
  - 96 included questions name the filing's fiscal year or the prior year.
  - 4 name only another year, each legitimately:
    - the 2017 Whole Foods acquisition, in the FY2019 filing's acquisition history;
    - a contribution planned for 2020, in the FY2019 filing;
    - the lease standard's February 1, 2019 adoption, the first day of Walmart's fiscal 2020;
    - fiscal 2020 impairments, from the FY2022 filing's three-year note.
  - 8 name no year, all for facts that do not change between filings: methodologies, governance, policies and
    definitions, and three risk statements made in one filing only (the ACA's expected impact, generative AI,
    regulators' powers).

  The period errors are the 3 revised pairs (§6).
- **Numbers:**
  - Every figure in the 85 numeric answers is in its quote (independent check).
  - The review found 1 wrong column (325) and 4 correct figures attributed to the wrong business (106, 112, 118, 123),
    plus Tesla's award figure given as the total (229).
  - Units: 71 answers state one, 14 give a bare figure from a table "in millions", and none states a wrong unit.

**Distribution by company:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Planned | 16 | 5 | 16 | 16 | 16 | 16 | 15 |
| Done | 16 | 5 | 16 | 16 | 16 | 16 | 15 |
| Pass validator | 28 | 9 | 22 | 19 | 27 | 28 | 21 |
| Evidence exclusions | 9 | 2 | 3 | 2 | 6 | 5 | 1 |
| Pairs built | 19 | 7 | 19 | 17 | 21 | 23 | 20 |
| **Included** | **17** | **5** | **15** | **12** | **18** | **23** | **18** |
| Batch 2 included | 19 | 13 | 16 | 21 | 18 | 14 | 14 |
| Batch 1 included | 17 | 17 | 11 | 16 | 15 | 15 | 10 |
| With batches 1–2 and the 66 (390) | 63 | 43 | 55 | 57 | 62 | 59 | 51 |

**Distribution by fiscal year:**

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Planned | 18 | 11 | 14 | 16 | 12 | 12 | 15 | 2 |
| Included | 18 | 11 | 16 | 15 | 12 | 16 | 16 | 4 |
| Batch 2 included | 13 | 15 | 16 | 19 | 17 | 17 | 16 | 2 |
| Batch 1 included | 15 | 13 | 17 | 15 | 12 | 14 | 13 | 2 |
| With batches 1–2 and the 66 (390) | 56 | 49 | 58 | 55 | 50 | 58 | 55 | 9 |

- **Companies:** 5–23 included each.
  - Amazon has 5: its last five groups, all FY2019, were planned.
  - Pfizer is lowest of the full companies (12). The validator rejected 12 of its 31 proposals, and the review did
    not accept 5 of its pairs.
  - Walmart is highest (23): every one of its pairs was accepted.
- **The 390 together:** Apple has the largest share, 16.2%, and Amazon the smallest, 11.0%. The others have 13.1–15.9%.
- **Years:** 11–18 per full year in the batch (2020 lowest, with 11 planned), and 49–58 in the 390.
- **Passage kind:** 64 table and 44 narrative pairs (61 and 39 planned passages; batch 2: 57 and 58).
- **Question type:**
  - `financial_table` 36 (batch 2: 42), `risk` 20, `other` 17;
  - `accounting_policy` 13, `revenue` 7, `debt_liquidity` 7;
  - `profitability` 4, `tax` 2, `mdna` 2.

## 11. Provenance and consistency of the included pairs

**The trial build (scratch copy only):**
- **Sources:**
  - `pilot_v3` and `pilot_v2` with their existing decisions;
  - P1's pair set (362 train pairs from batches 1–3), with a scratch decisions file. It holds batch 1's 112 and batch
    2's 124 decisions, re-keyed by `passage_id` and `question_sha256` to the 300-record pair IDs, and this batch's 126;
  - the scratch screening record.
- **Result:** 390 candidates (51 + 15 + 101 + 115 + 108).
- **Re-keying is mechanical:** all 236 earlier decisions found their pair by passage and question hash (112 and 121 of
  them under a new ID), and the committed files were not touched.
- **The 162 validation pairs** that `build-pairs` also builds have no decision and are excluded
  (`no_quality_decision`), as before.

**Provenance on all 108 included records:** the same block of 11 fields as batches 1 and 2:
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
- `verify_candidate_provenance` traces all 390 back to their records and decisions.
- `load_candidates`, the trainer's fail-closed loader, loads all 390.
- `load_validation_set` loads all 93 pairs of `validation_pairs_v4` apart from the 390.
- **The 66 are unchanged:** a rebuild of `train_candidates_v2`'s sources in scratch gives byte-identical files,
  `train_candidates.jsonl` `908d4c79…` and `candidates_log.jsonl` `c8d38f96…`.

## 12. Batch 3 against batches 1 and 2

| | Batch 1 | Batch 2 | Batch 3 |
|---|---:|---:|---:|
| Done / failed | 99 / 1 | 100 / 0 | 100 / 0 |
| Tokens per planned passage | 1,483 | 1,467 | 1,463 |
| Pass the validator | 151 of 198 (76.3%) | 159 of 200 (79.5%) | 154 of 199 (77.4%) |
| False validator rejections | 3 (1.5%) | 4 (2.0%) | 4 (2.0%) |
| Evidence exclusions | 39 (25.8%) | 35 (22.0%) | 28 (18.2%) |
| Pairs built | 112 | 124 | 126 |
| Accept / revise / reject | 101 / 6 / 5 | 115 / 6 / 3 | 108 / 14 / 4 |
| **Acceptance** | **90.2%** | **92.7%** | **85.7%** |
| **Included per planned passage** | **1.01** | **1.15** | **1.08** |
| Passages with no included pair | 30 | 20 | 23 |
| Factual errors (fail) | 4 | 6 | 13 |
| Period errors | 3 | 3 | 3 |
| Scope errors (part as the whole, wrong business) | 2 | 1 | 5 |
| Low-value-source exclusions | 5 | 3 | 5 |
| Reasoning answers among the proposals | 1 (rejected by the validator) | 1 (rejected by the review) | 0 |
| Highest semantic similarity | 0.8017 | 0.7929 | 0.7922 |
| Cosine ≥ 0.90 within the batch | 2 | 4 | 3 |
| Company range of included pairs | 10–17 | 13–21 | 5–23 (12–23 without Amazon) |
| HTTP 429 / 503 responses | 0 / 0 (resume) | 6 / 0 | 3 / 3 |

**Systematic differences:**
- **Lower acceptance, from more revisions** (14 against 6). The excess has a few identifiable sources.
  - JPMorgan's segment passages give 4 scope errors. The plan drew three times as many of them as before.
  - Two shareholder counts are misworded.
  - One each of the other kinds make up the rest.
- **The generator is unchanged:**
  - the same false-rejection rate and patterns;
  - no reasoning answer;
  - tokens per passage within 1.4% of batch 1.
- **Fewer evidence exclusions** (18.2%). Apple's statement lines lost the most.
- **Yield** of 1.08 included per passage, between batches 1 and 2.
- **Two same-fact repeats** across batches, both separated by the guard (§9).
- **Server errors:** Groq's 503 "over capacity" errors appear for the first time in the kept logs, and were handled
  (§2).

Nothing in batch 3 suggests a drift in the generator. The quality change is in what the plan drew. The review catches
it, and the gate absorbs it.

## 13. Reassessment for batches 4–6: sequencing, company balance and budget

**Sequencing (option A, approved):**
- `train_v3_p1_batch3_quality_decisions.jsonl` is bound to the pair set that the current 300 records build. The
  scratch replay confirmed that pair set exactly.
- Batch 4 will append records and renumber the pair IDs again.
- At the end of P1:
  1. the real `build-pairs` runs once;
  2. all six batches' decisions are carried into one decisions file keyed by the final pair IDs;
  3. the mapping by `passage_id` and `question_sha256` is mechanical (verified here for all 236 of batches 1–2).
- Batch 4's plan must be written only now, after batch 3's records exist, because `--after-generated` excludes
  generated passages, not other plans.

**Company balance:** scratch plans with the frozen options, excluding all three batches' groups, find the supply
below.

| Remaining eligible overlap groups | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| After batch 3 | 0 | 0 | 664 | 210 | 71 | 62 | 94 |
| After batches 4–6 under the frozen options | 0 | 0 | 604 | 150 | 11 | 2 | 34 |

- **A correction:** batch 2's audit put the other five companies at "75 or more each". That was the 400-passage
  plan's allocation, not their supply.
- **Batch 4 under the frozen options:**
  - JPMorgan, Pfizer, Tesla, Walmart and ExxonMobil at 20 each (20%, under the 25% cap);
  - 2–3 passages per company-year, and every company covers all its years;
  - 9 of JPMorgan's 20 are segment-results passages.
- **Batches 4–6 (300 passages):** 60 per company.
  - JPMorgan, Pfizer and ExxonMobil spread 8–9 per year.
  - **Tesla and Walmart run out of their thin years:**
    - Tesla would take 19 FY2019 passages and only 3 in each of FY2021–FY2023;
    - Walmart would take 13 each in FY2025 and FY2026 and 5 each in FY2021–FY2022.
  - The skew begins after batch 4.
- **All 600 planned passages:** Apple 44 (7.3%), Amazon 33 (5.5%), and 103–106 (17.2–17.7%) for each of the others.
- **Projected candidates** at batch 3's yield (1.08 per passage): about 714.
  - Apple would hold 63 (8.8%) and Amazon 43 (6.0%).
  - The others would hold 116–127 each (16.2–17.8%).
  - The cap holds. Today's 390 hold Apple 16.2% and Amazon 11.0%.
- **No way to add Apple or Amazon fits the frozen strategy.** It would take more than one passage per overlap group,
  that is, near-copies of passages already used. That would weaken the false-negative and leakage protections, so it is
  not recommended.

**Budget:**
- About 740,000 tokens were approved for batches 2–6. Batches 2 and 3 used 292,934 (146,678 and 146,256), so
  **447,066 remain**.
- **Batch 4:** about 147,700 (122,100 prompt from its 49,683 visible tokens, 25,600 completion), with a ceiling of
  198,900 at 768 tokens per reply. About 299,400 would remain after it.
- **Batches 4–6:** about 442,400 (365,700 prompt, 76,700 completion). That leaves a margin of about 4,700 tokens
  (1.0%).
  - At batch 1's rate per passage (1,483), the margin is about 2,200.
  - If every reply were as long as batch 3's longest (500 tokens), the total would be about 515,700, over the budget.
- **The model is accurate:** batch 3 came in 0.3% under its estimate.
- **So the budget does not constrain batch 4.** It does constrain batch 6, which may stop a few passages short unless
  a small contingency is approved before then.
- **Groq's daily limit** (1,000 requests) is far above a batch's ~106 requests. The binding limits are the unreported
  ones behind the 7–11 minute waits, which the generator handles.

## 14. Recommendation

- **Batch 3 quality gate: PASS** on every approved criterion (§1).
- **Batch 4: GO**, under five conditions:
  1. **One generator:** the same model, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…` and
     plan options. Only `--name` changes, to `train_v3_p1_batch4`, and the plan is written now that batch 3's records
     exist. Under the frozen options it holds the five remaining companies at 20 each, and it is deterministic.
  2. **Option A:** no real `build-pairs` or `build-candidates` in `bge_base_ft_v3` until batch 6 is generated and
     gated (§13).
  3. **Rate limits:** run with `--max-requests` set to the remaining passages, keep the console output with `tee`, and
     rerun the same command after each clean stop.
  4. **Stop rule:** batch 4 is replayed and gated on a scratch copy, as here, before batch 5 is planned.
  5. **Review focus:** the review keeps reading every answer, with particular care for three things:
     - JPMorgan segment scope;
     - executive and board tables;
     - the reasoning detector's flags, run on every proposal again.
- **The three-seed experiment: still NO-GO**, until B1–B3 of the readiness audit are resolved.

**Your decisions:**

| # | Decision | Recommendation |
|---|---|---|
| G1 | Company balance for batches 5–6 (needed before batch 5's plan, not batch 4's): (a) continue under the frozen options, with Tesla's and Walmart's thin years exhausted; (b) stop P1 after batch 4 (about 500 candidates, Apple about 12.7% and Amazon about 8.6%); (c) smaller batches 5–6 | **Run batch 4 now under (a) or (c).** Its plan is the same under both, and it is balanced by year for all five companies. Decide between (a), (b) and (c) after batch 4's gate. (a) if the planned data size matters more: the final mix keeps every company at or under 17.8%. (b) if an even company mix matters more. Do not loosen one passage per overlap group to add Apple or Amazon. |
| G2 | Budget margin for batches 4–6 (about 1%) | No change for batch 4. Before batch 6, approve a contingency of about 5% (some 22,000 tokens), or accept that batch 6 may stop a few passages short. |
| G3 | Validator weaknesses: 4 false rejections in two known patterns; reasoning in an answer; the model's knowledge in a question (pair 190) | Keep them unchanged through P1. Record them for a later validator version, with R1/R2 and question-word grounding as candidate rules. |
| G4 | The semantic review band | Keep the approved 0.90 exclusion and the 0.80–0.90 recorded review. Batch 3 put no question in the band. |

## Appendix: how the results were obtained

Everything below ran in a scratch copy of `data/processed/train/bge_base_ft_v3`. It used a copy of
`configs/train/data_prep_v3.yaml` that differs only in `output_dir`, the venv, and `HF_HUB_OFFLINE=1`.

```bash
python3 scripts/02_generate_queries.py revalidate --config SCRATCH/data_prep_v3_scratch.yaml --split train --dry-run
python3 scripts/02_generate_queries.py build-pairs --config SCRATCH/data_prep_v3_scratch.yaml
python3 scripts/02_generate_queries.py screen-questions --config SCRATCH/data_prep_v3_scratch.yaml \
    --name train_v3_p1_batch3 --pairs --candidates train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_v2check.yaml   # byte-identical to train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_trial.yaml     # pilot_v3, pilot_v2, P1 batches 1-3; semantic_screening
python3 scripts/02_generate_queries.py plan --config SCRATCH/data_prep_v3_scratch.yaml --split train \
    --name cap_after_b3_100 --size 100 --method balanced_years --max-company-share 0.25 --one-per group \
    --exclude-candidate-groups train_candidates_v2 \
    --after-generated configs/train/data_prep_v2.yaml configs/train/data_prep_v1.yaml
    # capacity only; also --size 300, and --size 1101 --max-company-share 1.0 for the whole remaining supply
```

**Other scripts:**
- A trace of every candidate through `build_pairs`' own functions. It matches the written pairs, ID for ID.
- The trainer's `load_candidates` and `plan_batches`, `verify_candidate_provenance`, and `load_validation_set` with
  the training pairs.
- The duplicate, repeated-fact and false-negative probes, and the quote and token-position checks.
- The reasoning detector and its looser variant over all 597 proposals of batches 1–3.
- Corpus look-ups for pairs 106, 118, 123, 190, 229 and 236.

No evaluation file was read except the 10 permitted question texts, through the screen's own loader and the lexical
check. No evaluation question text was printed or reviewed, because no batch 3 question reached the review band.
