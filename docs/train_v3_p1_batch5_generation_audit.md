# P1 batch 5 (`train_v3_p1_batch5`): generation audit and quality gate

Status: 2026-11-08.
- **What this is:**
  - an audit of the 100 generation records of plan `train_v3_p1_batch5` in `bge_base_ft_v3`;
  - an AI review (Claude) of the 120 pairs they build;
  - the reasoning-answer investigation, extended to all five batches;
  - the batch quality gate, a comparison with batches 1–4, and a reassessment of batch 6, the budget and the
    concentration risks.

  It uses the method of the batch 1–4 audits ([batch 4](train_v3_p1_batch4_generation_audit.md),
  [batch 3](train_v3_p1_batch3_generation_audit.md)). It is not human annotation, and no decision here is a gold label.
- **Where the decisions are:** `data/train_review/train_v3_p1_batch5_quality_decisions.jsonl`.
  - 120 rows (106 accept, 14 revise, 0 reject), each `reviewer: ai_review` and `gold: false`.
  - Each row is bound to its `pair_id`, `passage_id` and the SHA-256 of its exact question. The pair IDs are those of
    the pair set that the current 500 train records build (§13).
  - SHA-256 `f79763a4…7194`.
- **What was not done:**
  - no question was generated and no API request was sent;
  - no generation record, plan, candidate dataset, validation set or earlier decision was changed. The records of
    batches 1–4 and their decisions are byte-identical (§2);
  - `build-pairs`, `screen-questions`, `build-candidates` and `plan` were not run in `bge_base_ft_v3`. They ran only in a
    scratch copy of the preparation, which was then discarded. No final candidate dataset was built;
  - the validator is unchanged;
  - nothing was trained, committed or pushed.
- **Protected files:** 351 files under `data/`, `configs/`, `src/` and `scripts/` were checksummed before
  and after the audit. They are unchanged; the decisions file is the only file added.

## 1. Verdict

**The batch passes the approved quality gate on all four criteria.**

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | At least 0.8 included pairs per planned passage | 106 included for 100 planned passages: **1.06** | **Pass** |
| 2 | At least 75% AI-review acceptance | 106 of 120 built pairs: **88.3%** | **Pass** |
| 3 | Zero accepted leakage violations | **0**: no included pair fails the evidence, cross-split, lexical, semantic or duplicate checks (§8, §9) | **Pass** |
| 4 | Complete provenance for every included pair | **106 of 106** carry the full provenance block, and the fail-closed provenance check passes (§11) | **Pass** |

**The special-attention items:**
- **JPMorgan segment figures given to the whole Firm: none.**
  - The 5 segment-results passages gave 5 pairs: 3 accepted, 1 revised for scope within a segment (Asset & Wealth
    Management's provision given to its Global Private Bank business), and 1 revised for period.
  - Position 51, which holds the table behind batch 4's pairs 154–155, produced no Firm attribution.
- **ExxonMobil totals against components: 3 errors**, all in the supplemental oil-and-gas tables:
  - pairs 516 and 517 read the consolidated subsidiaries' December 31, 2018 block as ExxonMobil's 2019 figures;
  - pair 600 gives the consolidated subsidiaries' future cash inflows as ExxonMobil's total.

  Its production questions are all correct: Permian, Canada, U.S. upstream earnings, total upstream earnings,
  refining capacity.
- **Reasoning answers:** 1 among the 200 proposals (position 16), plus 1 answer that fills cut-off cells with another
  row's figures (position 73). The validator rejected both: the first by chance, the second by design. **No included
  answer or question is built from reasoning or outside knowledge** (§7).
- **Walmart's opioid litigation:** 6 included pairs, all new facts. One repeats batch 4's case-count template for
  another date, and NegativeGuard keeps the two apart. The topic now holds 13 of Walmart's 101 candidates (§9).
- **ExxonMobil's oil-equivalent conversion:** not repeated. Position 80 carries the footnote, but neither of its
  questions asks it, so the fact stays at 2 candidates (batches 3 and 4).
- **The 0.80–0.90 semantic review band:** no batch 5 question (highest 0.7908). Across batches 1–5 the band holds 2
  questions, both reviewed earlier: batch 1's pair 005, accepted, and batch 4's pair 154, revised (§8).
- **Existing data:** `train_candidates_v2` rebuilds byte-identical, and `validation_pairs_v4` loads apart from all 610
  trial training pairs. All 500 records use one model, prompt, validator and parameter set.

**Acceptance rises to 88.3%** (batches 3–4: 85.7%). Scope errors fall from 13 to 6. Period errors rise to 8: questions
whose answer depends on the filing (a recast or split-adjusted figure, a method that changed) and flattened tables that
carry the wrong year's labels (§6).

**GO for batch 6**, subject to your approval and the conditions in §14. Its expected benefit justifies its cost if the
planned data size (blocker B2) stands. Otherwise stop at batch 5 or run a 60-passage batch 6 (§13).
- **Remaining Groq budget: 155,688 tokens** of the approved ~740,000.
- **Batch 6's cost:** about 146,200 under the frozen options, leaving about 9,500 (6.1%).
- **Candidates:** 610 now, and **about 718–720** after batch 6.
- **Concentration:**
  - companies stay at or under about 17.6%, under the 25% cap;
  - Tesla's FY2019 and Walmart's FY2025–FY2026 would weigh more: about 25% and 37% of those companies' pairs (§13).

**Unchanged:** the three-seed A0 versus A1 experiment stays **NO-GO**. Blockers B1 (a selection-aware decision rule),
B2 (the planned data size) and B3 (the GPU rehearsal) still stand.

## 2. Records and provenance

| Check | Result |
|---|---|
| Plan | `train.train_v3_p1_batch5.json`, SHA-256 `bfcb220d…b660`, unchanged; 100 train passages |
| Records file | `generation/train.records.jsonl`, 500 lines, SHA-256 `51e8115b…7b00` |
| Batches 1–4 | Lines 1–400 hash to `398697df…3830` (and lines 1–300 to `3a8952fe…`), as the batch 4 audit recorded them. The runs only appended. |
| Batch 5's records | Lines 401–500: one record per plan position, in plan order; 500 distinct passages in the file |
| Outcome | **100 `done`, 0 `failed`**. Every record took 1 attempt, with no error. |
| Identity | All 500 records: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…`, record format 3 |
| Run | `train.run.json` is unchanged (`72c9e7bf…`); there is no continuation file |
| Replies | 100 distinct response hashes (499 across the file; batch 1's failed record has none) |
| Proposals | 200: every reply held two questions |
| Validator re-run | `revalidate --dry-run`: 499 up to date, 0 changed, 1 failed record (batch 1's position 87) |
| Tokens | **146,512** (121,640 prompt, 24,872 completion), equal in the records and the four run summaries. Batches 1–5 total 732,596. |
| Completion tokens per reply | 161–404, median 240. No reply came near the 768 limit. |
| Against the dry-run estimate | About 147,400 tokens were estimated; the run used 146,512 (0.6% less) in 105 HTTP requests |

**The runs:** four invocations, from 2026-11-07 22:07 to 2026-11-08 10:46 UTC (record timestamps), from
`logs/train_v3_p1_batch5.log`. The log holds no credential.

| Run | Request budget | HTTP requests | HTTP 429 | HTTP 503 | Done | Tokens | Ended by |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 100 | 48 | 2 | 0 | 46 | 67,658 | a rate-limit wait of 627 s, after a 35 s wait was taken |
| 2 | 52 | 19 | 1 | 0 | 18 | 26,344 | a wait of 454 s |
| 3 | 33 | 26 | 2 | 0 | 24 | 34,930 | a wait of 633 s, after a 135 s wait was taken |
| 4 | 33 | 12 | 0 | 0 | 12 | 17,580 | the end of the plan |
| **All** | | **105** | **5** | **0** | **100** | **146,512** | |

**How Groq's limits were handled:**
- **The generator worked as designed.** 105 HTTP requests gave 100 usable replies and 5 HTTP 429 responses, and no
  server errors.
  - It waited out the two 429 waits of at most 180 s and stopped cleanly at the three longer ones.
  - Each rerun skipped the completed passages, and the generator spent 4,468 s in pacing and rate-limit waits.
- **No passage was lost or retried** because of a rate limit.
- **What the headers show:** the last response of each run showed 1,000 requests per day (955 or more remaining) and
  8,000 tokens per minute. The 429 waits of 7.5–10.5 minutes are again longer than any per-minute window.

**Against batches 1–4 and the plan's per-batch estimate** ([training_full_v1.md](training_full_v1.md), §9):

| | Estimate | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Batch 5 |
|---|---|---:|---:|---:|---:|---:|
| Tokens | about 143,000 | 148,284 | 146,678 | 146,256 | 144,866 | 146,512 |
| Questions passing the validator | about 155 | 151 | 159 | 154 | 168 | 159 |
| Pairs built | about 140 | 112 | 124 | 126 | 133 | 120 |
| AI review accepts | about 115 | 101 | 115 | 108 | 114 | 106 |
| Candidates | about 100 (range about 80–115) | 101 | 115 | 108 | 114 | 106 |

## 3. The funnel

200 proposed → **159** pass the generator's validator (79.5%) → **120** pairs built → **106** accepted by the review →
**106** included in the trial candidate set.

| | Plan | Done | Proposed | Pass validator | Pairs built | Accepted (included) |
|---|---:|---:|---:|---:|---:|---:|
| Batch 1 | 100 | 99 | 198 | 151 (76%) | 112 | 101 |
| Batch 2 | 100 | 100 | 200 | 159 (80%) | 124 | 115 |
| Batch 3 | 100 | 100 | 199 | 154 (77%) | 126 | 108 |
| Batch 4 | 100 | 100 | 200 | 168 (84%) | 133 | 114 |
| Batch 5 | 100 | 100 | 200 | 159 (80%) | 120 | 106 |
| **All** | **500** | **499** | **997** | **791 (79%)** | **615** | **544** |

- **Exclusions at each step:**
  - `build-pairs` excluded 39 questions, every one for evidence leakage: `evaluation_evidence` 35 and
    `cross_split_evidence` 4 (§5).
  - Nothing was excluded for a validator re-check, a duplicate or evaluation overlap.
  - The trial candidate build excluded nothing beyond the 14 pairs the review did not accept.
- **Per passage:** 27 passages contribute no included pair, 40 contribute one and 33 contribute two (batch 4: 23, 40 and
  37).
- **Fewer pairs than batch 4** (120 against 133): more evidence exclusions (39 against 35, with Tesla losing 13) and 20
  `missing_company` rejections (8 in batch 4).
- **Batches 1–4 in this replay:**
  - The scratch `build-pairs` ran over all 500 records. Batches 1–4 keep the same pairs and exclusions, and their
    traced outcomes are identical to the batch 4 audit's.
  - 397 of the 495 earlier pair IDs differ from the 400-record numbering, as expected (§13).
  - The trace of every candidate through `build_pairs`' own functions matches the written pairs, ID for ID.

## 4. Generator acceptance and rejection (validator `sec-qcheck-v3`)

| Reason | Count | Verdict |
|---|---:|---|
| `missing_company` | 20 | All correct: the question names no company, or only a business ("Sam's Club U.S.", "the Consumer Healthcare JV", "Maxwell Technologies", "the Firm"). |
| `copied_from_passage` | 8 | 2 copy a run of prose (9 and 13 words). 5 are borderline: runs of 8–11 words, mostly names or an 8-word fragment. 1 is false (position 68): a row label of a cash-flow table flattened without `[TABLE]` tags, so the label exemption does not apply. |
| `evidence_length` | 5 | All correct by the rule: quotes of 348–541 characters (limit 300). |
| `evidence_not_in_visible_text` | 4 | 2 correct: position 17's answer is wrong and its quote's gap is too long; position 73 fills cut-off cells with another row's figures (§7). 2 borderline (positions 72 and 91): faithful quotes whose fragment after an ellipsis starts with a period, as batch 4's position 85. |
| `answer_not_supported` | 2 | Both false (positions 11 and 42): the answers are right, but their years are column headers, not in the quote. |
| `question_format` | 1 | Correct: position 16, whose answer is a reasoning answer (§7). |
| `number_not_in_passage` | 1 | Borderline (position 92): 4,000 shares is a correct conversion of the table's "(4)" thousand, but 4,000 is not in the text. |

**Totals:**
- 30 correct;
- 8 correct by the strict rule (borderline);
- **3 false rejections**, positions 11, 42 and 68: 1.5% of proposals. Batches 2–4 had 4 each (2.0%) and batch 1 had 3
  (1.5%).

The false rejections repeat the two known patterns:
- years missing from the quote (11 and 42);
- a row label outside a `[TABLE]` block (68; here the whole table is flattened without tags).

The validator stays as it is: changing it mid-P1 would change the generator's identity.

| Pos | Source | Reason | Question (verbatim) | Verdict |
|---:|---|---|---|---|
| 2 | PFE FY2019, narrative, Item 1 | `copied_from_passage` | What functions does the Worldwide Medical and Safety group, within WRDM, ensure Pfizer provides to stakeholders? | Correct by the rule, borderline: an 8-word run, mostly the group's name. |
| 6 | JPM FY2025, narrative, Item 1A | `evidence_length` | What specific cybersecurity enhancements does JPMorganChase identify as necessary to address vulnerabilities in its software and systems for fiscal 2025? | Correct by the rule: the quote has 541 characters (limit 300). |
| 6 | JPM FY2025, narrative, Item 1A | `evidence_length` | What are the potential negative consequences cited by JPMorganChase for fiscal 2025 if a successful circumvention of its systems or those of its third-party vendors occurs? | Correct by the rule: the quote has 397 characters (limit 300). |
| 11 | JPM FY2021, table, Business Segment Results | `answer_not_supported` | What was JPMorgan Chase & Co's total assets under management as of December 31, 2021, and how did it compare to the 2020 figure in billions of dollars? | False: the answer is right ($3,113 billion in 2021 and $2,716 billion in 2020, assets under management); its years are column headers, not in the quote. |
| 15 | XOM FY2025, narrative, Item 2 | `missing_company` | How long is the standard production period for an oil field under Guyana's petroleum agreements? | Correct: the question does not name the company. |
| 16 | JPM FY2024, narrative, Market Risk Management | `question_format` | How did the update in the second quarter of 2024 affect the reported sensitivities for JPMorgan Chase & Co in fiscal 2024 regarding a +100 basis point shift in short-term rates for a flatter yield curve in non-U.S. dollar currencies? | Correct: the answer is the model's reasoning ('The passage does not provide a specific value ... but for U.S. dollar it was (1.0)'): the non-U.S. rows lie past the window (section 7). |
| 17 | PFE FY2023, table, Item 8 | `evidence_not_in_visible_text` | What were the external acquisition-related costs for Pfizer's Biopharma segment in fiscal 2023, and what specific amount was recognized as post-closing compensation expense for Seagen employees in the fourth quarter of that year? | Correct: the quote's gap after the ellipsis exceeds the 600-character limit, and the answer is wrong: $672 million is the Biopharma segment's restructuring amount, not its acquisition costs. |
| 17 | PFE FY2023, table, Item 8 | `missing_company` | According to the restructuring accrual summary, what was the total provision for restructuring in the year ended December 31, 2023? | Correct: the question does not name the company. |
| 20 | XOM FY2022, table, 6. Additional Working Capital Informatio | `evidence_length` | What was the total amount of accounts payable and accrued liabilities for Exxon Mobil Corp in fiscal 2022? | Correct by the rule: the quote has 361 characters (limit 300). |
| 21 | JPM FY2023, table, Credit And Investment Risk Management | `copied_from_passage` | How much was JPMorgan Chase & Co's allowance for credit losses reduced due to the adoption of changes to the TDR accounting guidance on January 1, 2023? | Correct: copies a 13-word run of the sentence. |
| 24 | WMT FY2024, narrative, Item 1A | `missing_company` | In which U.S. District Court is the consolidated multidistrict litigation titled In re National Prescription Opiate Litigation pending, and what is its MDL number? | Correct: the question does not name the company. |
| 29 | WMT FY2025, table, Item 8 | `missing_company` | How much were total fuel-related expenses for Sam's Club U.S. in fiscal 2023? | Correct: the question does not name the company. |
| 38 | TSLA FY2019, table, Item 8 | `evidence_length` | What specific audit procedures did PricewaterhouseCoopers perform to evaluate the reasonableness of Tesla's assumptions regarding the nature and frequency of future claims for its automotive warranty reserve? | Correct by the rule: the quote has 348 characters (limit 300). |
| 42 | PFE FY2022, table, Item 8 | `answer_not_supported` | What was the total amount of short-term investments held by Pfizer as of December 31, 2022, and how did it compare to the figure for December 31, 2021? | False: the answer is right ($22,316 million and $29,125 million); its dates are column headers, not in the quote. |
| 42 | PFE FY2022, table, Item 8 | `copied_from_passage` | How much long-term equity securities did Pfizer hold in restricted trusts for U.S. non-qualified employee benefit plans at the end of fiscal 2022? | Correct by the rule, borderline: an 11-word run, mostly the benefit plans' name. |
| 46 | JPM FY2021, table, Business Segment Results | `missing_company` | What change in the Firm's GSIB surcharge was effective January 1, 2023, which contributed to the line of business capital allocation changes? | Correct: the question does not name the company. |
| 49 | WMT FY2023, narrative, Item 8 | `copied_from_passage` | What amount did Walmart pay in the fourth quarter of fiscal 2023 for settlements with Cherokee Nation, New Mexico, and Florida? | Correct by the rule, borderline: an 8-word run, mostly the three parties' names. |
| 50 | XOM FY2025, table, Consolidated Statement Of Comprehensive  | `missing_company` | How much was the foreign exchange translation adjustment for the year ended December 31, 2024, reported in millions? | Correct: the question does not name the company. |
| 57 | PFE FY2025, table, Item 8 | `missing_company` | According to the stock-based compensation activity table, what was the compensation cost recognized pre-tax for RSUs in the year ended December 31, 2024? | Correct: the question does not name the company. |
| 59 | WMT FY2020, narrative, Item 1A | `evidence_length` | What specific consequences does Walmart Inc. face for noncompliance with applicable laws and regulations in fiscal 2020? | Correct by the rule: the quote has 527 characters (limit 300). |
| 59 | WMT FY2020, narrative, Item 1A | `missing_company` | What are the claimants alleging regarding Asda's retail store employees versus warehouse employees in the Manchester Employment Tribunal? | Correct: the question does not name the company. |
| 62 | PFE FY2024, narrative, Item 1 | `missing_company` | Which agencies are identified as the primary regulatory authorities for medicines in the United Kingdom and Japan? | Correct: the question does not name the company. |
| 64 | WMT FY2025, table, Item 8 | `missing_company` | How many shares of Restricted Stock Units were granted during fiscal 2025, and what was their weighted-average grant-date fair value per share? | Correct: the question does not name the company. |
| 68 | TSLA FY2020, narrative, Item 8 | `copied_from_passage` | How much did Tesla spend on purchases of property and equipment, excluding finance leases, in fiscal 2020? | False: the run is a row label of a cash-flow table flattened without [TABLE] tags, so the label exemption does not apply; the answer is right (3,157). |
| 72 | PFE FY2019, narrative, Item 1A | `evidence_not_in_visible_text` | What inspection classification did the FDA assign to Pfizer's McPherson site following the July-August 2018 inspection, and what are the potential regulatory consequences of this status? | Correct by the rule, borderline: both fragments are verbatim and in order, but the one after the ellipsis starts with a period (batch 4's position 85). |
| 72 | PFE FY2019, narrative, Item 1A | `copied_from_passage` | Why has Pfizer experienced shortages of products from the legacy Hospira portfolio, and what progress has the company made in remediating these issues? | Correct by the rule, borderline: an 8-word run of the sentence. |
| 73 | TSLA FY2025, narrative, Item 8 | `evidence_not_in_visible_text` | What were the net cash flows from investing activities for Tesla, Inc. for fiscal 2025, fiscal 2024, and fiscal 2023? | Correct: the row is cut after its 2025 value, and the answer fills 2024 and 2023 with the operating cash-flow figures of the row above (section 7). |
| 74 | WMT FY2023, narrative, Item 8 | `missing_company` | For the consolidated securities class actions related to opioids, what period did the amended complaint revise the putative class of investors to? | Correct: the question does not name the company. |
| 77 | PFE FY2022, table, Item 8 | `missing_company` | What was the total net equity for Haleon/the Consumer Healthcare JV as of September 30, 2022? | Correct: the question does not name the company. |
| 77 | PFE FY2022, table, Item 8 | `missing_company` | How much gross profit did the Consumer Healthcare JV report for the twelve months ending September 30, 2021? | Correct: the question does not name the company. |
| 78 | TSLA FY2019, table, Item 8 | `copied_from_passage` | Why was goodwill not deductible for U.S. income tax purposes in Tesla, Inc.'s fiscal 2019 10-K? | Correct by the rule, borderline: an 8-word run of the sentence. |
| 87 | PFE FY2025, table, Item 8 | `missing_company` | How much were the reclassification adjustments for gains or losses included in net income related to derivative financial instruments for fiscal 2024, in millions? | Correct: the question does not name the company. |
| 88 | TSLA FY2020, narrative, Item 8 | `missing_company` | What was the fair value of the purchase consideration for the Maxwell Technologies acquisition completed in May 2019? | Correct: the question does not name the company. |
| 91 | JPM FY2023, narrative, Operational Risk Management | `copied_from_passage` | What specific responsibilities does JPMorgan Chase & Co's global Legal function hold for managing the Firm's exposure to legal risk during fiscal 2023? | Correct: copies a 9-word run of the sentence. |
| 91 | JPM FY2023, narrative, Operational Risk Management | `evidence_not_in_visible_text` | Who does the General Counsel of JPMorgan Chase & Co report to, and which specific committees does this officer serve on or advise according to the fiscal 2023 filing? | Correct by the rule, borderline: both fragments are verbatim and in order, but the one after the ellipsis starts with a period. |
| 92 | PFE FY2024, table, Item 8 | `number_not_in_passage` | How many stock options were exercised by Pfizer during fiscal 2024? | Correct by the rule, borderline: 4,000 shares is a correct conversion of the table's (4) thousand, but 4,000 is not in the visible text. |
| 93 | TSLA FY2025, narrative, Item 1 | `missing_company` | Which new AI training cluster is currently being constructed at Gigafactory Texas? | Correct: the question does not name the company. |
| 96 | JPM FY2019, table, Consolidated Results Of Operations | `missing_company` | How much Firmwide legal expense was included in the 'Other' noncompensation expense category for the year ended December 31, 2019? | Correct: the question does not name the company. |
| 98 | TSLA FY2019, table, Item 8 | `missing_company` | How many shares were reserved and available for issuance under the 2019 Equity Incentive Plan as of December 31, 2019? | Correct: the question does not name the company. |
| 98 | TSLA FY2019, table, Item 8 | `missing_company` | What was the weighted-average exercise price of the stock options balance at December 31, 2019? | Correct: the question does not name the company. |
| 99 | WMT FY2024, narrative, Item 8 | `missing_company` | What was the status of the defendants' motion to dismiss filed in November 2022 regarding the amended complaint in the securities class action? | Correct: the question does not name the company. |

## 5. Pair building: evidence-leakage exclusions (39)

`build-pairs` excludes a question whose evidence, quote or labelled figure is held exactly by:
- a blocked passage: `evaluation_evidence`;
- the visible text of a passage in another split: `cross_split_evidence`.

These rules fail closed by design, and every exclusion below follows them.

| Where the evidence is | Batch 5 | Batch 4 | Batch 3 | Batch 2 | Batch 1 |
|---|---:|---:|---:|---:|---:|
| An evaluation-pool or audit passage | 11 | 10 | 4 | 6 | 6 |
| An overlap-group copy of an evaluation passage | 7 | 8 | 5 | 14 | 11 |
| Only passages blocked for reusing evaluation text | 17 | 10 | 18 | 7 | 16 |
| The visible text of a validation passage | 4 | 7 | 1 | 8 | 6 |
| **All** | **39** (24.5% of 159) | **35** (20.8%) | **28** (18.2%) | **35** (22.0%) | **39** (25.8%) |

- **By company:**
  - Tesla lost 13 of its 32 passing questions, the most in any batch. Its statement lines (revenues, R&D, current
    assets, cash flows, taxes) recur in blocked passages.
  - ExxonMobil lost 10 of 37, Walmart 8 of 32, JPMorgan 5 of 31 and Pfizer 3 of 27.
- **Cross-split:** the 4 validation-passage holders (JPMorgan's position 41, Pfizer's 47, ExxonMobil's 95 and 100) keep
  the training set and `validation_pairs_v4` apart. Seven evaluation exclusions also have a validation copy.
- **The rule is unchanged** (decision E5).

| Pos | Source | Question (verbatim) | Answer (verbatim) | Rule | Where the evidence is |
|---:|---|---|---|---|---|
| 7 | PFE FY2022, table, Item 8 | What were the total acquisition-related and restructuring costs reported by Pfizer for the year ended December 31, 2022, expressed in millions? | $1,375 million | `evaluation_evidence` | an evaluation-pool or audit passage (6 blocked) |
| 8 | TSLA FY2020, narrative, Item 8 | What was the total amount of current liabilities reported by Tesla, Inc. as of December 31, 2020? | $14,248 million | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 8 | TSLA FY2020, narrative, Item 8 | How much did Tesla, Inc.'s total assets increase from December 31, 2019 to December 31, 2020? | Total assets were $52,148 million in 2020 compared to $34,309 million in 2019 | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 10 | XOM FY2024, table, Business Results | How much did industry refining margins decrease Exxon Mobil's Energy Products earnings in fiscal 2024, and what was the primary driver of this decline? | Earnings decreased by $6,280 million, driven by increased supply from industry capacity additions outpacing record global demand. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 10 | XOM FY2024, table, Business Results | What was Exxon Mobil's United States refinery throughput in thousands of barrels daily for fiscal 2024? | 1,865 thousands of barrels daily | `evaluation_evidence` | an evaluation-pool or audit passage (2 blocked) |
| 14 | WMT FY2023, narrative, Item 8 | What is Walmart Inc.'s ability to estimate potential financial losses related to the Asda Equal Value Claims in fiscal 2023? | The company cannot reasonably estimate any loss or range of loss that may arise related to these proceedings. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 23 | TSLA FY2024, narrative, Item 8 | What was Tesla's total cost of revenues for the year ended December 31, 2024? | 80,240 | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 23 | TSLA FY2024, narrative, Item 8 | How much did Tesla report for research and development expenses in fiscal 2023? | 3,969 | `evaluation_evidence` | an evaluation-pool or audit passage (5 blocked) |
| 24 | WMT FY2024, narrative, Item 1A | What amount of liability did Walmart Inc. accrue in connection with its opioid-related legal settlement framework, and how much of that amount was paid during fiscal 2024? | Approximately $3.3 billion, with almost all of it paid in fiscal 2024 | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked; also 1 validation) |
| 28 | TSLA FY2020, table, Item 8 | What was the total amount of accrued liabilities and other current liabilities reported by Tesla for the fiscal year ended December 31, 2020? | $3,855 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 28 | TSLA FY2020, table, Item 8 | How did Tesla account for proceeds from government incentives related to Gigafactory Shanghai on its consolidated balance sheets? | As a reduction to property, plant and equipment, net | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 29 | WMT FY2025, table, Item 8 | What was Walmart U.S.'s total assets in fiscal 2025? | $150,006 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 31 | JPM FY2020, table, Business Segment Results | How did JPMorgan Chase & Co describe the primary reason for changing its line of business capital allocations as of January 1, 2021? | changes in exposures for each LOB and an increase in the relative risk weighting toward Standardized RWA | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 32 | PFE FY2020, narrative, Item 7 | How did Pfizer describe the operational status of its global manufacturing sites during the 2020 pandemic period? | All manufacturing sites globally continued to operate at or near normal levels. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (4 blocked) |
| 34 | WMT FY2026, table, Item 8 | What was Walmart's total provision for income taxes in fiscal 2026? | $7,199 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 39 | WMT FY2021, narrative, Item 8 | What is the cap on Walmart Inc.'s indemnification obligation for the Asda equal pay claims following the sale of Asda in fiscal 2021? | up to a contractually determined amount | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 39 | WMT FY2021, narrative, Item 8 | For which MDL case number is Walmart Inc. named as a defendant in the consolidated multidistrict litigation concerning opioid abuse impacts? | MDL No. 2804 | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked; also 1 validation) |
| 41 | JPM FY2025, narrative, Business Segment & Corporate Results | How does JPMorgan Chase define a 5-star rating for mutual funds and ETFs in its Asset Management segment? | It represents the top 10% of industry-wide ranked funds | `cross_split_evidence` | the visible text of 1 validation passage |
| 43 | TSLA FY2024, table, Item 8 | What was Tesla's total provision for income taxes in fiscal 2024, and how did it compare to the provision in fiscal 2023? | $1,837 million in 2024 and $(5,001) million in 2023 | `evaluation_evidence` | only passages blocked for reusing evaluation text (7 blocked) |
| 47 | PFE FY2021, table, Item 1 | According to Pfizer's fiscal 2021 patent expiration table, in what year does the U.S. basic product patent for Eliquis expire? | 2026 | `cross_split_evidence` | the visible text of 1 validation passage |
| 51 | JPM FY2024, table, Business Segment & Corporate Results | What was JPMorgan Chase & Co's total net revenue for the Consumer & Community Banking segment in fiscal 2024? | $71,507 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 53 | TSLA FY2025, narrative, Item 8 | What were Tesla's total revenues for fiscal 2025, fiscal 2024, and fiscal 2023? | 94,827 million in fiscal 2025, 97,690 million in fiscal 2024, and 96,773 million in fiscal 2023 | `evaluation_evidence` | an evaluation-pool or audit passage (8 blocked) |
| 53 | TSLA FY2025, narrative, Item 8 | How much did Tesla report for research and development expenses in the year ended December 31, 2025? | 6,411 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 58 | TSLA FY2019, table, Item 8 | What was Tesla's net loss for the year ended December 31, 2019, as reported in the consolidated statements of cash flows? | $775 million | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 60 | XOM FY2021, table, 19. Income And Other Taxes | What was Exxon Mobil's total income tax expense (benefit) for the U.S. segment in fiscal 2021, in millions of dollars? | $1,602 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 63 | TSLA FY2024, table, Item 8 | What was Tesla's basic net income per share of common stock for the fiscal year ended December 31, 2024? | $2.23 | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 63 | TSLA FY2024, table, Item 8 | How much comprehensive income was attributable to noncontrolling interests and redeemable noncontrolling interests in subsidiaries for Tesla during the year ended December 31, 2023? | $(23) million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 68 | TSLA FY2020, narrative, Item 8 | What was Tesla's net cash provided by operating activities in fiscal 2020 compared to fiscal 2019? | 5,943 in fiscal 2020 and 2,405 in fiscal 2019 | `evaluation_evidence` | an evaluation-pool or audit passage (5 blocked) |
| 79 | WMT FY2024, narrative, Item 1 | What percentage of U.S. salaried management positions in store, club, and supply chain roles at Walmart Inc. were filled by individuals who began their careers in hourly positions during fiscal 2024? | 75% | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked; also 2 validation) |
| 81 | JPM FY2024, table, Consolidated Results Of Operations | What was JPMorgan Chase's total net revenue for the year ended December 31, 2024? | $177,556 million | `evaluation_evidence` | an evaluation-pool or audit passage (4 blocked) |
| 81 | JPM FY2024, table, Consolidated Results Of Operations | How much operating lease income did JPMorgan Chase include in other income for the year ended December 31, 2023? | $2.8 billion | `evaluation_evidence` | an evaluation-pool or audit passage (4 blocked; also 1 validation) |
| 85 | XOM FY2025, table, Business Results | What was the impact on Exxon Mobil's fiscal 2025 earnings from structural cost savings, and how much did expenses decrease earnings in the same period? | Structural cost savings increased earnings by $0.6 billion, while expenses decreased earnings by $0.5 billion. | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked; also 4 validation) |
| 85 | XOM FY2025, table, Business Results | According to the operational results table, what was Exxon Mobil's refinery throughput in the United States in thousands of barrels daily for fiscal 2025? | 1,927 thousands of barrels daily | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 89 | WMT FY2025, table, Item 8 | How much did Walmart report as its total income before income taxes for the fiscal year ended January 31, 2024? | $21,848 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (4 blocked) |
| 90 | XOM FY2022, table, Business Results | How did ExxonMobil's production volumes in the Permian region change in 2022 compared to the previous year, and what was the average daily rate? | Production averaged about 550 thousand oil-equivalent barrels per day, approximately 90 koebd higher than the previous year | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 95 | XOM FY2021, table, Consolidated Statement Of Cash Flows | What was the amount of net cash provided by operating activities for Exxon Mobil Corp in fiscal 2021? | $48,129 million | `evaluation_evidence` | an evaluation-pool or audit passage (6 blocked; also 2 validation) |
| 95 | XOM FY2021, table, Consolidated Statement Of Cash Flows | How much did Exxon Mobil Corp report for additions to property, plant and equipment in fiscal 2021? | ($12,076) million | `cross_split_evidence` | the visible text of 1 validation passage |
| 100 | XOM FY2020, table, Consolidated Statement Of Cash Flows | What was Exxon Mobil Corporation's net cash provided by operating activities in fiscal year 2020, millions of dollars? | 14,668 million dollars | `evaluation_evidence` | an evaluation-pool or audit passage (6 blocked; also 1 validation) |
| 100 | XOM FY2020, table, Consolidated Statement Of Cash Flows | How much did Exxon Mobil Corporation report for depreciation and depletion, including impairments, in fiscal year 2019, millions of dollars? | 18,998 million dollars | `cross_split_evidence` | the visible text of 2 validation passages |

## 6. AI review of the 120 built pairs

**How each pair was judged:** as in batches 1–4, each pair was read against its positive passage's full visible text.
- **Accept:** correct company, period, figure and unit; supported by the quote; a useful financial retrieval query.
- **Revise:** a specific error that a reworded question would fix.
- **Reject:** no financial content, or no real answer.

The full visible text of every passage was displayed (batch 4's display cut one passage short).

**Precedents, applied here:**
- a part given as the whole, or a figure given to the wrong business, is revised (batch 3's pairs 106–123; batch 4's
  pairs 154 and 419);
- a question naming no year is revised when the answer changes between filings (batch 3's pair 215; batch 4's pair
  248);
- a split-adjusted or recast figure asked without its filing is revised (batch 3's pair 236);
- a single-segment measure asked as "JPMorgan Chase's" is accepted when the Firm reports that segment's figure (batch
  4's pair 100).

**Corpus look-ups were used for:**
- pairs 416–417: the Sam's Club heading of Walmart's FY2020 MD&A;
- pair 157: JPMorgan's Multifamily exposure in its FY2021 and FY2022 filings;
- pairs 516–517: the year blocks of ExxonMobil's FY2019 capitalized-cost table;
- pair 217: Pfizer's FY2020 regulatory table;
- pair 226: Pfizer's 2019 net income in its FY2020 and FY2021 filings;
- pair 337: Tesla's Maxwell exchange ratio in its FY2019 and FY2020 filings;
- pairs 175 and 508: JPMorgan's capital-allocation and Walmart's debt fair-value methods across filings;
- pair 202: the WRDM description, which appears in the FY2019 filing only.

**Result:** 106 accept (88.3%), 14 revise, 0 reject.

| Aspect (120 pairs) | Pass | Minor | Fail | n/a |
|---|---:|---:|---:|---:|
| Factual accuracy | 102 | 4 | 14 (103, 157, 175, 193, 217, 226, 337, 416, 417, 499, 508, 516, 517, 600) | – |
| Financial relevance | 116 | 4 | 0 | – |
| Company | 120 | – | – | – |
| Period | 98 | 14 (names no year) | 8 (157, 175, 226, 337, 499, 508, 516, 517) | – |
| Figures (answer figures in the quote) | 85 | – | – | 35 |
| Units | 63 | 22 (no unit) | 0 | 35 |

**Issues noted on accepted pairs** (one pair can carry two):
- **Boilerplate:** 17, mostly risk-management and governance statements repeated yearly.
- **Inaccurate wording:** 4. A pending reform called "completed" (278). An expectation asked as an outcome (513). A
  geographic split called a "segment" (552). An investments line without its advances (566).
- **Weak financial relevance:** 4: an officer's trading plan, an undisclosed indemnity cap, human-capital programs and a
  litigation-governance step.
- **No issue:** 81.

Batch 4: 13, 10, 1 and 90. Batch 3: 22, 7, 11 and 72.

**Not accepted (14):**

| Pair | Pos | Decision | Reason |
|---:|---:|---|---|
| 103 | 61 | Revise | Wrong premise and an empty answer: the bullet is one of the drivers of CIB's increase, not the offset, and the answer only repeats the question's words ('the favorable impact'). |
| 157 | 36 | Revise | Wrong year: the Multifamily row follows the table's 'December 31, 2021' header (the flattened cells keep the 2022 labels); 84% is the December 31, 2021 figure (as the FY2021 filing shows), while December 31, 2022's is 82% (the preceding chunk). |
| 175 | 51 | Revise | Names no year, and the answer changes between filings: Standardized RWA and the GSIB surcharge (with a stress simulation) is the FY2023-FY2024 methodology; the FY2022 filing's also incorporates Basel III Advanced RWA (batch 3 precedent, pair 215). |
| 193 | 41 | Revise | Wrong scope: the provision of $97 million is the Asset & Wealth Management segment's (the paragraph's noninterest expense of $15.3 billion, up 6%, is AWM's; AWM's 2024 expense was $14,414 million); Global Private Bank is one of AWM's two businesses, not a segment. |
| 217 | 67 | Revise | Column not determinable: the flattened row holds one value with no column, so 'U.S.' is not supported; the row above gives Braftovi's U.S. approval as April 2020 and Japan's as November 2020. |
| 226 | 82 | Revise | Recast figure without its filing: $16,026 million is the FY2021 filing's recast 2019 figure; the FY2020 filing reports 2019 GAAP net income attributable to Pfizer of $16,273 million (batch 4's position 67; batch 3 precedent, pair 236). No unit (millions). |
| 337 | 88 | Revise | Split-adjusted without saying so: 0.0965 is the exchange ratio as adjusted for the 2020 stock split (the FY2020 filing); the FY2019 filing gives the merger's ratio as 0.0193 (batch 3 precedent, pair 236). |
| 416 | 19 | Revise | Wrong scope: the discussion is the Sam's Club segment's (membership income, club closures, Plus Member penetration; its heading is in the preceding chunk); the 11-basis-point decrease is Sam's Club's gross profit rate, not Walmart's. |
| 417 | 19 | Revise | Wrong scope: $122 million is the Sam's Club segment's operating income increase (same discussion as pair 416); the question names no segment. |
| 499 | 69 | Revise | Misreads the footnote: 5,743 is Walmart International's total including Canada; the footnote says the non-Canada counts are as of December 31, 2025 (Canada's as of January 31, 2026), not that Canada is excluded. |
| 508 | 94 | Revise | Names no year, and the answer changes between filings: observable prices of identical instruments is the FY2025-FY2026 method; the FY2020-FY2024 filings base the Level 2 estimate on the Company's current incremental borrowing rate (batch 4's positions 19 and 59; batch 3 precedent, pair 215). |
| 516 | 70 | Revise | Wrong year and scope: 39,173 lies in the 'Consolidated Subsidiaries As of December 31, 2018' block (its heading is before the window; the block that follows is the 2018 equity companies'); the 2019 block is in the preceding chunks (U.S. producing assets 99,405 against 95,532 here). |
| 517 | 70 | Revise | Wrong year and scope: 40,710 is the consolidated subsidiaries' Africa accumulated depreciation and depletion at December 31, 2018 (the same block as pair 516), not the 2019 fiscal year's, and not ExxonMobil's total. |
| 600 | 80 | Revise | Part given as the whole: 1,029,666 is the consolidated subsidiaries' future cash inflows at December 31, 2022 (the block is headed 'As of December 31, 2022 Consolidated Subsidiaries'); the equity companies' are listed separately, so it is not ExxonMobil's total. |

**Patterns:**
- **Scope:** 6, against 13 in batch 4.
  - **Walmart:** 416 and 417 give the Sam's Club segment's gross profit rate and operating income as Walmart's (the
    segment heading is in the preceding chunk).
  - **ExxonMobil:** 516 and 517 (consolidated subsidiaries' 2018 block, asked as ExxonMobil's 2019), and 600
    (consolidated subsidiaries' future cash inflows as the total).
  - **JPMorgan:** 193 (AWM's provision given to Global Private Bank).
- **Period:** 5 more (157, 175, 226, 337, 508), the largest category this batch.
  - **A figure that changes between filings, asked without the filing:** 226 (Pfizer's recast 2019 net income) and 337
    (Tesla's split-adjusted exchange ratio).
  - **A no-year question whose answer changed:** 175 (JPMorgan's capital-allocation methodology) and 508 (Walmart's
    debt fair-value method).
  - **A flattened table that kept the 2022 labels on its 2021 block:** 157 (JPMorgan's Multifamily exposure).
- **Reading:** 3.
  - 103: a wrong premise and an answer that repeats the question.
  - 217: a column the flattened table does not show.
  - 499: a footnote's date read as an exclusion.
- **Flattened tables:** 4 of the 14 (157, 217, 516, 517) come from tables whose flattening loses a block's year or a
  column. This is a new main source of error, ahead of the scope errors.
- **Rejections:** none. No board, age or boilerplate-only question was built.

**JPMorgan's 5 segment-results passages (positions 11, 31, 41, 46, 51):**
- **Accepted:** 3 pairs.
  - 122 and 135, CIB's pre-provision profit, name the segment.
  - 140 asks AWM's assets under management as the Firm's, which the Firm reports.
- **Revised:** 193 (scope within AWM) and 175 (period).
- **No segment figure was given to the whole Firm.**
- **Position 51,** batch 4's pairs 154–155 table:
  - its built pair asks the capital methodology;
  - its other question names Consumer & Community Banking and was excluded as evaluation evidence.
- **Batch 4 by comparison:** 5 of 11 segment pairs revised, all for Firm attribution. JPMorgan's other 15 passages in
  batch 5 gave 21 pairs, 19 accepted.

Every pair, verbatim (pair numbers are those of the 500-record pair set; positions are batch 5's plan positions):

| Pair | Pos | Source | Type | Question (verbatim) | Answer (verbatim) | Decision | Note |
|---:|---:|---|---|---|---|---|---|
| 099 | 26 | JPM FY2019, narrative, Item 1A | risk | What specific malicious objectives are viruses or code designed to achieve when they infiltrate JPMorgan Chase's systems? | Obtain unauthorized access to confidential information, manipulate or destroy data, disrupt or degrade service, or steal money. | Accept | Correct; names no year; a cybersecurity risk factor repeated yearly. |
| 100 | 26 | JPM FY2019, narrative, Item 1A | risk | Why does JPMorgan Chase state that it cannot provide absolute security against cyberattacks? | It may not always be possible to anticipate, detect, or recognize threats, or to implement effective preventive measures against all breaches. | Accept | Correct; names no year; a cybersecurity risk factor repeated yearly. |
| 102 | 61 | JPM FY2019, table, Consolidated Results Of Operations | revenue | What was the approximate amount of the adjustment to the credit card rewards liability included in the prior year that affected Card income for JPMorgan Chase & Co in fiscal 2019? | approximately $330 million | Accept | Correct; the prior year's adjustment, as asked. |
| 103 | 61 | JPM FY2019, table, Consolidated Results Of Operations | financial_table | According to the financial table describing the offset to the net increase in CIB, what was the specific impact of tighter funding spreads on derivatives in Credit Adjustments & Other for JPMorgan Chase & Co in fiscal 2019? | the favorable impact of tighter funding spreads on derivatives in Credit Adjustments & Other | **Revise** | Wrong premise and an empty answer: the bullet is one of the drivers of CIB's increase, not the offset, and the answer only repeats the question's words ('the favorable impact'). |
| 104 | 96 | JPM FY2019, table, Consolidated Results Of Operations | financial_table | What was JPMorgan Chase & Co's compensation expense for the year ended December 31, 2019? | $34,155 million | Accept | Correct; the Firm's consolidated expense. |
| 118 | 66 | JPM FY2020, narrative, Item 1A | risk | What specific capital metrics under Basel III rely on models that require regulatory approval before JPMorgan Chase can use them in fiscal 2020? | market risk RWA, credit risk RWA and operational risk RWA | Accept | Correct; a model-risk factor repeated yearly. |
| 119 | 66 | JPM FY2020, narrative, Item 1A | risk | According to JPMorgan Chase's fiscal 2020 filing, what are four examples of factors that could cause its risk management models to be ineffective? | reliance on historical trends, inherent limitations in forecasting, incomplete historical data, and technology performance issues | Accept | Correct; a model-risk factor repeated yearly. |
| 122 | 31 | JPM FY2020, table, Business Segment Results | financial_table | What was JPMorgan Chase & Co's pre-provision profit for the Corporate & Investment Bank segment in fiscal 2020? | $25,746 million | Accept | Correct; names the CIB segment, and the 2020 CIB column. |
| 135 | 46 | JPM FY2021, table, Business Segment Results | financial_table | What was the pre-provision profit for JPMorgan Chase & Co's Corporate & Investment Bank segment in the year ended December 31, 2021? | $26,424 million | Accept | Correct; names the CIB segment, and the 2021 CIB column. |
| 140 | 11 | JPM FY2021, table, Business Segment Results | mdna | According to the management discussion, what drove the 15% increase in assets under management for JPMorgan Chase & Co in fiscal 2021? | Cumulative net inflows and the impact of higher market levels | Accept | Correct; the Firm's assets under management are AWM's, so the attribution holds (batch 4 precedent for single-segment measures, pair 100). |
| 144 | 86 | JPM FY2021, narrative, Country Risk Management | accounting_policy | How does JPMorgan Chase & Co. measure securities financing exposures for its fiscal 2021 country risk management purposes? | At their receivable balance, net of eligible collateral received | Accept | Correct; a country-risk measurement convention repeated yearly. |
| 145 | 86 | JPM FY2021, narrative, Country Risk Management | risk | What is the primary objective of the stress testing component within JPMorgan Chase & Co.'s country risk management framework as of December 31, 2021? | To estimate and limit losses arising from a country crisis by measuring the impact of adverse asset price movements | Accept | Correct; a country-risk stress-testing objective repeated yearly. |
| 147 | 71 | JPM FY2022, table, Consolidated Results Of Operations | financial_table | What was JPMorgan Chase's compensation expense for the year ended December 31, 2022? | $41,636 million | Accept | Correct; the Firm's consolidated expense. |
| 148 | 71 | JPM FY2022, table, Consolidated Results Of Operations | risk | How much did JPMorgan Chase increase its consumer allowance for credit losses in 2022, and what were the primary drivers cited? | $1.2 billion, driven by Card Services with higher outstanding balances and a deteriorating macroeconomic outlook | Accept | Correct; the Firm's consolidated allowance discussion. |
| 156 | 36 | JPM FY2022, narrative, Credit And Investment Risk Management | financial_table | What was the total credit exposure for JPMorgan Chase & Co's real estate portfolio as of December 31, 2022? | $170,857 million | Accept | Correct; the December 31, 2022 total (it precedes the table's 2021 block). |
| 157 | 36 | JPM FY2022, narrative, Credit And Investment Risk Management | risk | What percentage of the Multifamily real estate credit exposure classified as investment-grade for JPMorgan Chase & Co at year-end 2022? | 84% | **Revise** | Wrong year: the Multifamily row follows the table's 'December 31, 2021' header (the flattened cells keep the 2022 labels); 84% is the December 31, 2021 figure (as the FY2021 filing shows), while December 31, 2022's is 82% (the preceding chunk). |
| 162 | 1 | JPM FY2022, table, Note 20 – Long-Term Debt | debt_liquidity | What was the total carrying value of JPMorgan Chase's parent company fixed-rate senior debt due after 5 years as of December 31, 2022? | $108,924 million | Accept | Correct; the parent company's fixed-rate senior debt due after 5 years. |
| 163 | 1 | JPM FY2022, table, Note 20 – Long-Term Debt | accounting_policy | How does JPMorgan Chase account for changes in the fair value of its equity-linked long-term debt instruments? | Changes in fair value are recorded in principal transactions revenue in the Consolidated statements of income, except for unrealized gains/(losses) due to DVA which are recorded in OCI. | Accept | Correct; names no year; a fair-value policy repeated yearly (the detector's R2 flag on 'which are' is a false positive). |
| 171 | 21 | JPM FY2023, table, Credit And Investment Risk Management | financial_table | What was the central case assumption for the U.S. unemployment rate in the fourth quarter of 2024, as reported by JPMorgan Chase & Co for the period ended December 31, 2023? | 4.4% | Accept | Correct; the central case at December 31, 2023 for 4Q24. |
| 172 | 56 | JPM FY2023, narrative, Operational Risk Management | risk | What specific governance framework does JPMorgan Chase & Co use to maintain oversight and coordination of its compliance risk? | The CCOR Management Framework | Accept | Correct; names no year; compliance governance repeated yearly. |
| 173 | 56 | JPM FY2023, narrative, Operational Risk Management | other | How frequently are JPMorgan Chase & Co employees required to affirm their compliance with the Firm's Code of Conduct? | Annually | Accept | Correct; names no year; a Code of Conduct requirement repeated yearly. |
| 175 | 51 | JPM FY2024, table, Business Segment & Corporate Results | accounting_policy | Which capital regulatory framework and surcharge are incorporated into JPMorgan Chase & Co's current equity allocation methodology? | Basel III Standardized risk-weighted assets ("RWA") and the global systemically important banks ("GSIB") surcharge | **Revise** | Names no year, and the answer changes between filings: Standardized RWA and the GSIB surcharge (with a stress simulation) is the FY2023-FY2024 methodology; the FY2022 filing's also incorporates Basel III Advanced RWA (batch 3 precedent, pair 215). |
| 184 | 16 | JPM FY2024, narrative, Market Risk Management | mdna | What was the impact on reported sensitivities for JPMorgan Chase & Co in fiscal 2024 from the update in the fourth quarter of 2024 for a -200 basis point parallel shift in U.S. dollar interest rates? | 1.3 | Accept | Correct; the fourth-quarter update for a -200 bp parallel shift; no unit (the table's unit is before the window). |
| 193 | 41 | JPM FY2025, narrative, Business Segment & Corporate Results | profitability | What was the provision for credit losses for JPMorgan Chase's Global Private Bank segment in fiscal 2025, and what was the primary driver for this amount? | $97 million, largely driven by the impact of a charge-off related to a client-specific exposure in the third quarter of 2025 | **Revise** | Wrong scope: the provision of $97 million is the Asset & Wealth Management segment's (the paragraph's noninterest expense of $15.3 billion, up 6%, is AWM's; AWM's 2024 expense was $14,414 million); Global Private Bank is one of AWM's two businesses, not a segment. |
| 198 | 76 | JPM FY2025, narrative, Operational Risk Management | risk | How does JPMorgan Chase & CO describe the scope of mandatory security training for its global workforce in fiscal 2025? | The training is mandatory for all employees globally on a periodic basis and is supplemented by Firmwide testing initiatives, including periodic phishing tests, alongside specialized training for specific roles like application developers. | Accept | Correct; security training governance repeated yearly. |
| 199 | 76 | JPM FY2025, narrative, Operational Risk Management | risk | What specific control requirements does the Technology Resiliency function establish for JPMorgan Chase & CO in fiscal 2025 to handle service degradations or outages? | It establishes control requirements for planning and testing the prioritized recovery of technology services, including incident response planning, data backup and retention, and recovery readiness. | Accept | Correct; technology resiliency governance repeated yearly. |
| 202 | 2 | PFE FY2019, narrative, Item 1 | other | How does Pfizer organize its Research Units within the WRDM organization for its Biopharma business? | They are organized by therapeutic area to enhance flexibility, cohesiveness and focus. | Accept | Correct; names no year; the WRDM description appears in the FY2019 filing only. |
| 207 | 37 | PFE FY2019, narrative, Item 1A | risk | Which specific biologic product did Pfizer identify as facing ongoing biosimilar competition in most European markets in its fiscal 2019 risk factors? | Enbrel | Accept | Correct; a biosimilar risk factor. |
| 208 | 37 | PFE FY2019, narrative, Item 1A | risk | What factors did Pfizer cite in fiscal 2019 that could lead to lower uptake of biosimilars? | Anti-competitive practices, physician reluctance to prescribe biosimilars for existing patients taking the originator product, or misaligned financial incentives | Accept | Correct. |
| 215 | 32 | PFE FY2020, narrative, Item 7 | mdna | When did Pfizer restart recruitment across its development portfolio, including new study starts, following the pandemic-related pauses in fiscal 2020? | Late-April 2020 | Accept | Correct. |
| 216 | 67 | PFE FY2020, table, Item 7 | financial_table | When did the FDA grant Emergency Use Authorization for Pfizer's PF-07302048 vaccine in fiscal 2020? | December 2020 | Accept | Correct. |
| 217 | 67 | PFE FY2020, table, Item 7 | financial_table | What was the status of the U.S. filing for Braftovi (encorafenib) in combination with Mektovi (binimetinib) for second or third-line BRAFV600E-mutant mCRC according to Pfizer's fiscal 2020 filing? | Approved Nov. 2020 | **Revise** | Column not determinable: the flattened row holds one value with no column, so 'U.S.' is not supported; the row above gives Braftovi's U.S. approval as April 2020 and Japan's as November 2020. |
| 223 | 47 | PFE FY2021, table, Item 1 | financial_table | What is the basic product patent expiration year for the Japan market for the drug Sutent, as reported by Pfizer for fiscal 2021? | 2024 | Accept | Correct. |
| 226 | 82 | PFE FY2021, narrative, Item 7 | profitability | What was the net income attributable to Pfizer Inc. common shareholders on a GAAP reported basis for fiscal 2019? | $16,026 | **Revise** | Recast figure without its filing: $16,026 million is the FY2021 filing's recast 2019 figure; the FY2020 filing reports 2019 GAAP net income attributable to Pfizer of $16,273 million (batch 4's position 67; batch 3 precedent, pair 236). No unit (millions). |
| 227 | 82 | PFE FY2021, narrative, Item 7 | accounting_policy | How much did purchase accounting adjustments reduce the amortization of intangible assets for Pfizer in fiscal 2019? | (4,158) | Accept | Correct; the 2019 purchase-accounting column of the FY2021 filing's reconciliation; no unit (millions). |
| 229 | 12 | PFE FY2021, narrative, Item 7 | mdna | What was the primary driver for the change in investing activities from continuing operations for Pfizer in the year ended December 31, 2021, compared to the prior year? | A $24.7 billion increase in purchases of short-term investments with original maturities of greater than three months. | Accept | Correct. |
| 230 | 12 | PFE FY2021, narrative, Item 7 | debt_liquidity | How much cash did Pfizer use to acquire Array, net of cash acquired, in fiscal 2019? | $10.9 billion | Accept | Correct; the 2019 Array acquisition, named by year. |
| 254 | 7 | PFE FY2022, table, Item 8 | other | How much of the total costs incurred under Pfizer's productivity program from its start through December 31, 2022, was attributed to Biopharma? | $1.4 billion | Accept | Correct; the program's Biopharma share through 2022. |
| 262 | 52 | PFE FY2023, narrative, Item 1 | risk | According to Pfizer's fiscal 2023 Form 10-K, what is the status of the Oslo Medicines Initiative (OMI) as of September 2022, and what organization was subsequently established by WHO/Europe? | The OMI concluded its work in September 2022, and the WHO/Europe Access to Novel Medicines Platform was established to enhance affordable and equitable access to effective, innovative and high-priced medicinal products in the region. | Accept | Correct; a pricing-policy development. |
| 263 | 52 | PFE FY2023, narrative, Item 1 | risk | In the context of Pfizer's fiscal 2023 Form 10-K, how does the centralized VBP program in China aim to contain healthcare costs? | It aims to contain healthcare costs by driving utilization of generics that have passed QCE, resulting in further lowering the price of medicines. | Accept | Correct; a China pricing statement. |
| 271 | 97 | PFE FY2023, narrative, Item 8 | accounting_policy | How did Pfizer account for the Blackstone research and development funding in its fiscal 2023 financial statements? | The funding is recognized as an obligation to perform contractual services and recorded as a reduction of Research and development expenses using an attribution model over the period of the related expenses. | Accept | Correct. |
| 272 | 97 | PFE FY2023, narrative, Item 8 | profitability | What was the amount of the reduction to Pfizer's research and development expenses in 2023 related to the Blackstone arrangement? | $175 million | Accept | Correct. |
| 278 | 62 | PFE FY2024, narrative, Item 1 | risk | What is Pfizer's assessment of the potential business impact of the completed EU pharmaceutical legislation reform? | Certain provisions of the reform could potentially have an adverse impact on business. | Accept | Correct; the reform was still pending ('once approved'), not completed. |
| 289 | 92 | PFE FY2024, table, Item 8 | financial_table | What was the aggregate intrinsic value of Pfizer's outstanding TSRUs as of December 31, 2024? | $122 million | Accept | Correct. |
| 291 | 22 | PFE FY2024, narrative, Item 8 | other | How many operating segments did Pfizer manage its commercial operations through in fiscal 2024, and what were their names? | Three operating segments: Biopharma, PC1, and Pfizer Ignite | Accept | Correct; the 2024 segment structure. |
| 292 | 22 | PFE FY2024, narrative, Item 8 | other | Who is identified as the Chief Operating Decision Maker (CODM) for Pfizer in the fiscal 2024 reporting period? | The Chairman and Chief Executive Officer | Accept | Correct; a governance fact of the segment note. |
| 298 | 87 | PFE FY2025, table, Item 8 | financial_table | What was Pfizer's net income before allocation to noncontrolling interests for the year ended December 31, 2025, in millions? | $7,812 million | Accept | Correct. |
| 301 | 27 | PFE FY2025, table, Item 8 | financial_table | What was the fair value of Pfizer's contingent consideration liabilities at the end of fiscal 2025? | $1,790 million | Accept | Correct. |
| 302 | 27 | PFE FY2025, table, Item 8 | debt_liquidity | What was the estimated fair value of Pfizer's long-term debt, excluding the current portion, as of December 31, 2024? | $54 billion | Accept | Correct; the prior-year figure, named by date. |
| 307 | 57 | PFE FY2025, table, Item 8 | profitability | What was the total share-based payment expense for Pfizer Inc. for fiscal 2025? | $799 million | Accept | Correct. |
| 315 | 38 | TSLA FY2019, table, Item 8 | financial_table | What was Tesla's total current assets balance in millions for the year ended December 31, 2019? | $12,103 million | Accept | Correct; a year-end balance asked 'for the year ended'. |
| 316 | 58 | TSLA FY2019, table, Item 8 | financial_table | How much did Tesla report for depreciation, amortization and impairment in the year ended December 31, 2018? | $1,901 million | Accept | Correct; the 2018 column. |
| 317 | 18 | TSLA FY2019, table, Item 8 | financial_table | What was the value of the property, plant and equipment, net balance for Tesla as of January 1, 2019, after the adjustments from the adoption of the new lease standard? | $9,713 million | Accept | Correct; the January 1, 2019 adoption balance. |
| 318 | 18 | TSLA FY2019, table, Item 8 | accounting_policy | How did Tesla determine the present value of lease payments for leases that did not provide an implicit rate of return during the adoption of the new lease standard? | Tesla used its incremental borrowing rate based on the information available at the adoption date. | Accept | Correct; the lease standard's adoption. |
| 319 | 78 | TSLA FY2019, table, Item 8 | financial_table | What was the total purchase price for the Maxwell acquisition reported by Tesla, Inc. for fiscal 2019? | $207 million | Accept | Correct; the 2019 Maxwell acquisition. |
| 334 | 48 | TSLA FY2020, narrative, Item 8 | financial_table | What was the change in accounts payable and accrued liabilities for Tesla, Inc. during the fiscal year ended December 31, 2020? | 2,102 | Accept | Correct; no unit (millions). |
| 335 | 48 | TSLA FY2020, narrative, Item 8 | financial_table | How much did inventory levels change for Tesla, Inc. in the year ended December 31, 2019? | (429) | Accept | Correct; the 2019 column; no unit (millions). |
| 337 | 88 | TSLA FY2020, narrative, Item 8 | other | What was the exchange ratio used to convert Maxwell common stock into Tesla common stock as part of the 2019 acquisition? | 0.0965 shares of Tesla common stock per Maxwell share | **Revise** | Split-adjusted without saying so: 0.0965 is the exchange ratio as adjusted for the 2020 stock split (the FY2020 filing); the FY2019 filing gives the merger's ratio as 0.0193 (batch 3 precedent, pair 236). |
| 378 | 83 | TSLA FY2024, table, Item 8 | financial_table | How many potentially dilutive stock-based awards, in millions, were excluded from Tesla's computation of diluted net income per share for the year ended December 31, 2024 because they were anti-dilutive? | 15 million | Accept | Correct. |
| 379 | 83 | TSLA FY2024, table, Item 8 | debt_liquidity | What specific types of cash does Tesla identify as comprising its restricted cash balances? | Cash held to service secured debt, collateral for sales to lease partners with a resale value guarantee, and deposits for insurance services | Accept | Correct; names no year; a restricted-cash policy repeated yearly. |
| 382 | 3 | TSLA FY2024, table, Item 8 | financial_table | What was Tesla's total finance lease expense in millions for the fiscal year ended December 31, 2024? | $430 million | Accept | Correct. |
| 383 | 3 | TSLA FY2024, table, Item 8 | financial_table | What was the weighted-average remaining lease term for Tesla's finance leases as of December 31, 2023? | 2.3 years | Accept | Correct; the prior-year column. |
| 386 | 43 | TSLA FY2024, table, Item 8 | tax | According to the tax rate reconciliation, what was the impact of GILTI and Subpart F inclusion on Tesla's tax provision for the year ended December 31, 2024? | $882 million | Accept | Correct; the tax reconciliation, in millions. |
| 389 | 93 | TSLA FY2025, narrative, Item 1 | other | When did Tesla begin initial customer deliveries for its new residential retrofit solar panel? | January 2026 | Accept | Correct. |
| 390 | 33 | TSLA FY2025, narrative, Item 1 | risk | What specific regulatory obligations does Tesla identify as potentially being imposed by additional layers of European Union frameworks for autonomous vehicle systems? | Risk management, data-governance, trainings, and technical documentation | Accept | Correct; names no year; an FY2025 regulatory statement. |
| 391 | 33 | TSLA FY2025, narrative, Item 1 | risk | How does Tesla describe the impact of legal restrictions in U.S. states and cities on the operation of autonomous vehicle ride-hailing fleets? | It increases the legal complexity with respect to autonomous vehicles and an autonomous vehicle ride-hailing business | Accept | Correct; names no year; a regulatory risk statement. |
| 401 | 73 | TSLA FY2025, narrative, Item 8 | financial_table | How much did Tesla, Inc. spend on purchases of investments in fiscal 2025 compared to fiscal 2024? | Purchases of investments were $(37,109) million in fiscal 2025 and $(35,955) million in fiscal 2024. | Accept | Correct (the detector's R4 flag is a false positive). |
| 407 | 13 | TSLA FY2025, narrative, Item 9B | other | What was the expiration date of the Rule 10b5-1 trading arrangement adopted by Tesla's Chief Financial Officer in November 2025? | January 31, 2027 | Accept | Correct; an officer's Rule 10b5-1 plan. |
| 408 | 13 | TSLA FY2025, narrative, Item 9B | other | How much did Tesla agree to invest in xAI's Series E Preferred Stock in January 2026? | approximately $2 billion | Accept | Correct; the January 2026 xAI investment. |
| 409 | 84 | WMT FY2020, narrative, Item 3 | risk | How many cases were pending in the National Prescription Opiate Litigation MDL as of March 6, 2020, according to Walmart's fiscal 2020 disclosure? | Over 2,000 cases | Accept | Correct; the MDL count as of March 6, 2020 (the same template as batch 4's pair 354, for March 5, 2021). |
| 410 | 84 | WMT FY2020, narrative, Item 3 | risk | What was the issuance date of the most recent grand jury subpoena received by Walmart regarding its money transfer services in fiscal 2020? | January 2020 | Accept | Correct. |
| 416 | 19 | WMT FY2020, narrative, Item 7 | profitability | What were the primary drivers behind the 11 basis point decrease in Walmart's gross profit rate for fiscal 2020? | Investments in price and higher eCommerce fulfillment costs, partially offset by reduced tobacco sales which have lower margins. | **Revise** | Wrong scope: the discussion is the Sam's Club segment's (membership income, club closures, Plus Member penetration; its heading is in the preceding chunk); the 11-basis-point decrease is Sam's Club's gross profit rate, not Walmart's. |
| 417 | 19 | WMT FY2020, narrative, Item 7 | mdna | How much did segment operating income increase for Walmart in fiscal 2020 compared to the previous fiscal year? | $122 million | **Revise** | Wrong scope: $122 million is the Sam's Club segment's operating income increase (same discussion as pair 416); the question names no segment. |
| 436 | 4 | WMT FY2021, table, Item 8 | debt_liquidity | What was the total repayment of matured debt for Walmart Inc. during fiscal 2021, in millions? | $5,382 million | Accept | Correct. |
| 437 | 4 | WMT FY2021, table, Item 8 | financial_table | How much operating lease cost did Walmart Inc. recognize in its Consolidated Statement of Income for the fiscal year ended January 31, 2021? | $2,626 million | Accept | Correct. |
| 446 | 44 | WMT FY2022, narrative, Item 8 | financial_table | What was the amount of net cash provided by operating activities for Walmart Inc. in the fiscal year ended January 31, 2022? | 24,181 | Accept | Correct; no unit (millions). |
| 447 | 44 | WMT FY2022, narrative, Item 8 | financial_table | How much did Walmart Inc. pay for property and equipment during the fiscal year ended January 31, 2021? | (10,264) | Accept | Correct; fiscal 2021 (the year ended January 31, 2021); no unit (millions). |
| 451 | 9 | WMT FY2022, narrative, Item 8 | risk | In Walmart Inc.'s fiscal 2022 10-K, what was the outcome of the liability phase trial in the opioid multidistrict litigation that concluded in November 2021? | The jury found in favor of the plaintiffs regarding the liability of all defendants, including the Company. | Accept | Correct; the November 23, 2021 liability verdict. |
| 452 | 9 | WMT FY2022, narrative, Item 8 | risk | According to Walmart Inc.'s fiscal 2022 filing, when was the abatement phase of the opioid litigation trial scheduled to begin? | May 10, 2022 | Accept | Correct. |
| 460 | 49 | WMT FY2023, narrative, Item 8 | debt_liquidity | How much of the remaining opioid settlement liability did Walmart record in accrued liabilities as of January 31, 2023? | $2.9 billion | Accept | Correct. |
| 461 | 74 | WMT FY2023, narrative, Item 8 | risk | In Walmart's fiscal 2023 filing, what was the status of the Company's motion to dismiss the DOJ's amended complaint regarding opioid litigation as of the filing date? | The motion remains pending. | Accept | Correct; the DOJ civil action's status in the FY2023 filing. |
| 462 | 14 | WMT FY2023, narrative, Item 8 | risk | Up to what limit has Walmart Inc. agreed to provide indemnification for certain Asda Equal Value Claims? | Up to a contractually determined amount. | Accept | Correct; the passage gives no figure, only that the indemnity is capped at a contractually determined amount. |
| 463 | 79 | WMT FY2024, narrative, Item 1 | other | Which specific programs does Walmart Inc. utilize in the U.S. to facilitate career advancement and educational opportunities for its associates during fiscal 2024? | Walmart Academy and Live Better U (LBU) | Accept | Correct; human-capital programs, not Walmart's finances. |
| 464 | 54 | WMT FY2024, narrative, Item 1A | risk | What specific operational consequences does Walmart warn could result from failing to adequately predict customer demand or optimize its distribution and fulfillment centers in fiscal 2024? | Excess or insufficient inventory, service interruptions, and increased costs. | Accept | Correct; a fulfillment risk factor (the same statement as batch 1's pair 408, from another filing). |
| 465 | 54 | WMT FY2024, narrative, Item 1A | risk | According to Walmart's risk factors for fiscal 2024, what factors are identified as being of primary importance to customers and increasing in significance due to digital tools and social media? | Price transparency, assortment of products, customer experience, convenience, ease and the speed and cost of shipping. | Accept | Correct; a consumer-trend risk factor. |
| 478 | 99 | WMT FY2024, narrative, Item 8 | risk | What specific authority did the Walmart Board of Directors grant to the independent Special Litigation Committee in May 2023? | The Board has authorized the SLC to retain independent legal counsel and such other advisors as the SLC deems appropriate in carrying out its duties. | Accept | Correct; a litigation-governance step. |
| 489 | 64 | WMT FY2025, table, Item 8 | accounting_policy | What were the weighted-average discount rates for the dividend yield used to calculate the fair value of performance-based restricted stock units for Walmart in fiscal 2025, 2024, and 2023? | 3.2% for fiscal 2025, 3.3% for fiscal 2024, and 3.3% for fiscal 2023 | Accept | Correct; all three years. |
| 492 | 89 | WMT FY2025, table, Item 8 | debt_liquidity | What was the fair value of Walmart's long-term debt, including amounts due within one year, as of January 31, 2025? | $33,790 million | Accept | Correct. |
| 498 | 69 | WMT FY2026, table, Item 2 | financial_table | What was the total retail unit count for Walmart U.S. Supercenters as of January 31, 2026? | 3,566 | Accept | Correct; no unit (a count). |
| 499 | 69 | WMT FY2026, table, Item 2 | financial_table | How many total retail units did Walmart International operate as of January 31, 2026, excluding Canada? | 5,743 | **Revise** | Misreads the footnote: 5,743 is Walmart International's total including Canada; the footnote says the non-Canada counts are as of December 31, 2025 (Canada's as of January 31, 2026), not that Canada is excluded. |
| 507 | 94 | WMT FY2026, table, Item 8 | debt_liquidity | What was Walmart's long-term debt carrying value and fair value as of January 31, 2026? | Carrying value of $38,166 million and fair value of $36,777 million | Accept | Correct. |
| 508 | 94 | WMT FY2026, table, Item 8 | accounting_policy | How did Walmart estimate the fair value of its long-term debt, and what inputs were used? | Using Level 2 inputs based on observable prices of identical instruments in less active markets | **Revise** | Names no year, and the answer changes between filings: observable prices of identical instruments is the FY2025-FY2026 method; the FY2020-FY2024 filings base the Level 2 estimate on the Company's current incremental borrowing rate (batch 4's positions 19 and 59; batch 3 precedent, pair 215). |
| 509 | 34 | WMT FY2026, table, Item 8 | tax | How much cash did Walmart pay in income taxes to Mexico during the fiscal year ended January 31, 2026? | $1,285 million | Accept | Correct; fiscal 2026 cash taxes paid to Mexico. |
| 512 | 35 | XOM FY2019, narrative, 2. Accounting Changes | accounting_policy | What was the amount of the operating lease right of use asset recorded by Exxon Mobil Corp at the adoption of Topic 842 on January 1, 2019? | $4.3 billion | Accept | Correct. |
| 513 | 35 | XOM FY2019, narrative, 2. Accounting Changes | accounting_policy | What was the impact of adopting the Financial Instruments – Credit Losses (Topic 326) update on Exxon Mobil Corp's financial statements as of January 1, 2020? | Not expected to have a material impact | Accept | Correct; the FY2019 filing's expectation for a standard adopted after it, asked as if it were the outcome. |
| 516 | 70 | XOM FY2019, narrative, Supplemental Information On Oil And Gas  | financial_table | What was the total capitalized cost for producing assets in the Asia region for Exxon Mobil as of December 31, 2019? | 39,173 | **Revise** | Wrong year and scope: 39,173 lies in the 'Consolidated Subsidiaries As of December 31, 2018' block (its heading is before the window; the block that follows is the 2018 equity companies'); the 2019 block is in the preceding chunks (U.S. producing assets 99,405 against 95,532 here). |
| 517 | 70 | XOM FY2019, narrative, Supplemental Information On Oil And Gas  | financial_table | How much accumulated depreciation and depletion did Exxon Mobil report for the Africa region in the 2019 fiscal year? | 40,710 | **Revise** | Wrong year and scope: 40,710 is the consolidated subsidiaries' Africa accumulated depreciation and depletion at December 31, 2018 (the same block as pair 516), not the 2019 fiscal year's, and not ExxonMobil's total. |
| 522 | 30 | XOM FY2020, table, Item 2 | financial_table | What was ExxonMobil's share of refining capacity at its Baytown, Texas facility as of year-end 2020, and what percentage interest did the company hold? | 561 KBD, with a 100% interest | Accept | Correct; ExxonMobil's share and interest. |
| 523 | 30 | XOM FY2020, table, Item 2 | financial_table | According to ExxonMobil's 2020 10-K, what was the total refining capacity in thousands of barrels per day for its United States operations? | 1,764 KBD | Accept | Correct; ExxonMobil's share for the United States. |
| 528 | 65 | XOM FY2020, table, 3. Miscellaneous Financial Information | financial_table | What was the total value of crude oil inventories held by Exxon Mobil at the end of fiscal 2020? | $5,354 million | Accept | Correct; the crude oil line of the inventory table. |
| 529 | 65 | XOM FY2020, table, 3. Miscellaneous Financial Information | other | How much before-tax goodwill impairment did Exxon Mobil recognize in its Upstream, Downstream, and Chemical reporting units during 2020? | $611 million | Accept | Correct; the 2020 goodwill impairment. |
| 550 | 25 | XOM FY2021, table, 11. Leases | financial_table | What was the weighted average discount rate for Exxon Mobil's operating leases as of December 31, 2021? | 2.3% | Accept | Correct. |
| 551 | 25 | XOM FY2021, table, 11. Leases | debt_liquidity | How much of Exxon Mobil's total lease liability in 2021 was attributed to drilling rigs and related equipment operating leases? | $935 million | Accept | Correct; the footnote to the operating-lease total. |
| 552 | 60 | XOM FY2021, table, 19. Income And Other Taxes | tax | How much did Exxon Mobil report for other taxes and duties included in production and manufacturing expenses for the Non-U.S. segment in fiscal 2020, in millions of dollars? | $663 million | Accept | Correct; the 2020 non-U.S. column ('segment' for a geographic split). |
| 560 | 90 | XOM FY2022, table, Business Results | financial_table | What was ExxonMobil's total Upstream Earnings (loss) (U.S. GAAP) in millions of dollars for fiscal 2022? | 36,479 million dollars | Accept | Correct; the Upstream segment's total, as asked. |
| 563 | 20 | XOM FY2022, table, 6. Additional Working Capital Informatio | debt_liquidity | What was the weighted-average interest rate on Exxon Mobil Corp's short-term borrowings outstanding as of December 31, 2022? | 1.5 percent | Accept | Correct. |
| 566 | 55 | XOM FY2022, table, 8. Investments, Advances And Long-Term R | financial_table | What was the balance of equity method company investments for Exxon Mobil Corp as of December 31, 2022? | $34,522 million | Accept | Correct; the Investments line, excluding advances (the total with advances is $42,571 million). |
| 567 | 55 | XOM FY2022, table, 8. Investments, Advances And Long-Term R | financial_table | What was the total amount of long-term receivables and miscellaneous, net of reserves, for Exxon Mobil Corp as of December 31, 2021? | $5,506 million | Accept | Correct; the prior-year column, net of reserves. |
| 573 | 5 | XOM FY2023, table, Item 5 | financial_table | How many shares did ExxonMobil purchase in November 2023 as part of its publicly announced share repurchase program? | 21,626,648 shares | Accept | Correct; the shares bought under the program, not the total. |
| 574 | 5 | XOM FY2023, table, Item 5 | financial_table | What was the average price per share paid by ExxonMobil for equity securities purchased in December 2023? | $101.06 | Accept | Correct. |
| 577 | 40 | XOM FY2023, table, Consolidated Statement Of Comprehensive  | financial_table | What was ExxonMobil's total comprehensive income attributable to the company for the year ended December 31, 2023, in millions of dollars? | 37,291 million dollars | Accept | Correct. |
| 578 | 40 | XOM FY2023, table, Consolidated Statement Of Comprehensive  | financial_table | How much did foreign exchange translation adjustments contribute to other comprehensive income for ExxonMobil in fiscal 2021, in millions of dollars? | (872) million dollars | Accept | Correct; the 2021 column. |
| 579 | 75 | XOM FY2023, table, 19. Income And Other Taxes | financial_table | What was Exxon Mobil's total net deferred tax liability in fiscal 2023? | $20,815 million | Accept | Correct. |
| 580 | 75 | XOM FY2023, table, 19. Income And Other Taxes | accounting_policy | Why did Exxon Mobil not record deferred income taxes on undistributed earnings from foreign subsidiaries as of December 31, 2023? | The earnings are expected to be indefinitely reinvested for the foreseeable future. | Accept | Correct; an indefinite-reinvestment policy repeated yearly. |
| 590 | 45 | XOM FY2024, table, Business Results | financial_table | What was ExxonMobil's United States upstream earnings for fiscal 2024, expressed in millions of dollars? | 6,426 million dollars | Accept | Correct; the Upstream segment's U.S. earnings, as asked. |
| 591 | 45 | XOM FY2024, table, Business Results | other | How many thousand oil-equivalent barrels per day did ExxonMobil produce in the Permian region on average in 2024? | approximately 1,185 thousand oil-equivalent barrels per day | Accept | Correct; the Permian's 2024 production, as asked. |
| 600 | 80 | XOM FY2024, table, Supplemental Information On Oil And Gas  | financial_table | What was the total future cash inflows from sales of oil and gas for Exxon Mobil Corp in millions of dollars as of December 31, 2022? | 1,029,666 million dollars | **Revise** | Part given as the whole: 1,029,666 is the consolidated subsidiaries' future cash inflows at December 31, 2022 (the block is headed 'As of December 31, 2022 Consolidated Subsidiaries'); the equity companies' are listed separately, so it is not ExxonMobil's total. |
| 601 | 80 | XOM FY2024, table, Supplemental Information On Oil And Gas  | accounting_policy | According to Exxon Mobil's 10-K, what discount factor is applied to net proved reserves to compute the standardized measure of discounted future net cash flows? | 10 percent | Accept | Correct; names no year; the standardized measure's prescribed discount factor. |
| 602 | 15 | XOM FY2025, narrative, Item 2 | other | What was the net production volume from ExxonMobil's Canadian operations in fiscal 2025? | 519 thousand oil-equivalent barrels per day | Accept | Correct; total operations in Canada, net. |
| 606 | 50 | XOM FY2025, table, Consolidated Statement Of Comprehensive  | financial_table | What was the comprehensive income attributable to ExxonMobil in fiscal 2025, in millions of dollars? | 32,600 million dollars | Accept | Correct. |

## 7. Special investigation: answers unsupported by the visible passage

**What was looked for:** an answer built from the model's reasoning or knowledge rather than from the visible passage
(batch 2's pair 032 is the known case), and any other answer the visible text does not support.

**The checks:**
1. **The detector** (rules R1–R4, unchanged) over every recorded proposal of batches 1–5: 996 answers.
2. **The looser grounding rule.**
3. **The flagged cut-off passages** from batch 5's plan check: positions 3, 11, 17, 43, 44, 46, 58 and 89, read first
   in the review.
4. **The review of all 120 pairs** (§6), with full visible text.
5. **A question-word scan** of the 106 included questions (pair 190's kind).

**The detector's batch 5 flags, each inspected by hand:**

| Position | Rules | Outcome | Verdict |
|---|---|---|---|
| 16 (JPMorgan FY2024, market risk) | R1 | rejected (`question_format`) | **Genuine.** "The passage does not provide a specific value for non-U.S. dollar flatter yield curve in Q2 2024, but for U.S. dollar it was (1.0)." The non-U.S. flatter-curve rows lie past the window. The validator rejected it by chance, through an unrelated rule. |
| 73 (Tesla FY2025, cash flows) | R4 (6 of 10) | rejected (`evidence_not_in_visible_text`) | **Genuine, of a different kind:** not meta-reasoning but invented cells. The investing row is cut after its 2025 value, and the answer gives 2024 and 2023 as $(14,923) and $(13,256) million: the operating cash flows of the row above. Its quote claims those cells, so the quote check rejected it, by design. |
| 1 (pair 163) | R2 | accepted | False positive: "which are" matches the question pattern in a verbatim answer. |
| 53 | R4 (6 of 6) | excluded as evaluation evidence | False positive: "fiscal" and "million" paraphrase the table; the figures are in the quote. |
| 73 (pair 401) | R4 (4 of 6) | accepted | False positive: the same paraphrase; both figures are in the quote. |

**Results:**
- **Batch 5 has 1 reasoning answer and 1 invented-cell answer among its 200 proposals. Neither reached a pair.**
  - Across batches 1–5: 6 reasoning answers in 997 proposals (1, 1, 0, 3 and 1 by batch), plus this invented-cell
    case.
  - Two of the 7 were rejected by a rule that bears on them: batch 4's position 39 and this batch's position 73,
    through the quote check. Four were rejected by unrelated rules, and batch 2's pair 032 by the review.
- **R1 (meta-language)** still matches exactly the 6 reasoning answers among the 996 answers and nothing else. The
  invented-cell answer has no meta-language, and only the quote check catches it.
- **The looser rule** flags 6 batch 5 answers: the 2 genuine cases and 4 paraphrases. Across batches 1–5 it flags 27:
  the 7 genuine cases and 20 paraphrases.
- **The cut-off scan does not predict them.**
  - It flagged 26 batch 5 passages. None produced a reasoning or invented-cell answer. The pairs of the eight read
    first (382–383, 140, 386, 446–447, 135, 316, 492) ask complete rows and were accepted.
  - Both batch 5 cases came from passages the scan did not flag. Position 16 ends inside a table whose closing tag is
    cut. Position 73 ends on a row whose last cell is a year.
  - The common setting is narrower than "ends mid-table": the question asks for exactly the part of a row or table
    that the window cuts off.
- **Position 17** (Pfizer FY2023, ending on "Balance,", the setting of pair 032) gave no unsupported answer. One
  proposal's answer is wrong ($672 million is a restructuring amount, not acquisition costs) and was rejected. The
  other was rejected for a missing company name.
- **Questions:** the question-word scan flags 9 included questions, all paraphrase. Two expand an abbreviation that
  the passage leaves short, both correctly: 216 writes out EUA as "Emergency Use Authorization", and 523 KBD as
  "thousands of barrels". No included question carries a wrong outside fact (pair 190's kind).

**Conclusion:**
- The failure stays rare, and the review remains the protection: no unsupported answer has reached the included data
  in five batches.
- **For batch 6:**
  - run the detector and the question scan on every proposal;
  - read first the answers to questions about rows the window cuts.
- **After P1:** an R1 rejection, and a rule that checks each quoted cell against the visible row, are the two
  candidate validator rules.

## 8. Evidence leakage and false-negative protections

**Leakage, on the 106 included pairs (all from the trial build, §11):**

| Check | Result |
|---|---|
| Evidence rule, re-checked at the candidate build | 0 leak passages; every included pair passes |
| Lexical overlap with the permitted evaluation questions (word Jaccard; reject at 0.8) | Highest 0.310 (batch 4: 0.345; batch 3: 0.263) |
| Semantic screen (`screen-questions --pairs --candidates train_candidates_v2`, in scratch) | 681 questions (the 615 pairs and the 66) against the 10 permitted question texts, `query_id` and text only. **0 flagged at 0.90.** Of batch 5's 120: highest 0.7908 (pair 184). Of its 106 included: highest 0.7908, 0 at or above 0.80, 4 at or above 0.75, 19 at or above 0.70; median 0.653. |
| **Every question in the 0.80–0.90 review band** (approved) | **2 across batches 1–5, none from batch 5:** batch 1's pair 005 (019 in the 500-record numbering), 0.8017 against `q_0002`, reviewed and accepted as a different fact in the batch 1 audit; and batch 4's pair 154 (176 here), 0.8014 against `q_0004`, reviewed as a different fact and revised for scope in the batch 4 audit. No evaluation question text was read for this audit. The threshold was not changed. |
| Cross-split | 4 excluded at `build-pairs`. `load_validation_set(validation_pairs_v4, training=…)` passes against all 610 trial training pairs. |
| Duplicates | None by normalised question; one near-identical same-fact pair, kept apart by the guard (§9) |

**Evidence within the model's window:**
- **The validator** located every accepted quote inside the tokenizer-visible part (512 tokens, special tokens
  included).
- **The independent re-check:** 104 of the 106 included quotes lie verbatim in the positive's visible text.
  - Pair 119 elides with "…".
  - Pair 447 joins a row label to its fiscal 2021 cell.
- **The window's end:** the latest quote ends at token 512, the window's last; 4 end after token 500 (batch 4: 5).
- **The window's limits:**
  - Pairs 416–417 lack the segment heading that lies before the window.
  - The two unsupported proposals ask for text past it (§7).

**False negatives:**
- **Equivalents:** 56 of the 106 record exact equivalents (123 passages), and 53 record uncertain ones (205 passages).
  The trainer never uses any of them as a negative.
- **Conflicts:** `NegativeGuard` finds 697 conflicting pairs among the 610 trial pairs, and the planner keeps them in
  different batches.
  - Involving batch 5: batch 5 with batch 5, 53; with batch 1, 22; with batch 2, 31; with batch 3, 28; with batch 4,
    49; with the 66, 13.
  - 90 of the 106 are in at least one conflict.

  | Batch size | Batches | Full | Mean fill (batch 4 trial) |
  |---:|---:|---:|---:|
  | 16 | 39 | 38 | 0.978 (0.984) |
  | 32 | 20 | 18 | 0.953 (0.927) |
  | 64 | 11 | 9 | 0.867 (0.875) |

- **An independent probe:** for each included pair, any other included pair's positive was checked for the answer's
  figure wherever the guard would still allow it as a negative.
  - 7 hits, all chance matches in other companies' passages: 4,158, "2,000" cases, 3,566 and 1,285.
  - No unrecorded equivalent was found.
- **Known gaps remain:** paraphrased evidence, rounded figures and changed row labels.

## 9. Duplicate questions and repeated financial facts

| Against | Same normalised question | Highest word Jaccard | Cosine ≥ 0.90 (BGE, zero-shot) |
|---|---:|---:|---|
| Within the 106 | 0 | 0.812 | 2 (highest 0.964): JPMorgan's compensation expense in 2019 against 2022; Walmart's long-term debt fair value at January 2025 against 2026 |
| Batch 1's 101 | 0 | **0.952** | 2 (highest **0.994**): Walmart's FY2022 operating cash flow, the same fact (below); Walmart's fulfillment risk statement, the same statement in another filing (below) |
| Batch 2's 115 | 0 | 0.800 | 5 (highest 0.943): Tesla's current assets in 2019 against 2022, JPMorgan's real estate exposure against its total credit portfolio, Pfizer's net income in 2025 against 2021, ExxonMobil's long-term receivables in 2021 against 2020, Walmart's operating against investing cash flow |
| Batch 3's 108 | 0 | 0.789 | 3 (highest 0.966): Walmart's debt repayments in fiscal 2021 against 2022, Tesla's current assets in 2019 against 2021, Walmart's operating cash flow in fiscal 2022 against 2026 |
| Batch 4's 114 | 0 | 0.800 | 3 (highest 0.954): Walmart's long-term debt fair value at January 2025 against 2024, Tesla's current assets in 2019 against 2023, ExxonMobil's equity-method investments in 2022 against 2025 |
| The 66 of `train_candidates_v2` | 0 | 0.750 | 1 (0.908): ExxonMobil's comprehensive income in 2025 against net income in 2022 |
| `validation_pairs_v4` (93) | 0 | 0.560 | 1 (0.921): Tesla's current assets in 2019 against 2025 |
| Earlier train generation (v1, v2) | 0 | 0.750 | 2 (highest 0.908): the same ExxonMobil question as the 66, and Walmart's operating against financing cash flow |

**Repeated financial facts:** every batch 5 pair was checked against every other trial candidate of the same company.
There are 18 neighbours, 14 of them in `NegativeGuard` conflict.
- **Two ask the same fact twice, from different passages:**
  - **Pair 446** and **batch 1's pair 092** (444 in the 500-record numbering) both ask Walmart's fiscal 2022 operating
    cash flow, $24,181 million, in near-identical words (cosine 0.994). The plan check had flagged position 44 as an
    exact equivalent.
  - **Pair 464** (FY2024 filing) and **batch 1's pair 101** (496; FY2026 filing) both ask the consequences of failing to
    predict customer demand, the same risk statement.
  - Both are in conflict: never in one batch, never each other's negative. Each positive holds the fact, so both pairs
    are valid and kept, as for batches 3–4's repeats.
- **Same template, same answer, another date:** pair 409 (MDL cases "over 2,000" as of March 6, 2020) and batch 4's
  pair 354 (427; as of March 5, 2021). In conflict.
- **A coincidence:** pair 317 shares the figure 9,713 with batch 4's pair 286 (344), for a different fact. In conflict
  anyway.
- **The other 14** ask the same template for another year, measure or line. 10 are in conflict; the 4 that are not
  are different facts.

**Walmart's opioid litigation:**
- **Batch 5's 6 included pairs ask new facts:** the 2021 liability verdict, the abatement date, the $2.9 billion
  remaining liability, the DOJ motion, the Special Litigation Committee's authority, and the 2020 MDL count.
- **Only the MDL count repeats a template** (above).
- **Across the 610,** 13 of Walmart's 101 candidates (12.9%) concern this one litigation, spread over FY2020–FY2024.
  That is a topic concentration, not a leakage or duplicate problem. Batch 6's Walmart passages (FY2020, FY2024–FY2026)
  may add more.

**ExxonMobil's oil-equivalent conversion:** no batch 5 pair asks it. The 610 hold 2 such candidates (batch 3's FY2021
and batch 4's FY2023), in conflict.

## 10. Company, fiscal-year and numerical accuracy, and distribution

**Accuracy:**
- **Company:** every included question names its positive's company (106 of 106; all 120 passed the validator's
  company rule).
- **Period:**
  - 87 included questions name the filing's fiscal year or the prior year.
  - 5 name only another year, each legitimately:
    - fiscal 2019 in the FY2021 filing's three-year reconciliation (227);
    - the 2019 Array acquisition (230);
    - the January 2026 xAI investment, a subsequent event in the FY2025 filing (408);
    - the January 1, 2020 adoption in the FY2019 filing (513);
    - fiscal 2021 in the FY2023 three-year statement (578).
  - 14 name no year. They cover facts repeated yearly or stated in one filing only (the WRDM description, the FY2025
    autonomous-vehicle regulations, the January 2026 solar deliveries).
  - The 8 period errors are revised (§6).
- **Numbers:**
  - Every figure in the 74 numeric included answers is in its quote (independent check).
  - Units: 58 state one, 16 give a bare figure from a table in millions, and none states a wrong unit.

**Distribution by company:**

| | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|
| Planned | 20 | 20 | 20 | 20 | 20 |
| Pass validator | 31 | 27 | 32 | 32 | 37 |
| Evidence exclusions | 5 | 3 | 13 | 8 | 10 |
| Pairs built | 26 | 24 | 19 | 24 | 27 |
| **Included** | **22** | **22** | **18** | **20** | **24** |
| Batch 4 included | 20 | 25 | 24 | 22 | 23 |
| Batch 3 included | 15 | 12 | 18 | 23 | 18 |

| The 610 trial candidates | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidates | 63 | 43 | 97 | 104 | 104 | 101 | 98 |
| Share | 10.3% | 7.0% | 15.9% | 17.0% | 17.0% | 16.6% | 16.1% |

**Distribution by fiscal year:**

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Planned | 13 | 15 | 11 | 11 | 12 | 18 | 17 | 3 |
| Included | 14 | 13 | 13 | 14 | 16 | 17 | 16 | 3 |
| The 610 together | 78 | 75 | 91 | 88 | 80 | 90 | 94 | 14 |

**Within companies (the 610):**

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TSLA | 18 | 13 | 13 | 10 | 12 | 16 | 22 | – |
| WMT | – | 13 | 18 | 16 | 10 | 15 | 15 | 14 |
| JPM | 18 | 13 | 12 | 16 | 12 | 12 | 14 | – |
| PFE | 15 | 6 | 15 | 18 | 16 | 18 | 16 | – |
| XOM | 7 | 17 | 17 | 16 | 15 | 14 | 12 | – |

- **Passage kind:** 56 table and 50 narrative pairs (54 and 46 planned).
- **Question type:**
  - `financial_table` 39, `risk` 25, `other` 12, `debt_liquidity` 10, `accounting_policy` 10;
  - `mdna` 4, `tax` 3, `profitability` 2, `revenue` 1.

## 11. Provenance and consistency of the included pairs

**The trial build (scratch copy only):**
- **Sources:**
  - `pilot_v3` and `pilot_v2` with their existing decisions;
  - P1's 615 train pairs (batches 1–5), with a scratch decisions file. It holds batches 1–4's 495 decisions, re-keyed
    by `passage_id` and `question_sha256` to the 500-record pair IDs, and this batch's 120;
  - the scratch screening record.
- **Result:** 610 candidates (51 + 15 + 101 + 115 + 108 + 114 + 106).
- **Re-keying:** all 495 earlier decisions found their pair (112, 121, 100 and 133 of them under an ID other than
  their file's), and the committed files were not touched.
- **The 162 validation pairs** that `build-pairs` also builds have no decision and are excluded, as before.

**Provenance on all 106 included records:** the same 11-field block as batches 1–4, identical to batch 4's. It names:
- corpus `ff9dadf3…4004`;
- preparation `bge_base_ft_v3`, `21c2293a…`;
- provider `groq`, model `qwen/qwen3.8-27b`;
- prompt `sec-qgen-v2`, `f42bdc54…`;
- parameters `42d5fe54…`;
- validator `sec-qcheck-v3`;
- origin `build-pairs`.

No field is missing or empty.

**Checks run on the trial set:**
- `verify_candidate_provenance` traces all 610 back to their records and decisions.
- `load_candidates`, the trainer's fail-closed loader, loads all 610.
- `load_validation_set` loads all 93 pairs of `validation_pairs_v4` apart from the 610.
- **The 66 are unchanged:** a rebuild of `train_candidates_v2`'s sources in scratch gives byte-identical files,
  `908d4c79…` and `c8d38f96…`.

## 12. Batch 5 against batches 1–4

| | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Batch 5 |
|---|---:|---:|---:|---:|---:|
| Done / failed | 99 / 1 | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 |
| Tokens per planned passage | 1,483 | 1,467 | 1,463 | 1,449 | 1,465 |
| Pass the validator | 151 (76.3%) | 159 (79.5%) | 154 (77.4%) | 168 (84.0%) | 159 (79.5%) |
| False validator rejections | 3 (1.5%) | 4 (2.0%) | 4 (2.0%) | 4 (2.0%) | 3 (1.5%) |
| Evidence exclusions | 39 (25.8%) | 35 (22.0%) | 28 (18.2%) | 35 (20.8%) | 39 (24.5%) |
| Pairs built | 112 | 124 | 126 | 133 | 120 |
| Accept / revise / reject | 101 / 6 / 5 | 115 / 6 / 3 | 108 / 14 / 4 | 114 / 17 / 2 | 106 / 14 / 0 |
| **Acceptance** | **90.2%** | **92.7%** | **85.7%** | **85.7%** | **88.3%** |
| **Included per planned passage** | **1.01** | **1.15** | **1.08** | **1.14** | **1.06** |
| Passages with no included pair | 30 | 20 | 23 | 23 | 27 |
| Scope errors | 2 | 1 | 5 | 13 | 6 |
| Period errors | 3 | 3 | 3 | 2 | 8 |
| Unsupported answers among the proposals | 1 | 1 | 0 | 3 | 2 (all rejected before review except batch 2's) |
| Highest semantic similarity (all pairs) | 0.8017 | 0.7929 | 0.7922 | 0.8014 | 0.7908 |
| Cosine ≥ 0.90 within the batch | 2 | 4 | 3 | 0 | 2 |
| Company range of included pairs | 10–17 | 13–21 | 5–23 | 20–25 | 18–24 |
| HTTP 429 / 503 responses | 0 / 0 | 6 / 0 | 3 / 3 | 3 / 1 | 5 / 0 |

**Systematic differences:**
- **Fewer scope errors, more period errors.** Batch 5 drew fewer JPMorgan segment tables (5 against 9) and ExxonMobil
  reserve tables. It drew more multi-year statements whose figures are recast, split-adjusted or flattened with the
  wrong labels.
- **No rejections:** no age, board or boilerplate-only questions.
- **The generator is unchanged:** the same false-rejection patterns, tokens per passage within 1.2% of batch 1, and
  completion lengths as before (mean 249).

Nothing in batch 5 suggests a drift in the generator. The review catches its errors, and the gate absorbs them.

## 13. Reassessment for batch 6: supply, concentration, budget and benefit

**Sequencing (option A, approved):**
- The decisions file is bound to the pair set that the 500 records build, and the scratch replay confirmed it.
- Batch 6, if run, appends records and renumbers the pair IDs again. At the end of P1, the real `build-pairs` runs
  once and all batches' decisions are carried over by `passage_id` and `question_sha256`, which is mechanical (verified
  here for all 495 decisions of batches 1–4).

**Supply** (scratch plans with the frozen options, excluding all five batches' groups):

| Remaining eligible overlap groups | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| After batch 5 | 0 | 0 | 624 | 170 | 31 | 22 | 54 |
| After batch 6 under the frozen options | 0 | 0 | 604 | 150 | 11 | 2 | 34 |

- **Tesla:** supply in only three years: FY2019 24, FY2020 1, FY2025 6.
- **Walmart:** four years: FY2020 4, FY2024 1, FY2025 9, FY2026 8.

**Batch 6 options** (deterministic scratch plans):

| Option | Passages | Tesla's years | Walmart's years | Visible tokens | Tokens (estimate) | Candidates added (about) |
|---|---:|---|---|---:|---:|---:|
| (a) frozen, 100 | 20 per company | FY2019 13, FY2020 1, FY2025 6 | FY2020 4, FY2024 1, FY2025 7, FY2026 8 | 49,160 | 146,200 | 108–110 |
| (c) frozen, 60 | 12 per company | FY2019 5, FY2020 1, FY2025 6 | FY2020 4, FY2024 1, FY2025 4, FY2026 3 | 29,546 | 87,800 | 65 |
| (b) stop at batch 5 | – | – | – | – | 0 | 0 |

- **The other three companies:** under (a), JPMorgan, Pfizer and ExxonMobil keep 2–3 passages per fiscal year.
- **JPMorgan:** (a) draws 2 segment-results passages.
- **ExxonMobil:** (a) draws 4 supplemental oil-and-gas passages, the kind behind batch 4's and 5's ExxonMobil errors.
- **The planner's `--max-per-company-year`** cannot rebalance the skew, because it counts the earlier batches'
  passages (batch 4 audit).

**Projected final candidates:**

| Stop after | Candidates | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch 5 (now) | 610 | 10.3% | 7.0% | 15.9% | 17.0% | 17.0% | 16.6% | 16.1% |
| Batch 6, 60 passages | about 675 | 9.3% | 6.4% | 16.2% | 17.3–17.4% | 17.3–17.4% | 16.9–17.0% | 16.4–16.6% |
| Batch 6, 100 passages | **about 718–720** | 8.8% | 6.0% | 16.3% | 17.4–17.6% | 17.4–17.6% | 17.1–17.3% | 16.6–16.9% |

The ranges apply each company's included-per-passage yield of batches 1–5 or of batches 3–5 (0.95–1.18). The 25% cap
holds in every case. The batch 4 audit projected about 725 at its higher yield.

**Concentration risks:**
- **Company:**
  - no company would exceed about 17.6%;
  - Apple and Amazon stay underweight at 8.8% and 6.0%, against an even 14.3% for seven companies, because their
    supply ran out in batch 3. No option changes that.
- **Fiscal year within a company:**
  - Tesla's FY2019 would rise from 18 of 104 today (17%) to about 32 of about 126 under (a), or about 24 of 117 under
    (c);
  - Walmart's FY2025–FY2026 from 29 of 101 (29%) to about 45 of about 123 (37%) under (a), or about 37 of 114 under
    (c);
  - across all companies, the years stay within 75–94 per full year.
- **Topic:** Walmart's opioid litigation (13 of 101 Walmart candidates) and same-template statement lines (Tesla's
  total current assets are now asked for four year-ends) will grow with more passages from the same filings.
- **Error-prone sources:** JPMorgan segment tables and ExxonMobil supplemental tables. The review catches their
  errors, but they cost yield.

**Budget:**
- About 740,000 tokens were approved for batches 2–6. Batches 2–5 used 584,312, so **155,688 remain**.
- **Batch 6 (a):**
  - about 146,200 (120,900 prompt, 25,300 completion), leaving about 9,500 (6.1%);
  - at batch 1's rate per passage, 148,300, leaving 7,400;
  - **it would exceed the budget only if replies averaged over 348 tokens.** The batches so far averaged 246–269 and
    the longest single reply was 504.
  - If every reply were as long as the longest yet, batch 6 would need about 171,300, so the run needs a stop rule:
    the generator counts requests, not tokens.
- **Batch 6 (c):** about 87,800, leaving about 67,900.
- **No additional contingency is needed** at the expected rate.

**Does batch 6's expected benefit justify its cost?**
- **Option (a): yes, if the planned data size stands (blocker B2).**
  - **For:** about 108 more candidates (+18%), and it completes the planned 600 passages. It makes the five remaining
    companies more even (16.3–17.6%). Its quality should match batches 3–5 (85.7–88.3% acceptance, with no
    unsupported answer reaching the included data so far). Its cost is the approved budget.
  - **Against:** it spends nearly all of the remaining budget. Tesla's FY2019 and Walmart's FY2025–FY2026 gain weight:
    a quarter of Tesla's and over a third of Walmart's pairs. It draws more of the error-prone tables.
- **Option (c)** buys 60% of the gain (about 65 candidates) for 60% of the cost, with less year skew. It falls short
  of the planned size, so B2 would still need relabelling.
- **Option (b)** keeps 610 candidates and the most even year mix. It leaves the planned size unmet and the budget
  unspent.
- **The marginal value:** an extra 108 pairs on a base of 610 is a modest gain for a fine-tune. It matters most if the
  three-seed experiment is to run at the planned size. If the experiment is relabelled as a pilot, (b) or (c) is the
  better use of the remaining budget.

## 14. Recommendation

- **Batch 5 quality gate: PASS** on every approved criterion (§1).
- **Batch 6: GO under option (a)**, on your approval, if the planned data size stands. The conditions:
  1. **One generator:** the same model, prompt, validator, parameters and plan options. Only `--name` changes, to
     `train_v3_p1_batch6`. Write the plan after approval, and verify it, its exclusions and a dry run before any request.
  2. **Option A:** no real `build-pairs` or `build-candidates` until batch 6 is generated and gated.
  3. **A token stop rule:**
     - stop the run if batch 6's cumulative tokens (summed from the run summaries) reach 150,000 before the plan ends,
       so the approved budget cannot be exceeded;
     - run it in two halves (`--max-requests 50`) to check the rate at the midpoint;
     - completed passages stay usable.
  4. **Stop rule:** batch 6 is replayed and gated on a scratch copy before the end-of-P1 build.
  5. **Review focus:**
     - scope in JPMorgan segment and ExxonMobil supplemental tables;
     - year labels in flattened multi-year tables;
     - figures that change between filings;
     - questions about rows the window cuts;
     - Walmart's opioid repeats.
- **If the planned size does not stand,** choose (b) stop at 610, or (c) a 60-passage batch 6.
- **The three-seed experiment: still NO-GO**, until B1–B3 of the readiness audit are resolved.

**Your decisions:**

| # | Decision | Recommendation |
|---|---|---|
| I1 | Batch 6: (a) 100 passages under the frozen options; (b) stop P1 at batch 5 (610 candidates); (c) a 60-passage batch 6 | **(a)** if B2's planned size stands: about 718–720 candidates, within the budget. **(c)** for less year skew at 60% of the gain. **(b)** if the experiment will be relabelled as a pilot. |
| I2 | Budget for batch 6 | No new contingency: about 9,500 tokens (6.1%) remain at the expected rate. Use the token stop rule (condition 3) instead. |
| I3 | Validator weaknesses: 3 false rejections in two known patterns; borderline elided quotes; unsupported answers caught partly by chance; flattened tables that lose a block's year or a column | Keep the validator unchanged through P1. Record an R1 rejection, a quoted-cell check against the visible row, and the ellipsis-period case for a later version. |
| I4 | The semantic review band | Keep the approved 0.90 exclusion and the 0.80–0.90 review. Batch 5 put no question in the band; the band's 2 questions are both reviewed. |

## Appendix: how the results were obtained

Everything below ran in a scratch copy of `data/processed/train/bge_base_ft_v3`. It used a copy of
`configs/train/data_prep_v3.yaml` that differs only in `output_dir`, the venv, and `HF_HUB_OFFLINE=1`.

```bash
python3 scripts/02_generate_queries.py revalidate --config SCRATCH/data_prep_v3_scratch.yaml --split train --dry-run
python3 scripts/02_generate_queries.py build-pairs --config SCRATCH/data_prep_v3_scratch.yaml
python3 scripts/02_generate_queries.py screen-questions --config SCRATCH/data_prep_v3_scratch.yaml \
    --name train_v3_p1_batch5 --pairs --candidates train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_v2check.yaml   # byte-identical to train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_trial.yaml     # pilot_v3, pilot_v2, P1 batches 1-5; semantic_screening
python3 scripts/02_generate_queries.py plan --config SCRATCH/data_prep_v3_scratch.yaml --split train \
    --name cap_after_b5_100 --size 100 --method balanced_years --max-company-share 0.25 --one-per group \
    --exclude-candidate-groups train_candidates_v2 \
    --after-generated configs/train/data_prep_v2.yaml configs/train/data_prep_v1.yaml
    # capacity only; also --size 60 and 80, and --size 901 --max-company-share 1.0 for the whole remaining supply
```

**Other scripts:**
- A trace of every candidate through `build_pairs`' own functions. It matches the written pairs, ID for ID, and
  reproduces the batch 4 audit's trace for batches 1–4.
- The trainer's `load_candidates` and `plan_batches`, `verify_candidate_provenance`, and `load_validation_set` with
  the training pairs.
- The duplicate, repeated-fact, opioid-topic and false-negative probes, and the quote and token-position checks.
- The reasoning detector and its looser variant over all 997 proposals of batches 1–5, and the question-word scan.
- Corpus look-ups for pairs 157, 175, 202, 217, 226, 337, 416–417, 508 and 516–517.

**Evaluation data:** no evaluation file was read except the 10 permitted question texts, through the screen's own
loader and the lexical check, and no evaluation question text was printed: no batch 5 question reached the review band.
No qrels, relevance labels, judged passages, pools or audits were read.
