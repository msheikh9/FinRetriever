# P1 batch 6 (`train_v3_p1_batch6_n60`): generation audit, final quality gate and P1 summary

Status: 2026-11-09.
- **What this is:**
  - an audit of the 60 generation records of plan `train_v3_p1_batch6_n60` in `bge_base_ft_v3`;
  - an AI review (Claude) of the 84 pairs they build;
  - the reasoning-answer investigation, extended to all six batches;
  - the batch quality gate, a comparison of all six batches, and a summary of P1 as a whole, ending with a GO/NO-GO
    for building `train_candidates_v3`.

  It uses the method of the batch 1–5 audits ([batch 5](train_v3_p1_batch5_generation_audit.md),
  [batch 4](train_v3_p1_batch4_generation_audit.md)). It is not human annotation, and no decision here is a gold label.
- **Where the decisions are:** `data/train_review/train_v3_p1_batch6_quality_decisions.jsonl`.
  - 84 rows (68 accept, 13 revise, 3 reject), each `reviewer: ai_review` and `gold: false`.
  - Each row is bound to its `pair_id`, `passage_id` and the SHA-256 of its exact question. The pair IDs are those of
    the pair set that the current 560 train records build (§11).
  - SHA-256 `4d0445c7…09f1`.
- **What was not done:**
  - no question was generated and no API request was sent;
  - no generation record, plan, candidate dataset, validation set or earlier decision was changed. The records of
    batches 1–5 and their decisions are byte-identical (§2);
  - `build-pairs`, `screen-questions` and `build-candidates` were not run in `bge_base_ft_v3`. They ran only in a scratch
    copy of the preparation, which was then discarded. No final candidate dataset was built;
  - the validator (`sec-qcheck-v3`) and the semantic-screen thresholds are unchanged;
  - nothing was trained, committed or pushed.
- **Protected files:** the 353 files under `data/`, `configs/`, `src/` and `scripts/` were checksummed before and after
  the audit. They are unchanged; the decisions file is the only file added.

## 1. Verdict

**Batch 6 passes the approved quality gate on all four criteria, and so does P1 as a whole.**

| # | Criterion | Batch 6 | Verdict | All six batches |
|---|---|---|---|---|
| 1 | At least 0.8 included pairs per planned passage | 68 included for 60 planned passages: **1.13** | **Pass** | 612 for 560: **1.09** |
| 2 | At least 75% AI-review acceptance | 68 of 84 built pairs: **81.0%** | **Pass** | 612 of 699: **87.6%** |
| 3 | Zero accepted leakage violations | **0**: no included pair fails the evidence, cross-split, lexical, semantic or duplicate checks (§8, §9) | **Pass** | **0** |
| 4 | Complete provenance for every included pair | **68 of 68** carry the full provenance block, and the fail-closed provenance check passes (§11) | **Pass** | **612 of 612** (678 of 678 with the 66) |

**The special-attention items:**
- **Tesla FY2019:** 5 passages gave 7 pairs, and 6 are included.
  - All are risk-factor or business narrative, and 4 of the 6 name no year.
  - One was revised: pair 348's answer only restates its question.
  - Tesla's FY2019 now holds 24 of its 117 trial candidates (20.5%, against 17.3% before), as projected. Its FY2025
    has become Tesla's largest year (28, 23.9%), so FY2019 and FY2025 together hold 44% of Tesla's pairs.
- **Walmart FY2025–FY2026:** 7 passages gave 8 pairs, and 7 are included.
  - One was revised: pair 566 gives fiscal 2024's $0.76 dividend as fiscal 2026's.
  - Three more questions were excluded as evaluation evidence.
  - These two years now hold 36 of Walmart's 115 candidates (31.3%, against 28.7%).
- **ExxonMobil supplemental tables:** 3 scope errors in the 6 pairs from supplemental and reserves tables, the same
  kind as batches 4–5. Two give the consolidated subsidiaries' block as ExxonMobil's total (679, 660); one gives
  developed reserves as total proved reserves (622). The other 17 ExxonMobil pairs have 1 error between them, so
  ExxonMobil is 19 of 23 (82.6%).
- **ExxonMobil's repeated evidence:**
  - Positions 45 and 55 carry the oil-equivalent conversion statement, but none of their questions asks it, so that
    fact stays at 2 candidates.
  - Pair 661 repeats batch 5's 10% discount fact. The two are in `NegativeGuard` conflict (§9).
- **Answers derived from model reasoning: none** among the 120 proposals.
  - One proposal filled a cut-off cell with the value of the row above (position 27). The quote check rejected it, by
    design.
  - One answer reads a window-cut table header (pair 201); the review rejected it.
  - **No included answer or question is built from reasoning or outside knowledge** (§7).
- **JPMorgan segment attribution:**
  - Of the 2 segment-results passages, position 31 built no pair. Position 41 built one, pair 185, and it gives
    Consumer & Community Banking's 1.40% net charge-off rate as the Firm's (the Firm's was 0.68%); revised.
  - Pair 179 gives the wholesale portfolio's net charge-offs as the Firm's.
  - JPMorgan is batch 6's weakest company: 7 of its 14 pairs accepted (§6).
- **Duplicates and repeated facts:**
  - No duplicate question.
  - Two included pairs ask a fact already asked from another passage: Tesla's 2020 inventory write-downs and
    ExxonMobil's 10% discount. Both are in conflict with their earlier pair (§9).
- **The 0.80–0.90 semantic review band:** no batch 6 question is in it (highest 0.7905). Across the six batches the band
  holds 2 questions, both reviewed earlier (§8).
- **Existing data:**
  - `train_candidates_v2` rebuilds byte-identical;
  - `validation_pairs_v4` loads apart from all 678 trial training pairs;
  - all 560 records use one model, prompt, validator and parameter set.

**Acceptance falls to 81.0%,** the lowest of the six batches (85.7–92.7% before), but above the gate. The fall comes from
JPMorgan, 7 of 14 accepted; the other four companies are 61 of 70 (87.1%). The yield per planned passage is not lower
(1.13), because evidence exclusions fell to 9.7% of passing questions (§3, §5).

**GO for building `train_candidates_v3`,** on your approval and under the conditions in §14.
- **P1's final result:** 612 included pairs from 560 planned passages, and **678 candidates** with the 66 of
  `train_candidates_v2`.
- **Budget:** batch 6 used 87,178 tokens, 58% of its 150,000 cap. 68,510 of the approved ~740,000 remain unspent.

**Unchanged:** the three-seed A0 versus A1 experiment stays **NO-GO**. Blockers B1 (a selection-aware decision rule) and
B3 (the GPU rehearsal) still stand, and B2 (the planned data size) now needs your decision on the final size (§14).

## 2. Records and provenance

| Check | Result |
|---|---|
| Plan | `train.train_v3_p1_batch6_n60.json`, SHA-256 `596920a8…4550`, unchanged; 60 train passages |
| Records file | `generation/train.records.jsonl`, 560 lines, SHA-256 `3d0b07ef…5dce` |
| Batches 1–5 | Lines 1–500 hash to `51e8115b…7b00` (lines 1–400 to `398697df…`, lines 1–300 to `3a8952fe…`), as the batch 5 audit recorded them. The runs only appended. |
| Batch 6's records | Lines 501–560: one record per plan position, in plan order; 560 distinct passages in the file |
| Outcome | **60 `done`, 0 `failed`**. Every record took 1 attempt, with no error. |
| Identity | All 560 records: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…`, record format 3 |
| Run | `train.run.json` is unchanged (`72c9e7bf…`); there is no continuation file |
| Replies | 60 distinct response hashes (559 across the file; batch 1's failed record has none) |
| Proposals | 120: every reply held two questions; one of them (position 43) has a null question |
| Validator re-run | `revalidate --dry-run`: 559 up to date, 0 changed, 1 failed record (batch 1's position 87) |
| Tokens | **87,178** (72,083 prompt, 15,095 completion), equal in the records and the two run summaries. Batches 1–6 total 819,774. |
| Completion tokens per reply | 165–505, median 246, mean 252. The longest reply of P1 (505) is still 263 below the 768 limit. |
| Against the dry-run estimate | About 87,800 tokens were estimated; the run used 87,178 (0.7% less) in 61 HTTP requests |
| Against the 150,000 cap | 58.1% used; the stop rule's bound was never approached |

**The runs:** two invocations, from 2026-11-08 21:12 to 22:39 UTC (record timestamps), from
`logs/train_v3_p1_batch6_n60.log`. The log holds no credential.

| Run | Request budget | HTTP requests | HTTP 429 | HTTP 503 | Done | Tokens | Ended by |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 60 | 60 | 0 | 1 | 59 | 86,000 | the request budget: the 503's retry used one request, so position 60 was left |
| 2 | 5 | 1 | 0 | 0 | 1 | 1,178 | the end of the plan |
| **All** | | **61** | **0** | **1** | **60** | **87,178** | |

**The token stop rule held:**
- After run 1 the rule allowed at most floor((150,000 − 86,000) / 2,246) = 28 more requests; run 2's budget was 5.
- In all, 61 HTTP requests were sent, below the 66 at which the cap could first be breached.
- **Groq's limits:** one HTTP 503 ("over capacity"), retried after 2 s, and no HTTP 429. The generator spent 2,638 s
  in pacing waits. The last responses showed 1,000 requests per day (941 and 969 remaining) and 8,000 tokens per
  minute.

**Against batches 1–5:**

| | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Batch 5 | Batch 6 (60 passages) |
|---|---:|---:|---:|---:|---:|---:|
| Tokens | 148,284 | 146,678 | 146,256 | 144,866 | 146,512 | 87,178 |
| Tokens per planned passage | 1,483 | 1,467 | 1,463 | 1,449 | 1,465 | 1,453 |
| Questions passing the validator | 151 | 159 | 154 | 168 | 159 | 93 |
| Pairs built | 112 | 124 | 126 | 133 | 120 | 84 |
| AI review accepts (candidates) | 101 | 115 | 108 | 114 | 106 | 68 |

## 3. The funnel

120 proposed → **93** pass the generator's validator (77.5%) → **84** pairs built → **68** accepted by the review → **68**
included in the trial candidate set.

| | Plan | Done | Proposed | Pass validator | Pairs built | Accepted (included) |
|---|---:|---:|---:|---:|---:|---:|
| Batch 1 | 100 | 99 | 198 | 151 (76%) | 112 | 101 |
| Batch 2 | 100 | 100 | 200 | 159 (80%) | 124 | 115 |
| Batch 3 | 100 | 100 | 199 | 154 (77%) | 126 | 108 |
| Batch 4 | 100 | 100 | 200 | 168 (84%) | 133 | 114 |
| Batch 5 | 100 | 100 | 200 | 159 (80%) | 120 | 106 |
| Batch 6 | 60 | 60 | 120 | 93 (78%) | 84 | 68 |
| **All** | **560** | **559** | **1,117** | **884 (79%)** | **699** | **612** |

- **Exclusions at each step:**
  - `build-pairs` excluded 9 questions, every one for evidence leakage: `evaluation_evidence` 8 and
    `cross_split_evidence` 1 (§5).
  - Nothing was excluded for a validator re-check, a duplicate or evaluation overlap.
  - The trial candidate build excluded nothing beyond the 16 pairs the review did not accept.
- **Per passage:** 10 passages contribute no included pair, 32 contribute one and 18 contribute two (batch 5, per 100:
  27, 40 and 33).
- **More pairs per passage than any earlier batch** (1.40 built per planned passage, against 1.12–1.33): evidence
  exclusions fell to 9 of 93 passing questions (§5).
- **Batches 1–5 in this replay:**
  - The scratch `build-pairs` ran over all 560 records. Batches 1–5 keep the same pairs and exclusions, and their
    traced outcomes are identical to the batch 5 audit's.
  - 474 of the 615 earlier pair IDs differ from the 500-record numbering, as expected (§11).
  - The trace of every candidate through `build_pairs`' own functions matches the written pairs, ID for ID.

## 4. Generator acceptance and rejection (validator `sec-qcheck-v3`)

| Reason | Count | Verdict |
|---|---:|---|
| `missing_company` | 14 | All correct: the question names no company, or only a business or a third party ("First Republic", "Enanta Pharmaceuticals", "the 2025 CEO Performance Award", "the CFPB lawsuit"). |
| `evidence_length` | 4 | All correct by the rule: quotes of 303–906 characters (limit 300). |
| `copied_from_passage` | 4 | 1 copies a 9-word run of prose (position 53). 1 is borderline, an 8-word run (49). 2 are false (positions 9 and 12): row labels of tables flattened without `[TABLE]` tags, so the label exemption does not apply. |
| `evidence_not_in_visible_text` | 3 | 1 correct: position 27 fills a cut-off cell with the row above's figure (§7). 2 borderline: faithful quotes whose fragment after an ellipsis starts with a period (position 1, as batch 5's 72 and 91) or with a year (12). |
| `answer_not_supported` | 1 | False (position 57): the answer is right, but its year is a column header, not in the quote. |
| `malformed` | 1 | Correct (position 43): the reply's second question is null. |

**Totals:**
- 21 correct;
- 3 correct by the strict rule (borderline);
- **3 false rejections**, positions 9, 12 and 57: 2.5% of proposals. Batches 1–5 had 1.5–2.0%.

The false rejections repeat the two known patterns:
- years missing from the quote (57);
- a row label outside a `[TABLE]` block (9 and 12).

The validator stays as it is (decision I3); its weaknesses are listed for a later version (§13).

| Pos | Source | Reason | Question (verbatim) | Verdict |
|---:|---|---|---|---|
| 1 | JPM FY2022, narrative, Note 2 – Fair Value Measurement | `evidence_not_in_visible_text` | For JPMorgan Chase & Co in fiscal 2022, how do increases in yield and credit spread generally affect the fair value measurement of mortgage-backed securities? | Correct by the rule, borderline: both fragments are verbatim and in order, but the one after the ellipsis starts with a period (batch 5's positions 72 and 91); the answer is right. |
| 3 | TSLA FY2020, table, Item 8 | `evidence_length` | According to Tesla's solar energy systems note, what was the net book value of solar energy systems in service as of December 31, 2020? | Correct by the rule: the quote has 394 characters (limit 300). |
| 9 | WMT FY2020, narrative, Item 8 | `copied_from_passage` | What was Walmart's diluted net income per common share attributable to the company for the full fiscal year ended January 31, 2020? | False: the run is the row label of a quarterly table flattened without [TABLE] tags, so the label exemption does not apply; the answer is right ($5.19). |
| 12 | PFE FY2021, narrative, Item 7 | `copied_from_passage` | What was the impact on Pfizer's 2019 net income attributable to common shareholders from the actuarial valuation and other pension and postretirement plan gains and losses? | False: the run is the row label of a reconciliation table flattened without [TABLE] tags; the answer is the row's net-income cell (750). |
| 12 | PFE FY2021, narrative, Item 7 | `evidence_not_in_visible_text` | How much did the gain on the completion of the Consumer Healthcare JV transaction reduce Pfizer's 2019 net income attributable to common shareholders? | Correct by the rule, borderline: the answer is right ((8,107), the row's net-income cell), but the fragment after the ellipsis starts with a year, not a word. |
| 19 | WMT FY2026, table, Item 8 | `missing_company` | What was the principal amount of the long-term debt maturing on September 9, 2025, and what was its interest rate? | Correct: the question does not name the company. |
| 21 | JPM FY2023, table, Credit And Investment Risk Management | `missing_company` | How much of the after 1 year through 5 years loan category was associated with First Republic in the context of the provided footnotes? | Correct: the question does not name the company. |
| 23 | TSLA FY2019, narrative, Item 1A | `missing_company` | Which specific convertible senior notes are identified in the risk factors for fiscal 2019, and what are their respective coupon rates and maturity years? | Correct: the question does not name the company. |
| 26 | JPM FY2019, narrative, Item 1 | `missing_company` | What is the effective date for the new EU capital and liquidity legislation for banks and investment firms that implements Basel III standards? | Correct: the question does not name the company. |
| 26 | JPM FY2019, narrative, Item 1 | `missing_company` | Under the final EU legislation, what condition allows a second intermediate parent undertaking to be established for non-EU banks operating in the EU? | Correct: the question does not name the company. |
| 27 | PFE FY2023, table, Item 8 | `evidence_not_in_visible_text` | How much were Pfizer Inc.'s research and development expenses in millions for the year ended December 31, 2022? | Correct: the research and development row is cut after its 2023 value, and the answer gives 2022 as $13,677 million, the 2022 cell of the selling, informational and administrative row above (section 7). |
| 28 | TSLA FY2025, table, Item 8 | `missing_company` | According to the reconciliation of taxes at the federal statutory rate, what was the dollar amount attributed to withholding tax for the year ended December 31, 2025? | Correct: the question does not name the company. |
| 29 | WMT FY2025, narrative, Item 1A | `missing_company` | What was the court order entered on February 20, 2025, regarding the CFPB lawsuit? | Correct: the question does not name the company. |
| 31 | JPM FY2020, table, Business Segment Results | `missing_company` | What specific business unit was realigned from Consumer & Community Banking to Corporate & Investment Bank in the first quarter of 2020? | Correct: the question does not name the company. |
| 31 | JPM FY2020, table, Business Segment Results | `missing_company` | How many technology and support staff were transferred during the second and third quarters of 2020 in connection with the business segment changes? | Correct: the question does not name the company. |
| 36 | JPM FY2022, narrative, Item 1A | `evidence_length` | What potential consequences could JPMorgan Chase face in fiscal 2022 if litigation or enforcement actions arise from privacy and data protection laws? | Correct by the rule: the quote has 303 characters (limit 300). |
| 41 | JPM FY2024, table, Business Segment & Corporate Results | `missing_company` | How much were the Home Lending loans held-for-sale and at fair value at the end of fiscal 2024? | Correct: the question does not name the company. |
| 43 | TSLA FY2019, narrative, Item 1 | `malformed` | (none) | Correct: the reply's second question is null. |
| 46 | JPM FY2025, narrative, Item 1A | `evidence_length` | What specific operational and structural changes might foreign regulatory initiatives require JPMorganChase to implement in its non-U.S. subsidiaries? | Correct by the rule: the quote has 906 characters (limit 300). |
| 46 | JPM FY2025, narrative, Item 1A | `evidence_length` | How might differences and conflicts in financial services laws between the U.S. and other countries impact JPMorganChase's business strategy and costs? | Correct by the rule: the quote has 584 characters (limit 300). |
| 48 | TSLA FY2025, table, Item 8 | `missing_company` | According to the reconciliation prior to ASU 2023-09 adoption, what was the amount for the change in valuation allowance for the year ended December 31, 2023? | Correct: the question does not name the company. |
| 49 | WMT FY2026, narrative, Item 1A | `copied_from_passage` | In which specific product areas is Walmart involved in multidistrict litigation regarding novel theories of personal injury or economic loss? | Correct by the rule, borderline: an 8-word run of the sentence. |
| 50 | XOM FY2025, table, Business Results | `missing_company` | According to the 2025 Chemical Products Earnings Driver Analysis, how much did oversupply-driven margin changes decrease earnings, and what was the primary cause? | Correct: the question does not name the company. |
| 52 | PFE FY2024, narrative, Item 8 | `missing_company` | In the legal proceedings involving Paxlovid in fiscal 2024, what was the outcome of the U.S. District Court for the District of Massachusetts regarding Enanta Pharmaceuticals' patent claim? | Correct: the question does not name the company. |
| 53 | TSLA FY2019, narrative, Item 1A | `copied_from_passage` | Which regulation began to fully apply in May 2018 to the processing of personal information collected from individuals in the European Union, and what were its primary effects mentioned by Tesla? | Correct: copies a 9-word run of the sentence. |
| 57 | PFE FY2025, table, Item 8 | `answer_not_supported` | What was the weighted-average common shares outstanding on a diluted basis for Pfizer Inc for the year ended December 31, 2024? | False: the answer is right (5,700 million, the 2024 column); its year is a column header, not in the quote. |
| 58 | TSLA FY2025, table, Item 8 | `missing_company` | What was the market capitalization milestone required to earn Tranche 2 of the 2025 CEO Performance Award? | Correct: the question does not name the company. |

## 5. Pair building: evidence-leakage exclusions (9)

`build-pairs` excludes a question whose evidence, quote or labelled figure is held exactly by:
- a blocked passage: `evaluation_evidence`;
- the visible text of a passage in another split: `cross_split_evidence`.

These rules fail closed by design, and every exclusion below follows them.

| Where the evidence is | Batch 6 | Batch 5 | Batch 4 | Batch 3 | Batch 2 | Batch 1 |
|---|---:|---:|---:|---:|---:|---:|
| An evaluation-pool or audit passage | 0 | 11 | 10 | 4 | 6 | 6 |
| An overlap-group copy of an evaluation passage | 2 | 7 | 8 | 5 | 14 | 11 |
| Only passages blocked for reusing evaluation text | 6 | 17 | 10 | 18 | 7 | 16 |
| The visible text of a validation passage | 1 | 4 | 7 | 1 | 8 | 6 |
| **All** | **9** (9.7% of 93) | **39** (24.5%) | **35** (20.8%) | **28** (18.2%) | **35** (22.0%) | **39** (25.8%) |

- **By company:** Walmart lost 5 of its 20 passing questions and Tesla 3 of 17 (Tesla lost 13 in batch 5). Pfizer lost
  1 of 19; JPMorgan and ExxonMobil lost none.
- **No question drew on an evaluation-pool or audit passage,** for the first time in P1.
- **Cross-split:** the one validation-passage holder (Pfizer's position 22) keeps the training set and
  `validation_pairs_v4` apart. Three evaluation exclusions also have a validation copy.
- **The rule is unchanged** (decision E5).

| Pos | Source | Question (verbatim) | Answer (verbatim) | Rule | Where the evidence is |
|---:|---|---|---|---|---|
| 4 | WMT FY2024, narrative, Item 8 | What was the total net cash used in financing activities for Walmart in the year ended January 31, 2022? | (22,828) | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 8 | TSLA FY2025, narrative, Item 1 | What is the name of the general purpose, autonomous humanoid robot that Tesla is developing and to which it applies its self-driving technology? | Optimus | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 8 | TSLA FY2025, narrative, Item 1 | How many electric motors are used in the vehicle powertrain technology for certain versions of the Model S, Model X, Cybertruck, and Tesla Semi to increase performance? | three | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 14 | WMT FY2025, narrative, Item 1A | What exclusive forum does Walmart Inc. designate in its bylaws for shareholder derivative claims regarding duty violations by directors or officers for fiscal 2025? | The Delaware Court of Chancery | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 22 | PFE FY2025, narrative, Item 8 | Who leads the Pfizer U.S. Commercial Division as part of the Biopharma reportable segment structure effective January 1, 2025? | The Chief U.S. Commercial Officer, Executive Vice President | `cross_split_evidence` | the visible text of 2 validation passages |
| 38 | TSLA FY2025, table, Item 8 | What was Tesla's reported net income for fiscal 2025 according to its consolidated statements of cash flows? | $3,855 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 39 | WMT FY2020, narrative, Item 8 | How much did Walmart record for the loss on extinguishment of debt in its operating activities adjustments for the fiscal year ended January 31, 2018? | 3,136 | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked; also 1 validation) |
| 44 | WMT FY2025, narrative, Item 3 | How much liability did Walmart Inc. accrue in fiscal 2023 for opioid-related settlements, and what was the status of that liability as of January 31, 2025? | Walmart Inc. accrued a liability of approximately $3.3 billion in fiscal year 2023, and as of January 31, 2025, all of the accrued liability has been paid. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked; also 1 validation) |
| 59 | WMT FY2025, narrative, Item 1 | How does Walmart describe the relationship between its website ESG disclosures and the content of its fiscal 2025 Form 10-K? | Nothing on the website, including ESG reporting, is deemed incorporated by reference into the Annual Report on Form 10-K or any other SEC filings. | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked; also 1 validation) |

## 6. AI review of the 84 built pairs

**How each pair was judged:** as in batches 1–5, each pair was read against its positive passage's full visible text.
- **Accept:** correct company, period, figure and unit; supported by the quote; a useful financial retrieval query.
- **Revise:** a specific error that a reworded question would fix.
- **Reject:** no financial content, or no real answer.

**Precedents, applied here:**
- a part given as the whole, or a figure given to the wrong business, is revised (batch 3's pairs 106–123; batch 4's
  reserves pairs; batch 5's pair 600);
- a flattened table whose later block keeps the first block's year labels is read by its block header (batch 5's pair
  157);
- a question naming no year is revised when the answer changes between filings (batch 3's pair 215; batch 5's 175);
- an answer that only restates the question is revised (batch 5's pair 103);
- a safe-harbour or reporting-basis statement with no financial content is rejected (batch 1's pair 106; batch 3's
  152; batch 4's 176), and so is a table-header lookup (batch 3's 169).

**Corpus look-ups were used for:**
- pairs 194 and 195: JPMorgan's Oil & Gas exposure at December 31, 2023 and 2024 (FY2023 and FY2024 filings);
- pair 177: JPMorgan's real estate exposure at both dates in the FY2023 filing's Note 4;
- pairs 179 and 185: the Firm's net charge-offs and rate (FY2023 segment results; FY2024 three-year summary);
- pair 339: Pfizer's other business activities in the FY2021–FY2025 filings;
- pair 622: ExxonMobil's FY2022 undeveloped reserves block.

**Result:** 68 accept (81.0%), 13 revise, 3 reject.

| Aspect (84 pairs) | Pass | Minor | Fail | n/a |
|---|---:|---:|---:|---:|
| Factual accuracy | 67 | 3 | 14 (177, 179, 185, 195, 201, 260, 280, 339, 348, 566, 621, 622, 660, 679) | – |
| Financial relevance | 79 | 2 | 3 (148, 200, 201) | – |
| Company | 84 | – | – | – |
| Period | 71 | 9 (names no year) | 4 (177, 195, 339, 566) | – |
| Figures (answer figures in the quote) | 68 | – | – | 16 |
| Units | 49 | 19 (no unit) | 0 | 16 |

**Issues noted on accepted pairs:**
- **Boilerplate:** 9, mostly risk factors and conventions repeated yearly.
- **Inaccurate wording:** 3. "Operate" for retail sites that are mostly distributors' (667). "Total LIFO inventory value"
  for the crude, products and merchandise total (635). "Non-operating items" for exclusions that are also
  transportation, LNG and power earnings (680).
- **Weak financial relevance:** 2: a yes/no supply outlook and an ESG aims statement.
- **No issue:** 54.

Batch 5: 17, 4, 4 and 81. Batch 4: 13, 10, 1 and 90.

**Not accepted (16):**

| Pair | Pos | Decision | Reason |
|---:|---:|---|---|
| 148 | 51 | Reject | The safe-harbour statement that the disclosures are forward-looking statements under the PSLRA: boilerplate with no financial content (batch 4 precedent, pair 176; batch 3, pair 152). |
| 177 | 56 | Revise | Wrong year: $170,857 million is the December 31, 2022 total (the FY2023 filing's Note 4 gives 2023's as $208,261 million and 2022's as $170,857 million); the flattened block keeps the 2023 labels. The fact of batch 5's FY2022 pair, asked for the wrong date. |
| 179 | 21 | Revise | Wrong scope: $879 million and $181 million are the wholesale portfolio's net charge-offs (the table's classes are wholesale: secured by real estate, commercial and industrial, other); the Firm's total also includes Consumer & Community Banking's $5,340 million for 2023 (the FY2023 segment results). |
| 185 | 41 | Revise | Segment figure given to the Firm: 1.40% is Consumer & Community Banking's total net charge-off rate (the table's rows are CCB's businesses, and footnote (b) names CCB); the Firm's 2024 rate was 0.68% (the FY2024 three-year summary). |
| 195 | 11 | Revise | Wrong year: the E&P row follows the table's 'December 31, 2023' header (the flattened cells keep the 2024 labels); 51% is the December 31, 2023 figure (as the FY2023 filing shows), while December 31, 2024's is 55% (the preceding chunk). Batch 5's pair 157 precedent. |
| 200 | 6 | Reject | Boilerplate with no financial content: every filing states that its statements are prepared under U.S. GAAP (batch 1 precedent, pair 106; batch 4, pair 176). |
| 201 | 6 | Reject | No financial fact: it asks the table's column labels, and the window cuts the header after 'Fully taxable-equivalent' (the columns continue with the adjustments and the managed basis), so the answer is incomplete; 'expressed in millions' and 'the year ended December 31' with no year are meaningless. |
| 260 | 47 | Revise | Misread table: (282) is the postretirement plans' 2020 net periodic benefit credit, not a cost (the answer drops the parentheses), and the postretirement column is not international (the international pension plans' 2020 cost is $141 million). |
| 280 | 7 | Revise | Wrong scope: $986 million is the unrealized loss on the investments in BioNTech, Allogene Therapeutics and Arvinas together, not on BioNTech alone. |
| 339 | 22 | Revise | Names no year, and the answer changes between filings: PC1 alone in the FY2021-FY2022 filings, Business Innovation in FY2023, PC1 and Pfizer Ignite in FY2024-FY2025 (batch 3 precedent, pair 215); and other business activities are not a segment. |
| 348 | 53 | Revise | Empty answer: the passage does not specify the obligations, and the answer restates the question ('certain legal obligations ... personal information'); batch 5's pair 103 precedent. |
| 566 | 34 | Revise | Wrong year: the $0.76 per share row follows 'Balances as of February 1, 2023', so it is the fiscal 2024 declaration, the statement's first year (its $6,140 million matches pair 528's fiscal 2024 dividends paid); the fiscal 2026 rows lie past the window. |
| 621 | 25 | Revise | One year asked for a range: Canada's open tax years are 2001-2021, so '2001' is only the earliest; reword as the earliest open year. The FY2023 tax table (batch 2) also starts Canada at 2001, and the guard allows that passage as this pair's negative (section 8). |
| 622 | 55 | Revise | Part given as the whole: 3,284 million barrels is the United States' proved developed reserves (the row is in the 'Developed' block); total proved reserves add the undeveloped 2,246, later in the table (batch 4's reserves precedent). |
| 660 | 45 | Revise | Part given as the whole: 736,138 is the consolidated subsidiaries' future cash inflows at December 31, 2021 (the block is headed 'As of December 31, 2021 Consolidated Subsidiaries'); the equity companies' are listed separately (batch 5's pair 600). |
| 679 | 40 | Revise | Part given as the whole: 70,818 and 26,781 are the consolidated subsidiaries' 2024 revenues (the block is headed '2024 Consolidated Subsidiaries'); the equity companies' are listed separately, so they are not ExxonMobil's total (batch 5's pair 600). |

**Patterns:**
- **Scope:** 6 (batch 5: 6).
  - **JPMorgan:** 185 gives Consumer & Community Banking's net charge-off rate as the Firm's; 179 gives the wholesale
    portfolio's net charge-offs as the Firm's.
  - **ExxonMobil:** 679 and 660 give the consolidated subsidiaries' block as ExxonMobil's total; 622 gives developed
    reserves as total proved reserves.
  - **Pfizer:** 280 gives a loss on three investments as BioNTech's alone.
- **Period:** 4 (batch 5: 8).
  - **Flattened tables that kept the first block's year labels:** 195 and 177 (both JPMorgan credit tables). 177 asks
    batch 5's real estate total for the wrong date.
  - **A statement's first year asked as its last:** 566 (Walmart's dividend per share).
  - **A no-year question whose answer changed between filings:** 339 (Pfizer's other business activities).
- **Reading:** 3.
  - 260: a pension table's sign and column.
  - 348: an answer that restates the question.
  - 621: a range answered by one year.
- **Rejections:** 3.
  - 148: the PSLRA safe harbour.
  - 200: the U.S. GAAP basis.
  - 201: the column labels of a header cut by the window.
- **Flattened tables:** 4 of the 16 (177, 195, 260, 566) come from tables whose flattening loses a block's year or a
  column's meaning, as in batch 5.

**JPMorgan (12 passages, 14 pairs, 7 accepted):**
- **Segment results (positions 31 and 41):**
  - 31's two questions were rejected by the validator for naming no company.
  - 41's one pair (185) gives a segment figure to the Firm; revised.
  - Batch 5 had none, and batch 4 had 5 of 11 segment pairs.
- **The other errors:**
  - one Firm-level scope error in a wholesale table (179);
  - two flattened-table year labels (177, 195);
  - three rejections from boilerplate and header passages (6 and 51).
- **What was accepted:** the 7 are credit-portfolio figures and risk statements that state their portfolio or industry:
  Oil & Gas, wholesale nonaccrual loans, Consumer & Retail criticized exposure.
- **A small sample:** with 12 passages the rate is uncertain. JPMorgan was 19 of 21 outside segment tables in batch 5.

**Tesla FY2019 (positions 13, 23, 33, 43, 53):**
- 7 pairs, 6 accepted: two Solar Roof and servicing risks, the market and supplier risks, $12.49 billion of
  indebtedness, and the 20-year lease terms. 4 of the 6 name no year.
- 348 (the CCPA) was revised.
- None of the five passages is a financial statement, so batch 6 adds no Tesla FY2019 figures beyond indebtedness.

**Walmart FY2025–FY2026 (positions 14, 19, 29, 34, 44, 49, 59):**
- 8 pairs, 7 accepted:
  - the forum provision;
  - the CFPB lawsuit;
  - the MDL count as of March 10, 2025;
  - self-insurance;
  - the ESG aims;
  - fiscal 2026 operating lease cost;
  - total shareholders' equity.
- 566 was revised (§6, patterns).
- Three more questions were excluded as evaluation evidence (14, 44, 59).

**ExxonMobil's supplemental and reserves tables (positions 40, 45 and 55):**
- 6 pairs, 3 revised for scope (679, 660, 622). Positions 40 and 45 put a consolidated-subsidiaries block in the window
  without its equity-company block, and position 55 the developed block without the undeveloped one.
- The other 3 are accepted: the excluded-earnings amount, the 10% discount factor and the SEC price basis.
- Across batches 4–6 these tables have given 13 scope errors; the review caught each.

Every pair, verbatim (pair numbers are those of the 560-record pair set; positions are batch 6's plan positions):

| Pair | Pos | Source | Type | Question (verbatim) | Answer (verbatim) | Decision | Note |
|---:|---:|---|---|---|---|---|---|
| 142 | 16 | JPM FY2021, table, Credit And Investment Risk Management | financial_table | What was the ending balance of the wholesale nonaccrual loan portfolio for JPMorgan Chase in the year ended December 31, 2021? | $2,445 million | Accept | Correct. |
| 143 | 16 | JPM FY2021, table, Credit And Investment Risk Management | risk | How much did JPMorgan Chase's secured exposure to the Oil & Gas sector amount to at December 31, 2021? | $18.0 billion | Accept | Correct; the Oil & Gas footnote's secured exposure. |
| 148 | 51 | JPM FY2021, narrative, Forward-Looking Statements | other | According to JPMorgan Chase's fiscal 2021 Form 10-K, which specific legislative act provides the legal framework for the forward-looking statements included in the disclosure? | Private Securities Litigation Reform Act of 1995 | **Reject** | The safe-harbour statement that the disclosures are forward-looking statements under the PSLRA: boilerplate with no financial content (batch 4 precedent, pair 176; batch 3, pair 152). |
| 149 | 51 | JPM FY2021, narrative, Forward-Looking Statements | risk | What are some of the key risk factors cited by JPMorgan Chase in its fiscal 2021 filing that could cause actual results to differ from forward-looking expectations? | Economic, financial, reputational and other impacts of the COVID-19 pandemic; Local, regional and global business, economic and political conditions and geopolitical events; Changes in laws, rules, and regulatory requirements | Accept | Correct; the first three factors of the forward-looking-statements list, repeated yearly (COVID-19 in the FY2020-FY2022 filings). |
| 151 | 36 | JPM FY2022, narrative, Item 1A | risk | How might compliance with conflicting privacy laws in different jurisdictions affect JPMorgan Chase's new product development? | It could hinder the development of new products or services, curtail the offering of existing products or services, or affect how products and services are offered to clients and customers. | Accept | Correct; names no year; a privacy risk factor repeated yearly. |
| 164 | 1 | JPM FY2022, narrative, Note 2 – Fair Value Measurement | risk | What specific characteristics of the underlying mortgages determine the inherent risk in mortgage-backed securities for JPMorgan Chase & Co's fiscal 2022 reporting? | Borrower FICO scores, LTV ratios for residential mortgages, and the nature of the property and/or any tenants for commercial mortgages. | Accept | Correct; a valuation-input description repeated yearly. |
| 177 | 56 | JPM FY2023, table, Credit And Investment Risk Management | financial_table | What was JPMorgan Chase's total real estate credit exposure, including loans and derivative receivables, as of December 31, 2023? | $170,857 million | **Revise** | Wrong year: $170,857 million is the December 31, 2022 total (the FY2023 filing's Note 4 gives 2023's as $208,261 million and 2022's as $170,857 million); the flattened block keeps the 2023 labels. The fact of batch 5's FY2022 pair, asked for the wrong date. |
| 178 | 56 | JPM FY2023, table, Credit And Investment Risk Management | risk | How did JPMorgan Chase's Consumer & Retail criticized exposure change between December 31, 2022, and December 31, 2023? | Increased by $409 million from $7.9 billion to $8.3 billion | Accept | Correct; the Consumer & Retail industry's criticized exposure, as asked. |
| 179 | 21 | JPM FY2023, table, Credit And Investment Risk Management | financial_table | What was the total net charge-off amount for JPMorgan Chase & Co. for the year ended December 31, 2023, and how did it compare to the total for the prior year? | $879 million in 2023 and $181 million in 2022 | **Revise** | Wrong scope: $879 million and $181 million are the wholesale portfolio's net charge-offs (the table's classes are wholesale: secured by real estate, commercial and industrial, other); the Firm's total also includes Consumer & Community Banking's $5,340 million for 2023 (the FY2023 segment results). |
| 185 | 41 | JPM FY2024, table, Business Segment & Corporate Results | financial_table | What was the total net charge-off rate for JPMorgan Chase & Co in fiscal 2024? | 1.40% | **Revise** | Segment figure given to the Firm: 1.40% is Consumer & Community Banking's total net charge-off rate (the table's rows are CCB's businesses, and footnote (b) names CCB); the Firm's 2024 rate was 0.68% (the FY2024 three-year summary). |
| 194 | 11 | JPM FY2024, narrative, Credit And Investment Risk Management | financial_table | What was JPMorgan Chase's total Oil & Gas credit exposure as of December 31, 2024? | $31,724 million | Accept | Correct; the December 31, 2024 block (the FY2024 filing's Note 4 confirms $31,724 million). |
| 195 | 11 | JPM FY2024, narrative, Credit And Investment Risk Management | risk | What percentage of JPMorgan Chase's Exploration & Production and Oil field Services Oil & Gas exposure was investment-grade as of December 31, 2024? | 51% | **Revise** | Wrong year: the E&P row follows the table's 'December 31, 2023' header (the flattened cells keep the 2024 labels); 51% is the December 31, 2023 figure (as the FY2023 filing shows), while December 31, 2024's is 55% (the preceding chunk). Batch 5's pair 157 precedent. |
| 200 | 6 | JPM FY2025, table, Explanation And Reconciliation Of The Fi | accounting_policy | What is the basis of accounting for the consolidated financial statements that JPMorgan Chase & Co presents for the fiscal year ended December 31, 2025? | U.S. GAAP | **Reject** | Boilerplate with no financial content: every filing states that its statements are prepared under U.S. GAAP (batch 1 precedent, pair 106; batch 4, pair 176). |
| 201 | 6 | JPM FY2025, table, Explanation And Reconciliation Of The Fi | financial_table | In the reconciliation of JPMorgan Chase & Co's non-GAAP measures for the year ended December 31, what are the specific basis labels used in the summary table, expressed in millions? | Reported and Fully taxable-equivalent | **Reject** | No financial fact: it asks the table's column labels, and the window cuts the header after 'Fully taxable-equivalent' (the columns continue with the adjustments and the managed basis), so the answer is incomplete; 'expressed in millions' and 'the year ended December 31' with no year are meaningless. |
| 217 | 2 | PFE FY2019, narrative, Item 1 | risk | What operational issue did Pfizer identify as the cause for periodic material shortages in 2019? | Constrained capacity or operational challenges with associated suppliers | Accept | Correct. |
| 218 | 2 | PFE FY2019, narrative, Item 1 | risk | Does Pfizer anticipate a significant impact to its operations from material shortages in 2020? | No significant impact is anticipated | Accept | Correct; a yes/no supply outlook. |
| 223 | 37 | PFE FY2019, narrative, Item 1A | risk | What specific policy initiative by private third-party payers does Pfizer describe as shifting cost burden to manufacturers and patients by excluding copay assistance from out-of-pocket cost calculations? | Copay accumulators | Accept | Correct; names no year; a payer-policy risk factor of the FY2019 filing. |
| 224 | 37 | PFE FY2019, narrative, Item 1A | risk | In which month and year did Pfizer's Lyrica product lose patent protection in the U.S.? | June 2019 | Accept | Correct. |
| 240 | 32 | PFE FY2020, narrative, Item 7 | financial_table | What was the amount of net cash used in financing activities from continuing operations for Pfizer Inc in fiscal 2020? | $(21,640) million | Accept | Correct. |
| 241 | 32 | PFE FY2020, narrative, Item 7 | mdna | What was the primary driver for the change in financing activities from continuing operations between 2020 and 2019 for Pfizer Inc? | $14.0 billion net payments on short-term borrowings in 2020 compared to $10.6 billion net proceeds raised in 2019 | Accept | Correct. |
| 259 | 47 | PFE FY2021, table, Item 8 | financial_table | What was the interest cost for Pfizer's U.S. pension plans in millions of dollars for the year ended December 31, 2021? | $455 million | Accept | Correct; the U.S. pension plans' 2021 column. |
| 260 | 47 | PFE FY2021, table, Item 8 | financial_table | How much did Pfizer report as the net periodic benefit cost for its international postretirement plans in millions of dollars for the year ended December 31, 2020? | $282 million | **Revise** | Misread table: (282) is the postretirement plans' 2020 net periodic benefit credit, not a cost (the answer drops the parentheses), and the postretirement column is not international (the international pension plans' 2020 cost is $141 million). |
| 274 | 42 | PFE FY2022, table, Item 8 | financial_table | What was Pfizer's comprehensive income attributable to the company for the year ended December 31, 2022? | $28,965 million | Accept | Correct. |
| 275 | 42 | PFE FY2022, table, Item 8 | debt_liquidity | How much did Pfizer report in cash and cash equivalents as of December 31, 2022? | $416 million | Accept | Correct. |
| 279 | 7 | PFE FY2022, table, Item 8 | debt_liquidity | What was the amount of capitalized interest for Pfizer Inc in fiscal 2022? | $124 million | Accept | Correct. |
| 280 | 7 | PFE FY2022, table, Item 8 | risk | How much unrealized loss did Pfizer Inc report on its investment in BioNTech in 2022? | $986 million | **Revise** | Wrong scope: $986 million is the unrealized loss on the investments in BioNTech, Allogene Therapeutics and Arvinas together, not on BioNTech alone. |
| 295 | 27 | PFE FY2023, table, Item 8 | revenue | What were Pfizer Inc.'s total product revenues in millions for the year ended December 31, 2023? | $50,914 million | Accept | Correct. |
| 314 | 17 | PFE FY2024, table, Item 8 | financial_table | What was the cost recognized in Comprehensive income for Pfizer's U.S. Pension Plans for the year ended December 31, 2024? | $116 | Accept | Correct; no unit (millions, stated before the window). |
| 315 | 17 | PFE FY2024, table, Item 8 | accounting_policy | What discount rate did Pfizer use for its U.S. Pension Plans to determine net periodic benefit cost in fiscal 2024? | 5.4% | Accept | Correct. |
| 319 | 52 | PFE FY2024, narrative, Item 8 | risk | According to Pfizer's fiscal 2024 disclosures, how many additional U.S. patents did GSK Group allege were infringed by Comirnaty in their August 2024 amended complaint? | Three additional U.S. patents | Accept | Correct. |
| 338 | 57 | PFE FY2025, table, Item 8 | accounting_policy | How many TSRUs were converted into PTUs during fiscal 2025 for Pfizer Inc, and what was the weighted-average grant price per unit for those conversions? | 20,968 TSRUs converted into 1,620 PTUs with a weighted-average grant price of $31.31 per unit in 2025 | Accept | Correct; the footnote's figures as stated. |
| 339 | 22 | PFE FY2025, narrative, Item 8 | other | Which entities are included in Pfizer's Other Business Activities segment for operating results? | PC1 and Pfizer Ignite | **Revise** | Names no year, and the answer changes between filings: PC1 alone in the FY2021-FY2022 filings, Business Innovation in FY2023, PC1 and Pfizer Ignite in FY2024-FY2025 (batch 3 precedent, pair 215); and other business activities are not a segment. |
| 340 | 43 | TSLA FY2019, narrative, Item 1 | other | What is the standard term length for Tesla's lease and power purchase agreement arrangements for solar energy systems in fiscal 2019? | 20-year terms | Accept | Correct. |
| 343 | 13 | TSLA FY2019, narrative, Item 1A | risk | What specific operational capabilities does Tesla identify as potential risks if they fail to keep pace with its increasing production for the Solar Roof? | Growing overall installation capability and avoiding unforeseen delays in the production ramp or inaccurate demand forecasts for the Solar Roof. | Accept | Correct; names no year; the FY2019 filing's Solar Roof ramp risk. |
| 344 | 13 | TSLA FY2019, narrative, Item 1A | risk | Which specific vehicle models does Tesla cite as higher-volume and newer additions to its fleet that could overburden its servicing capabilities and parts inventory? | Model 3 and Model Y. | Accept | Correct; names no year; the FY2019 filing's servicing risk. |
| 345 | 33 | TSLA FY2019, narrative, Item 1A | risk | Which specific automotive industry sector does Tesla identify as having a rapidly evolving market that could affect its vehicle sales? | The market for alternative fuel vehicles | Accept | Correct; names no year; a market risk factor repeated yearly. |
| 346 | 33 | TSLA FY2019, narrative, Item 1A | risk | What specific long-term agreement did Tesla enter into with Panasonic to help mitigate supply chain risk? | An agreement for Panasonic to be Tesla's manufacturing partner and supplier | Accept | Correct; names no year; a supply-chain risk factor repeated yearly. |
| 348 | 53 | TSLA FY2019, narrative, Item 1A | risk | What specific legal obligations did the California Consumer Privacy Act impose on Tesla starting in January 2020 regarding personal information? | Certain legal obligations on its use and processing of personal information related to California residents. | **Revise** | Empty answer: the passage does not specify the obligations, and the answer restates the question ('certain legal obligations ... personal information'); batch 5's pair 103 precedent. |
| 349 | 23 | TSLA FY2019, narrative, Item 1A | debt_liquidity | What was the aggregate principal amount of outstanding indebtedness for Tesla and its subsidiaries as of December 31, 2019? | $12.49 billion | Accept | Correct. |
| 377 | 3 | TSLA FY2020, table, Item 8 | accounting_policy | What was the total cost of inventory write-downs recorded by Tesla in its cost of revenues for the fiscal year ended December 31, 2020? | $145 million | Accept | Correct; the 2020 figure. |
| 441 | 38 | TSLA FY2025, table, Item 8 | financial_table | How much depreciation, amortization and impairment did Tesla record as an adjustment to reconcile net income to net cash provided by operating activities in the year ended December 31, 2024? | $5,368 million | Accept | Correct; the prior-year column. |
| 446 | 18 | TSLA FY2025, table, Item 8 | financial_table | What was Tesla's total lease expense for the year ended December 31, 2025? | $1,896 million | Accept | Correct. |
| 447 | 18 | TSLA FY2025, table, Item 8 | financial_table | What was the weighted-average remaining lease term for Tesla's operating leases as of December 31, 2025? | 7.7 years | Accept | Correct. |
| 450 | 58 | TSLA FY2025, table, Item 8 | financial_table | How many shares were subject to Tranche 1 of Tesla's 2025 CEO Performance Award? | 35,311,992 shares | Accept | Correct. |
| 451 | 28 | TSLA FY2025, table, Item 8 | financial_table | What was Tesla's total current federal income tax provision for the year ended December 31, 2023? | $48 million | Accept | Correct; the earliest column, named by year. |
| 452 | 48 | TSLA FY2025, table, Item 8 | tax | What was Tesla's cash paid for income taxes, net of refunds, in the foreign jurisdiction of China for the year ended December 31, 2025? | $751 million | Accept | Correct. |
| 465 | 39 | WMT FY2020, narrative, Item 8 | financial_table | What was the amount reported for adjustments related to Asda pension contribution in Walmart's consolidated statement of cash flows for fiscal 2020? | (1,036) | Accept | Correct; no unit (millions). |
| 466 | 54 | WMT FY2020, narrative, Item 8 | financial_table | What was the net cash provided by operating activities for Walmart Inc. in fiscal 2020? | 25,255 | Accept | Correct; no unit (millions). |
| 467 | 54 | WMT FY2020, narrative, Item 8 | financial_table | How much did Walmart Inc. pay for property and equipment during the fiscal year ended January 31, 2019? | (10,344) | Accept | Correct; the prior-year column; no unit (millions). |
| 468 | 24 | WMT FY2020, narrative, Item 8 | debt_liquidity | What was the total amount of dividends paid by Walmart Inc. to noncontrolling interest during fiscal 2018? | (690) | Accept | Correct; the earliest column, named by year; no unit (millions). |
| 469 | 24 | WMT FY2020, narrative, Item 8 | debt_liquidity | How much long-term debt did Walmart Inc. repay in the fiscal year ended January 31, 2019? | (3,784) | Accept | Correct; the prior-year column; no unit (millions). |
| 472 | 9 | WMT FY2020, narrative, Item 8 | revenue | How much net sales did Walmart report for the fourth quarter of fiscal 2020? | 137,743 | Accept | Correct; the fourth-quarter column; no unit (millions). |
| 528 | 4 | WMT FY2024, narrative, Item 8 | financial_table | How much did Walmart spend on dividends during fiscal 2024? | (6,140) | Accept | Correct; the parentheses mark an outflow; no unit (millions, stated before the window). |
| 532 | 59 | WMT FY2025, narrative, Item 1 | other | What are the primary environmental goals Walmart outlines for its sustainability efforts in fiscal 2025? | Enhance the sustainability of product supply chains, preserve protect and restore nature, reduce waste, procure renewable sources of energy, and reduce greenhouse gas emissions. | Accept | Correct; an ESG aims statement. |
| 540 | 29 | WMT FY2025, narrative, Item 1A | risk | What specific legal action did the Consumer Financial Protection Bureau take against Walmart Inc. on December 23, 2024? | Filed a lawsuit against the Company and Branch Messenger, Inc. in the District of Minnesota alleging violations of the Consumer Financial Protection Act. | Accept | Correct. |
| 541 | 14 | WMT FY2025, narrative, Item 1A | risk | According to Walmart Inc.'s fiscal 2025 filing, how might the exclusive forum provision for shareholder claims impact the company if a court deems it unenforceable? | It may face higher costs resolving such matters in other jurisdictions, which could negatively impact business and financial results | Accept | Correct; a forum-provision risk factor repeated yearly. |
| 543 | 44 | WMT FY2025, narrative, Item 3 | risk | How many cases with claims against Walmart Inc. were included in the National Prescription Opiate Litigation MDL as of March 10, 2025? | The MDL included approximately 250 cases with claims against the Company as of March 10, 2025. | Accept | Correct; the MDL count as of March 10, 2025 (the template of batch 4's and batch 5's counts for 2021 and 2020, with a new answer). |
| 555 | 49 | WMT FY2026, narrative, Item 1A | risk | How does Walmart primarily fund its claims for workers' compensation and general liability in fiscal 2026? | Predominantly through self-insurance | Accept | Correct; a self-insurance statement repeated yearly. |
| 565 | 34 | WMT FY2026, table, Item 8 | financial_table | What was the total shareholders' equity for Walmart Inc. as of January 31, 2026? | 105,887 | Accept | Correct; no unit (millions). |
| 566 | 34 | WMT FY2026, table, Item 8 | profitability | How much in cash dividends did Walmart declare per share during the period ending January 31, 2026? | $0.76 per share | **Revise** | Wrong year: the $0.76 per share row follows 'Balances as of February 1, 2023', so it is the fiscal 2024 declaration, the statement's first year (its $6,140 million matches pair 528's fiscal 2024 dividends paid); the fiscal 2026 rows lie past the window. |
| 567 | 19 | WMT FY2026, table, Item 8 | financial_table | What was the operating lease cost reported by Walmart Inc. for the fiscal year ended January 31, 2026? | $2,434 million | Accept | Correct. |
| 571 | 35 | XOM FY2019, narrative, Item 2 | other | What was ExxonMobil's net offshore acreage in Mozambique at year-end 2019, and how many net acres were relinquished during that year due to reduced working interests in specific Area 5 blocks? | Net acreage totaled approximately 1.8 million offshore acres at year-end 2019, and approximately 0.8 million net acres were relinquished in 2019. | Accept | Correct. |
| 572 | 35 | XOM FY2019, narrative, Item 2 | other | How many net development wells did ExxonMobil complete in Equatorial Guinea during the year ended December 31, 2019? | A total of 2.4 net development wells were completed during the year. | Accept | Correct. |
| 593 | 30 | XOM FY2020, table, 7. Equity Company Information | financial_table | What was ExxonMobil's share of total revenues from equity companies in fiscal 2020? | $21,282 million | Accept | Correct; ExxonMobil's share, as asked. |
| 594 | 30 | XOM FY2020, table, 7. Equity Company Information | revenue | What percentage of equity company revenues came from sales to ExxonMobil consolidated companies in 2019? | 13 percent | Accept | Correct; the prior year, named. |
| 609 | 60 | XOM FY2021, table, Consolidated Statement Of Comprehensive  | financial_table | What was ExxonMobil's comprehensive income attributable to the company for fiscal 2021, expressed in millions of dollars? | 25,981 million dollars | Accept | Correct. |
| 610 | 60 | XOM FY2021, table, Consolidated Statement Of Comprehensive  | financial_table | How much was the foreign exchange translation adjustment recorded in other comprehensive income for ExxonMobil in fiscal 2020, net of income taxes? | 1,916 million dollars | Accept | Correct; the prior-year column. |
| 620 | 25 | XOM FY2021, table, 19. Income And Other Taxes | tax | For fiscal 2021, what was Exxon Mobil Corp's net interest expense on income tax reserves, and how did it compare to the 2020 figure? | $0 million for 2021, compared to a credit of $6 million for 2020 | Accept | Correct. |
| 621 | 25 | XOM FY2021, table, 19. Income And Other Taxes | financial_table | According to the open tax years table, which tax year remains open for Exxon Mobil Corp's operations in Canada as of the fiscal 2021 filing? | 2001 | **Revise** | One year asked for a range: Canada's open tax years are 2001-2021, so '2001' is only the earliest; reword as the earliest open year. The FY2023 tax table (batch 2) also starts Canada at 2001, and the guard allows that passage as this pair's negative (section 8). |
| 622 | 55 | XOM FY2022, table, Item 2 | financial_table | What were the total oil-equivalent proved reserves for Exxon Mobil's consolidated subsidiaries in the United States at year-end 2022, in million barrels? | 3,284 million barrels | **Revise** | Part given as the whole: 3,284 million barrels is the United States' proved developed reserves (the row is in the 'Developed' block); total proved reserves add the undeveloped 2,246, later in the table (batch 4's reserves precedent). |
| 623 | 55 | XOM FY2022, table, Item 2 | accounting_policy | What price basis did Exxon Mobil use to report its proved reserves for the period ended December 31, 2022? | The average of the first-day-of-the-month price for each month during the last 12-month period | Accept | Correct; the SEC price basis, repeated yearly. |
| 634 | 20 | XOM FY2022, table, 3. Miscellaneous Financial Information | other | What were ExxonMobil's research and development expenses for the fiscal year 2022? | $824 million in 2022 | Accept | Correct. |
| 635 | 20 | XOM FY2022, table, 3. Miscellaneous Financial Information | financial_table | What was the total LIFO inventory value for ExxonMobil as of December 31, 2021? | $14,519 million | Accept | Correct figure; 'total LIFO inventory value' is loose: the table totals crude oil, products and merchandise. |
| 652 | 10 | XOM FY2023, table, 8. Investments, Advances And Long-Term R | financial_table | What was the total value of Exxon Mobil's equity method company investments and advances as of December 31, 2023? | 41,607 million dollars | Accept | Correct. |
| 653 | 10 | XOM FY2023, table, 8. Investments, Advances And Long-Term R | financial_table | How much were the long-term receivables and miscellaneous items, net of reserves, for Exxon Mobil at the end of fiscal 2022? | 6,944 million dollars | Accept | Correct; the prior-year column. |
| 660 | 45 | XOM FY2023, table, Supplemental Information On Oil And Gas  | financial_table | What was Exxon Mobil's total future cash inflows from sales of oil and gas in millions of dollars as of December 31, 2021? | 736,138 millions of dollars | **Revise** | Part given as the whole: 736,138 is the consolidated subsidiaries' future cash inflows at December 31, 2021 (the block is headed 'As of December 31, 2021 Consolidated Subsidiaries'); the equity companies' are listed separately (batch 5's pair 600). |
| 661 | 45 | XOM FY2023, table, Supplemental Information On Oil And Gas  | accounting_policy | What discount factor does Exxon Mobil apply when computing the standardized measure of discounted future net cash flows? | 10 percent | Accept | Correct; names no year; a standardized-measure convention repeated yearly (the fact of an earlier pair; section 9). |
| 667 | 5 | XOM FY2024, table, Item 2 | financial_table | How many total retail fuel sites did Exxon Mobil Corporation operate worldwide at the end of fiscal 2024? | 19,444 | Accept | Correct figure; 'operate' is loose: 18,842 of the 19,444 are distributor or reseller sites under its brands. |
| 668 | 5 | XOM FY2024, table, Item 2 | financial_table | What was the number of owned or leased retail fuel sites in Asia Pacific for Exxon Mobil at year-end 2024? | 264 | Accept | Correct. |
| 679 | 40 | XOM FY2024, table, Supplemental Information On Oil And Gas  | financial_table | What was ExxonMobil's total revenue from producing activities in fiscal 2024, and how much of that was generated by the United States region? | Total revenue was $70,818 million, with the United States contributing $26,781 million. | **Revise** | Part given as the whole: 70,818 and 26,781 are the consolidated subsidiaries' 2024 revenues (the block is headed '2024 Consolidated Subsidiaries'); the equity companies' are listed separately, so they are not ExxonMobil's total (batch 5's pair 600). |
| 680 | 40 | XOM FY2024, table, Supplemental Information On Oil And Gas  | accounting_policy | How much did ExxonMobil exclude from its producing activities results due to non-operating items and adjustments in fiscal 2024? | The excluded amounts totaled $1,395 million in 2024. | Accept | Correct figure; the excluded amounts are also transportation, LNG, power and other Upstream earnings, not only non-operating items. |
| 687 | 50 | XOM FY2025, table, Business Results | financial_table | What was Exxon Mobil's total Chemical Products earnings excluding Identified Items in fiscal 2025, reported in millions of dollars? | $1,070 million | Accept | Correct; the Chemical Products segment, named. |
| 691 | 15 | XOM FY2025, table, Note 14. Miscellaneous Financial Informa | financial_table | What was Exxon Mobil's total inventory value for crude oil, products, and merchandise in millions of dollars as of December 31, 2025? | $22,979 million | Accept | Correct. |
| 692 | 15 | XOM FY2025, table, Note 14. Miscellaneous Financial Informa | accounting_policy | How much did the aggregate replacement cost of Exxon Mobil's inventories exceed their LIFO carrying values by approximately at December 31, 2024? | $10 billion | Accept | Correct; the prior-year figure, named by date. |

## 7. Special investigation: answers unsupported by the visible passage

**What was looked for:** an answer built from the model's reasoning or knowledge rather than from the visible passage
(batch 2's pair 032 is the known case), an answer that invents cells (batch 5's position 73), and any other answer the
visible text does not support.

**The checks:**
1. **The detector** (rules R1–R4, unchanged) over every recorded answer of batches 1–6: 1,116 answers.
2. **The looser grounding rule.**
3. **The cut-off passages** flagged by batch 6's plan check (positions 3, 4, 5, 6, 10, 12, 18, 19, 21, 24, 27, 38, 39,
   40, 54, 55, 57 and 58), read first in the review.
4. **The review of all 84 pairs** (§6), with full visible text.
5. **A question-word scan** of the 68 included questions (pair 190's kind).

**Results:**
- **The detector flags no batch 6 answer, and neither does the looser rule.** Batches 1–5's flags are unchanged.
- **Batch 6 has no reasoning answer among its 120 proposals.**
  - Across batches 1–6: 6 reasoning answers in 1,117 proposals (1, 1, 0, 3, 1 and 0 by batch). R1 (meta-language)
    still matches exactly those 6 among the 1,116 answers, and nothing else.
- **One invented-cell answer (position 27, Pfizer FY2023 income statement).**
  - The research and development row is cut after its 2023 value. The answer gives 2022 as $13,677 million, the 2022
    cell of the selling, informational and administrative row above.
  - It has no meta-language, so no detector rule sees it. Its quote claims a cell the row does not show, so the quote
    check rejected it, by design.
  - Across P1 there are now 2 such answers (batch 5's position 73 and this one), both stopped by the quote check.
- **One incomplete answer (pair 201, JPMorgan FY2025 non-GAAP reconciliation):** the question asks the summary table's
  column labels, and the window cuts the header after "Fully taxable-equivalent". The review rejected it.
- **The cut-off scan predicted better this time.** Of its 18 flagged passages, 4 led to an error tied to the cut:
  - 27: the invented cell;
  - 6: the cut header;
  - 40 and 55: the block that completes a total lies past the window (679, 622).

  The other 14 gave complete-row answers. The common setting is the one batch 5 found: a question about exactly the
  part of a row or table that the window cuts off.
- **Questions:** the question-word scan flags 5 included questions, all paraphrase: "fourth quarter" for Q4, "lose" for
  "lost", "operate" in 667 (noted in the review). No included question carries a wrong outside fact.

**Conclusion:**
- No unsupported answer has reached the included data in six batches. The review and the quote check are the
  protection; the validator's other rules caught 4 of the 8 cases only by chance.
- **For a later validator version** (not applied; I3): an R1 rejection, and a rule that checks each quoted cell against
  the asked row of the visible text.

## 8. Evidence leakage and false-negative protections

**Leakage, on the 68 included pairs (all from the trial build, §11):**

| Check | Result |
|---|---|
| Evidence rule, re-checked at the candidate build | 0 leak passages; every included pair passes |
| Lexical overlap with the permitted evaluation questions (word Jaccard; reject at 0.8) | Highest 0.310 (batch 5: 0.310; batch 4: 0.345) |
| Semantic screen (`screen-questions --pairs --candidates train_candidates_v2`, in scratch) | 765 questions (the 699 pairs and the 66) against the 10 permitted question texts, `query_id` and text only. **0 flagged at 0.90.** Of batch 6's 84: highest 0.7905 (pair 185, revised). Of its 68 included: highest 0.7844 (pair 143), 0 at or above 0.80, 3 at or above 0.75, 14 at or above 0.70; median 0.655. |
| **Every question in the 0.80–0.90 review band** (approved) | **2 across batches 1–6, none from batch 6:** batch 1's pair 005 (019 in the 560-record numbering), 0.8017 against `q_0002`, reviewed and accepted as a different fact in the batch 1 audit; and batch 4's pair 154 (186 here), 0.8014 against `q_0004`, reviewed as a different fact and revised for scope in the batch 4 audit. No evaluation question text was read for this audit. The thresholds were not changed (I4). |
| Cross-split | 1 excluded at `build-pairs`. `load_validation_set(validation_pairs_v4, training=…)` passes against all 678 trial training pairs. |
| Duplicates | None by normalised question; two same-fact pairs, each kept apart from its earlier pair by the guard (§9) |

**Evidence within the model's window:**
- **The validator** located every accepted quote inside the tokenizer-visible part (512 tokens, special tokens
  included).
- **The independent re-check:** 66 of the 68 included quotes lie verbatim in the positive's visible text. Pairs 467
  and 653 join a row label to its cell.
- **The window's end:** the latest quote ends at token 512, the window's last; 1 ends after token 500 (batch 5: 4).
- **The window's limits:** pairs 201, 622 and 679 and position 27's proposal (§7).

**False negatives:**
- **Equivalents:** 32 of the 68 record exact equivalents (66 passages), and 27 record uncertain ones (99 passages).
  The trainer never uses any of them as a negative.
- **The plan's protections held:** the 24 planned passages that were recorded equivalents of earlier included pairs
  all stay inside those pairs' exclusions.
- **Conflicts:** `NegativeGuard` finds 913 conflicting pairs among the 678 trial pairs (697 among 610 after batch 5),
  and the planner keeps them in different batches.
  - Involving batch 6: batch 6 with batch 6, 35; with batch 1, 18; with batch 2, 26; with batch 3, 34; with batch 4,
    45; with batch 5, 49; with the 66, 9.
  - 56 of the 68 are in at least one conflict (82%; batch 5: 85%).
- **Batch fill:**

  | Batch size | Batches | Full | Mean fill (batch 5 trial) |
  |---:|---:|---:|---:|
  | 16 | 43 | 42 | 0.986 (0.978) |
  | 32 | 24 | 20 | 0.883 (0.953) |
  | 64 | 14 | 10 | 0.757 (0.867) |

  **New:** the denser conflict graph lowers the fill at the larger sizes. At 64 (the rehearsal's batch size, 63
  in-batch negatives) the mean batch holds 48 pairs, 4 of 14 batches are partly filled and the smallest holds 1. This
  is not leakage. It costs in-batch negatives, which matters for the training design (§13).
- **An independent probe:** for each included pair, any other included pair's positive was checked for the answer's
  figure wherever the guard would still allow it as a negative.
  - **Before the review's final pass,** there was 1 same-company hit. Pair 621's answer, 2001 (Canada's first open tax
    year), also starts Canada's range in batch 2's FY2023 tax table, and the guard would allow that passage as 621's
    negative.
  - The evidence is not equivalent (the ranges end in 2021 and 2023), and the question names the FY2021 filing. But
    the question also asks one year for a range, so 621 was revised and is not included.
  - **Now:** 3 hits, all chance matches in other companies' passages (1,896 and 22,979).
- **Known gaps remain:** paraphrased evidence, rounded figures, changed row labels, and the same answer inside another
  filing's differently-ended range (621's kind).

## 9. Duplicate questions and repeated financial facts

| Against | Same normalised question | Highest word Jaccard | Cosine ≥ 0.90 (BGE, zero-shot) |
|---|---:|---:|---|
| Within the 68 | 0 | 0.625 | 1 (0.907): JPMorgan's Oil & Gas secured exposure in 2021 against its total Oil & Gas exposure in 2024 |
| Batch 1's 101 | 0 | 0.708 | 4 (highest 0.942): Walmart's operating lease cost in fiscal 2026 against 2025, Tesla's 2023 current federal against total current tax, JPMorgan's Oil & Gas exposure in 2024 against 2021, Walmart's operating cash flow in fiscal 2020 against 2022 |
| Batch 2's 115 | 0 | 0.750 | 3 (highest 0.925): ExxonMobil's equity-method investments at 2023 against 2021, JPMorgan's Oil & Gas exposure against its total credit portfolio, ExxonMobil's tax-reserve interest in 2021 against 2023 |
| Batch 3's 108 | 0 | 0.875 | 3 (highest 0.948): Walmart's operating cash flow in fiscal 2020 against 2026, ExxonMobil's inventories at 2025 against 2024, ExxonMobil's share of equity-company revenues in 2020 against 2022 |
| Batch 4's 114 | 0 | 0.810 | 8 (highest **0.972**): Tesla's 2020 inventory write-downs, the same fact (below); Walmart's MDL count in 2025 against 2021; the rest the same statement line for another year |
| Batch 5's 106 | 0 | **0.895** | 10 (highest 0.955): Walmart's payments for property and equipment in fiscal 2019 against 2021; ExxonMobil's 10% discount factor, the same fact (below); the rest the same template for another year or measure |
| The 66 of `train_candidates_v2` | 0 | 0.600 | 0 (highest 0.887) |
| `validation_pairs_v4` (93) | 0 | 0.591 | 0 (highest 0.852) |
| Earlier train generation (v1, v2) | 0 | 0.600 | 1 (0.918): ExxonMobil's share of equity-company revenues against its share of total assets, 2020 |

**Repeated financial facts:** every batch 6 pair was checked against every other trial candidate of the same company.
There are 33 neighbours, 24 of them in `NegativeGuard` conflict.
- **Two ask the same fact twice, from different passages:**
  - **Pair 377** (FY2020 filing) and **batch 4's pair 290** (388 in the 560-record numbering; FY2021 filing) both ask
    Tesla's 2020 inventory write-downs, $145 million (cosine 0.972).
  - **Pair 661** (FY2023) and **batch 5's pair 601** (682; FY2024) both ask the 10% discount factor of the standardized
    measure.
  - Both are in conflict: never in one batch, never each other's negative. Each positive holds the fact, so both pairs
    are valid and kept, as for batches 3–5's repeats.
- **Same template, new answer:** pair 543 (MDL cases "approximately 250" as of March 10, 2025) against batch 4's pair
  354 (479; "over 2,000", March 5, 2021) and batch 5's pair 409 (455; March 6, 2020). All three are mutually in
  conflict.
- **A coincidence:** pair 667 (19,444 retail fuel sites) shares the figure with batch 3's pair 352 (674; $19,444 million
  of inventories). They are different facts, in conflict anyway.
- **The other 29** ask the same statement line or template for another year or measure. 20 are in conflict, because the
  multi-year statements share columns; the 9 that are not are different facts.
- **A revised repeat:** pair 177 asks batch 5's FY2022 real estate total for the wrong date; it is not included.

**Repeats across P1:** 6 included facts are now asked twice from different passages, all in conflict:
- Apple's State Aid and Walmart's comparable sales (batch 3);
- Walmart's fiscal 2022 operating cash flow and its demand-forecasting risk (batch 5);
- Tesla's inventory write-downs and ExxonMobil's discount factor (batch 6).

**Walmart's opioid litigation:**
- **Batch 6 adds 1 included pair,** the MDL count (template above).
  - The $3.3 billion accrual question was excluded as evaluation evidence.
  - Position 29's opioid paragraph gave no opioid question.
- **Across the 678,** 14 of Walmart's 115 candidates (12.2%; 12.9% before) concern this one litigation, over
  FY2020–FY2025. That is a topic concentration, not a leakage or duplicate problem.

**ExxonMobil's oil-equivalent conversion:**
- Positions 45 and 55 carry the statement, and neither's questions ask it.
- The 678 still hold 2 such candidates (batch 3's FY2021 and batch 4's FY2023), and the two are in conflict.

**Statement-line templates:** the year-balanced plan asks the same line for several year-ends, by design:
- ExxonMobil's equity-method investments (2021, 2022, 2023, 2025);
- Walmart's operating cash flow (fiscal 2020, 2022 twice, 2026) and operating lease cost (fiscal 2021, 2025, 2026);
- Tesla's total current assets (four year-ends, unchanged).

Each names its year.

## 10. Company, fiscal-year and numerical accuracy, and distribution

**Accuracy:**
- **Company:** every included question names its positive's company (68 of 68; all 84 passed the validator's company
  rule).
- **Period:**
  - 57 included questions name the filing's fiscal year or the prior year.
  - 3 name only another year, each legitimately:
    - the 2020 supply outlook in the FY2019 filing (218);
    - the 2023 column of the FY2025 filing's three-year tax table (451);
    - the fiscal 2018 column of the FY2020 cash-flow statement (468).
  - 8 name no year: risk factors and conventions repeated yearly, and Lyrica's June 2019 patent loss.
  - The 4 period errors are revised (§6).
- **Numbers:**
  - Every figure in the 55 numeric included answers is in its quote (independent check).
  - Units: 39 state one, 16 give a bare figure from a table in millions, and none states a wrong unit.

**Batch 6 by company:**

| | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|
| Planned | 12 | 12 | 12 | 12 | 12 |
| Pass validator | 14 | 19 | 17 | 20 | 23 |
| Evidence exclusions | 0 | 1 | 3 | 5 | 0 |
| Pairs built | 14 | 18 | 14 | 15 | 23 |
| **Included** | **7** | **15** | **13** | **14** | **19** |
| Included per planned passage | 0.58 | 1.25 | 1.08 | 1.17 | 1.58 |
| Batch 5 per planned passage | 1.10 | 1.10 | 0.90 | 1.00 | 1.20 |

**The final P1 included pairs and trial candidates, by company:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 included pairs | 53 | 35 | 91 | 111 | 106 | 108 | 108 | **612** |
| Share | 8.7% | 5.7% | 14.9% | 18.1% | 17.3% | 17.6% | 17.6% | |
| Trial candidates (with the 66) | 63 | 43 | 104 | 119 | 117 | 115 | 117 | **678** |
| Share | 9.3% | 6.3% | 15.3% | 17.6% | 17.3% | 17.0% | 17.3% | |

**By fiscal year:**

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch 6 planned | 9 | 8 | 6 | 6 | 5 | 7 | 16 | 3 |
| Batch 6 included | 12 | 11 | 7 | 8 | 5 | 8 | 14 | 3 |
| P1 included (612) | 80 | 76 | 89 | 90 | 76 | 87 | 98 | 16 |
| Trial candidates (678) | 90 | 86 | 98 | 96 | 85 | 98 | 108 | 17 |

**Within companies (the 678):**

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 9 | 7 | 11 | 7 | 10 | 9 | 10 | – | 63 |
| AMZN | 11 | 6 | 5 | 5 | 5 | 6 | 5 | – | 43 |
| JPM | 18 | 13 | 15 | 18 | 13 | 13 | 14 | – | 104 |
| PFE | 19 | 8 | 16 | 21 | 17 | 21 | 17 | – | 119 |
| TSLA | **24** | 14 | 13 | 10 | 12 | 16 | **28** | – | 117 |
| WMT | – | 19 | 18 | 16 | 10 | 16 | **19** | **17** | 115 |
| XOM | 9 | 19 | 20 | 19 | 18 | 17 | 15 | – | 117 |

Walmart's fiscal years end in January, so its "2026" is the year ended January 31, 2026; no other company has one.

- **Passage kind:** 37 table and 31 narrative included pairs (31 and 29 planned).
- **Question type:**
  - `financial_table` 27, `risk` 18, `accounting_policy` 7, `debt_liquidity` 5, `other` 5;
  - `revenue` 3, `tax` 2, `mdna` 1.

## 11. Provenance and consistency of the included pairs

**The trial build (scratch copy only):**
- **Sources:**
  - `pilot_v3` and `pilot_v2` with their existing decisions;
  - P1's 699 train pairs (batches 1–6), with a scratch decisions file. It holds batches 1–5's 615 decisions, re-keyed
    by `passage_id` and `question_sha256` to the 560-record pair IDs, and this batch's 84;
  - the scratch screening record.
- **Result:** 678 candidates (51 + 15 + 101 + 115 + 108 + 114 + 106 + 68).
- **Re-keying:** all 615 earlier decisions found their pair (112, 121, 100, 133 and 110 of them under an ID other than
  their file's), and the committed files were not touched.
- **The 162 validation pairs** that `build-pairs` also builds have no decision and are excluded, as before.

**Provenance on all 68 included records:** the same 11-field block as batches 1–5, identical to batch 5's. It names:
- corpus `ff9dadf3…4004` (`corpus_v1@1.2`);
- preparation `bge_base_ft_v3`, `21c2293a…`;
- provider `groq`, model `qwen/qwen3.8-27b`;
- prompt `sec-qgen-v2`, `f42bdc54…`;
- parameters `42d5fe54…`;
- validator `sec-qcheck-v3`;
- origin `build-pairs`.

No field is missing or empty.

**Checks run on the trial set:**
- `verify_candidate_provenance` traces all 678 back to their records and decisions.
- `load_candidates`, the trainer's fail-closed loader, loads all 678.
- `load_validation_set` loads all 93 pairs of `validation_pairs_v4` apart from the 678.
- **The 66 are unchanged:** a rebuild of `train_candidates_v2`'s sources in scratch gives byte-identical files,
  `908d4c79…` and `c8d38f96…`.

## 12. Quality across the six batches

| | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Batch 5 | Batch 6 |
|---|---:|---:|---:|---:|---:|---:|
| Planned passages | 100 | 100 | 100 | 100 | 100 | 60 |
| Done / failed | 99 / 1 | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 60 / 0 |
| Tokens per planned passage | 1,483 | 1,467 | 1,463 | 1,449 | 1,465 | 1,453 |
| Pass the validator | 151 (76.3%) | 159 (79.5%) | 154 (77.4%) | 168 (84.0%) | 159 (79.5%) | 93 (77.5%) |
| False validator rejections | 3 (1.5%) | 4 (2.0%) | 4 (2.0%) | 4 (2.0%) | 3 (1.5%) | 3 (2.5%) |
| Evidence exclusions | 39 (25.8%) | 35 (22.0%) | 28 (18.2%) | 35 (20.8%) | 39 (24.5%) | 9 (9.7%) |
| Pairs built per planned passage | 1.12 | 1.24 | 1.26 | 1.33 | 1.20 | 1.40 |
| Accept / revise / reject | 101 / 6 / 5 | 115 / 6 / 3 | 108 / 14 / 4 | 114 / 17 / 2 | 106 / 14 / 0 | 68 / 13 / 3 |
| **Acceptance** | **90.2%** | **92.7%** | **85.7%** | **85.7%** | **88.3%** | **81.0%** |
| **Included per planned passage** | **1.01** | **1.15** | **1.08** | **1.14** | **1.06** | **1.13** |
| Passages with no included pair | 30% | 20% | 23% | 23% | 27% | 17% |
| Scope errors | 2 | 1 | 5 | 13 | 6 | 6 |
| Period errors | 3 | 3 | 3 | 2 | 8 | 4 |
| Unsupported answers among the proposals | 1 | 1 | 0 | 3 | 2 | 1 (all rejected before review except batch 2's) |
| Highest semantic similarity (all pairs) | 0.8017 | 0.7929 | 0.7922 | 0.8014 | 0.7908 | 0.7905 |
| Cosine ≥ 0.90 within the batch | 2 | 4 | 3 | 0 | 2 | 1 |
| Company range of included pairs | 10–17 | 13–21 | 5–23 | 20–25 | 18–24 | 7–19 |
| HTTP 429 / 503 responses | 0 / 0 | 6 / 0 | 3 / 3 | 3 / 1 | 5 / 0 | 0 / 1 |

**Systematic differences:**
- **The generator did not drift.** Tokens per passage stay within 2.3% of batch 1, completion lengths are as before
  (mean 252), and the false rejections repeat the same two patterns in every batch.
- **Acceptance follows the sources drawn, not the batch:**
  - Batches 1–2 (90.2% and 92.7%) drew most of Apple's and Amazon's passages.
  - Batches 3–4 (85.7%) drew JPMorgan segment tables and ExxonMobil reserve tables.
  - Batch 5 (88.3%) drew fewer of them.
  - Batch 6 (81.0%) drew JPMorgan credit, boilerplate and segment passages, and three ExxonMobil supplemental and
    reserves tables.
- **The error types are stable:**
  - scope errors (33 in P1) come from JPMorgan segment and portfolio tables and ExxonMobil supplemental tables;
  - period errors (23) come from flattened multi-year tables and figures that change between filings;
  - unsupported answers (8 among 1,117 proposals) reached the included data in no batch.
- **Yield held up while acceptance fell:** batch 6's lower acceptance was offset by its low evidence-exclusion rate.

## 13. P1 as a whole: final counts, budget and remaining data-quality limitations

**Final counts:**
- **Generation:** 560 planned passages (batch 6 ran 60 of the 100 once planned), 559 done, 1,117 proposals, 884 passing
  the validator, 699 pairs.
- **Included:** **612 included P1 pairs** (87.6% of 699), and **678 trial candidates** with the 66 of
  `train_candidates_v2`.
- **Against the projections:** about 675 for a 60-passage batch 6 (batch 5 audit), 718–720 for a 100-passage one, and
  about 725 in the batch 4 audit.

**Budget:**
- About 740,000 tokens were approved for batches 2–6. Batches 2–6 used 671,490, so **68,510 remain unspent**.
- Batch 6 used 87,178 of its 150,000 cap.
- P1 used 819,774 tokens in all, batch 1 included.

**Concentration:**
- **Company:** no company exceeds 17.6% of the 678, under the 25% cap. Apple (9.3%) and Amazon (6.3%) stay
  underweight against an even 14.3%, because their supply ran out in batch 3.
- **Fiscal year:** 85–108 per full year (2026 is Walmart's alone).
- **Within companies:**
  - Tesla's FY2019 (24) and FY2025 (28) hold 44% of its 117 candidates;
  - Walmart's FY2025–FY2026 hold 36 of 115 (31.3%);
  - Amazon's FY2019 holds 11 of 43 (26%).

**Remaining data-quality limitations:**
1. **The labels are AI review, not gold.** Every decision is `reviewer: ai_review`, `gold: false`, from one reviewer in
   one pass, without human adjudication. Errors are possible in both directions; corpus look-ups were used only where
   the visible text left the answer in doubt.
2. **Error-prone sources remain in the data, filtered by the review.** JPMorgan segment and portfolio tables,
   ExxonMobil supplemental and reserves tables, and flattened multi-year tables gave most of the 70 revisions. The
   review read the visible text, so a subtler scope or year error in an accepted pair cannot be ruled out.
3. **Boilerplate and weak relevance:** 101 of the 612 included pairs carry the boilerplate note (risk factors and
   policies repeated yearly), 37 weak financial relevance and 36 inaccurate wording; 445 carry none.
4. **Company and year imbalance:** Apple and Amazon are underweight, and the year concentrations above remain. No
   further batch could correct them: the remaining Tesla and Walmart supply is mostly Tesla FY2019 and Walmart
   FY2025–FY2026.
5. **Topic and template concentration:**
   - Walmart's opioid litigation holds 14 of its 115 candidates.
   - Statement lines are asked for several year-ends (§9).
   - 6 facts are asked twice from different passages, all kept apart by the guard.
6. **False negatives:** `NegativeGuard` knows only recorded equivalents. Paraphrased evidence, rounded figures,
   changed row labels and shared answers in differently-ended ranges are not recorded (§8).
7. **Conflict density:** 913 conflicting pairs. At batch size 64 the mean fill is 0.757 (48 pairs, 47 in-batch
   negatives on average), and 4 of 14 batches are partly filled.
8. **Validator weaknesses** (frozen `sec-qcheck-v3`; recorded for a later version):
   - false rejections from years in column headers and from row labels in untagged tables (1.5–2.5% per batch);
   - the ellipsis rules for a fragment that starts with a period or a year;
   - no rule against reasoning answers (R1 matches exactly the 6);
   - no check that a quoted cell belongs to the asked row (the 2 invented-cell answers failed only because their
     quotes did).
9. **The window:** the 512-token window cuts tables. Headings before it (a segment's name) and blocks after it (the rest
   of a total) cause most scope errors, and questions about the cut part cause the unsupported answers.
10. **The semantic screen is uncalibrated:** its 0.90 threshold never bound (highest 0.8017). Paraphrase leakage relies
    on the lexical and evidence rules, and the review band holds 2 reviewed questions.
11. **Revised pairs are excluded, not repaired:** 70 revised and 17 rejected pairs are lost, and 10 of batch 6's
    passages (133 across P1) contribute no included pair.

## 14. Recommendation

- **Batch 6 quality gate: PASS** on every approved criterion, and P1 passes as a whole (§1).
- **Building `train_candidates_v3`: GO**, on your approval. The scratch build shows the result (678 candidates) and that
  every check passes on it. The conditions:
  1. **Real `build-pairs`, once** (option A), in `bge_base_ft_v3` over the 560 records, whose file must hash to
     `3d0b07ef…`. Verify that its 699 pairs equal the scratch replay's: pair ID, passage and question SHA-256.
  2. **One merged P1 decisions file**, re-keyed by `passage_id` and `question_sha256` from the six batch files, with
     the six left unchanged. It has 699 rows (612 accept, 70 revise, 17 reject), every one mapped to its pair.
  3. **`screen-questions`** in the real preparation, with the thresholds unchanged (0.90 exclusion, the 0.80–0.90
     band recorded); the band's 2 questions are already reviewed.
  4. **`build-candidates`** with a new `train_candidates_v3` config: sources `pilot_v3`, `pilot_v2` and P1 with the
     merged decisions, plus `semantic_screening`. Expect 678 candidates and a largest company share of 17.55%.
  5. **Verification:** `verify_candidate_provenance`, `load_candidates` (678), `load_validation_set` (93 apart),
     `train_candidates_v2` unchanged, and a protected-file diff. Write the result up before any training.
- **No training follows from this:** the three-seed experiment stays **NO-GO** until blockers B1–B3 of the readiness
  audit are resolved. B2 now needs only your decision on the final size (J2).

**Your decisions:**

| # | Decision | Recommendation |
|---|---|---|
| J1 | Build `train_candidates_v3` (real `build-pairs`, merged decisions, screening, `build-candidates`) | **GO**, under conditions 1–5. No API request, no training. |
| J2 | P1's final data size, for blocker B2: 678 candidates from 560 passages (the plan had 600), with 68,510 tokens unspent | Accept 678 as final, and end P1. Another batch would draw mostly Tesla FY2019 and JPMorgan, worsening the skew, for about 1.1 pairs per passage. |
| J3 | The 70 revised pairs | Keep them excluded. Rewording them would be a new, unreviewed generation step. |
| J4 | Validator weaknesses (§13, item 8) | Keep `sec-qcheck-v3` for `train_candidates_v3`. Record the R1 rejection, the quoted-cell row check, the ellipsis rules and the two false-rejection patterns for a later version. |

## Appendix: how the results were obtained

Everything below ran in a scratch copy of `data/processed/train/bge_base_ft_v3`. It used a copy of
`configs/train/data_prep_v3.yaml` that differs only in `output_dir`, the venv, and `HF_HUB_OFFLINE=1`.

```bash
python3 scripts/02_generate_queries.py revalidate --config SCRATCH/data_prep_v3_scratch.yaml --split train --dry-run
python3 scripts/02_generate_queries.py build-pairs --config SCRATCH/data_prep_v3_scratch.yaml
python3 scripts/02_generate_queries.py screen-questions --config SCRATCH/data_prep_v3_scratch.yaml \
    --name train_v3_p1_batch6_n60 --pairs --candidates train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_v2check.yaml   # byte-identical to train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_trial.yaml     # pilot_v3, pilot_v2, P1 batches 1-6; semantic_screening
```

**Other scripts:**
- A trace of every candidate through `build_pairs`' own functions. It matches the written pairs, ID for ID, and
  reproduces the batch 5 audit's trace for batches 1–5.
- The trainer's `load_candidates` and `plan_batches`, `verify_candidate_provenance`, and `load_validation_set` with
  the training pairs.
- The duplicate, repeated-fact, opioid-topic and false-negative probes, and the quote and token-position checks.
- The reasoning detector and its looser variant over all 1,117 proposals of batches 1–6, and the question-word scan.
- Corpus look-ups for pairs 177, 179, 185, 194, 195, 339 and 622.

**Evaluation data:** no evaluation file was read except the 10 permitted question texts, through the screen's own
loader and the lexical check, and no evaluation question text was printed: no batch 6 question reached the review band.
No qrels, relevance labels, judged passages, pools or audits were read.
