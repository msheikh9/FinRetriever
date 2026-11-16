# P1 batch 4 (`train_v3_p1_batch4`): generation audit and quality gate

Status: 2026-11-07.
- **What this is:**
  - an audit of the 100 generation records of plan `train_v3_p1_batch4` in `bge_base_ft_v3`;
  - an AI review (Claude) of the 133 pairs they build;
  - the reasoning-answer investigation, extended to all four batches;
  - the batch quality gate, a comparison with batches 1–3, and a reassessment of batches 5–6 and the budget.

  It uses the method of the [batch 1](train_v3_p1_batch1_generation_audit.md),
  [batch 2](train_v3_p1_batch2_generation_audit.md) and [batch 3](train_v3_p1_batch3_generation_audit.md) audits. It is
  not human annotation, and no decision here is a gold label.
- **Where the decisions are:** `data/train_review/train_v3_p1_batch4_quality_decisions.jsonl`.
  - 133 rows (114 accept, 17 revise, 2 reject), each `reviewer: ai_review` and `gold: false`.
  - Each row is bound to its `pair_id`, `passage_id` and the SHA-256 of its exact question. The pair IDs are those of
    the pair set that the current 400 train records build (§13).
  - SHA-256 `653897d6…eaad`.
- **What was not done:**
  - no question was generated and no API request was sent;
  - no generation record, plan, candidate dataset, validation set or earlier decision was changed. The records of
    batches 1–3 and their decisions are byte-identical (§2);
  - `build-pairs`, `screen-questions`, `build-candidates` and `plan` were not run in `bge_base_ft_v3`. They ran only in a
    scratch copy of the preparation, which was then discarded. No final candidate dataset was built;
  - the validator is unchanged;
  - nothing was trained, committed or pushed.
- **Protected files:** 349 files under `data/`, `configs/`, `src/` and `scripts/` were checksummed before
  and after the audit. They are unchanged; the decisions file is the only file added.
- **Earlier documents:** [the batch 3 audit](train_v3_p1_batch3_generation_audit.md),
  [the readiness audit](final_training_readiness_audit.md) and [the full-training plan](training_full_v1.md), §9.

## 1. Verdict

**The batch passes the approved quality gate on all four criteria.**

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | At least 0.8 included pairs per planned passage | 114 included for 100 planned passages: **1.14** | **Pass** |
| 2 | At least 75% AI-review acceptance | 114 of 133 built pairs: **85.7%** | **Pass** |
| 3 | Zero accepted leakage violations | **0**: no included pair fails the evidence, cross-split, lexical, semantic or duplicate checks (§8, §9) | **Pass** |
| 4 | Complete provenance for every included pair | **114 of 114** carry the full provenance block, and the fail-closed provenance check passes (§11) | **Pass** |

The further checks also hold:
- **Reasoning answers:** batch 4's proposals hold 3, more than any earlier batch, and **none reached a pair**. The
  validator rejected all three, two of them by chance (§7). No included answer or question is built from the model's
  reasoning or outside knowledge.
- **The two cut-off totals** (positions 12 and 13) produced no reasoning answer. Position 12 gave one rejected pair, a
  circular non-answer (§7).
- **Semantic screen:** no question is flagged at 0.90. One batch 4 question, pair 154 at 0.8014, falls in the
  0.80–0.90 review band. It asks a different fact from its nearest evaluation question, and the review excludes it
  anyway (§8).
- **Existing data:** `train_candidates_v2` rebuilds byte-identical, and `validation_pairs_v4` loads apart from all 504
  trial training pairs.
- **One generator:** all 400 records use one model, prompt, validator and parameter set.

**Acceptance equals batch 3's (85.7%), with a different mix of errors.** Thirteen of the 19 pairs not accepted
attribute a figure or a finding to the wrong scope, against 5 in batch 3 (§6):
- 5 come from JPMorgan's 9 segment-results passages, which gave 11 pairs (batch 3: 4 of 15);
- 7 come from ExxonMobil tables that state their scope in the visible text. Three take a proved-reserves table's
  *developed* block as total proved reserves. Three take the oil-and-gas *producing activities* results as company
  totals. One takes Canada's oil and gas operations as all of Canada;
- 1 is Pfizer's (a worldwide driver given as a U.S. one).

The review catches them, and the batch is 10.7 points above the gate.

**GO for batch 5**, subject to your approval and the conditions in §14:
1. **Same generator:** the same model, prompt, validator, parameters and plan options, with the validator unchanged.
2. **Option A:** no real `build-pairs` until all batches are generated.
3. **Company balance:** under the frozen options, batch 5 holds JPMorgan, Pfizer, Tesla, Walmart and ExxonMobil at 20
   passages each. Tesla now has only four fiscal years left (5 each); the others keep all their years (§13).
4. **Budget:** **302,200 tokens remain** of the approved ~740,000. Batch 5 needs about 147,400. Batches 5–6 together
   leave about 8,600 (2.9%) (§13).

**Batch 6 is a separate decision (H1, §14).** It is worth completing for the planned data size, at a cost in year
balance for Tesla and Walmart. The projected final count is **about 725 candidates** after batch 6, 614 after batch 5,
and 504 now.

**Unchanged:** the three-seed A0 versus A1 experiment stays **NO-GO**. Blockers B1 (a selection-aware decision rule),
B2 (the planned data size) and B3 (the GPU rehearsal) still stand.

## 2. Records and provenance

| Check | Result |
|---|---|
| Plan | `train.train_v3_p1_batch4.json`, SHA-256 `b5bc4567…9d32`, unchanged; 100 train passages |
| Records file | `generation/train.records.jsonl`, 400 lines, SHA-256 `398697df…3830` |
| Batches 1–3 | Lines 1–100 hash to `91116337…`, lines 1–200 to `79853cdb…` and lines 1–300 to `3a8952fe…217c`, as the earlier audits recorded them. The runs only appended. |
| Batch 4's records | Lines 301–400: one record per plan position, in plan order; no other plan's records; 400 distinct passages in the file |
| Outcome | **100 `done`, 0 `failed`**. Every record took 1 attempt, with no error. |
| Identity | All 400 records: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…`, record format 3 |
| Run | `train.run.json` is unchanged (`72c9e7bf…`). It binds prompt SHA-256 `f42bdc54…`, the same parameters and preparation `21c2293a…`. There is no continuation file. |
| Replies | 100 distinct response hashes (399 across the file; batch 1's failed record has none) |
| Proposals | 200: every reply held two questions |
| Validator re-run | `revalidate --dry-run`: 399 up to date, 0 changed, 1 failed record (batch 1's position 87) |
| Tokens | **144,866** (120,285 prompt, 24,581 completion), equal in the records and the four run summaries. Batches 1–4 total 586,084. |
| Completion tokens per reply | 172–483, median 241. No reply came near the 768 limit. |
| Against the dry-run estimate | About 147,700 tokens were estimated; the run used 144,866 (1.9% less) in 104 HTTP requests |

**The runs:** four invocations, from 2026-11-06 23:49 to 2026-11-07 16:49 UTC (record timestamps), from
`logs/train_v3_p1_batch4.log`. The log holds no credential.

| Run | Request budget | HTTP requests | HTTP 429 | HTTP 503 | Done | Tokens | Ended by |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 100 | 19 | 2 | 0 | 17 | 24,727 | a rate-limit wait of 411 s, after a 31 s wait was taken |
| 2 | 81 | 18 | 1 | 1 | 16 | 22,507 | a wait of 516 s |
| 3 | 63 | 63 | 0 | 0 | 63 | 91,403 | its request budget |
| 4 | 10 | 4 | 0 | 0 | 4 | 6,229 | the end of the plan |
| **All** | | **104** | **3** | **1** | **100** | **144,866** | |

**How Groq's limits were handled:**
- **The generator worked as designed.** 104 HTTP requests gave 100 usable replies, 3 HTTP 429 responses and one HTTP
  503.
  - It waited out the one 429 wait of at most 180 s and stopped cleanly at the two longer ones, without recording the
    passage.
  - The 503 was Groq's "over capacity" error, as in batch 3; it was retried after 2 s and the retry succeeded.
  - Each rerun skipped the completed passages, and the generator spent 4,035 s in pacing and rate-limit waits.
- **No passage was lost or retried** because of a rate limit or a server error.
- **What the headers show:** the last response of each run showed 1,000 requests per day (949 or more remaining) and
  8,000 tokens per minute. The 429 waits of 7–9 minutes are again longer than any per-minute window.

**Against batches 1–3 and the plan's per-batch estimate** ([training_full_v1.md](training_full_v1.md), §9):

| | Estimate | Batch 1 | Batch 2 | Batch 3 | Batch 4 |
|---|---|---:|---:|---:|---:|
| Tokens | about 143,000 | 148,284 | 146,678 | 146,256 | 144,866 |
| Questions passing the validator | about 155 | 151 | 159 | 154 | 168 |
| Pairs built | about 140 | 112 | 124 | 126 | 133 |
| AI review accepts | about 115 | 101 | 115 | 108 | 114 |
| Candidates | about 100 (range about 80–115) | 101 | 115 | 108 | 114 |

## 3. The funnel

200 proposed → **168** pass the generator's validator (84.0%) → **133** pairs built → **114** accepted by the review →
**114** included in the trial candidate set.

| | Plan | Done | Proposed | Pass validator | Pairs built | Accepted (included) |
|---|---:|---:|---:|---:|---:|---:|
| Batch 1 | 100 | 99 | 198 | 151 (76%) | 112 | 101 |
| Batch 2 | 100 | 100 | 200 | 159 (80%) | 124 | 115 |
| Batch 3 | 100 | 100 | 199 | 154 (77%) | 126 | 108 |
| Batch 4 | 100 | 100 | 200 | 168 (84%) | 133 | 114 |
| **All** | **400** | **399** | **797** | **632 (79%)** | **495** | **438** |

- **Exclusions at each step:**
  - `build-pairs` excluded 35 questions, every one for evidence leakage: `evaluation_evidence` 28 and
    `cross_split_evidence` 7 (§5). They are not training data, so they get no review decision.
  - Nothing was excluded for a validator re-check, a duplicate or evaluation overlap.
  - The trial candidate build excluded nothing beyond the 19 pairs the review did not accept.
- **Per passage:** 23 passages contribute no included pair, 40 contribute one and 37 contribute two (batch 3: 23, 46 and
  31; batch 2: 20, 45 and 35).
- **The higher validator pass rate** (84.0% against 76–80%) comes from fewer `missing_company` and `evidence_length`
  rejections (8 and 2, against 19 and 10 in batch 3). The validator is unchanged.
- **Batches 1–3 in this replay:** the scratch `build-pairs` ran over all 400 records.
  - Batches 1–3 keep the same 112, 124 and 126 pairs and the same exclusions; their traced outcomes are identical to the
    batch 3 audit's.
  - Every pair reappears with the same passage and question hash, but 263 of the 362 pair IDs differ from the
    300-record numbering, as expected (§13).
  - The trace of every candidate through `build_pairs`' own functions matches the written pairs, ID for ID.

## 4. Generator acceptance and rejection (validator `sec-qcheck-v3`)

| Reason | Count | Verdict |
|---|---:|---|
| `copied_from_passage` | 14 | 5 copy a run of prose (9–19 words), and one of these (position 19) also carries a reasoning answer. 8 are borderline: runs of 8–9 words that are mostly names, a line-item or defined term, or an 8-word fragment; they are correct by the strict rule. 1 is false (position 32): the row label of a table whose `[TABLE]` marker lies before the window, so the label exemption does not apply. |
| `missing_company` | 8 | All correct: the question does not name the company. One (position 54) also carries a reasoning answer. |
| `evidence_not_in_visible_text` | 4 | 1 correct (position 40: the quote joins one row's label to another row's values). 3 borderline (positions 71 and 85): the answers are right and the quotes faithful, but they elide text in a way the rule cannot verify. |
| `answer_not_supported` | 3 | 1 correct (position 39, a reasoning answer). 2 false (positions 11 and 91): the answers are right, but their years are column headers, not in the quote. |
| `evidence_length` | 2 | Both correct by the rule: quotes of 302 and 650 characters (limit 300). |
| `number_not_in_passage` | 1 | False (position 35): the answer is right, but the question's "December 31" is not in the visible text. |

**Totals:**
- 17 correct;
- 11 correct by the strict rule (borderline);
- **4 false rejections**, positions 11, 32, 35 and 91: 2.0% of proposals. Batches 2 and 3 had 4 each (2.0%) and batch 1
  had 3 (1.5%).

The four false rejections repeat the two known patterns:
- a row label outside a `[TABLE]` block (batch 3's positions 41 and 70);
- years or dates missing from the quote or the visible text (batch 3's 67 and 86; batch 2's 63; batch 1's 57).

**One new borderline form:** a quote that elides with "..." and resumes with a bare year or a period. Position 71's two
quotes jump across a multi-header JPMorgan table, and position 85's second fragment starts with a period. The rule
refuses a fragment after an ellipsis that does not start with a word, because a bare value could belong to any row.
That is the rule working as designed. All three answers are right, but the rejection costs three usable questions.

The validator stays as it is: changing it mid-P1 would change the generator's identity.

| Pos | Source | Reason | Question (verbatim) | Verdict |
|---:|---|---|---|---|
| 5 | XOM FY2024, table, Note 11. Leases | `copied_from_passage` | How does Exxon Mobil Corp account for assets leased for nearly all of their useful lives? | Correct by the rule, borderline: an 8-word run of the sentence. |
| 6 | JPM FY2025, table, Liquidity Risk Management | `evidence_length` | What were the long-term issuer credit ratings assigned by Moody's, Standard & Poor's, and Fitch to JPMorgan Chase & Co. as of December 31, 2025? | Correct by the rule: the quote has 650 characters (limit 300). |
| 11 | JPM FY2024, table, Business Segment & Corporate Results | `answer_not_supported` | What was JPMorgan Chase's Total Assets under custody as of December 31, 2024, and how does that compare to the figure reported for December 31, 2022? | False: the answer is right ($35,280 billion in 2024 and $28,635 billion in 2022, assets under custody); its years are column headers, not in the quote. |
| 19 | WMT FY2020, table, Item 8 | `copied_from_passage` | What was the carrying value of Walmart's long-term debt as of January 31, 2020, in millions of dollars? | Correct: the answer is a reasoning monologue about the cut-off table, not an answer (section 7); the copied run itself is borderline (9 words, mostly the date). |
| 26 | JPM FY2019, narrative, Item 1A | `copied_from_passage` | What specific litigation risks does JPMorgan Chase identify in its fiscal 2019 10-K regarding the transition from LIBOR and other benchmark rates? | Correct by the rule, borderline: an 8-word run of the sentence. |
| 29 | WMT FY2025, table, Item 2 | `missing_company` | What was the total square footage, in thousands, for Sam's Club U.S. locations as of January 31, 2025? | Correct: the question does not name the company. |
| 30 | XOM FY2020, narrative, Item 2 | `copied_from_passage` | When do the initial exploration phases for the offshore blocks A5-B, Z5-C, and Z5-D in Mozambique expire according to ExxonMobil's fiscal 2020 filing? | Correct by the rule, borderline: a 9-word run, mostly the three blocks' names. |
| 32 | PFE FY2020, narrative, Item 7 | `copied_from_passage` | What was the net-of-tax amount for total purchase accounting adjustments, acquisition-related items, discontinued operations and certain significant items for Pfizer in fiscal 2020? | False: the run is the table's row label, but the table's [TABLE] marker lies before the visible window, so the label exemption does not apply; the answer is right ($2,890 million). |
| 32 | PFE FY2020, narrative, Item 7 | `copied_from_passage` | According to the footnotes, how did Pfizer calculate the tax effect of associated pre-tax amounts for items included in Provision/(benefit) for taxes on income? | Correct by the rule, borderline: an 8-word run, mostly a line-item name. |
| 35 | XOM FY2019, table, Financial Information | `number_not_in_passage` | How much did ExxonMobil report as its net income attributable to ExxonMobil in millions of dollars for the year ended December 31, 2018? | False: the answer is right (20,840 for 2018); the question's 'December 31' is not in the visible text, whose columns read 2019, 2018 (batch 1 precedent, position 57; batch 2, position 63). |
| 36 | JPM FY2022, narrative, Note 4 – Credit Risk Concentrations | `copied_from_passage` | What amount of trading assets issued by U.S. state and municipal governments did JPMorgan Chase & Co hold as of December 31, 2022? | Correct by the rule, borderline: an 8-word run of the sentence. |
| 39 | WMT FY2021, narrative, Item 8 | `answer_not_supported` | What were the dividends paid by Walmart to noncontrolling interest in fiscal 2020? | Correct: the answer is the model's reasoning ('wait, looking at the text ... The 2020 value is cut off'), and its quote is another row (section 7). |
| 40 | XOM FY2023, narrative, Item 2 | `evidence_not_in_visible_text` | What was the total number of net wells drilled by ExxonMobil in fiscal 2023? | Correct: the quote joins the 'Total number of net wells drilled' label to another row's values; the answer itself is right (517). |
| 44 | WMT FY2022, narrative, Item 1A | `copied_from_passage` | What is the maximum potential fine Walmart Inc. could face for non-compliance with a material provision of the GDPR during fiscal 2022? | Correct: copies a 9-word run of the sentence. |
| 48 | TSLA FY2023, table, Item 8 | `missing_company` | What specific analysis did the auditors perform to evaluate the reasonableness of assumptions regarding future warranty claims? | Correct: the question does not name the company. |
| 51 | JPM FY2023, table, Business Segment Results | `missing_company` | According to the capital allocation methodology described, which specific regulatory standards and simulations are incorporated into the equity assigned to each business segment? | Correct: the question does not name the company. |
| 54 | WMT FY2020, table, Item 8 | `copied_from_passage` | What was the weighted-average discount for the dividend yield used to determine the fair value of restricted stock units granted by Walmart in fiscal 2020? | Correct: copies a 19-word run of the sentence. |
| 54 | WMT FY2020, table, Item 8 | `missing_company` | How many restricted stock and performance share units and restricted stock units were outstanding at the beginning of the period ending January 31, 2020, according to the activity table? | Correct: the question does not name the company; its answer is the model's reasoning ('Not explicitly stated in the provided text segment', section 7). |
| 57 | PFE FY2023, table, Item 8 | `missing_company` | According to the valuation assumptions table, what was the expected stock price volatility for TSRUs in 2023? | Correct: the question does not name the company. |
| 58 | TSLA FY2020, narrative, Item 8 | `copied_from_passage` | What was the total outstanding balance of MyPower customer notes receivable, net of allowance for credit losses, for Tesla at the end of fiscal 2020? | Correct: copies a 15-word run of the sentence. |
| 59 | WMT FY2024, table, Item 8 | `copied_from_passage` | How did Walmart describe the classification of its financial instruments primarily recorded in deferred income taxes and other in the Consolidated Balance Sheets? | Correct by the rule, borderline: an 8-word run, mostly a line-item name. |
| 67 | PFE FY2020, table, Item 7 | `evidence_length` | What was Pfizer's non-GAAP adjusted net income attributable to common shareholders in fiscal 2019, excluding the impact of discontinued operations and certain significant items? | Correct by the rule: the quote has 302 characters (limit 300). |
| 71 | JPM FY2022, table, Business Segment Results | `evidence_not_in_visible_text` | What was JPMorgan Chase's total net revenue for Consumer & Community Banking in fiscal 2022? | Correct by the rule, borderline: the answer is right ($55,017 million), but the quote jumps from the header to the row with '...' before a bare year, which the rule cannot attribute. |
| 71 | JPM FY2022, table, Business Segment Results | `evidence_not_in_visible_text` | How much provision for credit losses did JPMorgan Chase report for Corporate & Investment Bank in the year ended December 31, 2021? | Correct by the rule, borderline: the answer is right ($(1,174) million), but the quote jumps across the header rows with '...' before a bare year. |
| 81 | JPM FY2024, narrative, Consolidated Results Of Operations | `copied_from_passage` | How did the amortization of the fair value discount on acquired lending-related commitments associated with First Republic affect JPMorgan Chase's lending- and deposit-related fees in fiscal 2024? | Correct by the rule, borderline: an 8-word run of the sentence. |
| 84 | WMT FY2023, narrative, Item 3 | `missing_company` | How many cases were included in the In re National Prescription Opiate Litigation MDL as of March 3, 2023? | Correct: the question does not name the company. |
| 85 | XOM FY2025, narrative, Note 20. Mergers And Acquisitions | `evidence_not_in_visible_text` | What amount of cash did Exxon Mobil use to repay Denbury's credit facility and settle fractional shares, and how was the remainder of the purchase price primarily allocated? | Correct by the rule, borderline: both fragments are verbatim and in order, but the one after the ellipsis starts with a period; the answer adds 'remaining', which the passage does not say. |
| 87 | PFE FY2024, table, Item 8 | `missing_company` | What were the total assets of Haleon as of September 30, 2024, in millions? | Correct: the question does not name the company. |
| 91 | JPM FY2023, table, Consolidated Results Of Operations | `answer_not_supported` | What was JPMorgan Chase's reported Investment banking fees for the year ended December 31, 2023, and how does that compare to the amount recorded in 2021? | False: the answer is right ($6,519 million in 2023 and $13,216 million in 2021); its years are column headers, not in the quote. |
| 92 | PFE FY2023, narrative, Item 1 | `copied_from_passage` | What is the maximum out-of-pocket cap for Medicare beneficiaries set for 2025 under the Inflation Reduction Act as described by Pfizer? | Correct by the rule, borderline: a 9-word run, mostly the defined term. |
| 93 | TSLA FY2020, narrative, Item 8 | `copied_from_passage` | What was the amount of catch-up expense Tesla recognized in the second quarter of 2020 when the operational milestone of annualized Adjusted EBITDA of $4.5 billion became probable of being achieved? | Correct: copies a 16-word run of the sentence. |
| 93 | TSLA FY2020, narrative, Item 8 | `missing_company` | Which market capitalization milestones were certified by the Board of Directors for the vesting of the second and third tranches of the 2018 CEO Performance Award during the three months ended September 30, 2020? | Correct: the question does not name the company. |

## 5. Pair building: evidence-leakage exclusions (35)

`build-pairs` excludes a question whose evidence, quote or labelled figure is held exactly by:
- a blocked passage: `evaluation_evidence`;
- the visible text of a passage in another split: `cross_split_evidence`.

These rules fail closed by design, and every exclusion below follows them.

| Where the evidence is | Batch 4 | Batch 3 | Batch 2 | Batch 1 |
|---|---:|---:|---:|---:|
| An evaluation-pool or audit passage | 10 | 4 | 6 | 6 |
| An overlap-group copy of an evaluation passage | 8 | 5 | 14 | 11 |
| Only passages blocked for reusing evaluation text | 10 | 18 | 7 | 16 |
| The visible text of a validation passage | 7 | 1 | 8 | 6 |
| **All** | **35** (20.8% of 168) | **28** (18.2% of 154) | **35** (22.0% of 159) | **39** (25.8% of 151) |

- **By company:** Tesla lost 10 of its 36 questions and Walmart 9 of 32: statement lines (comprehensive income,
  investing and financing cash flows, equity, tax provision) that recur in blocked or validation passages. JPMorgan lost
  6, and Pfizer and ExxonMobil 5 each.
- **Cross-split:** the 7 validation-passage holders (Tesla's positions 53 and 88, Walmart's 14, 89 and 99, ExxonMobil's
  100) keep the training set and `validation_pairs_v4` apart. Four evaluation exclusions also have a validation copy.
- **The rule is unchanged** (decision E5).

| Pos | Source | Question (verbatim) | Answer (verbatim) | Rule | Where the evidence is |
|---:|---|---|---|---|---|
| 3 | TSLA FY2022, narrative, Item 1 | What hardware component does Tesla state runs its neural networks in its vehicles for self-driving purposes in fiscal 2022? | FSD Computer | `evaluation_evidence` | an overlap-group copy of an evaluation passage (3 blocked) |
| 10 | XOM FY2023, narrative, Business Environment | What percentage reduction in corporate-wide methane intensity does ExxonMobil target in its fiscal 2023 emission-reduction plans? | 70-80 percent | `evaluation_evidence` | an overlap-group copy of an evaluation passage (4 blocked) |
| 11 | JPM FY2024, table, Business Segment & Corporate Results | Why did JPMorgan Chase revise its prior-period Fixed Income Markets and Equity Markets net revenue figures in the fiscal 2024 report? | Certain net funding costs previously allocated to Fixed Income Markets were reclassified to Equity Markets in the fourth quarter of 2024. | `evaluation_evidence` | an evaluation-pool or audit passage (5 blocked; also 4 validation) |
| 12 | PFE FY2021, narrative, Item 7 | What was the total amount of the upfront payment and equity investment premium made by Pfizer to Arvinas in fiscal 2021? | $706 million | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 13 | TSLA FY2023, table, Item 8 | What was the comprehensive income attributable to common stockholders for Tesla, Inc. for the fiscal year ended December 31, 2023, in millions? | $15,215 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 13 | TSLA FY2023, table, Item 8 | What was the balance of Additional Paid-In Capital for Tesla, Inc. as of December 31, 2020, in millions? | $27,260 million | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 14 | WMT FY2023, table, Item 8 | What was the balance of long-term debt reported by Walmart Inc. as of January 31, 2023? | 34,649 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (3 blocked) |
| 14 | WMT FY2023, table, Item 8 | How much was the accumulated other comprehensive loss for Walmart shareholders as of January 31, 2022? | (8,766) million | `cross_split_evidence` | the visible text of 1 validation passage |
| 22 | PFE FY2023, narrative, Item 8 | What was the total amount of legally binding commitments for Pfizer as of December 31, 2023? | $5.2 billion | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 26 | JPM FY2019, narrative, Item 1A | When does JPMorgan Chase expect the LIBOR benchmark to be discontinued, according to its fiscal 2019 risk disclosures? | On or about the end of 2021 | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 33 | TSLA FY2019, narrative, Item 1A | Which vehicle models does Tesla offer leasing for directly through its local subsidiaries in the United States and in Canada as of fiscal 2019? | In the U.S., Tesla offers leasing for Model S, Model X, and Model 3; in Canada, it offers leasing for Model S and Model X. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 34 | WMT FY2026, table, Item 7 | What was the maximum amount of short-term borrowings outstanding at any month-end for Walmart in fiscal 2025? | $7,232 million | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 34 | WMT FY2026, table, Item 7 | What were the weighted-average interest rates on Walmart's short-term borrowings as of January 31, 2026 and January 31, 2025? | 4.0% and 5.3%, respectively | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 35 | XOM FY2019, table, Financial Information | What were ExxonMobil's total assets in millions of dollars for fiscal 2019? | 362,597 million dollars | `evaluation_evidence` | an overlap-group copy of an evaluation passage (8 blocked; also 2 validation) |
| 43 | TSLA FY2022, narrative, Item 8 | What was the net cash used in investing activities for Tesla, Inc. during the year ended December 31, 2022, in thousands? | 11,973 | `evaluation_evidence` | an evaluation-pool or audit passage (3 blocked) |
| 51 | JPM FY2023, table, Business Segment Results | What was the total net revenue reported by JPMorgan Chase & Co for the Corporate & Investment Bank segment for the year ended December 31, 2023? | $49,879 million | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 53 | TSLA FY2024, table, Item 8 | What was the balance of Tesla's short-term investments as of December 31, 2024? | $20,424 million | `cross_split_evidence` | the visible text of 1 validation passage |
| 53 | TSLA FY2024, table, Item 8 | How long has PricewaterhouseCoopers LLP served as Tesla's auditor? | Since 2005 | `cross_split_evidence` | the visible text of 1 validation passage |
| 56 | JPM FY2021, narrative, Business Segment Results | What was JPMorgan Chase's net revenue for the year ended December 31, 2021, and how did it change compared to the prior year? | Net revenue was $50.1 billion, a decrease of 2%. | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 72 | PFE FY2019, narrative, Item 1A | Which specific Pfizer products are mentioned as facing competition from generic manufacturers who claim their products do not infringe Pfizer's patents or that the patents are invalid? | Eliquis, Ibrance and Xeljanz | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked; also 1 validation) |
| 72 | PFE FY2019, narrative, Item 1A | What specific access challenges has Pfizer's biosimilar product Inflectra experienced among commercial payers? | Inflectra has experienced access challenges among commercial payers. | `evaluation_evidence` | only passages blocked for reusing evaluation text (2 blocked) |
| 73 | TSLA FY2021, narrative, Item 8 | What was the net cash used in investing activities by Tesla, Inc. for the year ended December 31, 2021? | (7,868) | `evaluation_evidence` | an evaluation-pool or audit passage (3 blocked) |
| 74 | WMT FY2021, narrative, Item 3 | What was the time frame during which investors in the opioids-related securities class actions against Walmart Inc. purportedly acquired stock, as stated in the fiscal 2021 filing? | March 30, 2016 through December 22, 2020 | `evaluation_evidence` | an overlap-group copy of an evaluation passage (1 blocked) |
| 79 | WMT FY2022, narrative, Item 8 | What was the total net cash used in financing activities for Walmart Inc. during the fiscal year ended January 31, 2022? | (22,828) millions | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 82 | PFE FY2021, narrative, Item 7 | What was Pfizer's worldwide revenue for Eliquis in fiscal 2021, and what was the percentage increase compared to the prior year? | $5,970 million, up 21% | `evaluation_evidence` | an evaluation-pool or audit passage (3 blocked) |
| 86 | JPM FY2021, table, Consolidated Results Of Operations | What was JPMorgan Chase's total net revenue for the year ended December 31, 2021? | $121,649 million | `evaluation_evidence` | an evaluation-pool or audit passage (5 blocked) |
| 88 | TSLA FY2024, table, Item 8 | What was the grant date fair value per share for Tesla's stock-based compensation awards during fiscal 2024? | $114.29 | `cross_split_evidence` | the visible text of 1 validation passage |
| 89 | WMT FY2020, table, Item 8 | What was Walmart's total current tax provision in millions of dollars for the fiscal year ended January 31, 2020? | $4,586 million | `cross_split_evidence` | the visible text of 2 validation passages |
| 90 | XOM FY2022, narrative, Consolidated Balance Sheet | What was the dollar amount of common stock held in treasury for ExxonMobil at the end of fiscal 2022? | (240,293) | `evaluation_evidence` | an overlap-group copy of an evaluation passage (2 blocked) |
| 90 | XOM FY2022, narrative, Consolidated Balance Sheet | How much total equity did ExxonMobil report as of December 31, 2022? | 202,473 | `evaluation_evidence` | an overlap-group copy of an evaluation passage (6 blocked) |
| 91 | JPM FY2023, table, Consolidated Results Of Operations | According to the footnotes for JPMorgan Chase's fiscal 2023 results, what specific component was included in Other income related to the First Republic acquisition, and what was its estimated value? | The estimated bargain purchase gain of $2.8 billion | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked; also 3 validation) |
| 94 | WMT FY2024, table, Item 8 | What was the weighted-average discount rate applied to finance leases by Walmart Inc. as of January 31, 2024? | 6.8% | `evaluation_evidence` | an evaluation-pool or audit passage (1 blocked) |
| 98 | TSLA FY2025, table, Item 8 | What was the comprehensive income attributable to common stockholders for Tesla, Inc. for the year ended December 31, 2025? | $4,825 million | `evaluation_evidence` | only passages blocked for reusing evaluation text (1 blocked) |
| 99 | WMT FY2025, table, Item 8 | What was Walmart's total shareholders' equity as of January 31, 2025, and how did it compare to the balance reported as of January 31, 2024? | Total shareholders' equity was $97,421 million as of January 31, 2025, compared to $90,349 million as of January 31, 2024. | `cross_split_evidence` | the visible text of 1 validation passage |
| 100 | XOM FY2020, narrative, Item 2 | What were ExxonMobil's total gross and net wells drilling counts for fiscal 2020 and fiscal 2019? | 1,297 gross and 787 net for 2020; 1,260 gross and 755 net for 2019 | `cross_split_evidence` | the visible text of 1 validation passage |

## 6. AI review of the 133 built pairs

**How each pair was judged:** as in batches 1–3, each pair was read against its positive passage's full visible text.
- **Accept:** correct company, period, figure and unit; supported by the quote; a useful financial retrieval
  query. Minor issues are noted but do not exclude.
- **Revise:** a specific error that a reworded question would fix.
- **Reject:** no financial content, or no real answer.

**Precedents, applied here:**
- a wrong unit, measure or scope fails, and a part given as the whole is revised (batch 1's pair 070; batch 3's pairs
  106, 112, 118, 123 and 229);
- a question naming no year is revised when the fact changes between filings, and accepted with a note when it does not
  (batch 1's pair 075; batch 3's pair 215);
- an ambiguous or contradictory measure is revised (batch 2's pair 232; batch 3's pair 181);
- a CIB business-line figure asked as "JPMorgan Chase's" is accepted when the business is CIB's alone (batch 3's pairs
  134 and 135), and revised when the measure exists Firm-wide with a different value (net income, total net revenue,
  provision);
- boilerplate with no financial content is rejected (batch 1's pair 106; batch 3's pair 152), and so is a non-answer
  (pilot_v3's pair 039).

**Corpus look-ups were used for:**
- pairs 154 and 155: JPMorgan's FY2024 consolidated results and the CIB segment's table;
- pair 127: JPMorgan's FY2021 Firm-wide provision for credit losses;
- pair 100: JPMorgan's FY2019 consolidated and CIB investment banking fees;
- pair 101: JPMorgan's FY2019 net income;
- pair 420: ExxonMobil's FY2019 income statement;
- pair 248: Pfizer's therapeutic areas in its FY2022–FY2025 filings;
- pairs 274 and 465 (no-year questions): Tesla's warranty risk factor in FY2019–FY2021 and ExxonMobil's CO2 pipeline
  network in FY2023–FY2024, both unchanged between filings;
- pair 109: JPMorgan's FY2019 risk types. All four are in the visible text, so the answer is supported.

**Result:** 114 accept (85.7%), 17 revise, 2 reject.

| Aspect (133 pairs) | Pass | Minor | Fail | n/a |
|---|---:|---:|---:|---:|
| Factual accuracy | 105 | 10 | 18 (101, 117, 127, 154, 155, 192, 193, 248, 269, 304, 391, 419, 420, 450, 460, 473, 474, 492) | – |
| Financial relevance | 131 | 1 | 1 (176) | – |
| Company | 133 | – | – | – |
| Period | 115 | 16 (names no year) | 2 (248, 304) | – |
| Figures (answer figures in the quote) | 99 | – | – | 34 |
| Units | 74 | 24 (no unit) | 1 (304) | 34 |

**Issues noted on accepted pairs** (one pair can carry two):
- **Boilerplate:** 13, mostly risk factors and governance, valuation or conversion statements repeated yearly.
- **Inaccurate wording:** 10. These include "segment" for JPMorgan's Equity and Fixed Income Markets businesses (pairs
  168 and 169), CIB's investment banking fees asked as the Firm's (100), "regulatory framework" for an industry standard
  (471), and a year named only as "the year following" a date (334).
- **Weak financial relevance:** 1 (staff experience in the cybersecurity section, pair 319).
- **No issue:** 90.

Batch 3: 22, 7, 11 and 72. Batch 2: 21, 7, 9 and 81. Batch 1: 19, 5, 10 and 67.

**Not accepted (19):**

| Pair | Pos | Decision | Reason |
|---:|---:|---|---|
| 101 | 61 | Revise | Wrong scope: $11.9 billion, up 1%, is the Corporate & Investment Bank segment's 2019 net income (its Banking and Markets discussion); the Firm's was $36,431 million (its FY2019 consolidated results). |
| 117 | 31 | Revise | Wrong scope: the table is the Asset & Wealth Management segment's (its mutual-fund rows, and the realignment from AWM in the note); 20,683 is AWM's headcount, while Wealth Management was one of its two businesses. |
| 127 | 56 | Revise | Wrong scope: the discussion is the Consumer & Community Banking segment's (Card, Auto, CBB, MSR results); the net benefit of $7.0 billion against a $12.3 billion expense is CCB's, while the Firm's 2021 provision was a net benefit of $9.3 billion against a $17.5 billion expense (its FY2021 consolidated results). |
| 154 | 46 | Revise | Part given as the whole: the table is the Commercial & Investment Bank segment's (after its Banking & Payments and Markets & Securities Services description); $70,114 million is CIB's 2024 revenue, the Firm's was $177,556 million (its FY2024 consolidated results). |
| 155 | 46 | Revise | Part given as the whole: $20,272 million is the Commercial & Investment Bank segment's 2023 net income (same table as pair 154); the Firm's was $49,552 million. |
| 176 | 37 | Reject | The safe-harbour statement that the risk factors are cautionary statements under the PSLRA: boilerplate with no financial content (batch 1 precedent, pair 106; batch 3, pair 152). |
| 192 | 82 | Revise | Wrong scope: the explanation is for the Prevnar family's worldwide decline (down 11% operationally); normalization of demand in Germany and other developed markets is an international driver, not a U.S. one. |
| 193 | 12 | Reject | Non-answer: the question asks which acquisition caused the $177 million charge, the passage does not name it, and the answer only restates the question (pilot_v3 precedent, pair 039). |
| 248 | 62 | Revise | Names no year, and the answer changes between filings: these four areas are the FY2024-FY2025 filings'; the FY2022-FY2023 filings also list rare diseases and anti-infectives (batch 3 precedent, pair 215). |
| 269 | 68 | Revise | Misreads the award: each tranche vests on its own market-capitalization milestone plus any one of the 16 operational milestones; $35.0 billion is the second revenue milestone, not a requirement of the second tranche. |
| 304 | 43 | Revise | Wrong year and unit: (3,364) is the 2022 column of the FY2022 filing (columns 2022, 2021, 2020), in millions; the question asks for 2023, in thousands. |
| 391 | 94 | Revise | Contradictory measure: the undiscounted total ($10,306 million) includes the imputed interest that is deducted to reach the finance lease liability; 'excluding imputed interest' describes the present value instead (batch 3 precedent, pair 181). |
| 419 | 70 | Revise | Part given as the whole: 5,223 (millions) is the consolidated subsidiaries' U.S. sales to third parties from oil and gas producing activities in 2017 (the supplemental results of operations), not ExxonMobil's total U.S. third-party sales. |
| 420 | 70 | Revise | Part given as the whole: 15,401 (millions) is the consolidated subsidiaries' depreciation and depletion for oil and gas producing activities in 2017; ExxonMobil's total was $19,893 million (its FY2019 income statement). |
| 450 | 20 | Revise | Part given as the whole: 4.3 million net acres is Canada's oil and gas operations; the in situ bitumen holdings (0.5 million net acres) are listed separately. |
| 460 | 75 | Revise | Part given as the whole: 1,208 million barrels is the developed reserves of U.S. consolidated subsidiaries (the row is under 'Developed'); total proved reserves add the undeveloped reserves, later in the table. |
| 473 | 80 | Revise | Part given as the whole: 4,998 million barrels is the developed oil-equivalent reserves of U.S. consolidated subsidiaries (the row is under 'Developed'); total proved reserves add the undeveloped reserves (as pair 460). |
| 474 | 80 | Revise | Part given as the whole: 296 billion cubic feet is Canada/Other Americas' developed natural gas reserves (the row is in the 'Developed' block); total proved reserves add the undeveloped reserves (as pair 460). |
| 492 | 15 | Revise | Part given as the whole: $66,880 million is the consolidated subsidiaries' revenue from oil and gas producing activities, which the passage says excludes other Upstream activities (transportation, LNG, power and others), not the Upstream function's revenue. |

**Patterns:**
- **Scope:** 13, against 5 in batch 3 and 1 in batch 2.
  - **JPMorgan's segment passages:** 5 (101, 117, 127, 154, 155). A segment's net income, revenue, provision or
    headcount is given as the Firm's, or a business's name is put on a whole segment's figure.
  - **ExxonMobil's reserve and supplemental tables:** 7.
    - 460, 473 and 474 read the *developed* block of the proved-reserves table as total proved reserves. The block's
      "Developed" label is in the visible text.
    - 419, 420 and 492 give the producing-activities results (oil and gas only, consolidated subsidiaries) as
      ExxonMobil's totals. The passage's title, and in 492 its first sentence, state the narrower scope.
    - 450 gives Canada's oil and gas acreage as all of Canada, with the in situ bitumen holdings listed separately.
  - **Pfizer:** 1 (192), a worldwide driver given as a U.S. one.
  - Unlike batch 3, the scope is usually in the visible text: of the 13, only pair 127's segment name and 154–155's lie
    outside it. The generator ignores a stated scope as readily as a missing one.
- **Measure or reading:** 2. Pair 391 calls an undiscounted total "excluding imputed interest". Pair 269 ties a revenue
  milestone to one tranche of Tesla's CEO award.
- **Period:** 2. Pair 248 names no year for a list that changes between filings. Pair 304 names the wrong year and unit.
- **Non-answer:** 1 (193, position 12): the passage does not name the acquisition the question asks for, and the
  answer only restates the question.
- **Boilerplate:** 1 (176): the safe-harbour statement of Pfizer's FY2019 risk factors. Batch 3's position 92 was the
  same filing's forward-looking-statements preamble.
- **The model's knowledge in a question:** 0 (§7).

**JPMorgan's 9 segment-results passages (positions 11, 21, 31, 41, 46, 51, 56, 61, 71):**
- **The source:** they gave 11 of JPMorgan's 25 pairs; three of the passages gave none (11, 51 and 71, all through
  evidence exclusions and validator rejections).
- **The outcome:** 6 accepted and 5 revised. JPMorgan's other 11 passages gave 14 pairs, all accepted.
- **The same pattern as batch 3:** 12 segment passages gave 15 pairs there, 4 revised. Segment passages remain
  JPMorgan's weak point, and the review catches the errors.
- **What is accepted:**
  - questions that name the business as the table does (CCB households, Consumer & Business Banking deposits, Wealth
    Management client advisors);
  - CIB-only lines (debt underwriting fees, Markets revenue), with a wording note where "segment" is used for a
    business.

Every pair, verbatim (pair numbers are those of the 400-record pair set; positions are batch 4's plan positions):

| Pair | Pos | Source | Type | Question (verbatim) | Answer (verbatim) | Decision | Note |
|---:|---:|---|---|---|---|---|---|
| 100 | 61 | JPM FY2019, table, Business Segment Results | financial_table | What were the debt underwriting fees for JPMorgan Chase in fiscal 2019, and what percentage change did they represent compared to the prior year? | $3.5 billion, up 8% | Accept | Correct for the CIB segment, which earns the Firm's investment banking fees (batch 3 precedent, pairs 134-135); the Firm's consolidated fees differ slightly ($7,501 million in total against CIB's $7,575 million). |
| 101 | 61 | JPM FY2019, table, Business Segment Results | profitability | What was the net income reported by JPMorgan Chase for the year ended December 31, 2019, and how did it compare to 2018? | $11.9 billion, up 1% | **Revise** | Wrong scope: $11.9 billion, up 1%, is the Corporate & Investment Bank segment's 2019 net income (its Banking and Markets discussion); the Firm's was $36,431 million (its FY2019 consolidated results). |
| 109 | 96 | JPM FY2019, narrative, Firmwide Risk Management | risk | What are the four main risk types JPMorgan Chase & Co categorizes its risks into for fiscal 2019? | Strategic risk, Credit and investment risk, Market risk, and Operational risk | Accept | Correct; all four risk types are in the visible text, which ends inside the operational-risk definition (the quote cites only the first). |
| 110 | 96 | JPM FY2019, narrative, Firmwide Risk Management | risk | According to JPMorgan Chase & Co's fiscal 2019 filing, what factors are cited as drivers of risk? | Economic environment, regulatory and government policy, competitor and market evolution, business decisions, process and judgment error, deliberate wrongdoing, dysfunctional markets, and natural disasters | Accept | Correct; a risk-governance definition repeated yearly. |
| 117 | 31 | JPM FY2020, table, Business Segment Results | financial_table | What was the headcount for JPMorgan Chase & Co's Wealth Management segment as of December 31, 2020? | 20,683 | **Revise** | Wrong scope: the table is the Asset & Wealth Management segment's (its mutual-fund rows, and the realignment from AWM in the note); 20,683 is AWM's headcount, while Wealth Management was one of its two businesses. |
| 118 | 31 | JPM FY2020, table, Business Segment Results | financial_table | How many Wealth Management client advisors did JPMorgan Chase & Co have in fiscal 2019? | 2,419 | Accept | Correct; the 2019 column, from AWM's metrics. |
| 123 | 66 | JPM FY2020, narrative, Operational Risk Management | risk | What specific responsibilities does JPMorgan Chase's Cybersecurity and Technology Control function hold regarding the firm's information security program in fiscal 2020? | Governance and oversight of the Information Security Program, and identifying information security risk issues. | Accept | Correct; cybersecurity governance repeated yearly. |
| 124 | 66 | JPM FY2020, narrative, Operational Risk Management | other | How did JPMorgan Chase modify its operational practices in fiscal 2020 to address cybersecurity risks arising from the impact of COVID-19? | The firm increased the use of remote access and third-party video conferencing solutions, and took additional precautionary measures to mitigate cybersecurity risks. | Accept | Correct; the 2020 pandemic measures. |
| 126 | 86 | JPM FY2021, table, Consolidated Results Of Operations | revenue | How much operating lease income did JPMorgan Chase include in other income for fiscal 2021? | $4.9 billion | Accept | Correct; the 2021 footnote figure. |
| 127 | 56 | JPM FY2021, narrative, Business Segment Results | profitability | How did JPMorgan Chase's provision for credit losses in fiscal 2021 compare to the expense reported in the prior year? | The provision for credit losses was a net benefit of $7.0 billion, compared with an expense of $12.3 billion in the prior year. | **Revise** | Wrong scope: the discussion is the Consumer & Community Banking segment's (Card, Auto, CBB, MSR results); the net benefit of $7.0 billion against a $12.3 billion expense is CCB's, while the Firm's 2021 provision was a net benefit of $9.3 billion against a $17.5 billion expense (its FY2021 consolidated results). |
| 129 | 21 | JPM FY2021, table, Business Segment Results | financial_table | What was JPMorgan Chase's Consumer & Business Banking average deposit level in billions of dollars for the year ended December 31, 2021? | $1,035.4 billion | Accept | Correct; the CCB segment's Consumer & Business Banking. |
| 130 | 21 | JPM FY2021, table, Business Segment Results | financial_table | How many CCB households, in millions, did JPMorgan Chase report as of December 31, 2020? | 63.4 million | Accept | Correct; the 2020 column. |
| 143 | 36 | JPM FY2022, narrative, Note 4 – Credit Risk Concentrations | risk | What percentage of the 'All other' credit risk category did SPEs represent at JPMorgan Chase & Co as of December 31, 2022? | approximately 95% | Accept | Correct. |
| 144 | 1 | JPM FY2022, narrative, Note 5 – Derivative Instruments | financial_table | What were the total gross derivative payables recognized on JPMorgan Chase's consolidated balance sheets for fiscal 2022 and fiscal 2021? | $637,513 for fiscal 2022 and $535,044 for fiscal 2021 | Accept | Correct; both years; no unit (millions). |
| 145 | 1 | JPM FY2022, narrative, Note 5 – Derivative Instruments | debt_liquidity | How much cash collateral was netted within net derivatives payable at JPMorgan Chase at December 31, 2022? | $69.2 billion | Accept | Correct. |
| 151 | 16 | JPM FY2023, narrative, Credit And Investment Risk Management | risk | What specific factors does JPMorgan Chase define centrally for stress testing scenarios in its fiscal 2023 credit risk management framework? | Macroeconomic factors | Accept | Correct; stress-testing governance repeated yearly. |
| 152 | 16 | JPM FY2023, narrative, Credit And Investment Risk Management | mdna | How does JPMorgan Chase monitor wholesale credit risk in its fiscal 2023 reporting, and what limits are applied? | Monitored regularly at an aggregate portfolio, industry, and individual client and counterparty level with established concentration limits | Accept | Correct; credit monitoring governance repeated yearly. |
| 153 | 81 | JPM FY2024, narrative, Consolidated Results Of Operations | revenue | What factors drove the increase in equity underwriting fees for JPMorgan Chase in fiscal 2024? | Higher industry-wide fees and wallet share gains in IPOs, and in follow-on and convertible securities offerings. | Accept | Correct; the Firm's 2024 equity underwriting drivers. |
| 154 | 46 | JPM FY2024, table, Business Segment & Corporate Results | revenue | What was JPMorgan Chase's total net revenue for the fiscal year 2024? | $70,114 million | **Revise** | Part given as the whole: the table is the Commercial & Investment Bank segment's (after its Banking & Payments and Markets & Securities Services description); $70,114 million is CIB's 2024 revenue, the Firm's was $177,556 million (its FY2024 consolidated results). |
| 155 | 46 | JPM FY2024, table, Business Segment & Corporate Results | profitability | How much net income did JPMorgan Chase report for the year ended December 31, 2023? | $20,272 million | **Revise** | Part given as the whole: $20,272 million is the Commercial & Investment Bank segment's 2023 net income (same table as pair 154); the Firm's was $49,552 million. |
| 163 | 76 | JPM FY2025, narrative, Consolidated Results Of Operations | revenue | What factors drove the increase in principal transactions revenue for JPMorgan Chase & Co's CIB segment in fiscal 2025? | Higher Fixed Income Markets revenue (primarily Rates and Commodities) and higher Equity Markets revenue (particularly Equity Derivatives) | Accept | Correct; the CIB's principal transactions drivers. |
| 164 | 76 | JPM FY2025, narrative, Consolidated Results Of Operations | profitability | What was the primary reason for the decrease in investment securities losses for JPMorgan Chase & Co in fiscal 2025? | Lower losses on sales of securities associated with repositioning the investment securities portfolio in Treasury and CIO | Accept | Correct. |
| 168 | 41 | JPM FY2025, narrative, Business Segment & Corporate Results | revenue | What was JPMorgan Chase's total net revenue for the Equity Markets segment in fiscal 2025? | $13,250 million | Accept | Correct; Equity Markets is a business within the CIB segment, not a segment. |
| 169 | 41 | JPM FY2025, narrative, Business Segment & Corporate Results | financial_table | How did JPMorgan Chase's net interest income for the Fixed Income Markets segment change from fiscal 2023 to fiscal 2024? | It increased from $3,171 million in fiscal 2023 to $6,347 million in fiscal 2024 | Accept | Correct (the detector's R4 flag is a false positive); Fixed Income Markets is a business within the CIB segment, not a segment. |
| 171 | 6 | JPM FY2025, table, Liquidity Risk Management | risk | How did Moody's modify the outlook for JPMorgan Chase & Co. and its principal subsidiaries in November 2025, and what change occurred to J.P. Morgan SE's long-term issuer rating? | Outlook revised to stable from positive; J.P. Morgan SE's long-term issuer rating revised to Aa2 from Aa3 | Accept | Correct; the November 3, 2025 revision covered the Parent Company and three subsidiaries, not JPMorgan Chase Bank, N.A. |
| 176 | 37 | PFE FY2019, narrative, Item 1A | risk | What specific legal act do Pfizer's risk factor statements constitute under in their fiscal 2019 Form 10-K? | Private Securities Litigation Reform Act of 1995 | **Reject** | The safe-harbour statement that the risk factors are cautionary statements under the PSLRA: boilerplate with no financial content (batch 1 precedent, pair 106; batch 3, pair 152). |
| 177 | 37 | PFE FY2019, narrative, Item 1A | risk | Which two companies did Pfizer agree to combine to create a new global pharmaceutical company named Viatris? | Upjohn and Mylan | Accept | Correct; names no year; the 2019 Upjohn-Mylan agreement. |
| 180 | 2 | PFE FY2019, narrative, Item 1A | risk | What specific challenges does Pfizer face in China regarding its medicines, and how do these challenges relate to government actions? | Government imposed pricing controls affecting certain Pfizer medicines | Accept | Correct; a China pricing risk factor. |
| 181 | 2 | PFE FY2019, narrative, Item 1A | risk | According to the risk factors, how does the use of utilization management tools like step edits in the U.S. affect Pfizer's specialty pharmaceutical business? | It can lead to higher negotiated rebates or discounts to health plans and PBMs in the U.S. | Accept | Correct; a specialty-pharmaceuticals risk factor. |
| 187 | 67 | PFE FY2020, table, Item 7 | accounting_policy | How much was the purchase accounting adjustment to income from continuing operations for Pfizer in fiscal 2019? | $3,363 million | Accept | Correct; the 2019 purchase-accounting column of the non-GAAP reconciliation. |
| 192 | 82 | PFE FY2021, narrative, Item 7 | mdna | According to Pfizer's MD&A for fiscal 2021, what was the primary driver for the decline in Prevnar family sales in the U.S.? | Normalization of demand in Germany and certain other developed markets following significantly increased adult demand in 2020, among other factors | **Revise** | Wrong scope: the explanation is for the Prevnar family's worldwide decline (down 11% operationally); normalization of demand in Germany and other developed markets is an international driver, not a U.S. one. |
| 193 | 12 | PFE FY2021, narrative, Item 7 | accounting_policy | Which specific asset acquisition did Pfizer complete in the second quarter of 2021 that resulted in a $177 million charge? | An asset acquisition completed in the second quarter of 2021 | **Reject** | Non-answer: the question asks which acquisition caused the $177 million charge, the passage does not name it, and the answer only restates the question (pilot_v3 precedent, pair 039). |
| 200 | 47 | PFE FY2021, table, Item 8 | financial_table | What was the total value of Pfizer's short-term investments as of December 31, 2021, in millions? | $29,125 million | Accept | Correct. |
| 201 | 47 | PFE FY2021, table, Item 8 | financial_table | How much did Pfizer report for held-to-maturity cash equivalents in fiscal 2020, in millions? | $89 million | Accept | Correct; the 2020 column. |
| 215 | 42 | PFE FY2022, table, Item 8 | financial_table | What was the amount of cash and cash equivalents and restricted cash and cash equivalents at the end of the period for Pfizer Inc. for the year ended December 31, 2022? | $468 | Accept | Correct; no unit (millions). |
| 216 | 42 | PFE FY2022, table, Item 8 | financial_table | How much cash did Pfizer Inc. pay for income taxes during the period for the year ended December 31, 2021? | $7,427 | Accept | Correct; the 2021 column; no unit (millions). |
| 217 | 7 | PFE FY2022, table, Item 8 | financial_table | What was Pfizer's income from continuing operations before provision for taxes on income in the United States for the year ended December 31, 2022? | $5,032 million | Accept | Correct; the United States line for 2022. |
| 218 | 7 | PFE FY2022, table, Item 8 | accounting_policy | According to Pfizer's fiscal 2022 reporting, what method was used to determine the fair value of the intangible assets discussed? | the multi-period excess earnings method, also known as the discounted cash flow method | Accept | Correct; 'the intangible assets discussed' leans on the passage (the impaired asset is named before the window). |
| 219 | 77 | PFE FY2022, table, Item 8 | accounting_policy | How does Pfizer determine the valuation of Total Shareholder Return Units (TSRUs) as of the grant date? | Using a Monte Carlo simulation model | Accept | Correct; names no year; a valuation policy repeated yearly. |
| 220 | 77 | PFE FY2022, table, Item 8 | accounting_policy | What is the settlement price definition for Pfizer's Total Shareholder Return Units (TSRUs)? | The average closing price of common stock during the 20 trading days ending on the fifth or seventh anniversary of the grant | Accept | Correct; names no year; an award term repeated yearly. |
| 223 | 92 | PFE FY2023, narrative, Item 1 | risk | When does Pfizer expect the new Medicare manufacturer discount program agreement to be signed, and when do its discounting obligations begin? | Signed in March 2024; obligations begin in 2025 | Accept | Correct. |
| 233 | 57 | PFE FY2023, table, Item 8 | accounting_policy | What was the total share-based payment expense reported by Pfizer for the fiscal year ended December 31, 2023? | $525 million | Accept | Correct. |
| 234 | 22 | PFE FY2023, narrative, Item 8 | accounting_policy | What was the estimated fair value of contingent consideration for acquisitions recorded in Other noncurrent liabilities for Pfizer at year-end 2023? | $512 million | Accept | Correct. |
| 239 | 52 | PFE FY2024, narrative, Item 1C | other | Which industry standards does Pfizer align its cybersecurity policies and practices with in its fiscal 2024 reporting? | NIST (National Institute of Standards and Technology) industry standards | Accept | Correct. |
| 240 | 52 | PFE FY2024, narrative, Item 1C | risk | What specific technical safeguards does Pfizer deploy to protect its information systems from cybersecurity threats? | Firewalls, intrusion prevention and detection systems, disaster recovery capabilities, malware and ransomware prevention, access controls and data protection | Accept | Correct; names no year; cybersecurity safeguards from Item 1C. |
| 241 | 17 | PFE FY2024, table, Item 8 | financial_table | What was the total equity reported by Pfizer Inc. for the fiscal year ended December 31, 2024? | $88,497 million | Accept | Correct; a year-end balance asked 'for the fiscal year'. |
| 242 | 17 | PFE FY2024, table, Item 8 | financial_table | How much was the accumulated other comprehensive loss for Pfizer Inc. shareholders as of December 31, 2023? | $(7,961) million | Accept | Correct; the prior-year column. |
| 243 | 87 | PFE FY2024, table, Item 8 | accounting_policy | How did Pfizer account for its investment in Haleon after reclassifying it to short-term investments? | As an equity investment with a readily determinable fair value | Accept | Correct; the Haleon investment's accounting. |
| 247 | 62 | PFE FY2025, narrative, Item 1 | mdna | What amount of R&D savings from Pfizer's cost base realignment in fiscal 2025 is scheduled for reinvestment into R&D programs in fiscal 2026? | approximately $500 million | Accept | Correct. |
| 248 | 62 | PFE FY2025, narrative, Item 1 | other | Which therapeutic areas does Pfizer identify as the primary focus of its research and development operations? | oncology, internal medicine, vaccines, and inflammation and immunology | **Revise** | Names no year, and the answer changes between filings: these four areas are the FY2024-FY2025 filings'; the FY2022-FY2023 filings also list rare diseases and anti-infectives (batch 3 precedent, pair 215). |
| 252 | 97 | PFE FY2025, table, Item 8 | profitability | What was the comprehensive income attributable to Pfizer Inc. for the year ended December 31, 2025, in millions? | $7,544 million | Accept | Correct. |
| 253 | 97 | PFE FY2025, table, Item 8 | financial_table | How much did total current assets amount to for Pfizer Inc. as of December 31, 2024, in millions? | $50,358 million | Accept | Correct; the prior-year column. |
| 256 | 27 | PFE FY2025, table, Item 8 | financial_table | What was the fair value of Pfizer's international pension plan investments valued using significant unobservable inputs at the beginning of the year ended December 31, 2025? | $1,433 million | Accept | Correct; the international plans' Level 3 assets at the start of 2025. |
| 257 | 27 | PFE FY2025, table, Item 8 | accounting_policy | According to Pfizer's description of valuation methods for fiscal 2025, how are Level 3 equity securities typically valued? | Using their last available price | Accept | Correct; a valuation policy repeated yearly. |
| 260 | 33 | TSLA FY2019, narrative, Item 1A | risk | What specific adjustments did Tesla make to vehicle prices during fiscal 2019, and what were the stated reasons for these changes? | Tesla adjusted vehicle prices to reflect anticipated changes to its cost structure from optimizing its retail strategy and as a limited accommodation to customers due to a reduction in the electric vehicle federal tax credit. | Accept | Correct; the 2019 price adjustments and their reasons. |
| 269 | 68 | TSLA FY2019, table, Item 8 | financial_table | What was the required total annualized revenue in billions of dollars for the second vesting tranche of Tesla's 2018 CEO Performance Award? | $35.0 billion | **Revise** | Misreads the award: each tranche vests on its own market-capitalization milestone plus any one of the 16 operational milestones; $35.0 billion is the second revenue milestone, not a requirement of the second tranche. |
| 270 | 68 | TSLA FY2019, table, Item 8 | other | How many operational milestones had been achieved by Tesla as of December 31, 2019, relative to the 2018 CEO Performance Award? | Two operational milestones | Accept | Correct. |
| 273 | 23 | TSLA FY2020, narrative, Item 1A | debt_liquidity | What was the total aggregate principal amount of Tesla's and its subsidiaries' debt outstanding as of December 31, 2020? | $10.57 billion | Accept | Correct. |
| 274 | 23 | TSLA FY2020, narrative, Item 1A | risk | Which specific Tesla products are cited as having recently been introduced or expected to be produced at significantly greater volumes, making warranty estimates more uncertain? | Model 3, Model Y and Solar Roof | Accept | Correct; a warranty-reserve risk factor. |
| 279 | 58 | TSLA FY2020, narrative, Item 8 | accounting_policy | How did Tesla measure expected credit losses for customer notes receivable under the adoption of ASC 326? | On a collective basis as the difference between the amortized cost basis and the present value of cash flows expected to be collected | Accept | Correct; names no year; the ASC 326 policy adopted on January 1, 2020. |
| 286 | 73 | TSLA FY2021, narrative, Item 8 | debt_liquidity | How much cash did Tesla, Inc. receive from proceeds of issuances of convertible and other debt during the year ended December 31, 2020? | 9,713 | Accept | Correct; the 2020 column; no unit (millions). |
| 287 | 8 | TSLA FY2021, table, Item 8 | accounting_policy | What was the adjustment to Tesla's accumulated deficit from the adoption of ASU 2020-06 as of January 1, 2021? | 211 million | Accept | Correct; the adoption adjustment, in millions. |
| 288 | 8 | TSLA FY2021, table, Item 8 | profitability | How did the adoption of ASU 2020-06 affect Tesla's net interest expense for the year ended December 31, 2021? | Decrease by $204 million | Accept | Correct. |
| 289 | 38 | TSLA FY2021, table, Item 8 | financial_table | What was the net carrying value of Tesla's solar energy systems as of December 31, 2021? | $5,765 million | Accept | Correct. |
| 290 | 38 | TSLA FY2021, table, Item 8 | accounting_policy | How much inventory write-down did Tesla record in cost of revenues for the year ended December 31, 2020? | $145 million | Accept | Correct; the 2020 figure. |
| 295 | 3 | TSLA FY2022, narrative, Item 1 | other | Which robotic humanoid did Tesla preview in 2022 that is controlled by the same AI system used for self-driving? | Optimus | Accept | Correct; a product preview from the business section. |
| 304 | 43 | TSLA FY2022, narrative, Item 8 | debt_liquidity | How much in repayments of convertible and other debt did Tesla, Inc. report for the year ended December 31, 2023, in thousands? | 3,364 | **Revise** | Wrong year and unit: (3,364) is the 2022 column of the FY2022 filing (columns 2022, 2021, 2020), in millions; the question asks for 2023, in thousands. |
| 305 | 83 | TSLA FY2022, narrative, Item 8 | accounting_policy | What was the total stock-based compensation expense recorded by Tesla for the year ended December 31, 2022, related to the 2018 CEO Performance Award? | $66 million | Accept | Correct. |
| 306 | 83 | TSLA FY2022, narrative, Item 8 | other | How much unrecognized stock-based compensation expense did Tesla have as of December 31, 2022, for the 2021 Performance-Based Stock Option and RSU Awards? | $204 million | Accept | Correct. |
| 309 | 48 | TSLA FY2023, table, Item 8 | financial_table | What was Tesla's total current assets as of December 31, 2023, in millions? | $49,616 million | Accept | Correct. |
| 312 | 78 | TSLA FY2023, table, Item 8 | debt_liquidity | What was the present value of Tesla's operating lease obligations as of December 31, 2023? | $4,343 million | Accept | Correct. |
| 313 | 78 | TSLA FY2023, table, Item 8 | debt_liquidity | How much in additional operating lease rent payments had not yet commenced for Tesla as of December 31, 2023? | $1.53 billion | Accept | Correct. |
| 318 | 18 | TSLA FY2024, narrative, Item 1C | risk | How often does Tesla's Audit Committee receive updates on cybersecurity matters from senior management in fiscal 2024? | Quarterly | Accept | Correct; cybersecurity governance. |
| 319 | 18 | TSLA FY2024, narrative, Item 1C | other | What is the average prior work experience in information technology roles, including security and compliance, for the leaders overseeing Tesla's cybersecurity risk management in fiscal 2024? | Over 15 years | Accept | Correct; staff experience from the cybersecurity section, not Tesla's finances. |
| 323 | 88 | TSLA FY2024, table, Item 8 | accounting_policy | How much unrecognized stock-based compensation expense did Tesla report as of December 31, 2024, for performance-based grants? | $490 million | Accept | Correct. |
| 328 | 63 | TSLA FY2025, narrative, Item 1C | risk | Which specific ISO standard has Tesla's Information Security Management System been certified to conform to as of 2025? | ISO/IEC 27001:2013 | Accept | Correct. |
| 329 | 63 | TSLA FY2025, narrative, Item 1C | risk | What are the four overarching and interconnected stages of Tesla's incident response and breach management processes? | 1) preparation for a cybersecurity incident, 2) detection and analysis of a security incident, 3) containment, eradication and recovery, and 4) post-incident analysis | Accept | Correct; names no year; an incident-response process repeated yearly. |
| 334 | 98 | TSLA FY2025, table, Item 8 | accounting_policy | How much stock-based compensation was recognized by Tesla, Inc. in the year following the balance date of December 31, 2022, according to the Consolidated Statements of Redeemable Noncontrolling Interests and Equity? | $2,013 million | Accept | Correct (2023's stock-based compensation in the equity statement); the question names the year only as 'the year following' December 31, 2022. |
| 338 | 28 | TSLA FY2025, table, Item 8 | debt_liquidity | What was the total long-term portion of Tesla's operating lease obligations as of December 31, 2025? | $5,389 million | Accept | Correct. |
| 339 | 28 | TSLA FY2025, table, Item 8 | other | How much in aggregate rent payments did Tesla exclude from its lease liability table for operating leases that had not yet commenced as of December 31, 2025? | $849 million | Accept | Correct. |
| 346 | 19 | WMT FY2020, table, Item 8 | risk | How much in impairment charges did Walmart record for the Walmart U.S. segment in fiscal 2020 related to eCommerce assets? | $0.5 billion | Accept | Correct; the Walmart U.S. segment's eCommerce impairment. |
| 347 | 89 | WMT FY2020, table, Item 8 | tax | How much deferred U.S. federal tax expense did Walmart report for the fiscal year ended January 31, 2019? | ($361) million | Accept | Correct; the fiscal 2019 column, a benefit. |
| 354 | 4 | WMT FY2021, narrative, Item 3 | risk | As of March 5, 2021, how many cases were included in the In re National Prescription Opiate Litigation MDL for Walmart Inc.? | over 2,000 cases | Accept | Correct. |
| 355 | 4 | WMT FY2021, narrative, Item 3 | risk | What is the new scheduled start date for the opioid dispensing and distribution claims trial involving Walmart Inc. that was previously set for May 10, 2021? | October 4, 2021 | Accept | Correct. |
| 356 | 74 | WMT FY2021, narrative, Item 3 | risk | In fiscal 2021, what specific legal sections did the securities class actions against Walmart Inc. allege were violated regarding the company's disclosures on opioids? | Sections 10(b) and 20(a) of the Securities Exchange Act of 1934, as amended | Accept | Correct. |
| 362 | 39 | WMT FY2021, narrative, Item 8 | financial_table | How much did Walmart pay for business acquisitions, net of cash acquired, in fiscal 2019? | (14,656) | Accept | Correct; the fiscal 2019 column; no unit (millions). |
| 366 | 44 | WMT FY2022, narrative, Item 1A | risk | Which specific state privacy laws in the U.S. did Walmart Inc. identify as significant regulatory developments affecting its data processing practices in fiscal 2022? | California Consumer Privacy Act (CCPA), California Privacy Rights Act (CPRA), and new comprehensive privacy legislation in Virginia and Colorado | Accept | Correct; a privacy-regulation risk factor. |
| 367 | 9 | WMT FY2022, narrative, Item 3 | risk | What was the status of the DOJ civil complaint against Walmart Inc. regarding opioid distribution as of November 19, 2021? | The District Court stayed further proceedings pending the decision of the United States Supreme Court in Ruan v. United States. | Accept | Correct. |
| 368 | 9 | WMT FY2022, narrative, Item 3 | risk | What is the class period for the securities class actions filed against Walmart regarding opioid disclosures? | March 30, 2016 through December 22, 2020 | Accept | Correct; names no year; the class period is fixed. |
| 371 | 79 | WMT FY2022, narrative, Item 8 | debt_liquidity | How much did Walmart Inc. report for premiums paid to extinguish debt in fiscal 2022? | (2,317) millions | Accept | Correct. |
| 378 | 49 | WMT FY2023, narrative, Item 1A | risk | What specific federal and state laws and regulations does Walmart Inc. state its U.S. health and wellness operations and Walmart Health locations are subject to regarding data privacy and health information? | Obligations and restrictions related to health information, such as those imposed under HIPAA, including data privacy and security and the sharing and interoperability of data. | Accept | Correct; a healthcare-regulation risk factor. |
| 379 | 49 | WMT FY2023, narrative, Item 1A | tax | According to Walmart's fiscal 2023 Form 10-K, what potential financial statement items may be materially affected if the ultimate tax outcome differs from the amounts recorded? | Income tax provision, net income, or cash flows in the period or periods for which such determination and settlement is made. | Accept | Correct; a tax risk statement repeated yearly. |
| 380 | 84 | WMT FY2023, narrative, Item 3 | risk | What total amount did the August 2022 judgment order Walmart and the other two defendants to pay over fifteen years regarding opioid dispensing claims? | approximately $651 million | Accept | Correct. |
| 385 | 24 | WMT FY2024, narrative, Item 1A | risk | What potential operational and financial consequences does Walmart Inc. identify for fiscal 2024 if its digital platforms suffer a successful cyberattack that disrupts operations? | Loss of sales, reductions in transactions, reputational damage, deterioration of competitive position, and liability for damage to customers and members. | Accept | Correct; a cybersecurity risk factor. |
| 386 | 24 | WMT FY2024, narrative, Item 1A | risk | According to Walmart Inc.'s risk factors for fiscal 2024, what specific internal and external factors other than external cyberattacks could lead to a compromise of its information systems? | Associate error or malfeasance, faulty password management, social engineering, vulnerabilities, irregularities, and hardware or software defects in design, coding, manufacture, or operations. | Accept | Correct; a cybersecurity risk factor. |
| 391 | 94 | WMT FY2024, table, Item 8 | financial_table | What were the total undiscounted finance lease obligations for Walmart Inc. as of the end of fiscal 2024, excluding imputed interest? | $10,306 million | **Revise** | Contradictory measure: the undiscounted total ($10,306 million) includes the imputed interest that is deducted to reach the finance lease liability; 'excluding imputed interest' describes the present value instead (batch 3 precedent, pair 181). |
| 392 | 59 | WMT FY2024, table, Item 8 | debt_liquidity | What was the fair value of Walmart's long-term debt, including amounts due within one year, as of January 31, 2024, in millions of dollars? | $38,431 million | Accept | Correct. |
| 398 | 64 | WMT FY2025, narrative, Item 1A | risk | What is the range of fines for PIPL violations in China as described in Walmart's fiscal 2025 Form 10-K? | Approximately RMB 50 million to up to 5% of the infringing company's previous year's revenues generated from within China | Accept | Correct; China's PIPL fines, from a privacy risk factor. |
| 399 | 64 | WMT FY2025, narrative, Item 1A | other | What specific consumer data does VIZIO collect from its internet-connected televisions according to Walmart's fiscal 2025 risk factors? | Viewing data from its opted-in internet-connected televisions | Accept | Correct; the VIZIO acquisition's data risk. |
| 400 | 29 | WMT FY2025, table, Item 2 | financial_table | How many Walmart U.S. retail units were owned as of January 31, 2025? | 4,039 units | Accept | Correct. |
| 402 | 99 | WMT FY2025, table, Item 8 | financial_table | According to the Consolidated Statements of Shareholders' Equity, what was the amount of consolidated net income allocated to Total Walmart Shareholders' for the period beginning February 1, 2022? | $11,680 million | Accept | Correct; fiscal 2023 (the year beginning February 1, 2022). |
| 410 | 69 | WMT FY2026, table, Item 5 | financial_table | What was the cumulative total shareholder return value for an initial $100 investment in Walmart Inc. common stock for the fiscal year ended January 31, 2026? | $272.28 | Accept | Correct. |
| 411 | 69 | WMT FY2026, table, Item 5 | other | How many holders of record for Walmart Inc. common stock were there as of March 11, 2026? | 185,190 | Accept | Correct; holders of record, correctly worded. |
| 419 | 70 | XOM FY2019, narrative, Supplemental Information On Oil And Gas  | revenue | What were Exxon Mobil's total third-party sales to external parties in the United States during fiscal 2017? | 5,223 | **Revise** | Part given as the whole: 5,223 (millions) is the consolidated subsidiaries' U.S. sales to third parties from oil and gas producing activities in 2017 (the supplemental results of operations), not ExxonMobil's total U.S. third-party sales. |
| 420 | 70 | XOM FY2019, narrative, Supplemental Information On Oil And Gas  | financial_table | How much did Exxon Mobil report as total depreciation and depletion across all regions in fiscal 2017? | 15,401 | **Revise** | Part given as the whole: 15,401 (millions) is the consolidated subsidiaries' depreciation and depletion for oil and gas producing activities in 2017; ExxonMobil's total was $19,893 million (its FY2019 income statement). |
| 423 | 100 | XOM FY2020, narrative, Item 2 | other | How many net acres of offshore holdings did ExxonMobil report in the United States at the end of 2020? | 0.4 million net acres | Accept | Correct. |
| 424 | 30 | XOM FY2020, narrative, Item 2 | other | What is the duration of the production term for ExxonMobil's operations in Equatorial Guinea for crude oil versus natural gas as of fiscal 2020? | Crude oil production period ranges from 25 to 30 years, while natural gas ranges from 25 to 50 years. | Accept | Correct; Equatorial Guinea's contract terms. |
| 433 | 65 | XOM FY2020, table, Supplemental Information On Oil And Gas  | financial_table | What was the total discounted future net cash flows relating to proved oil and gas reserves for ExxonMobil's consolidated and equity interests as of December 31, 2018? | $143,676 million | Accept | Correct; the 2018 total of the standardized measure. |
| 434 | 65 | XOM FY2020, table, Supplemental Information On Oil And Gas  | accounting_policy | How much of the discounted future net cash flows attributable to noncontrolling interests in ExxonMobil consolidated subsidiaries was recorded in 2019? | $1,064 million | Accept | Correct; the 2019 figure of the footnote. |
| 438 | 60 | XOM FY2021, narrative, Item 2 | other | What was ExxonMobil's net offshore acreage in Kazakhstan as of the end of fiscal 2021? | 0.2 million net acres | Accept | Correct. |
| 439 | 60 | XOM FY2021, narrative, Item 2 | other | How many net development wells did ExxonMobil complete in Qatar during the year ended December 31, 2021? | 4.8 net development wells | Accept | Correct. |
| 443 | 95 | XOM FY2021, table, 3. Miscellaneous Financial Information | financial_table | What was the total value of petroleum products held by ExxonMobil as of December 31, 2021, in millions of dollars? | $5,081 million | Accept | Correct. |
| 444 | 95 | XOM FY2021, table, 3. Miscellaneous Financial Information | other | How much did ExxonMobil spend on research and development expenses in fiscal 2021? | $843 million | Accept | Correct. |
| 445 | 25 | XOM FY2021, table, 5. Cash Flow Information | financial_table | What was the total cash interest paid by ExxonMobil in fiscal 2021, as reported in the cash flow information? | $1,474 million | Accept | Correct. |
| 446 | 25 | XOM FY2021, table, 5. Cash Flow Information | other | How much cash proceeds did ExxonMobil receive from the sale of its global Santoprene business in 2021? | $1.1 billion | Accept | Correct; the Santoprene sale. |
| 449 | 20 | XOM FY2022, narrative, Item 2 | other | How many net development wells did ExxonMobil complete in the Gulf of Mexico during fiscal 2022? | 0.9 net development wells | Accept | Correct; the Gulf of Mexico's 2022 completions. |
| 450 | 20 | XOM FY2022, narrative, Item 2 | other | What was ExxonMobil's total net acreage holding in Canada at the end of fiscal 2022? | 4.3 million net acres | **Revise** | Part given as the whole: 4.3 million net acres is Canada's oil and gas operations; the in situ bitumen holdings (0.5 million net acres) are listed separately. |
| 456 | 55 | XOM FY2022, table, Business Results | mdna | What was the impact of higher realizations on ExxonMobil's upstream earnings in fiscal 2022, and what were the contributing factors? | Higher realizations increased earnings by $21,290 million, reflecting tight supply and recovering demand, and favorable mark-to-market impacts of $2,800 million. | Accept | Correct. |
| 457 | 55 | XOM FY2022, table, Business Results | financial_table | How many thousands of barrels daily did ExxonMobil produce of crude oil, natural gas liquids, bitumen and synthetic oil in the United States in 2021? | 721 thousands of barrels daily | Accept | Correct; the 2021 column. |
| 460 | 75 | XOM FY2023, table, Item 2 | financial_table | What was Exxon Mobil's total proved crude oil reserves in the United States for consolidated subsidiaries at the end of fiscal 2023? | 1,208 million barrels | **Revise** | Part given as the whole: 1,208 million barrels is the developed reserves of U.S. consolidated subsidiaries (the row is under 'Developed'); total proved reserves add the undeveloped reserves, later in the table. |
| 461 | 75 | XOM FY2023, table, Item 2 | accounting_policy | How did Exxon Mobil convert natural gas volumes to an oil-equivalent basis for its proved reserves disclosure in fiscal 2023? | At six billion cubic feet per one million barrels | Accept | Correct; a conversion convention repeated yearly (the same fact as batch 3's pair 327). |
| 462 | 40 | XOM FY2023, narrative, Item 2 | financial_table | How many consolidated subsidiary productive development wells did ExxonMobil drill in Asia during fiscal 2022? | 5 | Accept | Correct; net wells, under the Consolidated Subsidiaries block (the table's heading is before the window). |
| 465 | 10 | XOM FY2023, narrative, Business Environment | other | How many miles of CO2 pipelines does ExxonMobil own and operate in the United States following the acquisition of Denbury Inc? | over 900 miles | Accept | Correct; the Denbury acquisition's CO2 pipeline network. |
| 471 | 45 | XOM FY2024, narrative, Item 1C | risk | Which regulatory framework does ExxonMobil incorporate into its cybersecurity program for fiscal 2024? | National Institute of Standards and Technology Cybersecurity Framework | Accept | Correct; the NIST Cybersecurity Framework is an industry standard, not a regulation. |
| 472 | 45 | XOM FY2024, narrative, Item 1C | other | Who is ultimately responsible for managing ExxonMobil's cybersecurity program? | Vice President of Information Technology (IT) | Accept | Correct; names no year; the program's manager (the Audit Committee oversees it). |
| 473 | 80 | XOM FY2024, table, Item 2 | financial_table | What was the oil-equivalent total of proved reserves for Exxon Mobil's consolidated United States operations in fiscal 2024? | 4,998 million barrels | **Revise** | Part given as the whole: 4,998 million barrels is the developed oil-equivalent reserves of U.S. consolidated subsidiaries (the row is under 'Developed'); total proved reserves add the undeveloped reserves (as pair 460). |
| 474 | 80 | XOM FY2024, table, Item 2 | financial_table | How many billion cubic feet of natural gas proved reserves did Exxon Mobil report for its Canada/Other Americas region in fiscal 2024? | 296 billion cubic feet | **Revise** | Part given as the whole: 296 billion cubic feet is Canada/Other Americas' developed natural gas reserves (the row is in the 'Developed' block); total proved reserves add the undeveloped reserves (as pair 460). |
| 483 | 5 | XOM FY2024, table, Note 11. Leases | financial_table | What was Exxon Mobil Corp's total operating lease cost in millions of dollars for fiscal 2024? | $4,343 million | Accept | Correct; the Operating Leases column's Total row ($4,343 million, including short-term and other); the operating lease cost line alone is $2,296 million. |
| 487 | 50 | XOM FY2025, table, Note 11. Investments, Advances, And Long | financial_table | What was the total amount of Exxon Mobil's equity method company investments and advances as of December 31, 2025? | $38,783 million | Accept | Correct. |
| 488 | 50 | XOM FY2025, table, Note 11. Investments, Advances, And Long | financial_table | What were the net long-term receivables and miscellaneous amounts for Exxon Mobil at the end of fiscal 2024? | $5,763 million | Accept | Correct; the prior-year column, net of reserves. |
| 491 | 85 | XOM FY2025, narrative, Note 20. Mergers And Acquisitions | accounting_policy | How did Exxon Mobil account for the Denbury acquisition under US GAAP, and what valuation basis was applied to the assets and liabilities? | It was accounted for as a business combination in accordance with ASC 805, requiring assets acquired and liabilities assumed to be recognized at fair values as of the acquisition date. | Accept | Correct; the Denbury acquisition's accounting, from a 609-character stub. |
| 492 | 15 | XOM FY2025, table, Supplemental Information On Oil And Gas  | financial_table | What was the total revenue for ExxonMobil's consolidated subsidiaries in the Upstream function for fiscal 2025? | $66,880 million | **Revise** | Part given as the whole: $66,880 million is the consolidated subsidiaries' revenue from oil and gas producing activities, which the passage says excludes other Upstream activities (transportation, LNG, power and others), not the Upstream function's revenue. |
| 493 | 15 | XOM FY2025, table, Supplemental Information On Oil And Gas  | accounting_policy | How much were the excluded earnings from other Upstream activities, such as LNG and transportation, for ExxonMobil in fiscal 2024? | $1.4 billion | Accept | Correct; the prior-year figure, named by year. |

## 7. Special investigation: answers from reasoning instead of the passage

**What was looked for:** an answer built from the model's own reasoning or knowledge rather than from the visible
passage. Batch 2's pair 032 is the known case. Its answer says the total is cut off, then proposes a different question.
Its only number is in the quote, so the validator passed it.

**Five checks were run:**
1. **The detector,** over every recorded proposal of batches 1–4: 796 answers (797 proposals, less one `malformed`
   reply with no answer). Its rules are unchanged:
   - **R1:** meta-language ("the provided text", "cut off", "not explicitly stated", "let's");
   - **R2:** an embedded question;
   - **R3:** a discursive answer;
   - **R4:** weak grounding.
2. **The looser grounding rule:** two or more content words, and over 25%, absent from the visible text.
3. **The two batch 4 passages at risk,** read in full: positions 12 and 13 end on a cut-off total.
4. **A broader scan for cut-off tables:** passages whose visible text ends inside a `[TABLE]` block with an empty last
   cell or a header only.
5. **The review of all 133 pairs** (§6), which reads every answer and question, plus a scan of the 114 included
   questions for words absent from the visible text (pair 190's kind).

**The detector's flags, each inspected by hand** (earlier batches' pair numbers are those of their own audits):

| Batch, position | Rules | Validator | Verdict |
|---|---|---|---|
| 4, 19 (Walmart FY2020, Item 8) | R1, R2, R3, R4 (69 of 94 words absent) | rejected (`copied_from_passage`) | **Genuine.** A 1,000-character monologue: "Not specified in the provided text segment … Let me re-read the prompt constraints … Wait … I will ask about the accounting policy". The passage ends on the debt table's header row with no values. The validator rejected it by chance. |
| 4, 39 (Walmart FY2021, Item 8) | R1, R3, R4 (11 of 27) | rejected (`answer_not_supported`) | **Genuine.** "(434) is not stated for 2020 … wait, looking at the text … The 2020 value is cut off … Let's use 'Purchase of Company stock'". The row is cut before its 2020 cell. The validator rejected it because its quote, another row, does not hold the (434) it cites. |
| 4, 54 (Walmart FY2020, Item 8) | R1, R4 (7 of 8) | rejected (`missing_company`) | **Genuine.** "Not explicitly stated in the provided text segment (table is cut off after the header/label)". The passage ends on a row label with no cells. Rejected by chance. |
| 4, 41 (pair 169) | R4 (5 of 5) | accepted | False positive: "increased", "million" and "fiscal" paraphrase the table; both figures are in the quote. |
| 1, 82 | R1, R3, R4 | rejected (`evidence_length`) | Genuine (batch 1). |
| 2, 8 (pair 032) | R1, R2, R3, R4 | accepted | Genuine: the known failure, rejected by batch 2's review. |
| 1, 46; 2, 91; 3, 70; 3, 74 | R3; R4; R2; R3 | various | False positives, as the batch 3 audit found. |

**Results:**
- **Batch 4 has 3 reasoning answers among its 200 proposals, and none reached a pair.** Batches 1–3 had 1, 1 and 0.
  All four batches together have 5 in 797 proposals (0.6%).
- **The validator caught all three in batch 4, but not by design.** Two were rejected by unrelated rules (a copied run,
  a missing company name). The third was rejected because its quote did not hold the figure it cites. Had the model
  written a fresh question and a quote of its first row, the review would have been the only protection, as for pair
  032.
- **The looser rule** flags 7 batch 4 answers: the same 3 genuine cases and 4 grounded paraphrases (pairs 144 and 169,
  and two proposals excluded for other reasons). Across batches 1–4 it flags 21: the 5 genuine cases and 16
  paraphrases.
- **Where they come from:** all 5 genuine cases come from passages that stop partway through a table row, after a label
  or header and before its value.
  - The setting is common: 69 of the 400 planned passages end inside a `[TABLE]` block with an empty last cell or a
    header only. Most of them produce ordinary questions about the rows that are visible.
  - The scan's `[TABLE]` rule catches 2 of the 5 genuine cases (batch 4's 19 and 54). The other 3 end mid-row in tables
    flattened without `[TABLE]` tags (batch 4's 39) or on a label such as "Net income" or "Total shareholders'"
    (batch 1's 82, batch 2's 8).
  - Batch 4's three come from three Walmart Item 8 statements cut mid-table: two from FY2020, one from FY2021.
- **The cut-off totals at positions 12 and 13:**
  - **Position 12** (Pfizer FY2021, ending "For 2020, the total of $") gave no reasoning answer.
    - One proposal, Arvinas's $706 million, was excluded as evaluation evidence.
    - The other became pair 193: "Which specific asset acquisition did Pfizer complete in the second quarter of 2021
      that resulted in a $177 million charge?", answered "An asset acquisition completed in the second quarter of
      2021". The passage never names the acquisition, so the answer only restates the question. It is rejected as a
      non-answer (§6). This is not reasoning, but it is the same root: a question about something the visible text does
      not give.
  - **Position 13** (Tesla FY2023, ending "Total Stockholders' Equity:") gave no reasoning answer. Both of its
    questions, comprehensive income and paid-in capital, ask about complete rows and were excluded as evaluation
    evidence.
- **Knowledge in place of the passage:**
  - **No included question carries outside knowledge.** The question scan flags 10 included questions with two or more
    words absent from the visible text. All 10 are paraphrase ("percentage change", "realignment", "US GAAP",
    "duration"), not facts (pair 190's kind).
  - **Pair 109's answer** lists JPMorgan's four risk types while quoting only the first. All four are in the visible
    text, which ends inside the operational-risk definition, so the answer is grounded and accepted.

**Conclusion:**
- **The failure is rarer than the setting but did recur in batch 4,** in 3 proposals, all from cut-off tables. None
  reached the training data, but only one was rejected by a rule aimed at it.
- **No change to the generator for P1.** The review reads every answer, and it rejected the one case that passed the
  validator (batch 2's pair 032). Changing the validator now would change the generator's identity mid-P1.
- **For batches 5–6:** run the detector and the question scan on every proposal again. The review should read answers
  from passages that end mid-table first.
- **After P1:** an R1 rejection (meta-language in an answer) is the strongest candidate rule for the next validator
  version. Among all 796 answers of batches 1–4, R1 matches exactly the 5 genuine cases and nothing else.

## 8. Evidence leakage and false-negative protections

**Leakage, on the 114 included pairs (all from the trial build, §11):**

| Check | Result |
|---|---|
| Evidence rule, re-checked at the candidate build | 0 leak passages; every included pair passes |
| Lexical overlap with the permitted evaluation questions (word Jaccard; reject at 0.8) | Highest 0.345 (batch 3: 0.263; batch 2: 0.333; batch 1: 0.346) |
| Semantic screen (`screen-questions --pairs --candidates train_candidates_v2`, in scratch) | 561 questions (the 495 pairs and the 66) against the 10 permitted question texts, `query_id` and text only. **0 flagged at 0.90.** The highest of all, 0.8017, is still batch 1's pair 005 (019 in the 400-record numbering). Of batch 4's 133: highest 0.8014 (pair 154, revised). Of its 114 included: highest 0.7558 (pair 126), 0 at or above 0.80, 3 at or above 0.75, 20 at or above 0.70; median 0.654. |
| The 0.80–0.90 review band (approved) | **One batch 4 question: pair 154** ("What was JPMorgan Chase's total net revenue for the fiscal year 2024?", 0.8014) against `q_0004`, the average liquidity coverage ratio in the fourth quarter of 2021. They share only the company: a different measure and a different year. **Not a paraphrase;** under the band rule it may stay. The review revises it anyway, for scope (§6), so it is not included. The threshold was not changed. |
| Cross-split | 7 excluded at `build-pairs`. `load_validation_set(validation_pairs_v4, training=…)` passes against all 504 trial training pairs. |
| Duplicates | None (§9) |

**Evidence within the model's window:**
- **The generator** sees only the tokenizer-visible part of each passage (512 tokens for BGE, special tokens
  included). The validator located every accepted quote inside that part.
- **The independent re-check:** 111 of the 114 included quotes lie verbatim in the positive's visible text.
  - The other 3 are found by segments or cells. Pairs 163 and 386 elide text with "…". Pair 488 joins a row label to its
    December 31, 2024 cell, skipping the 2025 cell.
  - Batch 3 had 105 of 108 verbatim.
  - Counted with the special tokens, as the window is, the latest quote ends at token 512, the window's last. 5 end
    after token 500 (batch 3: 2).
- **The window's limits:**
  - Batch 4's scope errors mostly ignore a scope that is visible (§6).
  - Three accepted answers state a unit whose header lies before the window (pairs 168, 169 and 371). The unit, millions,
    is right.

**False negatives:**
- **Equivalents:** 63 of the 114 record exact equivalents (137 passages), and 55 record uncertain ones (193
  passages). The trainer never uses any of them as a negative. Batch 3: 58 (151) and 41 (102).
- **Conflicts:** `NegativeGuard` finds 501 conflicting pairs among the 504 trial pairs, and the planner keeps them in
  different batches.
  - Involving batch 4: batch 4 with batch 4, 59; with batch 1, 17; with batch 2, 35; with batch 3, 35; with the 66, 8.
  - 93 of the 114 are in at least one conflict.
- **Batches:** with more pairs spread over more groups, the mean fill rose.

  | Batch size | Batches | Full | Mean fill (batch 3 trial) |
  |---:|---:|---:|---:|
  | 16 | 32 | 31 | 0.984 (0.938) |
  | 32 | 17 | 15 | 0.927 (0.813) |
  | 64 | 9 | 7 | 0.875 (0.677) |

- **An independent probe:** for each included pair, any other included pair's positive was checked for the answer's
  figure (4 or more digits, not a year) wherever the guard would still allow it as a negative.
  - 13 hits, all chance matches: "1934" from the Securities Exchange Act's name (7), "2,000" cases (2), the same
    $4,343 million at Tesla and ExxonMobil (3), and 5,081 in a Pfizer passage (1).
  - No unrecorded equivalent was found.
- **Repeated facts:** §9.
- **Known gaps remain:** paraphrased evidence, rounded figures and changed row labels.

## 9. Duplicate questions and repeated financial facts

| Against | Same normalised question | Highest word Jaccard | Cosine ≥ 0.90 (BGE, zero-shot) |
|---|---:|---:|---|
| Within the 114 | 0 | 0.619 | 0 (highest 0.899) |
| Batch 1's 101 | 0 | 0.778 | 1 (0.906): JPMorgan's Equity Markets revenue in 2025 against Total Markets in 2019 |
| Batch 2's 115 | 0 | 0.842 | 5 (highest 0.968): Pfizer's income taxes paid in 2021 against 2020, Tesla's current assets in 2023 against 2022, ExxonMobil's equity-method investments in 2025 against 2021, Tesla's debt proceeds in 2020 against 2023, and Pfizer's contingent consideration in 2023 against 2024 |
| Batch 3's 108 | 0 | 0.824 | 5 (highest 0.942): Tesla's current assets in 2023 against 2021, JPMorgan's Markets revenue lines, ExxonMobil's R&D in 2021 against 2023, ExxonMobil's oil-equivalent conversion (the same fact, below), and its equity-method investments in 2025 against 2024 |
| The 66 of `train_candidates_v2` | 0 | 0.778 | 1 (0.939): Tesla's present value of operating lease obligations in 2023 against 2020 |
| `validation_pairs_v4` (93) | 0 | 0.733 | 3 (highest 0.933): Tesla's current assets in 2023 against 2025, JPMorgan's gross derivative payables, and Walmart U.S. retail units owned in 2025 against owned and leased in 2022 |
| Earlier train generation (v1, v2) | 0 | 0.778 | 1 (0.939): the same Tesla lease question as the 66 |

**No question repeats another.** Every cosine match above asks a different year, measure or line, except the one
same-fact case below. The three validation-set matches ask other years or measures, and their positives are kept apart
by group (§8).

**Repeated financial facts:** every batch 4 pair was checked against every other trial candidate of the same company. A
neighbour shares a 4-or-more-digit answer figure (not a year), or has a question cosine of at least 0.90, or a word
Jaccard of at least 0.70. There are 14 such neighbours.
- **Two ask the same fact twice, from different passages.**
  - **Pair 488** (ExxonMobil's FY2025 filing) and **batch 3's pair 355** (FY2024 filing; 482 in the 400-record
    numbering) both give long-term receivables and miscellaneous, net, at December 31, 2024: $5,763 million.
  - **Pair 461** (FY2023 filing) and **batch 3's pair 327** (FY2021 filing; 436) both ask ExxonMobil's oil-equivalent
    conversion, six billion cubic feet per million barrels. The plan check flagged both passages as recorded
    equivalents.
  - Both are in `NegativeGuard` conflict, so they are never in one batch and never each other's negative. Each positive
    holds the fact, so both are valid pairs, and they are kept.
- **The other 12** ask the same template for a different year, country or line, with a different answer. 9 of them are
  in conflict anyway (adjacent filings). The 3 that are not are different facts: JPMorgan's Markets revenue lines twice,
  and ExxonMobil's equity-method investments at 2021 against 2025. None is an unrecorded equivalent.

**Same-template neighbours fell:** cosine matches of 0.90 or more against earlier batches were 21 in batch 3 (10
against batch 1, 11 against batch 2) and are 11 in batch 4 (1, 5 and 5 against batches 1–3). With five companies left,
batches 5–6 will draw these templates again, mostly statement lines of Tesla, Pfizer and ExxonMobil.

## 10. Company, fiscal-year and numerical accuracy, and distribution

**Accuracy:**
- **Company:** every question names its positive's company (114 of 114 included; all 133 passed the validator's
  company rule).
- **Period:**
  - 94 included questions name the filing's fiscal year or the prior year.
  - 4 name only another year, each a column of a multi-year statement:
    - 2023 in the FY2025 equity statement (334);
    - fiscal 2019 in the FY2021 cash-flow statement (362);
    - fiscal 2023 in the FY2025 equity statement (402);
    - 2018 in the FY2020 standardized-measure table (433).
  - 16 name no year. They cover facts that do not change between filings (valuation and award terms, cybersecurity
    processes, risk factors repeated yearly; 274 and 465 were checked across filings) and statements made in one filing
    only (the Upjohn–Mylan agreement, the Medicare discount program, the Haleon reclassification, the opioid class
    period, the Denbury accounting).

  The period errors are the 2 revised pairs (248, 304).
- **Numbers:**
  - Every figure in the 81 numeric answers is in its quote (independent check).
  - The review found 1 wrong year and unit (304), a misread award (269), and 12 correct figures given the wrong scope
    (§6).
  - Units: 63 included answers state one, 18 give a bare figure from a table "in millions", and none states a wrong
    unit.

**Distribution by company:**

| | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|
| Planned | 20 | 20 | 20 | 20 | 20 |
| Done | 20 | 20 | 20 | 20 | 20 |
| Pass validator | 31 | 34 | 36 | 32 | 35 |
| Evidence exclusions | 6 | 5 | 10 | 9 | 5 |
| Pairs built | 25 | 29 | 26 | 23 | 30 |
| **Included** | **20** | **25** | **24** | **22** | **23** |
| Batch 3 included | 15 | 12 | 18 | 23 | 18 |
| Batch 2 included | 16 | 21 | 18 | 14 | 14 |
| Batch 1 included | 11 | 16 | 15 | 15 | 10 |

| The 504 trial candidates (batches 1–4 and the 66) | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidates | 63 | 43 | 75 | 82 | 86 | 81 | 74 |
| Share | 12.5% | 8.5% | 14.9% | 16.3% | 17.1% | 16.1% | 14.7% |

**Distribution by fiscal year:**

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Planned | 10 | 13 | 15 | 15 | 15 | 15 | 15 | 2 |
| Included | 8 | 13 | 20 | 19 | 14 | 15 | 23 | 2 |
| Batch 3 included | 18 | 11 | 16 | 15 | 12 | 16 | 16 | 4 |
| Batch 2 included | 13 | 15 | 16 | 19 | 17 | 17 | 16 | 2 |
| Batch 1 included | 15 | 13 | 17 | 15 | 12 | 14 | 13 | 2 |
| The 504 together | 64 | 62 | 78 | 74 | 64 | 73 | 78 | 11 |

- **Companies:** 20–25 included each, the narrowest range so far. JPMorgan is lowest (20), through its segment
  revisions; Pfizer is highest (25).
- **The 504 together:** Tesla has the largest share, 17.1%, and Amazon the smallest, 8.5%. The cap holds easily.
- **Years:** 2019 is lowest (8 included of 10 planned). It lost 5 to evidence exclusions, and the batch 1–3 total
  already holds 46 FY2019 pairs.
- **Passage kind:** 55 table and 59 narrative pairs, from 50 planned passages of each.
- **Question type:**
  - `financial_table` 29 and `risk` 29 (batch 3: 36 and 20), `other` 18, `accounting_policy` 18;
  - `debt_liquidity` 8, `revenue` 4, `profitability` 3, `mdna` 3, `tax` 2.

## 11. Provenance and consistency of the included pairs

**The trial build (scratch copy only):**
- **Sources:**
  - `pilot_v3` and `pilot_v2` with their existing decisions;
  - P1's pair set (495 train pairs from batches 1–4), with a scratch decisions file. It holds batch 1's 112, batch 2's
    124 and batch 3's 126 decisions, re-keyed by `passage_id` and `question_sha256` to the 400-record pair IDs, and this
    batch's 133;
  - the scratch screening record.
- **Result:** 504 candidates (51 + 15 + 101 + 115 + 108 + 114).
- **Re-keying is mechanical:** all 362 earlier decisions found their pair by passage and question hash (112, 121 and 100
  of them under an ID other than their file's), and the committed files were not touched.
- **The 162 validation pairs** that `build-pairs` also builds have no decision and are excluded
  (`no_quality_decision`), as before.

**Provenance on all 114 included records:** the same block of 11 fields as batches 1–3, identical to batch 3's:
- corpus `ff9dadf3…4004` (`corpus_v1@1.2`);
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
- `verify_candidate_provenance` traces all 504 back to their records and decisions.
- `load_candidates`, the trainer's fail-closed loader, loads all 504.
- `load_validation_set` loads all 93 pairs of `validation_pairs_v4` apart from the 504.
- **The 66 are unchanged:** a rebuild of `train_candidates_v2`'s sources in scratch gives byte-identical files,
  `train_candidates.jsonl` `908d4c79…` and `candidates_log.jsonl` `c8d38f96…`.

## 12. Batch 4 against batches 1–3

| | Batch 1 | Batch 2 | Batch 3 | Batch 4 |
|---|---:|---:|---:|---:|
| Done / failed | 99 / 1 | 100 / 0 | 100 / 0 | 100 / 0 |
| Tokens per planned passage | 1,483 | 1,467 | 1,463 | 1,449 |
| Pass the validator | 151 of 198 (76.3%) | 159 of 200 (79.5%) | 154 of 199 (77.4%) | 168 of 200 (84.0%) |
| False validator rejections | 3 (1.5%) | 4 (2.0%) | 4 (2.0%) | 4 (2.0%) |
| Evidence exclusions | 39 (25.8%) | 35 (22.0%) | 28 (18.2%) | 35 (20.8%) |
| Pairs built | 112 | 124 | 126 | 133 |
| Accept / revise / reject | 101 / 6 / 5 | 115 / 6 / 3 | 108 / 14 / 4 | 114 / 17 / 2 |
| **Acceptance** | **90.2%** | **92.7%** | **85.7%** | **85.7%** |
| **Included per planned passage** | **1.01** | **1.15** | **1.08** | **1.14** |
| Passages with no included pair | 30 | 20 | 23 | 23 |
| Factual errors (fail) | 4 | 6 | 13 | 18 |
| Period errors | 3 | 3 | 3 | 2 |
| Scope errors (part as the whole, wrong business) | 2 | 1 | 5 | 13 |
| Low-value-source exclusions | 5 | 3 | 5 | 1 |
| Reasoning answers among the proposals | 1 (rejected by the validator) | 1 (rejected by the review) | 0 | 3 (rejected by the validator) |
| Highest semantic similarity | 0.8017 | 0.7929 | 0.7922 | 0.8014 (revised) |
| Cosine ≥ 0.90 within the batch | 2 | 4 | 3 | 0 |
| Company range of included pairs | 10–17 | 13–21 | 5–23 (12–23 without Amazon) | 20–25 |
| HTTP 429 / 503 responses | 0 / 0 (resume) | 6 / 0 | 3 / 3 | 3 / 1 |

**Systematic differences:**
- **The same acceptance, with more scope errors and fewer of everything else.**
  - Scope errors rose from 5 to 13: JPMorgan's segment tables, as in batch 3, and ExxonMobil's reserve and supplemental
    tables, new in batch 4 (§6).
  - Low-value sources fell from 5 to 1: no executive or board tables were drawn.
  - Rejections fell from 4 to 2.
- **The generator is unchanged:**
  - the same false-rejection rate and patterns;
  - tokens per passage within 2.3% of batch 1;
  - completion lengths slightly shorter (mean 246 against 256).
- **More reasoning answers** (3), all from cut-off tables and all rejected by the validator (§7).
- **The highest yield after batch 2** (1.14 included per passage), with the validator passing more questions (84.0%).
- **Two same-fact repeats** across batches, both separated by the guard (§9).

Nothing in batch 4 suggests a drift in the generator. Its errors follow the tables the plan drew. The review catches
them, and the gate absorbs them.

## 13. Reassessment for batches 5–6: sequencing, company balance, budget and final size

**Sequencing (option A, approved):**
- `train_v3_p1_batch4_quality_decisions.jsonl` is bound to the pair set that the current 400 records build. The scratch
  replay confirmed that pair set exactly.
- Batch 5 will append records and renumber the pair IDs again.
- At the end of P1:
  1. the real `build-pairs` runs once;
  2. all batches' decisions are carried into one decisions file keyed by the final pair IDs;
  3. the mapping by `passage_id` and `question_sha256` is mechanical (verified here for all 362 decisions of batches
     1–3).
- Batch 5's plan must be written only after its approval, and after batch 4's records exist (they do), because
  `--after-generated` excludes generated passages, not other plans.

**Company balance:** scratch plans with the frozen options, excluding all four batches' groups, find the supply below.

| Remaining eligible overlap groups | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| After batch 4 | 0 | 0 | 644 | 190 | 51 | 42 | 74 |
| After batches 5–6 under the frozen options | 0 | 0 | 604 | 150 | 11 | 2 | 34 |

- **Tesla's supply:** its remaining groups lie in four fiscal years: FY2019 29, FY2020 6, FY2024 5 and FY2025 11.
  FY2021–FY2023 are exhausted.
- **Walmart's supply:** FY2020 7, FY2021–FY2023 2–3 each, FY2024 5, FY2025 12 and FY2026 11.
- **Batch 5 under the frozen options** (scratch plan, deterministic like batches 3 and 4):
  - the five companies at 20 each (20%, under the 25% cap);
  - Tesla 5 in each of its four years, Walmart 2–4 per year, and JPMorgan, Pfizer and ExxonMobil 2–3 per year;
  - 50,034 visible tokens.
- **Batch 6 under the frozen options:**
  - Tesla would take 13 of its 20 from FY2019 and Walmart 15 of its 20 from FY2025–FY2026;
  - JPMorgan, Pfizer and ExxonMobil stay at 2–3 per year.
- **The year skew this buys** (included pairs, projected):
  - Tesla's FY2019 pairs would rise from 13 of 86 today to about 34 of 133;
  - Walmart's FY2025–FY2026 from 24 of 81 to about 47 of 126.
- **The planner's `--max-per-company-year` cannot rebalance it.** It counts the earlier batches' passages, and every
  company-year already exceeds a useful cap: a scratch plan refuses every size.
- **All 600 planned passages:** Apple 44 (7.3%), Amazon 33 (5.5%), and 103–106 (17.2–17.7%) for each of the others.

**Projected final candidates:**

| Stop after | Candidates | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch 4 (now) | 504 | 12.5% | 8.5% | 14.9% | 16.3% | 17.1% | 16.1% | 14.7% |
| Batch 5 | about 614 | 10.2–10.3% | 7.0% | 15.3–15.4% | 16.7–17.1% | 17.8–17.9% | 16.9–17.2% | 15.5–15.8% |
| Batch 6 | **about 725** (721–728) | 8.7% | 5.9–6.0% | 15.7–15.8% | 16.9–17.6% | 18.2–18.4% | 17.5–18.0% | 16.0–16.6% |

The ranges apply each company's included-per-passage yield of batches 1–4 or of batches 3–4. The batch 3 audit
projected about 714 at batch 3's yield; batch 4's higher yield raises it. The 25% cap holds in every case.

**Budget:**
- About 740,000 tokens were approved for batches 2–6. Batches 2–4 used 437,800 (146,678, 146,256 and 144,866), so
  **302,200 remain**.
- **Batch 5:** about 147,400 (122,000 prompt from its 50,034 visible tokens, 25,400 completion), with a ceiling of
  198,800 at 768 tokens per reply. About 154,800 would remain after it.
- **Batches 5–6:** about 293,600 (242,800 prompt, 50,800 completion). That leaves a margin of about 8,600 tokens
  (2.9%).
  - At batch 1's rate per passage (1,483), the margin is about 5,600; at batch 4's (1,449), about 12,400.
  - If every reply were as long as the longest yet (504 tokens), the total would be about 343,600, over the budget.
- **The model is accurate:** batch 3 came in 0.3% and batch 4 1.9% under their estimates. Batch 4's saving widened
  the margin from 1.0% to 2.9%.
- **Groq's daily limit** (1,000 requests) is far above a batch's ~104 requests. The binding limits are the unreported
  ones behind the 7–9 minute waits, which the generator handles.

**Are batches 5–6 worth completing?**
- **Batch 5: yes.**
  - It adds about 110 candidates (+22%), balanced across the five companies and across every year they have left.
  - Its quality is predictable from batches 3–4.
  - It needs about half of the remaining budget, with a wide margin.
- **Batch 6: yes, if the planned data size stands; otherwise stop after batch 5.**
  - **For:** it adds about 110 more candidates (to about 725). It completes the size that blocker B2 asks for. It keeps
    every company at or under about 18%. The approved budget covers it.
  - **Against:** most of its Tesla and Walmart passages come from one or two fiscal years. Its budget margin is about
    8,600 tokens, about six passages.
  - Stopping after batch 5 (about 614) keeps the year mix even, but B2 would then need the experiment's data size
    relabelled.

## 14. Recommendation

- **Batch 4 quality gate: PASS** on every approved criterion (§1).
- **Batch 5: GO**, on your approval, under five conditions:
  1. **One generator:** the same model, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, parameters `42d5fe54…` and
     plan options. Only `--name` changes, to `train_v3_p1_batch5`. Write the plan after approval, and verify it,
     its exclusions and a dry run before any request.
  2. **Option A:** no real `build-pairs` or `build-candidates` in `bge_base_ft_v3` until the last batch is generated and
     gated (§13).
  3. **Rate limits:** run with `--max-requests` set to the remaining passages, keep the console output with
     `tee -a`, and rerun the same command after each clean stop.
  4. **Stop rule:** batch 5 is replayed and gated on a scratch copy, as here, before batch 6 is planned or approved.
  5. **Review focus:** the review keeps reading every answer and question, with particular care for:
     - scope in JPMorgan's segment tables and ExxonMobil's reserve and supplemental tables;
     - passages that end mid-table, and the reasoning detector's flags on every proposal;
     - no-year questions whose answer may change between filings.
- **The three-seed experiment: still NO-GO**, until B1–B3 of the readiness audit are resolved.

**Your decisions:**

| # | Decision | Recommendation |
|---|---|---|
| H1 | Batch 6 (needed before its plan, after batch 5's gate): (a) run it under the frozen options; (b) stop P1 after batch 5, at about 614 candidates; (c) a smaller batch 6 under the frozen options | **(a)** if the planned data size (B2) stands: about 725 candidates, every company at or under about 18%, at the cost of Tesla's FY2019 and Walmart's FY2025–FY2026 weight (§13). **(b)** if an even year mix matters more than size. (c) only shortens the same skew. Decide after batch 5's gate. |
| H2 | Budget margin for batches 5–6 (about 2.9%, 8,600 tokens) | No change for batch 5. Before batch 6, approve a contingency of about 5% (some 15,000 tokens), or accept that batch 6 may stop a few passages short. |
| H3 | Validator weaknesses: 4 false rejections in two known patterns; 3 borderline elided quotes; reasoning answers caught only by chance | Keep the validator unchanged through P1. Record an R1 meta-language rejection, question-word grounding and the ellipsis case for a later validator version. |
| H4 | The semantic review band | Keep the approved 0.90 exclusion and the 0.80–0.90 recorded review. Batch 4 put one question in the band (pair 154): reviewed as a different fact, and excluded for scope anyway. |

## Appendix: how the results were obtained

Everything below ran in a scratch copy of `data/processed/train/bge_base_ft_v3`. It used a copy of
`configs/train/data_prep_v3.yaml` that differs only in `output_dir`, the venv, and `HF_HUB_OFFLINE=1`.

```bash
python3 scripts/02_generate_queries.py revalidate --config SCRATCH/data_prep_v3_scratch.yaml --split train --dry-run
python3 scripts/02_generate_queries.py build-pairs --config SCRATCH/data_prep_v3_scratch.yaml
python3 scripts/02_generate_queries.py screen-questions --config SCRATCH/data_prep_v3_scratch.yaml \
    --name train_v3_p1_batch4 --pairs --candidates train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_v2check.yaml   # byte-identical to train_candidates_v2
python3 scripts/02_generate_queries.py build-candidates --config SCRATCH/data_prep_v3_scratch.yaml \
    --candidates SCRATCH/candidates_trial.yaml     # pilot_v3, pilot_v2, P1 batches 1-4; semantic_screening
python3 scripts/02_generate_queries.py plan --config SCRATCH/data_prep_v3_scratch.yaml --split train \
    --name cap_after_b4_100 --size 100 --method balanced_years --max-company-share 0.25 --one-per group \
    --exclude-candidate-groups train_candidates_v2 \
    --after-generated configs/train/data_prep_v2.yaml configs/train/data_prep_v1.yaml
    # capacity only; also --size 200, --size 1001 --max-company-share 1.0 for the whole remaining supply,
    # and --max-per-company-year 8 (refused)
```

**Other scripts:**
- A trace of every candidate through `build_pairs`' own functions. It matches the written pairs, ID for ID, and
  reproduces the batch 3 audit's trace for batches 1–3.
- The trainer's `load_candidates` and `plan_batches`, `verify_candidate_provenance`, and `load_validation_set` with
  the training pairs.
- The duplicate, repeated-fact and false-negative probes, and the quote and token-position checks.
- The reasoning detector and its looser variant over all 797 proposals of batches 1–4, the broader cut-off-table scan,
  and the question-grounding scan of the included questions.
- Corpus look-ups for pairs 100, 101, 109, 127, 154, 155, 248, 274, 420 and 465.

**Evaluation data:** no evaluation file was read except the 10 permitted question texts, through the screen's own loader
and the lexical check. For the band review of pair 154, one of them, `q_0004`, was read through the screen's loader
(`query_id` and text only). No qrels, relevance labels, judged passages, pools or audits were read.
