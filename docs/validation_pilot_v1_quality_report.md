# Validation pilot `val_pilot_v1`: quality audit and validation set

Status: 2026-11-03. This audits the validation pilot (`bge_base_ft_v2`, plan
`val_pilot_v1`, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, 50
validation passages, 50 requests) and builds the validation set
`validation_pairs_v1` from it.

- Every decision below is an AI review (Claude), not human annotation, and
  none of it is a gold label.
- No request was sent beyond the 50 approved, no model was trained, and
  nothing was committed.
- **Unchanged:** the generation records, the 70 training pairs, the 66
  training candidates and their manifests, the corpus and the evaluation
  data.
- The decisions are in
  `data/validation_review/val_pilot_v1_quality_decisions.jsonl`. The set is
  in `data/processed/train/bge_base_ft_v2/validation_sets/validation_pairs_v1/`.

## 1. Decision

**The validation set is built and isolated, but it is not suitable for
choosing checkpoints.** Use it to test a validation evaluator end to end and
as a coarse sanity check; do not select or early-stop on it (§8).

- **16 of 50 passages give a validation question.**
  - The generator accepted 43.
  - The evidence rules removed 24: the training-pair rule (exact matches) 11,
    and the stricter validation rule (any match) 13 more.
  - The AI review removed 3 more.
- **16 questions are too few to rank checkpoints.**
  - A paired comparison is significant at p < 0.05 only if at least 6 of the
    16 questions move the same way: a Recall@10 difference of at least 37.5
    points.
  - Checkpoints of one run are expected to differ by a few points.
- **The current validation split cannot supply many more.**
  - Its 617 eligible sources form only 216 split units, and the pilot used
    50 of them.
  - Under the same rule the rest would give about 40–60 more questions,
    mostly JPMorgan's. Apple and Amazon are nearly exhausted.
- **Quality held.**
  - 36 of 43 questions are accepted (84%), 5 need revision and 2 are
    rejected.
  - Every question names the right company.
  - The 5 factual errors are table-scope and period misreadings, mostly in
    passages that start mid-table or mid-section.
- **Isolation holds.**
  - No positive is used by any training pair.
  - No included question's evidence is held by any train or blocked passage,
    exactly or possibly.
  - No question copies an evaluation or training question.
  - The trainer cannot read the set: by directory, by kind, by split and by
    batch, each tested.

## 2. Provenance

| Check | Result |
|---|---|
| Plan | `validation.val_pilot_v1.json`, SHA-256 `796b1873…2975`, unchanged; all 50 records follow it in plan order |
| Generation run | Bound, with no continuation. Prompt `sec-qgen-v2` (`f42bdc54…`), parameters `42d5fe54…`, model `qwen/qwen3.8-27b`, one question per passage |
| Records | 50 done, 0 failed, each on its first attempt. Sent 2026-11-03, 01:04–01:26 UTC. Completion tokens 77–330 (median 125) against the 768 limit |
| Validator | All 50 on `sec-qcheck-v3`. Re-run offline, it reproduces every acceptance and rejection |
| Records files | `validation.records.jsonl` `9289c518…`, `validation.run.json` `99bbd28c…`: read only, unchanged |
| Training data | `pairs.meta.json` `20271bbe…` (70 train pairs) and `train_candidates_v1` (manifest `6fb3aa72…`, 66 pairs) unchanged; the candidates re-verify and trace back to their records |
| Protected data | 236 files under `data/`, `configs/` and `results/`, checksummed before and after: none changed. Added: the set's 3 files, the decisions, their README and `configs/train/validation_v1.yaml` |

## 3. Rejections

| Stage | Kept | Removed | Why |
|---|---:|---:|---|
| Planned passages | 50 | | |
| Generator (`sec-qcheck-v3`) | 43 | 7 | `evidence_length` 3, `missing_company` 1, `copied_from_passage` 1, `evidence_not_in_visible_text` 1, `answer_not_supported` 1 |
| Validator, re-run | 43 | 0 | |
| Evidence, exact match (the training-pair rule) | 32 | 11 | A blocked passage holds the quote, a sentence or the labelled figure (9); a train passage's visible text does (2 more) |
| Evidence, any match (the validation rule) | 19 | 13 | A train passage (11) or a blocked passage (8) holds it uncertainly, or exactly beyond its visible window; some both |
| Question overlap | 19 | 0 | Highest Jaccard: 0.333 against an evaluation question, 0.619 against a training question (threshold 0.8) |
| AI review | **16** | 3 | Reject 01 and 41; revise 16 |

- **The AI review flagged 7 questions in all.** Four of them (07, 13, 17,
  18) were already removed by the evidence rules.
- **20 of the 36 accepted questions were removed mechanically.**

**Generator rejections:**

| Passage | Reason | Verdict |
|---|---|---|
| 11 Pfizer FY2024 | `missing_company` | Correct: the question names Eliquis, not Pfizer |
| 27 Walmart FY2020 | `copied_from_passage` | Correct by the rule: it repeats the 8-word row label. A natural query, though |
| 28, 31, 35 | `evidence_length` (448, 503, 649 characters) | Correct by the rule. 28 and 35 are table rows that repeat their headers (pilot_v3 §9.5) |
| 43 Apple FY2022 | `evidence_not_in_visible_text` | Correct: the quote joins two sentences with "…" |
| 46 Pfizer FY2021 | `answer_not_supported` | **False rejection**: the answer is right, but its years are not in the quoted table row |

**Where the excluded evidence was:**
- 24 of the 43 questions have their evidence somewhere outside the
  validation split.
  - 20 of those involve a train passage.
  - 7 involve an evaluation-pool passage itself: 09, 13, 23, 26, 32, 42, 47.
- **Why so many train passages.**
  - Neighbouring chunks of one filing can fall on opposite sides of the
    split, and a chunk's full text runs past its visible window.
  - 13 and 47: the previous chunk, a train passage, holds the quote in its
    hidden tail.
  - 07 and 37: a later filing repeats the year's table.
  - 17, 26 and 39: a train passage states the same figure, as a prior-year
    comparative (17), a balance-sheet total (26: total assets equal total
    liabilities and equity) or a reconciliation cell (39).
  - The training-pair rule sees none of these, because it rejects only exact
    matches in visible text.
- **8 of the 13 any-match exclusions hold the same fact:** 03, 07, 13, 17,
  26, 37, 39, 47.
- **5 look coincidental to the review:**
  - 12: a different $1.15 billion;
  - 18 and 34: long row labels;
  - 23: guidance phrasing and an unrelated $6.5 billion;
  - 45: risk-factor wording repeated in blocked copies.
- **Cost of the rule:** 4 of those 5 were accepted by the review, so the
  strict rule costs about 4 questions here. The rule is kept: it errs
  towards independence, which is what a validation set needs.

## 4. Per-question decisions

Each question was read against its passage's full visible text.
- **Accept:** correct company, period, figure and unit, supported by the
  quote, and a useful financial retrieval query.
- **Revise:** a specific error.
- **Reject:** no financial content, or an answer the question already gives.

The six reviewed aspects are recorded for every question (§5).

| # | Source | Question (verbatim) | Answer (verbatim) | AI review | Set | Note |
|---|---|---|---|---|---|---|
| 01 | AAPL FY2024, narrative/business | Where is Apple Inc. headquartered as of fiscal 2024? | Cupertino, California | **Reject** | Out: review | Headquarters location: no financial content (as pilot_v3 028). |
| 02 | AMZN FY2022, table/statements | What was Amazon.com, Inc.'s deferred U.S. Federal tax provision for the year ended December 31, 2022? | $6,686 million (benefit) | Accept | **In** | Correct; reads the parenthesised figure as a benefit, in millions. |
| 03 | JPM FY2021, narrative/risk | What specific forms of resolution have resulted from governmental investigations or enforcement actions involving JPMorgan Chase employees in the past? | Consent orders, deferred prosecution agreements, non-prosecution agreements and other civil or criminal sanctions | Accept | Out: possible train evidence | Correct; boilerplate risk factor repeated yearly; names no period ("in the past"). |
| 04 | PFE FY2025, table/mdna | What is the expected impact of the FASB guidance issued in November 2024 regarding disaggregated expense disclosures on Pfizer's consolidated financial statements for fiscal 2025? | This new guidance will result in increased disclosures in the notes to the financial statements. | Accept | Out: evaluation evidence, train evidence | Correct answer, but "for fiscal 2025" names the report: the guidance takes effect for 2027 annual reports and changes disclosures only. |
| 05 | TSLA FY2021, table/statements | What was the fair value of Tesla's Convertible Senior Notes as of December 31, 2021, in millions? | $2,016 million | Accept | Out: possible evaluation evidence, train evidence | Correct. |
| 06 | WMT FY2023, table/statements | What was the balance for Buildings and improvements in Walmart's property and equipment schedule as of January 31, 2023? | $104,554 million | Accept | **In** | Correct; the unit comes from the table header. |
| 07 | XOM FY2024, narrative/statements | What was Exxon Mobil's total discounted future net cash flows from oil and gas producing activities in fiscal 2024? | 151,494 | **Revise** | Out: possible train evidence | Calls the consolidated subsidiaries' 151,494 ExxonMobil's total: the visible text starts mid-table, below the heading, and with the equity companies' 34,170 the total is 185,664. The answer has no unit. |
| 08 | AAPL FY2020, table/statements | What was Apple Inc.'s effective tax rate for fiscal 2018, and how did it compare to the rate in fiscal 2020? | The effective tax rate was 18.3% in fiscal 2018 and 14.4% in fiscal 2020. | Accept | Out: evaluation evidence, train evidence | Correct; compares fiscal 2018 with fiscal 2020 from the report's three-year table. |
| 09 | AMZN FY2023, table/statements | What was the total estimated fair value of Amazon's cash, cash equivalents, and marketable securities as of December 31, 2022? | $70,026 | Accept | Out: evaluation evidence, possible train evidence | Correct prior-year figure; the answer has no unit (millions). |
| 10 | JPM FY2019, table/other | What transition method did JPMorgan Chase & Co elect when adopting the new lease accounting standard at the beginning of fiscal 2019? | The Firm elected the modified retrospective transition method, through a cumulative-effect adjustment to retained earnings without revising prior periods. | Accept | **In** | Correct. |
| 12 | TSLA FY2024, narrative/risk | What was the impact on Tesla's net income before taxes at the end of fiscal 2024 from a hypothetical 10% adverse change in foreign currency exchange rates, assuming no hedging? | $1.15 billion | Accept | Out: possible evaluation evidence | Correct. |
| 13 | WMT FY2022, narrative/mdna | What was the approximate contribution of Sam's Club eCommerce sales to comparable sales for Walmart Inc. in fiscal 2022? | approximately 2.2% | **Revise** | Out: possible evaluation evidence, possible train evidence | 2.2% is fiscal 2021's contribution; fiscal 2022's was 1.3%. The visible text starts after the "For fiscal 2021" lead-in. |
| 14 | XOM FY2025, table/business | How many net productive exploratory wells did Exxon Mobil Corp drill in total across its consolidated subsidiaries in fiscal 2025? | 4 wells | Accept | **In** | Correct. |
| 15 | AAPL FY2023, table/statements | What was Apple's total property, plant and equipment, net, as of September 30, 2023, in millions? | $43,715 million | Accept | Out: evaluation evidence, train evidence | Correct. |
| 16 | AMZN FY2024, narrative/statements | What was the total cost or amortized cost of Amazon's marketable securities as of December 31, 2024? | $101,532 | **Revise** | Out: review | $101,532 is the cost of cash, cash equivalents and marketable securities together (the table's subtotal, before restricted amounts), not of marketable securities. |
| 17 | JPM FY2020, table/mdna | What was the total provision for credit losses reported by JPMorgan Chase & Co for the year ended December 31, 2020? | $12.3 billion | **Revise** | Out: possible train evidence | $12.3 billion is the Consumer & Community Banking segment's provision (a segment result), not JPMorgan Chase's total. |
| 18 | PFE FY2022, table/mdna | What was Pfizer's reported diluted earnings per share attributable to common shareholders for the year ended December 31, 2021? | $4.06 | **Revise** | Out: possible evaluation evidence, possible train evidence | $4.06 is on the Non-GAAP Adjusted row; the question asks for reported diluted EPS. |
| 19 | TSLA FY2020, table/mdna | What was the production status of the Model Y at the Gigafactory Berlin as of the filing date for Tesla's fiscal 2020 10-K? | Constructing manufacturing facilities | Accept | **In** | Correct; the status as of the filing date. |
| 20 | WMT FY2021, table/statements | What was Walmart Inc.'s total comprehensive income attributable to the company for the fiscal year ended January 31, 2021, in millions? | $14,549 million | Accept | Out: evaluation evidence, train evidence | Correct. |
| 21 | XOM FY2023, table/other | What was the unrecognized compensation cost related to nonvested restricted awards at Exxon Mobil Corp as of December 31, 2023? | $2,120 million | Accept | **In** | Correct. |
| 22 | AAPL FY2025, table/statements | What was the weighted-average grant-date fair value per restricted stock unit for those granted by Apple Inc. in fiscal 2025? | $226.68 | Accept | **In** | Correct. |
| 23 | AMZN FY2020, narrative/mdna | What was Amazon's expected operating income range for the first quarter of 2021, and how does it compare to the first quarter of 2020? | Expected to be between $3.0 billion and $6.5 billion, compared with $4.0 billion in first quarter 2020 | Accept | Out: possible evaluation evidence, possible train evidence | Correct; guidance for the next quarter. |
| 24 | JPM FY2022, narrative/risk | What was JPMorgan Chase's total consumer credit exposure in fiscal 2022? | 1,351,352 | Accept | Out: possible evaluation evidence, train evidence | Correct; the answer has no unit (millions). |
| 25 | PFE FY2019, narrative/risk | Which specific currencies does Pfizer hedge for forecasted intercompany sales to protect against longer-term movements? | euro, Japanese yen, Chinese renminbi, U.K. pound, Canadian dollar, and Australian dollar | Accept | **In** | Correct; hedging policy repeated yearly; names no period. |
| 26 | TSLA FY2022, narrative/statements | What was Tesla's total liabilities and equity as of December 31, 2022? | $82,338 | Accept | Out: possible evaluation evidence, possible train evidence | Correct; the answer has no unit (millions). |
| 29 | AAPL FY2021, narrative/statements | What was the fair value of Apple Inc.'s corporate debt securities classified as Level 2 in the fiscal 2021 financial statements? | $84,858 | Accept | **In** | Correct; the answer has no unit (millions). |
| 30 | AMZN FY2019, table/statements | What was the adjustment amount applied to investing activities in Amazon's consolidated statements of cash flows for the year ended December 31, 2017, following the adoption of the ASU on restricted cash presentation? | $735 million | Accept | **In** | Correct; a 2017 restatement shown in the FY2019 report. |
| 32 | PFE FY2023, narrative/mdna | What was the primary driver of the decline in Pfizer's Paxlovid revenue for the year ended December 31, 2023? | A non-cash revenue reversal of $3.5 billion recorded in the fourth quarter of 2023, related to the expected return of an estimated 6.5 million treatment courses of EUA-labeled U.S. government inventory | Accept | Out: evaluation evidence, train evidence | Correct. |
| 33 | TSLA FY2023, narrative/business | What was Tesla's total worldwide employee headcount as of December 31, 2023? | 140,473 | Accept | Out: evaluation evidence | Correct. |
| 34 | WMT FY2026, table/mdna | What was Walmart's Return on Assets (ROA) for the fiscal year ended January 31, 2026? | 8.2% | Accept | Out: possible evaluation evidence, possible train evidence | Correct. |
| 36 | AAPL FY2022, narrative/statements | What was the fair value of Apple's Level 2 U.S. Treasury securities as of fiscal 2021? | 22,903 | Accept | **In** | Correct prior-year figure; "as of fiscal 2021" means its end; the answer has no unit (millions). |
| 37 | AMZN FY2021, narrative/statements | What was the net carrying amount of total acquired intangible assets for Amazon in fiscal 2021? | $5,107 | Accept | Out: possible train evidence | Correct; the answer has no unit (millions). |
| 38 | JPM FY2023, table/risk | What was the total long-term unsecured funding issuance by JPMorgan Chase's Parent Company for the year ended December 31, 2023? | $19,410 million | Accept | **In** | Correct; the Parent Company column. |
| 39 | PFE FY2020, table/mdna | What was the total purchase accounting adjustments net of tax for Pfizer Inc. for the year ended December 31, 2020? | $2,537 million | Accept | Out: possible train evidence | Correct. |
| 40 | TSLA FY2025, table/business | For fiscal 2025, what is the total initial term of the land use rights for Tesla's Gigafactory Shanghai, and how are these rights accounted for? | 50 years; treated as operating lease right-of-use assets | Accept | **In** | Correct; the footnote covers both Shanghai factories. |
| 41 | WMT FY2025, narrative/other | What does Walmart's Insider Trading Policy prohibit directors and officers from doing while in possession of material, nonpublic information? | Trading in Walmart securities | **Reject** | Out: review | Insider-trading-policy boilerplate in an Item 10 cross-reference passage; the answer follows from the question. |
| 42 | XOM FY2021, narrative/other | What was the total long-term debt reported by Exxon Mobil Corporation at December 31, 2021? | $43,428 million | Accept | Out: evaluation evidence, train evidence | Correct; the debt note's total, which includes finance lease obligations. |
| 44 | AMZN FY2025, narrative/statements | What was the approximate quantity of energy contracts subject to derivative accounting fair value measurements for Amazon as of December 31, 2025? | approximately 200 million megawatt-hours | Accept | **In** | Correct. |
| 45 | JPM FY2025, narrative/risk | What are the specific negative consequences JPMorganChase may face if it is required to admit wrongdoing in connection with resolving governmental investigations? | Disqualification from doing business with certain clients or customers or in specific jurisdictions, greater exposure in litigation, and reputational harm. | Accept | Out: possible evaluation evidence | Correct; boilerplate risk factor; names no period. |
| 47 | TSLA FY2019, table/mdna | What was Tesla's interest expense for the year ended December 31, 2019, and how did it change compared to the prior year? | Interest expense was $685 million in 2019, an increase of $22 million or 3% compared to 2018. | Accept | Out: possible evaluation evidence, possible train evidence | Correct. |
| 48 | WMT FY2024, narrative/business | What specific emissions reduction targets has Walmart Inc. established for fiscal 2024 operations and its value chain under Project Gigaton? | Zero emissions in operations by 2040 without offsets and one billion metric tons reduction, avoidance, or sequestration in the value chain by 2030 | Accept | **In** | Correct targets, but "for fiscal 2024 operations" misplaces them: they are for 2040 (operations) and 2030 (value chain, Project Gigaton). |
| 49 | XOM FY2020, narrative/risk | What specific reduction targets for operated upstream greenhouse gas emission intensity does ExxonMobil aim to achieve by 2025 compared to 2016 levels? | A reduction of 15 to 20 percent in 2025, compared to 2016 levels | Accept | **In** | Correct. |
| 50 | PFE FY2025, narrative/statements | Which U.S. Attorney's Office issued the subpoena to Pfizer in October 2018 regarding its relationship with Tris Pharma and Quillivant XR? | U.S. Attorney's Office for the Southern District of New York (SDNY) | Accept | Out: evaluation evidence | Correct; a legal-proceedings detail (which office issued a subpoena). |

## 5. Aspects reviewed

| Aspect | Pass | Minor | Fail | n/a | Notes |
|---|---:|---:|---:|---:|---|
| Factual accuracy | 38 | 0 | 5 | 0 | 07, 16, 18 misread a table's scope or row; 17 gives a segment figure as the firm's; 13 names the wrong year |
| Financial relevance | 40 | 1 | 2 | 0 | 01 headquarters, 41 insider-trading boilerplate; 50 is a legal-proceedings detail |
| Company | 43 | 0 | 0 | 0 | Every question names the filer correctly |
| Period | 35 | 7 | 1 | 0 | 13 wrong. Minor: 03, 25, 41, 45 name no period; 04 and 48 tie a future effect to the report's year; 36 says "as of fiscal 2021" |
| Figures | 31 | 0 | 3 | 9 | 07, 16, 18 |
| Units | 26 | 8 | 0 | 9 | 07, 09, 16, 24, 26, 29, 36, 37 give no unit (millions); none gives a wrong one |

**The errors share a cause: the visible text starts mid-table or
mid-section.**
- 07: the "Consolidated Subsidiaries" heading is cut off. The 151,494 is
  then read as ExxonMobil's total, which is 185,664 with equity companies.
- 13: the "For fiscal 2021" lead-in is cut off. 2.2% is then given as fiscal
  2022's contribution, which was 1.3%.
- 17: the Consumer & Community Banking heading is not visible, so the
  segment's $12.3 billion provision is given as the firm's.
- 16: the subtotal of cash, cash equivalents and marketable securities is
  read as marketable securities alone.
- 18: a Non-GAAP Adjusted figure is read as reported EPS.
- This repeats pilot_v3's main weakness (§9.2 there).

## 6. Leakage and isolation

**Positives.**
- All 16 are validation passages, none blocked, and each overlap group and
  split unit lies wholly in validation.
- None is a positive or an equivalent of any training pair: the 70 built
  train pairs and the 66 candidates use 255 passages.
- One passage per split unit, so no two questions share near-identical
  passages.

**Evidence.**
- For every included question, the locator finds no train or blocked
  passage holding the evidence, exactly or uncertainly, in full text.
- This is re-checked whenever the set is loaded.
- Within validation, 4 questions have exact equivalents and 3 uncertain
  ones. Both are recorded with the pair.

**Questions.**
- **Against the 10 evaluation questions:** highest Jaccard 0.333 (0.207
  among included questions).
  - Read side by side, none paraphrases one.
  - The closest topic is 32 (Paxlovid's 2023 decline) against `q_0005`
    (COVID product revenue, 2023 against 2022). It was removed anyway: an
    evaluation-pool passage holds its evidence.
- **Against the 115 training questions:** every question of the train
  generation records and the 66 candidates, highest Jaccard 0.619.
  - The closest included pair is 29, Apple's FY2021 Level 2 corporate debt
    fair value, against a training question on the same table for FY2023.
  - Different year and figure, but the same template.

**The trainer cannot use the set** (each tested, and checked on the real
data):
- it lives under `validation_sets/`, while the trainer opens only
  `candidates/<name>/`, whose manifests must be of kind `training_candidates`
  (a copy placed there is refused);
- `verify_pairs` refuses all 16 as train pairs;
- `NegativeGuard` allows none of their passages as a negative for any
  training candidate;
- `plan_batches` refuses a batch that mixes the splits.

**No evaluation data shaped the questions.**
- Plan and generation ran with every evaluation file out of reach (tested
  before the run).
- Evaluation questions were read only at build time, to reject overlaps.
- No development metric was read.

**Records are preserved.**
- The set binds the records file by its first 50 lines (SHA-256), so a
  later validation run may append to it.
- Any edit to those lines, to the plan, the run, the decisions or the set's
  own files makes `load_validation_set` refuse the set (tested).

## 7. The validation set

`data/processed/train/bge_base_ft_v2/validation_sets/validation_pairs_v1/`,
built by:

```bash
python3 scripts/02_generate_queries.py build-validation --config configs/train/data_prep_v2.yaml \
    --validation configs/train/validation_v1.yaml      # offline; written once
```

| File | SHA-256 | Content |
|---|---|---|
| `validation_pairs.jsonl` | `9f4c1e3b…8dce` | 16 contract pairs of the validation split |
| `validation_log.jsonl` | `f15017a4…8553` | All 50 plan passages, with reasons, decisions and every outside evidence holder |
| `manifest.json` | `3b071ef5…fa3e` | Inputs and their SHA-256, checks, counts |

| Company | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | | | 1 | 1 | | | 1 | 3 |
| AMZN | 1 | | | 1 | | | 1 | 3 |
| JPM | 1 | | | | 1 | | | 2 |
| PFE | 1 | | | | | | | 1 |
| TSLA | | 1 | | | | | 1 | 2 |
| WMT | | | | | 1 | 1 | | 2 |
| XOM | | 1 | | | 1 | | 1 | 3 |
| **All** | 3 | 2 | 1 | 2 | 3 | 1 | 4 | 16 |

- **Kinds:** 10 tables and 6 narrative.
- **Types:** `financial_table` 8, `accounting_policy` 4, `risk` 2, `tax` 1,
  `other` 1.
- **Largest company share:** 19%.
- **Clean:** 12 of the 16 pass every aspect.
  - 29 and 36 omit a unit, and 36 phrases its period loosely.
  - 25 is boilerplate naming no period.
  - 48 ties targets for 2030 and 2040 to "fiscal 2024".

Load it for an evaluator with `finretriever.training.validation_set.load_validation_set(prepared,
passages, config, "validation_pairs_v1", training=<the run's training pairs>)`. It reads no
evaluation file and fails closed.

## 8. Is it suitable for choosing checkpoints?

**No.** It can test an evaluator and catch a broken checkpoint, but it
cannot tell checkpoints apart.

1. **Too few questions.**
   - With 16, one question is 6.25 points of Recall@10, and a 95% interval
     at 50% is about ±25 points.
   - A paired sign test reaches p < 0.05 only when at least 6 questions move
     the same way, all of them.
   - Detecting a 5-point paired MRR@10 difference needs roughly 200–300
     questions. This assumes a per-question standard deviation of the
     difference of 0.25–0.30, which is not measured.
2. **Uneven coverage.**
   - Pfizer has 1 question; 2021 and 2024 have 1 each.
   - Half the questions are table lookups whose positive is chosen by the
     year among near-identical annual tables, so the set measures year
     discrimination more than retrieval in general.
3. **No room to grow.**
   - The validation split has 216 split units among its 617 eligible
     sources, and the pilot used 50.
   - Of the remaining 166 units, 69 are JPMorgan's, 4 Amazon's and 1
     Apple's.
   - Estimated from the share of each passage's sentences that no passage
     outside validation holds (41% predicted for this plan against 44%
     observed), the rest gives about 56 more evidence-clean questions at one
     per split unit, or about 85 at one per overlap group.
   - That is about 40–60 after generator and review losses (86% and 84%
     here): about 55–75 in all, mostly JPMorgan.
4. **Synthetic and AI-reviewed only.** One model wrote the questions and
   another reviewed them. No human has checked them, and none is a gold
   label.

**What it can do:**
- run a validation evaluator end to end;
- show that a checkpoint has not collapsed;
- give a base-model reference reported with its uncertainty.

The trainer's fixed-step checkpoints should stay as they are until a larger
set exists. Do not use the pilot_v3 development metrics instead.

## 9. Remaining limitations

- **Size and balance** (§8).
- **Lexical leakage checks.** The evaluation and training question checks
  are lexical. Paraphrases are ruled out only by reading the 10 evaluation
  questions; the 115 training questions were compared by Jaccard.
- **Evidence equivalence is approximate.**
  - A passage that restates the evidence in other words, or rounds its
    figure, is not found.
  - So a train passage could still answer a validation question unnoticed.
  - The any-match rule makes this less likely, at the cost of about 4
    coincidental exclusions here.
- **Template overlap with training.** Annual tables recur, so a validation
  question can share its template with a training question about another
  year (29). The fact differs; the wording does not.
- **Generator weaknesses repeat pilot_v3's.**
  - Mid-table and mid-section chunks cause misreadings (§5).
  - "Write 1 different questions" stays in the prompt, whose SHA-256 is
    pinned.
  - Years in an answer can trigger a false `answer_not_supported` (46).
- **Equivalents.** An evaluator must decide how to score them. Count the
  positive and its exact equivalents as relevant; leave uncertain
  equivalents unscored rather than counting them as misses.
- **The records file grows.** Later validation runs append to it. The set
  stays bound to the first 50 lines, but a new set needs a new version and
  its own plan.

## 10. Recommendation

1. **Now:** use `validation_pairs_v1` only to build and test the validation
   evaluator. Keep fixed-step checkpoints, and report any result from it with
   its interval.
2. **For a set that can choose checkpoints, decide between:**
   - **(a) A new preparation with a validation split sized for about 250
     questions,** stratified by company.
     - It re-splits the data, so the 66 training candidates would be
       re-checked through `carry-over`, as pilot_v2 was.
     - It is the only option that reaches checkpoint-selection size.
   - **(b) More validation plans from this split.**
     - One passage per overlap group instead of per split unit, preferring
       passages whose sentences no outside passage holds.
     - About 55–75 questions in all, for roughly 100–250 more requests, each
       batch with your approval.
     - Better than now, still short, and mostly JPMorgan.
   - **Not recommended: relaxing the evidence rule to exact matches
     anywhere.**
     - It would have kept 23 questions here instead of 16.
     - But 3 of the 7 extra ones (03, 26, 39) state a fact a train passage
       holds.
3. **Before any more generation,** a prompt `sec-qgen-v3` that avoids
   figures whose headings are not visible and states periods by their own
   year. Also a validator fix for years in answers (46). Both were already
   recommended after pilot_v3.
