# Validation plan `val_pilot_v2f`, batch 4: audit of records 148–195

Status: 2026-11-04.
- **What this is:** an AI review (Claude) of the 48 generation records for plan positions 148–195 of
  `val_pilot_v2f` in `bge_base_ft_v3`. It is not human annotation, and no decision here is a gold label.
- **What was not done:**
  - no generation record, plan, training pair, candidate, calibration, validation set or decision file
    was changed;
  - no validation set or validation pair was built, and no decision file was written: the decisions
    below exist only in this report until they are recorded;
  - no API request was sent and nothing was trained.
- **Checksums:** 274 files under `data/`, `configs/` and `results/` were checksummed before and after the
  audit, and none changed. Since positions 1–147 were recorded, only `validation.records.jsonl` has
  changed.
- **The checks:** every validation-set check (validator, evidence rule, question overlap, duplicates, unit
  isolation) was re-run in memory, as `build-validation` would run it, with nothing written.
- **Earlier documents:** [validation_plan_v2f.md](validation_plan_v2f.md) (the plan), the audits of
  [batch 1](validation_plan_v2f_batch1_audit.md) (positions 1–49), [batch 2](validation_plan_v2f_batch2_audit.md)
  (50–97), [batch 3](validation_plan_v2f_batch3_audit.md) (98–143) and
  [positions 144–147](validation_plan_v2f_positions_144_147_audit.md), and the decisions recorded for
  positions 1–147 in `data/validation_review/`.

## 1. Summary

**The records cover positions 148–198, not 148–195:**
- 51 records were appended, one for each plan position from 148 to 198, all `done` on their first
  attempt. Positions 196 (Pfizer FY2022), 197 (Tesla FY2025) and 198 (Walmart FY2021) are complete, and
  the validator accepts their questions again.
- The latest execution is positions 156–198: 43 records and 59,013 tokens, the figures reported. Positions
  148–155 came from three short executions earlier the same day.
- This audit covers 148–195, as asked. Positions 196–198 are not reviewed here.

**Batch 4 fell short of its estimate:**
- **The funnel:**
  - 48 passages;
  - the generator accepted 34 (71%);
  - the evidence rule left 8;
  - the review leaves **7 usable questions**.
- **Against the plan:** 9.5 expected at this plan's observed rates (the forecast before the batch was 9.7,
  or 8.0 if passages below 0.2 gave nothing).
- **Positions 1–195:** 57 usable, against 62.9 expected at the calibration's rates.
- **Running total:** 88 usable validation questions (16 carried, 15 from `val_pilot_v2e`, 57 from this
  plan).

**Where the shortfall came from:**
- **The generator:** it accepted 34 rather than about 37, and 10 of its 14 rejections fell on passages
  scoring 0.29 or more.
  - Three of the rejections are a new failure: Qwen returned an empty question or answer, with garbled
    type names (`deBT_liquidity`).
  - All three came in the latest execution; none of the 147 earlier positions had one.
- **The evidence rule:** 8 passes, against 9.8 predicted for the 34 accepted questions.
- **The review:** it removed 1 of the 8 (192), about the expected rate.

**Quality:**
- **One factual error in 34 (3%),** the lowest rate of any batch.
  - 178 gives one loan class's total ($326,409 million, residential real estate) as JPMorgan's total
    retained loans ($1,280,870 million). The class heading lies above the window.
  - The evidence rule had already excluded it.
- **The review excluded one evidence-passing question, 192:**
  - It asks when Tesla "plans to" adopt ASU 2021-08, which its FY2023 passage reports as adopted.
  - Tesla's FY2021 filing (in the annotation and dev pools) and FY2022 filing (blocked) give the effective
    date and method in other words, which the rule cannot match.
  - 192 was the new-standard notice flagged for this position.
- **Four questions drop a table's scope (minor):** 160, 166, 182 and 187 ask about the right row but omit
  the heading that qualifies it.
- **Elsewhere:** company attribution is right in all 34, and no answer gives a wrong unit.

**Low-scoring passages yield little, but not nothing:**
- **This batch:** 19 of the 48 passages score below 0.2.
  - They gave 2 evidence passes against 1.9 predicted, and 1 usable question (193).
  - That is 19 requests for one question, against 4.8 per question above 0.2.
- **All 195 positions:** passages below 0.2 gave 2 evidence passes against 4.1 predicted.

**FY2024:**
- 7 passages gave 2 usable questions (149, 166), so FY2024 rises from 6 to 8. It is still the thinnest
  full year.
- Only 10 FY2024 passages remain in the plan, worth about 1 question, so this plan cannot balance FY2024.

**Recommendation** (§12):
- **Positions 196–197 need no recovery:** both are complete records, and so is 198. Audit them with
  batch 5.
- **Batch 5 (positions 199–248) is worth running unchanged, as the last batch of this plan.**
  - It should give about 6–9 usable questions for 50 requests and about 68,000 tokens, roughly batch 4's
    cost per question, and bring the set to about 94–97.
  - Beyond position 248, each question costs 7–50 requests, and the yield is mostly 2019 and JPMorgan.

## 2. Provenance

| Check | Result |
|---|---|
| Plan | `validation.val_pilot_v2f.json` (`a7903bc7…`), unchanged. The 48 records audited are its positions 148–195, in order |
| Records beyond the batch | Positions 196–198 also have records: `done`, first attempt, no error, each question accepted by the generator, and the validator re-run accepts them again. Plan positions 1–198 each have exactly one record. Position 199 (ExxonMobil FY2022) is the next to generate |
| Run | The same bound run as batches 1–3: `qwen/qwen3.8-27b`, prompt `sec-qgen-v2`, parameters `42d5fe54…`, validator `sec-qcheck-v3`, one question per passage |
| Executions | Four: positions 148–149 at 01:15 UTC, 150–152 at 01:39, 153–155 at 02:10, then 156–198 from 12:23 to 12:44 UTC (43 records and 59,013 tokens, the reported latest execution) |
| Records | 48 done, 0 failed, each on its first response attempt, no errors. Records do not log HTTP retries, so the reported zero 429s cannot be checked from them. The 43 records of the latest execution match the 43 requests reported |
| Tokens | 65,189 for positions 148–195: 58,693 prompt and 6,496 completion, 1,358 per passage. Completion 88–329, median 122, against the 768 limit |
| Files | `validation.records.jsonl` is now `3cdab924…`, 248 lines. Its first 197 lines are byte-identical to the file positions 1–147 were audited and recorded on (`d7eb9d4b…`), so the run only appended. `validation.run.json` `34e6b74d…` is unchanged |
| Passages | None repeats a passage of `val_pilot_v2e` or of positions 1–147 |
| Validator re-run | Reproduces all 48 outcomes (34 accepted, 14 rejected, the same reasons) |
| Predictor re-run | All 48 plan scores recompute exactly from `configs/train/evidence_pass_v1.json` |

## 3. The funnel

| Stage | Kept | Removed | Why |
|---|---:|---:|---|
| Passages completed | 48 | | Positions 148–195. 196–198 are also complete, and not audited here |
| Generator (`sec-qcheck-v3`) | 34 | 14 | `copied_from_passage` 4, `missing_company` 4, `malformed` 3, `question_format` 1, `answer_not_supported` 1, `evidence_length` 1 |
| Evidence rule (validation: any match outside validation) | 8 | 26 | Possible training evidence 13, evaluation evidence 8, possible evaluation evidence 8, training evidence (exact, visible) 6; some have several |
| Question overlap (evaluation, training, duplicates) | 8 | 0 | Highest Jaccard: 0.310 against an evaluation question, 0.684 against any training question (threshold 0.8) |
| AI review | **7** | 1 | Reject 192 |

**Generator rejections** (14; verdict on each in §4):
- **13 are correct by the rule; one (164) is strict.** Its answer is right ($31.5 billion in 2022, $1.5
  billion less than 2021), but it names 2021, which its quote does not. A sound question was lost.
- **Empty fields (163, 168, 175):** new to this plan.
  - Two have no question and one has no answer.
  - Two give garbled type names (`deBT_liquidity`, `de_liquidity`).
  - All three came in the latest execution.
- **Copied wording (159, 172, 180, 195):** four, as in batch 3.
- **No company named (177, 181, 184, 190):**
  - two tables with no subject (177, 190);
  - a segment (Sam's Club, 181);
  - a benchmark administrator (ICE, 184).
  - 190's answer also drops the sign of a loss.
- **Length (154, 170):** 154's question has 41 words as the validator counts them, and 170 quotes a whole
  table row.

**Question types accepted:** `financial_table` 19, `risk` 4, `accounting_policy` 4, `other` 3,
`profitability` 2, `tax` 1, `revenue` 1.

## 4. Per-question decisions

- **Decisions:** ACCEPT, REVISE and REJECT apply to the 34 questions the generator accepted. For the 14 it
  rejected, the note gives a verdict on that rejection.
- **"Evidence rule":** "In" means no train or blocked passage holds the evidence, exactly or possibly, in
  its full text. "Out" lists the reasons. Only an "In" question with ACCEPT is usable.
- **REVISE** means the question would be acceptable reworded. Records are never edited, so a revised
  question is excluded like a rejected one.

| # | Source | Question (verbatim) | Answer (verbatim) | Decision | Evidence rule | Note |
|---|---|---|---|---|---|---|
| 148 | JPM FY2022, narrative/risk | How does JPMorgan Chase define wrong-way risk in its credit and investment risk management for fiscal 2022? | The risk that exposure to a counterparty is positively correlated with the impact of a default by the same counterparty, which could cause exposure to increase as the counterparty's capacity to meet obligations decreases | ACCEPT | Out: possible training evidence | Correct; JPMorgan's definition, repeated in its other years' filings (training copies). |
| 149 | PFE FY2024, narrative/risk | What specific EU regulatory measure took effect in January 2025 that Pfizer is monitoring for its application to new oncology and biologic products? | The European Regulation on Health Technology Assessment (EU HTA-R) | ACCEPT | **In** | Correct. Pfizer's FY2025 filing restates the fact ('took effect in January 2025') in a train passage the evidence rule cannot match; no training pair uses it (§7). |
| 150 | TSLA FY2024, table/statements | What was Tesla's total income before income taxes for the fiscal year ended December 31, 2024? | $8,990 million | ACCEPT | Out: evaluation evidence, possible training evidence | Correct: domestic, noncontrolling-interest and foreign rows sum to it. Batch 3's 131 asked the same row for 2023. |
| 151 | WMT FY2024, table/statements | What was the balance of Total Walmart shareholders' equity for Walmart Inc. as of January 31, 2024? | $83,861 million | ACCEPT | Out: training evidence | Correct. |
| 152 | XOM FY2023, narrative/other | What were the realized proceeds and net after-tax earnings from Exxon Mobil Corp's divestment activities in the year ended December 31, 2023? | Proceeds of approximately $4.1 billion and net after-tax earnings of approximately $0.6 billion | ACCEPT | Out: evaluation evidence, training evidence | Correct; both 2023 figures (the note also gives 2022's). |
| 153 | AMZN FY2019, table/statements | What was the consolidated goodwill balance for Amazon at the end of fiscal 2019? | $14,754 million | ACCEPT | Out: evaluation evidence, possible training evidence | Correct; the consolidated column of the roll-forward's closing row. |
| 154 | JPM FY2019, narrative/risk | According to JPMorgan Chase's fiscal 2019 10-K, what specific operational consequences does the company anticipate for its European operating framework if the U.K. and E.U. fail to reach an agreement before the end of the transition period? | The framework will be more complex, less efficient and more costly, potentially resulting in a more fragmented operating model across U.K., E.U. and other entities. | Generator: `question_format` | – | Correct by the rule: 41 words as the validator counts them (it splits U.K., E.U., 10-K and possessives), against a limit of 40. The answer adds a clause from outside the quote. |
| 155 | PFE FY2022, table/statements | What was Pfizer's total equity balance as of January 1, 2020, in millions? | $63,447 million | ACCEPT | Out: training evidence | Correct; the opening balance of the three-year equity statement, named by its date. |
| 156 | TSLA FY2022, table/statements | What was the total stock-based compensation expense for Tesla in fiscal 2022? | $1,560 million | ACCEPT | Out: possible training evidence | Correct. |
| 157 | WMT FY2026, table/statements | What were Walmart's total undiscounted operating lease obligations as of January 31, 2026, in millions? | $22,624 million | ACCEPT | **In** | Correct; the operating-lease column (the six maturities sum to 22,624). |
| 158 | XOM FY2024, table/other | What was the amount of net cash provided by operating activities for Exxon Mobil Corp for fiscal 2024, in millions of dollars? | 55,022 million dollars | ACCEPT | Out: evaluation evidence | Correct; the quote ends at token 505 of 512. |
| 159 | AMZN FY2025, table/statements | What was the total expected future amortization expense for Amazon's acquired finite-lived intangible assets as of December 31, 2025? | $8,034 million | Generator: `copied_from_passage` | – | Correct by the rule: copies the sentence that introduces the table, and the quote elides the table with '...'. The answer is right (the five years and thereafter sum to 8,034). |
| 160 | JPM FY2021, narrative/statements | What were JPMorgan Chase & Co's total securitized assets for residential mortgage commercial and other loans in fiscal 2021? | 93,698 | ACCEPT | Out: possible training evidence | Correct row; the parsed label nests 'Commercial and other' under residential mortgage and the question repeats it, but JPMorgan's securitization table treats it as a separate category. No unit (millions). |
| 161 | PFE FY2023, table/statements | What was the pre-tax expense recognized by Pfizer in 2021 to resolve an MDL relating to EpiPen? | $345 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; 2021's figure, named by year, in the FY2023 filing's discontinued-operations footnote (Meridian). |
| 162 | TSLA FY2020, narrative/mdna | What was the maximum potential payout Tesla could face from vehicle resale value guarantees or buyback options over the five years following December 31, 2020? | $1.84 billion over the next five years | ACCEPT | Out: possible evaluation evidence | Correct. |
| 163 | WMT FY2025, narrative/statements | What was the total amount of current liabilities reported by Walmart Inc. as of January 31, 2025? |  | Generator: `malformed` | – | Correct: no answer (null). |
| 164 | XOM FY2022, table/other | What was ExxonMobil's total other taxes and duties in fiscal 2022, and how did it compare to the amount in fiscal 2021? | Total other taxes and duties were $31.5 billion in 2022, a decrease of $1.5 billion from 2021. | Generator: `answer_not_supported` | – | Strict: the answer is right (32,955 - 31,455 = 1,500), but names 2021, which the quote does not. A sound question lost. |
| 165 | AMZN FY2019, narrative/mdna | What was Amazon's total operating expenses for the fiscal year ended December 31, 2019, and how does that compare to the prior year? | $265,981 million in fiscal 2019, compared to $220,466 million in fiscal 2018. | ACCEPT | Out: evaluation evidence, possible training evidence | Correct; both years. Same template as 183. |
| 166 | JPM FY2024, narrative/statements | What were JPMorgan Chase & Co's total gross derivative payables for interest rate contracts in fiscal 2024? | $273,133 | ACCEPT | **In** | Correct row; the total of the U.S. GAAP nettable interest-rate payables, a qualifier the visible heading gives and the question omits. No unit (millions). |
| 167 | PFE FY2025, table/statements | What were the confirmed obligations outstanding at the end of fiscal 2025 for suppliers participating in Pfizer's voluntary supply chain finance agreements? | $574 million | ACCEPT | **In** | Correct; 2025's ending balance (2024's is $688 million). |
| 168 | TSLA FY2023, narrative/statements |  | $892 million as of December 31, 2023; decreased by $6.46 billion during the year ended December 31, 2023 | Generator: `malformed` | – | Correct: no question (null). |
| 169 | WMT FY2021, table/statements | What was Walmart Inc.'s total liabilities and equity as of January 31, 2021, in millions of dollars? | $252,496 million | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; equal to total assets (batch 3's 111 template). |
| 170 | XOM FY2025, table/statements | What was the net carrying value of ExxonMobil's derivative assets as of December 31, 2025? | $854 million | Generator: `evidence_length` | – | Correct by the rule: the quote is the whole row (more than 300 characters). The answer is right. |
| 171 | AMZN FY2019, narrative/statements | What was the value of goodwill allocated from business acquisitions for Amazon.com, Inc. at December 31, 2019? | $189 | ACCEPT | Out: possible training evidence | Correct; goodwill from 2019's acquisitions, asked 'at December 31, 2019' as the parsed column reads. No unit (millions). 153's passage gives the same $189 million. |
| 172 | JPM FY2025, narrative/risk | What are the potential consequences for JPMorganChase in fiscal 2025 if it fails to comply with applicable law or meet supervisory expectations? | Increased regulatory scrutiny, enforcement actions by governmental authorities, the imposition of penalties, increased exposure to litigation, or reputational harm. | Generator: `copied_from_passage` | – | Correct by the rule: copies the sentence. A boilerplate regulatory risk. |
| 173 | PFE FY2021, narrative/statements | What was the carrying value of Pfizer's investment in the Consumer Healthcare JV as of December 31, 2021? | $16.3 billion | ACCEPT | Out: possible training evidence | Correct. |
| 174 | TSLA FY2025, narrative/statements | What offset amount per share applies to the 2025 CEO Performance Award shares upon vesting for Tesla, Inc.? | $334.09 per share | ACCEPT | **In** | Correct; a term of the 2025 CEO award (batch 3's 138 asked its expense). |
| 175 | WMT FY2022, table/statements |  | $26,060 million | Generator: `malformed` | – | Correct: no question (null). |
| 176 | XOM FY2019, narrative/risk | What was ExxonMobil's total distillation capacity across its refineries in fiscal 2019? | 4.7 million barrels per day | ACCEPT | Out: possible evaluation evidence | Correct; an operating capacity. |
| 177 | AMZN FY2025, table/statements | What was the number of restricted stock units outstanding as of December 31, 2025, and what was their weighted average grant-date fair value? | 222.5 million units with a weighted average grant-date fair value of $178 | Generator: `missing_company` | – | Correct: names no company. |
| 178 | JPM FY2023, narrative/statements | What was the total value of retained loans for JPMorgan Chase as of December 31, 2023, and how did it compare to the total for December 31, 2022? | $326,409 million for December 31, 2023, compared to $237,561 million for December 31, 2022 | **REVISE** | Out: possible evaluation evidence, training evidence | $326,409 million is the total of one consumer class, residential real estate (the LTV, FICO and state rows), whose heading lies above the window. JPMorgan's total retained loans were $1,280,870 million. |
| 179 | PFE FY2020, table/statements | What was Pfizer Inc.'s total assets for the year ended December 31, 2020? | $154,229 million | ACCEPT | Out: training evidence | Correct; a balance as of December 31, asked 'for the year ended'. |
| 180 | TSLA FY2019, narrative/risk | Which specific competitors does Tesla identify as facing competition with its Model 3 and Model Y in the entry-level premium sedan and compact SUV market? | BMW, Ford, Lexus, Mercedes and Volkswagen Group | Generator: `copied_from_passage` | – | Correct by the rule: copies the sentence. |
| 181 | WMT FY2020, table/statements | What was Sam's Club's eCommerce-related net sales for the fiscal year ended January 31, 2020? | $3.6 billion | Generator: `missing_company` | – | Correct: names Sam's Club but not Walmart. |
| 182 | XOM FY2020, table/other | What was the total amount of noncash right of use assets recorded in exchange for lease liabilities for Exxon Mobil in fiscal 2020? | $735 million | ACCEPT | **In** | Correct row; 735 is the operating-lease Total column, which the question calls the total (finance leases add 108). The FY2021 filing labels it 'Operating Leases 2020'. |
| 183 | AMZN FY2025, narrative/statements | What was Amazon's total operating expenses for the fiscal year ended December 31, 2025? | 636,949 | ACCEPT | Out: evaluation evidence | Correct; no unit (millions). Same template as 165. |
| 184 | JPM FY2020, narrative/other | When did ICE Benchmark Administration propose to cease publication of the principal tenors of U.S. dollar LIBOR? | Immediately following a final publication on June 30, 2023 | Generator: `missing_company` | – | Correct: names ICE Benchmark Administration, not JPMorgan; not JPMorgan's fact. |
| 185 | PFE FY2019, narrative/business | What specific federal and state healthcare programs are mentioned as requiring Pfizer to offer discounted pricing or rebates on pharmaceutical product purchases in fiscal 2019? | Medicaid Drug Rebate Program, the "federal ceiling price" drug pricing program, the 340B drug pricing program and the Medicare Part D Program | ACCEPT | Out: possible evaluation evidence | Correct; the standard list of U.S. pricing programmes, repeated in Pfizer's filings. |
| 186 | TSLA FY2020, narrative/business | In which state did Tesla, Inc. report having an insurance product available for customers as of fiscal 2020? | California | ACCEPT | Out: evaluation evidence | Correct for FY2020 (later filings report other states from 2021); a product's availability. |
| 187 | WMT FY2026, narrative/statements | What was the dollar amount and percentage impact of internal reorganization on Walmart's effective tax rate for fiscal 2026? | $1,814 (6.2%) | ACCEPT | Out: possible training evidence | Correct row, Luxembourg's (offset by its (6.1)% valuation-allowance change), which the question does not name; a second internal-reorganization row, (349) and (1.2)%, sits under nondeductible items. No unit (millions). |
| 188 | XOM FY2021, narrative/other | What was the amount of amortization of actuarial loss recorded in accumulated other comprehensive income for Exxon Mobil's U.S. pension benefits in fiscal 2021? | (733) | ACCEPT | Out: possible training evidence | Correct; keeps the table's parentheses; no unit (millions). |
| 189 | AMZN FY2019, table/statements | What was Amazon's total provision for income taxes for the fiscal year ended December 31, 2019? | $2,374 million | ACCEPT | Out: possible evaluation evidence, training evidence | Correct; the third of three years. |
| 190 | JPM FY2022, table/statements | What was the income statement impact of excluded components for interest rate fair value hedges in the year ended December 31, 2022? | $262 million | Generator: `missing_company` | – | Correct: names no company. The answer also drops the sign: the cell is (262), a loss. |
| 191 | PFE FY2024, narrative/business | What net revenue impact does Pfizer anticipate for 2025 resulting from the Medicare Part D redesign changes? | Approximately $1 billion unfavorable, year-over-year | ACCEPT | Out: evaluation evidence | Correct; a forecast for 2025, named. |
| 192 | TSLA FY2023, narrative/statements | When does Tesla, Inc. plan to adopt ASU No. 2021-08 regarding contract assets and liabilities, and how was it applied? | Prospectively on January 1, 2023 | **REJECT** | **In** | Asks when Tesla 'plans to' adopt a standard its passage reports adopted on January 1, 2023. Tesla's FY2021 filing (annotation and dev pools) and FY2022 filing (blocked) give the effective date and prospective method in other words, which the evidence rule cannot match. A standard-adoption notice, the pattern flagged for this position. |
| 193 | WMT FY2025, table/statements | What was the comprehensive income, net of income taxes, for Walmart Inc. in the fiscal year ended January 31, 2025? | $17,298 million | ACCEPT | **In** | Correct; consolidated, before noncontrolling interest. |
| 194 | XOM FY2024, narrative/business | What was the impact of Advantaged Volume Growth on Exxon Mobil's upstream earnings in fiscal 2024, and what primary factors drove this change? | Increased earnings by $3,760 million, driven by record production in the Permian and Guyana. | ACCEPT | Out: possible evaluation evidence, possible training evidence | Correct; Permian and Guyana (the second just past the quote). |
| 195 | JPM FY2021, narrative/statements | What were the contractual service fees, late fees, and other ancillary fees included in income for JPMorgan Chase & Co for the fiscal year 2021? | 1,298 | Generator: `copied_from_passage` | – | Correct by the rule: copies the row label. No unit. |

## 5. Aspects reviewed (34 accepted questions)

| Aspect | Pass | Minor | Fail | n/a | Notes |
|---|---:|---:|---:|---:|---|
| Factual correctness | 29 | 4 | 1 | 0 | Fail: 178 (one loan class's total given as the Firm's). Minor: 160, 166, 182 and 187 (a table qualifier omitted) |
| Financial relevance | 31 | 3 | 0 | 0 | Minor: 176 (refining capacity), 186 (a product's availability), 192 (a standard-adoption notice) |
| Company and segment attribution | 34 | 0 | 0 | 0 | Every question names the filer |
| Fiscal year | 31 | 2 | 1 | 0 | Fail: 192 (asks about a planned adoption already made). Minor: 171 ("at December 31, 2019" for a year's acquisitions), 179 ("for the year ended" for a balance) |
| Figures | 27 | 1 | 1 | 5 | Fail: 178. Minor: 188 (keeps the table's parentheses) |
| Units | 23 | 6 | 0 | 5 | Minor: 160, 166, 171, 183, 187 and 188 give no unit (millions); none gives a wrong unit |

**Table scope** (the main finding): Qwen reads a row's label and drops the heading that qualifies it. When
the heading is visible, the question is only imprecise; when it lies above the window, the figure is
wrong for the question.
- **178, wrong:** the passage starts inside a loan table whose heading lies above the window.
  - Its rows (loan-to-value and FICO bands, a breakdown by state) are those of JPMorgan's residential real
    estate loans, so "Total retained loans" is that class's total.
  - The question asks for JPMorgan Chase's total retained loans, which the same filing gives as
    $1,280,870 million.
- **Imprecise, accepted:** each passage still answers the question as asked, as with batch 1's 10 and
  batch 2's 57.
  - **160:** the parsed label nests "Commercial and other" under residential mortgage, and the question
    repeats it. The same filing's securitization table gives the two categories separate columns.
  - **166:** "U.S. GAAP nettable derivative payables" opens the passage. The question asks for total
    gross interest-rate payables without that qualifier.
  - **182:** the table's Total column is for operating leases; finance leases add 108. ExxonMobil's FY2021
    filing labels the same 735 "Operating Leases 2020".
  - **187:** the row is Luxembourg's internal reorganization, offset by its valuation-allowance change. A
    second internal-reorganization row, (349) and (1.2)%, sits under nondeductible items.

**Reporting periods:**
- **192:** "When does Tesla, Inc. plan to adopt ASU No. 2021-08" frames as future an adoption its FY2023
  passage reports as done ("We adopted this ASU prospectively on January 1, 2023").
  - The planned framing is that of Tesla's FY2021 and FY2022 filings, which list the standard as not yet
    adopted.
  - Both are blocked, the FY2021 passage as part of the annotation and dev pools. Each gives the effective
    date (annual periods after December 15, 2022) and the prospective method.
  - Rewording cannot help, because those passages state the same facts in any wording. So 192 is
    rejected rather than revised.
- **Prior periods named by date were read correctly:**
  - 155: Pfizer's January 1, 2020 equity, in its FY2022 filing;
  - 161: 2021's EpiPen settlement expense, in the FY2023 filing;
  - 152, 165 and 189: the right year among several columns or sentences.
- **171:** asks for goodwill "at December 31, 2019"; the parsed column reads that way, but the figure is
  goodwill from the year's acquisitions. Minor.

**Numerical accuracy:**
- **Every figure was checked** against its cell or sentence. Apart from 178, each is the right number for
  the right row, column and date.
- **Totals:** 150, 157 and 166 are totals, and their components sum to them.

**Evidence support:**
- **Quotes:** all 34 are verbatim, and the matcher accepts each.
- **Answers:** every answer is supported by its quote. The exception is the second half of 194's
  ("record production in Guyana"), which follows the quote inside the window.

**Visibility within BGE's 512-token limit:**
- **Where the text comes from:** 43 of 48 passages are truncated. 152, 160, 162, 164 and 182 are complete
  (195–376 tokens). Every quote lies inside the window.
- **Where the quotes end:** the latest ends at token 508 (192). Others after token 420: 158 (505), 193
  (465), 149 (442), 161 (430) and 185 (429).
- **Text outside the window:** 178's class heading lies above it, which is how its question misreads the
  total. 149's restatement in the FY2025 filing lies past the window of a validation passage, so it is
  not an equivalent (§7).
- **Context rules:** all 48 passages passed them (`headless_table_start`, `fragment_opening`).

## 6. Low-scoring passages and FY2024

**Passages scoring below 0.2:** 19 of the 48 (151, 155, 156, 161, 163, 165, 168, 169, 171, 175, 177, 183,
186, 187, 189, 191, 192, 193, 194).

| Plan score | Passages | Generator accepted | Evidence pass | Usable | Expected usable (calibration) |
|---|---:|---:|---:|---:|---:|
| Below 0.1 | 4 | 4 | 0 | 0 | 0.2 |
| 0.1–0.2 | 15 | 11 | 2 | 1 | 1.7 |
| 0.2–0.3 | 7 | 6 | 1 | 1 | 1.3 |
| 0.3–0.5 | 16 | 9 | 4 | 4 | 4.7 |
| 0.5 and above | 6 | 4 | 1 | 1 | 2.6 |

- **The low end is not zero in this batch:**
  - 15 of the 19 passages gave accepted questions;
  - those were predicted to give 1.9 evidence passes, and gave 2 (192, 193);
  - one is usable (193, Walmart's comprehensive income); 192 was rejected by the review.
- **Over all 195 positions, the low end is optimistic:** 35 accepted questions below 0.2 were predicted to
  give 4.1 passes, and gave 2, about half.
- **Its cost:**
  - below 0.2: 19 requests for one usable question;
  - at 0.2 and above: 29 passages gave 6 usable, 4.8 requests each.
- **The generator's losses fell at the top:**
  - 10 of its 14 rejections are at scores of 0.29 and above, and they took 34% of the batch's predicted
    yield from 29% of its passages.
  - Four are figure-free narrative passages (154, 172 and 180, risk factors; 184, an executive overview).
    There Qwen copied wording, ran long or named someone else.

**FY2024** (7 passages):

| # | Company | Score | Generator | Evidence rule (holders outside validation) | Review |
|---|---|---:|---|---|---|
| 149 | PFE | 0.23 | Accepted | **In** | ACCEPT, usable |
| 150 | TSLA | 0.28 | Accepted | Out: the FY2024 filing's own train chunk; FY2025 filing (train, blocked) | ACCEPT |
| 151 | WMT | 0.12 | Accepted | Out: the FY2024 filing's own train chunk; FY2025 and FY2026 filings (train) | ACCEPT |
| 158 | XOM | 0.25 | Accepted | Out: the FY2024 filing's own blocked chunk; FY2025 filing (blocked) | ACCEPT |
| 166 | JPM | 0.70 | Accepted | **In** | ACCEPT, usable |
| 191 | PFE | 0.14 | Accepted | Out: the FY2024 filing's own blocked chunk | ACCEPT |
| 194 | XOM | 0.20 | Accepted | Out: the FY2024 filing's own train chunk; earlier years' figures | ACCEPT |

- **The evidence rule alone limits FY2024:** the generator and the review lost none of the seven.
- **Why so few pass:** FY2024's values recur as the prior-year column of FY2025 filings and in neighbouring
  chunks of the FY2024 filing itself, and many of those are train or blocked.
- **Where FY2024 stands:**
  - FY2024 now has 8 usable questions, against 10–18 for each other full year from 2019 to 2025. FY2026
    exists only for Walmart, and has 1.
  - The plan's 163 unrun positions hold 10 FY2024 passages, worth about 1.0 question (3 of them in batch 5,
    about 0.6). FY2024 will end near 9.
- **Balancing it** needs new supply (a later plan or a larger validation split), not more of this plan.

## 7. Duplicates, templates, distribution and leakage

**Duplicates:**
- **None exact or near-exact.** No two questions share a normalised key, within the batch or with any
  earlier pool.
- **Highest Jaccard within the batch:** 0.684, between 150 and 189 (Tesla's pre-tax income and Amazon's tax
  provision): different companies and rows.
- **Highest against earlier batches:** 0.722, 150 and 189 each against an earlier question on Tesla's 2022
  tax provision.

**Repeated statement templates** (correlated, not duplicates):

| Questions | Template | Evidence rule | Usable |
|---|---|---|---|
| 165, 183 | Amazon's total operating expenses (FY2019, FY2025) | Both out | – |
| 150, 187, 189 (and 164, 168) | Income-tax notes | All out (164 and 168 generator-rejected) | – |
| 151, 169, 179 (and 163, 175) | Balance-sheet totals (Walmart's equity and total liabilities and equity; Pfizer's total assets) | All out (163 and 175 generator-rejected) | – |
| 153, 171 | Amazon's FY2019 goodwill (the same $189 million, in two tables) | Both out | – |
| 157, 182 | Lease tables (Walmart, ExxonMobil) | Both in | 157, 182 |
| 166 | JPMorgan's derivative netting table, as batch 2's 59 and batch 3's 129 | In | 166 |
| 156, 174 (and 177) | Stock-based compensation (Tesla; Amazon) | 174 in | 174 |

- **In the usable set:** no batch 4 usable question reaches 0.6 against any other of the 88.

**Evidence reuse across filing years** (the 26 excluded questions):
- **Where the holders are:**
  - 17 have holders both in their own filing and in other years' filings;
  - 5 only in other years' filings;
  - 4 only in their own filing.
- **By match:** figure matches alone excluded 5 (160, 161, 169, 173, 176).
- **By holder:**
  - training only 10, blocked only 7, both 9;
  - among the blocked holders, an evaluation pool itself for 10 questions, a passage only linked to one
    for 6.

**Paraphrased evidence the rule cannot match** (found by the review):
- **The limit:** the rule matches quotes and figures, so it cannot find a fact restated in other words, a
  limit noted since `val_pilot_v1` (§9 of its report).
- **192:** rejected (§5).
- **149: usable under the rules, with a caveat.**
  - Pfizer's FY2025 filing restates its fact ("the European Regulation on Health Technology Assessment
    (EU HTA-R) took effect in January 2025") in a train passage, and past the window of a validation
    passage.
  - No training pair uses either passage, and no training question mentions HTA-R.
  - 149 fails no reviewed aspect, so a review decision cannot exclude it (§12).

**Linked units and filings:**
- **Isolation:** every positive's linked unit and overlap group lie wholly in validation; checked for all
  48 passages.
- **The cap:** counting `val_pilot_v1`, `val_pilot_v2e` and positions 1–147, every unit holds at most 3
  generated passages. Three reach 3 with this batch: `grp_06910` (150 and 156), `grp_08861` (164) and
  `grp_00711` (189).
- **Filings revisited:** all 35 filings in the batch already gave a passage to positions 1–147. Amazon's
  FY2019 filing gave four here (153, 165, 171, 189), all excluded by the evidence rule.
- **Across all 88 usable questions:** they come from 46 filings.
  - 19 filings give one question, 15 give two and 9 give three.
  - 3 give four: Amazon's FY2019 and FY2025 filings, and Walmart's FY2025 filing (193 here).

**Distribution:**

| | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch passages | 0 | 7 | 9 | 8 | 8 | 8 | 8 | 48 |
| Usable, this batch | 0 | 0 | 1 | 2 | 1 | 2 | 1 | 7 |
| Usable so far (16 carried, 15, 50, 7) | 11 | 15 | 14 | 10 | 11 | 14 | 13 | 88 |
| Unrun plan passages (199–361) | 0 | 0 | 60 | 28 | 23 | 13 | 39 | 163 |
| Projected after batch 5 | 11 | 15 | 16.8 | 11.6 | 13.2 | 15.1 | 14.2 | 96.9 |
| Projected, whole plan | 11 | 15 | 21.6 | 13.3 | 15.2 | 15.3 | 16.2 | 107.5 |

- **About the projections:** they include positions 196–198, and count passages below 0.2 at half their
  predicted passes, as observed (§8).
- **Amazon's supply ended in this batch:** position 189 was its last, and none of its 7 passages gave a
  usable question. Apple's ended at position 134.
- **Fiscal years:**
  - The batch has 8 passages from 2019, 6 from 2020, 5 from 2021, 6 from 2022, 5 from 2023, 7 from 2024,
    9 from 2025 and 2 from 2026.
  - The 7 usable questions: 2020 1, 2024 2, 2025 3, 2026 1.
  - All 88 usable so far: 2019 16, 2020 13, 2021 11, 2022 10, 2023 11, 2024 8, 2025 18, 2026 1.
- **Kinds:** 27 narrative passages and 21 tables. The 7 usable are 3 narrative and 4 tables.

**Evidence overlap with training, blocked passages and evaluation pools:**
- **The 8 "In" questions:** no train or blocked passage holds the evidence by the rule's matching, and no
  positive is used by any training pair. 149's restatement is the exception noted above.
- **Question overlap:**

| Pool | Highest Jaccard | Question |
|---|---:|---|
| Evaluation questions | 0.310 | 193 |
| Training questions (all 150 ever generated) | 0.684 | 183 (against Amazon's total operating expenses, FY2021) |
| The 16 carried | 0.478 | 189 |
| `val_pilot_v1` (not carried) | 0.600 | 179 |
| `val_pilot_v2e` batch 1 | 0.625 | 193 (against Walmart's comprehensive income, FY2026) |
| `val_pilot_v2f` positions 1–147 | 0.722 | 150 (against Tesla's 2022 tax provision) |

## 8. Actual yield against the predictor

**By stage** (planned rates from the calibration, with this plan's observed rates in brackets):

| Stage | Planned | Observed |
|---|---:|---:|
| Generator acceptance | 0.82 (0.78) | 34 / 48 (0.71) |
| Evidence pass, of accepted | 9.8 predicted for these 34 (mean 0.29) | 8 (0.24) |
| Review acceptance, of passing | 0.86 (0.83) | 7 / 8 (0.88) |
| Usable | 10.5 at the calibration's rates; 9.5 at this plan's | **7** |
| Usable, positions 1–195 | 62.9 at the calibration's rates | **57** |

**Why batch 4 fell short:**
- **Given the 34 accepted questions,** the predictor expected 9.8 evidence passes and about 8.2 usable
  questions; the batch gave 8 and 7.
- **The rest of the gap (from 9.5) is the generator:** it accepted 34 rather than about 37, and its
  rejections fell on the higher-scoring passages (§6).

**The predictor's accuracy** (measured, out of sample):
- **This batch:** AUC 0.639, against 0.534 for the old sentence share: its weakest batch.
  - Its Brier score, 0.180, equals that of the batch's own constant rate and beats the calibration's
    constant (0.221).
  - So it was calibrated but ranked weakly in this narrow band of scores (mean 0.29).
- **All 195 positions (149 accepted questions):** AUC 0.796 (sentence share 0.683). Brier 0.184, against
  0.248 for either constant.
- **Calibration, all 195 positions:** right or conservative from 0.2 up, optimistic below 0.2.

| Predicted pass | Accepted questions | Predicted | Observed |
|---|---:|---:|---:|
| Below 0.2 | 35 | 4.1 | **2** |
| 0.2–0.4 | 35 | 10.6 | 15 |
| 0.4–0.6 | 40 | 19.7 | 22 |
| 0.6–0.8 | 28 | 19.0 | 19 |
| 0.8 and above | 11 | 10.0 | 10 |

**Re-estimate for the 163 unrun positions (199–361)** (projections):
- **Positions 196–198:** generated but not audited. Their passages score 0.07, 0.36 and 0.03, worth about
  0.4 usable questions.
- **The rates used:** this plan's observed rates over 195 positions, generator 0.76 and review 0.84.

| Model | 199–361 | Batch 5 (199–248) | Total with the 88 |
|---|---:|---:|---:|
| Observed rates | 22.1 | 8.9 | 111 |
| Observed rates, passages below 0.2 at half their predicted passes (as observed) | 19.1 (bootstrap 16.5–22.3) | 8.5 | 108 |
| Observed rates, passages below 0.2 giving nothing | 16.3 | 8.2 | 105 |
| The second row, at batch 4's realisation (7 of 9.5) | 14.1 | 6.3 | 102 |

**By block** (the second model):

| Positions | Mean score | Below 0.2 | Expected usable | Requests per usable | Cumulative total |
|---|---:|---:|---:|---:|---:|
| 199–248 (batch 5) | 0.28 | 16 | 8.5 | 5.9 | 97 |
| 249–297 | 0.24 | 21 | 6.7 | 7.3 | 104 |
| 298–347 | 0.14 | 38 | 3.3 | 15 | 107 |
| 348–361 | 0.12 | 12 | 0.6 | 22 | 108 |

- **Batch 5 split by score:** its 34 passages at 0.2 and above are expected to give 8.2 usable questions,
  4.2 requests each. Its 16 below 0.2 are expected to give 0.4, 44 requests each.

## 9. Did the predictor and the administrative filter help?

**The predictor:**
- **Ranking:** weak within this batch (AUC 0.64), still good over the plan (0.80 over 149 questions).
- **Calibration:** right from 0.2 up; below 0.2 it predicts about twice the passes observed. This batch's
  low end matched its prediction (2 against 1.9), so the low end is small rather than empty.
- **What it cannot see:** the generator's failures. Most of this batch's shortfall came from rejections
  on high-scoring narrative passages.

**The administrative filter:**
- **No administrative question reached the review:** no trading-plan, insider-trading, cybersecurity,
  Item 6, Section 13(r) or website-services question.
- **The flagged position:** 192 was the window's one flagged pattern (a new-standard notice). It gave a
  question the review rejected, so the flag was right.
- **What remains flagged:** two forward-looking-statement notices (265, 348) and one new-standard notice
  (295), none in batch 5's window.

## 10. Systematic weaknesses of Qwen's outputs in this batch

1. **Dropping the heading that scopes a row:** five questions (160, 166, 178, 182, 187). One is wrong
   (178, heading above the window); four are imprecise.
2. **Empty fields and garbled type names:** 163, 168 and 175, 3 of the latest execution's 43 records (7%).
   New to this plan.
3. **Period framing:** 192 (a planned adoption already made) and 171 (a year's acquisitions dated as a
   balance).
4. **Copied wording:** 4 of 48. **No company named:** 4 of 48. **Too long:** 154 (41 counted words).
5. **Units dropped:** 6 answers give no "millions" (batch 3: 3).
6. **Template repetition across years:**
   - Amazon's operating expenses (165, 183);
   - income-tax notes (five questions);
   - balance-sheet totals (five questions).
7. **Better than in earlier batches:**
   - one factual error in 34, the lowest rate yet;
   - no administrative question;
   - correct company attribution throughout;
   - no question depends on text past the window.

## 11. Comparison with earlier batches

| | Batch 1 | Batch 2 | Batch 3 | Batch 4 | Positions 1–195 |
|---|---|---|---|---|---|
| Positions | 1–49 | 50–97 | 98–143 | 148–195 | 1–195 (with 144–147) |
| Passages completed | 49 | 48 | 46 | 48 | 195 |
| Generator acceptance | 35 (71%) | 41 (85%) | 37 (80%) | 34 (71%) | 149 (76%) |
| Mean predicted pass, of accepted | 0.62 | 0.48 | 0.31 | 0.29 | 0.43 |
| Evidence pass, of accepted | **24 (69%)** | 21 (51%) | 14 (38%) | 8 (24%) | 68 (46%) |
| Review: accept / revise / reject, of accepted | 26 / 4 / 5 | 39 / 2 / 0 | 34 / 3 / 0 | 32 / 1 / 1 | 133 / 10 / 6 |
| Review acceptance, of evidence-passing | 0.75 | 0.95 | 0.79 | 0.88 | 0.84 |
| Factual errors, of accepted | 6 (17%) | 2 (5%) | 2 (5%) | **1 (3%)** | 11 (7%) |
| Usable per passage | 18 (37%) | **20 (42%)** | 11 (24%) | 7 (15%) | 57 (29%) |
| Tokens per passage | 1,314 | 1,353 | 1,397 | 1,358 | 1,355 |

## 12. Recommendation

**Positions 196–197: no recovery is needed.**
- **They are complete:** both records are `done` on the first attempt, with no error and one question
  each, which the generator accepted and the validator accepts again. So is 198. Plan positions 1–198
  each have exactly one record.
- **They came from the latest execution,** whose reported 43 requests and 59,013 tokens include them.
  Generating them again would only add second records for the same passages.
- **What to do:** audit 196–198 with batch 5's records. Their three questions are worth about 0.4 usable
  questions (the passages score 0.07, 0.36 and 0.03).

**Batch 5 (positions 199–248) is worth running unchanged, as the last batch of this plan.**
- **Cost:** 50 requests, about 68,000 tokens and about 25 minutes of generation, plus a full review.
- **Expected:**
  - about 6–9 usable questions: 8.5 projected, or 6.3 if it falls short as batch 4 did;
  - that is 6–8 requests and 8,000–11,000 tokens per question, about batch 4's actual rate (6.9 requests
    and 9,300 tokens).
- **What it adds:**
  - an even mix, 10 passages each from JPMorgan, Pfizer, Tesla, Walmart and ExxonMobil, which lifts Pfizer,
    now the thinnest company (10);
  - almost nothing for FY2024 (3 passages, about 0.6 questions).
- **What it buys:** a set of about 94–97 questions. A validation metric's standard error (at a 50% rate)
  falls from about 0.053 to 0.051.
- **Its low-scoring passages:** the 16 below 0.2 cost 16 requests for about 0.4 questions. Dropping them
  would need a new plan version, which is not worth it for about 22,000 tokens.
- **What to watch:** the empty-field rejections. If they exceed about 10% of batch 5's records, pause
  before continuing.

**Then stop this plan.**
- **Positions 249–297:**
  - about 5–7 questions at 7–9 requests each, for a total of about 104;
  - 18 of their 49 passages are from 2019, which would grow to about 22 questions, the largest year by
    far, while FY2024 stays near 9.
- **Positions 298–361:** as before, not worth running: 15–50 requests per question, nearly all JPMorgan and
  ExxonMobil.
- **What this plan supports:** about 95 questions, with FY2024 near 9.
  - A larger or better-balanced set needs new supply, such as a unit cap of 4–5 or a larger validation
    split, in a new plan version.
  - This revises the earlier advice to run through position 297: batch 4 gave 74% of its estimate, and
    what lies beyond 248 would worsen the year balance.

**Other decisions for you, none blocking batch 5:**
- **149:** usable under the rules, but a train passage of Pfizer's FY2025 filing restates its fact (§7).
  - I recommend keeping it, since no training pair uses that passage.
  - Excluding it would need a mechanism the decision format lacks: a revise or reject must name a failed
    aspect, and 149 fails none.
- **Recording:** batch 4's decisions (positions 148–195) can go into a new versioned file in
  `data/validation_review/`, in the format of positions 1–147, once you approve them.
- **Build-time caps:** still open. Three filings now give four questions each.
