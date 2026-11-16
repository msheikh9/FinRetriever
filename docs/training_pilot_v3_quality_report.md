# Training-data pilot `pilot_v3`: quality audit

Status: 2026-10-31. This audits the second real Qwen pilot: `bge_base_ft_v2`,
plan `pilot_v3`, prompt `sec-qgen-v2`, validator `sec-qcheck-v3`, 50 train
passages, 70 pairs.

- Every decision below is an AI review (Claude), not human annotation, and
  none of it is a gold label.
- Nothing was generated, trained or committed for this audit, and no pair
  was changed. The decisions are recorded separately in
  `data/train_review/pilot_v3_quality_decisions.jsonl`.

## 1. Decision

**NO-GO for a training smoke test on the pairs as built. GO once three
offline fixes and a small validation pilot are done (§10). Generation does not
need another prompt revision before the smoke test; it needs one before it is
scaled.**

- **Quality held and phrasing improved.**
  - 57 of 70 pairs are accepted (81%), 7 need revision and 6 should be
    rejected.
  - ISO dates fell from 19% of questions to none.
  - "fiscal YYYY" or "FY" phrasing rose from 22% to 46%.
  - The type `other` fell from 41% to 10%.
- **A leakage gap blocks training.** Prompt v2 asks answers to keep their
  units ("$57,411 million"). The evidence locator turns that into a scaled
  value that no longer matches the unscaled table cell, so the exact-figure
  check misses it. Six pairs that should have been rejected were built:
  - 002 and 052: the same labelled figure is in blocked passages;
  - 006, 007, 034 and 035: it is in a validation passage's visible text.
- **The fail-closed loader passes them too,** because it recomputes with the
  same locator.
- **The evaluation-evidence rejections were conservative, not wrong.**
  - 2 of the 7 had their evidence in an overlap-group copy of an evaluation
    passage.
  - 5 had it only in passages blocked for reusing *other* evaluation text.
  - None had it in an evaluation-pool passage itself.
  - The one cross-split rejection was correct.
- **There are no validation pairs,** so a trained model could not be
  selected or stopped early.

## 2. Provenance

| Check | Result |
|---|---|
| Corpus | `corpus_v1@1.2`, SHA-256 `ff9dadf3…4004`, 10,137 passages; matches the config |
| Preparation | `bge_base_ft_v2`, manifest `c9b8998e…7e15`; every file matches; the evidence-reuse index rebuilds to its recorded digest `d3321b40…5977` |
| Plan | `train.pilot_v3.json`, SHA-256 `55b33c98…c407`, unchanged; all 50 records follow it in plan order |
| Generation run | Bound, with no continuation; prompt `sec-qgen-v2` (`f42bdc54…`) and parameters `42d5fe54…` match the config |
| Records | 50 done, 0 failed, every one on its first attempt, on 2026-10-30 from 21:01 to 21:33 UTC. Completion tokens 164–420 (median 227) against the 768 limit; no reasoning tokens |
| Validator | All 50 on `sec-qcheck-v3`; `revalidate --dry-run` has nothing to change |
| Pairs | `train_pairs.jsonl` `b6d0bb4c…f762` matches `pairs.meta.json`; `check` finds 0 violations; validation pairs file empty |
| Trainer interface | All 70 load through `load_training_pairs` (contract, candidate, evidence, equivalence and lexical checks), 85 with the 15 carried-over pairs |
| Protected data | 109 files checksummed before and after this audit: unchanged |

## 3. Per-pair decisions

Each pair was read against its positive's full visible text.
- **Accept:** correct company, period, figure and unit; supported by the
  quote; a useful financial retrieval query.
- **Revise:** a specific error a regenerated question could fix.
- **Reject:** no financial content, or no answer.

"Hold: leak" marks pairs that are correct as questions but should have been
rejected by the evidence check (§6).

| Pair | Source | Type | Question (verbatim) | Answer (verbatim) | Decision | Note |
|---|---|---|---|---|---|---|
| 001 | AAPL FY2019, narrative/mdna | accounting_policy | What was the primary objective of Apple's marketable securities investment portfolio in fiscal 2019? | Minimizing the potential risk of principal loss | **Accept** | Boilerplate investment policy, repeated yearly. |
| 002 | AAPL FY2020, table/statements | financial_table | What was Apple's net income for the fiscal year ended September 26, 2020, in millions of dollars? | $57,411 million | **Accept** (hold: leak) | Correct, but a blocked passage holds the same figure: missed because the answer says "million". |
| 003 | AAPL FY2020, table/statements | financial_table | How much did Apple report as depreciation and amortization expense for the year ended September 28, 2019? | $12,547 million | **Accept** | Correct prior-year figure; names its year. |
| 004 | AAPL FY2021, narrative/risk | mdna | What was Apple's estimated maximum one-day loss in fair value for its investment portfolio, debt, and derivative positions as of September 25, 2021, and how does it compare to the prior year? | A maximum one-day loss in fair value of $550 million as of September 25, 2021, compared to $551 million as of September 26, 2020. | **Revise** | Misstates the scope: the $550 million is the foreign-currency VaR, not the investment portfolio, debt and derivatives. |
| 005 | AAPL FY2021, narrative/risk | risk | According to Apple's 95% confidence sensitivity analysis for fiscal 2021, what factors could cause actual future gains and losses to differ materially from the reported estimates? | Inherent limitations in predicting the timing and amount of changes in interest rates, foreign currency exchange rates, and the Company's actual exposures and positions. | **Accept** | Boilerplate caveat, repeated yearly. |
| 006 | AAPL FY2022, table/statements | financial_table | What was Apple's gross property, plant and equipment in millions of dollars as of September 24, 2022? | $114,457 million | **Accept** (hold: leak) | Correct, but a validation passage shows the same figure: missed because the answer says "million". |
| 007 | AAPL FY2022, table/statements | financial_table | How much interest and dividend income did Apple report in fiscal 2022, in millions? | $2,825 million | **Accept** (hold: leak) | Correct; same cross-split miss as 006. |
| 008 | AAPL FY2023, narrative/statements | financial_table | What was the fair value of Apple's Level 2 corporate debt securities held as non-current marketable securities in fiscal 2023? | 58,243 | **Accept** | Correct; answer has no unit (the header is not visible). |
| 009 | AAPL FY2023, narrative/statements | debt_liquidity | How much of Apple's Level 2 certificates of deposit and time deposits was classified as cash and cash equivalents in fiscal 2023? | 1,034 | **Accept** | Correct. |
| 010 | AAPL FY2024, table/risk | financial_table | What was the potential decline in fair value of Apple's investment portfolio in dollars in millions as of September 28, 2024, resulting from a hypothetical 100 basis point increase in interest rates across all tenors? | $2,755 million | **Accept** | Correct; long (35 words). |
| 011 | AAPL FY2024, table/risk | risk | What is the primary method Apple uses to assess the potential impact of foreign exchange rate fluctuations on its derivative positions? | A value-at-risk (VAR) model using a Monte Carlo simulation | **Accept** | Correct; names no period. |
| 012 | AAPL FY2025, narrative/statements | debt_liquidity | How much term debt did Apple report as a current liability in fiscal 2025? | $12,350 | **Accept** | Correct. |
| 013 | AMZN FY2020, table/risk | financial_table | What was the total principal amount of Amazon's corporate debt securities maturing in 2023 as of December 31, 2020? | $5,531 million | **Accept** | Correct. |
| 014 | AMZN FY2020, table/risk | risk | How does Amazon describe the impact of interest rate movements on the fair value of its fixed-rate long-term debt? | The fair value will generally fluctuate with movements of interest rates, increasing in periods of declining rates and declining in periods of increasing rates. | **Accept** | Correct; boilerplate; names no period. |
| 015 | AMZN FY2021, narrative/risk | risk | What were the estimated fair value declines for Amazon's foreign funds as of December 31, 2021, resulting from assumed 5%, 10%, and 20% adverse foreign exchange changes? | $1.1 billion, $2.3 billion, and $4.5 billion | **Accept** | Strong: three FX sensitivities. |
| 016 | AMZN FY2022, table/statements | financial_table | How many years remained in the weighted average useful life of Amazon's technology- and content-based finite-lived intangible assets at December 31, 2022? | 3.2 years | **Accept** | Correct. |
| 017 | AMZN FY2023, narrative/mdna | financial_table | What was the total value of cash provided by financing activities for Amazon in the year ended December 31, 2023? | $(15.9) billion | **Accept** | Correct; asks "provided by" for cash used. |
| 018 | AMZN FY2024, table/business | financial_table | What was the total leased square footage, in thousands, reported by Amazon for fiscal 2024? | 701,974 | **Accept** | Correct; operational. |
| 019 | AMZN FY2024, table/business | other | How many physical stores did Amazon operate in North America as of December 31, 2024? | 618 | **Accept** | Correct; operational. |
| 020 | AMZN FY2025, narrative/statements | debt_liquidity | How much did Amazon report as total finance lease liabilities, including interest, due in the year ended December 31, 2030? | 1,122 | **Accept** | Correct. |
| 021 | JPM FY2019, table/mdna | financial_table | What was the period-end value of JPMorgan Chase & Co's investment securities portfolio in fiscal 2019? | $396,416 million | **Accept** | Correct. |
| 022 | JPM FY2019, table/mdna | risk | What was the average credit rating assigned to the securities in JPMorgan Chase & Co's investment securities portfolio as of December 31, 2019? | AA+ | **Accept** | Correct. |
| 023 | JPM FY2020, narrative/mdna | profitability | What was the total net income or loss for JPMorgan Chase & Co's Treasury and CIO segment in fiscal 2020? | A loss of $1,403 million | **Accept** | Correct; Treasury and CIO is part of Corporate, not a segment. |
| 024 | JPM FY2020, narrative/mdna | accounting_policy | How much were the tax-equivalent adjustments for JPMorgan Chase & Co's Treasury and CIO segment for the year ended December 31, 2020? | $241 million | **Revise** | Attaches a footnote's $241 million to Treasury and CIO; the visible text does not show which line it qualifies. |
| 025 | JPM FY2022, narrative/business | other | How many percent of the Board of Directors at JPMorgan Chase were military veterans as of fiscal 2022? | 9% | **Reject** | Board demographics: no financial content. |
| 026 | JPM FY2023, narrative/statements | other | In what court was the putative class action regarding securities lending antitrust filed by JPMorgan Chase & Co in fiscal 2023? | United States District Court for the Southern District of New York. | **Revise** | JPMorgan is the defendant, not the filer; court-name trivia. |
| 027 | JPM FY2024, table/business | financial_table | What was the approximate square footage of JPMorganChase's facilities in the United States as of December 31, 2024? | 56.0 million square feet | **Accept** | Correct; operational. |
| 028 | JPM FY2024, table/business | other | Where is JPMorganChase's headquarters located, and how many stories does the building have? | New York City at 383 Madison Avenue, a 47-story office building | **Reject** | Headquarters building trivia; names no period. |
| 029 | JPM FY2025, table/mdna | financial_table | What were JPMorgan Chase's total assets under management as of December 31, 2025? | $4,791 billion | **Accept** | Correct. |
| 030 | JPM FY2025, table/mdna | mdna | What percentage increase did JPMorgan Chase report in client assets for 2025 compared with 2024? | 20% | **Accept** | Correct; omits the Asset & Wealth Management segment. |
| 031 | PFE FY2019, narrative/risk | risk | What is the duration of the Corporate Integrity Agreement Pfizer entered into with the HHS Office of Inspector General in May 2018? | five years | **Accept** | Correct. |
| 032 | PFE FY2019, narrative/risk | risk | According to Pfizer's fiscal 2019 10-K, what specific types of intellectual property rights does the company rely on to protect its proprietary technologies? | patent, trademark, trade dress, copyright, trade secret and domain name protection laws | **Accept** | Boilerplate IP list, repeated yearly. |
| 033 | PFE FY2021, table/mdna | debt_liquidity | What was the aggregate principal amount of the senior unsecured sustainability notes Pfizer completed a public offering of in August 2021? | $1 billion | **Accept** | Correct. |
| 034 | PFE FY2022, table/statements | revenue | What were the total Oncology revenues for Pfizer Inc. and Subsidiary Companies in fiscal 2022, stated in millions? | $12,132 million | **Accept** (hold: leak) | Correct; same cross-split miss as 006. |
| 035 | PFE FY2022, table/statements | financial_table | How much revenue did Pfizer Inc. and Subsidiary Companies generate from Ibrance for the year ended December 31, 2022? | 5,120 million | **Accept** (hold: leak) | Correct; same cross-split miss as 006. |
| 036 | PFE FY2023, narrative/other | risk | Who is responsible for overseeing Pfizer's cybersecurity risk management program in fiscal 2023? | The Chief Information Security Officer (CISO) | **Accept** | Correct; governance, little financial content. |
| 037 | PFE FY2024, table/statements | financial_table | What was the total amount of cash Pfizer paid for income taxes, net of refunds, during fiscal 2024? | $3,605 million | **Accept** | Correct (rejected in pilot 1, recovered by validator v3). |
| 038 | PFE FY2024, table/statements | accounting_policy | How does Pfizer report the remaining Transition Tax liability as of December 31, 2024? | In noncurrent Other taxes payable | **Accept** | Correct. |
| 039 | PFE FY2025, table/mdna | financial_table | How much cash did Pfizer provide from or use in operating activities for the fiscal year ended December 31, 2025? | The specific value is not provided in the excerpt | **Reject** | Non-answer ("not provided in the excerpt"): the figure is cut off. |
| 040 | TSLA FY2019, narrative/statements | accounting_policy | How much stock-based compensation expense did Tesla record related to specific performance-based awards for the year ended December 2017? | $7 million | **Accept** | Correct; vague "specific awards"; a 2017 figure. |
| 041 | TSLA FY2020, table/statements | financial_table | What was the present value of Tesla's operating lease obligations as of December 31, 2020, in millions? | $1,540 million | **Accept** | Correct. |
| 042 | TSLA FY2020, table/statements | debt_liquidity | How much cash did Tesla pay for amounts included in the measurement of lease liabilities from finance leases in the year ended December 31, 2020, in millions? | $338 million | **Accept** | Correct; copies a row label. |
| 043 | TSLA FY2021, table/statements | financial_table | What was the total assets reported for Tesla's Variable Interest Entities in millions at December 31, 2021? | $4,626 million | **Accept** | Correct. |
| 044 | TSLA FY2021, table/statements | other | How many shares of common stock did Tesla's CEO purchase from the company in May 2019? | 514,400 shares | **Accept** | Correct. |
| 045 | TSLA FY2023, narrative/mdna | revenue | What were Tesla's total revenues and the year-over-year change in those revenues for fiscal 2023? | Total revenues were $96.77 billion, representing an increase of $15.31 billion compared to the prior year. | **Accept** | Strong: revenue and change. |
| 046 | TSLA FY2023, narrative/mdna | profitability | How much net income attributable to common stockholders did Tesla report for fiscal 2023, and how did it change compared to the previous year? | Net income attributable to common stockholders was $15.00 billion, a favorable change of $2.44 billion compared to the prior year. | **Accept** | Strong: net income and change. |
| 047 | TSLA FY2024, narrative/risk | risk | What specific factors related to artificial intelligence development and data center operations does Tesla identify as challenges in its fiscal 2024 10-K? | Availability and cost of energy, processing power limitations, and substantial power requirements for data centers. | **Accept** | Correct. |
| 048 | TSLA FY2024, narrative/risk | risk | According to Tesla's fiscal 2024 10-K, what consequences could result from the unavailability of a component or supplier? | Production delays, idle manufacturing facilities, product design changes, loss of access to important technology and tools, and impacts on capacity expansion and contract fulfillment. | **Accept** | Correct; boilerplate risk factor. |
| 049 | TSLA FY2025, narrative/mdna | mdna | What specific artificial intelligence feature does Tesla, Inc. cite as a method to generate demand for its vehicles in fiscal 2025? | FSD (Supervised) | **Accept** | Correct. |
| 050 | TSLA FY2025, narrative/mdna | risk | How did the rapid increase in interest rates by central banks in developed countries affect the affordability of Tesla, Inc.'s vehicle lease and finance arrangements? | It impacted the affordability of vehicle lease and finance arrangements. | **Revise** | Names no period; the answer only restates the question. |
| 051 | WMT FY2020, narrative/mdna | tax | What was the amount of unrecognized tax benefits considered uncertain tax positions recorded as liabilities for Walmart Inc. as of January 31, 2020? | $1.8 billion | **Accept** | Correct. |
| 052 | WMT FY2021, table/mdna | financial_table | How much net cash did Walmart use in financing activities during the fiscal year ended January 31, 2021? | $(16,117) million | **Accept** (hold: leak) | Correct; same evaluation-evidence miss as 002. |
| 053 | WMT FY2021, table/mdna | mdna | What was the primary reason for the $1.8 billion increase in Walmart's net cash used in financing activities in fiscal 2021 compared to the previous year? | Timing of issuances and repayments of long-term debt | **Accept** | Strong: causal MD&A question. |
| 054 | WMT FY2022, narrative/mdna | risk | What was the impact of impairment charges on assets held and used for Walmart Inc. during fiscal 2022, and how sensitive was the fiscal 2022 impairment to a 10% decrease in undiscounted cash flows? | Impairment charges on assets held and used were immaterial in fiscal 2022, and the impairment would not change materially with a 10% decrease in undiscounted cash flows for stores or clubs with indicators of impairment. | **Accept** | Correct; typed risk, not accounting policy. |
| 055 | WMT FY2024, narrative/risk | risk | According to Walmart Inc.'s fiscal 2024 risk factors, what are the three specific reasons cited for why food products might fail to meet customer expectations or regulatory standards? | The product is a source of foodborne illness. | **Revise** | Wrong premise: the three reasons cover three product kinds; food products have one. |
| 056 | WMT FY2025, table/mdna | mdna | According to Walmart Inc.'s management discussion for fiscal 2025, what specific factors drove comparable sales growth in the U.S.? | Growth in transactions and unit volumes, with strong sales in grocery and health and wellness | **Accept** | Strong: comparable-sales drivers. |
| 057 | WMT FY2026, table/other | financial_table | How many shares did Walmart Inc. repurchase in the month of January 2026 under its publicly announced plans? | 3,073,037 shares | **Accept** | Correct. |
| 058 | XOM FY2019, table/statements | financial_table | What were the total consolidated costs incurred by Exxon Mobil Corp in fiscal 2019, and how did this compare to the amount reported for 2018? | Total consolidated costs incurred in 2019 were $19,240 million, up $2,912 million from 2018. | **Accept** | Correct. |
| 059 | XOM FY2019, table/statements | financial_table | According to the supplemental information, what was the total amount of development costs incurred by Exxon Mobil Corp's consolidated subsidiaries in the Asia region during 2019? | $1,317 million | **Accept** | Correct column reading. |
| 060 | XOM FY2020, table/other | financial_table | What was ExxonMobil's share of total assets for the fiscal year 2020? | 53,426 | **Revise** | Misreads the table: 53,426 is ExxonMobil's share of equity companies' total assets. |
| 061 | XOM FY2020, table/other | other | What percentage ownership interest did ExxonMobil hold in Aera Energy LLC as of December 31, 2020? | 48% | **Accept** | Correct. |
| 062 | XOM FY2021, narrative/other | accounting_policy | What monetary threshold did ExxonMobil set for disclosing environmental proceedings in its fiscal 2021 annual report? | $1 million | **Accept** | Correct; little financial content (a 45-word stub). |
| 063 | XOM FY2021, narrative/other | other | Where does ExxonMobil direct readers to find additional details on legal proceedings for the year ended December 31, 2021? | Note 16: Litigation and Other Contingencies of the Financial Section | **Reject** | Navigation pointer to a note, which prompt v2 forbids. |
| 064 | XOM FY2022, table/other | financial_table | What was the net income attributable to ExxonMobil in fiscal 2022, in millions of dollars? | 55,740 million dollars | **Accept** | Correct. |
| 065 | XOM FY2022, table/other | financial_table | How many weighted-average common shares were outstanding for ExxonMobil in fiscal 2021, in millions of shares? | 4,275 million shares | **Accept** | Correct. |
| 066 | XOM FY2023, narrative/business | financial_table | What was the total oil-equivalent production of all products for Exxon Mobil's developed consolidated subsidiaries in Asia in fiscal 2023? | 1,142 | **Revise** | Misreads the table: 1,142 is equity companies' developed reserves, not consolidated production. |
| 067 | XOM FY2023, narrative/business | financial_table | How many million barrels of crude oil did Exxon Mobil report as undeveloped resources in Canada/Other Americas for the year ended December 31, 2023? | 561 | **Accept** | Correct; "million barrels" is inferred, not shown in the visible text. |
| 068 | XOM FY2024, narrative/risk | risk | Where does Exxon Mobil Corp direct readers to find detailed information regarding its market risks in the fiscal 2024 annual report? | The section entitled "Market Risks" in the Financial Section of the report | **Reject** | Navigation pointer from a 406-character stub. |
| 069 | XOM FY2024, narrative/risk | risk | According to Exxon Mobil Corp's fiscal 2024 filing, what factors could cause the actual impact of future market changes to differ materially from forward-looking statements? | Factors discussed in the report, among other things | **Reject** | Vacuous answer to a forward-looking disclaimer. |
| 070 | XOM FY2025, table/statements | financial_table | What was the total amount of notes and accounts receivable for Exxon Mobil Corp as of December 31, 2025, in millions of dollars? | 44,562 million dollars | **Accept** | Correct. |

**Totals:** 57 Accept, 7 Revise, 6 Reject. Of the 57, 6 are held for the
leak, which leaves **51 clean pairs**.

## 4. Did prompt v2 improve phrasing?

| Measure | Pilot 1 (`sec-qgen-v1`, 37 pairs) | Pilot 3 (`sec-qgen-v2`, 70 pairs) |
|---|---|---|
| Questions with an ISO date | 7 (19%) | **0** |
| "fiscal YYYY" or "FY" phrasing | 8 (22%) | **32 (46%)** |
| Written full dates (when the period is a date) | 12 | 27 |
| Questions naming no year | 0 | 4 (011, 014, 028, 050) |
| Numeric answers with a unit | 14 of 22 | 37 of 50 |
| Type `other` | 41% | **10%** |
| Mean question length | 21.3 words | 20.0 words |
| Navigation-style questions ("where can … find", note numbers) | 0 | 2 accepted (063, 068) |

- **Yes, clearly, for dates and types.** Questions now read like an
  analyst's: "What was Apple's net income for the fiscal year ended September
  26, 2020, in millions of dollars?" A prior-year figure names its own year
  (003: "for the year ended September 28, 2019").
- **Rules still broken:**
  - 4 accepted questions name no period;
  - 6 rejected candidates name no company;
  - 2 navigation questions came from cross-reference stubs.

## 5. Rejections

**The generator rejected 22 of 100 candidates:**

| Reason | Count | Verdict |
|---|---|---|
| `missing_company` | 6 | Correct: the company is not named |
| `copied_from_passage` | 8 | 4 correct (whole prose sentences copied, 8–13 words); 2 uncertain (entity and place names in the run); 2 false (P02, where a chunk starting mid-table hides the row label from the exemption, and P48's "weighted-average interest rate on short-term borrowings") |
| `evidence_length` | 5 | Correct under the 300-character rule (301, 303, 335, 345, 365 characters). Three are table rows whose cells repeat a long header: the v2 prompt's "copy every cell up to the answer" pushes them over the limit, although validator v3 accepts skipped cells |
| `answer_not_supported` | 2 | Correct: one reads VaR columns wrongly; one quote leaves out the figure it answers with |
| `refers_to_context` | 1 | Correct ("According to the table…", and a note-number lookup) |

**Pair building rejected 8 of 78:**

| Candidate | Rule | Where the evidence is | Verdict |
|---|---|---|---|
| P08 Amazon lease maturities, FY2025 | evaluation | Overlap-group copy of an evaluation passage (beyond its visible text) | Legitimate, conservative |
| P29 Amazon capex 2023 | evaluation | Two overlap-group copies of evaluation passages | Legitimate, conservative |
| P12 Walmart $30.0 billion buyback authorization | evaluation | Only passages blocked by evidence reuse | Over-conservative |
| P15 Amazon FY2019 financing cash flow | evaluation | Only passages blocked by evidence reuse | Over-conservative |
| P15 Amazon FY2019 operating-lease cash | evaluation | Only a passage blocked by evidence reuse | Over-conservative |
| P28 Apple total assets, FY2025 | evaluation | Only a passage blocked by evidence reuse | Over-conservative |
| P36 Amazon contract-based intangibles, FY2022 | evaluation | Only a passage blocked by evidence reuse | Over-conservative |
| P23 JPMorgan veterans share, FY2022 | cross-split | Visible text of two validation passages (FY2021 and FY2024 Item 1) | Legitimate |

- **None of the 7 evaluation-evidence rejections had its evidence in an
  evaluation-pool passage.** The rule rejects evidence held by any blocked
  passage. That includes passages blocked because they substantially reuse
  *some* evaluation evidence.
- **Cost:** 5 of 78 candidates (6%).
- **Option:** scope the per-pair rule to evaluation passages and their
  overlap groups. Keeping it as it is is the safer choice for now.

## 6. Leakage and false-negative protections

**What worked:**
- Every positive is a train passage whose overlap group and split unit are
  wholly in train, and none is blocked.
- The highest word Jaccard against any evaluation question is 0.31, and no
  pair exceeds 0.5.
- Read side by side with the 10 evaluation questions, no pair paraphrases
  one. The closest topics are different facts:
  - 058, ExxonMobil's 2019 costs incurred, against `q_0009`, its 2022 capital
    spending;
  - 034 and 035, Pfizer's 2022 oncology and Ibrance revenue, against
    `q_0005`, COVID product revenue;
  - 056, Walmart's FY2025 comparable sales, against `q_0008`, FY2024 net
    sales by segment.
- The per-pair evidence check caught 8 candidates (§5). The Amazon FY2023
  capex sentence, the closest to `q_0003`, was among them.

**What failed: figures stated with a unit.**
- **The mechanism.** The locator matches the answer's figure to the
  evidence's figure under the same label; that is its "exact fact" match.
  An answer such as "$57,411 million" is normalised to `57411m`, while the
  table cell "57,411" is `57411`. The two never match, so a blocked or
  other-split passage holding the same labelled figure counts only as
  uncertain.
- **The trigger.** 29 of the 70 answers carry a scale word, encouraged by
  prompt v2's unit rule.
- **The result:**
  - 002 (Apple FY2020 net income) and 052 (Walmart FY2021 financing cash) have
    the figure in blocked passages (7 and 1);
  - 006, 007, 034 and 035 have it in a validation passage's visible text.
- **The fix.** Compare answer figures with and without their scale word when
  matching evidence figures, add a regression test, then rebuild the pairs
  offline. Those six pairs then fall out.

**False negatives:**
- 36 pairs record exact equivalents (82 passages) and 33 record uncertain
  ones (104 passages).
- The batch planner keeps them apart. With the 15 carried-over pairs, 85
  pairs fit in 6 batches of 16, 5 of them full, at 89% fill. At 64 per
  batch they need 3 batches at 44% fill.
- The unit bug also files some exact equivalents as uncertain. They are still
  kept out of negatives, but not counted as extra positives.
- Paraphrased or unit-converted evidence is still undetected.

## 7. Distribution

| | All 70 | Accepted 57 | Clean 51 |
|---|---|---|---|
| AAPL / AMZN / JPM / PFE / TSLA / WMT / XOM | 12 / 8 / 10 / 9 / 11 / 7 / 13 | 11 / 8 / 6 / 8 / 10 / 6 / 8 | 8 / 8 / 6 / 6 / 10 / 5 / 8 |
| JPMorgan share | 14% | 11% | 12% |
| Tables / narrative | 38 / 32 | 35 / 22 | 29 / 22 |

| Fiscal year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| All 70 | 8 | 11 | 10 | 9 | 9 | 13 | 9 | 1 |
| Clean 51 | 8 | 8 | 7 | 4 | 7 | 9 | 7 | 1 |

| Type | financial_table | risk | other | accounting_policy | debt_liquidity | mdna | profitability | revenue | tax |
|---|---|---|---|---|---|---|---|---|---|
| All 70 | 28 | 15 | 7 | 5 | 5 | 5 | 2 | 2 | 1 |
| Accepted 57 | 25 | 11 | 3 | 4 | 5 | 4 | 2 | 2 | 1 |

- **Duplicates:** none. The most similar pair of questions shares 52% of
  words (021 and 022, two facts from one JPMorgan table). No figure is asked
  twice, and no question is near one of the 15 carried-over pairs (maximum
  0.41).
- **Reused passages.** Four pairs reuse passages the first pilot used, all
  of whose pairs were ineligible. 002 asks the same fact as pilot 1's 003;
  the earlier copy is not in the training set.

## 8. Against the first pilot

| | Pilot 1 (`bge_base_ft_v1`) | Pilot 3 (`bge_base_ft_v2`) |
|---|---|---|
| Passages, proposals | 24, 48 | 50, 100 |
| Generator acceptance | 77% (validator v2) | 78% (validator v3) |
| Rejected at pair building | 0 (no evidence checks) | 8 |
| Pairs per request | 1.54 | 1.40 |
| Audit: Accept / Revise / Reject | 89% / 8% / 3% | 81% / 10% / 9% |
| Factual errors among pairs (scope, attribution, table reading, period) | 2 (5%) | 6 (9%): 004, 024, 026, 055, 060, 066 |
| No financial content or no answer | 1 | 6: 025, 028, 039, 063, 068, 069 |
| Clean after every check | 15 of 37 (41%) | 51 of 70 (73%) |
| Company balance | 2–7 pairs per company (XOM 2) | Every company 7–13 pairs |

- **Better:** phrasing, type spread, company balance, and the leakage
  checks, which keep 73% of pairs clean against 41%.
- **Worse:**
  - more table misreadings, because the chunk starts mid-table and the column
    headings are not visible (024, 060, 066);
  - more low-value questions from stub and non-financial passages.

## 9. Systematic weaknesses

1. **The unit bug in the leakage check** (6 pairs; §6). Blocking.
2. **Tables without visible headings.** When a chunk starts mid-table, the
   model guesses the entity, quantity or unit:
   - 060 and 066 misread the entity;
   - 024 attaches a footnote to a line it cannot see;
   - 067 infers a unit that is not shown;
   - 5 answers have no unit (008, 009, 012, 018, 020).
3. **Stub and cross-reference passages.** ExxonMobil's Item 3 (289
   characters) and Item 7A (406 characters) passed the source filter: 7A is a
   protected item, and both clear the 40-word minimum. They produced two
   navigation questions and one vacuous answer.
4. **Non-answers pass the validator.** 039's "The specific value is not
   provided in the excerpt" passed because "provided" appears in its quote.
5. **The prompt and the validator disagree on table quotes.** "Every cell up
   to the answer" yields quotes over 300 characters in rows that repeat
   headers (3 rejections), although validator v3 already accepts skipped
   cells.
6. **Low-value topics** from business and other sections: board
   demographics, headquarters, court names, disclosure thresholds (025, 026,
   028, 062).
7. **Prompt compliance:** 4 accepted questions name no period, and 6
   candidates name no company.

## 10. Recommendation

**Before a smoke test (no API calls except step 3):**
1. **Fix the locator's figure matching.** Make it insensitive to the scale
   word, add a regression test, and rebuild the `bge_base_ft_v2` pairs
   offline. 002, 052, 006, 007, 034 and 035 should fall out, leaving 64
   pairs.
2. **Apply the audit decisions explicitly.** Train only on accepted pairs,
   using `pilot_v3_quality_decisions.jsonl` the way the first pilot's
   decisions were used, never by editing pairs. That gives about 51 pairs,
   plus the 15 carried over: about 66.
3. **Run a small validation pilot** (for example 40–50 validation passages,
   at most 50 requests) under the fixed pipeline, with your approval. The
   training plan needs validation pairs for checkpoint choice.

**Then GO for a smoke test.** It checks mechanics, not quality:
- the loader and conflict-free batches of 16–32;
- the loss falls;
- checkpoints are written;
- a development evaluation runs.

About 66 pairs is far too few to measure domain adaptation, so no result
should be read as one.

**Before scaling generation:**
- a prompt `sec-qgen-v3` that does not ask for figures whose column headings
  or units are not visible, allows quotes that skip cells, and requires the
  period;
- a validator rule against non-answers;
- source filtering of stub and cross-reference passages, including protected
  items (a higher visible-word minimum);
- fewer business and other passages in plans;
- a decision on narrowing the per-pair evaluation rule (§5).
