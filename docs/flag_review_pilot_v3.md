# Review of the 12 new `pilot_v3` flags

Status: 2026-10-29.

> **Auditors: do not read this page before finishing the blind audit in
> `data/eval/audit/pilot_v3/`.** It shows AI grades for passages in that
> sample. The same applies to [pool_review_pilot_v3.md](pool_review_pilot_v3.md).

**Who made these recommendations.** They are an AI second look by
`claude-opus-5-5`, the model that proposed the labels. They are not human
decisions.

**Effect on the labels.** No label was changed. A change can only enter
through a new label version, made by a person:

1. a spec with curated changes;
2. `dev-labels`;
3. `export-dev`.

The flags are `flg_013`–`flg_024` in `data/eval/dev_ai/pilot_v3/flags.jsonl`.
The rubric is v1.1 ([eval_methodology.md](eval_methodology.md) §3).

## Summary

| Flag | Question | Passage evidence | AI grade (confidence) | Why flagged | Recommendation | Uncertainty |
|---|---|---|---|---|---|---|
| flg_013 | q_0001 Apple FY2024: risks of hardware built by a few manufacturers abroad | FY2024 Item 1A: "vendor non-trade receivables resulting from purchases of components by outsourcing partners … concentrated among a few individual vendors located primarily in Asia … no assurance such procedures will … avoid losses" | 1 (medium) | P1 open question; candidate 2 | **Retain 1** | Medium. 2 is defensible if any risk tied to the partners counts. |
| flg_014 | q_0003 Amazon: reasons for the 2023 AWS profit change | FY2023 MD&A: "The increase in technology and infrastructure costs … in 2023 … is primarily due to an increase in spending on infrastructure and increased payroll"; sales and marketing include "sales commissions related to AWS" | 0 (low) | Low confidence | **Revise to 1**, together with flg_015 | High. Grade the pair together: both 1 or both 0. |
| flg_015 | q_0003 | FY2023 MD&A: infrastructure costs are "necessary to support AWS and other Amazon businesses", "allocated to segments based on usage", and rose on "spending on infrastructure and increased payroll" | 1 (low) | P1 open question; candidate 2 | **Retain 1**; reject 2 | Medium. Between 0 and 1, not 2. |
| flg_016 | q_0003 | Same passage as flg_015 | 1 (low) | Low confidence | **Retain 1** | Medium |
| flg_017 | q_0006 Tesla: revolving credit facility undrawn at end of 2023 | FY2022 MD&A: "In January 2023, we entered into an unsecured revolving credit facility providing for a commitment of up to $5.0 billion." | 1 (low) | Low confidence | **Retain 1** | Medium. The strict temporal reading gives 0. |
| flg_018 | q_0009 ExxonMobil: 2022 capital spending by segment and U.S./non-U.S. | FY2024 financial information: "Additions to property, plant and equipment … 2022: 18,338" | 1 (low) | Low confidence | **Retain 1 under the current convention**; the reviewer should decide the convention | High |
| flg_019 | q_0009 | FY2023 liquidity: 2022 "spending for additions to property, plant and equipment of $18.4 billion" | 1 (low) | Low confidence | As flg_018 | High |
| flg_020 | q_0009 | FY2023 financial information: "Additions to property, plant and equipment … 2022: 18,338" | 1 (low) | Low confidence | As flg_018 | High |
| flg_021 | q_0009 | FY2022 financial information: "Additions to property, plant and equipment … 2022: 18,338" | 1 (low) | Low confidence | As flg_018 | High |
| flg_022 | q_0009 | FY2022 liquidity: 2022 "spending for additions to property, plant and equipment of $18.4 billion" | 1 (low) | Low confidence | As flg_018 | High |
| flg_023 | q_0005 Pfizer: COVID-19 revenue reduction, 2023 vs 2022 | FY2023 product table: "Comirnaty … Worldwide \| $11,220 \| $37,806 \| (70)"; "Declines largely driven by lower contracted deliveries and demand"; footnote on the $3.5B Paxlovid reversal | 1 (medium) | P5; candidate 2 | **Retain 1** | Low to medium |
| flg_024 | q_0008 Walmart: fiscal 2024 net sales by segment | FY2026 disaggregated tables, "Fiscal Years Ended January 31," 2026 \| 2025 \| 2024: Walmart U.S. "Total … $441,817", International "Total … $114,641", Sam's Club U.S. "Total … $86,179" | 2 (medium) | P3; the only new grade 2 | **Retain 2** | Low |

**Result.** Retain 11 grades. Revise one (flg_014, 0 → 1); flg_015 and
flg_016 must be decided with it.

## Reasoning

### flg_013: retain 1

The passage links a credit-loss risk to the same outsourcing partners,
concentrated among a few vendors in Asia. Under P1, a passage that states one
element of the answer is 2.

The element this passage states is a receivables risk. The question asks
about risks of relying on a few manufacturers abroad: supply disruption,
control, geopolitics. Those are stated in FY2024 Item 1A chunks 4 and 5,
which are already grade 2. Reading the receivables risk as an answer element
stretches the question, so the passage stays 1.

### flg_014 to flg_016: retain chunk 12 at 1; raise chunk 13 to 1

Both chunks contain the same sentence on the 2023 increase in technology and
infrastructure costs. The AWS answer, in chunk 10 (already grade 2), cites
higher payroll and technology infrastructure spending as offsets.

- **Why 1.** Neither passage says anything about AWS operating income, so
  neither is 2. Under P1, a passage that "needs outside information to
  connect to the question" is 1, which fits both.
- **Why not 2 or 0.** Grading the passages differently would treat two
  copies of the same evidence inconsistently.
- **If a reviewer reads the topic strictly.** Company-wide expense lines are
  not AWS profit reasons. Both should then be 0, decided together.

### flg_017: retain 1

The passage names the requested facility and its $5.0B commitment, the
upper bound of the answer. Being from the FY2022 report, it cannot say what
was drawn at the end of 2023.

This is consistent with `pilot_v2`, which labels the FY2023 description of
the facility without its balance as 1. A strict reading of "information
about other years is 0" gives 0. The passage describes January 2023, inside
the asked year, so 1 is the closer fit.

### flg_018 to flg_022: retain 1, but decide the convention

All five passages give one total of a related measure:
- additions to property, plant and equipment of $18,338M;
- or cash spending of $18.4B.

None gives the Capex measure the question's answer uses ($22,704M) or any
segment or U.S./non-U.S. split.

**Why 1 under the current convention.** Since `pilot_v1`, totals without the
requested split have been 1. `pilot_v2` labels the same kind of rows 1 with
low confidence, so 1 is consistent.

**Why the convention is in doubt.** The passages answer no part of a
breakdown question. Under the rubric's "a table that contains the line item
but not the requested breakdown" they qualify for 1 only loosely, because
the line item is a different measure.

**Recommendation.** A person should decide this class once, for `pilot_v2`
and `pilot_v3` together, in the next label version. These five labels carry
most of E5's gain on `q_0009` (nDCG@10 0.434 → 0.681).

### flg_023: retain 1

The passage gives the complete vaccine part:
- 2023 and 2022 revenues;
- the 70% decline;
- the demand drivers.

It does not give the Paxlovid part. Under P5 option A, if the full answer is
normally reported in one table, a passage with only some of its elements is
1. This answer is: the FY2023 revenue bridge has both products, and `pilot_v2`
already grades it 2. This is a clear application of an adopted option.

### flg_024: retain 2

All three fiscal 2024 segment totals are in the passage, and so is their
period: the "Fiscal Years Ended January 31," column headers. The passage
also says the tables disaggregate segment net sales.

Under P3, a later filing's comparative is graded on content. The renamed
"Sam's Club U.S." segment is the same segment, and its totals equal those in
the FY2024 report's own segment table, which is already a `pilot_v2` grade 2.

## What the recommendations would change

The six stored rankings, re-scored in memory with modified `pilot_v3`
qrels. Nothing was written.

| Variant | BM25 g / c | BGE g / c | E5 g / c |
|---|---|---|---|
| `pilot_v3` as is (nDCG@10) | 0.170 / 0.194 | 0.286 / 0.297 | 0.361 / 0.376 |
| Recommendations applied (flg_014 → 1) | 0.170 / 0.194 | 0.283 / 0.294 | 0.363 / 0.378 |
| Strict alternative (flg_015/016, flg_017, flg_018–022 → 0) | 0.170 / 0.196 | 0.280 / 0.284 | 0.336 / 0.351 |

g = global mode; c = company mode.

- **Recommendations applied.** Small changes: at most 0.003 nDCG@10. BM25
  Recall@100 falls by 0.010, because the denominator grows by one positive.
- **Strict alternative.** The nDCG@10 order is unchanged. Recall@100 returns
  to favouring BM25 company (0.694 against 0.642 for E5 company), and BGE
  and E5 tie on global MRR@10 (0.550).

These decisions belong to the human audit. All 11 flagged passages (12
flags) are in its sample.
