# Expanded development pool `pilot_v3`: coverage, labels and quality review

Status: 2026-10-29. Everything here is **provisional development data**. The
new labels are AI proposals by `claude-opus-5-5`, the model that drafted the
questions and the `pilot_v1` labels. No person has reviewed them, and nothing
in this document is a benchmark result. See
[eval_methodology.md](eval_methodology.md) §11 for the tiers and rules.

## 1. Summary

- **What was done.** `pilot_v3` adds to the `pilot_v2` pool every passage
  that any of the six stored zero-shot runs ranked in its top 10. The six
  runs are BM25, BGE-base and E5-base, each in global and company mode. That
  added **117 passages** (92 from the companies asked about, 25 from other
  companies), each graded by an AI proposal under rubric v1.1. The 579
  `pilot_v2` labels, their 3 rule changes, 12 flags and audit dispositions
  are carried over unchanged.
- **What the new labels say.** 102 are grade 0, 14 grade 1 and 1 grade 2,
  so 15 new positives. Every one of the 15 was in a dense run's top 10 and
  in no BM25 run's top 10. 7 of the 15 are low-confidence grade-1 labels.
- **What changed in the metrics.** Every run's top 10 is now fully judged
  (judged@10 = 1.000 for all six). The dense runs' nDCG@10 rose by 0.04–0.05
  and their MRR@10 by 0.05–0.13. BM25's nDCG@10 fell by about 0.01, because
  the new positives raise the ideal ranking for questions where BM25 does
  not find them. The order on nDCG@10 and MRR@10 is the same as with
  `pilot_v2`: E5 > BGE > BM25, in both modes.
- **What cannot be concluded.** Not that one retriever is better. The
  labels are AI-generated and unaudited, there are 10 questions, and ranks
  11–100 are still 64–75% unjudged. If the 7 low-confidence positives are
  wrong, the Recall@100 ranking flips back towards BM25, and BGE and E5 tie
  on global MRR@10 (§6.3).

## 2. What `pilot_v3` is

| Step | Command | Output | Pinned inputs |
|---|---|---|---|
| Pool | `expand-pool --spec configs/eval/dev_pool_pilot_v3.yaml` | `data/eval/dev_ai/pilot_v3/pool/` | `pilot_v2` labels (SHA-256 of `labels.meta.json`), `data/eval/annotation/pool.jsonl`, each of the six `run.jsonl` and `metrics.json` files |
| AI proposals | library API (`create_pool_proposals`) | `data/eval/dev_ai/pilot_v3/proposals/` | the pool (corpus and annotation version) |
| Labels | `dev-labels --spec configs/eval/dev_labels_pilot_v3.yaml` | `data/eval/dev_ai/pilot_v3/{labels,changes,flags,questions}.jsonl`, `labels.meta.json` | `pilot_v2` labels, pool and proposals |
| Export | `export-dev --labels data/eval/dev_ai/pilot_v3` | `dev_queries.jsonl`, `dev_qrels.jsonl`, provenance, metadata, report | the label set |
| Benchmark | `04_run_baselines.py --eval-set dev_pilot_v3` | `results/dev/pilot_v3/` | the six stored runs in `results/dev/pilot_v2/` (SHA-256 checked) |

**How pooling was kept apart from judging.**

- A pool entry records how it entered the pool: the original candidate
  sources, plus every stored run that ranked it, with run, mode, rank and
  score.
- It also records `judged_in`, the label version that judged it: `pilot_v2`
  or `null`. A `null` entry is unjudged. It is absent from the qrels, never
  grade 0, until a label set grades it.
- Proposals can be made only for unjudged entries. A passage `pilot_v2`
  judged cannot receive a new proposal.
- Every passage ID was checked against the frozen corpus (`corpus_v1@1.2`,
  SHA-256 `ff9dadf3…`).

**Checks on the stored runs.**

- Each stored run's `run.jsonl` was checked against the SHA-256 in its
  `metrics.json`.
- Each run's judged@10 under the `pilot_v2` labels had to equal the value
  that run recorded. All six matched.

**Unchanged inputs.** The corpus, the annotation workspace, `pilot_v1`,
`pilot_v2`, the audit, the six runs and the embedding caches are
byte-identical to before this work.

**Separation from the gold set.** This pool is a development pool. The gold
annotation workspace still takes no dense-retriever candidates (§4 of the
methodology).

## 3. Pooling coverage

### Before expansion (`pilot_v2` labels)

| Run | judged@10 | Unjudged in top 10 (same / other company) | New pool entries contributed |
|---|---|---|---|
| BM25, global | 0.880 | 12 (0 / 12) | 12 |
| BM25, company | 1.000 | 0 | 0 |
| BGE, global | 0.490 | 51 (40 / 11) | 51 |
| BGE, company | 0.520 | 48 (48 / 0) | 48 |
| E5, global | 0.430 | 57 (44 / 13) | 57 |
| E5, company | 0.480 | 52 (52 / 0) | 52 |

- **Passages to assess.** The six top 10s hold 220 unjudged (question,
  passage) slots. Removing duplicates leaves **117 unique unjudged passages**,
  all of them assessed, so none is left unjudged. Across the six runs the
  top 10s contain 262 unique pairs, of which 145 were already judged.
- **New candidates by system.** Counts overlap when more than one system
  proposed a passage:
  - E5: 65
  - BGE: 59
  - BM25: 12, all from global mode and all from companies the question does
    not ask about
- **New candidates by exclusive source.**
  - E5 only: 50
  - BGE only: 44
  - BGE and E5: 11
  - BM25 only: 6
  - BM25 and E5: 2
  - BM25 and BGE: 2
  - all three: 2

### Overlap between BM25, BGE and E5

Over all 262 top-10 pairs, judged or not:

| Proposed by | Pairs |
|---|---|
| BM25 only | 74 |
| BGE only | 64 |
| E5 only | 66 |
| BGE and E5 only | 20 |
| BM25 and BGE only | 11 |
| BM25 and E5 only | 11 |
| All three | 16 |

- **BM25 and the dense models barely overlap.** BM25's top 10s (112 pairs
  over both modes) share 38 pairs with either dense model.
- **The two dense models also overlap little.** BGE (111 pairs) and E5 (113)
  share 36.

### Per-question pool sizes

| Question | `pilot_v2` pool | New (same / other company) | `pilot_v3` pool | Unique pairs in the six top 10s |
|---|---|---|---|---|
| q_0001 | 64 | 15 (8 / 7) | 79 | 34 |
| q_0002 | 53 | 2 (2 / 0) | 55 | 22 |
| q_0003 | 59 | 7 (7 / 0) | 66 | 24 |
| q_0004 | 54 | 10 (10 / 0) | 64 | 23 |
| q_0005 | 64 | 11 (11 / 0) | 75 | 25 |
| q_0006 | 59 | 34 (17 / 17) | 93 | 45 |
| q_0007 | 58 | 1 (0 / 1) | 59 | 18 |
| q_0008 | 60 | 16 (16 / 0) | 76 | 26 |
| q_0009 | 56 | 11 (11 / 0) | 67 | 24 |
| q_0010 | 52 | 10 (10 / 0) | 62 | 21 |
| **Total** | **579** | **117 (92 / 25)** | **696** | **262** |

## 4. How the new passages were labelled

Method `llm-full-passage-read-v2`; the full description is in
`proposals/proposals.meta.json`.

- **Reading.** Claude (`claude-opus-5-5`, in a Claude Code session) read the
  full text of all 117 passages, one question at a time.
- **What the model saw.** The question, its companies and years, and each
  passage's company, form, filing date, report date and section. Passages
  were listed by passage ID.
- **What the model did not see.** Runs, ranks, scores, and which retriever
  proposed each passage. The model did know that the batch came from the
  retrieval runs.
- **Prior labels shown.** For consistency, the `pilot_v2` positive labels
  and their explanations were reviewed before grading each question. For
  `q_0008`, the `pilot_v1` grades were also listed. This makes the new
  labels consistent with the old ones, but also correlated with them.
- **Rubric.** v1.1, as for `pilot_v2`:
  - the company and filing year must match the question;
  - another filing year's copy of a qualitative disclosure is 0 (P4);
  - a figure needs its period or basis in the passage (P2);
  - later filings' comparative columns are graded on content (P3);
  - part of an answer normally reported in one table is 1 (P5).

  Explanations cite only the passage.
- **What each proposal records.** Grade, confidence (`low`, `medium` or
  `high`), a one-sentence explanation, the model, the method and a
  timestamp.
- **Separation from human data.** The proposals are stored apart from the
  annotation workspace. No human judgment was created or changed, and no
  reviewer identity was used.
- **Rules applied by `dev-labels`.**
  - The P2 pattern check ran on new grade-2 labels. It changed nothing: the
    only new grade 2 states "January 31, 2024".
  - P1 flags were added to new grade-1 labels on open questions.
  - Low-confidence flags were added.
  - Two curated flags were added in the spec.
  - Flags never change a grade.

## 5. Quality review

### 5.1 Grade distribution of the new labels

| | Grade 0 | Grade 1 | Grade 2 | Total |
|---|---|---|---|---|
| Same company as the question | 77 | 14 | 1 | 92 |
| Other company | 25 | 0 | 0 | 25 |
| **All new** | **102** | **14** | **1** | **117** |
| `pilot_v3` overall | 596 | 65 | 35 | 696 |

**Confidence of the new labels.** 83 high, 26 medium, 8 low.

**Why some topical passages are 0.**

- **Repeated text.** 11 same-company passages repeat a qualitative
  disclosure from another filing year:
  - Apple's outsourcing and vendor-concentration text from FY2019, FY2020,
    FY2022, FY2023 and FY2025;
  - ExxonMobil's OPEC and supply text from FY2019 and FY2022–FY2025.

  All 11 are 0 under P4, 9 of them with medium confidence.
- **Wrong topic.** The JPMorgan question (`q_0004`) received 10 dense
  results about long-term debt and capital. None mentions the liquidity
  coverage ratio.

### 5.2 Low-confidence and flagged labels (for human review)

12 new flags are open.

**Low confidence (8).**

| Question | Passage | Grade | What is uncertain |
|---|---|---|---|
| q_0003 | `amzn…d3a00c1d8a74_000012` | 1 | FY2023 technology and infrastructure cost increase. It explains AWS cost pressure only at company level (also flagged P1). |
| q_0003 | `amzn…d3a00c1d8a74_000013` | 0 | The same cost sentence plus sales and marketing. Its AWS link is weaker. |
| q_0006 | `tsla…615485ed6438_000020` | 1 | FY2022 report: Tesla entered a $5.0B revolving credit facility in January 2023. Right facility, but nothing on year-end 2023. |
| q_0009 | five ExxonMobil passages | 1 each | 2022 additions to PP&E as a single total ($18.4B cash or $18,338M), with no segment or U.S./non-U.S. split. This follows the `pilot_v2` convention for total-only figures (1, low). |

**Judgment flags (4).**

- **P1 (2).** `q_0001`: the FY2024 risk factor on vendor non-trade
  receivables concentrated among a few vendors in Asia. Also `q_0003` above.
- **P5 (1).** `q_0005`: the FY2023 product table with Comirnaty in 2023
  against 2022 (−70%, with the demand drivers), but no Paxlovid. Whether one
  product's complete figures are a complete evidence unit (2) needs a
  reader.
- **P3 (1).** `q_0008`: the only new grade 2. It is the FY2026 report's
  disaggregated revenue tables, whose fiscal 2024 columns give all three
  segments. There, the third segment is named Sam's Club U.S.

### 5.3 Potentially relevant passages found only by dense retrieval

All 15 new positives come from dense top 10s.

- **Which model found them.** BGE only: 6. E5 only: 7. Both: 2.
- **Where BM25 ranks them.** None is in a BM25 run's top 10. In
  company-mode BM25, which is close to how the pools were built, 8 rank
  31–97 and 7 are not in the top 100.

| Question | Passage | Grade (confidence) | Found by (rank) | BM25 company rank |
|---|---|---|---|---|
| q_0001 | FY2024 1A vendor receivables risk (`…907129e41fcc_000021`) | 1 (medium) | BGE company (10) | 64 |
| q_0003 | FY2023 technology and infrastructure costs (`…d3a00c1d8a74_000012`) | 1 (low) | E5 both modes (10) | >100 |
| q_0005 | FY2023 Comirnaty 2023 vs 2022 (`…cde2c92bc62c_000016`) | 1 (medium) | BGE (7), E5 (2) | >100 |
| q_0005 | FY2025 note: $3.5B Paxlovid reversal, 2023 revenues (`…d0cb001f40ca_000106`) | 1 (medium) | E5 (8) | 97 |
| q_0006 | FY2022: $5.0B RCF entered January 2023 (`…615485ed6438_000020`) | 1 (low) | BGE company (4) | 44 |
| q_0008 | FY2026 disaggregated revenues, all three segments for fiscal 2024 (`…40ceee83bdf8_000051`) | **2** (medium) | BGE (8) | 59 |
| q_0008 | FY2024 MD&A International table, fiscal 2024 $114,641M (`…8736c816bde6_000013`) | 1 (high) | BGE (1), E5 (2) | 31 |
| q_0008 | FY2024 MD&A Sam's Club table, fiscal 2024 $86,179M (`…8736c816bde6_000014`) | 1 (high) | E5 (3) | 70 |
| q_0008 | FY2025 MD&A Sam's Club U.S., fiscal 2024 comparative (`…6fa57f3e89fa_000014`) | 1 (medium) | BGE (5) | 88 |
| q_0008 | FY2026 Walmart U.S. table, fiscal 2024 comparative (`…40ceee83bdf8_000011`) | 1 (medium) | E5 (6) | 35 |
| q_0009 | five 2022 PP&E totals (FY2022, FY2023, FY2024 filings) | 1 (low) each | BGE (3, 9), E5 (1, 3, 4) | >100 |

### 5.4 Questions whose original pools missed relevant evidence

Six of the ten questions gain positives; none was without a positive
before.

- **`q_0008` (Walmart segment sales).** The clearest miss. The FY2024
  report's own MD&A segment tables for International and Sam's Club were
  never pooled (BM25 ranks 31 and 70, against a pool depth of 30). Neither
  was a later filing's complete three-segment table. Known positives go
  from 7 to 12, and the question gains its second grade-2 passage.
- **`q_0009` (ExxonMobil 2022 capex).** Five total-only figures. These are
  weak evidence (grade 1, low confidence), and none has a BM25 rank in the
  top 100: the passages say "additions to property, plant and equipment",
  not "capital spending".
- **`q_0005` (Pfizer COVID revenue).** Two partial passages. One is the
  FY2023 per-product table for the vaccine, a possible 2 under P5.
- **Weaker single additions.** `q_0001`, `q_0003` and `q_0006` gain one
  each, all grade 1, two of them low confidence.
- **No new positives.** `q_0002`, `q_0004`, `q_0007` and `q_0010`.
  For these, the dense top 10s added only negatives: other filing years,
  other companies or other topics.

### 5.5 judged@10 before and after the provisional labels

| Run | `pilot_v2` | `pilot_v3` |
|---|---|---|
| BM25 global / company | 0.880 / 1.000 | 1.000 / 1.000 |
| BGE global / company | 0.490 / 0.520 | 1.000 / 1.000 |
| E5 global / company | 0.430 / 0.480 | 1.000 / 1.000 |

- **Every question in every run is now at 1.000.** Before, the dense runs
  were below the 0.7 coverage threshold on 7–8 of the 10 questions.
- **Beyond rank 10, most passages are still unjudged.** In ranks 11–100,
  64% (BM25 company) to 75% (E5) of passages are unjudged.

### 5.6 Evidence of systematic BM25 pooling bias

**Yes, the evidence points that way, though its size is uncertain.**

- **Every new positive came from a dense model.** No passage that only a
  BM25 run surfaced was relevant: its 12 additions were all other-company
  passages. All 15 new positives came from dense top 10s, and 7 of them are
  not in BM25's top 100.
- **The pool depth of 30 was a binding cut-off.** 8 new positives sit at
  BM25 company ranks 31–97, so a deeper BM25 pool would have found them.
  The others use wording BM25 does not match, such as "additions to
  property, plant and equipment" for capital spending.
- **Dense scores under `pilot_v2` were lower bounds, as expected.** The
  dense runs' newly judged top-10 passages were relevant 12–17% of the time.
  Their already judged top-10 passages were relevant 33–47% of the time.
  The BM25-built pools therefore held most, not all, of the relevance the
  dense models reach.
- **Recall@100 was the most biased metric.** Under `pilot_v2`, BM25 company
  had the highest Recall@100 (0.665, against 0.637 for E5 company). Under
  `pilot_v3` they are level (0.674 against 0.673).
- **Size of the effect.** The bias added 15 positives to 85, a 17.6%
  increase, mostly grade 1. It did not change the order on nDCG@10 or
  MRR@10.

### 5.7 Earlier labels

No `pilot_v2` grade was changed or re-flagged; the derivation refuses to
re-flag carried labels. Reading the new passages raised no contradiction
with an earlier grade. Three observations could inform a human review of
`pilot_v2`:

- **`q_0008`.** The FY2024 MD&A segment tables now labelled 1 are as
  informative as the FY2025 and FY2026 comparatives that `pilot_v2` already
  labelled 1.
- **`q_0005`.** The new P5 flag concerns the same kind of evidence unit as
  the existing `pilot_v2` P5 flag on `q_0007`.
- **`q_0009`.** Both versions label total-only PP&E or Capex figures 1 with
  low confidence. If a reader decides these are 0, the decision should be
  applied to both versions together.

## 6. Benchmark: the same rankings, `pilot_v2` against `pilot_v3` labels

`results/dev/pilot_v3/` scores the six stored rankings from
`results/dev/pilot_v2/` without running any retriever, computing any
embedding or re-ranking anything.

- **Rankings are unchanged.** The rank, passage and score columns of every
  `run.jsonl` are identical. Each `metrics.json` records the source run file
  and its SHA-256.
- **Metric definitions are unchanged.** nDCG@10, MRR@10, Recall@100 and
  judged@10 are computed by the same function.

### 6.1 Results

| Run | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Relevant in top 10 |
|---|---|---|---|---|---|
| BM25 global | 0.181 → 0.170 | 0.323 → 0.323 | 0.606 → 0.578 | 0.880 → 1.000 | 18 → 18 |
| BM25 company | 0.206 → 0.194 | 0.423 → 0.423 | 0.665 → 0.674 | 1.000 → 1.000 | 21 → 21 |
| BGE global | 0.247 → 0.286 | 0.450 → 0.550 | 0.572 → 0.627 | 0.490 → 1.000 | 17 → 23 |
| BGE company | 0.247 → 0.297 | 0.450 → 0.575 | 0.592 → 0.643 | 0.520 → 1.000 | 17 → 25 |
| E5 global | 0.314 → 0.361 | 0.550 → 0.600 | 0.628 → 0.666 | 0.430 → 1.000 | 20 → 29 |
| E5 company | 0.330 → 0.376 | 0.575 → 0.625 | 0.637 → 0.673 | 0.480 → 1.000 | 21 → 30 |

Known positives across the 10 questions: 85 → 100. Per-question nDCG@10 in
both versions is in `results/dev/pilot_v3/report.md`.

### 6.2 Why the metrics moved

- **Dense nDCG@10 and MRR@10 rose.** Passages that were unjudged, and so
  counted as non-relevant, are now judged positives in the dense top 10s:
  6–9 per run. The largest moves are on `q_0008` and `q_0009`:
  - BGE `q_0008`: 0.000 → 0.299 nDCG@10. Its first known relevant passage
    moves from rank 27 to rank 1.
  - E5 `q_0009`: 0.434 → 0.681.
- **BM25's nDCG@10 fell, without any change to its rankings.** nDCG
  divides by the ideal ranking of the known positives. The new positives
  raise that ideal for `q_0006` and `q_0008`, where BM25 does not retrieve
  them in its top 10:
  - `q_0008`: 0.371 → 0.268;
  - `q_0006` in company mode: 0.223 → 0.211.

  The same effect lowers some dense per-question scores, for example BGE on
  `q_0003` (0.478 → 0.441).
- **BM25's MRR@10 is unchanged.** None of the new positives is in a BM25
  top 10.
- **Recall@100 moved both ways.** Its denominator grew from 85 to 100. BM25
  global's top 100 holds only 5 of the 15 new positives, so it fell (0.606
  → 0.578). The dense runs' top 100s hold 13–14 of them, so theirs rose.
  BM25 company's top 100 holds 8, so it rose slightly.
- **The judged-only scores overstated the dense runs.** These are the
  condensed-list sensitivity scores under `pilot_v2`. E5 global was 0.451
  judged-only against 0.361 fully judged, and BGE global 0.338 against
  0.286. The fully judged values fall between the `pilot_v2` standard score
  and the judged-only score. Read the condensed list as an upper bound, not
  an estimate.

### 6.3 How robust the changes are

The new positives are AI labels, mostly grade 1, and nearly half of them are
low confidence. As a sensitivity check (not saved as a label version), the 7
low-confidence new positives were set to 0:

| Run | nDCG@10 (`pilot_v3`) | nDCG@10, low-confidence new positives as 0 | Recall@100, same variant |
|---|---|---|---|
| BM25 global / company | 0.170 / 0.194 | 0.170 / 0.196 | 0.611 / 0.694 |
| BGE global / company | 0.286 / 0.297 | 0.280 / 0.284 | 0.603 / 0.619 |
| E5 global / company | 0.361 / 0.376 | 0.336 / 0.351 | 0.633 / 0.642 |

- **nDCG@10 holds its order.** E5 > BGE > BM25 in both modes, also with
  binary relevance.
- **MRR@10 nearly does.** Only in global mode do BGE and E5 tie, at 0.550.
- **Recall@100 does not.** BM25 company leads again (0.694). On these 10
  questions, the Recall@100 comparison depends on 7 low-confidence labels.

**What this does and does not show.** It shows that BM25-built pools
under-scored the dense runs on these development labels, and by roughly how
much. It does not show that any retriever is better. Adding AI judgments
does not establish ground-truth relevance. The labels come from one model,
which also drafted the questions and saw the earlier labels. This
expansion's labels have not been audited, and the `pilot_v1` audit was
AI-assisted.

## 7. Limitations

- **The labels are unverified.** All 117 new labels are AI proposals with no
  human or blind audit. The earlier audit covered `pilot_v1` proposals only
  and was AI-assisted.
- **Errors are likely correlated.** The same model wrote the questions,
  graded `pilot_v1`, and graded these passages after reading the `pilot_v2`
  positives.
- **Only the top 10 was deepened.** Ranks 11–100 are 64–75% unjudged, so
  Recall@100 and any deeper-rank metric still favour BM25-like rankings.
- **Ten questions.** One question (`q_0008`) or one label decision
  (`q_0009`'s totals) moves the averages noticeably. No bootstrap intervals
  are reported.
- **The dense models see only the first 512 tokens.** About 91% of passages
  are truncated for BGE and E5, so these runs do not reflect what a model
  reading full passages would retrieve.
- **The pool now favours these six systems.** It is deepened with exactly
  the systems being compared, so their top 10s are fully judged, while
  another system (for example a fine-tuned model) would face unjudged
  passages again. This is why the gold pool rules still exclude
  model-proposed candidates.
- **Config provenance.** `configs/eval/benchmark.yaml` gained the
  `dev_pilot_v3` entry after the `pilot_v2` runs. Their recorded
  `benchmark_config_sha256` (`7f0f2bce…`) is for the file before that
  addition; nothing else in the file changed.

## 8. What a person should review next

1. **The 12 new flags (`flags.jsonl`, `flg_013`–`flg_024`).** Above all, the
   15 new positives and the `q_0009` total-only convention.
2. **A blind audit of the `pilot_v3` proposals.** For example, all 15
   positives plus a seeded sample of the 102 zeros, graded without AI
   assistance, before these labels inform any decision.
3. **Pool deepening below rank 10.** Deepen the dense runs to ranks 11–30
   before comparing Recall@100.
