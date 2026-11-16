# Training-data preparation v2 (`bge_base_ft_v2`): evidence-level leakage control

Status: 2026-10-30.

- **v2 extends v1 with evidence-level leakage control.** v1 (`bge_base_ft_v1`,
  [training_data_v1.md](training_data_v1.md)) blocked evaluation passages and
  their whole-passage copies. v2 also blocks passages that substantially reuse
  their sentences, spans or figures. It also keeps evidence from crossing the
  train/validation split, and records which passages could answer each
  training question.
- **v1 is unchanged.** Its preparation, pilot records and pairs stay as they
  are. The new code reproduces v1's data files byte for byte from its config.
  Only the manifest's config hash differs, because `data_prep_v1.yaml` was
  edited after v1 was prepared.
- **No pair is rebound.** The pilot pairs are re-checked explicitly
  (`carry-over`, §6), not attached to the v2 split.
- **Not done here:** no API request was made and no model was trained. The
  frozen corpus, evaluation questions, labels and retrieval runs were only
  read.
- **Added since (§10–13):** prompt `sec-qgen-v2`, balanced sampling plans,
  the training-pair contract with a false-negative-safe trainer interface,
  and the second pilot plan `pilot_v3` (50 requests, not yet run).
- **§14:** the validation pilot `val_pilot_v1` (50 requests, run on
  2026-11-03), the `balanced_years` method, write-once pairs and the
  validation set `validation_pairs_v1` (16 questions; not enough to choose
  checkpoints).
- **Successor:** `bge_base_ft_v3` keeps this preparation's data pinned in
  place and enlarges the validation split
  ([training_data_v3.md](training_data_v3.md)). Nothing here changed.

```bash
python3 scripts/02_generate_queries.py prepare --config configs/train/data_prep_v2.yaml     # offline, ~50 s
python3 scripts/02_generate_queries.py carry-over --config configs/train/data_prep_v2.yaml \
    --from-config configs/train/data_prep_v1.yaml \
    --decisions data/train_review/pilot_v2_quality_decisions.jsonl --name pilot_v2
```

## 1. The gap v2 closes

The pilot audit ([training_pilot_v2_quality_report.md](training_pilot_v2_quality_report.md))
found 6 pairs whose evidence also sat in evaluation-pool passages, and 4 that
shared evidence with validation passages.

- **Why the blocklist missed them.** It worked on whole passages. Overlap
  groups need 50% of 5-word shingles in common, and shingles found in more than
  50 passages are skipped.
- **What slips through.** 10-K evidence is reused in smaller pieces:
  - consecutive chunks of a section share 80 words by construction;
  - a risk factor or policy paragraph reappears in the next filing among new
    text, often with a word or a year changed;
  - a comparative table repeats last year's columns.

## 2. Method (`finretriever.training.evidence_reuse`)

Everything is computed from corpus text. No evaluation grade, judgment or
question is read; the evaluation pools contribute passage IDs only, as in v1.

**Units.**

| Unit | Definition |
|---|---|
| Span | 12 consecutive case-folded `[a-z0-9]+` words, 64-bit BLAKE2b hash. Catches sentences edited by a word or two, and lists without sentences. |
| Sentence | A prose sentence (tables removed) of at least 12 words. Used for exact evidence matches. |
| Fact | A significant number with its company and label. "Significant" means at least 4 digits, or at least 2 followed by thousand, million, billion or trillion; years and percentages never count. The label is the nearest earlier word that is not a number, a period word or a stop word: "Total operating expenses \| Year Ended December 31, 2021: 444,943" gives (AMZN, 444943, *expenses*) in the FY2021 MD&A and the FY2023 income statement alike. |

**Generic units are ignored.** A span or sentence used by more than one
company, or any unit held by more than 50 passages, cannot relate two
passages. This covers auditor's-report wording, standard definitions and legal
boilerplate. Terminology alone never forms a unit.

**Substantial reuse.** A passage's visible text (what BGE reads and what
queries are written from) substantially reuses a set of passages when either
holds:
- the spans it shares with them cover at least 25% of its visible words;
- it shares at least 3 facts with them, and those are at least half of its
  facts (for example, a comparative table's prior-year columns).

Spans that only an adjacent chunk of the same section shares do not count.
That 80-word overlap is about a fifth of the visible text, and the per-pair
check below catches any question drawn from it.

| Setting (`configs/train/data_prep_v2.yaml`, `reuse:`) | Value | Why |
|---|---|---|
| `span_words` | 12 | Long enough that matches are deliberate reuse, short enough to survive one-word edits |
| `min_sentence_words` | 12 | Terminology and short phrases never form a unit |
| `max_text_companies` | 1 | Language two companies share is generic |
| `max_unit_passages` | 50 | As v1's `max_shingle_df` |
| `min_fact_digits` / `min_scaled_fact_digits` | 4 / 2 | Figures specific enough to identify a fact |
| `label_window_tokens` | 40 | Reaches a row's label past its other cells |
| `substantial_word_share` | 0.25 | Above the structural chunk overlap |
| `substantial_facts` / `substantial_fact_share` | 3 / 0.5 | Most of a table's figures, not three totals quoted in a 44-figure note |

**How it is applied.**
- **Blocklist (one step).** Every passage whose visible text substantially
  reuses the full text of an evaluation passage, or of a member of its overlap
  group, is blocked with reason `evidence_reuse`, along with its whole overlap
  group. The step starts from the evaluation passages only, never from
  passages it added, so it cannot cascade.
- **Split units.** Remaining passages that substantially reuse each other
  (visible text on both sides) are linked. Linked overlap groups are assigned
  to train or validation as one unit, stratified by company and year as
  before. Without links the split reproduces v1.
- **Per pair** (`build-pairs` and `carry-over`). `EvidenceLocator` finds
  every passage of the same company holding a pair's evidence.
  - **Exact matches:** the quote itself (located as validator v3 does), a
    whole evidence sentence, or the answer's figure under the evidence's
    label.
  - **Uncertain matches:** another figure of the evidence, the answer's figure
    under another label, or 8 consecutive evidence words.
  - **Rejection:** a pair is rejected as `evaluation_evidence` if a blocked
    passage holds it exactly, and as `cross_split_evidence` if the visible
    text of an other-split passage does.
- **Fail closed.**
  - A format-1 preparation builds no pairs, and `check` reports the leakage
    checks as unavailable.
  - Reloading a preparation rebuilds the index and compares its SHA-256 with
    the one recorded at preparation time; a mismatch stops every check.
  - `prepare` refuses to write if any remaining passage substantially reuses
    evaluation evidence, or if a link crosses the split.

## 3. Results on the frozen corpus

Index: 2,064,631 spans (838,275 shared, 3,442 generic), 53,311 sentence units
and 62,433 fact units. Digest `d3321b40…5977`; manifest
`c9b8998e…7e15`.

| | v1 | v2 |
|---|---|---|
| Blocked passages | 2,110 | **3,950** |
| … evaluation candidates / their overlap groups | 667 / 1,443 | 667 / 1,443 |
| … evidence reuse (spans 932, facts 59, both 170) / their overlap groups | – | 1,161 / 679 |
| Train passages (overlap groups) | 7,200 (3,253) | **5,554** (2,627) |
| Validation passages (overlap groups) | 827 (419) | **633** (358) |
| Generation sources, ≥ 40 visible words (train / validation) | 7,185 / 827 | 5,539 / 633 |
| … kept by the source-quality filter | 6,822 / 819 | 5,254 / 617 |
| Split units (links) | groups | 1,329 (6,113 links; 362 units join groups; largest 413 passages, JPMorgan, all 7 years) |

**Verification** (recorded in the manifest):
- 0 remaining passages substantially reuse evaluation evidence, and 0 links
  cross the split.
- Below the thresholds, 1,807 usable passages still share some unit with
  evaluation passages (largest share 0.2493), and 1,463 train passages share
  some unit with validation passages. Both are enforced per pair.
- `prepare` run twice gives byte-identical files.

**Effect on the distribution** (train passages; validation share of usable
passages in brackets):

| Company | v1 | v2 | Change |
|---|---|---|---|
| AAPL | 302 | 164 | −46% (10.7% → 10.4%) |
| AMZN | 410 | 228 | −44% (11.8% → 10.6%) |
| JPM | 2,975 | 2,784 | −6% (10.1% → 10.3%) |
| PFE | 1,092 | 793 | −27% (10.3% → 10.4%) |
| TSLA | 713 | 386 | −46% (10.1% → 11.9%) |
| WMT | 754 | 521 | −31% (10.3% → 8.4%) |
| XOM | 954 | 678 | −29% (10.2% → 10.0%) |

- **By company.** Companies whose evaluation pools are large relative to
  their filings lose most. JPMorgan is now half the training passages (50%,
  up from 41%).
- **By year.** Every fiscal year loses 17–25%. Validation shares stay between
  9.0% and 11.2% per year and between 8.4% and 11.9% per company.

## 4. False positives and false negatives

**Possible false positives (over-blocking).**
- 123 of the 1,102 span-blocked passages would fall below the 25% share if
  spans held by 15 or more passages were discounted. Those spans are one
  company's recurring boilerplate:
  - JPMorgan's "Prior period amounts have been revised to conform with the
    current presentation" (38 passages);
  - Walmart's "Any repurchased shares are constructively retired and
    returned to an unissued status" (25).
- Examples:
  - an Amazon FY2020 note blocked at 0.33, which would be 0.04 without its
    payment-processing policy text;
  - a Walmart FY2022 store-count table, 0.31 → 0.14.
- The 679 passages blocked only as overlap-group members of a reusing
  passage do not themselves reach the threshold.
- Each is still the same company's text, so these errors cost data, not
  correctness.

**False negatives (missed reuse).**
- With sentence units alone, 287 usable passages had at least 30% of their
  visible 8-word spans in evaluation passages. The worst were Apple exhibit
  indexes at 97%, with no full sentences and no four-digit figures, and prose
  edited by a word. Spans cut this to 59.
- Of those 59:
  - 29 are high only because of the 80-word overlap with an adjacent
    evaluation chunk, which the per-pair check handles. Several Walmart FY2024–2026
    statement passages fall here, with span shares of 0.16–0.23.
  - The other 30 match auditor's-report wording that Walmart's and Apple's
    auditors share ("such procedures included examining on a test basis
    evidence regarding the amounts and disclosures in the financial
    statements"). This is ignored as generic, correctly.
- **Still missed:**
  - figures written differently ("$14.6 billion" in prose, "14,600" in a
    table) by the index's links and blocklist. The per-pair check compares
    figures by amount since `evidence-reuse-v2` (docs/training_trainer_v1.md);
  - rows whose label changes between filings;
  - paraphrased sentences;
  - reuse below the thresholds that no pair's evidence touches.

## 5. Validator `sec-qcheck-v3`

Validator v3 (`finretriever.training.quotes`, `query_generation`) fixes the
false rejections the audit found and keeps the strict checks. The original
generation records are not rewritten.

| v2 false rejection | v3 |
|---|---|
| A quote skipping cells of its row ("Total operating expenses \| … 2023: 537,933") | Accepted: cells found in order, at most 6 skipped, no row or table boundary between |
| Fragments joined by "..." | Accepted: each fragment verbatim and in order, gap ≤ 600 characters, next fragment starts with a word |
| A row label restating the parent the flattened table shortened | Accepted only if the nearest earlier full parent in the same table is the quoted one |
| Copying "allowance for credit losses" or a table title | Not counted: table titles and labels, and 41 standard line items and accounting terms (`STANDARD_TERMS`), are exempt; prose copying still counts |

**Still rejected:**
- invented cells or labels, a value after an ellipsis, a value from another
  row, an edited quote;
- invented or converted numbers, unsupported dates;
- the wrong company;
- evidence outside the BGE-visible text.

**On the pilot's stored candidates:** v3 accepts all 37 earlier pairs and 9 of
the 11 `pilot_v2` rejections. The 2 left are an invented quote and a quote
with a changed phrase. The 9 recovered candidates have not been reviewed and
are not carried over.

## 6. Pilot pairs: quality decisions and carry-over

The audit's decisions are recorded explicitly in
[`data/train_review/pilot_v2_quality_decisions.jsonl`](../data/train_review/pilot_v2_quality_decisions.jsonl):
- 33 accept, 3 revise, 1 reject;
- issues: 9 `boilerplate`, 2 `weak_financial_relevance`, 1
  `misinterpreted_table_value`, 1 `incorrect_period`, 1 `inaccurate_wording`.

Each decision is bound to its pair by passage and question SHA-256. They are
an AI review, not gold. No question was edited.

`carry-over` wrote `data/processed/train/bge_base_ft_v2/carryover/pilot_v2.jsonl`
with its meta file, write-once.

**15 of the 37 pairs are eligible as development training candidates**, all
from the 33 accepted:

| Reason (a pair can have several) | Pairs |
|---|---|
| Evidence held exactly by an evaluation-pool passage (`evaluation_evidence`) | 16 |
| Positive now blocked by evidence reuse (`positive_now_blocked`) | 15 |
| Quality decision revise / reject | 3 / 1 |

- **Eligible:** 004, 005, 014–020, 023, 024, 029, 030, 031, 036. JPMorgan
  supplies 7 of them, and Amazon none.
- **Repeated evidence:** 9 eligible pairs have evidence repeated across
  fiscal years. They carry 19 exact and 20 uncertain equivalent passages to
  keep out of their negatives.
- **Newly caught by v2:** 003 (Apple's FY2020 net income, held by 5
  evaluation passages) and 013 (the AWS server useful-life sentence, verbatim
  in an evaluation-pool passage for `q_0003`). Neither was flagged in the
  audit.
- **Recovered:** 030 and 031 had a cross-split duplicate under v1. Their
  Walmart MD&A lineage is now on one side.

## 7. False-negative protection for training (`finretriever.training.false_negatives`)

- **What each pair records.** v2 pairs store `equivalents.exact` and
  `equivalents.uncertain`: same-split passages whose visible text holds the
  evidence.
- **Exclusions.** `negative_exclusions` returns these together with the
  positive's overlap group; uncertain ones are included, because excluding one
  too many only costs a negative. `filter_negatives` applies them to mined
  candidates.
- **Batching.** `conflict_free_batches` places pairs deterministically so that
  no batch holds a pair beside a passage that answers it.
- The trainer uses these functions (docs/training_trainer_v1.md); the miner is still a stub.

## 8. Remaining risks

- **False negatives remain possible:**
  - evidence restated in other words or units;
  - uncertain matches beyond the 8-word span rule;
  - answers that sit beyond another passage's visible window but which a
    reader of the full text would still accept.
- **Semantic leakage** against evaluation questions is still checked
  lexically; the span and fact checks cover shared evidence text, not
  paraphrased questions.
- **Over-blocking** of company boilerplate (§4) costs data, especially for
  Apple, Amazon and Tesla.
- **JPMorgan imbalance.** JPMorgan holds 49% of the kept training groups.
  Plans should cap it.

## 9. Generation capacity

At Groq's 1,000 output tokens per minute the paced resume ran at about 2.1
requests per minute (768 tokens reserved). In the pilot:
- the generator accepted about 1.5 candidates per request, about 1.8 with
  validator v3;
- 4 of the 19 accepted pairs whose positive stayed usable failed the
  per-pair evidence check.

That gives about 1.2–1.4 usable pairs per training request.

| Scope | Requests | Time at ~2.1/min | Expected pairs |
|---|---|---|---|
| One passage per overlap group: 2,595 train + 354 validation | 2,949 | ~23 h | ~3,100–3,600 train, ~240 validation |
| Every kept passage: 5,254 train + 617 validation | 5,871 | ~47 h | ~6,300–7,400 train, ~420 validation |

Daily request and token limits are unknown. Read the `x-ratelimit-*` headers
that `generate` prints before a multi-day run. The second pilot recommended
in the audit (about 200 train and 100 validation groups) fits in about
2.5 hours.

## 10. Prompt `sec-qgen-v2`

`configs/train/data_prep_v2.yaml` sets `generation.prompt_version:
sec-qgen-v2`. Compared with `sec-qgen-v1`, the new prompt:

- **Names periods naturally.** It gives the model "fiscal 2024 (period ended
  September 28, 2024), filed November 1, 2024" instead of ISO dates. It asks
  for "fiscal 2024" or "FY2024", or a period the passage states, and a full
  date only when the question depends on it, never as YYYY-MM-DD. A figure for
  an earlier year shown in the passage must name that year, the mistake
  behind audit pair 028.
- **Asks about substance:** reported figures, tables, revenue, profitability,
  liquidity and debt, risks, accounting policies. It rules out exhibit or note
  numbers, form or item names, filing dates and where documents are filed.
- **Forbids calculation.** It allows only stated figures, each kept with its
  unit and period: "Do not add, subtract, divide, convert or otherwise
  calculate new figures."
- **Keeps quotes intact.** It allows at most five copied words, though line
  items and accounting terms may be used as written. It asks for one exact
  quote; for a table row, the label and every cell up to the answer, never
  shortened with "...".
- **Offers more question types:** `profitability`, `accounting_policy` and
  `tax`. Validator v3 accepts every type either prompt offers.
- **Has no example questions,** so no evaluation text can reach a reply
  through the prompt. A test checks every template against the evaluation
  questions.

**Versioning.**
- **Registry and default.** Prompts live in a registry (`PROMPTS`). A config
  without `prompt_version` keeps `sec-qgen-v1`, whose text and SHA-256
  (`81626417…`, pinned by a test) are unchanged.
- **Run identity.** A run records the prompt version and SHA-256 in
  `run.json`; `sec-qgen-v2` is `f42bdc54…`. The version is not part of the
  parameter hash, so the v1 run and its continuation stay bound and its 25
  records stay valid under `sec-qgen-v1`.
- **A new prompt is a new run.** Changing a run's prompt is refused, never
  continued.
- **Same preparation.** Editing only generation settings leaves a preparation
  as it is: `prepare` now reports "unchanged … the config file changed only in
  sections the preparation does not use" instead of refusing. Any change to
  data or a recorded setting is still refused.

## 11. Balanced sampling plans (`plan --method balanced`)

- **Company quotas.** Quotas are as equal as the sources allow, and no company
  takes more than `--max-company-share` (default 0.25). A plan that cannot fit
  under the cap fails rather than stretching it.
- **Distinct sources.** At most one passage per overlap group and per split
  unit, so no two planned passages substantially share evidence.
  `--exclude-carryover NAME` also skips the split units of a carry-over's
  eligible pairs.
- **Diversity.** Within a company, each pick takes the passage kind (table or
  narrative) and fiscal year it has used least. Across the plan it weights
  section families: statements and MD&A 3, risk 2, business and other 1.
  Picks alternate between companies, so a run stopped early is still
  balanced.
- **Eligibility.** Only approved, non-blocked passages of the split that the
  source-quality filter keeps.
- **Reproducibility.** The seed is the split seed; plans are written once.
  Question types are only known after generation; passage kind and section
  family are what a plan can balance beforehand.
- **`balanced_years` (§14).** The same constraints with fiscal years first.
  `balanced` is unchanged, so `pilot_v3` is reproduced as written.

## 12. Training-pair contract and the trainer interface

**Contract (`finretriever.training.pair_contract`, version 1).**
`build-pairs` now writes every pair as a contract record:
- the query, answer, evidence and question type;
- the positive and its source metadata: company, fiscal year, filing and
  report dates, section, item, visible length;
- its overlap group and split unit;
- `equivalents`: exact and uncertain same-split passages, the method
  (`evidence-reuse-v1`; `evidence-reuse-v2` since the figure fix, which makes
  v1 pairs stale) and the index digest;
- `provenance`: corpus, preparation version and manifest SHA-256, provider,
  model, parameters, prompt version and SHA-256, validator version, and
  origin.

`check` validates every pair against its preparation. A positive outside the
pair's split is a violation, so validation pairs come only from the
validation split.

**Loading (`finretriever.training.training_data.load_training_pairs`).**
- **Sources.** Built pairs and, on request, eligible carry-over rows, converted
  to the same contract.
- **Fail closed.** Nothing is returned if any pair:
  - lacks equivalence metadata or has metadata from another preparation or
    index;
  - was checked by an older validator;
  - fails the candidate checks;
  - holds evidence a blocked or other-split passage holds;
  - has equivalents that differ from what the evidence locator finds now;
  - overlaps an evaluation question, or repeats another pair's query.
- **On the real data,** the 15 carried-over pilot pairs load through this
  path.

**Protection.**
- **Multi-positive.** `NegativeGuard.positives` is the positive with its exact
  equivalents; each holds the evidence, so a trainer may use them as extra
  positives.
- **Exclusions.** `NegativeGuard.exclusions` adds the overlap groups of those
  passages, the uncertain equivalents, and passages directly linked to the
  positive by substantial evidence reuse. Blocked and other-split passages are
  never negatives.
- **Batches.** `plan_batches` places pairs deterministically from a seed so
  that no batch holds a pair beside a passage excluded for it, then verifies
  every batch pair by pair.
- **Hard negatives.** `hard_negative_candidates` filters mined candidates the
  same way.
- **On the 15 carried-over pairs,** 6 pair combinations conflict, mostly two
  questions on one passage. Batches of 4 come out as 4, 4, 4 and 3.

**What remains.** Equivalence is found from shared text and figures, so these
can still be sampled as negatives:
- evidence restated in other words;
- a figure in other units ("$14.6 billion" against "14,600"), until
  `evidence-reuse-v2` compared figures by amount; a rounded figure is still
  missed;
- a table row whose label changed between filings;
- an answer only a reader could assemble from several passages.

Uncertain equivalents are excluded but not used as positives, so a passage
that truly answers a question can be ignored rather than learnt from. The
trainer is built on this interface (docs/training_trainer_v1.md); the
hard-negative miner is still a stub.

## 13. Second pilot (`pilot_v3`, not yet run)

`data/processed/train/bge_base_ft_v2/plans/train.pilot_v3.json` (SHA-256
`55b33c98…c407`): 50 train passages, `balanced`, seed 20261029.

| | |
|---|---|
| Companies | AAPL 7, AMZN 7, JPM 7 (14%), PFE 8, TSLA 7, WMT 7, XOM 7 |
| Fiscal years | 2019 6, 2020 7, 2021 7, 2022 7, 2023 7, 2024 8, 2025 7, 2026 1; every company covers 7 years |
| Kinds | 24 tables, 26 narrative; every company has both |
| Section families | statements 15, MD&A 15, risk 10, business 5, other 5 |
| Constraints | 50 distinct overlap groups and split units; the 71 groups in the units of the 15 carried-over pairs excluded |
| Eligible pool | 5,254 of 5,539 approved train passages (source filter) |

The dry run sends nothing:

```bash
python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v2.yaml --split train --plan pilot_v3 --max-requests 50
```

The real run needs `GROQ_API_KEY` in the environment:

```bash
python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v2.yaml --split train --plan pilot_v3 --max-requests 50 --confirm-api-use
```

- **The cap includes retries.** `--max-requests 50` counts every HTTP attempt.
  A malformed reply or an HTTP 429 uses budget, so a run that needs retries
  stops cleanly with passages left. Those resume later, with fresh approval
  for more requests.
- **Duration.** At the 1,000 output tokens per minute limit the run takes about
  25–50 minutes.
- **Next steps.** Run `build-pairs` and `check` with the same config, then
  review the pairs as in the first audit.


## 14. Validation pilot (`val_pilot_v1`, run 2026-11-03)

**Purpose.** A synthetic validation set for choosing checkpoints, generated
from the validation split only. Its questions are AI-generated and
AI-checked: not human-validated and not gold labels.

**Outcome.**
- **The run:** 50 requests, 50 records, 0 failures, 0 rate limits.
- **The set:** `validation_pairs_v1` holds 16 questions, out of 43 that
  passed the generator.
  - Under the validation set's evidence rule, no passage outside validation
    may hold a question's evidence, even possibly.
  - The audit and the reasons are in
    [validation_pilot_v1_quality_report.md](validation_pilot_v1_quality_report.md).
- **Too small to choose checkpoints,** and the split cannot supply many more
  (§8 there).
- The text below is the plan as it was approved before the run.

**How validation passages are processed** (the code as it stands; nothing
new):

- **Sources.** Approved validation-split passages with at least 40 visible
  words: 633. Blocked passages and their groups are never sources. The
  source-quality filter keeps 617, dropping 1 exhibit index, 6 cross
  references, 7 repeated administrative passages and 2 with little financial
  content.
- **One question per passage.** `queries_per_passage.validation: 1` is part
  of the recorded run, so the config is not edited. The `sec-qgen-v2` prompt
  (`f42bdc54…`) then asks for "1 different questions". A second candidate in
  a reply is rejected as `excess_question` unless the first fails the
  validator.
- **Validator.** `sec-qcheck-v3` runs on every candidate.
- **Records.** The run has its own files: `generation/validation.run.json`
  and `generation/validation.records.jsonl`. No train file, pair file or
  candidate dataset changes. The candidate dataset pins only the train run's
  files and the pair files (tested).

**Plan.** `data/processed/train/bge_base_ft_v2/plans/validation.val_pilot_v1.json`
(SHA-256 `796b1873…2975`): 50 passages, `balanced_years`, seed 20261029.

```bash
python3 scripts/02_generate_queries.py plan --config configs/train/data_prep_v2.yaml --split validation \
    --name val_pilot_v1 --size 50 --method balanced_years --max-company-share 0.25 \
    --exclude-evidence-of train_candidates_v1      # written once
```

| Company | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Total | Tables / narrative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AAPL | – | 1 | 1 | 2 | 1 | 1 | 1 | – | 7 | 4 / 3 |
| AMZN | 1 | 1 | 1 | 1 | 1 | 1 | 1 | – | 7 | 3 / 4 |
| JPM | 1 | 1 | 1 | 1 | 1 | 1 | 1 | – | 7 | 3 / 4 |
| PFE | 1 | 1 | 1 | 1 | 1 | 1 | 2 | – | 8 | 4 / 4 |
| TSLA | 1 | 1 | 1 | 1 | 1 | 1 | 1 | – | 7 | 4 / 3 |
| WMT | – | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 7 | 3 / 4 |
| XOM | 1 | 1 | 1 | 1 | 1 | 1 | 1 | – | 7 | 3 / 4 |
| **All** | 5 | 7 | 7 | 8 | 7 | 7 | 8 | 1 | 50 | 24 / 26 |

- **Years.** Every company covers every fiscal year it has, except Apple:
  6 of 7. Walmart's fiscal years run from 2020 to 2026.
  - Apple has 19 validation sources in 8 split units.
  - Its FY2019 sources lie in only two units: the Item 2 Properties chain
    (FY2019–2025) and the Item 7A chain, which is excluded (below). The
    Properties chain is also the only FY2024 source.
  - One plan can therefore hold only one of FY2019 and FY2024.
- **`balanced_years`.** Within a company it picks by fiscal year first, then
  by the year the fewest remaining units can still supply. The kind and
  section balance of `balanced` comes after that. `balanced` alone took
  kinds first and covered 5 of Apple's 7 years.
- **Sections.** Statements 19, MD&A 11, risk 8, business 6, other 6. By item:
  Item 8 17, Item 7 10, sections without an item number (JPMorgan, Exxon
  notes) 10, and 1–3 each from Items 1, 1A, 2, 6, 7A and 10.
- **Constraints.** No company takes more than 25% of the plan. The plan uses
  50 distinct overlap groups and split units.
  - **Exclusion.** `--exclude-evidence-of train_candidates_v1` excludes 14
    overlap groups. They sit in the 5 split units of the 12 validation
    passages that may hold the evidence of 6 of the 66 training candidates.
  - **Why these are not caught later.** All 12 are matches build-pairs never
    rejects: 11 are uncertain (a figure, value or 8-word span), and 1 is exact
    but outside the passage's visible window.
  - **Without the exclusion,** `balanced` would have planned 2 such passages.
- **Weakest source:** Walmart FY2025 Item 10. It opens with a cross
  reference, then describes the insider trading policy. It passes the source
  filter, so it stays; the plan is not edited by hand.

**Requests and yield.**
- **Requests:** 50 if every reply is usable. `--max-requests 50` counts every
  HTTP attempt, retries included, so a retry leaves the last planned passage
  unrequested. Picks alternate between companies, so an early stop stays
  balanced.
- **The cap is per command.** A second run is a new budget and needs new
  approval, for at most 50 minus the requests the first run reports.
- **Duration:** about 25–50 minutes at 1,000 output tokens per minute.
  pilot_v3 sent exactly 50 requests for 50 passages, with no retries.

Expected yield (estimates from pilot_v3, not measurements):

| Stage | Expected count | Basis |
|---|---|---|
| Passing the validator | about 39 | 78% of pilot_v3 candidates |
| Passing the build-time leakage checks | about 30 | A visible sentence of a planned passage is held by a blocked passage (most) or a train passage in 21% of cases on average (median 9%), so a quote there rejects the pair |
| Accepted by an AI review | about 25 | 81% of pilot_v3 pairs |

That is about 25–33 usable questions from 50 requests, not 50. More would
need a second validation plan with its own approval.

**Commands.** The dry run sends nothing and needs no key:

```bash
python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v2.yaml --split validation --plan val_pilot_v1 --max-requests 50
```

The real run needs `GROQ_API_KEY` in the environment:

```bash
python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v2.yaml --split validation --plan val_pilot_v1 --max-requests 50 --confirm-api-use
```

**Safeguards.**
- **Validation examples never enter training.**
  - The trainer loads only the train-split candidates of
    `train_candidates_v1`.
  - `verify_pairs` refuses a pair of another split.
  - The candidate dataset pins its pair files and generation records by
    SHA-256.
- **Validation evidence never becomes a training negative.**
  - `NegativeGuard` allows only same-split, non-blocked negatives, for
    in-batch and mined negatives alike, and the trainer checks every batch.
  - A validation pair whose evidence a train passage's visible text holds
    exactly is rejected (`cross_split_evidence`), and so is a train pair whose
    evidence a validation passage holds.
  - Validation units that may answer a training candidate are excluded from
    the plan (above).
- **Pairs are written once (`write_pairs`).**
  - The old `build-pairs` overwrote the pair files. Today's checks would
    rebuild pilot_v3's pairs as 64, not 70, and only one would keep its ID
    and query. That would have broken the 70 quality decisions and
    `train_candidates_v1`.
  - A build that would change the written pairs is now refused, and nothing
    is written.
  - So the validation pairs need a pair set of their own (next steps).
- **No evaluation data is used to make or choose validation questions.**
  - The prompt has no examples.
  - The plan reads the corpus and the preparation only. Evaluation pools
    contributed passage IDs to the blocklist when the data was prepared.
  - `plan` and `generate` run with every evaluation question, label, pool,
    audit and run file moved out of reach (tested).
  - Evaluation questions are read only when pairs are built, to reject
    overlapping questions.
  - No pilot_v3 development metric is read.

**After the run (done offline, 2026-11-03):**
1. **Validation pairs** (`finretriever.training.validation_set`,
   `build-validation`, `configs/train/validation_v1.yaml`) are built into
   their own write-once set, `validation_sets/validation_pairs_v1/`, never
   into `pairs/`. Beyond the per-pair checks:
   - any match of the evidence in a train or blocked passage excludes the
     question, not only an exact match in visible text;
   - so does a question that copies any training question;
   - every positive must be unused by the training pairs and candidates.
2. **The AI quality review** is in
   `data/validation_review/val_pilot_v1_quality_decisions.jsonl`. It records
   an outcome for each of six aspects per question.
3. **Not built yet:** a synthetic validation evaluator. MRR@10 counts the
   positive and its exact equivalents as relevant and leaves uncertain ones
   unscored.
