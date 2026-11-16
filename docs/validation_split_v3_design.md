# A larger validation split (`bge_base_ft_v3`, proposed): split comparison and recommendation

Status: 2026-11-03. **Approved and implemented:** the split, the pins and the
target, with `sec-qgen-v2` kept. The result is
[training_data_v3.md](training_data_v3.md). Everything below is the design
as it was proposed, unchanged.

**Design only** (when written):
- Nothing was prepared or implemented, no API request was sent and no model
  was trained.
- **Unchanged:** the frozen corpus, `bge_base_ft_v2`, its candidates and
  `validation_pairs_v1`.
- **Where the numbers come from:** simulations run in memory over
  `bge_base_ft_v2`'s stored overlap groups, blocklist and evidence-reuse
  links.
  - At a 10% share the simulation reproduces today's split exactly, passage
    for passage.
  - The statistical figures come from re-scoring stored passage embeddings.
    Only the 82 query texts were encoded.

## 1. Recommendation

- **Split: company-tuned shares, not one larger uniform share.**
  - The validation shares are AAPL 45%, AMZN 45%, TSLA 35%, WMT 35%, XOM
    30%, PFE 25% and JPM 15%.
  - That is 22.6% of usable passages overall: 1,397 validation and 4,790
    training passages.
- **Existing data is pinned.**
  - Every unit that the existing training data uses stays in train: the
    candidates and the pilot pairs, with every passage that holds their
    evidence.
  - Every current validation unit stays in validation.
  - So all 66 candidates, all 64 currently valid pilot_v3 pairs and all 16
    validation questions survive unchanged. Nothing moves between splits
    without being listed.
- **Target: about 120 validation questions** (100 at least, 150
  comfortably), not 200–300.
  - That is enough to choose between checkpoints of one run by paired
    comparison, if they differ about as much as the measured small-change
    proxy (§5).
  - It is not enough to prove small gains between different systems, which
    is the evaluation set's job.
- **Expected yield:**
  - about 124 new balanced questions at one per split unit, plus the 16
    carried over: about 140;
  - about 100 new after about 230 requests, and the ceiling after about 434.
- **Why not simply 30% or 35%.**
  - A uniform 35% split costs 724 more training passages than T1.
  - It still gives Apple 6 questions and Amazon 4, with JPMorgan capped at
    25%.
  - Its balanced yield (131) is barely above T1's (124).
  - Without pins it would also invalidate 20 of the 66 candidates.

## 2. How the options were evaluated

- **Same inputs as v2.**
  - Overlap groups, blocklist (the same evaluation sources) and
    evidence-reuse links are unchanged.
  - So a new preparation differs from `bge_base_ft_v2` only in which split
    units go to validation.
- **The split algorithm is v2's** (`splits.assign_splits`).
  - Units are assigned whole, stratified by company × fiscal year, in seed
    order.
  - Each unit goes to validation if that brings its stratum closer to the
    target share.
  - **Pinned variants** assign the pinned units first and count them toward
    their stratum's target.
  - **Company-tuned variants** use a different target share per company.
- **Expected usable questions.**
  - For every visible sentence of every eligible usable passage (5,871
    passages), I recorded every passage the evidence locator finds holding
    it.
  - A sentence is clean under a split if no train or blocked passage holds
    it. That is the validation set's evidence rule.
  - The expected questions from requesting a passage are 0.86 × its clean
    share × 0.84.
    - 0.86 is the generator acceptance in `val_pilot_v1` (43/50).
    - 0.84 is the review acceptance among evidence-clean questions there
      (16/19).
  - **Calibration:** in `val_pilot_v1` the model predicted 41% clean and
    44% was observed.
  - Plans take one passage per split unit, as now, skipping the 50 units
    `val_pilot_v1` already used. No company may exceed 25% of the set.

## 3. Split comparison

All rows except the first are pinned. "Eligible" means at least 40 visible
words and kept by the source-quality filter; eligible validation units
exclude the 50 already used.

| Option | Share of usable | Train | Validation | Eligible validation | Validation overlap groups | Validation units | Eligible units left | Eligible train |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Today (v2) | 10.2% | 5,554 | 633 | 617 | 358 | 219 | 166 | 5,254 |
| 20% | 20.3% | 4,933 | 1,254 | 1,161 | 699 | 431 | 374 | 4,710 |
| 25% | 25.5% | 4,607 | 1,580 | 1,474 | 819 | 481 | 423 | 4,397 |
| 30% | 29.3% | 4,374 | 1,813 | 1,700 | 958 | 571 | 512 | 4,171 |
| 35% | 34.3% | 4,066 | 2,121 | 1,994 | 1,097 | 656 | 595 | 3,877 |
| **T1 (tuned)** | **22.6%** | **4,790** | **1,397** | **1,270** | **798** | **495** | **434** | **4,601** |

**Validation passages by company:**

| Option | AAPL | AMZN | JPM | PFE | TSLA | WMT | XOM | JPM share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Today | 19 | 27 | 320 | 92 | 52 | 48 | 75 | 51% |
| 20% | 42 | 54 | 612 | 207 | 98 | 83 | 158 | 49% |
| 25% | 46 | 73 | 831 | 235 | 107 | 97 | 191 | 53% |
| 30% | 53 | 75 | 950 | 267 | 129 | 112 | 227 | 52% |
| 35% | 60 | 85 | 1,086 | 318 | 152 | 125 | 295 | 51% |
| **T1** | **72** | **112** | **474** | **235** | **152** | **125** | **227** | **34%** |

**Validation passages by fiscal year:**

| Option | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Today | 88 | 101 | 97 | 78 | 88 | 84 | 89 | 8 |
| 20% | 168 | 203 | 179 | 152 | 195 | 162 | 183 | 12 |
| 25% | 222 | 260 | 235 | 204 | 206 | 208 | 231 | 14 |
| 30% | 253 | 296 | 265 | 233 | 231 | 247 | 272 | 16 |
| 35% | 293 | 332 | 309 | 274 | 283 | 301 | 311 | 18 |
| **T1** | **198** | **233** | **200** | **163** | **196** | **182** | **207** | **18** |

Only Walmart has fiscal 2026 (its year ends in January), and Walmart has no
2019.

**Expected usable validation questions** (new questions; add the 16
carried over):

| Option | Uncapped | Balanced (no company > 25%) | With today's planner | Two per unit, uncapped | Requests for 100 balanced | Requests at the balanced ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Today | 41 | 24 | 36 | — | not reachable | — |
| 20% | 98 | 67 | 86 | 118 | not reachable | 224 for 70 |
| 25% | 112 | 82 | 98 | 137 | not reachable | 272 for 86 |
| 30% | 146 | 109 | 127 | 176 | 276 | 330 for 110 |
| 35% | 180 | 129 | 159 | 220 | 218 | 368 for 131 |
| **T1** | **124** | **123** | **108** | **147** | **228** | **434 for 124** |

- **Today's planner** takes a random passage per unit. The other columns
  assume a planner that takes each unit's cleanest passage (§8).

**Expected questions per company under T1:** AAPL 8.5, AMZN 8.2, JPM 31.1,
PFE 21.7, TSLA 20.1, WMT 13.6 and XOM 20.4. Under 30%, they are 4.1, 2.8,
64.4, 26.6, 15.4, 12.3 and 20.4.

**Existing training data** (§6 has the identities):

| Option | Candidates kept of 66, pinned | Candidates kept, not pinned | Of which pilot_v2 (15) | pilot_v3 pairs kept of 64, not pinned | `validation_pairs_v1` kept of 16 |
|---|---:|---:|---:|---:|---:|
| 20% | 66 | 59 | 15 | 55 | 16 |
| 25% | 66 | 52 | 13 | 50 | 16 |
| 30% | 66 | 49 | 8 | 53 | 16 |
| 35% | 66 | 46 | 6 | 52 | 16 |
| T1 | 66 | 51 | 11 | 52 | 16 |

- **pilot_v3 pairs:** 6 of the 70 already fail today's checks (the unit bug
  found in the pilot_v3 audit), so 64 is the baseline.
- **Pinned variants keep all 64.** No current validation passage leaves
  validation, and no pinned training unit leaves training.

## 4. Company and year diversity

- **Apple and Amazon are limited by the corpus, not the share.**
  - They have only 60 and 63 usable split units, because most of their
    passages are evaluation-blocked (435 and 521).
  - 29% and 35% of their visible sentences are held by a blocked passage,
    which no split can change.
  - Most of the rest (61% and 56%) is held only by other usable passages.
    Moving more of their units into validation therefore makes their
    questions cleaner, so T1 raises their shares.
- **The cost to their training data.** Eligible training passages fall
  from 150 to 111 for Apple and from 185 to 107 for Amazon. Neither has
  much to give.
- **Even so, about 8 questions each** (roughly 6% of the set) is the most
  either can supply without giving up most of its training data. That is
  "represented", not "equal".
- **No company dominates.**
  - Under T1 JPMorgan supplies 34% of validation passages but, at 31 of 124
    expected questions, 25% of the set.
  - Under the uniform shares JPMorgan holds about half the validation
    passages, and the 25% cap binds.
- **Fiscal years.** Under T1 every company has eligible validation units in
  every year it reports (2019–2025, Walmart 2020–2026), at least 5 per cell
  and at least 10 for most:

| T1 eligible validation units | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 13 | 10 | 11 | 5 | 6 | 5 | 5 | – |
| AMZN | 15 | 12 | 12 | 6 | 8 | 11 | 13 | – |
| JPM | 33 | 41 | 28 | 34 | 59 | 43 | 48 | – |
| PFE | 19 | 28 | 25 | 17 | 20 | 17 | 20 | – |
| TSLA | 32 | 16 | 13 | 12 | 13 | 11 | 20 | – |
| WMT | – | 21 | 16 | 14 | 9 | 14 | 16 | 17 |
| XOM | 25 | 34 | 29 | 21 | 22 | 24 | 32 | – |

A unit spanning several years counts in each year it has an eligible
passage.

**Leakage protections are unchanged:**
- **Groups stay whole.** Every overlap group and every split unit (groups
  joined by evidence reuse) is assigned whole, so none crosses train and
  validation. Every simulated option passes `validate_split`.
- **Evaluation evidence stays out.**
  - The blocklist is v2's: the same evaluation sources, with their overlap
    groups and the passages that reuse their evidence.
  - Training pairs are still rejected for evidence a blocked passage holds.
  - Validation questions are rejected for any evidence held outside
    validation, exactly or possibly.
- **Equivalent passages never become false negatives.**
  - `NegativeGuard` still allows only same-split, non-blocked negatives and
    excludes every exact and uncertain equivalent.
  - A larger validation split only moves possible holders of training
    evidence to the other split, where they are never negatives anyway.
  - The validation evaluator, once built, will count the positive and its
    exact equivalents as relevant and leave uncertain ones unscored.
- **The existing checkpoints stay independent of v3 validation.** With
  pins, no v3 validation passage was a training positive or in-batch
  negative of the smoke-test checkpoints. Without pins, 7–20 of their
  training positives would land in validation.

## 5. The statistical target

**Measured** on the 82 synthetic questions (the 66 training candidates and
the 16 validation questions):
- Each question ranks the full 10,137-passage corpus. The relevant passages
  are the positive and every passage holding the evidence exactly; possible
  holders are removed from the ranking.
- Base models only: no checkpoint was used, and no evaluation question or
  label was read.

| Paired comparison | Mean difference in MRR@10 | SD of per-question differences | 95% bootstrap CI of the SD | Questions that changed |
|---|---:|---:|---:|---:|
| BGE vs BGE without its query instruction (a small change) | +0.031 | 0.109 | 0.060–0.147 | 21 of 82 (26%) |
| BGE vs E5 (different dense models) | −0.092 | 0.354 | 0.266–0.427 | 39 (48%) |
| BGE vs BM25 (dense vs lexical) | −0.281 | 0.443 | 0.375–0.505 | 53 (65%) |

- **MRR@10 levels:** BM25 0.536, E5 0.347, BGE 0.255. On the 16
  validation questions alone, BGE scores 0.165.
- **Clustering by company is negligible.**
  - The intra-company correlation of the differences is 0.00–0.04.
  - A company-cluster bootstrap gives the same standard error as treating
    questions as independent (0.043 against 0.039 for BGE vs E5).

**Questions needed for two checkpoints, paired**, n = (z × SD / δ)²:

| SD of differences | Pick the better one with ≥ 90% probability: δ = 0.02 / 0.03 / 0.05 | Detect the difference (α = 0.05, power 0.8): δ = 0.02 / 0.03 / 0.05 |
|---|---|---|
| 0.11 (measured small change) | 49 / 22 / 8 | 234 / 104 / 38 |
| 0.15 (upper end of its interval) | 93 / 42 / 15 | 442 / 197 / 71 |
| 0.25 (assumed earlier) | 257 / 115 / 42 | 1,227 / 546 / 197 |
| 0.35 (measured, different models) | 515 / 229 / 83 | 2,460 / 1,093 / 394 |

**Assumptions:**
- Checkpoints of one run differ like the small-change proxy (SD about
  0.11–0.15). The proxy changes the query side only, while fine-tuning
  changes both sides. Checkpoints far apart in training may differ more.
- Choosing a checkpoint is a ranking decision, so the probability of
  picking the better one is the relevant criterion. Proving that a gain is
  real belongs to the evaluation set.
- The 82 synthetic questions behave like future validation questions:
  same generator, prompt and corpus.

**Estimates:**
- **At 120 questions** with SD 0.11–0.15, the better of two checkpoints
  0.02 apart is chosen 93–98% of the time.
- **The same 120 questions** detect a 0.03 difference with power 0.59–0.85.
- **Where 200–300 came from:** it was the "detect δ = 0.05 at SD
  0.25–0.30" figure (197–283). It is right for comparing different systems,
  and more than choosing within a run needs.

**Unknowns:**
- How much real checkpoints differ (the smoke-test checkpoints are
  trivial, 15 steps, and not available here).
- Whether validation rankings track evaluation rankings. On these
  synthetic questions BM25 beats BGE by 0.28, so they reward lexical
  matching more than the evaluation questions may.
- The yields in §3 rest on a 50-passage calibration. The generator
  acceptance of 86% has a 95% interval of 74–93%, and the review acceptance
  of 84% one of 62–95%. Expect ±20–25%.

## 6. Existing training data under each split

**Without pins, these candidates would become invalid,** all because their
positive would land in validation:
- **20%** (7): pilot_v3 03, 15, 16, 20, 49, 57, 70.
- **25%** (14): pilot_v2 19, 20; pilot_v3 03, 15, 16, 20, 21, 22, 49, 51,
  53, 54, 57, 70.
- **30%** (17): pilot_v2 14, 17, 18, 19, 20, 23, 24; pilot_v3 03, 15, 16,
  20, 45, 46, 51, 53, 57, 70.
- **35%** (20): pilot_v2 14, 17, 18, 19, 20, 23, 24, 30, 31; pilot_v3 03,
  15, 16, 20, 33, 41, 42, 45, 46, 57, 70.
- **T1** (15): pilot_v2 19, 20, 30, 31; pilot_v3 01, 03, 15, 16, 20, 41,
  42, 45, 46, 57, 70.

Of the 64 currently valid pilot_v3 pairs, 9, 14, 11, 12 and 12 would become
invalid.

**With pins, none.**
- **What is pinned to train:** 109 split units (1,901 passages), every unit
  holding a positive of the candidates or of the built pairs, or one of the
  50 pilot_v3 plan passages, or any passage that holds their evidence.
- **What is pinned to validation:** every current validation unit (633
  passages), so v3 validation contains all of v2's.
- **Nothing moves silently.** The pins are listed in the configuration, and
  each is resolved to passage IDs recorded in the manifest. Carried-over
  candidates are re-checked, and any exclusion would be logged with its
  reason.

## 7. Proposed configuration

`configs/train/data_prep_v3.yaml`, to be created only after approval. It is
v2 with three changes. `split.validation_share_by_ticker` and `split.pins`
are new and need code (§8).

```yaml
# Leakage-safe training-data preparation v3 (bge_base_ft_v3): v2 with a larger, company-tuned validation split.
# Everything but the split is v2's, so groups, blocklist and evidence-reuse files must come out byte-identical.
version: bge_base_ft_v3
corpus_path: data/processed/corpus_v1.jsonl
corpus_version: corpus_v1@1.2
corpus_sha256: ff9dadf37bb9820e2c87df70a411fd189bb02776609e5ec06173dd03f2344004
output_dir: data/processed/train/bge_base_ft_v3
min_source_words: 40
overlap: {...}            # as data_prep_v2.yaml
blocklist: {...}          # as data_prep_v2.yaml (same evaluation sources)
visible_text: {...}       # as data_prep_v2.yaml
reuse: {...}              # as data_prep_v2.yaml
generation: {...}         # as data_prep_v2.yaml (sec-qgen-v2, one validation question per passage)
leakage: {...}            # as data_prep_v2.yaml
source_quality: {...}     # as data_prep_v2.yaml

split:
  seed: 20261029                      # v2's seed, so the split stays close to v2's
  validation_share: 0.25              # strata of any company not listed below
  validation_share_by_ticker:         # NEW: more validation for companies with few units, less for JPMorgan
    AAPL: 0.45
    AMZN: 0.45
    TSLA: 0.35
    WMT: 0.35
    XOM: 0.30
    PFE: 0.25
    JPM: 0.15
  stratify_by: [ticker, fiscal_year]
  pins:                               # NEW: units fixed before the split; resolved to passage IDs in the manifest
    from_preparation: configs/train/data_prep_v2.yaml   # its manifest SHA-256 is recorded and checked
    validation: all                   # every bge_base_ft_v2 validation unit stays validation (v3 contains v2's)
    train:                            # these units stay train: positives and every passage holding their evidence
      candidates: [train_candidates_v1]
      pairs: true                     # bge_base_ft_v2 pairs/train_pairs.jsonl
      plans: [pilot_v3]
```

- **Failure checks.** `prepare` would refuse any of these:
  - a unit pinned both ways;
  - a pin on a blocked passage;
  - a v2 file whose SHA-256 differs from the one recorded.
- **A post-prepare check:** the v2 and v3 overlap, blocklist and
  evidence-reuse files must be byte-identical.

## 8. Migration and revalidation plan

Each step is offline unless marked, and each waits for approval.
`bge_base_ft_v2` and `validation_pairs_v1` are checksummed before and after
every step.

1. **Code, with tests, before any preparation:**
   - **`splits.py`:**
     - per-company target shares;
     - pinned units, assigned first and counted toward their stratum;
     - refusals for conflicting or blocked pins;
     - a regression test that no pins and no per-company shares reproduce v2
       exactly (as the simulation does).
   - **`data_prep.py`:**
     - resolve pins from the v2 artifacts (candidates, built pairs, plans and
       their evidence holders; v2 validation units);
     - record them, with every input's SHA-256, in the manifest.
   - **`validation_set.py`:** carry a validation set into a new preparation.
     It re-runs every check, keeps decisions bound by question SHA-256, and
     logs any exclusion.
   - **`sampling_plan.py`:**
     - an evidence-aware option that takes, in each unit, the passage with
       the most sentences that nothing outside validation holds (about +15%
       yield, §3);
     - an option to skip units already requested.
2. **`prepare --config configs/train/data_prep_v3.yaml`.** Verify:
   - identical groups, blocklist and evidence-reuse files and index digest;
   - v2 validation ⊆ v3 validation;
   - every pinned unit on its side;
   - the split statistics match this report.
3. **Training data.**
   - `build-candidates` in v3, with sources pilot_v3 (from v2) and pilot_v2
     (from v1), re-checked against v3, into a new version
     (`train_candidates_v2`).
   - Expect 66 of 66. Any other count is a finding to report, not to repair.
4. **Validation carry-over.** `validation_pairs_v1`'s 16 are re-checked into
   v3. Expect 16 of 16.
5. **New validation plans** under v3:
   - `balanced_years`, at most 25% per company, one passage per unit,
     evidence-aware;
   - skipping the 50 units `val_pilot_v1` used;
   - `--exclude-evidence-of` the new candidate dataset.
6. **Generation (API, approval per batch).** Batches of at most 50 requests,
   each with its dry run and its own `--max-requests` cap.
7. **AI review, then `build-validation` in v3,** to a new version
   (`validation_pairs_v2`). Then the validation evaluator.

## 9. Requests and generation time

| Goal (total including the 16) | New questions | Requests (evidence-aware planner) | At the observed 2.3 per minute | At the worst-case 1 per minute |
|---|---:|---:|---:|---:|
| About 116 | about 100 | about 230 | about 1.7 h | about 3.8 h |
| About 140 (T1's ceiling at one per unit) | about 124 | about 434 | about 3.2 h | about 7.2 h |

- **The observed rate:** `val_pilot_v1` sent 50 requests in 22 minutes,
  with one question per reply.
- **The worst case:** each request reserves 768 of the account's 1,000
  output tokens per minute.
- **With today's planner** (a random passage per unit), yields are about
  13% lower, so more requests are needed.
- **Batching:** 5 to 9 batches of up to 50 requests.

## 10. Risks and unresolved assumptions

1. **Pins tie v3 to v2.** 31% of usable passages are pinned to train, so the
   split is no longer a function of seed and share alone. The pins come from
   balanced pilot plans and use no evaluation label, but they are a
   constraint, and they are recorded.
2. **Apple and Amazon stay thin:** about 8 questions each, with wide
   intervals. Their training supply falls by a quarter and by 42%.
3. **The validation content is unrepresentative.**
   - The evidence rule, and an evidence-aware planner even more so, select
     validation content that is the part of the corpus least like the
     evaluation pools.
   - Synthetic questions favor lexical matching (BM25 0.54 against BGE
     0.26).
   - Checkpoint choice on them may not track evaluation results.
4. **The statistics rest on a proxy.** Real checkpoint differences are
   unmeasured; 82 questions give the SD a wide interval (0.06–0.15 for the
   small change).
5. **The yield model is a sentence-level proxy without the answer.** It was
   calibrated once (41% predicted against 44% observed).
6. **Groq:** daily limits, model availability (`qwen/qwen3.8-27b`) and run
   time (3–7 hours in total) are not guaranteed.
7. **Blocklist drift.** If evaluation pools grow before v3 is prepared, more
   passages are blocked and these numbers shift. The byte-identity check in
   §8 would catch it.
8. **New code.** Per-company shares, pins, validation carry-over and
   evidence-aware plans are untested until written.
9. **Prompt choice.**
   - Keeping `sec-qgen-v2` keeps new questions comparable with the carried
     ones, but its mid-table misreadings remain (about 12% factual errors in
     `val_pilot_v1`, caught by review).
   - A `sec-qgen-v3` would need its own pilot. This is a separate decision.
10. **Correlated questions.** Taking two passages per unit would add about
    20% more questions, but those questions are correlated. It is not part
    of the recommendation.
