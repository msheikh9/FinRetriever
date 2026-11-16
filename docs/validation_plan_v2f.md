# Validation plans `val_pilot_v2f` and `val_pilot_v2g`: a calibrated predictor and the whole remaining pool

Status: 2026-11-03. For approval; nothing generated.
- **What was not done:** no Groq request was sent and no model was trained.
- **Unchanged:** the corpus, `bge_base_ft_v3`, the 66 training candidates, the 16 validation questions, batch
  1's 15 accepted questions, the generation records, the evaluation data and every earlier plan.
- **Checksums:** of the 267 files under `data/`, `configs/` and `results/`, none changed except the review
  README. The added files are listed in §10.

## 1. Summary

**Measured so far:** 31 usable validation questions (the 16 carried, and 15 from batch 1 of
`val_pilot_v2e`). The plan's first 50 requests gave 15 usable questions (30%).

**Batch 1 is recorded** in `data/validation_review/`:
- the 39 AI-review decisions, which `build-validation` can read: 36 accept, 1 revise, 2 reject;
- an outcomes log of all 50 passages: generator rejection (11), evidence rule (22), review (2 rejected, 1
  revision), and 15 usable.

**The predictor (`evidence-pass-v1`, calibrated on 82 observed questions):**
- **Ranking:** AUC 0.81, against 0.70 for the isolation share that ordered `val_pilot_v2e`.
- **Accuracy:** leave-one-out Brier score 0.172, against 0.246 for a constant.
- **It ranks passages and assigns no labels.**

**Administrative filter:** a new opt-in source rule removes 31 governance passages: Item 9B trading
plans, Item 1C cybersecurity and Item 10 insider-trading policy. It removes no risk factor, MD&A or financial
statement.

**The whole remaining pool was inspected:** 667 eligible validation passages in 436 overlap groups.

**Two plans, written once:**
- `val_pilot_v2f`: all 361 passages the caps allow, best-predicted first within the company and year
  balance. 8 batches.
- `val_pilot_v2g`: 238 passages predicted to pass at 0.2 or more, under the same caps. 5 batches. 221 of them
  are also in `val_pilot_v2f`.

**The 120-question target** (89 more usable questions) is not reachable with certainty under the current rules:

| Plan | Requests | New usable: central (10th–90th percentile) | Total with the 31 |
|---|---:|---|---|
| `val_pilot_v2f` | 361 | 87.5 (70.8–105.1) | 118.5 (102–136) |
| `val_pilot_v2g` | 238 | 78.3 (64.6–91.4) | 109 (96–122) |

- **Even all of `val_pilot_v2f`** is a little below 89 on the central estimate.
- **The binding constraint** is the cap of three passages per linked unit (§5).

**Recommendation:** `val_pilot_v2f`, approved one batch of 50 at a time.
- Audit each batch, and stop once 120 usable questions are reached.
- After two batches, recalibrate if the observed yield falls outside the projected range.
- `val_pilot_v2g` is the request-efficient alternative: about 9 fewer questions for 123 fewer requests.

## 2. Batch 1, recorded

| File | Rows | What |
|---|---:|---|
| `data/validation_review/val_pilot_v2e_batch1_quality_decisions.jsonl` | 39 | One AI-review decision per generator-accepted question (`load_validation_decisions` format); binds `pair_id` (`validation_000001`…), `passage_id` and `question_sha256` |
| `data/validation_review/val_pilot_v2e_batch1_outcomes.jsonl` | 50 | Every passage: plan position, the record's SHA-256, the question and its SHA-256, generator outcome and reasons, evidence-rule outcome and reasons, review decision and issues, `excluded_by`, `usable` |

- **Why two files:** a decision needs a question the generator accepted, so the 11 generator-rejected
  passages appear only in the outcomes log.
- **Labelling:** both files mark every row `reviewer: ai_review`, `gold: false`.
- **Loading checked:** the real loader reads the decisions file without error.

## 3. The predictor (`src/finretriever/training/evidence_prediction.py`)

**What it predicts:** the probability that a question generated from a passage passes the validation
evidence rule. It ranks passages for a plan; it never labels relevance.

**Measurements on the visible text** (corpus only; no evaluation label is read):
- **Unique financial figures:** the share of the passage's significant figures whose amount no passage of the
  company outside validation holds. This is the evidence locator's own amount test, on full text.
  - Training and blocked holders count, so evaluation pools and their linked passages count too.
  - The old score ignored this entirely. Figure matches alone excluded 9 of batch 1's 22.
- **Isolated sentences:** the old score, the share of prose sentences no outside passage holds.

**Mixture and calibration:**
- **The mixture:** the two shares weighted by how often the generator quotes a significant figure (measured:
  tables 0.70, narrative 0.50).
- **The fit:** a logistic fit on its logit, with two parameters (intercept −0.361, slope 0.838) and a ridge
  pull toward the uncalibrated mixture.
- **Expected usable questions:** predicted pass × generator acceptance 0.82 × review acceptance 0.86 (pooled
  observed rates).

**Against overfitting:**
- **Few fitted numbers:** two fitted parameters and two measured rates.
- **Variants tried:** four (all figures, labelled figures, amounts only, sentence share alone), each judged by
  an unfitted AUC: 0.78 for each figure variant, 0.70 for the sentence share alone, measured before
  percentages were matched correctly. The simplest figure variant was kept.
- **Not fitted:** company, year and opening multipliers. They are confounded, with too few review failures
  (10) to fit.
- **Error estimate:** the leave-one-out error is recorded in the calibration file.

**Calibration** (`configs/train/evidence_pass_v1.json`, written once by `calibrate-predictor`). It holds
every observation and the files they came from, by SHA-256.

| Measured | Value |
|---|---|
| Questions | 82 (`val_pilot_v1` 43 in `bge_base_ft_v2`, batch 1 39 in `bge_base_ft_v3`) |
| AUC: predictor / old isolation share | 0.806 / 0.697 |
| Leave-one-out Brier: predictor / constant | 0.172 / 0.246 |
| Passes predicted (leave-one-out) / observed: `val_pilot_v1` | 15.4 / 19 |
| Passes predicted (leave-one-out) / observed: batch 1 | 20.9 / 17 |

**The caveat (measured):**
- **The over-prediction:** on batch 1, a plan chosen by a score, the predictor over-predicts by 23%. Plans
  chosen by this predictor will select on its errors too.
- **The stress case:** the projections include a "selection stress" scaled by 17/20.9.

## 4. The administrative rule

- **Name:** `administrative_disclosure` in `source_quality`, opt-in. Plans enable it with
  `--administrative-rule`, so `data_prep_v3.yaml` is unchanged and earlier plans rebuild identically
  (checked).
- **When it excludes a passage:**
  - it is not a protected item (1A, 7, 7A, 8);
  - it has fewer than 2 financial figures per 100 words;
  - it holds at least 3 insider-trading, trading-plan or cybersecurity-governance markers, or at least 1 in
    Item 1C, 9B or 10.
- **On `bge_base_ft_v3`'s 1,397 validation passages** it excludes 31, every one a governance disclosure:
  - Item 9B trading arrangements (AAPL, JPM, PFE, WMT, XOM);
  - Item 1C cybersecurity (AMZN, WMT);
  - Item 10 insider-trading policy (AAPL, WMT).
- **What it leaves:** passages that mention a code of conduct inside business or MD&A text are kept.

## 5. The remaining supply

**The rules applied** (as requested):
- exclude every previously generated passage and its overlap group (`val_pilot_v1`'s 50 and batch 1's 50);
- keep the units that may hold a training candidate's evidence excluded (leakage);
- exclude the overlap groups of the 16 carried questions and their equivalents;
- at most three passages per linked unit, counting the 100 generated ones;
- one passage per overlap group; at most 25% per company.

| Step | Validation passages |
|---|---:|
| Approved in `bge_base_ft_v3` | 1,397 |
| Kept by the source rules (with the context and administrative rules) | 880 |
| Outside training-candidate evidence units | 864 |
| Outside the generated passages' groups | 672 |
| Outside the carried questions' equivalents' groups | 667 (436 groups) |
| Within the unit cap and company cap: `val_pilot_v2f` | 361 |

**Differences from `val_pilot_v2e`'s exclusions:**
- E excluded `val_pilot_v1`'s whole units. Reopening their never-generated groups, within the cap, adds 21
  passages and about 7 expected questions.
- 128 of E's 166 ungenerated passages are in `val_pilot_v2f`. 6 are now excluded by the administrative rule.

**The cap per unit is the binding constraint** (expected new usable, central):
- cap 3: 361 passages, 87.5;
- cap 4: 381, 93.3;
- cap 5: 394, 97.5;
- no cap: about 106 if every group's best passage were generated.

## 6. The plans

**Common settings:**
- `balanced_years`, seed 20261029, one passage per overlap group, at most 3 per linked unit (counting
  earlier generation), at most 25% per company;
- scores from `evidence-pass-v1`, with score priority (score in 0.1 steps right after the year balance);
- the administrative rule, and the exclusions of §5.

| | `val_pilot_v2f` | `val_pilot_v2g` |
|---|---|---|
| Minimum predicted pass | none | 0.2 |
| Passages = requests | 361 | 238 |
| Batches of ≤ 50 | 8 (7 × 50, 11) | 5 (4 × 50, 38) |
| Mean predicted pass | 0.34 | 0.47 |
| AAPL / AMZN / JPM / PFE / TSLA / WMT / XOM | 20 / 29 / 90 / 58 / 53 / 43 / 68 | 11 / 21 / 59 / 39 / 43 / 23 / 42 |
| Expected usable by company | 4.7 / 6.4 / 23.9 / 15.0 / 13.5 / 9.1 / 14.9 | 4.0 / 6.3 / 21.6 / 13.6 / 12.8 / 7.4 / 12.6 |
| Fiscal years 2019 → 2026 | 69 / 66 / 55 / 36 / 42 / 36 / 52 / 5 | 54 / 46 / 28 / 21 / 24 / 25 / 37 / 3 |
| FY2019 share | 19% | 23% |
| Narrative / table | 192 / 169 | 133 / 105 |
| Units with 2–3 passages; largest filing | 58; Tesla FY2019 19 | 38; Tesla FY2019 17 |
| Time at batch 1's pace, 2.5 per minute (worst case 1 per minute) | 2.4 h (6.0 h) | 1.6 h (4.0 h) |
| Tokens at batch 1's 1,322 per request | about 477,000 | about 315,000 |
| SHA-256 | `a7903bc7…` | `4c90eaf6…` |

**Balance comes first:**
- the first 150 passages of `val_pilot_v2f` hold 20–22 per company, 21 from FY2019 (14%);
- the JPMorgan and FY2019 excess sits in the low-yield tail.

## 7. Expected yield (projections, not measurements)

- **Central:** the calibrated predictor.
- **Conservative and optimistic:** the 10th and 90th percentiles over 400 bootstrap refits of the calibration
  (observations resampled, rates and fit re-estimated).
- **Selection stress:** central × 17/20.9, batch 1's over-prediction.

| New usable questions | Central | Conservative | Optimistic | Selection stress |
|---|---:|---:|---:|---:|
| `val_pilot_v2f`, all 361 | 87.5 | 70.8 | 105.1 | 71.2 |
| `val_pilot_v2g`, all 238 | 78.3 | 64.6 | 91.4 | 63.6 |
| `val_pilot_v2e`'s 160 remaining eligible passages (for comparison) | 53.1 | 44.2 | 62.2 | — |

**Cumulative, central** (new usable questions; total with the 31 in brackets):

| After requests | 50 | 100 | 150 | 200 | 250 | 300 | 361 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `val_pilot_v2f` | 23.5 (55) | 40.0 (71) | 53.1 (84) | 63.7 (95) | 73.6 (105) | 81.9 (113) | 87.5 (119) |
| `val_pilot_v2g` | 23.7 (55) | 41.2 (72) | 56.2 (87) | 70.2 (101) | 78.3 (109) at 238 | | |

**For comparison:** `val_pilot_v2e`'s next 50 passages are expected to give 15.8; `val_pilot_v2f`'s first 50,
23.5.

**Against 120:**
- **`val_pilot_v2f`:** reaches it only above its central estimate.
- **`val_pilot_v2g`:** reaches it only near its optimistic bound.
- **Under the selection stress,** both end near 95–102.

## 8. Original and improved sampling, compared

| | `val_pilot_v2e` (original) | `val_pilot_v2f` / `v2g` (improved) |
|---|---|---|
| Score | Isolated sentence share: tables and answer figures not seen | Calibrated: unique figures and isolated sentences |
| Its accuracy (measured) | Predicted 75% of batch 1 to pass the evidence rule; 44% did. AUC 0.70 | AUC 0.81; batch 1 over-predicted by 23% (leave-one-out) |
| Role of the score | Eligibility at 0.3, then the last tie-break | Ranks right after the year balance; optional minimum |
| Administrative passages | Kept (4 of batch 1's 50, none usable) | Excluded (31 in the split) |
| Exclusions | `val_pilot_v1`'s whole units | Generated passages' groups; units capped at 3, counting them |
| Filing cap | 8 per company-year | None: it would cost about 10 questions (`val_pilot_v2f`) |
| Expected usable per request (central) | 0.33 for its remainder; 0.30 observed in batch 1 | 0.24 (`v2f`, whole) / 0.33 (`v2g`); 0.47 in each plan's first batch |
| Expected new usable | 53 (remainder) | 87.5 / 78.3 |

## 9. Safeguards, checked on both plans

**Both plans pass every check:**
- validation passages only, kept by every source rule;
- one passage per overlap group;
- no generated passage or its group, no carried question's equivalent group, no training-candidate evidence
  unit;
- at most 3 per linked unit including the 100 generated passages;
- company cap held;
- scores equal to a fresh prediction;
- made from the `bge_base_ft_v3` manifest (`21c2293a…`).

**Not touched:**
- **The split:** a plan changes no split, so the pins and evidence-reuse-v2 units are as they were.
- **Leakage rules:** the strict evidence rule, the evaluation and training overlap checks and the duplicate
  check apply again when a validation set is built. Nothing is relaxed.

**Determinism and write-once:**
- every earlier plan (`pilot_v3`, `val_pilot_v1`, `val_pilot_v2`, `v2b`–`v2e`) rebuilds byte-identically;
- the calibration rebuilds "unchanged".

## 10. Commands and files

```bash
# The calibration (written once).
python3 scripts/02_generate_queries.py calibrate-predictor --config configs/train/data_prep_v3.yaml \
    --out configs/train/evidence_pass_v1.json \
    --pilot configs/train/data_prep_v2.yaml val_pilot_v1 data/validation_review/val_pilot_v1_quality_decisions.jsonl 50 \
    --pilot configs/train/data_prep_v3.yaml val_pilot_v2e data/validation_review/val_pilot_v2e_batch1_quality_decisions.jsonl 50

# The plans (written once). val_pilot_v2g adds --min-predicted-pass 0.2 and has size 238.
python3 scripts/02_generate_queries.py plan --config configs/train/data_prep_v3.yaml --split validation \
    --name val_pilot_v2f --size 361 --method balanced_years --max-company-share 0.25 --one-per group \
    --max-per-unit 3 --predictor configs/train/evidence_pass_v1.json --score-priority --administrative-rule \
    --after-generated configs/train/data_prep_v2.yaml --exclude-evidence-of train_candidates_v2 \
    --exclude-equivalents-of configs/train/data_prep_v3.yaml:validation_pairs_v2

# A dry run (sends nothing); each approved batch adds --confirm-api-use.
python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v3.yaml --split validation \
    --plan val_pilot_v2f --max-requests 50
```

| File | Tracked | What |
|---|---|---|
| `src/finretriever/training/evidence_prediction.py` | yes | The predictor, calibration and write-once file |
| `configs/train/evidence_pass_v1.json` | yes | The calibration (SHA-256 `3b2e6448…`) |
| `data/validation_review/val_pilot_v2e_batch1_*.jsonl` | no (gitignored data) | Batch 1 decisions and outcomes; README updated |
| `data/processed/train/bge_base_ft_v3/plans/validation.val_pilot_v2{f,g}.json` | no (gitignored data) | The plans |

**Building the validation set later:**
- `val_pilot_v2e`'s batch-1 questions are decided under that plan (positions 1–50).
- A later set can build from `val_pilot_v2f` and carry the batch-1 set, which `build-validation` supports
  within one preparation.
- Records are per passage, so a passage generated under `val_pilot_v2f` is never generated twice.

## 11. Risks and decisions

1. **The target:**
   - choose between generating all of `val_pilot_v2f` (central 119), `val_pilot_v2g` (109), or more supply;
   - more supply could come from a cap of 4 per unit (+6) or 5 (+10), which weakens independence;
   - or from a larger validation split, a new preparation that costs training data.
   - Relaxing the evidence rule is not recommended.
2. **Selection effect:** the predictor was fitted on 82 questions and over-predicted the score-selected batch
   by 23%. The conservative and stress columns cover it. Recalibrating after two batches would show whether
   it persists.
3. **Concentration in the tail:** JPMorgan reaches the 25% cap, Tesla FY2019 gives up to 19 passages, and
   Apple and Amazon stay small (about 5 and 6 new questions). Early stopping avoids most of the tail.
4. **Correlated questions:** 58 units give 2–3 passages each in `val_pilot_v2f`. A validation set records
   their equivalents.
5. **The predictor is a ranking aid** fitted on AI-reviewed synthetic questions. It is not evidence about
   retrieval quality, and none of its numbers are labels.
