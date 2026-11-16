# Training-data preparation v3 (`bge_base_ft_v3`): a larger, company-tuned validation split

Status: 2026-11-03.
- **Prepared, migrated and planned offline.** No Groq request was sent and
  no model was trained.
- **Unchanged:** the frozen corpus, the evaluation data,
  `bge_base_ft_v2` and everything built on it (pairs, candidates,
  `validation_pairs_v1`, generation records).
- **The design** is in [validation_split_v3_design.md](validation_split_v3_design.md),
  as approved: company-tuned shares, pins and a target of about 120
  validation questions.

## 1. Summary

- **Split:**
  - 4,790 training passages (v2 had 5,554), 1,397 validation (v2 had 633)
    and 3,950 blocked (unchanged): 22.6% of usable passages in validation;
  - 2,187 training and 798 validation overlap groups.
- **Nothing moved except train → validation.**
  - 764 passages left training; no validation or blocked passage changed.
  - Every overlap group, edge, blocklist entry and evidence-reuse link is
    byte-identical to v2's, and so is the evidence index.
- **Existing data preserved, and re-checked rather than assumed:**
  - all 66 training candidates (`train_candidates_v2`);
  - all 16 validation questions (`validation_pairs_v2`).
- **Plan:** `val_pilot_v2` has 185 validation passages.
  - It is expected to give about 106 usable questions, so about 122 with
    the 16.
  - That is 4 batches of at most 50 requests, about 1.3–3 hours, about
    250,000 tokens.
  - It is an estimate from `val_pilot_v1`'s rates, not a measurement.
  - A band-calibrated check suggests about 92 (108 with the 16). Capped
    alternatives (`val_pilot_v2b`–`v2e`) are compared in
    [validation_plan_alternatives.md](validation_plan_alternatives.md),
    awaiting a choice.
- **120 questions is an initial target for choosing checkpoints.** It is not
  enough evidence for any broad claim that retrieval improved (§7).

## 2. What changed from v2

| | Where | Effect |
|---|---|---|
| Per-company validation shares | `split.validation_share_by_ticker` (`splits.py`) | AAPL 45%, AMZN 45%, TSLA 35%, WMT 35%, XOM 30%, PFE 25%, JPM 15% |
| Split pins | `split.pins`, a file written once by `pin-split` (`split_pins.py`, `pinning.py`) | 334 training passages (109 units, 1,901 passages with their units) stay train; all 633 v2 validation passages (219 units) stay validation |
| Context rules | `source_quality.context_rules` (`source_quality.py`) | Plans skip passages whose opening hides what a figure refers to (§5) |
| Plan options | `plan --one-per group --min-isolated-share F --exclude-validation-set CONFIG:VERSION` (`sampling_plan.py`, `data_cli.py`) | One passage per overlap group; a preference for passages whose evidence nothing outside validation holds; the earlier validation sets' units are skipped |
| Validation carry-over | `carry_over` in a validation-set config (`validation_set.py`) | Earlier validation questions are re-checked against the new preparation, one by one |

- **Everything else is v2's:** corpus, overlap, blocklist sources, evidence
  reuse, visible text, generation (`sec-qgen-v2`, validator
  `sec-qcheck-v3`, one validation question per passage) and leakage.
- **The new settings are recorded only when set.**
  - Re-running `prepare` on `data_prep_v2.yaml` reports the preparation
    unchanged.
  - Rebuilding `train_candidates_v1` and `validation_pairs_v1` also reports
    them unchanged.
  - Both written v2 plans (`pilot_v3`, `val_pilot_v1`) rebuild identically
    in memory.
- **One planner fix applies everywhere.**
  - Four overlap groups contain passages of two or more companies (shared
    boilerplate).
  - Company quotas now count only groups and units that no other company
    draws from, so a pick for one company can no longer leave another short.
  - The two v2 plans are unaffected.

```bash
python3 scripts/02_generate_queries.py pin-split --config configs/train/data_prep_v2.yaml \
    --candidates train_candidates_v1 --pairs --plan pilot_v3 --keep-validation \
    --out configs/train/split_pins_bge_base_ft_v3.json                         # written once
python3 scripts/02_generate_queries.py prepare --config configs/train/data_prep_v3.yaml
python3 scripts/02_generate_queries.py build-candidates --config configs/train/data_prep_v3.yaml \
    --candidates configs/train/candidates_v2.yaml
python3 scripts/02_generate_queries.py build-validation --config configs/train/data_prep_v3.yaml \
    --validation configs/train/validation_v2.yaml
```

## 3. The preparation

**Verified against v2 and against the design** (all hold):
- `overlap_groups.jsonl`, `overlap_edges.jsonl`, `blocklist.jsonl` and
  `reuse_links.jsonl` are byte-identical.
- `passages.jsonl` and `evidence_reuse.jsonl` are identical apart from each
  passage's split. The evidence-reuse index digest is unchanged.
- **Transitions:**

| v2 → v3 | Passages |
|---|---:|
| train → train | 4,790 |
| validation → validation | 633 |
| blocked → blocked | 3,950 |
| train → validation | 764 |

- Every pinned passage, and its whole unit, is on its pinned side.
- The split is passage-for-passage the one the design simulated.
- The manifest records the pins file (SHA-256 `838eef04…`) and the
  per-company shares. Its own SHA-256 is `21c2293a…`.

| Company | Train v2 → v3 | Validation v2 → v3 | Validation share of usable | Validation overlap groups |
|---|---:|---:|---:|---:|
| AAPL | 164 → 111 | 19 → 72 | 39% | 43 |
| AMZN | 228 → 143 | 27 → 112 | 44% | 68 |
| JPM | 2,784 → 2,630 | 320 → 474 | 15% | 247 |
| PFE | 793 → 650 | 92 → 235 | 27% | 125 |
| TSLA | 386 → 286 | 52 → 152 | 35% | 91 |
| WMT | 521 → 444 | 48 → 125 | 22% | 82 |
| XOM | 678 → 526 | 75 → 227 | 30% | 143 |
| **All** | **5,554 → 4,790** | **633 → 1,397** | **22.6%** | **798** |

- **Shares below their targets:** Walmart's 22% and Apple's 39% fall short
  because 59% and 45% of their usable passages sit in units pinned to
  train (Walmart's 224-passage unit among them). Units move whole.
- **Eligible sources:** every validation passage has at least 40 visible
  words (1,397). 4,775 training passages do.

| Validation passages | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 17 | 13 | 13 | 7 | 8 | 7 | 7 | – |
| AMZN | 21 | 17 | 16 | 11 | 13 | 16 | 18 | – |
| JPM | 59 | 69 | 57 | 59 | 88 | 71 | 71 | – |
| PFE | 29 | 41 | 38 | 30 | 33 | 31 | 33 | – |
| TSLA | 38 | 24 | 18 | 16 | 18 | 14 | 24 | – |
| WMT | – | 25 | 20 | 16 | 12 | 17 | 17 | 18 |
| XOM | 34 | 44 | 38 | 24 | 24 | 26 | 37 | – |
| **All** | 198 | 233 | 200 | 163 | 196 | 182 | 207 | 18 |

- **Training by fiscal year:** 2019 630, 2020 740, 2021 665, 2022 658,
  2023 666, 2024 684, 2025 676, 2026 71.
- **Protections are unchanged.**
  - Overlap groups and split units are whole.
  - The evaluation blocklist and its evidence-reuse expansion are v2's.
  - `NegativeGuard` still allows only same-split, non-blocked negatives and
    excludes every exact and uncertain equivalent. A passage that moved to
    validation can therefore never be a training negative.

## 4. Existing data, re-checked against v3

**Training candidates** (`train_candidates_v2`, manifest `fbd30a59…`).
- **Sources:** the same as `train_candidates_v1`, re-checked against v3:
  pilot_v3's 70 pairs from v2 and pilot_v2's 37 from v1.

| Source | Included | Excluded (each pair, with every reason) |
|---|---:|---|
| pilot_v3 | 51 of 70 | AI review: reject 6, revise 7. Evidence: a validation passage holds it exactly (`cross_split_evidence`) 4, a blocked passage holds it (`evaluation_evidence`) 2 |
| pilot_v2 | 15 of 37 | positive now blocked 15, `evaluation_evidence` 17, AI review: revise 3, reject 1 |

- **Same outcomes as in v2:** all 107 source pairs have exactly their
  `train_candidates_v1` outcome and reasons, the same 66 are included, and
  every candidate's question, answer, evidence, positive, split and
  equivalents are unchanged.
- **Origin:** only the recorded origin changes, to `carryover:pilot_v3` and
  `carryover:pilot_v2`.
- **Why nothing moved:** every positive and every training passage holding
  their evidence is pinned.

**Validation questions** (`validation_pairs_v2`, manifest `105f7888…`).
- **All 16 of `validation_pairs_v1` are included**, under their original
  IDs (`val_pilot_v1:validation_000002` … `_000049`), with origin
  `carryover:validation_pairs_v1`.
- **Each passes every check against v3:**
  - its positive is a validation passage whose group and unit lie wholly
    in validation;
  - no training pair or candidate uses it;
  - no train or blocked passage holds its evidence;
  - the validator passes;
  - it overlaps no evaluation or training question.
- **Their equivalents are unchanged,** and their AI decisions come with
  them (not gold).
- **Fail-closed reload:** the set reloads against the 66 v3 candidates with
  no isolation problem, and none of its passages may be a training
  negative.
- **An unpinned migration logs its losses** (tested): a carried question
  whose passage left validation is excluded as `positive_not_validation`,
  logged and reported. It is never moved or repaired.

## 5. Better validation passages

`val_pilot_v1`'s factual errors came from passages whose visible text starts
after the context a figure needs (docs/validation_pilot_v1_quality_report.md
§5). v3 turns on two source-quality rules for its plans:
- **`headless_table_start`:** the text starts inside a table (rows before any
  `[TABLE]`, or rows only), and no visible column heading names a period.
- **`fragment_opening`:** the text starts mid-sentence, and before its first
  financial figure it names no period (a year, or a month and day) or no
  entity (the company, "we", "our", "the Company", "the Firm").

**Measured on `val_pilot_v1`'s 50 passages:**
- The rules exclude 8 passages (4 of each).
- **Errors caught:** 3 of the 5 factual errors (07, 13 and 17).
- **Errors missed:** 16 (a subtotal read as one line) and 18 (Non-GAAP read
  as reported). Their column headings name periods, so no text rule sees
  them.
- **Good questions lost:** 2 of the 16 included (21 and 30), and 2 accepted
  ones that the evidence rule had removed anyway.

**On v3's validation sources:** 1,397 approved, 911 kept.

| Exclusions | Passages |
|---|---:|
| `headless_table_start` | 209 |
| `fragment_opening` | 150 |
| Earlier rules | 127 |

Tables and narrative fall alike, so the mixture holds.

**Plan-time preferences:**
- **Self-contained evidence first.** Plans score each passage by its
  isolated-evidence share: the share of its visible sentences that no train
  or blocked passage holds by any evidence-locator match.
- **The score predicts the evidence rule.** In `val_pilot_v1` the mean score
  was 41% and 44% of accepted questions passed the validation set's
  evidence rule.
- **How the plan uses it:** passages below 0.5 are skipped, and among
  balance ties the higher score wins.
- **Balance stays as before:** per-company quotas (at most 25%), each
  company's least-used fiscal year first, then table or narrative and
  section family.

## 6. The validation plan `val_pilot_v2`

```bash
python3 scripts/02_generate_queries.py plan --config configs/train/data_prep_v3.yaml --split validation \
    --name val_pilot_v2 --size 185 --method balanced_years --max-company-share 0.25 --one-per group \
    --min-isolated-share 0.5 --exclude-evidence-of train_candidates_v2 \
    --exclude-validation-set configs/train/data_prep_v2.yaml:validation_pairs_v1 \
    --exclude-validation-set configs/train/data_prep_v3.yaml:validation_pairs_v2   # written once
```

**Plan file:** `data/processed/train/bge_base_ft_v3/plans/validation.val_pilot_v2.json`,
SHA-256 `91648ca3…9bf9a`, seed 20261029.

**Exclusions** (118 overlap groups in 55 units):
- every unit of `val_pilot_v1`'s 50 passages, plus the 16 carried
  questions' equivalents;
- every validation unit that may hold a training candidate's evidence.

**Mean isolated share:** 0.79 (at least 0.5 throughout).

| Company | Passages | Expected usable | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 13 | 7.9 | 4 | 3 | 2 | – | 1 | 2 | 1 | – |
| AMZN | 16 | 9.7 | 6 | 3 | 1 | 2 | 1 | 1 | 2 | – |
| JPM | 37 | 22.0 | 5 | 5 | 5 | 6 | 6 | 5 | 5 | – |
| PFE | 28 | 14.7 | 10 | 3 | 4 | 3 | 3 | 2 | 3 | – |
| TSLA | 28 | 14.1 | 13 | 2 | 3 | 3 | 1 | 1 | 5 | – |
| WMT | 26 | 15.5 | – | 7 | 4 | 1 | 2 | 3 | 3 | 6 |
| XOM | 37 | 21.8 | 7 | 7 | 7 | 4 | 4 | 3 | 5 | – |
| **All** | **185** | **105.8** | 45 | 30 | 26 | 19 | 18 | 17 | 24 | 6 |

- **Years:** every company covers every fiscal year it still has eligible
  passages in. Apple has none left in 2022 after the exclusions and the
  0.5 threshold. JPMorgan and ExxonMobil take 20%
  each, the cap being 25%.
- **Kinds:** 112 narrative, 73 tables (39%).
- **Sections:** statements 85, risk 30, business 21, MD&A 13, other 36.
- **FY2019 is over-represented** (45 of 185, 24%), mostly Tesla (13 of 28)
  and Pfizer (10 of 28).
  - FY2019 passages are the most isolated: 69% score at least 0.5, against
    51–59% for other years.
  - Tesla's and Pfizer's other years run out of isolated, unused overlap
    groups first.
  - A per-company-year cap would fix it, but would move picks to JPMorgan
    and ExxonMobil. That decision is open (§9).
- **Several passages per unit:**
  - **What happens:** one passage per overlap group lets 19 split units give
    more than one passage (66 passages, at most 8 from one unit), mostly
    JPMorgan and ExxonMobil.
  - **The cost:** their questions are less independent than one per unit.
  - **Safeguard:** a later validation set records each question's
    equivalents, and its duplicate check removes repeated questions.

**Expected yield (an estimate):**
- **The estimate:** 105.8 usable questions, the sum over the plan of its
  isolated share × 0.86 (`val_pilot_v1`'s generator acceptance, 43/50) ×
  0.84 (its review acceptance among evidence-clean questions, 16/19).
- **Uncertainty:** those rates have 95% intervals of about 74–93% and
  62–95%, so expect roughly 80–130.
- **The context rules may raise the review acceptance,** but that is not
  assumed.
- **Probably optimistic:** at isolation scores of 0.5 or more, 11 of 16
  `val_pilot_v1` questions passed the evidence rule, against about 87%
  predicted. On those band rates the plan gives about 92
  ([validation_plan_alternatives.md](validation_plan_alternatives.md) §7).
- **With the 16 carried questions:** about 122.

**Requests and time:**

| | Estimate |
|---|---|
| Requests | 185, one per passage |
| Batches | 4 approved batches of at most 50: 50, 50, 50, 35 |
| Time | about 1.3 h at `val_pilot_v1`'s 2.3 requests per minute, at most about 3.1 h at the worst-case pacing of 1 per minute (768-token reservation against 1,000 output tokens per minute) |
| Tokens | about 1,344 per request (1,206 prompt, 137 completion in `val_pilot_v1`): about 249,000 |

- **Retries** count against each batch's cap. A retry leaves a passage for
  the next batch, so a fifth, small batch may be needed.
- **An early stop stays balanced:** plan order alternates companies.

**Dry run** (sends nothing, needs no key):

```bash
python3 scripts/02_generate_queries.py generate --config configs/train/data_prep_v3.yaml --split validation \
    --plan val_pilot_v2 --max-requests 50
```

**Each approved batch:** the same command with `--confirm-api-use`, which
needs `GROQ_API_KEY` in the environment. Completed passages are skipped on
the next batch.

**After generation:**
1. An AI review of every question, in the `data/validation_review/` format.
2. `build-validation` into a new version that carries `validation_pairs_v2`
   and adds `val_pilot_v2`.

## 7. Statistics: what is measured and what is not

**Measured** (docs/validation_split_v3_design.md §5): 82 synthetic questions
over the full corpus, base models only, paired per question.
- SD of per-question MRR@10 differences: 0.109 for a small change (BGE with
  or without its query instruction; 95% CI 0.06–0.15), 0.354 for BGE vs
  E5, 0.443 for BGE vs BM25.
- Intra-company correlation of the differences: 0.00–0.04.

**Assumptions:**
- Checkpoints of one fine-tuning run differ about like the small-change
  proxy.
- Synthetic validation questions behave like the 82 measured.
- Choosing a checkpoint needs a reliable ranking, not significance.

**Estimates:**
- With about 120 questions and an SD of 0.11–0.15, the better of two
  checkpoints 0.02 MRR@10 apart is chosen 93–98% of the time.
- The same questions detect a 0.03 difference with power 0.59–0.85.

**Unknowns:**
- How much real checkpoints differ: the smoke-test checkpoints are not
  available here and are trivial.
- Whether validation rankings follow evaluation rankings. On the synthetic
  questions BM25 beats BGE, so they reward lexical matching.
- The real yield of `val_pilot_v2`.

**What 120 questions are for:**
- They are an initial target for choosing among checkpoints of one run.
- They are not sufficient evidence for a broad scientific claim about
  retrieval improvement. Such a claim needs the evaluation set, with its
  own labels and uncertainty, and comparisons against strong baselines (BM25
  included).

## 8. Files

| File | Tracked | What |
|---|---|---|
| `configs/train/data_prep_v3.yaml` | yes | The v3 preparation |
| `configs/train/split_pins_bge_base_ft_v3.json` | yes | Pins from v2, every passage with its reasons (SHA-256 `838eef04…`) |
| `configs/train/candidates_v2.yaml` | yes | `train_candidates_v2` |
| `configs/train/validation_v2.yaml` | yes | `validation_pairs_v2` (carry-over) |
| `data/processed/train/bge_base_ft_v3/` | no (gitignored) | The preparation, `candidates/train_candidates_v2/`, `validation_sets/validation_pairs_v2/`, `plans/validation.val_pilot_v2.json` |

## 9. Risks and open items

1. **The yield is an estimate** from one 50-passage pilot (roughly 80–130
   new questions). If it falls short, a second plan from the remaining
   pool, with approval, is the fallback: 211 unplanned passages score at
   least 0.5, in fewer distinct groups, since one per group was already
   taken. A band-calibrated estimate gives about 92 new questions, not 106
   ([validation_plan_alternatives.md](validation_plan_alternatives.md) §7).
2. **FY2019 is over-represented** in the plan, especially Tesla and Pfizer
   (§6). Plans with per-company-year and per-unit caps are compared in
   [validation_plan_alternatives.md](validation_plan_alternatives.md); your
   decision.
3. **Several questions may come from one unit** (19 units, up to 8
   passages), which reduces independence. Effective sizes and a cap of 3
   per unit are in
   [validation_plan_alternatives.md](validation_plan_alternatives.md) §6.
4. **The context rules are heuristics.** They miss table-scope errors whose
   headings name periods (16, 18) and drop some good passages (2 of 16 in
   the pilot).
5. **Validation content is unrepresentative.** Isolated evidence and the
   context rules favor self-contained, less repeated content, unlike the
   evaluation pools, and synthetic questions favor lexical matching.
6. **Apple and Amazon remain small:** about 8 and 10 expected questions.
   Their training supply fell to 111 and 143 passages.
7. **The trainer still points at v2.** `configs/train/biencoder_bge_base*.yaml`
   uses `train_candidates_v1` in `bge_base_ft_v2`. Switching to v3's
   identical candidates is a separate, deliberate change.
8. **Groq risks:** availability of `qwen/qwen3.8-27b`, daily limits and rate
   limits.
