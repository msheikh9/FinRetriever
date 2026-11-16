# Gold evaluation set: methodology

> **Closed, incomplete (2026-11-15).** Gold set v1 human annotation was stopped after one graded question (q_0011, 45 judgments). The 100-question gold benchmark, the inter-annotator agreement analysis and the gold-set decision test were not completed, and no result comes from gold set v1. The methodology below describes the intended design; no hand-annotated benchmark exists. See [gold_set_v1_closure.md](gold_set_v1_closure.md).

This document defines how FinRetriever's hand-annotated evaluation set is
built and how results on it must be read. It covers v1 (target: 100
questions), which will later grow to 300. The tooling is
`scripts/03_annotate_gold_set.py` (code in `src/finretriever/eval_set/`); the
configuration is `configs/eval/gold_set_v1.yaml`.

**Status:** the annotation infrastructure exists and is tested. A 10-question
pilot (`q_0001`–`q_0010`, AI-assisted questions approved by a human reviewer;
see §2) has candidate pools (579 passages). Manual grading of the full pilot
pools was stopped after 4 human grades (on `q_0001`), which are kept. The
other 575 pooled passages carry **AI-proposed development labels**, stored
apart from human judgments. A 30-passage blind audit of those labels is
complete (25/30 exact agreement, 29/30 relevant-vs-not, κ 0.750). It is
**AI-assisted**: the reviewer consulted ChatGPT for some judgments, so it is
not an independent human validation (§11). The rubric clarifications that
audit prompted are adopted as rubric v1.1 (§3), and the development label
version `pilot_v2` applies them. BM25 and the zero-shot BGE and E5
baselines have been run on it over the whole corpus and restricted to the
question's company. The pools were then deepened with those six runs' top
10s, and the 117 passages this added were AI-labelled (`pilot_v3`), so every
run's top 10 is judged. All these numbers are provisional development
metrics (§11). There are no human gold qrels yet: `data/eval/gold_queries.jsonl` and
`data/eval/gold_qrels.jsonl` stay empty until an export follows real human
grading.

The set exists to compare BM25, zero-shot BGE, zero-shot E5 and, later, the
fine-tuned FinRetriever model on the same questions. Every rule below serves
that comparison: it must not favour one retriever by construction, and its
gaps must be visible.

## 1. Corpus provenance

All judgments refer to one frozen corpus:

| | |
|---|---|
| Corpus | `data/processed/corpus_v1.jsonl` |
| Version | `corpus_v1@1.2` |
| SHA-256 | `ff9dadf37bb9820e2c87df70a411fd189bb02776609e5ec06173dd03f2344004` |
| Content | 10,137 passages from 49 10-K filings of AAPL, AMZN, JPM, PFE, TSLA, WMT, XOM |

Passage IDs are positional (filing, section index, chunk index), not derived
from the text. A rebuilt corpus can therefore put different text behind an
existing ID. To prevent silent mismatches:

1. Every command re-hashes the corpus and compares it with its manifest
   (`load_corpus_identity`) and with the SHA-256 and version pinned in the
   config. Any difference stops the command (exit code 2) before anything is
   read or written.
2. The annotation workspace records the corpus SHA-256, corpus version and
   annotation version when it is created (`workspace.json`). Opening it
   against any other corpus fails.
3. Every pooled and judged passage ID is checked against the corpus when the
   workspace is opened and again at export.
4. The export writes `gold_eval.metadata.json` (`corpus_sha256`,
   `corpus_version`, `annotation_version`). Evaluation code must call
   `verify_evaluation_corpus` with it before scoring.

If the corpus ever has to be rebuilt, the gold set must be re-validated:
judgments are not transferred automatically.

## 2. Writing questions

### What a question is

Write the question an analyst, investor or credit researcher would actually
type when looking for information in these filings. Questions are written by
a person, or drafted with AI assistance and approved by a person (see
*AI-assisted questions* below). They are never generated in bulk and never
derived automatically from passages.

### Independence from the passages

Questions written while looking at a passage tend to reuse its wording. That
makes lexical retrieval look better than it is and rewards memorising
surface forms. Therefore:

- **Write first, search later.** Draft the question from an information need
  (a company, a topic and a period you want to know about) before running
  any search.
- **Record how the question was written.** Use `--authoring independent` for
  questions drafted before reading candidate passages, and
  `--authoring passage_inspired` when the idea came from reading a passage.
  Passage-inspired questions are allowed but must use your own wording, not
  the passage's. The status report shows the mix; keep passage-inspired
  questions under about 20%.
- Avoid copying distinctive phrases from filings (for example "Organizing
  for Growth", "Identified Items") unless the question is explicitly about
  that defined term.

### AI-assisted questions

A question drafted by an AI model is recorded as exactly that:

- `--ai-assisted`, with the model as author (`--annotator claude`) and the
  person who approved the text as `--reviewed-by`. The tool refuses an
  AI-assisted question without a reviewer.
- `--authoring passage_inspired`: the model drafted it after searching the
  corpus to check that it can be answered. The tool refuses `independent`
  for AI-assisted questions.
- `--notes` saying how the question was drafted and when it was approved.

AI-assisted questions count towards the ~20% passage-inspired share. The v1
pilot (`q_0001`–`q_0010`) is entirely AI-assisted, so the remaining questions
should be written by people. `status` shows the AI-assisted count, and the
export carries `ai_assisted` and `reviewed_by` in each query's metadata, so
results can be reported with and without these questions. A model never
records a relevance judgment in the human workspace: the tool refuses the
configured AI identities (`ai_annotators`) for `annotate` and `audit`, and
the export refuses any judgment recorded under them. AI relevance labels
exist only as separate development data (§11).

### Coverage targets for v1 (100 questions)

Each question records a `topic`, a `question_type`, a `comparison`, the
companies it concerns (`--ticker`) and, where relevant, fiscal years
(`--year`). Aim for roughly:

| Dimension | Values (target share) |
|---|---|
| topic | risk_factors (~20%), mdna (~20%), financial_statements (~20%), debt_liquidity (~15%), revenue_profitability (~20%), other (~5%) |
| question_type | exact_fact (~30%): a figure or fact stated in one place; semantic (~30%): a concept described in different words than the question; table (~25%): answered by financial-table rows; multi_passage (~15%): needs two or more passages |
| comparison | cross_year (≥15%): change over fiscal years; cross_company (≥10%): the same financial concept across companies, where terminology differs (e.g. "allowance for credit losses" vs "inventory reserves") |
| companies | every company in at least 10 questions; JPM and XOM appended-report sections included |

These are targets for the mix, not quotas to fill with weak questions.
`status` reports the current distribution.

### Practical rules

- Name the company unless the question is deliberately cross-company.
- Name the fiscal year when the answer depends on it. Note that a fiscal
  year's figures also appear in the next two years' 10-Ks as comparatives.
- One information need per question. Split compound questions.
- Withdraw (`withdraw q_NNNN`) rather than delete a question that turns out to
  be ambiguous. IDs are never reused.

## 3. Relevance grading rubric

Grade each candidate for the question as written, using the passage text
alone (the filing and section shown help to locate it).

| Grade | Meaning | Financial-filing guidance |
|---|---|---|
| **2: directly relevant** | The passage answers the question, or the part of it the passage could answer for a multi-passage question. | It states the requested figure for the requested period and company (including as a prior-year comparative column), or explains the asked-about risk, policy or driver specifically. |
| **1: partially relevant** | Useful but incomplete: the right company, period and topic, with only part of the answer. | A summary sentence without the requested figure; a table that contains the line item but not the requested breakdown; context needed to interpret the answer. |
| **0: not relevant** | Does not help answer the question. | Same keywords in another context (e.g. "credit" in a credit-card risk factor when the question is about a credit facility); a different company; boilerplate. |
| **skip** | You looked but cannot decide. | Ambiguous scope, needs domain knowledge you lack, or the passage is cut mid-table. Add a note (`s <note>`). |

Rules:

- **A skip is not a 0.** Skipped passages stay unjudged and never appear in
  the qrels. Use skip whenever you would otherwise guess.
- Judge relevance, not truth: a passage that answers the question is
  relevant even if a later filing restates the number.
- A cross-reference stub ("see Note 7") is 0 unless the question asks where
  the information is.
- Notes can follow any grade (`1 covers FY2023 only`) and are kept in the
  history.

### Temporal relevance

Judge every passage against the company and the fiscal year (or years) the
question names, not against the topic alone.

- **Repeated text is not assumed relevant.** Filings repeat risk factors and
  policy descriptions from year to year. A passage from a different filing
  year is not relevant just because its wording matches the requested
  year's disclosure. A question about the fiscal 2024 report is answered by
  the fiscal 2024 report; the same paragraph in the fiscal 2023 10-K is 0
  unless the question's wording makes other years relevant.
- **Other filings can still hold the requested period.** A later 10-K that
  reports the requested year's figure, for example in a prior-year
  comparative column, is graded on that content (often 2).
- **Information about other years** is 0, unless the question asks about a
  change or comparison that it helps answer (1 or 2).
- **Multi-passage and cross-year questions:** grade each passage by its
  contribution to the requested answer. It is a 2 if it supplies a required
  part (for example one requested year's figure), a 1 if it only supports or
  interprets the answer, and a 0 otherwise. A passage does not need to answer
  the whole question to be a 2.

### Rubric v1.1 clarifications (adopted 2026-10-28)

These were proposed after the pilot audit
([audit_review_pilot_v1.md](audit_review_pilot_v1.md) §5) and adopted with
the options shown. They clarify the rules above. No existing human judgment
was changed, and labels made under them record the rubric version (`v1.1`).

- **P1, open questions.** Some questions ask for a set of risks, reasons,
  offerings or mechanisms. For these, a passage that explicitly states one
  element of the answer, from the requested company and report or period, is
  **2**, even if most of the passage is about something else. It is **1** if
  it only names the topic, is cut off before the statement, or needs outside
  information to connect to the question.
- **P2, passage-alone evidence.** For a figure, the requested period and
  basis must be identifiable from the passage itself: its text, table
  headers and any footnote inside the chunk. The filing shown with the
  passage identifies the company and the report. A figure whose period or
  basis is defined only outside the chunk is **1**. An example is an
  "average" whose three-month basis is given in a footnote in the next
  chunk. The explanation of a grade may cite only what the passage says.
- **P3, comparatives.** A later 10-K that states the requested figure for the
  requested period is graded on content, like the original filing: 2 if
  complete under P2, 1 if partial. It does not matter how many years later
  it was filed.
- **P4, repeated text from another filing year (option A).** When the
  question names a report or year, another filing year's copy of a
  qualitative disclosure is **0**, even when it is verbatim.
- **P5, closed multi-part questions (option A).** Decide by the evidence,
  not the `question_type`. Sometimes no single filing can contain the whole
  answer, for example five years of figures when a 10-K shows three. Then a
  passage that states a complete required evidence unit (the requested
  figures for at least one requested period) may be **2**. If the full
  answer is normally reported in one table, a passage with only some of its
  elements is **1**.
- **P6, notes and assistance.** Write a note for every grade 1, every 0 given
  under P4 and every comparative, in annotation and in audits. Record any
  assistance (another model, a colleague) in the note of each affected
  judgment. The tool does not enforce this yet.

## 4. Candidate pooling

Annotators judge only the passages in a question's pool. The pool therefore
decides which relevant passages can ever count. Four sources exist, and each
pooled passage records every source that proposed it (method, rank, score
and parameters), so a pool can be audited and rebuilt:

| Source | What it does | Default |
|---|---|---|
| `bm25` | The project's BM25 retriever over the whole frozen corpus, restricted to the question's companies. It runs on the question text and on each reformulation the annotator writes (`--reformulation`). The pure-Python backend is always used, so pools do not depend on optional packages. Tokens ignore punctuation, so a question's "supplier?" matches "supplier." in a passage. | top 30 per search |
| `manual` | Passages found with `search`/`show` or by reading the filing (`--passage-id`, with `--manual-note` saying how, e.g. "AI-assisted answerability inspection"). Adding a passage to the pool is not a judgment. | — |
| `stratified` | A seeded sample from every in-scope filing, so years and companies that BM25 ranks low are still seen. Use it for cross-year questions. | 0 per filing |
| `random` | A seeded uniform sample of in-scope passages: a rough check on what lexical search misses. | 5 |

Rules:

- **No model-based sources.** Dense retrievers do not select candidates, and
  the fine-tuned FinRetriever model never may: a pool built by the system
  under test favours that system. The schema rejects any other source.
- **Reformulate.** For semantic and cross-company questions, add at least two
  reformulations using the vocabulary a filing would use (e.g. "allowance
  for loan losses", "provision for credit losses").
- **Years are not used to filter by default**, because later filings repeat
  earlier years' figures. `--use-years` exists for questions about a single
  filing.
- Pools are shown in the order candidates were added (BM25 rank first), and
  the source and rank are displayed. This is a known position bias (see §7).

## 5. Annotation workflow

```bash
# once: create the workspace bound to the frozen corpus
python scripts/03_annotate_gold_set.py init --annotator XY

# per question
python scripts/03_annotate_gold_set.py add-question --annotator XY \
    --text "How much undrawn revolving credit capacity did Tesla report?" \
    --topic debt_liquidity --type exact_fact --ticker TSLA --authoring independent
python scripts/03_annotate_gold_set.py pool q_0001 --annotator XY \
    --reformulation "unused committed amounts credit facilities"
python scripts/03_annotate_gold_set.py search "revolving credit facility availability" --ticker TSLA
python scripts/03_annotate_gold_set.py pool q_0001 --annotator XY --passage-id <id from search>
python scripts/03_annotate_gold_set.py annotate q_0001 --annotator XY --hide-scores

# any time
python scripts/03_annotate_gold_set.py status
python scripts/03_annotate_gold_set.py history q_0001
python scripts/03_annotate_gold_set.py export
```

(`FINRETRIEVER_ANNOTATOR=XY` can replace `--annotator`. The question above
only illustrates the command; it is not part of the gold set.)

In `annotate`, each candidate shows the query ID and question, the passage
ID, the company, filing date and accession number, the section (Item number,
or the report heading for JPM/XOM appended sections), every pool source with
its rank and score, the current judgment and the passage text.
`--hide-scores` hides ranks and scores while grading (use it by default).
It only changes the display: candidates keep their order and their stored
provenance. `--blind` (added 2026-11-12, required for gold set v1) goes
further: it hides every pool source and shows the candidates in an order
shuffled per question, the same in every session, again without changing
anything stored. `leakage-check` compares the written questions with the
training and validation questions (texts only) before pooling. See
[gold_set_v1_protocol.md](gold_set_v1_protocol.md). Commands: `0`/`1`/`2` grade, `s [note]` skip, `c` clear,
`n`/`p`/`g N`/`u` navigate, `f` full text, `h` history, `done` mark
complete, `q` quit.

- **Saving and resuming:** every grade, skip and clear is written to
  `history.jsonl` immediately. A session can stop at any time, and the next
  `annotate` resumes at the first pending candidate.
- **Editing:** navigate back (`p`, `g N`) and grade again. The earlier grade
  stays in the history.
- **Completing:** `done` (or `complete q_NNNN`) requires every candidate to be
  graded or skipped. A complete question is frozen until `reopen`.

Workspace files (`data/eval/annotation/`): `workspace.json` (corpus binding),
`questions.jsonl`, `pool.jsonl` (candidates with provenance) and
`history.jsonl` (append-only revision log; current judgments are its replay).
They are part of the audit trail and should be committed with the export.

## 6. Integrity safeguards

- **Corpus binding:** see §1.
- **Unique questions:** IDs are assigned sequentially (`q_0001`, ...) and
  never reused. A question whose normalised text repeats an existing one is
  refused.
- **Valid judgments:** grades must be 0, 1 or 2. Only pooled passages of an
  open question can be judged, so every judgment has a recorded selection
  path.
- **No conflicting judgments:** each history event records the judgment it
  replaces. Replaying the log on open rejects out-of-order, duplicated or
  edited events. A store refuses to write if `history.jsonl` changed since it
  was opened, so two sessions cannot interleave edits.
- **Deterministic export:** the export has no timestamps, and the same
  judgments always produce byte-identical files.

## 7. Judgment completeness and its limits

The qrels are **pooled judgments**. They are incomplete by design:

- A passage that was never pooled was never judged. It may be relevant.
- Skipped and pending pool entries are unjudged too.
- The export contains only graded passages (0, 1, 2). Unjudged passages are
  absent, **not** recorded as 0.

The metrics (`finretriever.evaluation.metrics`), like trec_eval, still score
any unjudged passage a retriever returns as non-relevant. That is an
assumption, not a judgment. Its main consequence: a retriever that finds
relevant passages BM25 did not pool (likely for dense models on paraphrased
questions) is under-scored. To keep this visible:

- `status` and the export report show, per question, the pool size, graded,
  skipped and pending counts, grade counts (0/1/2), the pool sources, and
  flags: `no_positive`, `small_pool` (fewer than 10 candidates),
  `pending_candidates`, `skipped_candidates`, `empty_pool`.
- Questions without a positive judgment are not exported: they score 0 for
  every retriever and cannot rank them. The report lists them so they can be
  re-pooled or withdrawn.
- Every reported metric must come with **judged@10**
  (`qrels_builder.judged_at_k`): the share of each retriever's top 10 that
  was judged. If a retriever's judged@10 is clearly lower than BM25's, its
  score is a lower bound. Its unjudged top-ranked passages should then be
  judged (pool deepening) before conclusions are drawn.
- Pool deepening may add the top-k of the *zero-shot* baselines after their
  first evaluation, recorded as manual candidates whose `--manual-note` names
  the run. It must never use the fine-tuned model's output. Development
  pools are deepened separately, outside the annotation workspace, with
  each run recorded as a source (`expand-pool`, §11: `pilot_v3`).

### How incomplete judgments bias nDCG and Recall

Let *R* be the passages truly relevant to a question and *J* the relevant
passages that were actually judged. Pooling guarantees only that J ⊆ R.

- **nDCG@10, numerator (DCG).** An unjudged passage in a retriever's top 10
  earns gain 0. A retriever that finds relevant passages outside the pool is
  scored as if it had returned irrelevant ones. The penalty falls on the
  systems least like the pooling sources (BM25 here), which is the dense
  models this project exists to evaluate.
- **nDCG@10, denominator (IDCG).** The ideal ranking is built from J only.
  When R is larger than J, IDCG is too small. That inflates nDCG for systems
  that retrieve the known positives, and the inflation is larger for
  questions with few judged positives. Absolute nDCG values are therefore
  not comparable across pools of different depth.
- **Recall@100.** The denominator is |J|, not |R|. Recall is measured against
  the pool, not against the truth, so it is overstated for every system. A
  system that retrieves unjudged relevant passages gets no credit for them.
  Recall differences between systems that reach different parts of the
  corpus are therefore biased towards the pooling sources.
- **MRR@10.** An unjudged relevant passage ranked above the first judged one
  is treated as irrelevant, so MRR is a lower bound for systems that surface
  unpooled answers.
- **Noisy labels add a second bias.** If some judgments come from a fallible
  labeller (for example the AI development labels of §11), false negatives
  behave like unjudged positives (they lower DCG and recall for the system
  that retrieves them) and false positives inflate both. The audit (§11)
  estimates how often each happens.

Consequences for reporting: compare systems on the same judged pool, report
judged@10 with every metric, prefer relative comparisons to absolute values,
and deepen the pool when a system's judged@10 falls clearly below BM25's. A
condensed-list score, computed after removing unjudged passages from each
ranking, is a useful sensitivity check. Named evaluation sets report it as
`ndcg@10_judged_only` and `mrr@10_judged_only` (§11).

## 8. Sources of evaluation bias

| Bias | Effect | Mitigation |
|---|---|---|
| BM25 pooling | Relevant passages without lexical overlap are missed; favours lexical retrievers. | Reformulations, manual lookup, stratified and random samples, judged@k reporting, pool deepening with zero-shot runs. |
| Author wording | Questions echo filing language, which favours BM25. | Write-first rule, `authoring` field, cap on passage-inspired questions. |
| AI-drafted questions | A model that searched the corpus may favour questions it knows are answerable, in phrasings close to the filings. | `ai_assisted` and `reviewed_by` recorded, human approval, counted in the passage-inspired share, results reportable with and without them. |
| Single annotator | Grades reflect one person's reading; no agreement estimate yet. | Rubric, notes, full history. A second annotator re-grades about 20% of questions blind, and Cohen's κ is reported before v1 is final (`regrade-init`, `audit`, `regrade-report`; [gold_set_v1_protocol.md](gold_set_v1_protocol.md) §2.4). |
| AI development labels | One model labelled 575 pilot passages; its errors are systematic, not random, and it wrote the questions it labelled. | Kept apart from human judgments and never exported (§11); blind 30-passage audit with agreement and confusion reported (AI-assisted in `pilot_v1`, so not independent); final results use human judgments only. |
| Position and rank display | Showing BM25 rank and ordering by it may bias grades towards the top. | `--hide-scores` hides ranks and scores; grade from the text; skip when unsure. Candidates are still ordered BM25-first. `--blind` (gold set v1) hides the sources too and shuffles the order. |
| Repeated text across years | The same risk factor appears in seven filings, so a question can have many equally relevant passages. | Judge each on content; nDCG uses graded relevance; the report shows grade counts. |
| Topic and company imbalance | Results may reflect the question mix more than retriever quality. | Distribution targets (§2) and per-dimension counts in `status`. |
| Chunk boundaries | An answer split across passages is only partly contained in each. | Grade 1 for partial passages; `multi_passage` questions. |
| Small sample | 100 questions give wide confidence intervals. | Report per-query scores and bootstrap intervals; grow to 300. |

## 9. Data leakage prevention

- Gold questions live only in `data/eval/` (`source: "gold"`). Synthetic
  training queries (`scripts/02_generate_queries.py`, not yet implemented)
  must be written elsewhere and must never read `data/eval/`.
- Before any training run, synthetic queries must pass
  `qrels_builder.check_no_gold_leakage`, which rejects texts whose word sets
  overlap a gold question by 80% or more. This lexical check catches copies
  and light rewordings, not deep paraphrases, so the rule above (no access to
  gold files) is the primary safeguard.
- Gold questions and their paraphrases are never used as training or
  validation data, and never as prompts or examples for query generation.
- Hyperparameters and model selection use a separate development set, not
  the gold set. The gold set is scored once per finished model.
- Gold passages may appear as training passages: the corpus is shared by
  design. What must not leak is the question–passage pairing.
- The same rules cover every question in `data/eval/`, whatever tier its
  labels belong to (§11): the annotation workspace, the AI development labels
  in `data/eval/dev_ai/` and the audit in `data/eval/audit/`. The AI labels
  are question–passage pairings for evaluation questions, so they are never
  training data, never hard negatives, and never examples for query
  generation. The leakage check must be run against all questions in
  `data/eval/annotation/questions.jsonl`, not only the exported ones.

## 10. Reproducing results

A result on the gold set is reproducible from:

1. the corpus with the SHA-256 above (rebuilt with `make corpus
   CORPUS_ARGS=--strict` from the same raw filings and code; the manifest
   records the builder source hashes);
2. `data/eval/gold_queries.jsonl`, `gold_qrels.jsonl`,
   `gold_eval.metadata.json` and the workspace files, all committed;
3. the retriever configuration in `configs/eval/benchmark.yaml`.

Evaluation must first call `verify_evaluation_corpus(gold_eval.metadata.json,
corpus)`, then score with `finretriever.evaluation.metrics`, and report
nDCG@10 (headline), MRR@10, Recall@100 and judged@10. The report must state
the annotation version and the number of exported questions. Results on a
partial gold set must say so.

**The metric definition is authoritative and independent of the environment**
(2026-11-12). The nDCG gain is `2**grade - 1` and the discount
`1 / log2(rank + 1)`. Every metric is computed by the project's own kernels,
and pytrec_eval is never used, even when installed. Gold set v1's comparison
of A1 against A0 is frozen in [gold_set_v1_protocol.md](gold_set_v1_protocol.md).

## 11. AI-assisted development labels and the independent human audit

Grading every pooled passage by hand (579 passages for the 10-question pilot)
is slow, and most candidates are clearly irrelevant. The pilot therefore
uses AI-proposed labels as **development data**, checks them with a **blind
human audit**, and keeps the final benchmark for human judgments only.

### Three tiers, never mixed

| Tier | Where | Labels from | Used for | Never used for |
|---|---|---|---|---|
| 1. AI development labels | `data/eval/dev_ai/<set>/` | An AI model (`claude-opus-5-5` for `pilot_v1`) | Building and debugging the evaluation pipeline; rough, clearly labelled development comparisons of the zero-shot baselines | Gold qrels, reported benchmark results, training, hard negatives, query-generation prompts |
| 2. Independent human audit | `data/eval/audit/<set>/` | A person, blind to the AI labels | Measuring how far the tier-1 labels can be trusted | Gold qrels, training |
| 3. Human final evaluation | `data/eval/annotation/` → `gold_queries.jsonl`, `gold_qrels.jsonl` | People, through `annotate` | The benchmark reported for every retriever | Training, tuning |

A tier-2 audit is **independent** only if the grader used no AI assistance.
The `pilot_v1` audit does not meet this condition (see *Pilot audit results*
below).

How the tiers are kept apart:

- Each tier lives in its own directory. A proposal set cannot be created in
  the annotation workspace directory, and an audit refuses both the
  workspace and the proposal directory; it is a separate workspace.
- The export reads only the annotation workspace. It refuses audit
  workspaces and any judgment recorded under an AI identity
  (`ai_annotators` in the config), and AI proposals are never read by it.
- `annotate` and `audit` refuse AI identities, so an AI label cannot be
  entered as a human grade.
- An AI proposal is refused for a passage that already has a human grade,
  and nothing copies proposals into the human workspace. AI labels never
  replace human judgments; a person can grade a passage that has a proposal,
  and the human grade is the only one that can reach the gold set.
- Tier-1 and tier-2 numbers are always reported as such ("AI-labelled
  development set", "audit agreement"), never as gold results.

The pilot questions (`q_0001`–`q_0010`) are AI-drafted and their passages
were AI-labelled, so they are development questions. If any of them later
enters the final benchmark, every candidate must be graded by a person, and
results must be reported with and without them. The final benchmark should
be built mainly from new, human-written questions.

### How the pilot proposals were produced (`data/eval/dev_ai/pilot_v1`)

- **Coverage:** every pooled passage of `q_0001`–`q_0010` without a human
  grade: 575 proposals. The 4 human grades on `q_0001` are untouched and
  were not proposed for.
- **Method `llm-full-passage-read-v1`:** the model read the full text of
  each passage, in batches of 12–16 per question, with the question, company,
  filing year and section visible and retrieval ranks, scores and sources
  hidden. For each passage it gave a grade (0/1/2), a confidence
  (`low`/`medium`/`high`) and a one-sentence explanation tied to the passage
  text. Where the date of a comparative table was unclear, other corpus
  passages were searched to identify it.
- **Rules applied:** the rubric and temporal rules of §3. In practice:
  repeated text from another filing year is 0 (confidence `medium` when the
  topic matches); a later filing's prior-year comparative that contains the
  requested figure is 2 with confidence `medium`; a total without the
  requested split, a forecast, or context that explains the answer is 1;
  for multi-passage questions a passage that states a requested year's figure
  is 2.
- **Provenance:** `proposals.meta.json` records the corpus SHA-256, corpus
  version and annotation version (the workspace's binding), the source
  workspace, the model, the method and its description, the creator and the
  creation time, and a notice that these are not human judgments. Each line
  of `proposals.jsonl` repeats the model and method and has its own
  timestamp. The set is append-only, and an audit pins its SHA-256.

The resulting labels are provisional. They come from one model, which also
drafted the questions, so its errors can be correlated with the questions'
wording. The audit measures the error rate but cannot remove it.

### The blind audit (`data/eval/audit/pilot_v1`)

**Sample.** 30 proposals, 3 per question. Within a question the slots go
round-robin over the proposed grades 2, 1, 0, so every proposed class that
exists for a question is represented even though most proposals are 0. The
draw is seeded (`audit_seed: 20261027`) and reproducible: passages within a
(question, proposed grade) stratum are drawn with
`random.Random(f"{seed}:{query_id}:{grade}")`, and the display order across
questions is shuffled with `random.Random(f"{seed}:order")`. The sample and
its rules are recorded in `audit.json`, together with the SHA-256 of the
proposals it audits.

**Blindness.** The audit workspace holds only the questions, the 30 passage
IDs and a neutral source (`stratified`, "blind audit sample"). It contains no
AI grade, confidence or explanation, and the audit screen shows no ranks,
scores or sources. `audit-status` shows progress only. The comparison with
the AI labels is computed only by `audit-report`, which refuses to run while
any item is pending. The auditor should not open `data/eval/dev_ai/` before
finishing.

**Grading.**

```bash
python3 scripts/03_annotate_gold_set.py audit --annotator XY
```

Use the §3 rubric. Commands: `0`/`1`/`2 [note]` grade, `s [note]` skip,
`c` clear, `n`/`p` move, `g N` go to item N (to correct an earlier grade),
`u` next ungraded, `f` full text, `?` help, `q` quit. Every action is saved
at once; rerunning the command resumes at the first ungraded item, and a
corrected grade keeps the earlier one in the audit history.
`python3 scripts/03_annotate_gold_set.py audit-status` shows progress.

**Report.** When every item is graded or skipped:

```bash
python3 scripts/03_annotate_gold_set.py audit-report   # add --json for the full record
```

It writes `agreement_report.json` with: raw agreement (exact grade),
relevant-vs-not agreement (grade ≥ 1), unweighted Cohen's κ, a
stratum-weighted agreement (each stratum weighted by its share of all
proposals, to estimate agreement over the whole proposal set instead of the
balanced sample), the confusion matrix (human × AI grade), agreement broken
down by question type, human grade, AI grade, AI confidence and question,
and the list of disagreements with both grades. Skipped items are counted
but excluded from agreement. The report never changes either set of labels.

**Reading it.** 30 items give wide intervals: one disagreement moves raw
agreement by more than 3 points, and per-class counts are small. Treat the
audit as a check for systematic problems (for example the AI calling
comparative-column passages 2 when the human says 1), not as a precise error
rate. Consider the AI labels usable for development only if the audit
shows no systematic false negatives on human-relevant passages; otherwise
revise the method and draw a new audit from a new proposal set.

### Pilot audit results (`pilot_v1`)

All 30 items were graded by the human reviewer, with no skips, corrections or notes.

| Statistic | Value | 95% interval |
|---|---|---|
| Exact agreement | 25/30 = 83.3% | 66.4–92.7% |
| Relevant-vs-not agreement | 29/30 = 96.7% | 83.3–99.4% |
| Cohen's κ (unweighted) | 0.750 | 0.53–0.95 (bootstrap) |
| Stratum-weighted exact agreement | 87.3% | — |

**Independence limitation.** The reviewer consulted ChatGPT for some
judgments. Which items, and how, was not recorded.

- The audit therefore compares Claude's labels with an AI-assisted human,
  not with an independent one.
- Agreement between two language models can be inflated by shared ways of
  reading, so these figures are likely optimistic by an unknown amount.
- They must be quoted with this qualifier and do not make the AI labels
  "human-validated".
- The audit still stands as a blind, reproducible check, and its
  disagreements are informative.

**Findings.**

- Four of the five disagreements are between grades 1 and 2.
- The one relevant-vs-not disagreement (#17) concerns repeated text from
  another filing year. Whether it is an AI false negative depends on a rubric
  decision (P4).
- Three disagreements trace to the AI. Two were explicit statements
  marked down for being a small part of the passage. One was a grade that
  relied on a footnote outside the passage.
- Two trace to departures from written rules that need restating.

The AI labels are judged usable for preliminary, clearly labelled
development experiments (binary relevance first, judged@10 always, with
rule-variant sensitivity checks). They are not usable for benchmark claims.
The full review, including each disagreement, the proposed clarifications
and the human evaluation still required, is
[audit_review_pilot_v1.md](audit_review_pilot_v1.md).

### Development label versions (`pilot_v2`)

A development label version is derived from immutable inputs and never
edited by hand. `pilot_v2` (`data/eval/dev_ai/pilot_v2/`) comes from the
`pilot_v1` proposals and the 4 human grades under rubric v1.1. The spec that
derives it, `configs/eval/dev_labels_pilot_v2.yaml` (kept in the private archive),
is pinned by its SHA-256.

- **Inputs stay immutable.** `labels.meta.json` records the SHA-256 of the
  `pilot_v1` proposals, the audit and its report, and the workspace history.
  The export refuses to run if the proposals or the audit change afterwards.
- **Only deterministic rules change grades, and only AI grades.** P2 is
  applied as a pattern check per figure question: an AI grade-2 label whose
  passage does not state the requested period or basis becomes 1. Each
  change is recorded in `changes.jsonl`, with the previous and new grade,
  the rule, the reason, the check, the evidence and the original explanation.
  A grade that differs from its base without a change record is refused.
- **Judgment calls are flagged, not applied.** Every AI grade-1 label on an
  open question is flagged under P1. Curated flags cover explanation
  grounding (P2) and evidence units (P5). Flags never change a grade.
- **Audit grades are evidence, not labels.** Every audited label whose
  AI-assisted human grade differs from its base gets an explicit
  disposition: changed by a rule, flagged, or kept under a named rule. The
  derivation fails if a disagreement has none.
- **Nothing is invented.** Pool entries with neither a human grade nor an AI
  proposal stay unlabelled. Unpooled passages stay unjudged.

`pilot_v2`: 579 labels (575 AI proposals, 4 human grades), with 494 at grade
0, 51 at grade 1 and 34 at grade 2.

- 3 grades changed under P2: the three JPMorgan three-year-highlights rows
  whose fourth-quarter basis is in a footnote outside the passage.
- 12 flags are open: 9 under P1, 2 under P2 and 1 under P5.
- The five audit disagreements: #26 changed (P2), #13 and #21 flagged (P1),
  #17 kept (P4) and #20 kept (P5).
- Every question has positive and grade-2 evidence.

### Development evaluation workflow

```bash
python3 scripts/03_annotate_gold_set.py dev-labels --spec configs/eval/dev_labels_pilot_v2.yaml
python3 scripts/03_annotate_gold_set.py export-dev --labels data/eval/dev_ai/pilot_v2
python3 scripts/04_run_baselines.py --config configs/eval/benchmark.yaml --eval-set dev_pilot_v2
```

`export-dev` writes `dev_queries.jsonl` (`source: dev_ai`), `dev_qrels.jsonl`
and `dev_qrels.provenance.jsonl`. The provenance file records, per qrel, the
label source, whether it is AI-generated, its assistance and whether it
changed. The export also writes `dev_eval.metadata.json` (the corpus binding
read by `verify_evaluation_corpus`) and `dev_eval.report.json`, which holds
the status, provenance, audit independence, per-question coverage, exceptions
and file hashes.

The export is reproducible. It refuses conflicting labels, unknown passages,
the gold export directory, the workspace and the immutable inputs.

`--eval-set` runs a named set from `configs/eval/benchmark.yaml`:

- The corpus is verified against the set's metadata, and the qrels against
  the SHA-256 in the export report, before anything is scored.
- Each query gets its own record: nDCG@10, MRR@10, Recall@100, judged@10 and
  the judged-only sensitivity scores. Its top 10 is split into judged
  positives, confirmed negatives and unjudged passages (in scope: the
  company asked about; out of scope: other companies).
- Coverage flags use pre-set thresholds (`min_positives`,
  `min_judged_at_10`).
- Development sets write only under `results/dev/`. Every output carries the
  provisional banner, the corpus SHA-256, the label version, the metric
  definitions and the configuration.

Every retriever in `baselines` runs in every mode in `retrieval_modes`, and
each output records its mode:

- **`global`:** each question ranks the whole corpus.
- **`company`:** each question ranks only the passages of the companies it
  names (its `tickers`). Other passages are removed before ranking, so a
  constrained run returns up to `top_k` eligible passages rather than a
  filtered global list. BM25 and the dense models get exactly the same
  restriction (`finretriever.retrieval.base.eligible_indices`).

All runs are scored by the same function with the same metric definitions.
Each writes `results/dev/<set>/<mode>/<retriever>/metrics.json` and
`run.jsonl`, and one `report.md` compares them. The report also discloses
where the judged passages came from (`candidate_pool_path`): the pools were
built mainly with BM25, which favours BM25-like rankings (§7).

The BM25 settings are pinned in the config. The `word` tokenizer is the
analyzer the candidate pools used. The pure-Python backend gives the same
ranking whether or not `rank_bm25` is installed; the default `auto` backend
does not, which is why the committed mini-demo results reproduce only with
the pure-Python engine.

The dense baselines (`finretriever.retrieval.biencoder`) are pinned to
Hugging Face commits: `BAAI/bge-base-en-v1.5` at
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` and `intfloat/e5-base-v2` at
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. Each model's input format is
built into the code, and a prefix in the config must match it:

- **BGE** puts its retrieval instruction ("Represent this sentence for
  searching relevant passages: ") before queries only. Its model card
  recommends the instruction for short queries against passages and says
  v1.5 also works, slightly worse, without it. The evaluation questions are
  short queries over long passages, so it is used. Passages get no prefix.
- **E5** needs "query: " before every query and "passage: " before every
  passage.

The models' own configured prompts are never added on top. Embeddings are
L2-normalised, so the dot product used for ranking is the cosine; ties go to
the earlier corpus position. Passages are encoded in batches of 32, one
batch at a time, on CUDA when available and otherwise on the CPU (Apple
MPS only when set explicitly). Each query is encoded on its own, so its
vector does not depend on the other questions in the run.

Passage embeddings are cached under `data/cache/embeddings/` (gitignored).
The cache key is the corpus SHA-256, the model, the revision and the
encoding settings: passage prefix, maximum sequence length, pooling,
normalisation, batch size, device type, and the versions of
sentence-transformers, transformers and torch. An entry is reused only if
its metadata, passage order and file hashes all match. Anything else is
rejected with an error, never silently reused or rebuilt.

A model that cannot be loaded is listed as skipped with the reason. The
other retrievers still run, and the script exits 1.

### First BM25 development benchmark (provisional)

These are provisional development metrics on AI-generated labels, not
benchmark results. They come from the first run, which compared two BM25
analyzers with global ranking only. `results/dev/pilot_v2/report.md` now
holds the run described in the next subsection, whose `global/bm25` run
reproduces the `bm25` row below exactly.

| Run | nDCG@10 | MRR@10 | Recall@100 | judged@10 |
|---|---|---|---|---|
| `bm25` (word tokens) | 0.181 | 0.323 | 0.606 | 0.880 |
| `bm25_whitespace` (diagnostic) | 0.190 | 0.363 | 0.520 | 0.690 |

For `bm25`, the top 10 across the 10 questions holds 18 judged-relevant
passages, 70 confirmed negatives and 12 unjudged passages, all 12 from other
companies. Only `q_0006` falls below the coverage thresholds. Its top 10 is
entirely Walmart and Amazon credit-facility passages. None of its five
positive passages contains the word "Tesla" (27% of Tesla's passages do), so
the company named in the question gives BM25 no signal.

The failures are informative:

- **`q_0001`:** a vocabulary gap. "Third-party manufacturers abroad" in the
  question against "outsourcing partners" in the filings; no positive
  reaches the top 100.
- **`q_0003`:** other years' AWS explanations and unrelated Amazon notes
  outrank 2023's.
- **`q_0010`:** six other-year copies of the OPEC risk factor, all 0 under
  P4, outrank the FY2020 passage (rank 7).

The two analyzers cannot be compared on these numbers. They differ in
judged@10 (0.88 against 0.69), with 7 unjudged passages from the asked-about
companies in the whitespace run. On 10 questions, differences of this size
are noise.

### Retrieval modes and zero-shot dense baselines (provisional)

Status (2026-10-29): all six runs are complete: BM25, BGE and E5, each in
both retrieval modes, on CPU. Both passage-embedding caches are in
`data/cache/embeddings/`. The runs below are scored with the `pilot_v2`
labels; the next subsection re-scores the same rankings after deepening the
pools.

```bash
pip install -r requirements.txt   # torch + sentence-transformers
python scripts/04_run_baselines.py --config configs/eval/benchmark.yaml --eval-set dev_pilot_v2
```

Provisional development metrics; the report is at
`results/dev/pilot_v2/report.md`.

| Run | nDCG@10 | MRR@10 | Recall@100 | judged@10 | Unjudged in top 10 (same / other company) |
|---|---|---|---|---|---|
| `bm25`, global | 0.181 | 0.323 | 0.606 | 0.880 | 12 (0 / 12) |
| `bm25`, company | 0.206 | 0.423 | 0.665 | 1.000 | 0 (0 / 0) |
| `bge_zero_shot`, global | 0.247 | 0.450 | 0.572 | 0.490 | 51 (40 / 11) |
| `bge_zero_shot`, company | 0.247 | 0.450 | 0.592 | 0.520 | 48 (48 / 0) |
| `e5_zero_shot`, global | 0.314 | 0.550 | 0.628 | 0.430 | 57 (44 / 13) |
| `e5_zero_shot`, company | 0.330 | 0.575 | 0.637 | 0.480 | 52 (52 / 0) |

- **What the restriction changes.** All 12 unjudged passages in the global
  top 10s come from companies the question does not ask about, so the
  restriction removes exactly them. The main change is `q_0006`: its global
  top 10 is all Walmart and Amazon credit-facility text, and restricted to
  Tesla, a relevant passage is ranked first. The other questions keep their
  top 10s, apart from one passage each for `q_0001` and `q_0007`.
- **Why company-mode BM25 is fully judged.** The pools were built with this
  same BM25 (`word` analyzer, pure-Python engine), restricted to the
  question's company, 30 deep on the question and its reformulations. 511
  of the 579 judged passages (88%) and 80 of the 85 relevant ones (94%) came
  from BM25. The other 68 are 50 random, 13 stratified and 5 hand-found
  passages, and no dense retriever proposed any. Company-mode BM25 is
  therefore close to the system that built the pools. Its judged@10 of 1.0
  is a property of the pool, not evidence of quality, and its scores are the
  most favoured of any run.
- **What the dense runs faced.** Every passage a dense model ranked highly
  that BM25 had not pooled was unjudged and counted as non-relevant. Half or
  more of each dense top 10 was unjudged, so these scores are lower bounds.
  The pools were therefore deepened with the six runs' top 10s; see
  *Expanded development pool* below.
- **Truncation.** Both models read at most 512 tokens, including special
  tokens. 9,204 of 10,137 passages (90.8%) are longer for BGE and 9,208 for
  E5 (with its `passage: ` prefix). The median passage has about 683 tokens,
  and 27.6–27.7% of all passage tokens fall beyond the limit, so a passage
  is embedded from its first 512 tokens only. Evidence near the end of a
  long passage is invisible to the dense models but not to BM25. No
  question is truncated (at most 42 tokens).

### Expanded development pool and labels (`pilot_v3`)

`pilot_v3` deepens the `pilot_v2` pools with the six runs' top 10s (§7) and
labels the passages that brings in. The full coverage report, label review
and metric comparison are in
[pool_review_pilot_v3.md](pool_review_pilot_v3.md).

```bash
python3 scripts/03_annotate_gold_set.py expand-pool --spec configs/eval/dev_pool_pilot_v3.yaml
# AI proposals for the pool's unjudged entries go to data/eval/dev_ai/pilot_v3/proposals/
python3 scripts/03_annotate_gold_set.py dev-labels --spec configs/eval/dev_labels_pilot_v3.yaml
python3 scripts/03_annotate_gold_set.py export-dev --labels data/eval/dev_ai/pilot_v3
python3 scripts/04_run_baselines.py --config configs/eval/benchmark.yaml --eval-set dev_pilot_v3
```

- **Pool (`pilot_v3/pool/`).** The pool holds every `pilot_v2` pool entry
  plus every passage a stored run ranked in its top 10.
  - Each entry keeps its original candidate sources. It also lists every run
    that ranked it, with run, mode, rank and score, and `judged_in`: the
    label version that judged it, or `null`.
  - Adding a passage to the pool is not a judgment. A `null` entry stays out
    of the qrels until a label set grades it.
  - The stored runs are read with their SHA-256 checked. Each run's
    judged@10 under the base labels must equal what it recorded.
  - The result: 117 new entries (92 from the companies asked about, 25 from
    other companies) and 696 entries in all.
- **Proposals (`pilot_v3/proposals/`).** AI proposals (`claude-opus-5-5`,
  method `llm-full-passage-read-v2`) cover the unjudged entries only. A
  passage `pilot_v2` judged is refused, so no earlier label is re-graded.
- **Labels (`pilot_v3/`).** The `pilot_v2` labels, changes, flags and audit
  dispositions are carried over unchanged; each label records
  `introduced_in`.
  - The new labels go through the same rubric v1.1 rules: the P2 check, P1
    flags, flags for low-confidence proposals, and curated flags. Curated
    flags may be added only to new labels.
  - New entries without a proposal would stay unjudged; there are none.
  - The set records the SHA-256 of the base labels, the pool and the
    proposals. `export-dev` refuses to run if any of them, or the base
    set's own inputs, change.
- **Benchmark (`--eval-set dev_pilot_v3`).** The set has `runs_from:
  results/dev/pilot_v2`, so it runs no retriever. It scores the stored
  rankings unchanged, with the same metric functions.
  - Each output records its source run file and SHA-256.
  - With `compare_with: dev_pilot_v2`, the report puts the earlier scores
    beside the new ones.

Results:

- **Labels.** 696 labels: 579 carried over and 117 new AI labels (102 at
  grade 0, 14 at grade 1, 1 at grade 2). 12 new flags are open; 8 of them
  are for low confidence.
- **Coverage.** Every run's judged@10 is now 1.000.
- **New positives.** All 15 come from dense top 10s, and none was in a BM25
  run's top 10.
- **Metrics.** The dense runs' nDCG@10 rose by 0.04–0.05. BM25's fell by
  about 0.01, because its ideal ranking now includes positives it does not
  retrieve. The order on nDCG@10 is unchanged.

These are still provisional development metrics. The new labels are
unaudited AI proposals, ranks 11–100 are 64–75% unjudged, and the
comparison is sensitive to 7 low-confidence labels (see the review).

### Flag review and blind audit of `pilot_v3`

The 12 new flags were reviewed in
[flag_review_pilot_v3.md](flag_review_pilot_v3.md). That review is an AI
second look, not a human decision, and it changed no label. It recommends
retaining 11 grades and raising one. It also asks a person to settle two
conventions:
- the pair of Amazon cost passages (`q_0003`);
- total-only figures for the ExxonMobil breakdown question (`q_0009`).

A blind priority audit of 20 new passages waits for a person in
`data/eval/audit/pilot_v3/`:

```bash
python3 scripts/03_annotate_gold_set.py audit-init-pool --spec configs/eval/audit_pilot_v3.yaml --annotator <creator>  # done
python3 scripts/03_annotate_gold_set.py audit --audit-dir data/eval/audit/pilot_v3 --annotator <initials>
python3 scripts/03_annotate_gold_set.py audit-report --audit-dir data/eval/audit/pilot_v3
```

- **How the sample was drawn** (`finretriever.eval_set.pool_audit`). It is
  purposive, not representative, so its agreement describes these cases
  only. Candidates are the new passages from the companies each question asks
  about.
  - **Tier 1 first:** proposed relevant or low confidence. That covers all 15
    new positives and all 8 low-confidence labels.
  - **Then balance:** questions are kept balanced, less confident labels and
    those with more review features come first, and ties are broken by seed
    20261029.
  - **Review features:** a financial table, a report year the question does
    not ask about, or a dense top-10 passage absent from company-mode BM25's
    top 100.
- **Blindness.** The workspace holds no AI grade, confidence, explanation or
  selection reason.
- **What the auditor must not open first.** The AI grades for these passages
  appear in `data/eval/dev_ai/pilot_v3/` and in the two review pages. The
  auditor should not open them before finishing.
- **AI assistance.** If the auditor uses any, it should be recorded in each
  affected note.
- **No grade is recorded automatically.** `audit-report` runs only when
  every item is graded or skipped, and it changes no label.
