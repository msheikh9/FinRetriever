# Training-data preparation (`bge_base_ft_v1`)

Status: 2026-10-30.

> **Superseded for new generation by `bge_base_ft_v2`** ([training_data_v2.md](training_data_v2.md)),
> which adds sentence-, span- and fact-level leakage control, evidence-aware split units, per-pair
> evidence checks and validator `sec-qcheck-v3`. This preparation, its pilot records and pairs are
> kept unchanged; its pairs are re-checked against v2 with `carry-over`, never rebound.

- **The pipeline is implemented and tested.** `prepare` has been run on the
  frozen corpus.
- **Real generation so far covers 15 of the 24 `pilot_v2` passages**, plus
  one earlier Qwen request (section 10). A run stopped at its request budget
  after HTTP 429s from Groq's limit of 1,000 output tokens per minute. The
  generator now paces requests to that limit, and the remaining 9 passages
  resume as a continuation of the same run. No training pair has been built
  yet. Further generation needs explicit confirmation and `GROQ_API_KEY`.
- **No model has been trained.**

The configuration is
[`configs/train/data_prep_v1.yaml`](../configs/train/data_prep_v1.yaml) and the
code is in `finretriever.training`. Everything generated goes under
`data/processed/train/bge_base_ft_v1/`, which is gitignored.

```bash
python3 scripts/02_generate_queries.py prepare --config configs/train/data_prep_v1.yaml        # offline
python3 scripts/02_generate_queries.py sources --split train        # source-quality report, offline
python3 scripts/02_generate_queries.py plan --split train --name pilot_v2 --size 24   # offline, write-once
python3 scripts/02_generate_queries.py generate --split train --plan pilot_v2 --config configs/train/data_prep_v1.yaml
#   prints the plan only; add --confirm-api-use (with GROQ_API_KEY set) to send requests; resumable;
#   --continue-run when only request settings (token limit, pacing, retries) changed since the run began
python3 scripts/02_generate_queries.py revalidate --split train     # re-run the checks on records, offline
python3 scripts/02_generate_queries.py generate --split validation --config configs/train/data_prep_v1.yaml
python3 scripts/02_generate_queries.py build-pairs --config configs/train/data_prep_v1.yaml
python3 scripts/02_generate_queries.py check --config configs/train/data_prep_v1.yaml
```

`make queries QUERIES_ARGS="prepare"` (or `generate --split train`, and so
on) runs the same commands.

## 1. Overlap groups (`finretriever.training.overlap`)

10-K filings repeat text from year to year, and chunk boundaries shift. A
passage's copies must therefore stay on one side of the train/validation
split, and none of them may be used as a negative for it. The method is
deterministic:

1. **Shingles.** Each passage becomes case-folded `[a-z0-9]+` words, then
   runs of 5 consecutive words ("shingles"), each hashed with 64-bit BLAKE2b.
2. **Exact duplicates.** Passages with identical word sequences are linked
   directly, at any length.
3. **Candidates without comparing all pairs.** An inverted index maps each
   shingle to the passages containing it. Shingles found in more than 50
   passages are treated as boilerplate and skipped. A pair is a candidate if
   it shares at least 10 of the remaining shingles.
4. **Exact verification.** For each candidate pair, compute on the full
   shingle sets:
   - Jaccard: shared shingles ÷ all shingles of both passages;
   - containment: shared shingles ÷ the smaller passage's shingles.

   A pair is an edge if containment is at least 0.5, and is labelled:
   - `exact`: identical word sequences;
   - `near_duplicate`: Jaccard at least 0.8;
   - `overlap`: otherwise.
5. **Groups.** A group is a connected component of the edges, so a chain
   A~B~C forms one group even when A and C don't overlap.

**When a pair can be missed.** Only when nearly all the text it shares is
boilerplate found in more than 50 passages. Exact duplicates are never
missed.

**On the frozen corpus** (about 10 seconds):

| Statistic | Value |
|---|---|
| Candidate pairs verified | 152,342, instead of 51 million all-pairs comparisons |
| Edges | 15,327: 104 exact, 1,567 near-duplicate, 13,656 overlap |
| Edges across filings / across companies | 14,998 / 1,103 |
| Groups | 4,068: 2,871 single passages, 1,197 groups of two or more |
| Passages in groups of two or more | 7,266 (72%) |
| Groups linked only through a chain | 529 |
| Largest group | 143 passages |

**The cross-company edges are boilerplate.** They come from auditor reports,
controls-and-procedures sections and financial-statement indexes shared
across companies, which is also why the largest group spans companies. Such
passages make weak training sources; the source-quality filter excludes them (§9).

## 2. Evaluation blocklist (`finretriever.training.blocklist`)

**What is blocked:**
- every candidate passage of the development evaluation:
  - the annotation workspace pool;
  - the expanded `pilot_v3` pool;
  - the `pilot_v1` and `pilot_v3` audit samples;
- every gold-qrel passage (none exist yet);
- every member of these passages' overlap groups.

**What is read.** Passage IDs only. Grades, judgments and question texts are
never read: every pooled passage is blocked whether relevant or not, so no
label can shape the training data.

**Failing closed.** Each source is opened with its own corpus-binding check.
The build stops when a source:
- is missing;
- is bound to another corpus;
- names a passage outside the corpus;
- is a gold-qrels file with rows but no metadata.

It also stops when no evaluation source is configured.

**On the frozen corpus.** 2,110 passages and 396 groups are blocked: 667
direct evaluation candidates and 1,443 more through their overlap groups.

| Source | Passages |
|---|---|
| Annotation workspace pool | 558 |
| `pilot_v3` expanded pool | 667 |
| `pilot_v1` audit sample | 30 |
| `pilot_v3` audit sample | 20 |
| Gold qrels | 0 (empty) |

`blocklist.jsonl` lists every blocked passage with its reasons: the sources
it appears in, or `overlap_group:<id>`.

## 3. Train/validation split (`finretriever.training.splits`)

**How groups are assigned.**
- Whole overlap groups are assigned, never single passages.
- Each group belongs to the (company, fiscal year) of most of its members.
- Within each (company, fiscal year), groups are ordered by
  SHA-256(`"{seed}:{group_id}"`).
- A group moves to validation while that brings its stratum closer to 10% of
  its passages.

**Checks.** `validate_split` and `open_prepared` refuse:
- a group with members on two sides;
- a blocked passage in either split;
- an unblocked passage marked blocked;
- a group that is only partly blocked.

**On the frozen corpus** (seed 20261029). Train has 7,200 passages in 3,253
groups; validation has 827 passages in 419 groups (10.3% of the usable
passages). No group crosses.

| Company | Train | Validation | Blocked |
|---|---|---|---|
| AAPL | 302 | 36 | 280 |
| AMZN | 410 | 55 | 311 |
| JPM | 2,975 | 336 | 318 |
| PFE | 1,092 | 125 | 288 |
| TSLA | 713 | 80 | 448 |
| WMT | 754 | 87 | 169 |
| XOM | 954 | 108 | 296 |

| Fiscal year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| Train | 891 | 1,106 | 1,012 | 982 | 1,022 | 1,015 | 1,065 | 107 |
| Validation | 115 | 147 | 128 | 109 | 112 | 108 | 93 | 15 |
| Blocked | 213 | 280 | 302 | 335 | 334 | 338 | 286 | 22 |

Fiscal 2026 is Walmart's year ending January 2026.

## 4. Visible text (`finretriever.training.visible_text`)

**What the encoder can read.** The pinned BGE tokenizer (`tokenizer.json` at
`a5beb1e3…`, loaded offline with `tokenizers`) truncates at 512 tokens,
including special tokens and any passage prefix. Its offsets give where
truncation ends in the text, backed off to the last whole word.

**On the frozen corpus.**
- 9,204 passages (90.8%) are truncated, the same count as the zero-shot runs.
- A visible passage has 510 content tokens, a median of 365 words.
- 7,185 train and 827 validation passages have at least 40 visible words and
  can give queries.

**How it constrains generation.** The generator sees only this visible text.
A query's evidence must lie inside it, so no accepted query depends on
truncated text.

## 5. Query generation (`finretriever.training.query_generation`)

**Prompt `sec-qgen-v1`.** Its SHA-256 is recorded with every output. The
model receives:
- the company, filing, fiscal year and section;
- the visible passage only;
- for train passages, a request for 2 questions; for validation, 1;
- a hint for one figure question when a table is present.

**Rules given to the model:**
- answerable from the passage alone;
- name the company and the period;
- no run of more than five copied words;
- only numbers that appear in the passage;
- no reference to "the passage" or "the table".

Each question comes back with a type (risk, MD&A, financial table, debt and
liquidity, revenue, other), a short answer and a verbatim evidence quote.

**Checks on every candidate.** It is rejected, with the reason recorded,
when it:
- is malformed, or has an unknown type;
- is not a 6–40-word question ending in "?";
- refers to "the passage", "the table" and similar;
- does not name its company, or names another corpus company;
- copies 8 or more consecutive words from the passage;
- has evidence shorter than 15 or longer than 300 characters;
- has evidence not found in the visible text (the answerability check);
- contains a number in the question or answer that is not in the visible
  text or the filing's years (no invented or converted figures);
- contains a complete date, in the question or answer, that the visible text
  does not support (`date_not_in_passage`, see below);
- has an answer whose numbers or content are not in its evidence;
- is a near-duplicate within its passage;
- goes beyond the requested count.

**Dates (validator `sec-qcheck-v2`).** The prompt gives the fiscal year end
as an ISO date ("fiscal year ended 2025-09-27") and asks the model to name the
period, so questions repeat it. Validator v1 split that date into 2025, 09
and 27 and rejected the question because 09 and 27 were not in the passage.
v2 reads dates as dates:
- A complete date in the **question** equal to the filing's report date or
  filing date (from the corpus metadata) is exempt from the number check as
  a whole. It is recognised as 2025-09-27, 9/27/2025, 9/27/25,
  September 27, 2025 (also Sept. 27, 2025, Sep 27 2025, September 27th,
  2025) or 27 September 2025.
- Its day and month are never allowed on their own: "27 global notes" still
  fails, as do invented figures in the same question.
- Every other complete date in the question or answer must be supported by
  the visible text: the same date in any of those forms, or the same month
  and day with a year the passage or filing gives ("December 31, 2024 and
  2023", or a table header "December 31, | 2023"). Otherwise the candidate
  is rejected as `date_not_in_passage`. Under v1, "September 28, 2025"
  passed whenever 28 and 2025 appeared anywhere in the passage.
- Strings that are not real dates (2025-27-09, 2025-02-30) are checked as
  numbers.
- Answers get no exemption: an answer's numbers must still be in its
  evidence.

**What a record keeps (record format 3).** Each passage's record stores the
model, the prompt version, the validator version, the SHA-256 of the
generation parameters (`params_sha256`, new in format 3) and every proposed
candidate in reply order: question, type, answer and evidence exactly as
given (non-text values as null). Rejected candidates also keep their
position and rejection reason. The record does not store the raw reply, the
provider payload or any reasoning. The reply's SHA-256 is kept, `<think>`
blocks and `message.reasoning` are discarded before parsing, and every
record line is redacted for the API key before it is written.

**Local revalidation.** `revalidate --split train` re-runs the current checks
on each passage's latest done record that an older validator checked. It
uses the stored candidates and sends no request.
- It appends a revision record. That record carries the original
  candidates, the new accepted and rejected lists, zero usage, and a
  `revision` block: the SHA-256 of the record it revises, the previous
  validator version, whether the outcome changed, and when.
- Earlier lines are never rewritten. The latest done record counts, and a
  second run changes nothing.
- `--dry-run` reports without writing.
- Records from before format 2 kept only the rejected question text. They
  are listed as not revalidatable rather than reconstructed.
- `build-pairs` reports the done records per validator version
  (`done_by_validator`).

**API behaviour (Groq, OpenAI-compatible chat completions).**
- Model: `qwen/qwen3.8-27b`, after `llama-3.3-70b-versatile` returned 404
  for this key (section 10). Check a model is offered to the key before a
  run.
- Request settings, all in the config and recorded with every run:
  - temperature 0.7, the top of Groq's recommended 0.5–0.7 for Qwen;
  - `max_completion_tokens` 768, sent under that name (2048 until
    2026-10-30). The 15 `pilot_v2` replies used 181–265 completion tokens
    (median 230). The one numbers-heavy reply on record, an Apple exhibit
    list, used 399. Qwen's tokenizer splits figures into single digits, so a
    table-heavy reply with two 300-character quotes can reach roughly 700.
    512 would leave only 28% headroom over the 399-token reply; 768 leaves
    1.9×. Groq's default of 1024 counts reasoning tokens too;
  - JSON object mode;
  - `reasoning_effort: none`, so no reasoning tokens are spent on a task
    that extracts rather than solves;
  - `reasoning_format: parsed`, so any reasoning arrives in a separate
    field. `raw` (reasoning inside `<think>` tags) is refused with JSON
    mode, as Groq answers HTTP 400.

  Reasoning settings that are not set are not sent, so the client still
  works with models that do not accept them.
- Reasoning never becomes the reply:
  - a separate `message.reasoning` field is ignored;
  - `<think>...</think>` blocks are removed from the text;
  - an unclosed `<think>`, or a reply that is empty once reasoning is
    removed, counts as malformed.

  Reasoning tokens, when reported, are added to the usage as
  `reasoning_tokens`. Tokens are counted even when the reply is unusable.
- A reply cut off at the token limit (`finish_reason` `length`) is recorded
  as a failed request with status 200 and code `length`, without a retry:
  the same limit would usually cut it again. It counts toward the
  consecutive-failure stop.
- **Pacing.** Requests are spaced to 20 per minute. With
  `output_tokens_per_minute` set (1,000, the account's Groq limit for this
  model), every request also waits until the output tokens of the last
  61 seconds (one minute plus a 1-second margin) plus its own
  `max_completion_tokens` fit within that limit.
  - A reply is charged the completion tokens it reports, or its whole
    reservation if it reports none (or the connection failed). A refused
    request is charged nothing.
  - A sliding window this strict also keeps within a fixed-minute or
    token-bucket limit.
  - At 768 tokens per request, a reply of about 230 tokens leaves room for
    one more request in the same minute. The worst case is one request per
    minute.
- **Oversized requests never go out.** A `max_completion_tokens` above
  `output_tokens_per_minute` is refused when the config loads, and again
  when the client is built, before any request.
- **HTTP 429 is temporary.** The client waits as long as the API asks, then
  retries:
  - first `Retry-After` (seconds or an HTTP date), then the reset of an
    exhausted `x-ratelimit-remaining-*` limit (`x-ratelimit-reset-*`, e.g.
    `2m59.56s`), then "try again in …" in the error message, and
    exponential backoff only when none of these names a wait. A 1-second
    margin is added to each;
  - `Retry-After` is no longer cut to `max_backoff_seconds` (60). That cut
    made the pilot retry too early and spend its budget on repeated 429s;
  - a limit left exhausted on a good reply, such as a daily request limit
    at 0, also delays the next request.
- **Clean stops on rate limits.** When a wait would exceed
  `max_rate_limit_wait_seconds` (180), or 429s outlast the retries, the run
  stops cleanly: it exits 0 and says why, the passage is not recorded, and
  a rerun resumes it. Rate limits therefore never mark a passage failed.
- **Too-large requests are permanent.** HTTP 413, "request too large", or an
  error showing more tokens requested than the limit ("Limit 1000,
  Requested 2048") stops the run at once with exit code 2. It is never
  retried, and the passage is not recorded. A limit that was only reached
  ("Limit 1000, Used 980, Requested 768") is a temporary 429.
- 5xx and network errors are retried with exponential backoff.
- HTTP 401/403 stop the run at once, as does a missing key (exit code 2).
- HTTP 404, or the error codes `model_not_found` / `model_decommissioned`,
  also stop the run at once (exit code 2), with nothing recorded: the model
  is unavailable to the key, so every request would fail.
- Three consecutive failed requests stop the run, which can be resumed.
  Rejected requests count as well as transient errors. Failed records keep
  the HTTP status and error code.
- A reply that is not the expected JSON is retried once.

**Credential safety.** The key is read only from `GROQ_API_KEY` and sent only
in the Authorization header. It is redacted from every recorded error and
never written or printed; the tests check every output file and both output
streams.

**Resuming.**
- Each passage's outcome is appended to `generation/<split>.records.jsonl`
  and flushed before the next request. A rerun skips passages already done,
  and failed passages are retried only with `--retry-failed`.
- `generation/<split>.run.json` binds the records to the prompt, model,
  parameters and preparation. A rerun with different settings is refused,
  so runs never mix.
- **Request settings may change through a continuation.** These are
  `max_completion_tokens`, `output_tokens_per_minute`,
  `requests_per_minute`, `max_rate_limit_wait_seconds` and the retry,
  timeout and backoff settings (`REQUEST_PARAMS`).
  - They decide how and how fast requests are sent, never what an accepted
    reply says. The token limit only ends a reply early, and a reply that
    reaches it is recorded as failed, never accepted.
  - With `--continue-run`, the run is continued instead of refused. One
    line is appended to `generation/<split>.run.continuations.jsonl` with:
    the SHA-256 of `run.json`, the previous and new parameters' SHA-256,
    the new parameters, each changed setting (from/to), and the number and
    SHA-256 of the records written so far.
  - `run.json` and the records are never rewritten. Later records carry the
    new `params_sha256`; records without it predate any continuation.
  - Every later check confirms that the counted records still begin the
    file byte for byte. It also checks that each continuation names the run
    and the binding before it.
  - Any other change (prompt, model, temperature, JSON or reasoning mode,
    queries per passage, validation thresholds, preparation) is still a
    different run, with or without the flag.
  - The dry run shows the changes and what `--continue-run` would record.
    `revalidate` and `build-pairs` accept a continued run. Each pair's
    `generation_params_sha256` is that of the record it came from, and
    `pairs.meta.json` lists the run's parameter hashes and the
    continuations file's SHA-256.

**Separation of inputs.** Evaluation questions and labels never enter a
prompt. Train and validation records and pairs are separate files.

## 6. Pairs and quality controls (`finretriever.training.pair_quality`)

`build-pairs` turns the accepted candidates into
`pairs/train_pairs.jsonl` and `pairs/validation_pairs.jsonl`. It adds two
global rejections:
- **`duplicate_query`:** a query repeated anywhere, with train taking
  precedence over validation;
- **`evaluation_overlap`:** a query whose word-set Jaccard with any
  evaluation question is at least 0.8. The question texts come from
  `annotation/questions.jsonl` and `gold_queries.jsonl` and are used only
  for this comparison.

**What each pair records.** Its positive passage, group, company, year,
question type, answer, evidence, and the highest similarity to any
evaluation question. It also records the provider, model, prompt version
and a hash of the generation parameters.

**Fail-closed re-check.** `check_pairs` re-verifies everything before any
file is written, and `check` repeats it later:
- the positive exists and belongs to the pair's split;
- the positive is not blocked;
- its group lies wholly in that split;
- no (query, passage) pair or query repeats;
- every candidate check still passes;
- no query overlaps an evaluation question.

**Leakage limitation.** The leakage check is lexical, the minimum safeguard.
A paraphrase of an evaluation question can pass, so semantic leakage is not
ruled out. The blocklist means no evaluation passage can be a positive or a
negative, which limits the damage. A semantic check (cosine similarity to
evaluation questions with the zero-shot encoder) is the obvious next
safeguard.

## 7. Manifests and reproducibility

**`manifest.json` (preparation).** No timestamps; reruns are byte-identical,
and a version directory is never overwritten with different content. It
records:
- the frozen corpus's SHA-256, version and passage count;
- the config's SHA-256;
- the overlap method, settings and statistics;
- the blocklist sources, each with its SHA-256, and their statistics;
- the split seed, strategy and distributions;
- the tokenizer's identity: the SHA-256 of its serialised state, its limit
  and the passage prefix;
- the SHA-256 of every file.

**`pairs/pairs.meta.json` (pairs).** It records:
- the corpus identity;
- the preparation manifest's SHA-256 and the blocklist's SHA-256;
- the split and overlap settings;
- the provider, model, prompt version and its SHA-256;
- the full generation parameters and their hash;
- each run's record and run-file hashes;
- the leakage method, threshold, and the SHA-256 of each question file;
- accepted and rejected counts by reason, question-type and company/year
  distributions;
- the quality-check result and the output hashes.

Pairs are rebuilt deterministically from the records.

## 8. Example with mocked responses

These replies were **written by hand to stand in for the model**. They are
not Groq output. They were run through the real checks on three approved
training passages:

| Passage | Mocked question | Result |
|---|---|---|
| Walmart FY2022 Item 1 | What share of Walmart's fiscal 2022 consolidated net sales came from the Walmart U.S. segment? (answer 69%; evidence quoted from the passage) | Accepted |
| Walmart FY2022 Item 1 | How did Walmart U.S. net sales grow from $341.0 billion to $395 billion by fiscal 2022? | Rejected: `number_not_in_passage` ($395 billion is invented) |
| Walmart FY2022 Item 1 | Which segment does the passage call Walmart's largest in fiscal 2022? | Rejected: `refers_to_context` |
| Pfizer FY2022 Item 1 | What drug pricing and reimbursement risks did Pfizer describe in its fiscal 2022 annual report? | Accepted |
| Pfizer FY2022 Item 1 | What must drug makers offer to have their products covered by Medicaid in fiscal 2022? | Rejected: `missing_company` |
| Amazon FY2021 Item 1 (executive officers table) | Who led Amazon Web Services as its CEO according to Amazon's executive officer list of January 2022? | Accepted |
| Amazon FY2021 Item 1 | Did Amazon's executive chair also lead Apple in fiscal 2021? | Rejected: `other_company` |
| Amazon FY2021 Item 1 | Mr. Bezos founded Amazon.com in 1994 and has served as Executive Chair since July 2021? | Rejected: `copied_from_passage` |

## 9. Source quality and sampling plans

### Source-quality filter (`finretriever.training.source_quality`, `source-quality-v1`)

Approved passages are not all worth a request. The first pilot sample
included exhibit lists and cross-reference pages, and the first real Qwen
request went to an exhibit index (section 10). The filter decides which
approved passages generation may use.
- It changes nothing in the preparation: the split and blocklist stay as
  they are, and an excluded passage simply gives no queries.
- Its rules are generic and text-based, with no company names or
  company-specific lists. They read the BGE-visible text, the 10-K item
  number and how many filings repeat the text (the overlap group).
- Thresholds are per 100 visible words and set in the config's
  `source_quality` section.

A passage is excluded by the first rule it matches, in this order:

| Rule | Excludes a passage when | Typical catch |
|---|---|---|
| `exhibit_index` | exhibit-list markers ≥ 2.5 per 100 words (exhibit numbers and descriptions, "incorporated by reference to/from", "filed/furnished herewith", certifications, indentures, XBRL, references to other SEC forms and agreements) | Item 15 exhibit indexes |
| `cross_reference` | ≥ 50% of the words outside tables are in sentences that point elsewhere ("incorporated by reference", "see Note 5", "information required by this Item", "is included in", page numbers), and fewer than 2 financial figures per 100 words | Items 10–14 proxy references; Item 7/8 pages that only point to the annual report |
| `repetitive_administrative` | administrative markers ≥ 1.5 per 100 words (disclosure controls, internal control over financial reporting, certifications, signatures, "not applicable", officer rosters with ages) **and** the text repeats in ≥ 3 filings | Item 9A; management's report on internal control; executive-officer rosters |
| `low_financial_content` | finance terms < 0.5% of words, < 1 financial figure and < 3 table numbers per 100 words; **never** in Items 1A, 7, 7A or 8 | human-capital culture text; product-software descriptions; a Section 13(r) notice |

Notes on the rules:
- Financial figures are dollar amounts, percentages, numbers with thousands
  separators and decimals; bare years do not count.
- Table numbers include plain counts and years, but not ages. This keeps
  operating tables (production volumes) and patent-expiry tables, which a
  first version of the rule wrongly excluded.
- Risk factors, MD&A, market risk and financial statements are protected
  from the low-content rule, because their legal and qualitative passages
  (litigation, patents, macro risks) have few finance words but are
  substantive.

**What it excludes.** Counts are from `sources --split train` and
`sources --split validation` on the frozen corpus. A passage can match more
than one rule; only the first excludes it.

| Rule | Train: excluded (first match) | Train: any match | Validation: excluded | Validation: any match |
|---|---|---|---|---|
| `exhibit_index` | 186 | 186 | 1 | 1 |
| `cross_reference` | 44 | 56 | 2 | 2 |
| `repetitive_administrative` | 114 | 123 | 3 | 3 |
| `low_financial_content` | 19 | 44 | 2 | 2 |
| **Excluded** | **363 of 7,185 (5.1%)** | | **8 of 827 (1.0%)** | |

**What is excluded and kept, by section:**
- **Excluded:** Item 15 (184), Item 9A (63), Item 10 (42), Item 1 (23:
  officer rosters, investor-relations boilerplate and culture text), Items
  11 and 14 (7 each), Item 8 (7: the "see the annual report" pages and none
  of the statements or notes), and Item 7 (1, likewise).
- **Untouched:** no Item 1A, 7A or 1C passage is excluded.
- **Kept:** 680 risk-factor, 492 MD&A, 55 market-risk and 1,493
  financial-statement passages, plus 3,344 notes and MD&A subsections
  without an item number.

**Known residue.** A few weak passages still pass, and the pilot will show
whether they matter:
- 4 Item 15 passages: lists of XBRL financial statements and one list of
  credit-agreement amendments;
- some executive-officer rosters without "Age:" markers;
- auditor reports.

### Sampling plans (`finretriever.training.sampling_plan`)

A plan is a write-once JSON file,
`<prepared>/plans/<split>.<name>.json`. It holds:
- the selection method, seed and size;
- the prepared manifest it was made from;
- the source-quality version and settings, plus counts per rule;
- the passages in order, with the counts by company, fiscal year and
  section.

Writing a plan again with the same content is a no-op; different content
under the same name is refused.

`generate --plan <name>` sends the plan's passages in plan order, after
checking they are still approved sources of the same preparation. Records
stay in one `generation/<split>.records.jsonl`, so a passage already done is
skipped under any plan. `--sample` remains the original unfiltered pilot
selector.

| Plan | Method | Filter | Passages | Companies | Fiscal years |
|---|---|---|---|---|---|
| `pilot_v1` | `company_round_robin` (the original `--sample 25`, seed 20261029) | none | 25 | AAPL 4, AMZN 4, JPM 4, PFE 4, TSLA 3, WMT 3, XOM 3 | 2019 5, 2020 2, 2021 7, 2022 3, 2023 1, 2024 1, 2025 5, 2026 1 |
| `pilot_v2` | `company_year_round_robin`, seed 20261029 | `source-quality-v1` | 24 | AAPL 4, AMZN 4, JPM 4, PFE 3, TSLA 3, WMT 3, XOM 3 | 2019 3, 2020 4, 2021 3, 2022 3, 2023 3, 2024 4, 2025 3, 2026 1 (only Walmart has FY2026) |

**`company_year_round_robin`.** Companies take turns. Each company's next
passage comes from the fiscal year it has used least, then the year least
used across the plan, then seeded year order. Within a year, passages are
taken in seeded order. Only passages the filter keeps are eligible.

**`pilot_v2` by section:**
- Item 8: 10;
- notes and MD&A subsections without an item number: 6;
- Item 1A: 3;
- Item 7: 3;
- Items 1 and 5: 1 each.

Seven of its passages are also in `pilot_v1`, because the same seed orders
each company's passages. None of them has a Qwen record.

`pilot_v1` reproduces the original sample exactly. Its 25 passage IDs, in
order, equal those of the 25 archived llama records, and the one Qwen record
is its first passage. The new filter would exclude 4 of them:
- the AAPL FY2025 and two AMZN exhibit indexes;
- the Walmart Item 10 cross-reference.

## 10. Running a pilot

`generate` takes these options for a bounded pilot:

- **`--plan NAME`:** the passages of a written sampling plan, in plan order.
- **`--sample N`:** the original selection, kept so `pilot_v1` stays
  reproducible: N approved passages spread over companies, unfiltered,
  seeded with `--sample-seed` (default: the split seed).
- **`--limit N`:** at most N passages in this run.
- **`--max-requests N`:** a hard cap on HTTP requests, retries included. The
  run stops cleanly at the cap. A passage with no request yet is left
  unrecorded so a rerun resumes it. If the cap stops a passage after an
  unusable reply, that passage is recorded as failed with
  `budget_exhausted: true` and the error, so a one-request test still shows
  what came back.
- **`--continue-run`:** continue a run whose request settings changed
  (section 5, "Resuming"); without it such a run is refused.
- **Usage reporting:** each run prints the requests sent, the token usage the
  API reported, the HTTP 429 responses, the time spent pacing and waiting,
  and the latest `x-ratelimit-*` headers. Each record also stores its own
  token usage.

Without `--plan` or `--sample`, a full run uses every approved passage the
source-quality filter keeps.

### History of the first pilot

**Plan `pilot_v1`** (the original 25-passage sample):
- 4 passages each from AAPL, AMZN, JPM and PFE, and 3 each from TSLA, WMT
  and XOM;
- fiscal years 2019–2026;
- unfiltered, including 3 exhibit indexes and a cross-reference page.

**Attempt 1 (2026-10-29): `llama-3.3-70b-versatile`, failed.** It sent 25
requests and every one returned HTTP 404 `model_not_found`.
- No reply arrived, so no token usage was reported and no candidate was
  checked.
- The key was accepted: a bad key returns 401.
- The run then treated a refused model as a per-passage failure, so it
  repeated the same 404 for all 25 passages. It now stops after one request.
- The 25 failed records are kept unchanged in
  `generation.failed-llama-3.3-70b-versatile/`: `train.records.jsonl`
  SHA-256 `ccea8a87…c566`, `train.run.json` `f2af7e2b…d11c`.

**Attempt 2 (2026-10-29): one request to `qwen/qwen3.8-27b`.** The request
succeeded: 1,068 prompt and 399 completion tokens, no reasoning tokens. It
went to `pilot_v1`'s first passage, the Apple FY2025 Item 15 exhibit index.
- Qwen proposed 2 questions, and both were rejected as
  `number_not_in_passage`. The cause was the date bug above: the
  questions repeated "fiscal year ended 2025-09-27" from the prompt.
- Under validator v2, both questions pass the number check. The regression
  tests run them, with answers and evidence written for the test.
- **Not recoverable without a new request.** That record predates format 2
  and kept only the question text, not Qwen's answers or evidence.
  `revalidate` lists it as not revalidatable and writes nothing. The
  answers are not reconstructed, because made-up answers would be passed
  off as the model's.
- The record stays `done` with 0 accepted. Its passage is now excluded by
  the source filter, so nothing is lost for training. The run binding
  (`generation/train.run.json`) is unchanged and still matches the
  configuration.

### `pilot_v2` so far

**Attempt 3 (2026-10-29): one request on `pilot_v2`.** The first passage
was Apple's FY2024 risk factors (macroeconomic risks), with fiscal year end
2024-09-28. The reply used 868 prompt and 244 completion tokens. Both
questions were accepted; each carried the ISO report date, which validator
v2 exempted as verified metadata.

**Attempt 4 (2026-10-30, `--max-requests 23`): 14 more passages, then rate
limits.**
- Passages 2–15 completed in about two minutes, one request each, using
  181–265 completion tokens.
- The other 9 requests of the budget were answered with HTTP 429. The
  account's Groq limit for this model is 1,000 output tokens per minute,
  and each request reserved 2048. The client then paced only to 20 requests
  per minute and cut `Retry-After` to 60 seconds.
- The run stopped cleanly at the budget, and nothing was recorded for
  passage 16. No passage is marked failed, and 429 responses are not
  recorded.
- Over the 15 passages: 30 candidates proposed, 24 accepted, 6 rejected
  (4 `evidence_not_in_visible_text`, 2 `copied_from_passage`). Of the
  accepted, 11 are typed `other`, 7 `financial_table`, 4 `risk`, and 1 each
  `debt_liquidity` and `mdna`.
- Requests to Groq so far: 25 (llama), 1 (`pilot_v1`), 1 and 23 (`pilot_v2`),
  50 in all.

**Resuming the remaining 9 passages.** The token limit and pacing changed
(section 5), so the run continues under continuation 1 rather than starting
again. `run.json` and the 16 existing records stay as they are, and the 15
completed passages are skipped. First the dry run, which writes nothing:

```bash
python3 scripts/02_generate_queries.py generate --split train --plan pilot_v2 --continue-run
```

It should report 15 done, 0 failed, 9 to request, and the changes
`max_completion_tokens 2048 -> 768, max_rate_limit_wait_seconds unset ->
180.0, output_tokens_per_minute unset -> 1000`. Then:

```bash
python3 scripts/02_generate_queries.py generate --split train --plan pilot_v2 --continue-run --max-requests 12 --confirm-api-use
```

At 768 tokens per request this takes about 5–10 minutes. The budget of 12
covers the 9 passages plus 3 retries. If a rate limit or the budget stops
it, rerun the same command: completed passages are skipped and no second
continuation is recorded.

## 11. Limitations and next steps

- **Little real generation yet, and the output-token limit sets the pace.**
  One question per validation passage and two per training passage, over
  the filtered sources, means about 7,640 requests.
  - At this account's 1,000 output tokens per minute and 768 reserved per
    request, a second request fits in a minute only after a reply of 232
    tokens or fewer, about half the pilot's replies. That is one to two
    requests per minute: roughly 64–127 hours. The old estimate of 6.4
    hours at 20 requests per minute ignored that limit.
  - A full run needs a higher Groq limit, or a lower `max_completion_tokens`
    once the pilot's longest replies are known. Either change is a
    continuation of the run, not a new one.
  - Finish `pilot_v2` first, and read the rejection reasons and the
    question-type balance (11 of the first 24 accepted are `other`).
- **Company imbalance.** JPMorgan provides 42% of the kept training passages
  (2,894 of 6,822). Plans balance companies; a full run does not. Cap
  passages per company, or weight sampling, if the question mix skews.
- **The source filter is heuristic.** Its thresholds were set by reading
  the corpus's own passages, and the residue listed in section 9 passes.
  Recheck the per-rule counts after any corpus change, and read what the
  pilot's rejected candidates say about the remaining sources.
- **Lexical leakage check only** (§6). Semantic leakage remains possible.
- **The answerability check is lightweight.** A verbatim quote plus
  number and keyword support do not prove that the answer is correct or
  complete. Read a sample of accepted pairs before training.
- **Year-to-year copies across groups.** Copies below 50% shingle
  containment are not grouped. Lightly edited or reordered copies could
  still sit in different splits; the 0.5 threshold is a deliberate trade-off.
- **Tokenizer coupling.** The visible text depends on BGE's tokenizer. E5
  (ablation A3) reads a `passage: ` prefix with its own tokenizer and needs
  its own visible-text pass.
