# Data pipeline: SEC EDGAR to the passage corpus

How `corpus_v1` is built: downloading 10-K filings from SEC EDGAR, then parsing, filtering and chunking them into passages. Moved here unchanged from the README (2026-11-15); the system overview is in [architecture.md](architecture.md).

## Download SEC filings

`make download` runs [`scripts/01_download_filings.py`](../scripts/01_download_filings.py),
which fetches 10-K filings directly from SEC EDGAR for the tickers in
[`configs/data/corpus_v1.yaml`](../configs/data/corpus_v1.yaml) and stores them
under `data/raw/` (gitignored). `make corpus` then turns them into the
passage corpus offline; see [Build the passage corpus](#build-the-passage-corpus).

### Set `SEC_USER_AGENT`

SEC fair-access policy requires every request to identify the requester with
a name and contact email. The downloader reads this from `SEC_USER_AGENT` and
refuses to make any request without it; there is no default. Keep the real
value out of Git:

```bash
# Option A: current shell only
export SEC_USER_AGENT="Your Name your.email@example.com"

# Option B: the gitignored .env file (quote the value, it contains spaces)
cp .env.example .env              # then set SEC_USER_AGENT="..." in .env
set -a; source .env; set +a       # the scripts do not read .env themselves
```

Never put the value in `configs/`, scripts, or any other committed file.

### Dry-run vs. download

| Command | SEC requests | Writes |
|---|---|---|
| `make download DOWNLOAD_ARGS=--dry-run` | ticker map + submissions JSON per ticker (+ older history pages when needed) | nothing |
| `make download` | the above + one archive request per filing not already on disk | `data/raw/<TICKER>/10-K/<accession>/filing.html` and `metadata.json` |

A dry-run lists each selected filing's accession number, filing date, CIK,
primary document and archive URL without fetching any filing document. A real
download is idempotent: a filing whose `filing.html` matches the SHA-256 in its
`metadata.json` is skipped on re-run. Requests are spaced 0.2 s apart and
429/5xx responses are retried with bounded backoff. Unknown or misspelled
config keys (e.g. `filing_per_ticker`) are rejected rather than ignored.

For a first live check, use one ticker and one filing via a throwaway config:

```bash
printf 'tickers: [AAPL]\nfilings_per_ticker: 1\noutput_dir: /tmp/finretriever_raw\n' > /tmp/sec_smoke.yaml
python scripts/01_download_filings.py --config /tmp/sec_smoke.yaml --dry-run
```

## Build the passage corpus

```
SEC EDGAR ──01_download_filings.py──▶ data/raw/<TICKER>/10-K/<accession>/{filing.html, metadata.json}
          ──01_build_corpus.py──────▶ data/processed/corpus_v1.jsonl + corpus_v1.manifest.json
```

`make corpus` (or `python scripts/01_build_corpus.py --config configs/data/corpus_v1.yaml`)
runs entirely offline:

1. **Validate** every raw filing: `metadata.json` against the downloader's
   schema, `filing.html` against the SHA-256 recorded at download time.
   Incomplete, corrupt, duplicated or unparseable filings are *rejected*;
   valid filings outside the configured tickers, form type or newest
   `filings_per_ticker` are *excluded*.
2. **Parse** each filing into Item sections
   ([`filing_parser.py`](../src/finretriever/data/filing_parser.py)): hidden
   inline-XBRL metadata and page footers removed, sections located through the
   linked table of contents, financial tables rendered one self-describing row
   per line (`Net sales > Products | 2025: $307,003 | 2024: $294,866`).
3. **Filter** boilerplate sections whose whole body is `None.`, `Not
   applicable.`, `[Reserved]`, or a one-sentence proxy-statement reference.
   Sections are never dropped for being short, and sections with tables are
   always kept. Rules live in `SectionFilterRules` and every drop is recorded.
4. **Chunk** each section separately (no passage spans two Items) into
   passages of at most `chunk_size_tokens` (512) whitespace tokens with up to
   `chunk_overlap_tokens` (80) of overlap ([`chunker.py`](../src/finretriever/data/chunker.py)).
5. **Write** the corpus JSONL (one `Passage` per line, deterministic IDs) and a
   manifest, each atomically.

Both outputs live in `data/processed/` and are gitignored. The manifest records
the corpus SHA-256, counts, every filtered section and why, rejected/excluded
filings, the source accessions, chunk and filter settings, table-boundary
diagnostics, and the git commit plus source-file hashes of the builder. Build
time appears only in the manifest, so rebuilding from the same filings and
settings produces a byte-identical corpus.

Exit codes: `0` complete corpus; `1` corpus written but some filings were
rejected or a configured ticker has no filings (or `--strict` refused to build
at all); `2` configuration error or nothing buildable. Use
`make corpus CORPUS_ARGS=--strict` to fail on any problem instead of building a
partial corpus.

### Table-aware chunking and its limits

Prose sections are chunked with fixed 512-token windows and 80-token overlap.
Sections containing tables keep every table row whole: a passage never starts
or ends inside a row, so values are never separated from their row label,
group (`Net sales >`) or column labels. Remaining limitations:

- Near tables, overlap is made of whole rows, so it can be shorter than 80
  tokens (or zero when the last row alone is longer than the overlap).
- A row longer than the whole 512-token budget is split between cells, and each
  continuation piece repeats the row label followed by `(continued)` — text
  that is not in the filing itself. The manifest counts these rows.
- A passage that starts in the middle of a long table carries the rows' own
  labels but not the table's title (e.g. `CONSOLIDATED BALANCE SHEETS`) or the
  `[TABLE]` marker, which stay in the earlier passage.
- Table quality is bounded by the parser: tables without detectable column
  headers keep values in row order without per-value labels.
- Cover page, table of contents and signatures are not part of any Item and
  are excluded; exact duplicate passages are removed within a filing but not
  across filings (the manifest counts cross-filing duplicates).
