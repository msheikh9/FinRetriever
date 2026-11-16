"""Build the passage corpus from downloaded SEC filings.

The pipeline is offline and deterministic:

    <raw_dir>/<TICKER>/<FORM>/<accession>/{filing.html, metadata.json}
      -> validate: metadata against ``FilingMetadata``, filing.html against
         the SHA-256 recorded at download time
      -> scope: configured form type and tickers, newest N filings per ticker
      -> parse: ``parse_filing`` -> Item sections
      -> filter: explicit, configurable boilerplate rules (``SectionFilterRules``)
      -> chunk: ``chunk_section`` per section, so no passage spans two Items
      -> write: corpus JSONL, then a JSON manifest, each atomically

Problems are split into two kinds. A *rejected* filing is incomplete,
corrupt, duplicated or unparseable; it never enters the corpus, and in strict
mode it fails the whole build. An *excluded* filing is valid but outside the
configured scope (another form type or ticker, or older than the newest N per
ticker); it is reported but is not a problem.

Order: filings by (ticker, filing date, accession), sections in document
order, chunks in window order. The corpus JSONL contains no timestamps, so
the same inputs and settings always produce byte-identical output; the
build time lives only in the manifest.

Tables and chunk boundaries: the parser renders each table row as one
self-describing line (``Net sales > Products | 2025: $307,003 | ...``), and
``chunk_section`` keeps those rows whole (see ``chunk_text_table_aware``):
no passage starts or ends inside a row, and a row longer than the whole
chunk budget is split into pieces that each repeat the row label plus
``(continued)``. The manifest ``diagnostics`` verify this independently on
the written passages — counting partial rows at passage edges, unlabelled
fragments (the target is zero), rows needing continuation, and rows not
represented whole or through labelled continuation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from finretriever.data.chunker import chunk_section, chunk_text_table_aware
from finretriever.data.edgar_downloader import (
    HTML_FILENAME,
    METADATA_FILENAME,
    EdgarDownloadConfig,
    FilingMetadata,
)
from finretriever.data.filing_parser import TABLE_END, TABLE_START, analyze_filing
from finretriever.data.schemas import Passage, Section, make_filing_id
from finretriever.utils.io import read_yaml, write_jsonl

CORPUS_BUILDER_VERSION = "1.2"  # 1.1: table-aware chunks; 1.2: appended reports, coverage
MANIFEST_VERSION = 1

DIAGNOSTIC_KEYS = (
    "table_rows",
    "table_rows_requiring_continuation",
    "table_rows_not_represented",
    "passages_starting_mid_table_row",
    "passages_ending_mid_table_row",
    "unlabelled_partial_row_fragments",
    "passages_not_located",
    "passages_duplicating_another_filing",
    "max_passage_tokens",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_MODULES = ("data/corpus_builder.py", "data/filing_parser.py", "data/chunker.py", "data/schemas.py")

DEFAULT_BOILERPLATE_PATTERNS: dict[str, str] = {
    "none": r"none\.?",
    "not_applicable": r"(?:not applicable|n/a)\.?",
    "reserved": r"\[?reserved\]?\.?",
    # One sentence (no inner period) deferring the Item to the proxy statement,
    # with "proxy statement" and "incorporated by reference" in either order.
    "proxy_reference": (
        r"(?=[^.]*\bproxy statement\b)(?=[^.]*\bincorporated (?:herein )?by reference\b)"
        r"(?:the )?information required by (?:this|these) items?\b[^.]*\.?"
    ),
}


class CorpusConfigError(ValueError):
    """The corpus configuration is missing, unreadable or invalid."""


class CorpusBuildError(RuntimeError):
    """No valid corpus can be built from the available filings."""


class StrictModeError(CorpusBuildError):
    """Strict mode found rejected filings or configured tickers without filings."""

    def __init__(self, message: str, build: CorpusBuild) -> None:
        super().__init__(message)
        self.build = build


class SectionFilterRules(BaseModel):
    """Rules deciding which parsed sections carry no retrievable information.

    A section is dropped only if its body (the text after the ``Item N.``
    heading line) has no word characters, or *entirely* matches one of
    ``boilerplate_patterns`` (case-insensitive, whitespace-collapsed) while
    being at most ``max_boilerplate_words`` long. Length alone never drops a
    section, and sections containing a table are always kept.
    """

    model_config = ConfigDict(extra="forbid")

    boilerplate_patterns: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_BOILERPLATE_PATTERNS)
    )
    max_boilerplate_words: int = Field(default=100, gt=0)
    drop_empty_sections: bool = True
    drop_duplicate_passages: bool = True  # exact duplicate text within one filing
    # Sections of at most this many words that only point elsewhere ("Refer to
    # Note 30 ...", "Reference is made to the section entitled ...",
    # "... appears on pages 53-160") carry no content of their own. 0 disables.
    cross_reference_max_words: int = Field(default=80, ge=0)

    @field_validator("boilerplate_patterns")
    @classmethod
    def _patterns_compile(cls, v: dict[str, str]) -> dict[str, str]:
        for name, pattern in v.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"boilerplate pattern {name!r} is not a valid regex: {exc}") from exc
        return v


class CorpusBuildConfig(BaseModel):
    """Everything that determines the corpus. ``manifest_path`` defaults to
    ``<output stem>.manifest.json`` next to the corpus."""

    model_config = ConfigDict(extra="forbid")

    name: str = "corpus"
    raw_dir: Path
    output_path: Path
    manifest_path: Path | None = None
    tickers: list[str] | None = None
    form_type: str = "10-K"
    filings_per_ticker: int | None = Field(default=None, gt=0)
    chunk_size_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    strict: bool = False
    filters: SectionFilterRules = Field(default_factory=SectionFilterRules)
    # Filings whose kept sections cover less of their source text than this
    # are listed under manifest["coverage"]["warnings"]; nothing is rejected.
    coverage_warning_below: float = Field(default=0.8, ge=0, le=1)

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else [t.strip().upper() for t in v]

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def _consistent(self) -> CorpusBuildConfig:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        if self.resolved_manifest_path == self.output_path:
            raise ValueError("manifest_path must differ from output_path")
        return self

    @property
    def resolved_manifest_path(self) -> Path:
        if self.manifest_path is not None:
            return self.manifest_path
        return manifest_path_for(self.output_path)


def manifest_path_for(corpus_path: str | Path) -> Path:
    """Default manifest location for a corpus: ``<stem>.manifest.json`` beside it."""
    path = Path(corpus_path)
    return path.with_name(f"{path.stem}.manifest.json")


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        (f"{'.'.join(str(p) for p in error['loc'])}: " if error["loc"] else "")
        + error["msg"].removeprefix("Value error, ")
        for error in exc.errors()
    )


def load_corpus_build_config(
    config_path: str | Path,
    *,
    raw_dir: str | Path | None = None,
    output: str | Path | None = None,
    manifest: str | Path | None = None,
    strict: bool = False,
) -> CorpusBuildConfig:
    """Load the shared corpus YAML (e.g. ``configs/data/corpus_v1.yaml``).

    The YAML is first validated exactly as the downloader validates it, so
    unknown or misspelled keys are rejected here too. The raw directory is
    the downloader's ``output_dir``, the corpus path is ``paths.corpus_jsonl``;
    the keyword arguments override them. Errors are one-line
    ``CorpusConfigError``s.
    """
    path = Path(config_path)
    if not path.is_file():
        raise CorpusConfigError(f"config file not found: {path} (pass --config PATH)")
    try:
        raw = read_yaml(path)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise CorpusConfigError(f"invalid YAML in {path}{where}: {getattr(exc, 'problem', exc)}") from None
    except (OSError, ValueError) as exc:
        raise CorpusConfigError(f"cannot load config {path}: {exc}") from None

    try:
        download = EdgarDownloadConfig.model_validate(raw)
    except ValidationError as exc:
        raise CorpusConfigError(f"invalid config settings in {path}: {_format_validation_error(exc)}") from None
    paths = raw.get("paths") or {}
    output_path = output if output is not None else paths.get("corpus_jsonl")
    if output_path is None:
        raise CorpusConfigError(f"{path} has no paths.corpus_jsonl; pass --output PATH")

    settings: dict[str, Any] = {
        "name": raw.get("name", "corpus"),
        "raw_dir": raw_dir if raw_dir is not None else download.output_dir,
        "output_path": output_path,
        "manifest_path": manifest,
        "tickers": download.tickers,
        "form_type": download.form_type,
        "filings_per_ticker": download.filings_per_ticker,
        "strict": strict,
    }
    for key in ("chunk_size_tokens", "chunk_overlap_tokens"):
        if key in raw:
            settings[key] = raw[key]
    try:
        return CorpusBuildConfig.model_validate(settings)
    except ValidationError as exc:
        raise CorpusConfigError(f"invalid corpus settings in {path}: {_format_validation_error(exc)}") from None


# --- raw filings ------------------------------------------------------------------


class FilingIssue(BaseModel):
    """A filing that was rejected (invalid) or excluded (out of scope)."""

    path: str
    reason: str
    detail: str
    ticker: str | None = None
    accession_number: str | None = None


class FilingRejected(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class LoadedFiling:
    """A raw filing whose metadata validated and whose HTML matched its checksum."""

    directory: Path
    metadata: FilingMetadata
    html: bytes

    @property
    def filing_id(self) -> str:
        m = self.metadata
        return make_filing_id(m.ticker, m.form_type, m.accession_number)


def discover_raw_filings(raw_dir: str | Path) -> list[Path]:
    """Directories under ``raw_dir`` holding a ``filing.html`` or ``metadata.json``.

    Directories with only one of the two are returned too, so that
    incomplete downloads are reported rather than silently ignored.
    Sorted for a deterministic build order.
    """
    root = Path(raw_dir)
    if not root.is_dir():
        raise CorpusBuildError(f"raw filing directory not found: {root}")
    found = {
        path.parent
        for name in (HTML_FILENAME, METADATA_FILENAME)
        for path in root.rglob(name)
        if path.is_file()
    }
    return sorted(found)


def load_raw_filing(directory: str | Path) -> LoadedFiling:
    """Load and verify one raw filing directory; raise ``FilingRejected`` if unusable."""
    directory = Path(directory)
    html_path = directory / HTML_FILENAME
    metadata_path = directory / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FilingRejected("missing_metadata", f"{METADATA_FILENAME} not found")
    if not html_path.is_file():
        raise FilingRejected("missing_html", f"{HTML_FILENAME} not found")
    try:
        metadata = FilingMetadata.model_validate_json(metadata_path.read_bytes())
    except ValidationError as exc:
        raise FilingRejected("invalid_metadata", _format_validation_error(exc)) from None
    if metadata.html_sha256 is None:
        raise FilingRejected("missing_checksum", f"{METADATA_FILENAME} has no html_sha256")
    html = html_path.read_bytes()
    digest = hashlib.sha256(html).hexdigest()
    if digest != metadata.html_sha256:
        raise FilingRejected(
            "sha256_mismatch",
            f"{HTML_FILENAME} hashes to {digest[:12]}..., metadata records "
            f"{metadata.html_sha256[:12]}...",
        )
    if not html.strip():
        raise FilingRejected("empty_html", f"{HTML_FILENAME} is empty")
    return LoadedFiling(directory, metadata, html)


# --- quality filtering ------------------------------------------------------------


def section_body(section: Section) -> str:
    """Section text after its ``Item N. Title`` heading line."""
    _, _, body = section.text.partition("\n")
    return body


_REFERENCE_START_RE = re.compile(
    r"^(?:refer to|reference is made to|(?:please )?see\b|for (?:a|an|more|further)\b[^.]{0,100}?\bsee\b"
    r"|(?:the )?information (?:required by|about|regarding)\b[^.]{0,120}?\b(?:is|are|will be)\s+"
    r"(?:incorporated|included|set forth|contained|presented)\b|incorporated (?:herein )?by reference)"
)
_APPEARS_ON_PAGES_RE = re.compile(r"\bappears? on pages?\s+\d")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z\"“(])")


def _is_cross_reference(body: str, max_words: int) -> bool:
    """Whether every sentence of ``body`` only points to content elsewhere.

    A reference followed by any other sentence ("... Our code of ethics is
    posted at example.com.") is kept, since that sentence is content.
    """
    text = re.sub(r"\s+", " ", body).strip()
    if not text or len(text.split()) > max_words or not _REFERENCE_TARGET_RE.search(text.lower()):
        return False
    return all(
        _REFERENCE_START_RE.match(sentence.lower()) or _APPEARS_ON_PAGES_RE.search(sentence.lower())
        for sentence in _SENTENCE_BREAK_RE.split(text)
    )
_REFERENCE_TARGET_RE = re.compile(
    r"\b(?:notes? \d|items? \d|part i|section|pages?\s+\d|heading|caption|proxy statement"
    r"|financial (?:section|report|statements)|this (?:report|form 10-k|annual report))"
)


def section_filter_reason(section: Section, rules: SectionFilterRules) -> str | None:
    """Why ``section`` should be dropped, or ``None`` to keep it.

    A bare 1-3 digit last line (a page number that escaped page-furniture
    removal, as in ``None.`` / ``28``) is ignored for this decision only; the
    section text itself is never changed here.
    """
    body = re.sub(r"\n\s*\d{1,3}\s*$", "", section_body(section))
    if TABLE_START in body:
        return None
    normalized = re.sub(r"\s+", " ", body).strip().lower()
    if not re.search(r"\w", normalized):
        return "empty" if rules.drop_empty_sections else None
    words = len(normalized.split())
    if words > rules.max_boilerplate_words:
        return None
    for name, pattern in rules.boilerplate_patterns.items():
        if re.fullmatch(pattern, normalized):
            return f"boilerplate:{name}"
    if _is_cross_reference(body, rules.cross_reference_max_words):
        return "boilerplate:cross_reference"
    return None


# --- build ------------------------------------------------------------------------------


class FilteredSection(BaseModel):
    filing_id: str
    accession_number: str
    item_number: str | None
    title: str
    reason: str
    words: int


class FilingRecord(BaseModel):
    """Per-filing provenance and source-text coverage recorded in the manifest.

    ``source_words`` counts the filing's own visible text (table cells, not
    rendered labels; page furniture excluded). ``covered_source_words`` is the
    part inside kept sections, and ``excluded_source_words`` accounts for the
    rest by category, including ``filtered_boilerplate``. ``passage_tokens``
    is generated volume — it includes chunk overlap and table labels, so it
    is not comparable with the source counts.
    """

    filing_id: str
    ticker: str
    cik: str
    company_name: str
    form_type: str
    accession_number: str
    filing_date: str
    html_sha256: str
    path: str
    sections_parsed: int
    sections_kept: int
    appended_sections_kept: int
    passages: int
    passage_tokens: int
    source_words: int
    covered_source_words: int
    coverage: float
    excluded_source_words: dict[str, int]
    report_toc_entries: int
    report_toc_unmatched: list[str]


@dataclass
class CorpusBuild:
    """In-memory result of ``build_corpus``; ``write_corpus`` persists it."""

    config: CorpusBuildConfig
    discovered: int
    passages: list[Passage] = field(default_factory=list)
    filings: list[FilingRecord] = field(default_factory=list)
    rejected: list[FilingIssue] = field(default_factory=list)
    excluded: list[FilingIssue] = field(default_factory=list)
    filtered_sections: list[FilteredSection] = field(default_factory=list)
    filtered_passages: Counter[str] = field(default_factory=Counter)
    missing_tickers: list[str] = field(default_factory=list)
    diagnostics: dict[str, int] = field(default_factory=dict)

    @property
    def sections_parsed(self) -> int:
        return sum(f.sections_parsed for f in self.filings)

    @property
    def sections_kept(self) -> int:
        return sum(f.sections_kept for f in self.filings)

    @property
    def problems(self) -> list[str]:
        """Human-readable reasons the build is not complete for its config."""
        issues = [f"rejected {i.path}: {i.reason}: {i.detail}" for i in self.rejected]
        if self.missing_tickers:
            issues.append(f"no filings for configured tickers: {', '.join(self.missing_tickers)}")
        return issues


def _relative(directory: Path, raw_dir: Path) -> str:
    try:
        return directory.relative_to(raw_dir).as_posix()
    except ValueError:
        return directory.as_posix()


def _issue(filing: LoadedFiling | Path, reason: str, detail: str, raw_dir: Path) -> FilingIssue:
    directory = filing.directory if isinstance(filing, LoadedFiling) else filing
    metadata = filing.metadata if isinstance(filing, LoadedFiling) else None
    return FilingIssue(
        path=_relative(directory, raw_dir),
        reason=reason,
        detail=detail,
        ticker=metadata.ticker if metadata else None,
        accession_number=metadata.accession_number if metadata else None,
    )


def _select_filings(
    loaded: list[LoadedFiling], config: CorpusBuildConfig, build: CorpusBuild
) -> list[LoadedFiling]:
    raw_dir = config.raw_dir
    in_scope: list[LoadedFiling] = []
    seen: dict[str, LoadedFiling] = {}
    for filing in loaded:
        m = filing.metadata
        if m.form_type != config.form_type:
            build.excluded.append(
                _issue(filing, "form_type_not_configured", f"{m.form_type} != {config.form_type}", raw_dir)
            )
        elif config.tickers is not None and m.ticker not in config.tickers:
            build.excluded.append(_issue(filing, "ticker_not_configured", m.ticker, raw_dir))
        elif m.accession_number in seen:
            first = _relative(seen[m.accession_number].directory, raw_dir)
            build.rejected.append(
                _issue(filing, "duplicate_accession", f"{m.accession_number} already loaded from {first}", raw_dir)
            )
        else:
            seen[m.accession_number] = filing
            in_scope.append(filing)

    if config.filings_per_ticker is not None:
        by_ticker: dict[str, list[LoadedFiling]] = defaultdict(list)
        for filing in in_scope:
            by_ticker[filing.metadata.ticker].append(filing)
        in_scope = []
        for filings in by_ticker.values():
            filings.sort(key=lambda f: (f.metadata.filing_date, f.metadata.accession_number), reverse=True)
            in_scope.extend(filings[: config.filings_per_ticker])
            for filing in filings[config.filings_per_ticker:]:
                build.excluded.append(
                    _issue(filing, "beyond_filings_per_ticker",
                           f"older than the newest {config.filings_per_ticker}", raw_dir)
                )
    return sorted(
        in_scope,
        key=lambda f: (f.metadata.ticker, f.metadata.filing_date, f.metadata.accession_number),
    )


def _contains_table(text: str) -> bool:
    return TABLE_START in text or TABLE_END in text or " | " in text


def _row_spans(section_text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Table rows of a section and their token spans, as the chunker tokenises it."""
    rows: list[str] = []
    spans: list[tuple[int, int]] = []
    offset, inside = 0, False
    for line in section_text.split("\n"):
        tokens = line.split()
        stripped = " ".join(tokens)
        if stripped == TABLE_START:
            inside = True
        elif stripped == TABLE_END:
            inside = False
        elif inside and tokens:
            rows.append(stripped)
            spans.append((offset, offset + len(tokens)))
        offset += len(tokens)
    return rows, spans


def _locate_passages(tokens: list[str], passages: list[Passage]) -> list[tuple[int, int] | None]:
    """Token span of each passage within its section, in order; ``None`` if not a slice.

    Passages are contiguous token windows of the section, except labelled
    continuation pieces of over-long rows, which carry ``(continued)`` text
    that is not in the section and so cannot be located.
    """
    positions: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(tokens):
        positions[token].append(index)
    located: list[tuple[int, int] | None] = []
    previous = -1
    for passage in passages:
        part = passage.text.split()
        starts = positions.get(part[0], []) if part else []
        found = next(
            (s for s in starts[bisect_right(starts, previous):] if tokens[s:s + len(part)] == part),
            None,
        )
        located.append(None if found is None else (found, found + len(part)))
        if found is not None:
            previous = found
    return located


def _represented(row: str, passages: list[Passage]) -> bool:
    """Whether ``row`` appears whole in a passage or through labelled continuation."""
    if any(row in p.text for p in passages):
        return True
    label, _, rest = row.partition(" | ")
    if not rest:
        label, rest = "", row
    labelled = [p.text for p in passages if label in p.text]
    vocab = {token for text in labelled for token in text.split()}
    # A cell longer than one piece is itself split, so fall back to its tokens.
    return all(
        any(cell in text for text in labelled) or all(t in vocab for t in cell.split())
        for cell in rest.split(" | ")
    )


def _table_diagnostics(section_text: str, passages: list[Passage], continued_rows: int) -> Counter[str]:
    """Exact table-boundary checks for one section's passages.

    Each passage is located as a token span of the section, and its start and
    end are compared with the token spans of the section's table rows: a
    boundary strictly inside a row is a real mid-row cut, whatever the text
    looks like. Passages that are not plain slices (labelled continuation
    pieces) are counted in ``passages_not_located`` instead of being guessed.
    """
    rows, spans = _row_spans(section_text)
    diagnostics: Counter[str] = Counter(
        table_rows=len(rows), table_rows_requiring_continuation=continued_rows
    )
    if not rows:
        return diagnostics
    tokens = section_text.split()
    inside = bytearray(len(tokens) + 1)  # 1 where a boundary would fall strictly inside a row
    for a, e in spans:
        inside[a + 1:e] = b"\x01" * (e - a - 1)
    located = _locate_passages(tokens, passages)
    for i, span in enumerate(located):
        if span is None:
            diagnostics["passages_not_located"] += 1
            continue
        start, end = span
        if i > 0 and inside[start]:
            # A located passage that starts inside a row lacks that row's label.
            diagnostics["passages_starting_mid_table_row"] += 1
            diagnostics["unlabelled_partial_row_fragments"] += 1
        if i < len(passages) - 1 and inside[end]:
            diagnostics["passages_ending_mid_table_row"] += 1
    found = [s for s in located if s is not None]
    starts = [s for s, _ in found]
    for row, (a, e) in zip(rows, spans, strict=True):
        k = bisect_right(starts, a) - 1  # windows' ends grow with their starts
        if not (k >= 0 and found[k][1] >= e) and not _represented(row, passages):
            diagnostics["table_rows_not_represented"] += 1
    return diagnostics


def _check_strict(build: CorpusBuild) -> None:
    if build.config.strict and build.problems:
        raise StrictModeError(
            f"strict mode: {len(build.problems)} problem(s); first: {build.problems[0]}", build
        )


def build_corpus(config: CorpusBuildConfig) -> CorpusBuild:
    """Validate, parse, filter and chunk every filing under ``config.raw_dir``.

    Nothing is written. Raises ``StrictModeError`` in strict mode if any
    filing is rejected or a configured ticker has no filing, and
    ``CorpusBuildError`` if no passages can be produced at all.
    """
    directories = discover_raw_filings(config.raw_dir)
    build = CorpusBuild(config=config, discovered=len(directories))

    loaded: list[LoadedFiling] = []
    for directory in directories:
        try:
            loaded.append(load_raw_filing(directory))
        except FilingRejected as exc:
            build.rejected.append(_issue(directory, exc.reason, exc.detail, config.raw_dir))
    selected = _select_filings(loaded, config, build)
    if config.tickers is not None:
        present = {f.metadata.ticker for f in selected}
        build.missing_tickers = [t for t in config.tickers if t not in present]
    _check_strict(build)

    rules = config.filters
    first_filing_for_text: dict[str, str] = {}
    diagnostics: Counter[str] = Counter(dict.fromkeys(DIAGNOSTIC_KEYS, 0))
    for filing in selected:
        m = filing.metadata
        try:
            parsed = analyze_filing(filing.html, filing.filing_id)
        except Exception as exc:  # unexpected markup or parser bug: report, never drop silently
            build.rejected.append(
                _issue(filing, "parse_error", f"{type(exc).__name__}: {exc}", config.raw_dir)
            )
            continue
        sections, coverage = parsed.sections, parsed.coverage
        if not sections:
            build.rejected.append(
                _issue(filing, "no_sections", "no Item sections could be identified", config.raw_dir)
            )
            continue

        filing_passages: list[Passage] = []
        seen_texts: set[str] = set()
        kept = kept_appended = filtered_words = 0
        for index, section in enumerate(sections):
            reason = section_filter_reason(section, rules)
            if reason is not None:
                build.filtered_sections.append(
                    FilteredSection(
                        filing_id=filing.filing_id,
                        accession_number=m.accession_number,
                        item_number=section.item_number,
                        title=section.title,
                        reason=reason,
                        words=len(section_body(section).split()),
                    )
                )
                filtered_words += coverage.section_words[index]
                continue
            kept += 1
            kept_appended += section.item_number is None
            chunks = chunk_section(
                section,
                ticker=m.ticker,
                form_type=m.form_type,
                filing_date=m.filing_date.isoformat(),
                chunk_size_tokens=config.chunk_size_tokens,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
            )
            if TABLE_START in section.text:
                continued = chunk_text_table_aware(
                    section.text, config.chunk_size_tokens, config.chunk_overlap_tokens
                ).continued_rows
                diagnostics.update(_table_diagnostics(section.text, chunks, continued))
            for passage in chunks:
                if rules.drop_duplicate_passages and passage.text in seen_texts:
                    build.filtered_passages["duplicate_text_in_filing"] += 1
                    continue
                seen_texts.add(passage.text)
                origin = first_filing_for_text.setdefault(passage.text, filing.filing_id)
                if origin != filing.filing_id:
                    diagnostics["passages_duplicating_another_filing"] += 1
                metadata = {
                    **passage.metadata,
                    "accession_number": m.accession_number,
                    "cik": m.cik,
                    "company_name": m.company_name,
                    "report_date": m.report_date.isoformat() if m.report_date else None,
                    "section_index": index,
                    "section_kind": "item" if section.item_number else "appended_report",
                    "num_chunks": len(chunks),
                    "contains_table": _contains_table(passage.text),
                }
                filing_passages.append(passage.model_copy(update={"metadata": metadata}))

        build.passages.extend(filing_passages)
        excluded = {k: v for k, v in coverage.words_by_category.items() if k != "sections"}
        excluded["filtered_boilerplate"] = filtered_words
        covered = coverage.words_by_category["sections"] - filtered_words
        build.filings.append(
            FilingRecord(
                filing_id=filing.filing_id,
                ticker=m.ticker,
                cik=m.cik,
                company_name=m.company_name,
                form_type=m.form_type,
                accession_number=m.accession_number,
                filing_date=m.filing_date.isoformat(),
                html_sha256=m.html_sha256 or "",
                path=_relative(filing.directory, config.raw_dir),
                sections_parsed=len(sections),
                sections_kept=kept,
                appended_sections_kept=kept_appended,
                passages=len(filing_passages),
                passage_tokens=sum(p.token_count or 0 for p in filing_passages),
                source_words=coverage.source_words,
                covered_source_words=covered,
                coverage=round(covered / coverage.source_words, 4) if coverage.source_words else 0.0,
                excluded_source_words=excluded,
                report_toc_entries=coverage.report_toc_entries,
                report_toc_unmatched=list(coverage.report_toc_unmatched),
            )
        )

    diagnostics["max_passage_tokens"] = max((p.token_count or 0 for p in build.passages), default=0)
    build.diagnostics = dict(diagnostics)
    _check_strict(build)
    if not build.passages:
        raise CorpusBuildError(
            f"no passages could be built from {build.discovered} filing director"
            f"{'y' if build.discovered == 1 else 'ies'} under {config.raw_dir}"
            + (f"; {build.problems[0]}" if build.problems else "")
        )
    _check_invariants(build)
    return build


def _check_invariants(build: CorpusBuild) -> None:
    ids = Counter(p.passage_id for p in build.passages)
    duplicates = [pid for pid, n in ids.items() if n > 1]
    if duplicates:
        raise CorpusBuildError(f"duplicate passage IDs: {duplicates[:3]}")
    limit = build.config.chunk_size_tokens
    over = [p.passage_id for p in build.passages if (p.token_count or 0) > limit]
    if over:
        raise CorpusBuildError(f"passages exceed {limit} tokens: {over[:3]}")


# --- output -----------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _provenance() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit else None
    return {
        "version": CORPUS_BUILDER_VERSION,
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "source_sha256": {
            module: hashlib.sha256((package_root / module).read_bytes()).hexdigest()
            for module in _SOURCE_MODULES
        },
        "python": platform.python_version(),
        "json_serializer": "orjson" if importlib.util.find_spec("orjson") else "json",
    }


def _coverage_summary(build: CorpusBuild) -> dict[str, Any]:
    by_ticker: dict[str, list[float]] = defaultdict(list)
    for record in build.filings:
        by_ticker[record.ticker].append(record.coverage)
    threshold = build.config.coverage_warning_below
    return {
        "measure": "share of source words (filing text after page furniture; table cells, "
        "not rendered labels) inside kept sections",
        "warning_below": threshold,
        "by_ticker": {
            ticker: {"min": min(v), "max": max(v), "mean": round(sum(v) / len(v), 4)}
            for ticker, v in sorted(by_ticker.items())
        },
        "warnings": [
            {"ticker": r.ticker, "filing_date": r.filing_date, "accession_number": r.accession_number,
             "coverage": r.coverage, "unassigned_words": r.excluded_source_words.get("unassigned", 0)}
            for r in build.filings
            if r.coverage < threshold
        ],
    }


def build_manifest(
    build: CorpusBuild, *, corpus_sha256: str, corpus_bytes: int, built_at: datetime
) -> dict[str, Any]:
    """Machine-readable record of how the corpus was built."""
    config = build.config
    return {
        "manifest_version": MANIFEST_VERSION,
        "corpus_version": f"{config.name}@{CORPUS_BUILDER_VERSION}",
        "built_at": built_at.astimezone(UTC).isoformat(timespec="seconds"),
        "corpus": {
            "name": config.name,
            "path": config.output_path.as_posix(),
            "sha256": corpus_sha256,
            "bytes": corpus_bytes,
            "passages": len(build.passages),
        },
        "settings": {
            "raw_dir": config.raw_dir.as_posix(),
            "form_type": config.form_type,
            "tickers": config.tickers,
            "filings_per_ticker": config.filings_per_ticker,
            "chunk_size_tokens": config.chunk_size_tokens,
            "chunk_overlap_tokens": config.chunk_overlap_tokens,
            "strict": config.strict,
            "filters": config.filters.model_dump(),
        },
        "counts": {
            "filing_directories_discovered": build.discovered,
            "filings_included": len(build.filings),
            "filings_rejected": len(build.rejected),
            "filings_excluded": len(build.excluded),
            "sections_parsed": build.sections_parsed,
            "sections_kept": build.sections_kept,
            "sections_filtered": len(build.filtered_sections),
            "passages": len(build.passages),
            "passages_filtered": sum(build.filtered_passages.values()),
        },
        "filtered_sections_by_reason": dict(sorted(Counter(s.reason for s in build.filtered_sections).items())),
        "filtered_passages_by_reason": dict(sorted(build.filtered_passages.items())),
        "source_accessions": [f.accession_number for f in build.filings],
        "filings": [f.model_dump() for f in build.filings],
        "filtered_sections": [s.model_dump() for s in build.filtered_sections],
        "rejected_filings": [i.model_dump() for i in build.rejected],
        "excluded_filings": [i.model_dump() for i in build.excluded],
        "missing_tickers": build.missing_tickers,
        "coverage": _coverage_summary(build),
        "diagnostics": build.diagnostics,
        "builder": _provenance(),
    }


def write_corpus(build: CorpusBuild, *, built_at: datetime | None = None) -> dict[str, Any]:
    """Write the corpus JSONL and its manifest, each atomically; return the manifest.

    The corpus is replaced first and the manifest second, so a crash in
    between leaves a manifest whose ``corpus.sha256`` no longer matches.
    """
    config = build.config
    write_jsonl(config.output_path, build.passages, atomic=True)
    data = config.output_path.read_bytes()
    manifest = build_manifest(
        build,
        corpus_sha256=hashlib.sha256(data).hexdigest(),
        corpus_bytes=len(data),
        built_at=built_at or datetime.now(UTC),
    )
    _atomic_write_text(config.resolved_manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = [
    "CORPUS_BUILDER_VERSION",
    "DEFAULT_BOILERPLATE_PATTERNS",
    "CorpusBuild",
    "CorpusBuildConfig",
    "CorpusBuildError",
    "CorpusConfigError",
    "FilingIssue",
    "FilingRecord",
    "FilingRejected",
    "FilteredSection",
    "LoadedFiling",
    "SectionFilterRules",
    "StrictModeError",
    "build_corpus",
    "build_manifest",
    "discover_raw_filings",
    "load_corpus_build_config",
    "load_raw_filing",
    "manifest_path_for",
    "section_body",
    "section_filter_reason",
    "write_corpus",
]
