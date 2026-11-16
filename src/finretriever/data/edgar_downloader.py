"""Download raw SEC filings directly from the official EDGAR endpoints.

Three official endpoints are used, and nothing else:

* ``https://www.sec.gov/files/company_tickers.json`` — ticker → CIK mapping.
* ``https://data.sec.gov/submissions/CIK##########.json`` — a company's filing
  history, newest first. ``filings.recent`` holds roughly the latest 1,000
  filings; older filings live in paged files listed under ``filings.files``.
  Prolific filers (e.g. JPM, with thousands of prospectus supplements a year)
  need those pages to reach seven years of 10-Ks.
* ``https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<document>`` —
  the filing documents themselves.

SEC fair-access policy (https://www.sec.gov/os/accessing-edgar-data) requires
every request to carry a User-Agent that identifies the requester with a
contact email, and caps traffic at 10 requests/second. ``EdgarClient`` reads
the User-Agent from ``SEC_USER_AGENT`` and refuses to run without one, spaces
requests by ``request_delay_seconds`` (0.2s by default, i.e. at most
5 requests/second), and retries 429/5xx responses with bounded exponential
backoff.

Raw filings are stored idempotently as::

    <output_dir>/<TICKER>/<FORM>/<accession-number>/filing.html
                                                   /metadata.json

``filing.html`` is the primary document byte-for-byte as served by the SEC.
``metadata.json`` records where it came from plus its SHA-256, which is what
lets a re-run recognise a complete download and skip it.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol
from urllib.parse import quote

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

SEC_USER_AGENT_ENV = "SEC_USER_AGENT"

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"
ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

HTML_FILENAME = "filing.html"
METADATA_FILENAME = "metadata.json"

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0
_RETRY_AFTER_MAX_SECONDS = 60.0

_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SUBMISSIONS_PAGE_RE = re.compile(r"^CIK\d{10}-submissions-\d{3,}\.json$")
_REQUIRED_FILING_COLUMNS = ("accessionNumber", "filingDate", "form", "primaryDocument")

# Keys of the shared corpus YAML that belong to later pipeline stages (corpus
# naming, chunking, processed-data layout). The downloader accepts and ignores
# them; any other key that is not a downloader field is rejected, so a typo
# such as ``filing_per_ticker`` cannot silently fall back to a default.
CORPUS_LATER_STAGE_KEYS = frozenset({"name", "chunk_size_tokens", "chunk_overlap_tokens", "paths"})
CORPUS_PATHS_KEYS = frozenset({"interim_dir", "processed_dir", "corpus_jsonl"})


class EdgarError(RuntimeError):
    """Base class for SEC EDGAR acquisition errors."""


class MissingUserAgentError(EdgarError):
    """``SEC_USER_AGENT`` is unset or does not identify a contact."""


class UnknownTickerError(EdgarError):
    """A configured ticker is absent from the SEC ticker mapping."""


class EdgarResponseError(EdgarError):
    """An SEC response was empty or did not have the expected structure."""


class EdgarRequestError(EdgarError):
    """An HTTP request failed permanently or exhausted its retries."""

    def __init__(
        self,
        message: str,
        *,
        url: str,
        status_code: int | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.attempts = attempts


class EdgarDownloadConfig(BaseModel):
    """Downloader settings, usually loaded from ``configs/data/corpus_v1.yaml``.

    The same corpus YAML also carries settings for later pipeline stages.
    Those keys (``CORPUS_LATER_STAGE_KEYS``, and ``CORPUS_PATHS_KEYS`` inside
    ``paths``) are accepted and ignored; every other unknown key is rejected
    with a "did you mean" hint.

    ``cik_overrides`` maps a configured ticker to the registrant CIK whose
    filings should be used instead of the CIK the SEC ticker map currently
    gives it — e.g. when a holding-company reorganisation moved the ticker
    to a new registrant that has no 10-K history yet. It is applied only to
    the tickers listed; nothing is ever substituted automatically.
    """

    model_config = ConfigDict(extra="ignore")

    tickers: list[str] = Field(min_length=1)
    form_type: str = "10-K"
    filings_per_ticker: int = Field(default=7, gt=0)
    output_dir: Path = Path("data/raw")
    request_delay_seconds: float = Field(default=0.2, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    cik_overrides: dict[str, str] = Field(default_factory=dict)
    # Optional pin: ticker -> the exact accession numbers to download, instead of the newest
    # ``filings_per_ticker``. A pinned corpus stays the same after companies file new 10-Ks.
    accessions: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        problems = _unknown_key_errors(data, set(cls.model_fields) | CORPUS_LATER_STAGE_KEYS)
        paths = data.get("paths")
        if paths is not None:
            if isinstance(paths, Mapping):
                problems += _unknown_key_errors(paths, CORPUS_PATHS_KEYS, prefix="paths.")
            else:
                problems.append(
                    f"'paths' must be a mapping of {sorted(CORPUS_PATHS_KEYS)} to paths, "
                    f"got {type(paths).__name__}"
                )
        if problems:
            raise ValueError("; ".join(problems))
        return data

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, v: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in v:
            ticker = raw.strip().upper()
            if not ticker:
                raise ValueError("tickers must not contain blank entries")
            if ticker not in normalized:
                normalized.append(ticker)
        return normalized

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, v: str) -> str:
        form_type = v.strip().upper()
        if not form_type:
            raise ValueError("form_type must be non-empty")
        return form_type

    @field_validator("cik_overrides", mode="before")
    @classmethod
    def _normalize_cik_overrides(cls, v: Any) -> Any:
        if not isinstance(v, Mapping):
            return v
        normalized: dict[str, str] = {}
        for ticker, cik in v.items():
            if not isinstance(cik, str):
                # YAML reads unquoted numbers with leading zeros unreliably
                # (0000034077 would be octal), so require the quoted form.
                raise ValueError(
                    f"cik_overrides.{ticker}: CIK must be a quoted string such as "
                    f"'0000034088', got {type(cik).__name__} {cik!r}"
                )
            try:
                normalized[str(ticker).strip().upper()] = normalize_cik(cik)
            except ValueError as exc:
                raise ValueError(f"cik_overrides.{ticker}: {exc}") from None
        return normalized

    @field_validator("accessions", mode="before")
    @classmethod
    def _normalize_accessions(cls, v: Any) -> Any:
        if not isinstance(v, Mapping):
            return v
        normalized: dict[str, list[str]] = {}
        for ticker, numbers in v.items():
            if not isinstance(numbers, list) or not numbers:
                raise ValueError(f"accessions.{ticker}: expected a non-empty list of accession numbers")
            cleaned: list[str] = []
            for number in numbers:
                if not isinstance(number, str) or not _ACCESSION_RE.match(number.strip()):
                    raise ValueError(f"accessions.{ticker}: {number!r} is not an accession number "
                                     "such as '0000320193-23-000106' (quote it)")
                if number.strip() in cleaned:
                    raise ValueError(f"accessions.{ticker}: {number.strip()} is listed twice")
                cleaned.append(number.strip())
            normalized[str(ticker).strip().upper()] = cleaned
        return normalized

    @model_validator(mode="after")
    def _overrides_name_configured_tickers(self) -> EdgarDownloadConfig:
        stray = sorted(set(self.cik_overrides) - set(self.tickers))
        if stray:
            raise ValueError(f"cik_overrides for tickers not in tickers: {', '.join(stray)}")
        stray = sorted(set(self.accessions) - set(self.tickers))
        if stray:
            raise ValueError(f"accessions for tickers not in tickers: {', '.join(stray)}")
        return self


def _unknown_key_errors(
    data: Mapping[Any, Any], allowed: set[str] | frozenset[str], prefix: str = ""
) -> list[str]:
    errors: list[str] = []
    for key in data:
        if key in allowed:
            continue
        close = difflib.get_close_matches(str(key), sorted(allowed), n=1)
        hint = (
            f"did you mean '{prefix}{close[0]}'?"
            if close
            else f"allowed: {', '.join(prefix + name for name in sorted(allowed))}"
        )
        errors.append(f"unknown config key '{prefix}{key}' ({hint})")
    return errors


CikSource = Literal["ticker_map", "cik_override"]


class CompanyInfo(BaseModel):
    """One row of the SEC ticker mapping."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    cik: str
    company_name: str
    cik_source: CikSource = "ticker_map"


class FilingMetadata(BaseModel):
    """Provenance for one filing; serialised as ``metadata.json``.

    ``local_html_path`` and ``html_sha256`` are filled in once the document
    has been written to disk. ``cik_source`` says whether ``cik`` came from
    the SEC ticker map or from a configured ``cik_overrides`` entry; in the
    latter case ``company_name`` is the registrant's name from its SEC
    submissions record.
    """

    model_config = ConfigDict(extra="forbid")

    ticker: str
    cik: str
    company_name: str
    form_type: str
    accession_number: str
    filing_date: date
    report_date: date | None = None
    primary_document: str
    filing_url: str
    local_html_path: str | None = None
    html_sha256: str | None = None
    cik_source: CikSource = "ticker_map"

    @field_validator("cik")
    @classmethod
    def _ten_digit_cik(cls, v: str) -> str:
        if not (len(v) == 10 and v.isdigit()):
            raise ValueError(f"cik must be a 10-digit zero-padded string, got {v!r}")
        return v

    @field_validator("accession_number")
    @classmethod
    def _accession_format(cls, v: str) -> str:
        if not _ACCESSION_RE.match(v):
            raise ValueError(f"accession_number must look like 0000320193-23-000106, got {v!r}")
        return v


class DownloadResult(BaseModel):
    """Outcome of ``download_filing`` for one filing."""

    metadata: FilingMetadata
    status: Literal["downloaded", "skipped"]


class DownloadFailure(BaseModel):
    """A ticker or filing that could not be acquired, with the reason."""

    ticker: str
    accession_number: str | None = None
    error: str


class DownloadSummary(BaseModel):
    """Result of ``download_corpus``."""

    requested_tickers: list[str]
    output_dir: Path
    dry_run: bool = False
    selected: list[FilingMetadata] = Field(default_factory=list)
    downloaded: int = 0
    skipped: int = 0
    failures: list[DownloadFailure] = Field(default_factory=list)
    cik_overrides_used: list[CikOverrideUse] = Field(default_factory=list)

    @property
    def filings_selected(self) -> int:
        return len(self.selected)


class CikOverrideUse(BaseModel):
    """A configured ``cik_overrides`` entry applied during a run."""

    ticker: str
    cik: str
    company_name: str
    ticker_map_cik: str | None  # what the SEC ticker map says today, if anything
    ticker_map_company_name: str | None


def normalize_cik(cik: str | int) -> str:
    """Return ``cik`` as the 10-digit zero-padded string EDGAR uses.

    The ticker mapping reports CIKs as integers (``320193``) while the
    submissions API expects ``CIK0000320193.json``; normalising once avoids
    mixing the two forms.
    """
    if isinstance(cik, bool):
        raise TypeError("cik must be a str or int, not bool")
    if isinstance(cik, int):
        text = str(cik)
    elif isinstance(cik, str):
        text = cik.strip()
    else:
        raise TypeError(f"cik must be a str or int, got {type(cik).__name__}")
    if not text.isdigit() or len(text) > 10 or int(text) == 0:
        raise ValueError(f"Invalid CIK {cik!r}: expected 1-10 digits, not all zero")
    return text.zfill(10)


def accession_without_dashes(accession_number: str) -> str:
    """Strip dashes from an accession number (``0000320193-23-000106`` → 18 digits).

    The Archives path uses the dash-free form as the filing folder name.
    """
    if not isinstance(accession_number, str) or not _ACCESSION_RE.match(accession_number):
        raise ValueError(
            f"Invalid accession number {accession_number!r}: expected ##########-##-######"
        )
    return accession_number.replace("-", "")


def build_filing_url(cik: str | int, accession_number: str, primary_document: str) -> str:
    """Build the sec.gov Archives URL for a filing's primary document.

    Format: ``https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<document>``
    where ``<cik>`` has **no** leading zeros and ``<accession>`` has **no**
    dashes, e.g. ``.../data/320193/000032019323000106/aapl-20230930.htm``.
    The document name is validated and percent-encoded so a malformed
    submissions row cannot point outside the filing folder.
    """
    cik_path = str(int(normalize_cik(cik)))
    accession_path = accession_without_dashes(accession_number)
    document = _validate_primary_document(primary_document)
    return f"{ARCHIVES_BASE_URL}/{cik_path}/{accession_path}/{quote(document)}"


def _validate_primary_document(primary_document: str) -> str:
    if not isinstance(primary_document, str) or not primary_document.strip():
        raise ValueError("primary_document must be a non-empty string")
    if primary_document != primary_document.strip():
        raise ValueError(f"primary_document has surrounding whitespace: {primary_document!r}")
    if "\\" in primary_document or primary_document.startswith("/"):
        raise ValueError(f"primary_document must be a relative path: {primary_document!r}")
    if any(part in ("", ".", "..") for part in primary_document.split("/")):
        raise ValueError(f"primary_document has an invalid path segment: {primary_document!r}")
    return primary_document


def resolve_user_agent(user_agent: str | None = None) -> str:
    """Return the SEC User-Agent, falling back to ``$SEC_USER_AGENT``.

    The SEC asks for ``"<organisation or name> <contact email>"`` and blocks
    undeclared clients with HTTP 403. No default is provided on purpose: the
    identity must be the operator's own.
    """
    value = user_agent if user_agent is not None else os.environ.get(SEC_USER_AGENT_ENV, "")
    value = value.strip()
    if not value:
        raise MissingUserAgentError(
            f"{SEC_USER_AGENT_ENV} is not set. SEC EDGAR requires a descriptive "
            "User-Agent with a contact email, e.g. "
            f'export {SEC_USER_AGENT_ENV}="FinRetriever research you@your-domain.com"'
        )
    if "@" not in value:
        raise MissingUserAgentError(
            f"{SEC_USER_AGENT_ENV}={value!r} is missing a contact email; SEC fair-access "
            "policy requires one in the User-Agent."
        )
    return value


class _HTTPSession(Protocol):
    def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> Any: ...


class EdgarClient:
    """Throttled, retrying HTTP GET client for SEC endpoints.

    Every request waits until at least ``request_delay_seconds`` have passed
    since the previous one. Responses 429/500/502/503/504 and connection
    errors/timeouts are retried up to ``max_retries`` times with exponential
    backoff (1s, 2s, 4s, ... capped at 30s); a 429 ``Retry-After`` header is
    honoured up to 60s. Any other non-2xx status raises immediately.

    ``session``, ``sleep`` and ``clock`` are injectable so tests never touch
    the network or wait in real time.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        request_delay_seconds: float = 0.2,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        session: _HTTPSession | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if request_delay_seconds < 0:
            raise ValueError("request_delay_seconds must be >= 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.user_agent = resolve_user_agent(user_agent)
        self.request_delay_seconds = request_delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    @classmethod
    def from_config(
        cls,
        config: EdgarDownloadConfig,
        *,
        user_agent: str | None = None,
        session: _HTTPSession | None = None,
    ) -> EdgarClient:
        return cls(
            user_agent,
            request_delay_seconds=config.request_delay_seconds,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            session=session,
        )

    @property
    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    def get(self, url: str) -> bytes:
        """GET ``url`` and return the response body."""
        attempts = self.max_retries + 1
        status_code: int | None = None
        last_error = ""
        for attempt in range(1, attempts + 1):
            self._throttle()
            retry_after: float | None = None
            try:
                response = self._session.get(
                    url, headers=self.headers, timeout=self.timeout_seconds
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                status_code = None
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                status_code = response.status_code
                if 200 <= status_code < 300:
                    return response.content
                if status_code not in RETRYABLE_STATUS_CODES:
                    hint = (
                        " (SEC returns 403 when the User-Agent is missing or not compliant)"
                        if status_code == 403
                        else ""
                    )
                    raise EdgarRequestError(
                        f"GET {url} failed with HTTP {status_code}{hint}",
                        url=url,
                        status_code=status_code,
                        attempts=attempt,
                    )
                last_error = f"HTTP {status_code}"
                if status_code == 429:
                    retry_after = _parse_retry_after(response.headers)
            if attempt < attempts:
                self._sleep(self._backoff_seconds(attempt, retry_after))
        raise EdgarRequestError(
            f"GET {url} failed after {attempts} attempt(s); last error: {last_error}",
            url=url,
            status_code=status_code,
            attempts=attempts,
        )

    def get_json(self, url: str) -> Any:
        """GET ``url`` and decode it as JSON; empty or invalid bodies raise."""
        body = self.get(url)
        if not body.strip():
            raise EdgarResponseError(f"Empty response body from {url}")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise EdgarResponseError(f"Invalid JSON from {url}: {exc}") from exc

    def _throttle(self) -> None:
        if self._last_request_at is not None and self.request_delay_seconds > 0:
            wait = self.request_delay_seconds - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    @staticmethod
    def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, _RETRY_AFTER_MAX_SECONDS)
        return min(_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), _BACKOFF_MAX_SECONDS)


def _parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:  # HTTP-date form; fall back to exponential backoff
        return None
    return seconds if seconds >= 0 else None


def parse_company_tickers(payload: Any) -> dict[str, CompanyInfo]:
    """Parse ``company_tickers.json`` into ``{TICKER: CompanyInfo}``.

    The payload is an object keyed by row index:
    ``{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}``.
    Several tickers can share one CIK (share classes); each ticker maps once.
    """
    if not isinstance(payload, Mapping):
        raise EdgarResponseError(
            f"company_tickers.json: expected a JSON object, got {type(payload).__name__}"
        )
    if not payload:
        raise EdgarResponseError("company_tickers.json: mapping is empty")
    companies: dict[str, CompanyInfo] = {}
    for key, row in payload.items():
        try:
            ticker = str(row["ticker"]).strip().upper()
            info = CompanyInfo(
                ticker=ticker,
                cik=normalize_cik(row["cik_str"]),
                company_name=str(row["title"]).strip(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EdgarResponseError(
                f"company_tickers.json: malformed row {key!r}: {exc!r}"
            ) from exc
        companies.setdefault(ticker, info)
    return companies


def load_company_tickers(client: EdgarClient) -> dict[str, CompanyInfo]:
    """Fetch and parse the official SEC ticker → CIK mapping."""
    return parse_company_tickers(client.get_json(COMPANY_TICKERS_URL))


def resolve_ticker(companies: Mapping[str, CompanyInfo], ticker: str) -> CompanyInfo:
    """Look up ``ticker`` in the SEC mapping, raising ``UnknownTickerError``."""
    key = ticker.strip().upper()
    try:
        return companies[key]
    except KeyError:
        raise UnknownTickerError(
            f"Ticker {key!r} is not in the SEC company_tickers.json mapping "
            "(SEC uses '-' for share classes, e.g. BRK-B)"
        ) from None


def submissions_url(cik: str | int) -> str:
    """URL of a company's submissions JSON on data.sec.gov."""
    return f"{SUBMISSIONS_BASE_URL}/CIK{normalize_cik(cik)}.json"


def select_recent_filings(
    submissions: Any,
    company: CompanyInfo,
    form_type: str = "10-K",
    limit: int = 7,
) -> list[FilingMetadata]:
    """Select the newest ``limit`` filings of ``form_type`` from ``filings.recent``.

    The SEC serves filings as parallel arrays (``accessionNumber[i]``,
    ``form[i]``, ``filingDate[i]``, ...), normally newest first. Selection
    does not rely on that order: see ``_newest_first``. ``form_type`` must
    match exactly, so ``10-K`` excludes amendments (``10-K/A``) and legacy
    ``10-K405`` filings.
    """
    return select_filings_from_columns(
        _recent_columns(submissions, company),
        company,
        form_type,
        limit,
        source=f"submissions for {company.ticker} (filings.recent)",
    )


def select_filings_from_columns(
    columns: Any,
    company: CompanyInfo,
    form_type: str,
    limit: int,
    *,
    source: str,
) -> list[FilingMetadata]:
    """Select the newest ``limit`` filings from one columnar block.

    ``columns`` is ``filings.recent`` or the top level of an older page.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")
    rows = _matching_rows(columns, form_type, source=source)
    return [_row_to_metadata(row, company) for row in _newest_first(rows)[:limit]]


class _FilingRow(NamedTuple):
    """One unvalidated row of a columnar filings block that matched the form type."""

    source: str
    index: int
    form_type: str
    accession_number: Any
    filing_date: Any
    report_date: Any
    primary_document: Any


def _recent_columns(submissions: Any, company: CompanyInfo) -> Any:
    if not isinstance(submissions, Mapping):
        raise EdgarResponseError(
            f"submissions for {company.ticker}: expected a JSON object, "
            f"got {type(submissions).__name__}"
        )
    filings = submissions.get("filings")
    if not isinstance(filings, Mapping) or "recent" not in filings:
        raise EdgarResponseError(
            f"submissions for {company.ticker}: missing 'filings.recent' block"
        )
    return filings["recent"]


def _matching_rows(columns: Any, form_type: str, *, source: str) -> list[_FilingRow]:
    if not isinstance(columns, Mapping):
        raise EdgarResponseError(
            f"{source}: expected an object of filing columns, got {type(columns).__name__}"
        )
    missing = [name for name in _REQUIRED_FILING_COLUMNS if name not in columns]
    if missing:
        raise EdgarResponseError(f"{source}: missing filing columns {missing}")
    column_names = [*_REQUIRED_FILING_COLUMNS, *(["reportDate"] if "reportDate" in columns else [])]
    lengths = set()
    for name in column_names:
        if not isinstance(columns[name], list):
            raise EdgarResponseError(f"{source}: column {name!r} is not a list")
        lengths.add(len(columns[name]))
    if len(lengths) != 1:
        raise EdgarResponseError(f"{source}: filing columns have mismatched lengths")

    wanted = form_type.strip().upper()
    report_dates = columns.get("reportDate")
    return [
        _FilingRow(
            source=source,
            index=i,
            form_type=wanted,
            accession_number=columns["accessionNumber"][i],
            filing_date=columns["filingDate"][i],
            report_date=(report_dates[i] or None) if report_dates else None,
            primary_document=columns["primaryDocument"][i],
        )
        for i, form in enumerate(columns["form"])
        if str(form).strip().upper() == wanted
    ]


def _newest_first(rows: Iterable[_FilingRow]) -> list[_FilingRow]:
    """Deduplicate by accession number (first occurrence wins), newest first.

    Rows are sorted by ``(filingDate, accessionNumber)`` descending, so the
    result does not depend on the order in which the SEC returns blocks,
    pages or rows. ISO dates sort chronologically as strings. A row with an
    unusable date sorts last: it is only selected, and then rejected as
    malformed, when too few well-formed rows exist.
    """
    unique: dict[str, _FilingRow] = {}
    for row in rows:
        unique.setdefault(str(row.accession_number), row)

    def sort_key(row: _FilingRow) -> tuple[str, str]:
        filing_date = row.filing_date
        if not (isinstance(filing_date, str) and _ISO_DATE_RE.match(filing_date)):
            filing_date = ""
        return filing_date, str(row.accession_number)

    return sorted(unique.values(), key=sort_key, reverse=True)


def _row_to_metadata(row: _FilingRow, company: CompanyInfo) -> FilingMetadata:
    try:
        return FilingMetadata(
            ticker=company.ticker,
            cik=company.cik,
            company_name=company.company_name,
            form_type=row.form_type,
            accession_number=row.accession_number,
            filing_date=row.filing_date,
            report_date=row.report_date,
            primary_document=row.primary_document,
            filing_url=build_filing_url(company.cik, row.accession_number, row.primary_document),
            cik_source=company.cik_source,
        )
    except (ValueError, ValidationError) as exc:
        raise EdgarResponseError(
            f"{row.source}: malformed {row.form_type} row {row.index} "
            f"(accession {row.accession_number!r}): {exc}"
        ) from exc


def _older_page_names(submissions: Mapping[str, Any], company: CompanyInfo) -> list[str]:
    pages = submissions["filings"].get("files") or []
    if not isinstance(pages, list):
        raise EdgarResponseError(f"submissions for {company.ticker}: 'filings.files' is not a list")
    names: set[str] = set()
    for page in pages:
        name = page.get("name") if isinstance(page, Mapping) else None
        if not isinstance(name, str) or not _SUBMISSIONS_PAGE_RE.match(name):
            raise EdgarResponseError(
                f"submissions for {company.ticker}: unexpected page entry {page!r}"
            )
        names.add(name)
    return sorted(names)


def resolve_cik_override(
    client: EdgarClient, ticker: str, cik: str
) -> tuple[CompanyInfo, Mapping[str, Any]]:
    """Company info for a configured CIK override, from its official submissions JSON.

    The registrant name is taken from the submissions record (the ticker map
    no longer lists the overridden registrant), and the record's own CIK
    must match the configured one. Returns the submissions payload too, so
    ``collect_filings`` does not fetch it a second time.
    """
    submissions = client.get_json(submissions_url(cik))
    if not isinstance(submissions, Mapping):
        raise EdgarResponseError(f"submissions for CIK {cik}: expected a JSON object")
    name = submissions.get("name")
    reported = submissions.get("cik")
    try:
        reported_cik = normalize_cik(reported) if reported is not None else None
    except (TypeError, ValueError):
        reported_cik = None
    if reported_cik != cik:
        raise EdgarResponseError(
            f"submissions for CIK override {ticker} -> {cik} report CIK {reported!r}"
        )
    if not isinstance(name, str) or not name.strip():
        raise EdgarResponseError(f"submissions for CIK override {ticker} -> {cik} have no registrant name")
    company = CompanyInfo(ticker=ticker, cik=cik, company_name=name.strip(), cik_source="cik_override")
    return company, submissions


def collect_filings(
    client: EdgarClient,
    company: CompanyInfo,
    form_type: str,
    limit: int,
    *,
    submissions: Mapping[str, Any] | None = None,
) -> list[FilingMetadata]:
    """Fetch submissions and select the newest ``limit`` filings of ``form_type``.

    ``filings.recent`` always holds a company's newest filings, so when it
    has ``limit`` matches the older pages cannot contain anything newer and
    are not fetched. Otherwise every page in ``filings.files`` is fetched
    once, in sorted name order, and merged before selecting: the SEC does
    not promise any page order, and only the full set guarantees the newest
    matches. Selection deduplicates by accession number and sorts newest
    first (see ``_newest_first``), so fewer than ``limit`` filings are
    returned only when the company's whole history has fewer. Pass
    ``submissions`` if the company's submissions JSON was already fetched.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if submissions is None:
        submissions = client.get_json(submissions_url(company.cik))
    rows = _matching_rows(
        _recent_columns(submissions, company),
        form_type,
        source=f"submissions for {company.ticker} (filings.recent)",
    )
    if len({str(row.accession_number) for row in rows}) < limit:
        for name in _older_page_names(submissions, company):
            rows.extend(
                _matching_rows(
                    client.get_json(f"{SUBMISSIONS_BASE_URL}/{name}"),
                    form_type,
                    source=f"submissions page {name}",
                )
            )
    return [_row_to_metadata(row, company) for row in _newest_first(rows)[:limit]]


PINNED_SEARCH_LIMIT = 1000  # how far back to look for pinned filings (a company files one 10-K a year)


def select_pinned_filings(
    client: EdgarClient,
    company: CompanyInfo,
    form_type: str,
    accessions: Sequence[str],
    *,
    submissions: Mapping[str, Any] | None = None,
) -> list[FilingMetadata]:
    """Exactly the filings named in ``accessions`` (in that order), from the company's whole filing history.

    Raises :class:`EdgarResponseError` if any of them is not a ``form_type`` filing of this company, so a pinned
    corpus is either complete or not downloaded at all.
    """
    available = {f.accession_number: f for f in collect_filings(
        client, company, form_type, PINNED_SEARCH_LIMIT, submissions=submissions)}
    missing = [number for number in accessions if number not in available]
    if missing:
        raise EdgarResponseError(
            f"pinned {form_type} accession(s) not found for {company.ticker} (CIK {company.cik}): {missing}"
        )
    return [available[number] for number in accessions]


def filing_dir(output_dir: str | Path, metadata: FilingMetadata) -> Path:
    """Deterministic raw-storage folder: ``<output_dir>/<TICKER>/<FORM>/<accession>``."""
    form_dir = metadata.form_type.replace("/", "_")
    return Path(output_dir) / metadata.ticker / form_dir / metadata.accession_number


def download_filing(
    client: EdgarClient,
    metadata: FilingMetadata,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> DownloadResult:
    """Download one filing into its raw-storage folder.

    If a valid ``filing.html`` + ``metadata.json`` pair already exists, the
    filing is skipped without any request. Valid means the metadata parses,
    names the same accession number, and its ``html_sha256`` matches the HTML
    on disk; anything else (e.g. an interrupted earlier run) is re-downloaded.
    Both files are written atomically, HTML first, so ``metadata.json`` only
    ever appears next to a complete document.
    """
    target = filing_dir(output_dir, metadata)
    html_path = target / HTML_FILENAME
    metadata_path = target / METADATA_FILENAME

    if not overwrite:
        existing = _load_existing(html_path, metadata_path, metadata.accession_number)
        if existing is not None:
            return DownloadResult(metadata=existing, status="skipped")

    body = client.get(metadata.filing_url)
    if not body.strip():
        raise EdgarResponseError(f"Empty filing document from {metadata.filing_url}")

    target.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(html_path, body)
    stored = metadata.model_copy(
        update={
            "local_html_path": html_path.as_posix(),
            "html_sha256": hashlib.sha256(body).hexdigest(),
        }
    )
    _atomic_write_bytes(metadata_path, (stored.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return DownloadResult(metadata=stored, status="downloaded")


def _load_existing(
    html_path: Path, metadata_path: Path, accession_number: str
) -> FilingMetadata | None:
    if not (html_path.is_file() and metadata_path.is_file()):
        return None
    try:
        stored = FilingMetadata.model_validate_json(metadata_path.read_bytes())
    except ValidationError:
        return None
    if stored.accession_number != accession_number or stored.html_sha256 is None:
        return None
    body = html_path.read_bytes()
    if not body or hashlib.sha256(body).hexdigest() != stored.html_sha256:
        return None
    return stored


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def download_corpus(
    config: EdgarDownloadConfig,
    *,
    client: EdgarClient | None = None,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> DownloadSummary:
    """Resolve tickers, select filings and download them per ``config``.

    Failures are isolated: an unknown ticker or a bad submissions payload
    fails that ticker, a failed document download fails that filing, and
    both are recorded in ``DownloadSummary.failures``. Failing to load the
    ticker mapping aborts the run, since nothing can proceed without it.
    With ``dry_run=True`` only metadata is fetched; no filing is written.
    """
    if client is None:
        client = EdgarClient.from_config(config)
    report = progress or (lambda _message: None)
    summary = DownloadSummary(
        requested_tickers=list(config.tickers),
        output_dir=config.output_dir,
        dry_run=dry_run,
    )

    companies = load_company_tickers(client)
    for ticker in config.tickers:
        try:
            submissions: Mapping[str, Any] | None = None
            if ticker in config.cik_overrides:
                company, submissions = resolve_cik_override(client, ticker, config.cik_overrides[ticker])
                mapped = companies.get(ticker)
                summary.cik_overrides_used.append(
                    CikOverrideUse(
                        ticker=ticker,
                        cik=company.cik,
                        company_name=company.company_name,
                        ticker_map_cik=mapped.cik if mapped else None,
                        ticker_map_company_name=mapped.company_name if mapped else None,
                    )
                )
                report(
                    f"{ticker}: using configured CIK override {company.cik} ({company.company_name})"
                    + (f" instead of ticker-map CIK {mapped.cik} ({mapped.company_name})" if mapped else "")
                )
            else:
                company = resolve_ticker(companies, ticker)
            pinned = config.accessions.get(ticker)
            if pinned:
                filings = select_pinned_filings(client, company, config.form_type, pinned, submissions=submissions)
            else:
                filings = collect_filings(
                    client, company, config.form_type, config.filings_per_ticker, submissions=submissions
                )
            if not filings:
                raise EdgarResponseError(
                    f"No {config.form_type} filings found for {ticker} (CIK {company.cik})"
                )
        except EdgarError as exc:
            summary.failures.append(DownloadFailure(ticker=ticker, error=str(exc)))
            report(f"{ticker}: FAILED: {exc}")
            continue

        summary.selected.extend(filings)
        report(f"{ticker} (CIK {company.cik}): {len(filings)} x {config.form_type} selected")
        if dry_run:
            continue

        for metadata in filings:
            try:
                result = download_filing(client, metadata, config.output_dir)
            except EdgarError as exc:
                summary.failures.append(
                    DownloadFailure(
                        ticker=ticker,
                        accession_number=metadata.accession_number,
                        error=str(exc),
                    )
                )
                report(f"{ticker} {metadata.accession_number}: FAILED: {exc}")
                continue
            if result.status == "downloaded":
                summary.downloaded += 1
            else:
                summary.skipped += 1
            report(f"{ticker} {metadata.accession_number} ({metadata.filing_date}): {result.status}")
    return summary


__all__ = [
    "ARCHIVES_BASE_URL",
    "COMPANY_TICKERS_URL",
    "CORPUS_LATER_STAGE_KEYS",
    "CORPUS_PATHS_KEYS",
    "HTML_FILENAME",
    "METADATA_FILENAME",
    "SEC_USER_AGENT_ENV",
    "SUBMISSIONS_BASE_URL",
    "CikOverrideUse",
    "CompanyInfo",
    "DownloadFailure",
    "DownloadResult",
    "DownloadSummary",
    "EdgarClient",
    "EdgarDownloadConfig",
    "EdgarError",
    "EdgarRequestError",
    "EdgarResponseError",
    "FilingMetadata",
    "MissingUserAgentError",
    "UnknownTickerError",
    "accession_without_dashes",
    "build_filing_url",
    "collect_filings",
    "download_corpus",
    "download_filing",
    "filing_dir",
    "load_company_tickers",
    "normalize_cik",
    "parse_company_tickers",
    "resolve_cik_override",
    "resolve_ticker",
    "resolve_user_agent",
    "select_filings_from_columns",
    "select_pinned_filings",
    "select_recent_filings",
    "submissions_url",
]
