"""Unit tests for finretriever.data.edgar_downloader.

All HTTP goes through ``FakeSession``; an autouse fixture additionally blocks
DNS and socket connects so an accidental real request fails loudly.
"""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from finretriever.data.edgar_downloader import (
    COMPANY_TICKERS_URL,
    HTML_FILENAME,
    METADATA_FILENAME,
    CompanyInfo,
    EdgarClient,
    EdgarDownloadConfig,
    EdgarRequestError,
    EdgarResponseError,
    FilingMetadata,
    MissingUserAgentError,
    UnknownTickerError,
    accession_without_dashes,
    build_filing_url,
    collect_filings,
    download_corpus,
    download_filing,
    filing_dir,
    normalize_cik,
    parse_company_tickers,
    resolve_ticker,
    select_recent_filings,
    submissions_url,
)

UA = "FinRetriever tests tests@example.com"
AAPL = CompanyInfo(ticker="AAPL", cik="0000320193", company_name="Apple Inc.")


class NetworkBlockedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise NetworkBlockedError("real network access attempted in a unit test")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


def json_response(payload: Any) -> FakeResponse:
    return FakeResponse(200, json.dumps(payload).encode("utf-8"))


class FakeSession:
    """Serves queued responses per URL and records every request."""

    def __init__(self, routes: dict[str, list[FakeResponse | Exception]]) -> None:
        self.routes = {url: list(queue) for url, queue in routes.items()}
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        self.calls.append((url, dict(headers), timeout))
        queue = self.routes.get(url)
        if not queue:
            raise AssertionError(f"unexpected request to {url}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def urls(self) -> list[str]:
        return [url for url, _, _ in self.calls]


def make_client(
    session: FakeSession, sleeps: list[float] | None = None, **kwargs: Any
) -> EdgarClient:
    sleep_log = sleeps if sleeps is not None else []
    kwargs.setdefault("request_delay_seconds", 0.0)
    return EdgarClient(UA, session=session, sleep=sleep_log.append, **kwargs)


TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 19617, "ticker": "JPM", "title": "JPMORGAN CHASE & CO"},
    "2": {"cik_str": 1018724, "ticker": "amzn", "title": "AMAZON COM INC"},
}


def submissions_payload(rows: list[tuple[str, str, str, str, str]], files: list | None = None):
    """rows: (accessionNumber, filingDate, reportDate, form, primaryDocument), newest first."""
    return {
        "cik": "320193",
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "accessionNumber": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "reportDate": [r[2] for r in rows],
                "form": [r[3] for r in rows],
                "primaryDocument": [r[4] for r in rows],
            },
            "files": files or [],
        },
    }


AAPL_ROWS = [
    ("0000320193-24-000081", "2024-08-02", "2024-06-29", "10-Q", "aapl-20240629.htm"),
    ("0000320193-23-000106", "2023-11-03", "2023-09-30", "10-K", "aapl-20230930.htm"),
    ("0000320193-23-000077", "2023-08-04", "2023-07-01", "8-K", "aapl-20230803.htm"),
    ("0000320193-22-000108", "2022-10-28", "2022-09-24", "10-K", "aapl-20220924.htm"),
    ("0000320193-22-000050", "2022-03-01", "", "10-K/A", "aapl-10ka.htm"),
    ("0000320193-21-000105", "2021-10-29", "2021-09-25", "10-K", "aapl-20210925.htm"),
]


# --- config -------------------------------------------------------------------


def test_config_normalizes_tickers() -> None:
    config = EdgarDownloadConfig(tickers=[" aapl", "Jpm ", "AAPL"])
    assert config.tickers == ["AAPL", "JPM"]


def test_config_normalizes_form_type() -> None:
    assert EdgarDownloadConfig(tickers=["AAPL"], form_type=" 10-k ").form_type == "10-K"


def test_config_rejects_empty_ticker_list() -> None:
    with pytest.raises(ValidationError):
        EdgarDownloadConfig(tickers=[])


def test_config_rejects_blank_ticker() -> None:
    with pytest.raises(ValidationError, match="blank"):
        EdgarDownloadConfig(tickers=["AAPL", "  "])


@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_non_positive_filings_per_ticker(value: int) -> None:
    with pytest.raises(ValidationError):
        EdgarDownloadConfig(tickers=["AAPL"], filings_per_ticker=value)


@pytest.mark.parametrize(
    "field,value",
    [("request_delay_seconds", -0.1), ("timeout_seconds", 0), ("max_retries", -1)],
)
def test_config_rejects_invalid_network_settings(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        EdgarDownloadConfig(tickers=["AAPL"], **{field: value})


def test_config_defaults_and_ignores_later_stage_fields() -> None:
    config = EdgarDownloadConfig.model_validate(
        {"tickers": ["AAPL"], "chunk_size_tokens": 512, "paths": {"processed_dir": "x"}}
    )
    assert config.form_type == "10-K"
    assert config.filings_per_ticker == 7
    assert config.output_dir == Path("data/raw")
    assert config.request_delay_seconds == 0.2
    assert config.timeout_seconds == 30.0
    assert config.max_retries == 3


def test_config_rejects_misspelled_filings_per_ticker_instead_of_defaulting() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EdgarDownloadConfig.model_validate({"tickers": ["AAPL"], "filing_per_ticker": 1})
    assert (
        "unknown config key 'filing_per_ticker' (did you mean 'filings_per_ticker'?)"
        in str(excinfo.value)
    )


def test_config_accepts_every_allowlisted_later_stage_key() -> None:
    config = EdgarDownloadConfig.model_validate(
        {
            "tickers": ["AAPL"],
            "name": "corpus_v1",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 80,
            "paths": {"interim_dir": "a", "processed_dir": "b", "corpus_jsonl": "c.jsonl"},
        }
    )
    assert config.tickers == ["AAPL"]
    assert not hasattr(config, "chunk_size_tokens")


def test_config_unknown_key_without_close_match_lists_allowed_keys() -> None:
    with pytest.raises(ValidationError, match="unknown config key 'zzz' \\(allowed: accessions, chunk_overlap"):
        EdgarDownloadConfig.model_validate({"tickers": ["AAPL"], "zzz": 1})


def test_config_rejects_misspelled_paths_key() -> None:
    with pytest.raises(
        ValidationError,
        match="unknown config key 'paths.procesed_dir' \\(did you mean 'paths.processed_dir'\\?\\)",
    ):
        EdgarDownloadConfig.model_validate({"tickers": ["AAPL"], "paths": {"procesed_dir": "x"}})


def test_config_rejects_non_mapping_paths() -> None:
    with pytest.raises(ValidationError, match="'paths' must be a mapping"):
        EdgarDownloadConfig.model_validate({"tickers": ["AAPL"], "paths": "data/processed"})


def test_config_reports_every_unknown_key_at_once() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EdgarDownloadConfig.model_validate(
            {"tickers": ["AAPL"], "form": "10-K", "chunk_size": 512, "paths": {"raw_dir": "x"}}
        )
    message = str(excinfo.value)
    assert "'form' (did you mean 'form_type'?)" in message
    assert "'chunk_size' (did you mean 'chunk_size_tokens'?)" in message
    assert "'paths.raw_dir'" in message


# --- identifiers and URLs -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [(320193, "0000320193"), ("320193", "0000320193"), ("0000320193", "0000320193"),
     (" 19617 ", "0000019617"), (1234567890, "1234567890")],
)
def test_normalize_cik_pads_to_ten_digits(raw: str | int, expected: str) -> None:
    assert normalize_cik(raw) == expected


@pytest.mark.parametrize("raw", ["", "abc", "12345678901", "0", "-5", "CIK320193"])
def test_normalize_cik_rejects_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_cik(raw)


def test_normalize_cik_rejects_bool() -> None:
    with pytest.raises(TypeError):
        normalize_cik(True)


def test_accession_without_dashes() -> None:
    assert accession_without_dashes("0000320193-23-000106") == "000032019323000106"


@pytest.mark.parametrize("raw", ["000032019323000106", "0000320193-23-00010", "", "a-b-c"])
def test_accession_without_dashes_rejects_malformed(raw: str) -> None:
    with pytest.raises(ValueError, match="Invalid accession number"):
        accession_without_dashes(raw)


def test_build_filing_url_matches_sec_archive_layout() -> None:
    url = build_filing_url("0000320193", "0000320193-23-000106", "aapl-20230930.htm")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    )


def test_build_filing_url_strips_cik_zeros_for_int_and_str() -> None:
    from_int = build_filing_url(19617, "0000019617-24-000225", "jpm-20231231.htm")
    from_str = build_filing_url("0000019617", "0000019617-24-000225", "jpm-20231231.htm")
    assert from_int == from_str
    assert "/data/19617/000001961724000225/" in from_int


def test_build_filing_url_percent_encodes_document() -> None:
    url = build_filing_url(320193, "0000320193-23-000106", "form 10k.htm")
    assert url.endswith("/000032019323000106/form%2010k.htm")


@pytest.mark.parametrize(
    "document", ["", "  ", "../escape.htm", "/abs.htm", "a//b.htm", "a\\b.htm", " x.htm"]
)
def test_build_filing_url_rejects_unsafe_document(document: str) -> None:
    with pytest.raises(ValueError):
        build_filing_url(320193, "0000320193-23-000106", document)


def test_build_filing_url_rejects_bad_accession() -> None:
    with pytest.raises(ValueError):
        build_filing_url(320193, "000032019323000106", "aapl.htm")


def test_submissions_url_uses_padded_cik() -> None:
    assert submissions_url(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"


# --- ticker mapping -----------------------------------------------------------


def test_parse_company_tickers_resolves_ticker_cik_and_name() -> None:
    companies = parse_company_tickers(TICKERS_PAYLOAD)
    assert companies["AAPL"] == AAPL
    assert companies["JPM"].cik == "0000019617"
    assert companies["AMZN"].company_name == "AMAZON COM INC"


def test_resolve_ticker_is_case_insensitive() -> None:
    assert resolve_ticker(parse_company_tickers(TICKERS_PAYLOAD), " aapl ") == AAPL


def test_unknown_ticker_produces_clear_error() -> None:
    with pytest.raises(UnknownTickerError, match="'ZZZZ' is not in the SEC company_tickers"):
        resolve_ticker(parse_company_tickers(TICKERS_PAYLOAD), "zzzz")


@pytest.mark.parametrize(
    "payload,match",
    [
        ({}, "mapping is empty"),
        ([], "expected a JSON object"),
        ({"0": {"ticker": "AAPL", "title": "Apple"}}, "malformed row '0'"),
        ({"0": {"cik_str": "x", "ticker": "AAPL", "title": "Apple"}}, "malformed row '0'"),
        ({"0": "AAPL"}, "malformed row '0'"),
    ],
)
def test_parse_company_tickers_rejects_malformed(payload: Any, match: str) -> None:
    with pytest.raises(EdgarResponseError, match=match):
        parse_company_tickers(payload)


# --- filing selection ---------------------------------------------------------


def test_select_recent_filings_filters_to_exact_form_type() -> None:
    filings = select_recent_filings(submissions_payload(AAPL_ROWS), AAPL, "10-K", 10)
    assert [f.form_type for f in filings] == ["10-K", "10-K", "10-K"]
    assert "0000320193-22-000050" not in {f.accession_number for f in filings}  # 10-K/A


def test_select_recent_filings_respects_limit() -> None:
    filings = select_recent_filings(submissions_payload(AAPL_ROWS), AAPL, "10-K", 2)
    assert len(filings) == 2


def test_select_recent_filings_preserves_newest_first_order() -> None:
    filings = select_recent_filings(submissions_payload(AAPL_ROWS), AAPL, "10-K", 10)
    assert [f.accession_number for f in filings] == [
        "0000320193-23-000106",
        "0000320193-22-000108",
        "0000320193-21-000105",
    ]


def test_select_recent_filings_extracts_fields() -> None:
    first = select_recent_filings(submissions_payload(AAPL_ROWS), AAPL, "10-k", 1)[0]
    assert first.ticker == "AAPL"
    assert first.cik == "0000320193"
    assert first.company_name == "Apple Inc."
    assert first.form_type == "10-K"
    assert str(first.filing_date) == "2023-11-03"
    assert str(first.report_date) == "2023-09-30"
    assert first.primary_document == "aapl-20230930.htm"
    assert first.filing_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    )
    assert first.local_html_path is None


def test_select_recent_filings_empty_report_date_becomes_none() -> None:
    rows = [("0000320193-23-000106", "2023-11-03", "", "10-K", "aapl.htm")]
    assert select_recent_filings(submissions_payload(rows), AAPL)[0].report_date is None


def test_select_recent_filings_without_matches_returns_empty() -> None:
    rows = [("0000320193-23-000077", "2023-08-04", "", "8-K", "x.htm")]
    assert select_recent_filings(submissions_payload(rows), AAPL) == []


@pytest.mark.parametrize(
    "payload,match",
    [
        ([], "expected a JSON object"),
        ({}, "missing 'filings.recent'"),
        ({"filings": {}}, "missing 'filings.recent'"),
        ({"filings": {"recent": []}}, "expected an object of filing columns"),
        ({"filings": {"recent": {"form": ["10-K"]}}}, "missing filing columns"),
        (
            {"filings": {"recent": {"accessionNumber": "x", "filingDate": [], "form": [],
                                    "primaryDocument": []}}},
            "column 'accessionNumber' is not a list",
        ),
        (
            {"filings": {"recent": {"accessionNumber": ["0000320193-23-000106"],
                                    "filingDate": [], "form": ["10-K"],
                                    "primaryDocument": ["a.htm"]}}},
            "mismatched lengths",
        ),
    ],
)
def test_malformed_submissions_payload_raises_clear_error(payload: Any, match: str) -> None:
    with pytest.raises(EdgarResponseError, match=match):
        select_recent_filings(payload, AAPL)


@pytest.mark.parametrize(
    "row",
    [
        ("bad-accession", "2023-11-03", "", "10-K", "a.htm"),
        ("0000320193-23-000106", "", "", "10-K", "a.htm"),
        ("0000320193-23-000106", "2023-11-03", "", "10-K", ""),
        ("0000320193-23-000106", "2023-11-03", "", "10-K", "../x.htm"),
    ],
)
def test_malformed_matching_row_raises_clear_error(row: tuple[str, ...]) -> None:
    with pytest.raises(EdgarResponseError, match="malformed 10-K row 0"):
        select_recent_filings(submissions_payload([row]), AAPL)


def test_collect_filings_follows_older_pages_when_recent_is_short() -> None:
    page_name = "CIK0000320193-submissions-001.json"
    recent = submissions_payload(AAPL_ROWS[:2], files=[{"name": page_name}])
    page = submissions_payload(AAPL_ROWS[2:])["filings"]["recent"]
    session = FakeSession(
        {
            submissions_url(AAPL.cik): [json_response(recent)],
            f"https://data.sec.gov/submissions/{page_name}": [json_response(page)],
        }
    )
    filings = collect_filings(make_client(session), AAPL, "10-K", 3)
    assert [f.accession_number for f in filings] == [
        "0000320193-23-000106",
        "0000320193-22-000108",
        "0000320193-21-000105",
    ]


def test_collect_filings_skips_older_pages_when_recent_suffices() -> None:
    recent = submissions_payload(AAPL_ROWS, files=[{"name": "CIK0000320193-submissions-001.json"}])
    session = FakeSession({submissions_url(AAPL.cik): [json_response(recent)]})
    assert len(collect_filings(make_client(session), AAPL, "10-K", 2)) == 2
    assert session.urls() == [submissions_url(AAPL.cik)]


def test_collect_filings_rejects_unexpected_page_name() -> None:
    recent = submissions_payload(AAPL_ROWS[:2], files=[{"name": "../../evil.json"}])
    session = FakeSession({submissions_url(AAPL.cik): [json_response(recent)]})
    with pytest.raises(EdgarResponseError, match="unexpected page entry"):
        collect_filings(make_client(session), AAPL, "10-K", 5)


# --- ordering and deduplication across blocks/pages ----------------------------

PAGE_1 = "CIK0000320193-submissions-001.json"
PAGE_2 = "CIK0000320193-submissions-002.json"


def ten_k(year: int, seq: int = 1) -> tuple[str, str, str, str, str]:
    return (
        f"0000320193-{year % 100:02d}-{seq:06d}",
        f"{year}-11-01",
        f"{year}-09-30",
        "10-K",
        f"aapl-{year}0930.htm",
    )


def page_url(name: str) -> str:
    return f"https://data.sec.gov/submissions/{name}"


def page_columns(rows: list[tuple[str, str, str, str, str]]) -> dict[str, Any]:
    return submissions_payload(rows)["filings"]["recent"]


def years(filings: list[FilingMetadata]) -> list[int]:
    return [f.filing_date.year for f in filings]


def test_select_recent_filings_keeps_newest_when_rows_are_out_of_order() -> None:
    rows = [ten_k(2019), ten_k(2023), ten_k(2021), ten_k(2022)]
    assert years(select_recent_filings(submissions_payload(rows), AAPL, "10-K", 2)) == [2023, 2022]


def test_select_recent_filings_dedupes_accession_numbers() -> None:
    rows = [ten_k(2023), ten_k(2023), ten_k(2022), ten_k(2021)]
    filings = select_recent_filings(submissions_payload(rows), AAPL, "10-K", 3)
    assert years(filings) == [2023, 2022, 2021]
    assert len({f.accession_number for f in filings}) == 3


@pytest.mark.parametrize("listing", [[PAGE_1, PAGE_2], [PAGE_2, PAGE_1]])
def test_collect_filings_selects_newest_across_out_of_order_pages(listing: list[str]) -> None:
    # PAGE_2 holds the oldest filings and its rows are themselves reversed.
    recent = submissions_payload([ten_k(2023)], files=[{"name": n} for n in listing])
    session = FakeSession(
        {
            submissions_url(AAPL.cik): [json_response(recent)],
            page_url(PAGE_1): [json_response(page_columns([ten_k(2021), ten_k(2022)]))],
            page_url(PAGE_2): [json_response(page_columns([ten_k(2018), ten_k(2020)]))],
        }
    )
    filings = collect_filings(make_client(session), AAPL, "10-K", 4)
    assert years(filings) == [2023, 2022, 2021, 2020]
    assert session.urls()[1:] == [page_url(PAGE_1), page_url(PAGE_2)]


def test_collect_filings_dedupes_overlap_and_preserves_limit() -> None:
    recent = submissions_payload([ten_k(2023), ten_k(2022)], files=[{"name": PAGE_1}])
    page = page_columns([ten_k(2022), ten_k(2021), ten_k(2020), ten_k(2019)])
    session = FakeSession(
        {
            submissions_url(AAPL.cik): [json_response(recent)],
            page_url(PAGE_1): [json_response(page)],
        }
    )
    filings = collect_filings(make_client(session), AAPL, "10-K", 4)
    assert years(filings) == [2023, 2022, 2021, 2020]
    assert len({f.accession_number for f in filings}) == 4


def test_collect_filings_fetches_each_listed_page_once() -> None:
    recent = submissions_payload([ten_k(2023)], files=[{"name": PAGE_1}, {"name": PAGE_1}])
    session = FakeSession(
        {
            submissions_url(AAPL.cik): [json_response(recent)],
            page_url(PAGE_1): [json_response(page_columns([ten_k(2022)]))],
        }
    )
    assert years(collect_filings(make_client(session), AAPL, "10-K", 5)) == [2023, 2022]
    assert session.urls().count(page_url(PAGE_1)) == 1


def test_collect_filings_returns_whole_history_when_shorter_than_limit() -> None:
    recent = submissions_payload([ten_k(2023)], files=[{"name": PAGE_1}])
    session = FakeSession(
        {
            submissions_url(AAPL.cik): [json_response(recent)],
            page_url(PAGE_1): [json_response(page_columns([ten_k(2022)]))],
        }
    )
    assert years(collect_filings(make_client(session), AAPL, "10-K", 7)) == [2023, 2022]


def test_collect_filings_ignores_malformed_rows_older_than_the_selection() -> None:
    legacy = ("0000320193-96-000001", "1996-12-19", "", "10-K", "")  # no primary document
    recent = submissions_payload([ten_k(2023)], files=[{"name": PAGE_1}])
    session = FakeSession(
        {
            submissions_url(AAPL.cik): [json_response(recent)],
            page_url(PAGE_1): [json_response(page_columns([legacy, ten_k(2022)]))],
        }
    )
    assert years(collect_filings(make_client(session), AAPL, "10-K", 2)) == [2023, 2022]


def test_collect_filings_rejects_malformed_row_inside_the_selection() -> None:
    broken = ("0000320193-22-000001", "2022-11-01", "", "10-K", "")
    recent = submissions_payload([ten_k(2023)], files=[{"name": PAGE_1}])
    session = FakeSession(
        {
            submissions_url(AAPL.cik): [json_response(recent)],
            page_url(PAGE_1): [json_response(page_columns([broken, ten_k(2021)]))],
        }
    )
    with pytest.raises(EdgarResponseError, match=f"submissions page {PAGE_1}: malformed 10-K row 0"):
        collect_filings(make_client(session), AAPL, "10-K", 2)


def test_rows_with_unusable_dates_sort_after_well_formed_rows() -> None:
    undated = ("0000320193-24-000009", "not-a-date", "", "10-K", "x.htm")
    rows = [undated, ten_k(2023), ten_k(2022)]
    assert years(select_recent_filings(submissions_payload(rows), AAPL, "10-K", 2)) == [2023, 2022]
    with pytest.raises(EdgarResponseError, match="malformed 10-K row 0"):
        select_recent_filings(submissions_payload(rows), AAPL, "10-K", 3)


# --- HTTP client --------------------------------------------------------------


def test_sec_headers_include_configured_user_agent() -> None:
    session = FakeSession({COMPANY_TICKERS_URL: [json_response(TICKERS_PAYLOAD)]})
    client = make_client(session, timeout_seconds=12.5)
    client.get_json(COMPANY_TICKERS_URL)
    _, headers, timeout = session.calls[0]
    assert headers["User-Agent"] == UA
    assert "gzip" in headers["Accept-Encoding"]
    assert timeout == 12.5


def test_user_agent_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "  Env Person env@example.com ")
    client = EdgarClient(session=FakeSession({}))
    assert client.headers["User-Agent"] == "Env Person env@example.com"


@pytest.mark.parametrize("env_value", [None, "", "   "])
def test_missing_user_agent_is_rejected(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    if env_value is None:
        monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    else:
        monkeypatch.setenv("SEC_USER_AGENT", env_value)
    with pytest.raises(MissingUserAgentError, match="SEC_USER_AGENT is not set"):
        EdgarClient(session=FakeSession({}))


def test_user_agent_without_contact_email_is_rejected() -> None:
    with pytest.raises(MissingUserAgentError, match="contact email"):
        EdgarClient("FinRetriever", session=FakeSession({}))


def test_download_corpus_without_user_agent_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(MissingUserAgentError):
        download_corpus(EdgarDownloadConfig(tickers=["AAPL"]))


def test_http_429_retries_with_bounded_backoff_and_retry_after() -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession(
        {
            url: [
                FakeResponse(429, headers={"Retry-After": "2"}),
                FakeResponse(429, headers={"Retry-After": "999"}),
                FakeResponse(429),
                FakeResponse(200, b"ok"),
            ]
        }
    )
    sleeps: list[float] = []
    assert make_client(session, sleeps, max_retries=3).get(url) == b"ok"
    assert len(session.calls) == 4
    assert sleeps == [2.0, 60.0, 4.0]  # Retry-After honoured, capped; then backoff


def test_http_429_gives_up_after_max_retries() -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession({url: [FakeResponse(429) for _ in range(10)]})
    sleeps: list[float] = []
    with pytest.raises(EdgarRequestError, match="after 3 attempt"):
        make_client(session, sleeps, max_retries=2).get(url)
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_transient_5xx_is_retried(status: int) -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession({url: [FakeResponse(status), FakeResponse(200, b"ok")]})
    sleeps: list[float] = []
    assert make_client(session, sleeps).get(url) == b"ok"
    assert len(session.calls) == 2
    assert sleeps == [1.0]


def test_connection_errors_and_timeouts_are_retried() -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession(
        {
            url: [
                requests.ConnectionError("reset"),
                requests.Timeout("slow"),
                FakeResponse(200, b"ok"),
            ]
        }
    )
    assert make_client(session).get(url) == b"ok"
    assert len(session.calls) == 3


@pytest.mark.parametrize("status", [400, 403, 404])
def test_non_retryable_4xx_raises_immediately(status: int) -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession({url: [FakeResponse(status), FakeResponse(200, b"ok")]})
    sleeps: list[float] = []
    with pytest.raises(EdgarRequestError, match=f"HTTP {status}") as excinfo:
        make_client(session, sleeps).get(url)
    assert len(session.calls) == 1
    assert sleeps == []
    assert excinfo.value.status_code == status
    assert excinfo.value.attempts == 1


def test_403_error_mentions_user_agent() -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession({url: [FakeResponse(403)]})
    with pytest.raises(EdgarRequestError, match="User-Agent"):
        make_client(session).get(url)


def test_retry_exhaustion_raises_clear_error() -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession({url: [FakeResponse(503) for _ in range(4)]})
    with pytest.raises(EdgarRequestError) as excinfo:
        make_client(session, max_retries=3).get(url)
    message = str(excinfo.value)
    assert url in message
    assert "after 4 attempt(s)" in message
    assert "last error: HTTP 503" in message
    assert excinfo.value.attempts == 4
    assert excinfo.value.status_code == 503


def test_zero_retries_makes_a_single_attempt() -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession({url: [FakeResponse(503)]})
    with pytest.raises(EdgarRequestError, match="after 1 attempt"):
        make_client(session, max_retries=0).get(url)


def test_requests_are_throttled_by_request_delay() -> None:
    url = "https://www.sec.gov/x"
    session = FakeSession({url: [FakeResponse(200, b"a"), FakeResponse(200, b"b")]})
    now = [100.0]
    sleeps: list[float] = []
    client = EdgarClient(
        UA,
        request_delay_seconds=0.5,
        session=session,
        sleep=sleeps.append,
        clock=lambda: now[0],
    )
    client.get(url)
    now[0] += 0.1
    client.get(url)
    assert sleeps == [pytest.approx(0.4)]


def test_empty_sec_response_does_not_silently_succeed() -> None:
    session = FakeSession({COMPANY_TICKERS_URL: [FakeResponse(200, b"  ")]})
    with pytest.raises(EdgarResponseError, match="Empty response body"):
        make_client(session).get_json(COMPANY_TICKERS_URL)


def test_invalid_json_response_raises() -> None:
    session = FakeSession({COMPANY_TICKERS_URL: [FakeResponse(200, b"<html>")]})
    with pytest.raises(EdgarResponseError, match="Invalid JSON"):
        make_client(session).get_json(COMPANY_TICKERS_URL)


# --- download_filing ----------------------------------------------------------

HTML = b"<html><body>Apple 10-K</body></html>"


def aapl_filing() -> FilingMetadata:
    return select_recent_filings(submissions_payload(AAPL_ROWS), AAPL, "10-K", 1)[0]


def test_downloader_creates_deterministic_directory_structure(tmp_path: Path) -> None:
    metadata = aapl_filing()
    session = FakeSession({metadata.filing_url: [FakeResponse(200, HTML)]})
    result = download_filing(make_client(session), metadata, tmp_path)
    expected = tmp_path / "AAPL" / "10-K" / "0000320193-23-000106"
    assert filing_dir(tmp_path, metadata) == expected
    assert result.status == "downloaded"
    assert sorted(p.name for p in expected.iterdir()) == [HTML_FILENAME, METADATA_FILENAME]


def test_downloader_writes_filing_html(tmp_path: Path) -> None:
    metadata = aapl_filing()
    session = FakeSession({metadata.filing_url: [FakeResponse(200, HTML)]})
    download_filing(make_client(session), metadata, tmp_path)
    assert (filing_dir(tmp_path, metadata) / HTML_FILENAME).read_bytes() == HTML


def test_downloader_writes_metadata_json(tmp_path: Path) -> None:
    metadata = aapl_filing()
    session = FakeSession({metadata.filing_url: [FakeResponse(200, HTML)]})
    result = download_filing(make_client(session), metadata, tmp_path)
    target = filing_dir(tmp_path, metadata)
    stored = json.loads((target / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert stored["ticker"] == "AAPL"
    assert stored["cik"] == "0000320193"
    assert stored["accession_number"] == "0000320193-23-000106"
    assert stored["filing_date"] == "2023-11-03"
    assert stored["report_date"] == "2023-09-30"
    assert stored["primary_document"] == "aapl-20230930.htm"
    assert stored["filing_url"] == metadata.filing_url
    assert stored["local_html_path"] == (target / HTML_FILENAME).as_posix()
    assert len(stored["html_sha256"]) == 64
    assert FilingMetadata.model_validate(stored) == result.metadata
    assert not list(target.glob("*.tmp"))


def test_rerun_skips_already_downloaded_valid_filing(tmp_path: Path) -> None:
    metadata = aapl_filing()
    session = FakeSession({metadata.filing_url: [FakeResponse(200, HTML)]})
    client = make_client(session)
    first = download_filing(client, metadata, tmp_path)
    second = download_filing(client, metadata, tmp_path)
    assert second.status == "skipped"
    assert second.metadata == first.metadata
    assert len(session.calls) == 1


@pytest.mark.parametrize("damage", ["truncate_html", "delete_metadata", "corrupt_metadata"])
def test_invalid_existing_files_are_redownloaded(tmp_path: Path, damage: str) -> None:
    metadata = aapl_filing()
    session = FakeSession(
        {metadata.filing_url: [FakeResponse(200, HTML), FakeResponse(200, HTML)]}
    )
    client = make_client(session)
    download_filing(client, metadata, tmp_path)
    target = filing_dir(tmp_path, metadata)
    if damage == "truncate_html":
        (target / HTML_FILENAME).write_bytes(HTML[:10])
    elif damage == "delete_metadata":
        (target / METADATA_FILENAME).unlink()
    else:
        (target / METADATA_FILENAME).write_text("{not json", encoding="utf-8")
    assert download_filing(client, metadata, tmp_path).status == "downloaded"
    assert (target / HTML_FILENAME).read_bytes() == HTML
    assert len(session.calls) == 2


def test_overwrite_forces_redownload(tmp_path: Path) -> None:
    metadata = aapl_filing()
    session = FakeSession(
        {metadata.filing_url: [FakeResponse(200, HTML), FakeResponse(200, HTML)]}
    )
    client = make_client(session)
    download_filing(client, metadata, tmp_path)
    assert download_filing(client, metadata, tmp_path, overwrite=True).status == "downloaded"


def test_empty_filing_document_raises_and_writes_nothing(tmp_path: Path) -> None:
    metadata = aapl_filing()
    session = FakeSession({metadata.filing_url: [FakeResponse(200, b"")]})
    with pytest.raises(EdgarResponseError, match="Empty filing document"):
        download_filing(make_client(session), metadata, tmp_path)
    assert not filing_dir(tmp_path, metadata).exists()


# --- download_corpus ----------------------------------------------------------


def corpus_session(extra: dict[str, list] | None = None) -> FakeSession:
    routes: dict[str, list] = {
        COMPANY_TICKERS_URL: [json_response(TICKERS_PAYLOAD)],
        submissions_url(AAPL.cik): [json_response(submissions_payload(AAPL_ROWS))],
    }
    for filing in select_recent_filings(submissions_payload(AAPL_ROWS), AAPL, "10-K", 10):
        routes[filing.filing_url] = [FakeResponse(200, HTML)]
    routes.update(extra or {})
    return FakeSession(routes)


def test_download_corpus_summary_counts(tmp_path: Path) -> None:
    config = EdgarDownloadConfig(tickers=["AAPL"], filings_per_ticker=2, output_dir=tmp_path)
    summary = download_corpus(config, client=make_client(corpus_session()))
    assert summary.requested_tickers == ["AAPL"]
    assert summary.filings_selected == 2
    assert summary.downloaded == 2
    assert summary.skipped == 0
    assert summary.failures == []


def test_download_corpus_records_unknown_ticker_and_continues(tmp_path: Path) -> None:
    config = EdgarDownloadConfig(tickers=["ZZZZ", "AAPL"], filings_per_ticker=1, output_dir=tmp_path)
    summary = download_corpus(config, client=make_client(corpus_session()))
    assert summary.downloaded == 1
    assert len(summary.failures) == 1
    assert summary.failures[0].ticker == "ZZZZ"
    assert "not in the SEC company_tickers" in summary.failures[0].error


def test_download_corpus_records_ticker_without_matching_filings(tmp_path: Path) -> None:
    jpm = parse_company_tickers(TICKERS_PAYLOAD)["JPM"]
    rows = [("0000019617-24-000001", "2024-01-02", "", "424B2", "x.htm")]
    session = corpus_session({submissions_url(jpm.cik): [json_response(submissions_payload(rows))]})
    config = EdgarDownloadConfig(tickers=["JPM"], output_dir=tmp_path)
    summary = download_corpus(config, client=make_client(session))
    assert summary.filings_selected == 0
    assert "No 10-K filings found for JPM" in summary.failures[0].error


def test_download_corpus_records_per_filing_failure(tmp_path: Path) -> None:
    newest = aapl_filing()
    session = corpus_session({newest.filing_url: [FakeResponse(404)]})
    config = EdgarDownloadConfig(tickers=["AAPL"], filings_per_ticker=2, output_dir=tmp_path)
    summary = download_corpus(config, client=make_client(session))
    assert summary.downloaded == 1
    assert len(summary.failures) == 1
    failure = summary.failures[0]
    assert (failure.ticker, failure.accession_number) == ("AAPL", newest.accession_number)
    assert "HTTP 404" in failure.error


def test_download_corpus_dry_run_fetches_metadata_only(tmp_path: Path) -> None:
    session = corpus_session()
    config = EdgarDownloadConfig(tickers=["AAPL"], filings_per_ticker=2, output_dir=tmp_path)
    summary = download_corpus(config, client=make_client(session), dry_run=True)
    assert summary.dry_run is True
    assert summary.filings_selected == 2
    assert summary.downloaded == summary.skipped == 0
    assert not any("/Archives/" in url for url in session.urls())
    assert list(tmp_path.iterdir()) == []


# --- network guard ------------------------------------------------------------


def test_unit_tests_cannot_reach_the_network() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("www.sec.gov", 443), timeout=1)
    with pytest.raises(NetworkBlockedError):
        EdgarClient(UA, max_retries=0, request_delay_seconds=0).get(COMPANY_TICKERS_URL)


# --- CIK overrides --------------------------------------------------------------

XOM_MAP = {"0": {"cik_str": 2115436, "ticker": "XOM", "title": "ExxonMobil Holdings Corp"}}
XOM_HISTORICAL = "0000034088"
XOM_ROWS = [
    ("0000034088-26-000045", "2026-02-18", "2025-12-31", "10-K", "xom-20251231.htm"),
    ("0000034088-25-000010", "2025-02-19", "2024-12-31", "10-K", "xom-20241231.htm"),
]


def registrant(cik: str, name: str, rows: list) -> dict[str, Any]:
    payload = submissions_payload(rows)
    payload["cik"], payload["name"] = cik, name
    return payload


def override_config(tmp_path: Path, **kwargs: Any) -> EdgarDownloadConfig:
    settings: dict[str, Any] = {
        "tickers": ["XOM"],
        "filings_per_ticker": 2,
        "output_dir": tmp_path,
        "cik_overrides": {"XOM": XOM_HISTORICAL},
    }
    settings.update(kwargs)
    return EdgarDownloadConfig(**settings)


def test_config_normalizes_cik_overrides() -> None:
    config = EdgarDownloadConfig(tickers=["xom"], cik_overrides={" xom ": "34088"})
    assert config.cik_overrides == {"XOM": "0000034088"}
    assert EdgarDownloadConfig(tickers=["AAPL"]).cik_overrides == {}


@pytest.mark.parametrize("cik", ["abc", "00000340880", "0000000000", "", "34-088"])
def test_config_rejects_invalid_override_cik(cik: str) -> None:
    with pytest.raises(ValidationError, match="cik_overrides.XOM: Invalid CIK"):
        EdgarDownloadConfig(tickers=["XOM"], cik_overrides={"XOM": cik})


def test_config_requires_quoted_override_cik() -> None:
    with pytest.raises(ValidationError, match="must be a quoted string"):
        EdgarDownloadConfig(tickers=["XOM"], cik_overrides={"XOM": 34088})


def test_config_rejects_override_for_unconfigured_ticker() -> None:
    with pytest.raises(ValidationError, match="cik_overrides for tickers not in tickers: XON"):
        EdgarDownloadConfig(tickers=["XOM"], cik_overrides={"XON": "0000034088"})


def test_config_rejects_misspelled_override_key() -> None:
    with pytest.raises(ValidationError, match="did you mean 'cik_overrides'"):
        EdgarDownloadConfig.model_validate({"tickers": ["XOM"], "cik_override": {"XOM": "0000034088"}})


def test_override_resolves_historical_registrant(tmp_path: Path) -> None:
    session = FakeSession(
        {
            COMPANY_TICKERS_URL: [json_response(XOM_MAP)],
            submissions_url(XOM_HISTORICAL): [
                json_response(registrant("34088", "EXXON MOBIL CORP", XOM_ROWS))
            ],
        }
    )
    messages: list[str] = []
    summary = download_corpus(
        override_config(tmp_path), client=make_client(session), dry_run=True, progress=messages.append
    )
    assert summary.failures == []
    assert [(f.ticker, f.cik, f.company_name, f.cik_source) for f in summary.selected] == [
        ("XOM", XOM_HISTORICAL, "EXXON MOBIL CORP", "cik_override")
    ] * 2
    assert [f.accession_number for f in summary.selected] == [r[0] for r in XOM_ROWS]
    assert summary.selected[0].filing_url.startswith("https://www.sec.gov/Archives/edgar/data/34088/")
    assert [u.model_dump() for u in summary.cik_overrides_used] == [
        {
            "ticker": "XOM",
            "cik": XOM_HISTORICAL,
            "company_name": "EXXON MOBIL CORP",
            "ticker_map_cik": "0002115436",
            "ticker_map_company_name": "ExxonMobil Holdings Corp",
        }
    ]
    assert (
        "XOM: using configured CIK override 0000034088 (EXXON MOBIL CORP) instead of "
        "ticker-map CIK 0002115436 (ExxonMobil Holdings Corp)"
    ) in messages
    # The submissions record is fetched once, and the ticker-map registrant never.
    assert session.urls() == [COMPANY_TICKERS_URL, submissions_url(XOM_HISTORICAL)]


def test_no_override_means_no_substitution(tmp_path: Path) -> None:
    session = FakeSession(
        {
            COMPANY_TICKERS_URL: [json_response(XOM_MAP)],
            submissions_url("0002115436"): [
                json_response(registrant("2115436", "ExxonMobil Holdings Corp", AAPL_ROWS[:1]))
            ],
        }
    )
    summary = download_corpus(
        override_config(tmp_path, cik_overrides={}), client=make_client(session), dry_run=True
    )
    assert summary.selected == []
    assert summary.cik_overrides_used == []
    assert summary.failures[0].error == "No 10-K filings found for XOM (CIK 0002115436)"
    assert submissions_url(XOM_HISTORICAL) not in session.urls()


def test_unknown_ticker_without_override_still_fails(tmp_path: Path) -> None:
    session = FakeSession({COMPANY_TICKERS_URL: [json_response(XOM_MAP)]})
    config = override_config(tmp_path, tickers=["XOM", "ZZZZ"])
    session.routes[submissions_url(XOM_HISTORICAL)] = [
        json_response(registrant("34088", "EXXON MOBIL CORP", XOM_ROWS))
    ]
    summary = download_corpus(config, client=make_client(session), dry_run=True)
    assert [f.ticker for f in summary.failures] == ["ZZZZ"]
    assert "not in the SEC company_tickers" in summary.failures[0].error


def test_override_works_when_ticker_map_lacks_the_ticker(tmp_path: Path) -> None:
    session = FakeSession(
        {
            COMPANY_TICKERS_URL: [json_response(TICKERS_PAYLOAD)],
            submissions_url(XOM_HISTORICAL): [
                json_response(registrant("0000034088", "EXXON MOBIL CORP", XOM_ROWS))
            ],
        }
    )
    summary = download_corpus(override_config(tmp_path), client=make_client(session), dry_run=True)
    assert len(summary.selected) == 2
    assert summary.cik_overrides_used[0].ticker_map_cik is None


@pytest.mark.parametrize(
    "payload,match",
    [
        (registrant("2115436", "EXXON MOBIL CORP", XOM_ROWS), "report CIK '2115436'"),
        (registrant("34088", "  ", XOM_ROWS), "no registrant name"),
    ],
)
def test_override_submissions_must_match_configured_registrant(
    tmp_path: Path, payload: dict[str, Any], match: str
) -> None:
    session = FakeSession(
        {COMPANY_TICKERS_URL: [json_response(XOM_MAP)], submissions_url(XOM_HISTORICAL): [json_response(payload)]}
    )
    summary = download_corpus(override_config(tmp_path), client=make_client(session), dry_run=True)
    assert summary.selected == []
    assert re.search(match, summary.failures[0].error)


def test_override_provenance_is_written_to_metadata_and_rerun_is_idempotent(tmp_path: Path) -> None:
    routes = {
        COMPANY_TICKERS_URL: [json_response(XOM_MAP), json_response(XOM_MAP)],
        submissions_url(XOM_HISTORICAL): [
            json_response(registrant("34088", "EXXON MOBIL CORP", XOM_ROWS)),
            json_response(registrant("34088", "EXXON MOBIL CORP", XOM_ROWS)),
        ],
    }
    for accession, _, _, _, document in XOM_ROWS:
        routes[build_filing_url(XOM_HISTORICAL, accession, document)] = [FakeResponse(200, HTML)]
    session = FakeSession(routes)
    config = override_config(tmp_path)
    first = download_corpus(config, client=make_client(session))
    assert (first.downloaded, first.skipped, first.failures) == (2, 0, [])
    stored = json.loads(
        (tmp_path / "XOM" / "10-K" / XOM_ROWS[0][0] / "metadata.json").read_text(encoding="utf-8")
    )
    assert (stored["ticker"], stored["cik"], stored["company_name"], stored["cik_source"]) == (
        "XOM", XOM_HISTORICAL, "EXXON MOBIL CORP", "cik_override"
    )
    second = download_corpus(config, client=make_client(session))
    assert (second.downloaded, second.skipped) == (0, 2)


def test_ticker_map_filings_record_their_cik_source(tmp_path: Path) -> None:
    config = EdgarDownloadConfig(tickers=["AAPL"], filings_per_ticker=1, output_dir=tmp_path)
    summary = download_corpus(config, client=make_client(corpus_session()), dry_run=True)
    assert summary.selected[0].cik_source == "ticker_map"
    assert summary.cik_overrides_used == []


# --- pinned accessions -------------------------------------------------------------


def test_pinned_accessions_download_exactly_those_filings_whatever_the_newest_are(tmp_path: Path) -> None:
    pinned = ["0000320193-21-000105", "0000320193-23-000106"]  # an older 10-K first; not the newest two
    config = EdgarDownloadConfig(tickers=["AAPL"], filings_per_ticker=1, output_dir=tmp_path,
                                 accessions={"aapl": pinned})
    summary = download_corpus(config, client=make_client(corpus_session()))
    assert [f.accession_number for f in summary.selected] == pinned
    assert summary.downloaded == 2 and summary.failures == []
    assert sorted(p.name for p in (tmp_path / "AAPL" / "10-K").iterdir()) == sorted(pinned)


@pytest.mark.parametrize("missing", ["0000320193-99-000001", "0000320193-24-000081"])  # unknown; a 10-Q
def test_a_pinned_accession_that_is_not_a_10k_of_the_company_fails_the_ticker_and_downloads_nothing(
    tmp_path: Path, missing: str
) -> None:
    config = EdgarDownloadConfig(tickers=["AAPL"], output_dir=tmp_path,
                                 accessions={"AAPL": ["0000320193-23-000106", missing]})
    summary = download_corpus(config, client=make_client(corpus_session()))
    assert summary.downloaded == 0 and summary.selected == []
    assert "pinned 10-K accession(s) not found for AAPL" in summary.failures[0].error and missing in \
        summary.failures[0].error
    assert not (tmp_path / "AAPL").exists()


@pytest.mark.parametrize(("accessions", "message"), [
    ({"AAPL": ["320193-23-106"]}, "is not an accession number"),
    ({"AAPL": ["0000320193-23-000106", "0000320193-23-000106"]}, "listed twice"),
    ({"AAPL": []}, "non-empty list"),
    ({"MSFT": ["0000789019-23-000014"]}, "accessions for tickers not in tickers: MSFT"),
])
def test_pinned_accessions_are_validated(accessions: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EdgarDownloadConfig(tickers=["AAPL"], accessions=accessions)
