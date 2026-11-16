"""Unit tests for finretriever.data.corpus_builder on synthetic raw filings.

``write_raw_filing`` reproduces the downloader's on-disk layout
(``<raw>/<TICKER>/<FORM>/<accession>/{filing.html, metadata.json}``) with a
correct SHA-256, so each test can then damage exactly one thing. Network
access is blocked: the corpus build must be fully offline.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from finretriever.data.chunker import chunk_text
from finretriever.data.corpus_builder import (
    CORPUS_BUILDER_VERSION,
    DIAGNOSTIC_KEYS,
    CorpusBuildConfig,
    CorpusBuildError,
    CorpusConfigError,
    FilingRejected,
    SectionFilterRules,
    StrictModeError,
    build_corpus,
    discover_raw_filings,
    load_corpus_build_config,
    load_raw_filing,
    section_filter_reason,
    write_corpus,
)
from finretriever.data.edgar_downloader import FilingMetadata, build_filing_url
from finretriever.data.filing_parser import TABLE_START, parse_filing
from finretriever.data.schemas import Section, make_filing_id, make_passage_id
from finretriever.utils.io import read_jsonl

FINANCIAL_TABLE = (
    "<table><tr><td></td><td>2025</td><td>2024</td></tr>"
    "<tr><td>Net sales:</td></tr>"
    "<tr><td>Products</td><td>$300</td><td>$290</td></tr>"
    "<tr><td>Services</td><td>100</td><td>90</td></tr>"
    "<tr><td>Total net sales</td><td>400</td><td>380</td></tr></table>"
)

STANDARD_ITEMS: list[tuple[str, str, str]] = [
    ("1", "Business", "<div>The Company designs and sells industrial widgets to manufacturers.</div>"),
    ("1A", "Risk Factors",
     "<div>Demand for widgets is cyclical and a downturn could materially adversely affect results.</div>"),
    ("1B", "Unresolved Staff Comments", "<div>None.</div>"),
    ("2", "Properties", "<div>The Company's headquarters is located in Ohio.</div>"),
    ("4", "Mine Safety Disclosures", "<div>Not applicable.</div>"),
    ("7", "Management's Discussion and Analysis", "<div>Net sales increased 6% on higher volumes.</div>"),
    ("8", "Financial Statements and Supplementary Data",
     "<div>CONSOLIDATED STATEMENTS OF OPERATIONS</div>" + FINANCIAL_TABLE),
    ("9B", "Other Information",
     "<div>On October 1, 2025, the CFO announced a transition.</div>"
     "<div>Insider Trading Arrangements</div><div>None.</div>"),
    ("10", "Directors, Executive Officers and Corporate Governance",
     "<div>The information required by this Item will be included in the 2026 Proxy Statement, "
     "and is incorporated herein by reference.</div>"),
]
KEPT_ITEMS = ["1", "1A", "2", "7", "8", "9B"]


def filing_html(items: list[tuple[str, str, str]] = STANDARD_ITEMS, company: str = "ACME CORP") -> str:
    """A small 10-K with a linked table of contents, like EDGAR inline-XBRL filings."""
    toc = "".join(
        f'<tr><td><a href="#i{item}">Item {item}.</a></td><td>{title}</td><td>{page}</td></tr>'
        for page, (item, title, _) in enumerate(items, start=1)
    )
    body = "".join(
        f'<div id="i{item}"></div><div><span style="font-weight:700">Item {item}. {title}</span></div>{html}'
        for item, title, html in items
    )
    return (
        f"<html><body><div>{company}</div><div>FORM 10-K</div><table>{toc}</table>{body}"
        "<div>SIGNATURES</div><div>Signed on behalf of the registrant.</div></body></html>"
    )


def write_raw_filing(
    raw_dir: Path,
    *,
    ticker: str = "ACME",
    accession: str = "0000000001-25-000001",
    filing_date: str = "2025-10-31",
    form_type: str = "10-K",
    cik: str = "0000000001",
    html: str | None = None,
    directory: Path | None = None,
    metadata_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write ``filing.html`` + ``metadata.json`` exactly as the downloader does."""
    body = (html if html is not None else filing_html(company=f"{ticker} CORP")).encode("utf-8")
    target = directory or raw_dir / ticker / form_type.replace("/", "_") / accession
    target.mkdir(parents=True, exist_ok=True)
    (target / "filing.html").write_bytes(body)
    metadata = FilingMetadata(
        ticker=ticker,
        cik=cik,
        company_name=f"{ticker} CORP",
        form_type=form_type,
        accession_number=accession,
        filing_date=filing_date,
        report_date="2025-09-27",
        primary_document=f"{ticker.lower()}-10k.htm",
        filing_url=build_filing_url(cik, accession, f"{ticker.lower()}-10k.htm"),
        local_html_path=(target / "filing.html").as_posix(),
        html_sha256=hashlib.sha256(body).hexdigest(),
    )
    payload = metadata.model_dump(mode="json") | (metadata_overrides or {})
    (target / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def make_config(tmp_path: Path, **overrides: Any) -> CorpusBuildConfig:
    settings: dict[str, Any] = {
        "name": "test_corpus",
        "raw_dir": tmp_path / "raw",
        "output_path": tmp_path / "out" / "corpus.jsonl",
        "tickers": ["ACME"],
    }
    settings.update(overrides)
    return CorpusBuildConfig(**settings)


def section(text: str) -> Section:
    return Section(section_id="s", filing_id="f", title="T", item_number="1", text=text)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted during a corpus build test")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    write_raw_filing(raw_dir)
    return raw_dir


# --- discovery and validation ---------------------------------------------------


def test_discovery_is_recursive_sorted_and_reports_incomplete_dirs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    beta = write_raw_filing(raw_dir, ticker="BETA", accession="0000000002-25-000001")
    acme = write_raw_filing(raw_dir, ticker="ACME")
    (beta / "metadata.json").unlink()
    (raw_dir / "notes.txt").write_text("not a filing", encoding="utf-8")
    (acme / "filing.html.abc.tmp").write_text("partial", encoding="utf-8")
    assert discover_raw_filings(raw_dir) == [acme, beta]


def test_missing_raw_dir_is_a_build_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusBuildError, match="raw filing directory not found"):
        discover_raw_filings(tmp_path / "nope")


@pytest.mark.parametrize(
    "damage,reason",
    [
        (lambda d: (d / "metadata.json").unlink(), "missing_metadata"),
        (lambda d: (d / "filing.html").unlink(), "missing_html"),
        (lambda d: (d / "metadata.json").write_text("{not json", encoding="utf-8"), "invalid_metadata"),
        (lambda d: (d / "filing.html").write_bytes(b"<html>tampered</html>"), "sha256_mismatch"),
    ],
    ids=["missing-metadata", "missing-html", "invalid-json", "sha256-mismatch"],
)
def test_incomplete_or_corrupt_filings_are_rejected(raw: Path, damage, reason: str) -> None:
    directory = discover_raw_filings(raw)[0]
    damage(directory)
    with pytest.raises(FilingRejected) as excinfo:
        load_raw_filing(directory)
    assert excinfo.value.reason == reason


def test_metadata_failing_schema_is_rejected(tmp_path: Path) -> None:
    directory = write_raw_filing(tmp_path / "raw", metadata_overrides={"accession_number": "bad"})
    with pytest.raises(FilingRejected) as excinfo:
        load_raw_filing(directory)
    assert excinfo.value.reason == "invalid_metadata"
    assert "accession_number" in excinfo.value.detail


def test_metadata_without_checksum_is_rejected(tmp_path: Path) -> None:
    directory = write_raw_filing(tmp_path / "raw", metadata_overrides={"html_sha256": None})
    with pytest.raises(FilingRejected, match="missing_checksum"):
        load_raw_filing(directory)


def test_invalid_filings_never_reach_the_corpus(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_filing(raw_dir)
    bad = write_raw_filing(raw_dir, ticker="BETA", accession="0000000002-25-000001")
    (bad / "filing.html").write_bytes(b"<html>tampered</html>")
    build = build_corpus(make_config(tmp_path, tickers=["ACME", "BETA"]))
    assert [(i.path, i.reason) for i in build.rejected] == [
        ("BETA/10-K/0000000002-25-000001", "sha256_mismatch")
    ]
    assert {p.ticker for p in build.passages} == {"ACME"}
    assert build.missing_tickers == ["BETA"]


def test_duplicate_accession_is_rejected_once(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_filing(raw_dir)
    write_raw_filing(raw_dir, directory=raw_dir / "copies" / "ACME")
    build = build_corpus(make_config(tmp_path))
    assert [(i.path, i.reason) for i in build.rejected] == [
        ("copies/ACME", "duplicate_accession")
    ]
    assert len(build.filings) == 1
    assert len({p.passage_id for p in build.passages}) == len(build.passages)


# --- scope ------------------------------------------------------------------------


def test_out_of_scope_filings_are_excluded_not_rejected(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_filing(raw_dir, accession="0000000001-24-000001", filing_date="2024-11-01")
    write_raw_filing(raw_dir, accession="0000000001-25-000001", filing_date="2025-10-31")
    write_raw_filing(raw_dir, ticker="ZZZ", accession="0000000009-25-000001")
    write_raw_filing(raw_dir, accession="0000000001-25-000077", form_type="10-Q")
    build = build_corpus(make_config(tmp_path, filings_per_ticker=1))
    assert sorted((i.reason, i.accession_number) for i in build.excluded) == [
        ("beyond_filings_per_ticker", "0000000001-24-000001"),
        ("form_type_not_configured", "0000000001-25-000077"),
        ("ticker_not_configured", "0000000009-25-000001"),
    ]
    assert build.rejected == []
    assert build.problems == []
    assert [f.accession_number for f in build.filings] == ["0000000001-25-000001"]


def test_filings_are_ordered_by_ticker_then_date(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_filing(raw_dir, ticker="BETA", accession="0000000002-25-000001", filing_date="2025-01-15")
    write_raw_filing(raw_dir, accession="0000000001-25-000001", filing_date="2025-10-31")
    write_raw_filing(raw_dir, accession="0000000001-24-000001", filing_date="2024-11-01")
    build = build_corpus(make_config(tmp_path, tickers=["ACME", "BETA"]))
    assert [(f.ticker, f.filing_date) for f in build.filings] == [
        ("ACME", "2024-11-01"), ("ACME", "2025-10-31"), ("BETA", "2025-01-15")
    ]
    passage_filings = list(dict.fromkeys(p.filing_id for p in build.passages))
    assert passage_filings == [f.filing_id for f in build.filings]


# --- filtering ----------------------------------------------------------------------


def test_boilerplate_sections_are_filtered_with_reasons(raw: Path, tmp_path: Path) -> None:
    build = build_corpus(make_config(tmp_path))
    assert [(s.item_number, s.reason) for s in build.filtered_sections] == [
        ("1B", "boilerplate:none"),
        ("4", "boilerplate:not_applicable"),
        ("10", "boilerplate:proxy_reference"),
    ]
    kept = list(dict.fromkeys(p.metadata["item_number"] for p in build.passages))
    assert kept == KEPT_ITEMS  # short Item 2 and 9B ("... None.") stay
    assert (build.sections_parsed, build.sections_kept) == (9, 6)


@pytest.mark.parametrize(
    "body,reason",
    [
        ("None.", "boilerplate:none"),
        ("  NOT APPLICABLE  ", "boilerplate:not_applicable"),
        ("[Reserved]", "boilerplate:reserved"),
        ("", "empty"),
        ("—", "empty"),
        ("The information required by this Item is incorporated herein by reference to the "
         "Company's Proxy Statement for its 2026 annual meeting.", "boilerplate:proxy_reference"),
        ("The Company leases one office.", None),
        ("None. The Company has no unresolved comments, but notes a pending SEC inquiry.", None),
        ("The information required by this Item is incorporated by reference to the Proxy "
         "Statement. Our code of ethics is posted at example.com.", None),
        (f"{TABLE_START}\nNone | 1\n[/TABLE]", None),
    ],
)
def test_section_filter_reason(body: str, reason: str | None) -> None:
    assert section_filter_reason(section(f"Item 1. Title\n{body}"), SectionFilterRules()) == reason


def test_long_body_is_never_boilerplate() -> None:
    rules = SectionFilterRules(max_boilerplate_words=5)
    body = "The information required by this Item is incorporated by reference to the Proxy Statement."
    assert section_filter_reason(section(f"Item 10. T\n{body}"), rules) is None


def test_filter_rules_are_configurable(raw: Path, tmp_path: Path) -> None:
    keep_all = SectionFilterRules(boilerplate_patterns={}, cross_reference_max_words=0)
    assert build_corpus(make_config(tmp_path, filters=keep_all)).filtered_sections == []
    custom = SectionFilterRules(boilerplate_patterns={"ohio_hq": r".*located in ohio\."}, cross_reference_max_words=0)
    build = build_corpus(make_config(tmp_path, filters=custom))
    assert [(s.item_number, s.reason) for s in build.filtered_sections] == [("2", "boilerplate:ohio_hq")]


def test_invalid_filter_regex_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a valid regex"):
        SectionFilterRules(boilerplate_patterns={"bad": "("})


def test_duplicate_passages_within_a_filing_are_filtered(tmp_path: Path) -> None:
    items = [
        ("1", "Business", "<div>Widgets.</div>"),
        ("7", "Results", "<div>" + "w1 w2 w3 w4 " * 3 + "</div>"),
    ]
    write_raw_filing(tmp_path / "raw", html=filing_html(items))
    config = make_config(tmp_path, chunk_size_tokens=4, chunk_overlap_tokens=0)
    build = build_corpus(config)
    assert dict(build.filtered_passages) == {"duplicate_text_in_filing": 1}
    assert len({p.text for p in build.passages}) == len(build.passages)
    keep = build_corpus(config.model_copy(update={"filters": SectionFilterRules(drop_duplicate_passages=False)}))
    assert len(keep.passages) == len(build.passages) + 1


# --- passages ---------------------------------------------------------------------


def test_passage_ids_are_deterministic_and_unique(raw: Path, tmp_path: Path) -> None:
    first = build_corpus(make_config(tmp_path))
    second = build_corpus(make_config(tmp_path))
    filing_id = make_filing_id("ACME", "10-K", "0000000001-25-000001")
    assert [p.passage_id for p in first.passages] == [p.passage_id for p in second.passages]
    for passage in first.passages:
        assert passage.filing_id == filing_id
        assert passage.passage_id == make_passage_id(
            filing_id, passage.section_id, passage.metadata["chunk_index"]
        )
    assert len({p.passage_id for p in first.passages}) == len(first.passages)


def test_passage_metadata(raw: Path, tmp_path: Path) -> None:
    build = build_corpus(make_config(tmp_path))
    item8 = next(p for p in build.passages if p.metadata["item_number"] == "8")
    assert (item8.ticker, item8.form_type, item8.filing_date) == ("ACME", "10-K", "2025-10-31")
    assert item8.section_title == "Financial Statements and Supplementary Data"
    assert item8.metadata == {
        "chunk_index": 0,
        "section_title": "Financial Statements and Supplementary Data",
        "item_number": "8",
        "accession_number": "0000000001-25-000001",
        "cik": "0000000001",
        "company_name": "ACME CORP",
        "report_date": "2025-09-27",
        "section_index": 6,
        "section_kind": "item",
        "num_chunks": 1,
        "contains_table": True,
    }
    assert "Net sales > Products | 2025: $300 | 2024: $290" in item8.text
    item1a = next(p for p in build.passages if p.metadata["item_number"] == "1A")
    assert item1a.metadata["contains_table"] is False


def test_chunks_respect_token_budget_and_item_boundaries(tmp_path: Path) -> None:
    long_body = "<div>" + " ".join(f"risk{i}" for i in range(60)) + ".</div>"
    items = [("1", "Business", long_body), ("1A", "Risk Factors", long_body), ("7", "Results", long_body)]
    html = filing_html(items)
    write_raw_filing(tmp_path / "raw", html=html)
    build = build_corpus(make_config(tmp_path, chunk_size_tokens=12, chunk_overlap_tokens=3))
    assert max(p.token_count for p in build.passages) <= 12
    assert all(len(p.text.split()) <= 12 for p in build.passages)
    sections = {s.section_id: s for s in parse_filing(html, build.filings[0].filing_id)}
    for passage in build.passages:
        assert passage.text in " ".join(sections[passage.section_id].text.split())
    assert all(p.metadata["num_chunks"] > 1 for p in build.passages)


def statement_table(rows: int, *, wide_values: int = 0) -> str:
    """A grouped two-period table, optionally with one very wide row."""
    body = "<tr><td>Net sales:</td></tr>" + "".join(
        f"<tr><td>Line item {i}</td><td>${i},000</td><td>{i}%</td></tr>" for i in range(1, rows + 1)
    )
    if wide_values:
        body += "<tr><td>Wide row</td>" + "".join(f"<td>{n},111</td>" for n in range(wide_values)) + "</tr>"
    return f"<table><tr><td></td><td>2025</td><td>Change</td></tr>{body}</table>"


def table_build(tmp_path: Path, table_html: str, size: int, overlap: int):
    items = [
        ("1", "Business", "<div>" + " ".join(f"word{i}" for i in range(70)) + "</div>"),
        ("8", "Statements", "<div>CONSOLIDATED STATEMENTS OF OPERATIONS</div>" + table_html),
        ("9A", "Controls", "<div>" + " ".join(f"ctl{i}" for i in range(50)) + "</div>"),
    ]
    html = filing_html(items)
    write_raw_filing(tmp_path / "raw", html=html)
    build = build_corpus(make_config(tmp_path, chunk_size_tokens=size, chunk_overlap_tokens=overlap))
    sections = {s.section_id: s for s in parse_filing(html, build.filings[0].filing_id)}
    return build, sections


def test_table_passages_never_begin_or_end_inside_a_row(tmp_path: Path) -> None:
    build, _ = table_build(tmp_path, statement_table(15), size=24, overlap=10)
    diagnostics = build.diagnostics
    assert diagnostics["table_rows"] == 15
    assert diagnostics["table_rows_requiring_continuation"] == 0
    assert diagnostics["passages_starting_mid_table_row"] == 0
    assert diagnostics["passages_ending_mid_table_row"] == 0
    assert diagnostics["unlabelled_partial_row_fragments"] == 0
    assert diagnostics["table_rows_not_represented"] == 0
    table_passages = [p for p in build.passages if p.metadata["item_number"] == "8"]
    assert len(table_passages) > 3
    for passage in table_passages:
        assert passage.token_count <= 24
        for i in range(1, 16):  # every row present is present whole, with its parent group
            if f"Line item {i} " in passage.text:
                assert f"Net sales > Line item {i} | 2025: ${i},000 | Change: {i}%" in passage.text


def test_long_row_is_continued_with_its_label(tmp_path: Path) -> None:
    build, _ = table_build(tmp_path, statement_table(3, wide_values=20), size=24, overlap=6)
    diagnostics = build.diagnostics
    assert diagnostics["table_rows_requiring_continuation"] == 1
    assert diagnostics["passages_starting_mid_table_row"] == 0
    assert diagnostics["unlabelled_partial_row_fragments"] == 0
    assert diagnostics["table_rows_not_represented"] == 0
    assert diagnostics["passages_ending_mid_table_row"] >= 1  # labelled first piece of the wide row
    continued = [p for p in build.passages if "Wide row (continued)" in p.text]
    assert continued and all(p.text.startswith("Net sales > Wide row (continued) |") for p in continued)
    all_text = " ".join(p.text for p in build.passages)
    assert all(f"{n},111" in all_text for n in range(20))
    assert max(p.token_count for p in build.passages) <= 24


def test_every_financial_value_and_token_is_preserved(tmp_path: Path) -> None:
    build, sections = table_build(tmp_path, statement_table(15, wide_values=6), size=24, overlap=10)
    numbers = re.compile(r"^\(?\$?\d[\d,]*(\.\d+)?\)?%?$")
    for section_id, section in sections.items():
        passages = [p for p in build.passages if p.section_id == section_id]
        if not passages:
            continue
        present = {t for p in passages for t in p.text.split()}
        missing = [t for t in section.text.split() if t not in present]
        assert missing == [], (section.item_number, missing[:5])
        assert all(t in present for t in section.text.split() if numbers.match(t))


def test_narrative_sections_keep_fixed_window_chunks(tmp_path: Path) -> None:
    build, sections = table_build(tmp_path, statement_table(15), size=24, overlap=10)
    for section_id, section in sections.items():
        if TABLE_START in section.text:
            continue
        passages = [p.text for p in build.passages if p.section_id == section_id]
        assert passages == chunk_text(section.text, 24, 10)


def test_table_chunks_never_cross_item_boundaries(tmp_path: Path) -> None:
    build, sections = table_build(tmp_path, statement_table(15), size=24, overlap=10)
    for passage in build.passages:
        section = sections[passage.section_id]
        assert passage.text in " ".join(section.text.split())
        assert passage.metadata["item_number"] == section.item_number


def test_manifest_lists_every_table_diagnostic(raw: Path, tmp_path: Path) -> None:
    manifest = write_corpus(build_corpus(make_config(tmp_path)))
    assert set(manifest["diagnostics"]) == set(DIAGNOSTIC_KEYS)
    assert manifest["diagnostics"]["unlabelled_partial_row_fragments"] == 0


# --- strictness and failure -------------------------------------------------------


def test_strict_mode_fails_on_any_rejected_filing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_filing(raw_dir)
    (write_raw_filing(raw_dir, accession="0000000001-24-000001") / "metadata.json").unlink()
    config = make_config(tmp_path, strict=True)
    with pytest.raises(StrictModeError, match="missing_metadata") as excinfo:
        build_corpus(config)
    assert excinfo.value.build.rejected[0].reason == "missing_metadata"
    assert not config.output_path.exists()
    lenient = build_corpus(config.model_copy(update={"strict": False}))
    assert len(lenient.filings) == 1 and len(lenient.rejected) == 1


def test_strict_mode_fails_on_configured_ticker_without_filings(raw: Path, tmp_path: Path) -> None:
    with pytest.raises(StrictModeError, match="no filings for configured tickers: BETA"):
        build_corpus(make_config(tmp_path, tickers=["ACME", "BETA"], strict=True))


def test_no_valid_filings_is_a_build_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pytest.raises(CorpusBuildError, match="no passages could be built"):
        build_corpus(make_config(tmp_path))
    bad = write_raw_filing(raw_dir)
    (bad / "filing.html").write_bytes(b"tampered")
    with pytest.raises(CorpusBuildError, match="sha256_mismatch"):
        build_corpus(make_config(tmp_path))


def test_unparseable_filing_is_rejected(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_filing(raw_dir)
    write_raw_filing(raw_dir, ticker="BETA", accession="0000000002-25-000001",
                     html="<html><body><div>No items here.</div></body></html>")
    build = build_corpus(make_config(tmp_path, tickers=["ACME", "BETA"]))
    assert [(i.ticker, i.reason) for i in build.rejected] == [("BETA", "no_sections")]
    assert build.missing_tickers == []  # BETA was selected, then failed to parse


# --- output and manifest ------------------------------------------------------------


def test_write_corpus_and_manifest(raw: Path, tmp_path: Path) -> None:
    config = make_config(tmp_path)
    build = build_corpus(config)
    manifest = write_corpus(build, built_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    data = config.output_path.read_bytes()
    assert read_jsonl(config.output_path) == [p.model_dump(mode="json") for p in build.passages]
    on_disk = json.loads(config.resolved_manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert config.resolved_manifest_path.name == "corpus.manifest.json"
    assert manifest["corpus_version"] == f"test_corpus@{CORPUS_BUILDER_VERSION}"
    assert manifest["built_at"] == "2026-01-02T03:04:05+00:00"
    assert manifest["corpus"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["corpus"]["bytes"] == len(data)
    assert manifest["counts"] == {
        "filing_directories_discovered": 1,
        "filings_included": 1,
        "filings_rejected": 0,
        "filings_excluded": 0,
        "sections_parsed": 9,
        "sections_kept": 6,
        "sections_filtered": 3,
        "passages": len(build.passages),
        "passages_filtered": 0,
    }
    assert manifest["filtered_sections_by_reason"] == {
        "boilerplate:none": 1, "boilerplate:not_applicable": 1, "boilerplate:proxy_reference": 1
    }
    assert manifest["source_accessions"] == ["0000000001-25-000001"]
    assert manifest["settings"]["chunk_size_tokens"] == 512
    assert manifest["settings"]["chunk_overlap_tokens"] == 80
    builder = manifest["builder"]
    assert set(builder["source_sha256"]) == {
        "data/corpus_builder.py", "data/filing_parser.py", "data/chunker.py", "data/schemas.py"
    }
    assert "git_commit" in builder and "git_dirty" in builder


def test_repeated_builds_write_identical_corpus(raw: Path, tmp_path: Path) -> None:
    first = make_config(tmp_path, output_path=tmp_path / "a" / "corpus.jsonl")
    second = make_config(tmp_path, output_path=tmp_path / "b" / "corpus.jsonl")
    m1 = write_corpus(build_corpus(first))
    m2 = write_corpus(build_corpus(second))
    assert first.output_path.read_bytes() == second.output_path.read_bytes()
    for manifest in (m1, m2):
        manifest.pop("built_at")
        manifest["corpus"].pop("path")
    assert m1 == m2


def test_failed_write_leaves_previous_corpus_untouched(raw: Path, tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.output_path.parent.mkdir(parents=True)
    config.output_path.write_text("previous corpus\n", encoding="utf-8")
    build = build_corpus(config)
    build.passages.append("not a passage")  # type: ignore[arg-type]  # fails mid-write
    with pytest.raises(TypeError):
        write_corpus(build)
    assert config.output_path.read_text(encoding="utf-8") == "previous corpus\n"
    assert sorted(p.name for p in config.output_path.parent.iterdir()) == ["corpus.jsonl"]


# --- configuration --------------------------------------------------------------------


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "corpus.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_from_shared_corpus_yaml(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        "name: corpus_v9\ntickers: [aapl, JPM]\nform_type: 10-K\nfilings_per_ticker: 3\n"
        "output_dir: data/raw\nchunk_size_tokens: 256\nchunk_overlap_tokens: 32\n"
        "paths:\n  corpus_jsonl: data/processed/corpus_v9.jsonl\n",
    )
    config = load_corpus_build_config(path)
    assert (config.name, config.tickers, config.filings_per_ticker) == ("corpus_v9", ["AAPL", "JPM"], 3)
    assert (config.chunk_size_tokens, config.chunk_overlap_tokens) == (256, 32)
    assert config.raw_dir == Path("data/raw")
    assert config.output_path == Path("data/processed/corpus_v9.jsonl")
    assert config.resolved_manifest_path == Path("data/processed/corpus_v9.manifest.json")
    override = load_corpus_build_config(
        path, raw_dir="/r", output="/o/c.jsonl", manifest="/o/m.json", strict=True
    )
    assert (override.raw_dir, override.output_path, override.resolved_manifest_path, override.strict) == (
        Path("/r"), Path("/o/c.jsonl"), Path("/o/m.json"), True
    )


def test_repo_corpus_yaml_loads() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_corpus_build_config(root / "configs" / "data" / "corpus_v1.yaml")
    assert (config.chunk_size_tokens, config.chunk_overlap_tokens) == (512, 80)
    assert config.output_path == Path("data/processed/corpus_v1.jsonl")


@pytest.mark.parametrize(
    "text,match",
    [
        ("tickers: [AAPL]\nfiling_per_ticker: 3\npaths: {corpus_jsonl: c.jsonl}\n", "did you mean 'filings_per_ticker'"),
        ("tickers: [AAPL]\n", "no paths.corpus_jsonl"),
        ("tickers: [AAPL]\nchunk_size_tokens: 80\nchunk_overlap_tokens: 80\npaths: {corpus_jsonl: c.jsonl}\n",
         "chunk_overlap_tokens must be smaller"),
        ("tickers: [AAPL\n", "invalid YAML"),
    ],
)
def test_invalid_config_is_reported(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(CorpusConfigError, match=match):
        load_corpus_build_config(write_yaml(tmp_path, text))


def test_manifest_path_must_differ_from_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest_path must differ"):
        make_config(tmp_path, manifest_path=tmp_path / "out" / "corpus.jsonl")


# --- exact table diagnostics -------------------------------------------------------

from finretriever.data.chunker import CONTINUATION_MARKER  # noqa: E402
from finretriever.data.corpus_builder import _table_diagnostics  # noqa: E402
from finretriever.data.schemas import Passage  # noqa: E402

SP500_SECTION = (
    "Item 5. Market\n"
    "The graph compares cumulative total return for the Company, the S&P 500\n"
    "[TABLE]\n"
    "S&P 500 Index | 2016: $100 | 2017: $119 | 2018: $140\n"
    "Dow Jones Index | 2016: $100 | 2017: $125 | 2018: $121\n"
    "[/TABLE]"
)


def passages_from(texts: list[str]) -> list[Passage]:
    return [
        Passage(passage_id=f"p{i}", filing_id="f", section_id="s", ticker="T", form_type="10-K",
                filing_date="2025-01-01", text=t)
        for i, t in enumerate(texts)
    ]


def test_prose_that_resembles_a_row_is_not_a_mid_row_fragment() -> None:
    tokens = SP500_SECTION.split()
    cut = tokens.index("[TABLE]")  # prose ends "... the S&P 500"; the row "S&P 500 Index | ..." follows
    diagnostics = _table_diagnostics(SP500_SECTION, passages_from([" ".join(tokens[:cut]), " ".join(tokens[cut:])]), 0)
    assert diagnostics["table_rows"] == 2
    assert diagnostics["passages_starting_mid_table_row"] == 0
    assert diagnostics["passages_ending_mid_table_row"] == 0
    assert diagnostics["unlabelled_partial_row_fragments"] == 0
    assert diagnostics["table_rows_not_represented"] == 0


def test_genuine_mid_row_cut_is_counted() -> None:
    tokens = SP500_SECTION.split()
    cut = tokens.index("2017:")  # inside the S&P 500 row
    diagnostics = _table_diagnostics(SP500_SECTION, passages_from([" ".join(tokens[:cut]), " ".join(tokens[cut:])]), 0)
    assert diagnostics["passages_ending_mid_table_row"] == 1
    assert diagnostics["passages_starting_mid_table_row"] == 1
    assert diagnostics["unlabelled_partial_row_fragments"] == 1


def test_labelled_continuation_pieces_are_uncertain_not_fragments() -> None:
    tokens = SP500_SECTION.split()
    cut = tokens.index("2017:")
    continuation = f"S&P 500 Index {CONTINUATION_MARKER} | " + " ".join(tokens[cut:])
    diagnostics = _table_diagnostics(
        SP500_SECTION, passages_from([" ".join(tokens[:cut]), continuation]), continued_rows=1
    )
    assert diagnostics["passages_not_located"] == 1
    assert diagnostics["unlabelled_partial_row_fragments"] == 0
    assert diagnostics["table_rows_requiring_continuation"] == 1
    assert diagnostics["table_rows_not_represented"] == 0


# --- coverage and cross-references ----------------------------------------------------


def test_manifest_records_per_filing_coverage(raw: Path, tmp_path: Path) -> None:
    # The synthetic filing is tiny, so its TOC and boilerplate dominate (~47%).
    manifest = write_corpus(build_corpus(make_config(tmp_path, coverage_warning_below=0.3)))
    record = manifest["filings"][0]
    excluded = record["excluded_source_words"]
    assert record["covered_source_words"] + sum(excluded.values()) == record["source_words"]
    assert excluded["filtered_boilerplate"] > 0
    assert record["coverage"] == round(record["covered_source_words"] / record["source_words"], 4)
    assert record["passage_tokens"] >= 1
    assert manifest["coverage"]["by_ticker"]["ACME"]["min"] == record["coverage"]
    assert manifest["coverage"]["warnings"] == []


def test_low_coverage_is_a_warning_not_a_rejection(raw: Path, tmp_path: Path) -> None:
    manifest = write_corpus(build_corpus(make_config(tmp_path, coverage_warning_below=0.999)))
    assert [w["ticker"] for w in manifest["coverage"]["warnings"]] == ["ACME"]
    assert manifest["counts"]["filings_included"] == 1


@pytest.mark.parametrize(
    "body,reason",
    [
        ("Refer to Item 10.", "boilerplate:cross_reference"),
        ("Refer to Note 30 for a description of the Firm's material legal proceedings.", "boilerplate:cross_reference"),
        ('Reference is made to the section entitled "Market Risks" in the Financial Section of this report.',
         "boilerplate:cross_reference"),
        ("Management's discussion and analysis appears on pages 53-160 of this report.", "boilerplate:cross_reference"),
        ('See Item 8 of Part II, "Financial Statements and Supplementary Data - Note 7 - Legal Proceedings."',
         "boilerplate:cross_reference"),
        ("Information required by Item 11 of Part III is included in our Proxy Statement and is incorporated "
         "herein by reference.", "boilerplate:cross_reference"),
        ("ExxonMobil has elected to use a $1 million threshold for environmental proceedings. Refer to Note 16.", None),
        ("Refer to Note 16. Separately, the Company settled a $40 million claim in March.", None),
        ("See the table below for the properties we lease.", None),
        ("None.\n28", "boilerplate:none"),
        ("The Company recorded revenue of\n28", None),
    ],
)
def test_cross_reference_and_page_number_filtering(body: str, reason: str | None) -> None:
    assert section_filter_reason(section(f"Item 3. Legal Proceedings\n{body}"), SectionFilterRules()) == reason


def test_appended_report_passages_are_marked(tmp_path: Path) -> None:
    from tests.unit.test_filing_parser import XOM_LIKE

    write_raw_filing(tmp_path / "raw", html=XOM_LIKE)
    build = build_corpus(make_config(tmp_path))
    kinds = Counter(p.metadata["section_kind"] for p in build.passages)
    assert kinds["appended_report"] > 0 and kinds["item"] > 0
    report = [p for p in build.passages if p.metadata["section_kind"] == "appended_report"]
    assert all("item_number" not in p.metadata for p in report)
    assert {p.section_title for p in report} >= {"BUSINESS PROFILE", "CONSOLIDATED BALANCE SHEET"}
    record = build.filings[0]
    assert record.appended_sections_kept == 8
    assert {s.reason for s in build.filtered_sections} >= {"boilerplate:cross_reference", "boilerplate:none"}
