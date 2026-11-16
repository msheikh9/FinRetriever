"""Regression tests for section extraction on the locally downloaded 10-K corpus.

They run against the raw filings ``scripts/01_download_filings.py`` writes to
``data/raw`` (or ``FINRETRIEVER_RAW_DIR``): JPMorgan's annual report appended
after cross-reference Items, ExxonMobil's Financial Section after Item 16, and
the standard Item layouts of the other companies. Nothing is downloaded;
when the filings are absent — as in CI — every test here is skipped.
"""

from __future__ import annotations

import functools
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import pytest
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from finretriever.data.corpus_builder import LoadedFiling, discover_raw_filings, load_raw_filing
from finretriever.data.filing_parser import (
    COVERAGE_CATEGORIES,
    TABLE_START,
    ParsedFiling,
    analyze_filing,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = Path(os.environ.get("FINRETRIEVER_RAW_DIR", _REPO_ROOT / "data" / "raw"))
FILINGS = discover_raw_filings(RAW_DIR) if RAW_DIR.is_dir() else []

pytestmark = pytest.mark.skipif(
    not FILINGS, reason=f"no raw SEC filings under {RAW_DIR} (set FINRETRIEVER_RAW_DIR)"
)


def filings_of(*tickers: str) -> list[Path]:
    return [d for d in FILINGS if d.parent.parent.name in tickers]


def oldest_and_newest(ticker: str) -> list[Path]:
    found = filings_of(ticker)
    return [found[0], found[-1]] if found else []


def _id(directory: Path) -> str:
    return f"{directory.parent.parent.name}-{directory.name}"


@functools.cache
def parsed(directory: Path) -> tuple[LoadedFiling, ParsedFiling]:
    filing = load_raw_filing(directory)
    return filing, analyze_filing(filing.html, filing.filing_id)


def appended(result: ParsedFiling) -> list:
    return [s for s in result.sections if s.item_number is None]


def items(result: ParsedFiling) -> dict:
    return {s.item_number: s for s in result.sections if s.item_number is not None}


def words(text: str) -> int:
    return len(text.split())


# --- every filing ---------------------------------------------------------------


@pytest.mark.parametrize("directory", FILINGS, ids=_id)
def test_every_filing_is_sectioned_with_high_coverage(directory: Path) -> None:
    _, result = parsed(directory)
    coverage = result.coverage
    assert len(items(result)) >= 15
    assert coverage.coverage >= 0.9, coverage.words_by_category
    assert coverage.words_by_category["unassigned"] / coverage.source_words < 0.03
    # Every source word is counted exactly once.
    assert set(coverage.words_by_category) == set(COVERAGE_CATEGORIES)
    assert sum(coverage.words_by_category.values()) == coverage.source_words
    assert sum(coverage.section_words) == coverage.words_by_category["sections"]
    assert len(coverage.section_words) == len(result.sections)


@pytest.mark.parametrize("directory", FILINGS, ids=_id)
def test_sections_are_ordered_disjoint_and_uniquely_identified(directory: Path) -> None:
    _, result = parsed(directory)
    sections = result.sections
    assert all(a.end_char <= b.start_char for a, b in zip(sections, sections[1:], strict=False))
    assert len({s.section_id for s in sections}) == len(sections)
    item_numbers = [s.item_number for s in sections if s.item_number is not None]
    assert len(item_numbers) == len(set(item_numbers))
    # Items come first; appended report sections never carry an invented Item.
    kinds = [s.item_number is None for s in sections]
    assert kinds == sorted(kinds)
    assert not any(re.match(r"item\s+\d", s.title, re.I) for s in appended(result))


@pytest.mark.parametrize("directory", FILINGS, ids=_id)
def test_no_post_signature_or_duplicated_content(directory: Path) -> None:
    _, result = parsed(directory)
    assert result.coverage.words_by_category["signatures_and_after"] > 0
    text = "\n".join(s.text for s in result.sections)
    assert "has duly caused this report to be signed on its behalf" not in text
    # Sections are disjoint slices, so repeats come from the filing itself:
    # footnotes under several tables, repeated column headers, MD&A figures
    # restated in the statements.
    lines = [line for line in text.split("\n") if len(line) > 100]
    duplicated = sum(n - 1 for n in Counter(lines).values() if n > 1)
    assert duplicated / len(lines) < 0.08


@pytest.mark.parametrize("directory", FILINGS, ids=_id)
def test_no_bare_page_numbers_in_sections(directory: Path) -> None:
    _, result = parsed(directory)
    for section in result.sections:
        bare = [line for line in section.text.split("\n") if re.fullmatch(r"(?:page )?\d{1,3}", line, re.I)]
        assert bare == [], section.title


# --- JPMorgan: annual report appended after cross-reference Items ---------------

# A rendered balance-sheet row, e.g. "Total assets [a] | 2019: $2,687,379 | ...";
# tables whose multi-row headers cannot be aligned keep unlabelled values.
_TOTAL_ASSETS_RE = re.compile(
    r"^Total assets(?: \[\w+\])? \| (?:[^|\n]+: )?\$?\d{1,3}(?:,\d{3})+ \|", re.I | re.M
)


@pytest.mark.parametrize("directory", filings_of("JPM"), ids=_id)
def test_jpm_appended_annual_report_is_recovered(directory: Path) -> None:
    _, result = parsed(directory)
    report = appended(result)
    titles = [s.title.casefold() for s in report]
    assert result.coverage.coverage >= 0.98
    assert result.coverage.report_toc_unmatched == ()
    for title in (
        "executive overview",
        "consolidated results of operations",
        "firmwide risk management",
        "liquidity risk management",
        "report of independent registered public accounting firm",
        "consolidated balance sheets",
        "consolidated statements of cash flows",
    ):
        assert title in titles, title
    notes = [t for t in titles if re.match(r"note \d{1,2} [-–] ", t)]
    assert len(notes) >= 30
    assert notes[0].startswith("note 1 ")


@pytest.mark.parametrize("directory", filings_of("JPM"), ids=_id)
def test_jpm_cross_reference_items_are_stubs_not_report_copies(directory: Path) -> None:
    _, result = parsed(directory)
    by_item = items(result)
    for item in ("7", "8"):
        assert words(by_item[item].text) < 200, item
    overview = next(s for s in appended(result) if s.title.casefold() == "executive overview")
    assert words(overview.text) > 1000


@pytest.mark.parametrize("directory", oldest_and_newest("JPM"), ids=_id)
def test_jpm_financial_tables_are_preserved(directory: Path) -> None:
    _, result = parsed(directory)
    balance = next(s for s in appended(result) if s.title.casefold() == "consolidated balance sheets")
    assert balance.text.count(TABLE_START) >= 1
    assert _TOTAL_ASSETS_RE.search(balance.text)


# --- ExxonMobil: Financial Section after Item 16 --------------------------------

_XOM_NOTE_RE = re.compile(r"^(?:note )?\d{1,2}\. \w")


@pytest.mark.parametrize("directory", filings_of("XOM"), ids=_id)
def test_xom_item_16_does_not_absorb_the_financial_section(directory: Path) -> None:
    _, result = parsed(directory)
    report = appended(result)
    titles = [s.title.casefold() for s in report]
    assert words(items(result)["16"].text) < 50
    assert result.coverage.coverage >= 0.95
    assert result.coverage.report_toc_unmatched == ()
    for title in (
        "business profile",
        "liquidity and capital resources",
        "report of independent registered public accounting firm",
        "consolidated statement of income",
        "consolidated balance sheet",
        "consolidated statement of cash flows",
    ):
        assert title in titles, title
    notes = [t for t in titles if _XOM_NOTE_RE.match(t)]
    assert len(notes) >= 18
    assert titles.index(notes[0]) > titles.index("consolidated balance sheet")


@pytest.mark.parametrize("directory", oldest_and_newest("XOM"), ids=_id)
def test_xom_financial_statements_keep_values(directory: Path) -> None:
    _, result = parsed(directory)
    balance = next(s for s in appended(result) if s.title.casefold() == "consolidated balance sheet")
    assert balance.text.count(TABLE_START) >= 1
    assert _TOTAL_ASSETS_RE.search(balance.text)


def test_xom_note_heading_without_space_after_number_is_matched() -> None:
    found = [d for d in filings_of("XOM") if d.name.startswith("0000034088-20-")]
    if not found:
        pytest.skip("XOM FY2019 10-K not downloaded")
    _, result = parsed(found[0])
    titles = [s.title for s in appended(result)]
    # The report TOC lists "4. Other ..."; the body heading reads "4.Other ...".
    assert titles.index("4.Other Comprehensive Income Information") == titles.index("3. Miscellaneous Financial Information") + 1
    assert result.coverage.report_toc_unmatched == ()


# --- standard Item layouts -------------------------------------------------------


@pytest.mark.parametrize("directory", filings_of("AAPL", "AMZN", "PFE", "TSLA", "WMT"), ids=_id)
def test_standard_filings_have_only_item_sections(directory: Path) -> None:
    _, result = parsed(directory)
    by_item = items(result)
    assert appended(result) == []
    for item in ("1", "1A", "7", "8"):
        assert item in by_item, item
    assert words(by_item["1A"].text) > 2000


@pytest.mark.parametrize("directory", filings_of("AAPL", "AMZN", "TSLA", "WMT"), ids=_id)
def test_standard_filings_keep_mdna_and_statements_in_their_items(directory: Path) -> None:
    _, result = parsed(directory)
    by_item = items(result)
    assert words(by_item["7"].text) > 2000
    assert by_item["8"].text.count(TABLE_START) >= 10


def test_pfe_fy2019_financials_live_in_an_exhibit_not_in_the_document() -> None:
    found = [d for d in filings_of("PFE") if d.name.startswith("0000078003-20-")]
    if not found:
        pytest.skip("PFE FY2019 10-K not downloaded")
    _, result = parsed(found[0])
    by_item = items(result)
    # Items 7 and 8 incorporate the annual report exhibit by reference; nothing is invented.
    assert words(by_item["7"].text) < 100
    assert words(by_item["8"].text) < 100
    assert appended(result) == []


@pytest.mark.parametrize("directory", filings_of("AAPL"), ids=_id)
def test_apple_item_boundaries(directory: Path) -> None:
    _, result = parsed(directory)
    item_numbers = [s.item_number for s in result.sections]
    assert item_numbers[:3] == ["1", "1A", "1B"]
    assert item_numbers[-2:] == ["15", "16"]
    assert item_numbers.index("7") < item_numbers.index("7A") < item_numbers.index("8")
    by_item = items(result)
    assert by_item["1A"].title == "Risk Factors"
    assert by_item["7"].title.startswith("Management's Discussion and Analysis")
    assert by_item["8"].title == "Financial Statements and Supplementary Data"


# --- determinism and value preservation -----------------------------------------


@pytest.mark.parametrize(
    "directory", [*oldest_and_newest("JPM")[-1:], *oldest_and_newest("XOM")[-1:]], ids=_id
)
def test_section_ids_are_deterministic(directory: Path) -> None:
    filing, first = parsed(directory)
    second = analyze_filing(filing.html, filing.filing_id)
    assert [s.model_dump() for s in second.sections] == [s.model_dump() for s in first.sections]
    assert second.coverage == first.coverage


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@pytest.mark.parametrize(
    "directory", [*oldest_and_newest("JPM"), *oldest_and_newest("XOM")], ids=_id
)
def test_xbrl_facts_survive_into_sections(directory: Path) -> None:
    filing, result = parsed(directory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(filing.html, "lxml")
    for header in soup.find_all("ix:header"):
        header.decompose()
    facts = [
        match.group()
        for fact in soup.find_all("ix:nonfraction")
        if (match := _NUMBER_RE.search(fact.get_text()))
    ]
    present = set(_NUMBER_RE.findall("\n".join(s.text for s in result.sections)))
    missing = [fact for fact in facts if fact not in present]
    assert len(facts) > 1000
    # Only cover-page facts (public float, shares outstanding) fall outside sections.
    assert len(missing) <= 3, missing
