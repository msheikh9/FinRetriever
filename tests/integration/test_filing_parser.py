"""Integration tests for the filing parser on Apple's real FY2025 10-K.

The filing is an optional local fixture: download it with
``scripts/01_download_filings.py`` (see README) or point
``FINRETRIEVER_AAPL_10K`` at a saved copy. When it is absent — as in CI,
which never downloads SEC data — every test here is skipped.
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from pathlib import Path

import pytest
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from finretriever.data.filing_parser import (
    TABLE_START,
    extract_table_text,
    extract_visible_text,
    parse_filing,
)
from finretriever.data.schemas import make_filing_id

_DEFAULT_PATH = "/tmp/finretriever_smoke_raw/AAPL/10-K/0000320193-25-000079/filing.html"
FILING_PATH = Path(os.environ.get("FINRETRIEVER_AAPL_10K", _DEFAULT_PATH))
FILING_ID = make_filing_id("AAPL", "10-K", "0000320193-25-000079")

pytestmark = pytest.mark.skipif(
    not FILING_PATH.is_file(),
    reason=f"Apple 10-K fixture not found at {FILING_PATH} (set FINRETRIEVER_AAPL_10K)",
)


@pytest.fixture(scope="module")
def html() -> bytes:
    return FILING_PATH.read_bytes()


@pytest.fixture(scope="module")
def sections(html: bytes):
    return parse_filing(html, FILING_ID)


@pytest.fixture(scope="module")
def visible(html: bytes) -> str:
    return extract_visible_text(html)


@pytest.fixture(scope="module")
def by_item(sections) -> dict:
    return {s.item_number: s for s in sections}


@pytest.fixture(scope="module")
def soup(html: bytes) -> BeautifulSoup:
    """The raw filing with only the hidden XBRL header removed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        parsed = BeautifulSoup(html, "lxml")
    parsed.find("ix:header").decompose()
    return parsed


@pytest.fixture(scope="module")
def visible_lines(visible: str) -> set[str]:
    return set(visible.split("\n"))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def test_core_items_exist(by_item) -> None:
    for item in ("1", "1A", "7", "8"):
        assert item in by_item, f"Item {item} missing"
    assert by_item["1A"].title == "Risk Factors"
    assert by_item["7"].title.startswith("Management's Discussion and Analysis")
    assert by_item["8"].title == "Financial Statements and Supplementary Data"


def test_items_are_unique_and_in_document_order(sections) -> None:
    items = [s.item_number for s in sections]
    assert [s.start_char for s in sections] == sorted(s.start_char for s in sections)
    assert items == [
        "1", "1A", "1B", "1C", "2", "3", "4", "5", "7", "7A", "8",
        "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
    ]


def test_risk_factors_contain_meaningful_text(by_item) -> None:
    text = by_item["1A"].text
    assert len(text.split()) > 5000
    assert text.lower().count("adversely") >= 20
    assert "macroeconomic" in text.lower()


def test_mdna_contains_mdna_content(by_item) -> None:
    text = by_item["7"].text
    assert "Segment Operating Performance" in text
    assert "Gross Margin" in text or "Gross margin" in text
    assert "Americas | 2025: $178,353 | Change: 7%" in text


def test_financial_statements_keep_values(by_item) -> None:
    text = by_item["8"].text
    assert "CONSOLIDATED STATEMENTS OF OPERATIONS" in text
    assert "CONSOLIDATED BALANCE SHEETS" in text
    assert "Products | Years ended September 27, 2025: $307,003" in text
    assert "Total net sales | Years ended September 27, 2025: 416,161" in text
    assert text.count(TABLE_START) >= 20


def test_every_visible_xbrl_number_survives(soup: BeautifulSoup, visible: str) -> None:
    facts = [_clean(fact.get_text()) for fact in soup.find_all("ix:nonfraction")]
    assert len(facts) == 969
    assert [f for f in facts if f and f not in visible] == []


def test_every_numeric_table_cell_survives(soup: BeautifulSoup, visible: str) -> None:
    cells = [_clean(td.get_text()) for td in soup.find_all("td")]
    numbers = [c for c in cells if re.fullmatch(r"\(?\$?\d[\d,]*(\.\d+)?\)?%?", c)]
    assert len(numbers) == 1162
    assert [n for n in numbers if n not in visible] == []


def test_repeated_line_items_are_distinguished_by_parent(visible_lines: set[str]) -> None:
    ended = "Years ended September 27, 2025"
    assert f"Net sales > Products | {ended}: $307,003 | Years ended September 28, 2024: $294,866 | Years ended September 30, 2023: $298,085" in visible_lines
    assert f"Cost of sales > Products | {ended}: 194,116 | Years ended September 28, 2024: 185,233 | Years ended September 30, 2023: 189,282" in visible_lines
    assert f"Net sales > Services | {ended}: 109,158 | Years ended September 28, 2024: 96,169 | Years ended September 30, 2023: 85,200" in visible_lines
    assert "ASSETS > Current assets > Marketable securities | September 27, 2025: 18,763 | September 28, 2024: 35,228" in visible_lines
    assert "ASSETS > Non-current assets > Marketable securities | September 27, 2025: 77,723 | September 28, 2024: 91,479" in visible_lines


def test_totals_and_result_lines_carry_no_closed_group(visible: str) -> None:
    lines = visible.split("\n")
    for label in (
        "Total net sales | Years ended",
        "Total cost of sales | Years ended",
        "Gross margin | Years ended",
        "Cash generated by operating activities | Years ended",
        "Total assets | September 27, 2025: $359,241",
    ):
        assert any(line.startswith(label) for line in lines), label
    assert not any("> Total net sales" in line or "> Gross margin |" in line for line in lines)


def test_first_column_header_tables_are_aligned(visible_lines: set[str]) -> None:
    assert (
        "Investment portfolio | Hypothetical Interest Rate Increase: 100 basis points, all tenors | "
        "Potential Impact: Decline in fair value | 2025: $2,416 | 2024: $2,755"
    ) in visible_lines
    assert any(
        line.startswith(
            "June 29, 2025 to August 2, 2025 > Open market and privately negotiated purchases | "
            "Total Number of Shares Purchased: 33,265 | Average Price Paid Per Share: $210.43"
        )
        for line in visible_lines
    )


def test_rowspan_headers_label_the_right_debt_values(visible_lines: set[str]) -> None:
    assert any(
        line.startswith(
            "2013 – 2023 debt issuances > Fixed-rate 0.000% – 4.850% notes | "
            "Maturities (calendar year): 2025 – 2062 | 2025 Amount (in millions): $86,781 | "
            "2025 Effective Interest Rate: 0.03% – 5.75% | 2024 Amount (in millions): $97,341"
        )
        for line in visible_lines
    )


def test_group_paths_never_come_from_another_table(soup: BeautifulSoup) -> None:
    checked = 0
    for table in soup.find_all("table"):
        if table.find_parent("table"):
            continue
        source = _clean(table.get_text(" ")).replace("’", "'")
        for line in extract_table_text(table).split("\n"):
            label = line.split(" | ")[0]
            if " > " not in label:
                continue
            for group in label.split(" > ")[:-1]:
                assert re.sub(r" \[\d\]", "", group) in re.sub(r"\s?\(\d\)", "", source), (group, line)
                checked += 1
    assert checked > 100


def test_footnote_definitions_are_marked(visible_lines: set[str]) -> None:
    assert any(line.startswith("[1] Services net sales include amortization") for line in visible_lines)


def test_no_hidden_xbrl_or_page_artifacts(visible: str) -> None:
    for noise in ("fasb.org", "xbrli", "P1Y", "0000320193", "2025 Form 10-K |", " "):
        assert noise not in visible, noise


def test_sections_are_disjoint_slices_without_toc_rows(sections, visible: str) -> None:
    previous_end = 0
    for section in sections:
        assert visible[section.start_char:section.end_char] == section.text
        assert section.start_char >= previous_end
        previous_end = section.end_char
        assert "Risk Factors | 5" not in section.text  # TOC row


def test_no_major_text_duplication(sections) -> None:
    lines = [line for s in sections for line in s.text.split("\n") if len(line) > 100]
    duplicated = sum(n - 1 for n in Counter(lines).values() if n > 1)
    # The filing itself repeats a few lines (statement index, boilerplate).
    assert duplicated / len(lines) < 0.03
