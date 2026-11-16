"""Unit tests for finretriever.data.filing_parser on synthetic 10-K fixtures.

The fixtures mimic the structures seen in real inline-XBRL 10-Ks: a hidden
``ix:header``, a linked table of contents, TOC/body/cross-reference copies of
each Item heading, repeated page footers around page breaks, and financial
tables whose ``$``/``%``/``)`` symbols sit in their own cells.
"""

from __future__ import annotations

import warnings

import pytest
from bs4 import BeautifulSoup

from finretriever.data.filing_parser import (
    TABLE_END,
    TABLE_START,
    FilingParseError,
    extract_filing_sections,
    extract_table_text,
    extract_visible_text,
    normalize_filing_text,
    parse_filing,
)
from finretriever.data.schemas import make_section_id

FILING_ID = "filing_acme_10-k_test"


def wrap(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">'
        f"<head><title>acme-10k</title></head><body>{body}</body></html>"
    )


def footer_and_break(page: int) -> str:
    return f'<div>ACME Corp | 2025 Form 10-K | {page}</div><hr style="page-break-after:always"/>'


HIDDEN_HEADER = (
    '<div style="display:none"><ix:header><ix:hidden>'
    '<ix:nonNumeric name="dei:AmendmentFlag" contextRef="c-1">false</ix:nonNumeric>'
    "</ix:hidden><ix:resources><xbrli:context id=\"c-1\"><xbrli:entity>"
    '<xbrli:identifier scheme="http://www.sec.gov/CIK">0000999999</xbrli:identifier>'
    "</xbrli:entity></xbrli:context></ix:resources></ix:header></div>"
)

FINANCIAL_TABLE = """
<table>
  <tr><td></td><td colspan="6">Years ended</td><td colspan="2"></td></tr>
  <tr><td></td><td colspan="3">2025</td><td colspan="3">2024</td><td colspan="2">Change</td></tr>
  <tr><td>Net sales</td><td>$</td><td>1,234</td><td></td><td>$</td><td>1,100</td><td></td>
      <td>12</td><td>%</td></tr>
  <tr><td>Operating loss</td><td>$</td><td>(56</td><td>)</td><td>$</td><td>&#8212;</td><td></td>
      <td>(4)</td><td>%</td></tr>
  <tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
  <tr><td>Total</td><td colspan="2">1,178</td><td></td><td colspan="2">1,100</td><td></td>
      <td colspan="2">7%</td></tr>
</table>
"""

LINKED_FILING = wrap(
    HIDDEN_HEADER
    + """
<div>ACME CORP</div><div>FORM 10-K</div>
<table>
  <tr><td><a href="#p1">Part I</a></td></tr>
  <tr><td><a href="#i1">Item 1.</a></td><td><a href="#i1">Business</a></td><td><a href="#i1">1</a></td></tr>
  <tr><td><a href="#i1a">Item 1A.</a></td><td>Risk Factors</td><td>2</td></tr>
  <tr><td><a href="#i1b">Item 1B.</a></td><td>Unresolved Staff Comments</td><td>3</td></tr>
  <tr><td><a href="#p2">Part II</a></td></tr>
  <tr><td><a href="#i7">Item 7.</a></td><td>Management&#8217;s Discussion and Analysis</td><td>3</td></tr>
  <tr><td><a href="#i8">Item 8.</a></td><td>Financial Statements and Supplementary Data</td><td>4</td></tr>
</table>
<hr style="page-break-after:always"/>
<div id="p1"></div><div>PART I</div>
<div id="i1"></div><div><span style="font-weight:700">Item 1.</span><span style="font-weight:700">&#160;Business</span></div>
<div>ACME makes anvils. See Part I, <a href="#i1a">Item 1A</a> of this Form 10-K under the heading &#8220;Risk Factors.&#8221;</div>
<div>Item 7 of this report discusses results of operations.</div>
"""
    + footer_and_break(1)
    + """
<div id="i1a"></div><div><span style="font-weight:700">Item 1A. Risk Factors</span></div>
<div>Demand for the Company&#8217;s anvils could decline because roadrunners are&#160;fast.</div>
<div style="display:none">SECRET HIDDEN TEXT</div>
<table><tr><td>&#8226;</td><td>Supply of iron may be interrupted.</td></tr></table>
"""
    + footer_and_break(2)
    + """
<div id="i1b"></div><div>Item 1B. Unresolved Staff Comments</div>
<div id="p2"></div><div>PART II</div>
<div id="i7"></div><div>Item 7. Management&#8217;s Discussion and Analysis</div>
<div>Net sales were $<ix:nonFraction name="us-gaap:Revenues" contextRef="c-1" unitRef="usd"
  decimals="-6" scale="6">1,234</ix:nonFraction> million for fiscal year ended
  <ix:nonNumeric name="dei:DocumentPeriodEndDate" contextRef="c-1">September&#160;27, 2025</ix:nonNumeric>, which</div>
"""
    + footer_and_break(3)
    + """
<div>increased compared to 2024.</div>
<div id="i8"></div><div>Item 8. Financial Statements and Supplementary Data</div>
<div>CONSOLIDATED STATEMENTS OF OPERATIONS</div>
"""
    + FINANCIAL_TABLE
    + footer_and_break(4)
    + """
<div>SIGNATURES</div>
<div>Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934.</div>
"""
)

# No internal links at all; an unlinked TOC (with page numbers) precedes the body.
UNLINKED_FILING = wrap(
    """
<div>Item 1. Business 1</div>
<div>Item 1A. Risk Factors 4</div>
<div>Item 7. Management's Discussion and Analysis 9</div>
<div>Item 1. Business</div>
<div>The Company sells widgets. Item 1A of this report describes risks.</div>
<div>Item 1A. Risk Factors</div>
<div>Widget demand is cyclical.</div>
<div>Item 7. Management's Discussion and Analysis</div>
<div>Revenue grew 4%.</div>
"""
)


def table(html: str):
    return BeautifulSoup(html, "lxml").find("table")


@pytest.fixture(scope="module")
def linked_sections():
    return parse_filing(LINKED_FILING, FILING_ID)


@pytest.fixture(scope="module")
def linked_text():
    return extract_visible_text(LINKED_FILING)


def by_item(sections) -> dict:
    return {s.item_number: s for s in sections}


# --- linked TOC ---------------------------------------------------------------


def test_linked_toc_resolves_items_in_document_order(linked_sections) -> None:
    assert [s.item_number for s in linked_sections] == ["1", "1A", "7", "8"]
    starts = [s.start_char for s in linked_sections]
    assert starts == sorted(starts)


def test_sections_preserve_item_numbers_and_titles(linked_sections) -> None:
    assert [(s.item_number, s.title) for s in linked_sections] == [
        ("1", "Business"),
        ("1A", "Risk Factors"),
        ("7", "Management's Discussion and Analysis"),
        ("8", "Financial Statements and Supplementary Data"),
    ]
    assert linked_sections[0].text.startswith("Item 1. Business\n")


def test_toc_entries_are_not_included_in_sections(linked_sections) -> None:
    for section in linked_sections:
        assert "Unresolved Staff Comments | 3" not in section.text
        assert "Risk Factors | 2" not in section.text


def test_cross_references_do_not_become_boundaries(linked_sections) -> None:
    item1 = by_item(linked_sections)["1"].text
    assert "See Part I, Item 1A of this Form 10-K" in item1
    assert "Item 7 of this report discusses results of operations." in item1
    assert "Demand for the Company's anvils" not in item1


def test_empty_section_is_skipped_and_part_heading_is_a_boundary(linked_sections) -> None:
    assert "1B" not in by_item(linked_sections)
    item1a = by_item(linked_sections)["1A"].text
    assert "PART II" not in item1a
    assert "Unresolved Staff Comments" not in item1a


def test_last_section_stops_at_signatures(linked_sections) -> None:
    item8 = by_item(linked_sections)["8"].text
    assert "SIGNATURES" not in item8
    assert "Pursuant to the requirements" not in item8


def test_sections_are_exact_slices_of_visible_text(linked_sections, linked_text) -> None:
    previous_end = 0
    for section in linked_sections:
        assert linked_text[section.start_char:section.end_char] == section.text
        assert section.start_char >= previous_end
        previous_end = section.end_char


def test_section_ids_are_deterministic(linked_sections) -> None:
    again = parse_filing(LINKED_FILING, FILING_ID)
    assert [s.section_id for s in again] == [s.section_id for s in linked_sections]
    for index, section in enumerate(linked_sections):
        assert section.section_id == make_section_id(FILING_ID, section.title, index)
        assert section.filing_id == FILING_ID
    assert len({s.section_id for s in linked_sections}) == len(linked_sections)


MISANCHORED_TOC = (
    '<table><tr><td><a href="#a">Item 1.</a></td><td>Business</td></tr>'
    '<tr><td><a href="#b">Item 1A.</a></td><td>Risk Factors</td></tr>'
    '<tr><td><a href="#c">Item 7.</a></td><td>MD&amp;A</td></tr></table>'
    '<div id="a">Item 1. Business</div><div>Text one.</div>'
)


def test_wrong_toc_anchor_is_rejected_and_gap_filled_from_body_heading() -> None:
    html = wrap(
        MISANCHORED_TOC
        + "<div>Item 1A. Risk Factors</div><div>Risk text.</div>"
        + '<div id="c">Item 7. MD&amp;A</div><div id="b">A results paragraph, not a heading.</div>'
    )
    sections = extract_filing_sections(html, FILING_ID)
    assert [s.item_number for s in sections] == ["1", "1A", "7"]
    assert "Risk text." not in sections[0].text
    assert sections[1].text == "Item 1A. Risk Factors\nRisk text."


def test_item_without_verifiable_heading_is_not_fabricated() -> None:
    html = wrap(
        MISANCHORED_TOC
        + '<div id="b">Risk discussion without any Item 1A heading.</div>'
        + '<div id="c">Item 7. MD&amp;A</div><div>Results.</div>'
    )
    sections = extract_filing_sections(html, FILING_ID)
    assert [s.item_number for s in sections] == ["1", "7"]
    assert "Risk discussion without any Item 1A heading." in sections[0].text


# --- hidden content, XBRL, entities ----------------------------------------------


def test_hidden_xbrl_header_and_display_none_are_removed(linked_text) -> None:
    assert "0000999999" not in linked_text
    assert "false" not in linked_text
    assert "SECRET HIDDEN TEXT" not in linked_text


def test_visible_inline_xbrl_facts_are_preserved(linked_sections) -> None:
    item7 = by_item(linked_sections)["7"].text
    assert "Net sales were $1,234 million for fiscal year ended September 27, 2025" in item7


def test_numeric_entities_and_nbsp_are_normalized(linked_sections, linked_text) -> None:
    assert " " not in linked_text
    assert "&#" not in linked_text
    assert "Demand for the Company's anvils could decline because roadrunners are fast." in (
        by_item(linked_sections)["1A"].text
    )
    assert "—" in by_item(linked_sections)["8"].text  # em dash kept as a value


# --- page artifacts ---------------------------------------------------------------


def test_repeated_page_footers_are_removed(linked_text) -> None:
    assert "2025 Form 10-K |" not in linked_text


def test_paragraph_split_by_page_break_is_rejoined(linked_sections) -> None:
    item7 = by_item(linked_sections)["7"].text
    assert "September 27, 2025, which increased compared to 2024." in item7


def test_footer_like_text_is_kept_when_it_does_not_repeat() -> None:
    html = wrap(
        "<div>Item 1. Business</div><div>Page 7 of the prospectus explains this.</div>"
        "<hr/><div>Item 1A. Risk Factors</div><div>Risks.</div>"
    )
    assert "Page 7 of the prospectus explains this." in extract_visible_text(html)


# --- tables -------------------------------------------------------------------------


def test_financial_table_rows_keep_column_headers_and_symbols() -> None:
    assert extract_table_text(table(FINANCIAL_TABLE)) == "\n".join(
        [
            "Net sales | Years ended 2025: $1,234 | Years ended 2024: $1,100 | Change: 12%",
            "Operating loss | Years ended 2025: $(56) | Years ended 2024: $— | Change: (4)%",
            "Total | Years ended 2025: 1,178 | Years ended 2024: 1,100 | Change: 7%",
        ]
    )


def test_financial_table_is_marked_and_not_duplicated(linked_sections) -> None:
    item8 = by_item(linked_sections)["8"].text
    assert f"CONSOLIDATED STATEMENTS OF OPERATIONS\n{TABLE_START}\nNet sales |" in item8
    assert item8.count("Operating loss") == 1
    assert item8.rstrip().endswith(TABLE_END)


def test_table_without_header_rows_keeps_cells_separated() -> None:
    text = extract_table_text(
        table("<table><tr><td>Exhibit 3.1</td><td>Restated Articles</td><td>10-Q</td></tr>"
              "<tr><td>Exhibit 10.1</td><td>Stock Plan</td><td>8-K</td></tr></table>")
    )
    assert text == "Exhibit 3.1 | Restated Articles | 10-Q\nExhibit 10.1 | Stock Plan | 8-K"


def test_bullet_layout_table_becomes_narrative(linked_sections) -> None:
    item1a = by_item(linked_sections)["1A"].text
    assert "• Supply of iron may be interrupted." in item1a
    assert TABLE_START not in item1a


def test_nested_table_text_is_not_duplicated() -> None:
    text = extract_table_text(
        table("<table><tr><td>Outer</td><td><table><tr><td>Inner A</td><td>1</td></tr>"
              "</table></td></tr><tr><td>Second</td><td>2</td></tr></table>")
    )
    assert text == "Outer | Inner A | 1\nSecond | 2"


def test_labelled_first_row_is_not_treated_as_column_header() -> None:
    text = extract_table_text(
        table("<table><tr><td>Exhibit Number</td><td>Description</td></tr>"
              "<tr><td>3.1</td><td>Restated Articles</td></tr></table>")
    )
    assert text == "Exhibit Number | Description\n3.1 | Restated Articles"


def test_empty_table_renders_nothing() -> None:
    assert extract_table_text(table("<table><tr><td></td><td>&#160;</td></tr></table>")) == ""


def test_item_text_inside_financial_table_is_not_a_heading() -> None:
    html = wrap(
        "<div>Item 1. Business</div><div>Text.</div>"
        "<table><tr><td>Item 7. Reconciliation</td><td>$</td><td>5</td></tr>"
        "<tr><td>Other</td><td>$</td><td>6</td></tr></table>"
        "<div>Item 1A. Risk Factors</div><div>Risks.</div>"
    )
    sections = extract_filing_sections(html, FILING_ID)
    assert [s.item_number for s in sections] == ["1", "1A"]
    assert "Item 7. Reconciliation | $5" in sections[0].text


# --- fallback and failure modes -----------------------------------------------------


def test_missing_toc_uses_heading_fallback_and_skips_toc_lines() -> None:
    sections = parse_filing(UNLINKED_FILING, FILING_ID)
    assert [(s.item_number, s.text.split("\n")[1]) for s in sections] == [
        ("1", "The Company sells widgets. Item 1A of this report describes risks."),
        ("1A", "Widget demand is cyclical."),
        ("7", "Revenue grew 4%."),
    ]


def test_unlinked_toc_without_page_numbers_is_ignored() -> None:
    html = wrap(
        "<div>Item 1. Business</div><div>Item 1A. Risk Factors</div><div>Item 7. MD&amp;A</div>"
        "<div>Cover text.</div>"
        "<div>Item 1. Business</div><div>Body one.</div>"
        "<div>Item 1A. Risk Factors</div><div>Body two.</div>"
    )
    sections = extract_filing_sections(html, FILING_ID)
    assert [(s.item_number, s.text) for s in sections] == [
        ("1", "Item 1. Business\nBody one."),
        ("1A", "Item 1A. Risk Factors\nBody two."),
    ]


def test_too_few_headings_produce_no_sections() -> None:
    html = wrap("<div>Item 1. Business</div><div>Only one item here.</div>")
    assert extract_filing_sections(html, FILING_ID) == []
    with pytest.raises(FilingParseError, match="no Item sections found"):
        parse_filing(html, FILING_ID)


def test_parse_filing_rejects_blank_filing_id() -> None:
    with pytest.raises(ValueError, match="filing_id"):
        parse_filing(LINKED_FILING, "  ")


def test_malformed_html_is_handled() -> None:
    html = (
        "<html><body><div><b>Item 1. Business<div>Unclosed <span>text</div></b>"
        "</p></td><div>Item 1A. Risk Factors<div>Risky &amp; bold"
    )
    sections = extract_filing_sections(html, FILING_ID)
    assert [s.item_number for s in sections] == ["1", "1A"]
    assert "Risky & bold" in sections[1].text


@pytest.mark.parametrize("html", ["", b"", "<html></html>", "   "])
def test_empty_input(html: str | bytes) -> None:
    assert extract_visible_text(html) == ""
    assert extract_filing_sections(html, FILING_ID) == []


def test_bytes_and_str_input_agree() -> None:
    assert extract_visible_text(LINKED_FILING.encode("utf-8")) == extract_visible_text(LINKED_FILING)


def test_parser_warning_filter_is_not_global() -> None:
    extract_visible_text(LINKED_FILING)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("probe", UserWarning, stacklevel=2)
    assert any(str(w.message) == "probe" for w in caught)
    assert not any(
        f[0] == "ignore" and getattr(f[2], "__name__", "") == "XMLParsedAsHTMLWarning"
        for f in warnings.filters
    )


# --- normalize_filing_text ----------------------------------------------------------


def test_normalize_filing_text() -> None:
    raw = "  The Company’s  “net” sales​ — up\r\n\n\t 5%­  \n"
    assert normalize_filing_text(raw) == "The Company's \"net\" sales — up\n5%"


# --- hierarchical row context ------------------------------------------------------

RULE = ' style="border-top:1px solid #000000"'
YEAR_HEADER = "<tr><td></td><td colspan='2'>2025</td><td colspan='2'>2024</td></tr>"
BLANK_ROW = "<tr><td></td><td></td><td></td><td></td><td></td></tr>"


def row(label: str, a: str, b: str, *, dollar: bool = False, rule: bool = False) -> str:
    sign, border = ("$" if dollar else ""), (RULE if rule else "")
    return (
        f"<tr><td>{label}</td><td{border}>{sign}</td><td{border}>{a}</td>"
        f"<td{border}>{sign}</td><td{border}>{b}</td></tr>"
    )


def group(label: str, style: str = "") -> str:
    return f'<tr><td style="{style}">{label}</td></tr>'


def render(rows: str) -> list[str]:
    return extract_table_text(table(f"<table>{YEAR_HEADER}{rows}</table>")).split("\n")


INCOME_STATEMENT = (
    group("Net sales:")
    + row("Products", "300", "290", dollar=True)
    + row("Services", "100", "90")
    + row("Total net sales", "400", "380", rule=True)
    + BLANK_ROW
    + group("Cost of sales:")
    + row("Products", "190", "185", dollar=True)
    + row("Services", "30", "25")
    + row("Total cost of sales", "220", "210", rule=True)
    + row("Gross margin", "180", "170", rule=True)
)


def test_parent_group_is_carried_into_child_rows() -> None:
    assert render(INCOME_STATEMENT) == [
        "Net sales > Products | 2025: $300 | 2024: $290",
        "Net sales > Services | 2025: 100 | 2024: 90",
        "Total net sales | 2025: 400 | 2024: 380",
        "Cost of sales > Products | 2025: $190 | 2024: $185",
        "Cost of sales > Services | 2025: 30 | 2024: 25",
        "Total cost of sales | 2025: 220 | 2024: 210",
        "Gross margin | 2025: 180 | 2024: 170",
    ]


def test_repeated_labels_stay_distinguishable_by_parent() -> None:
    products = [line for line in render(INCOME_STATEMENT) if "Products" in line]
    assert products == [
        "Net sales > Products | 2025: $300 | 2024: $290",
        "Cost of sales > Products | 2025: $190 | 2024: $185",
    ]


def test_parent_does_not_leak_into_another_table() -> None:
    html = wrap(
        "<div>Item 1. Business</div>"
        f"<table>{YEAR_HEADER}{group('Net sales:')}{row('Products', '300', '290')}</table>"
        "<div>Segment information follows.</div>"
        f"<table>{YEAR_HEADER}{row('Products', '50', '40')}{row('Services', '5', '4')}</table>"
        "<div>Item 1A. Risk Factors</div><div>Risks.</div>"
    )
    lines = extract_visible_text(html).split("\n")
    assert "Net sales > Products | 2025: 300 | 2024: 290" in lines
    assert "Products | 2025: 50 | 2024: 40" in lines
    assert "Services | 2025: 5 | 2024: 4" in lines
    assert sum("Net sales >" in line for line in lines) == 1


@pytest.mark.parametrize(
    "separator,next_row",
    [
        (row("Total net sales", "400", "380"), "Gross margin | 2025: 180 | 2024: 170"),
        (BLANK_ROW, "Gross margin | 2025: 180 | 2024: 170"),
        (group("Commitments and contingencies"), "Gross margin | 2025: 180 | 2024: 170"),
    ],
    ids=["after-total", "after-blank-row", "after-label-only-row"],
)
def test_parent_does_not_carry_into_unrelated_category(separator: str, next_row: str) -> None:
    lines = render(
        group("Net sales:") + row("Products", "300", "290") + separator + row("Gross margin", "180", "170")
    )
    assert next_row in lines
    assert not any(line.startswith("Net sales > Gross margin") for line in lines)


def test_ruled_row_closes_group_but_a_ruled_first_child_does_not() -> None:
    lines = render(
        group("Net sales:")
        + row("Products", "300", "290", rule=True)
        + row("Services", "100", "90")
        + row("Net sales, all categories", "400", "380", rule=True)
        + row("Operating income", "120", "110")
    )
    assert lines == [
        "Net sales > Products | 2025: 300 | 2024: 290",
        "Net sales > Services | 2025: 100 | 2024: 90",
        "Net sales, all categories | 2025: 400 | 2024: 380",
        "Operating income | 2025: 120 | 2024: 110",
    ]


def test_indented_groups_nest_and_survive_blank_row_only_for_deeper_group() -> None:
    lines = render(
        group("ASSETS:", "padding:2px 1pt")
        + group("Current assets:", "padding:2px 1pt 2px 10pt")
        + row("Marketable securities", "18", "35")
        + row("Total current assets", "148", "153")
        + BLANK_ROW
        + group("Non-current assets:", "padding-left:10pt")
        + row("Marketable securities", "78", "91")
        + row("Total non-current assets", "211", "212")
        + row("Total assets", "359", "365")
    )
    assert lines == [
        "ASSETS > Current assets > Marketable securities | 2025: 18 | 2024: 35",
        "ASSETS > Total current assets | 2025: 148 | 2024: 153",
        "ASSETS > Non-current assets > Marketable securities | 2025: 78 | 2024: 91",
        "ASSETS > Total non-current assets | 2025: 211 | 2024: 212",
        "Total assets | 2025: 359 | 2024: 365",
    ]


def test_groups_do_not_nest_without_indentation() -> None:
    lines = render(
        group("ASSETS:") + group("Current assets:") + row("Cash", "10", "9")
        + row("Total current assets", "10", "9")
    )
    assert lines == [
        "ASSETS:",
        "Current assets > Cash | 2025: 10 | 2024: 9",
        "Total current assets | 2025: 10 | 2024: 9",
    ]


def test_unlabelled_continuation_row_keeps_group_path_only_inside_group() -> None:
    lines = render(
        row("", "1", "2")
        + group("Lease liabilities:")
        + row("Operating leases", "5", "4")
        + row("", "6", "7")
        + row("Total lease liabilities", "11", "11")
    )
    assert lines == [
        "2025: 1 | 2024: 2",
        "Lease liabilities > Operating leases | 2025: 5 | 2024: 4",
        "Lease liabilities | 2025: 6 | 2024: 7",
        "Total lease liabilities | 2025: 11 | 2024: 11",
    ]


def test_subtotal_closes_group() -> None:
    lines = render(group("Level 1:") + row("Money market funds", "5", "4") + row("Subtotal", "5", "4"))
    assert lines == ["Level 1 > Money market funds | 2025: 5 | 2024: 4", "Subtotal | 2025: 5 | 2024: 4"]


# --- column-header alignment ------------------------------------------------------


def test_header_row_with_first_column_caption_is_aligned() -> None:
    text = extract_table_text(
        table(
            "<table><tr><td>Periods</td><td>Shares Purchased</td><td>Average Price</td></tr>"
            "<tr><td>June 2025:</td></tr>"
            "<tr><td>Open market purchases</td><td>33,265</td><td>$210.43</td></tr></table>"
        )
    )
    assert text == (
        "Periods | Shares Purchased | Average Price\n"
        "June 2025 > Open market purchases | Shares Purchased: 33,265 | Average Price: $210.43"
    )


def test_text_only_data_row_after_captioned_header_is_not_a_header() -> None:
    text = extract_table_text(
        table("<table><tr><td>Title of class</td><td>Symbol</td><td>Exchange</td></tr>"
              "<tr><td>Common Stock</td><td>AAPL</td><td>Nasdaq</td></tr>"
              "<tr><td>0.000% Notes due 2025</td><td>&#8212;</td><td>Nasdaq</td></tr></table>")
    )
    assert text.split("\n") == [
        "Title of class | Symbol | Exchange",
        "Common Stock | Symbol: AAPL | Exchange: Nasdaq",
        "0.000% Notes due 2025 | Symbol: — | Exchange: Nasdaq",
    ]


def test_period_header_with_first_column_caption_is_aligned() -> None:
    text = extract_table_text(
        table("<table><tr><td>Fiscal year</td><td>2025</td></tr>"
              "<tr><td>Revenue</td><td>$100</td></tr><tr><td>Costs</td><td>(40)</td></tr></table>")
    )
    assert text == "Fiscal year | 2025\nRevenue | 2025: $100\nCosts | 2025: (40)"


def test_header_spanning_two_values_is_not_assigned_to_either() -> None:
    text = extract_table_text(
        table("<table><tr><td></td><td colspan='2'>2025</td></tr>"
              "<tr><td>Revenue</td><td>$100</td><td>10%</td></tr>"
              "<tr><td>Costs</td><td>$40</td><td>4%</td></tr></table>")
    )
    assert text == "2025\nRevenue | $100 | 10%\nCosts | $40 | 4%"


def test_shifted_value_is_not_given_the_neighbouring_column_label() -> None:
    text = extract_table_text(
        table("<table><tr><td></td><td>2025</td><td></td><td>2024</td></tr>"
              "<tr><td>Revenue</td><td>$100</td><td></td><td>$90</td></tr>"
              "<tr><td>Costs</td><td></td><td colspan='2'>(40)</td><td>(35)</td></tr></table>")
    )
    assert text.split("\n") == [
        "Revenue | 2025: $100 | 2024: $90",
        "Costs | (40) | (35)",
    ]


def test_rowspan_header_keeps_lower_header_cells_in_their_columns() -> None:
    text = extract_table_text(
        table(
            "<table>"
            "<tr><td rowspan='2'></td><td rowspan='2'>Maturities</td><td colspan='2'>2025</td></tr>"
            "<tr><td>Amount</td><td>Rate</td></tr>"
            "<tr><td>Fixed-rate notes</td><td>2025 &#8211; 2062</td><td>$86,781</td><td>0.03% &#8211; 5.75%</td></tr>"
            "<tr><td>Total principal</td><td></td><td>91,281</td><td></td></tr>"
            "</table>"
        )
    )
    assert text.split("\n") == [
        "Fixed-rate notes | Maturities: 2025 – 2062 | 2025 Amount: $86,781 | 2025 Rate: 0.03% – 5.75%",
        "Total principal | 2025 Amount: 91,281",
    ]


def test_rowspan_row_label_applies_to_the_rows_it_spans() -> None:
    text = extract_table_text(
        table(
            "<table><tr><td></td><td></td><td>2025</td></tr>"
            "<tr><td rowspan='2'>Operating leases</td><td>Current</td><td>5</td></tr>"
            "<tr><td>Non-current</td><td>6</td></tr>"
            "<tr><td>Finance leases</td><td>Current</td><td>7</td></tr></table>"
        )
    )
    assert text.split("\n") == [
        "Operating leases | Current | 2025: 5",
        "Operating leases | Non-current | 2025: 6",
        "Finance leases | Current | 2025: 7",
    ]


# --- footnotes vs negative values ---------------------------------------------------


def test_superscript_footnote_is_distinguished_from_negative_value() -> None:
    sup = '<span style="position:relative;top:-2.8pt;vertical-align:baseline">(1)</span>'
    text = extract_table_text(
        table(f"<table>{YEAR_HEADER}<tr><td>Services{sup}</td><td></td><td>(1)</td>"
              "<td></td><td>2</td></tr></table>")
    )
    assert text == "Services [1] | 2025: (1) | 2024: 2"


def test_footnote_definition_marker_is_bracketed() -> None:
    html = wrap(
        "<div>Item 1. Business</div><div><span>(1)</span><span>Services net sales include amortization.</span></div>"
        "<div>(a) Documents filed as part of this report.</div>"
        "<div>Item 1A. Risk Factors</div><div>Risks.</div>"
    )
    lines = extract_visible_text(html).split("\n")
    assert "[1] Services net sales include amortization." in lines
    assert "(a) Documents filed as part of this report." in lines


# --- appended reports (non-standard layouts) -----------------------------------------

from finretriever.data.filing_parser import analyze_filing  # noqa: E402


def page(n: int) -> str:
    """A page break preceded by a bare page number, as many filers lay pages out."""
    return f'<div>{n}</div><hr style="page-break-after:always"/>'


def item_toc(items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f'<tr><td><a href="#i{n}">Item {n}.</a></td><td>{title}</td><td>{p}</td></tr>'
        for p, (n, title) in enumerate(items, start=1)
    )
    return f"<table>{rows}</table>"


def item(n: str, title: str, body: str) -> str:
    return f'<div id="i{n}"></div><div><b>Item {n}. {title}</b></div>{body}'


XOM_ITEMS = [("1", "Business"), ("1A", "Risk Factors"), ("7", "MD&amp;A"), ("8", "Financial Statements"),
             ("15", "Exhibits"), ("16", "Form 10-K Summary")]
XOM_LIKE = wrap(
    item_toc(XOM_ITEMS) + page(1)
    + item("1", "Business", "<div>The Corporation explores for and produces crude oil.</div>")
    + item("1A", "Risk Factors", "<div>Commodity prices may fall sharply.</div>") + page(2)
    + item("7", "MD&amp;A", "<div>Reference is made to the section entitled MD&amp;A in the Financial Section of this report.</div>")
    + item("8", "Financial Statements", "<div>Reference is made to the Financial Section of this report.</div>")
    + item("15", "Exhibits", "<div>See Index to Exhibits of this report.</div>")
    + item("16", "Form 10-K Summary", "<div>None.</div>") + page(3)
    + "<div>FINANCIAL SECTION</div><table>"
    "<tr><td>TABLE OF CONTENTS</td></tr>"
    "<tr><td>Business Profile</td><td>30</td></tr>"
    "<tr><td>Management's Discussion and Analysis</td></tr>"
    "<tr><td>Overview</td><td>31</td></tr>"
    "<tr><td>Liquidity and Capital Resources</td><td>33</td></tr>"
    "<tr><td>Consolidated Financial Statements</td></tr>"
    "<tr><td>Statement of Income</td><td>40</td></tr>"
    "<tr><td>Balance Sheet</td><td>41</td></tr>"
    "<tr><td>Notes to Consolidated Financial Statements</td></tr>"
    "<tr><td>1. Summary of Accounting Policies</td><td>42</td></tr>"
    "<tr><td>2. Earnings Per Share</td><td>44</td></tr>"
    "<tr><td>Supplemental Information on Oil and Gas</td><td>50</td></tr></table>" + page(4)
    + "<div>BUSINESS PROFILE</div><div>Upstream earnings were $5,063 million.</div>"
    "<div>OVERVIEW</div><div>Results reflected lower prices.</div>" + page(5)
    + "<table><tr><td>LIQUIDITY AND CAPITAL RESOURCES</td><td>2025</td><td>2024</td></tr>"
    "<tr><td>Cash from operations</td><td>51,970</td><td>55,022</td></tr></table>"
    "<div>Liquidity remained strong.</div>" + page(6)
    + "<div>CONSOLIDATED STATEMENT OF INCOME</div>"
    "<table><tr><td></td><td>2025</td><td>2024</td></tr><tr><td>Revenue</td><td>323,905</td><td>339,247</td></tr></table>"
    "<div>CONSOLIDATED BALANCE SHEET</div>"
    "<table><tr><td></td><td>2025</td><td>2024</td></tr><tr><td>Total assets</td><td>448,980</td><td>453,475</td></tr></table>"
    + page(7)
    + "<div>NOTES TO CONSOLIDATED FINANCIAL STATEMENTS</div>"
    "<div>Note 1. Summary of Accounting Policies</div><div>Principles of consolidation apply.</div>" + page(8)
    + "<div>NOTES TO CONSOLIDATED FINANCIAL STATEMENTS</div>"
    "<div>Note 2. Earnings Per Share</div><div>Earnings per share were $6.70.</div>" + page(9)
    + "<div>SUPPLEMENTAL INFORMATION ON OIL AND GAS (unaudited)</div><div>Proved reserves grew.</div>"
    "<div>INDEX TO EXHIBITS</div><div>3(i) Restated Certificate of Incorporation.</div>"
    "<div>SIGNATURES</div><div>Pursuant to the requirements of the Act.</div><div>POWER OF ATTORNEY</div>"
)


@pytest.fixture(scope="module")
def xom_like():
    return analyze_filing(XOM_LIKE, FILING_ID)


def test_appended_financial_section_is_split_by_its_own_headings(xom_like) -> None:
    appended = [(s.title, s.item_number) for s in xom_like.sections if s.item_number is None]
    assert appended == [
        ("BUSINESS PROFILE", None),
        ("OVERVIEW", None),
        ("LIQUIDITY AND CAPITAL RESOURCES", None),  # heading is a table's first cell
        ("CONSOLIDATED STATEMENT OF INCOME", None),  # "Statement of Income" + prefix
        ("CONSOLIDATED BALANCE SHEET", None),
        ("Note 1. Summary of Accounting Policies", None),
        ("Note 2. Earnings Per Share", None),
        ("SUPPLEMENTAL INFORMATION ON OIL AND GAS (unaudited)", None),
    ]
    assert xom_like.coverage.report_toc_unmatched == ()


def test_item_16_no_longer_absorbs_the_appended_report(xom_like) -> None:
    item16 = next(s for s in xom_like.sections if s.item_number == "16")
    assert item16.text == "Item 16. Form 10-K Summary\nNone."
    assert [s.item_number for s in xom_like.sections if s.item_number] == ["1", "1A", "7", "8", "15", "16"]


def test_appended_sections_keep_order_values_and_exclude_exhibits_and_signatures(xom_like) -> None:
    starts = [s.start_char for s in xom_like.sections]
    assert starts == sorted(starts)
    text = "\n".join(s.text for s in xom_like.sections)
    for value in ("$5,063", "51,970", "323,905", "448,980", "$6.70"):
        assert value in text
    for excluded in ("Restated Certificate", "Pursuant to the requirements", "POWER OF ATTORNEY", "TABLE OF CONTENTS"):
        assert excluded not in text
    assert text.count("NOTES TO CONSOLIDATED FINANCIAL STATEMENTS") == 2  # running headers stay content, not sections


def test_coverage_accounts_for_every_source_word(xom_like) -> None:
    coverage = xom_like.coverage
    words = coverage.words_by_category
    assert sum(words.values()) == coverage.source_words
    assert sum(coverage.section_words) == words["sections"]
    assert words["report_toc"] > 0 and words["exhibit_index"] > 0 and words["signatures_and_after"] > 0
    assert words["unassigned"] == 0
    assert 0.5 < coverage.coverage < 1.0
    assert coverage.report_sections == 8 and coverage.report_toc_entries == 8


def test_appended_section_ids_are_deterministic(xom_like) -> None:
    again = analyze_filing(XOM_LIKE, FILING_ID)
    assert [s.section_id for s in again.sections] == [s.section_id for s in xom_like.sections]
    for index, section in enumerate(xom_like.sections):
        assert section.section_id == make_section_id(FILING_ID, section.title, index)


JPM_ITEMS = [("1", "Business"), ("1A", "Risk Factors"), ("3", "Legal Proceedings"), ("7", "MD&amp;A"), ("15", "Exhibits")]
JPM_LIKE = wrap(
    item_toc(JPM_ITEMS) + page(1)
    + item("1", "Business", "<div>The Firm is a global financial services company.</div>")
    + item("1A", "Risk Factors", "<div>Refer to the Risk Management section on pages 9-12.</div>")
    + item("3", "Legal Proceedings", "<div>Refer to Note 2 for a description of legal proceedings.</div>")
    + item("7", "MD&amp;A", "<div>Management's discussion and analysis appears on pages 5-12.</div>")
    + "<div>PART IV</div>"
    + item("15", "Exhibits", "<table><tr><td>3.1</td><td>Certificate of Incorporation</td></tr></table>") + page(2)
    + "<div>Part IV</div>"  # running page header on the exhibit-list pages
    + "<table><tr><td>10.1</td><td>Deferred Compensation Plan</td></tr></table>" + page(3)
    + "<div>Table of contents</div><table>"
    "<tr><td>Financial:</td><td></td><td>Audited financial statements:</td></tr>"
    "<tr><td>5</td><td>Executive Overview</td><td>20</td><td>Report of Independent Registered Public Accounting Firm</td></tr>"
    "<tr><td>9</td><td>Capital Risk Management</td><td>22</td><td>Consolidated Financial Statements</td></tr>"
    "<tr><td>12</td><td>Market Risk Management</td><td>27</td><td>Notes to Consolidated Financial Statements</td></tr>"
    "</table>" + page(4)
    + "<div>EXECUTIVE OVERVIEW</div><div>Net income was $58.5 billion.</div>" + page(5)
    + "<div>CAPITAL RISK MANAGEMENT</div><div>The CET1 ratio was 15.7%.</div>"
    "<div>MARKET RISK MANAGEMENT</div><div>Average VaR was $45 million.</div>" + page(6)
    + "<div>Report of Independent Registered Public Accounting Firm</div><div>We have audited the statements.</div>"
    + page(7)
    + "<div>Consolidated statements of income</div>"
    "<table><tr><td></td><td>2025</td><td>2024</td></tr><tr><td>Total net revenue</td><td>182,447</td><td>177,556</td></tr></table>"
    "<div>Consolidated balance sheets</div>"
    "<table><tr><td></td><td>2025</td><td>2024</td></tr><tr><td>Total assets</td><td>4,424,900</td><td>4,002,814</td></tr></table>"
    + page(8)
    + "<div>Note 1 – Basis of presentation</div><div>The statements are prepared under U.S. GAAP.</div>"
    "<div>Note 2 – Litigation</div><div>The Firm is defending several matters.</div>" + page(9)
    + "<div>Signatures</div><div>Pursuant to the requirements of Section 13.</div>"
)


@pytest.fixture(scope="module")
def jpm_like():
    return analyze_filing(JPM_LIKE, FILING_ID)


def test_appended_annual_report_after_items_is_recovered(jpm_like) -> None:
    appended = [s.title for s in jpm_like.sections if s.item_number is None]
    assert appended == [
        "EXECUTIVE OVERVIEW",
        "CAPITAL RISK MANAGEMENT",
        "MARKET RISK MANAGEMENT",
        "Report of Independent Registered Public Accounting Firm",
        "Consolidated statements of income",  # statement title, not a TOC entry
        "Consolidated balance sheets",
        "Note 1 – Basis of presentation",
        "Note 2 – Litigation",
    ]
    assert jpm_like.coverage.report_toc_unmatched == ()
    text = "\n".join(s.text for s in jpm_like.sections)
    for value in ("$58.5 billion", "15.7%", "182,447", "4,424,900"):
        assert value in text


def test_running_part_header_does_not_cut_an_item_short(jpm_like) -> None:
    item15 = next(s for s in jpm_like.sections if s.item_number == "15")
    assert "Certificate of Incorporation" in item15.text
    assert "Deferred Compensation Plan" in item15.text  # after the "Part IV" page header
    assert "Table of contents" not in item15.text  # the report starts at its TOC


def test_bare_page_numbers_and_breaks_are_removed(jpm_like) -> None:
    for section in jpm_like.sections:
        assert not any(line.strip().isdigit() for line in section.text.split("\n")), section.title


def test_post_signature_text_is_never_a_section(jpm_like) -> None:
    assert all("Pursuant to the requirements" not in s.text for s in jpm_like.sections)
    assert jpm_like.coverage.words_by_category["signatures_and_after"] > 0


def test_standard_filing_has_no_appended_sections() -> None:
    parsed = analyze_filing(LINKED_FILING, FILING_ID)
    assert all(s.item_number for s in parsed.sections)
    assert parsed.coverage.report_sections == 0
    assert parsed.coverage.words_by_category["report_toc"] == 0


def test_numeric_table_after_last_item_is_not_mistaken_for_a_report_toc() -> None:
    html = wrap(
        item_toc([("1", "Business"), ("15", "Exhibits")])
        + item("1", "Business", "<div>Widgets.</div>")
        + item("15", "Exhibits", "<table>" + "".join(
            f"<tr><td>Segment {k}</td><td>{10 + k}</td></tr>" for k in range(8)) + "</table>"
            "<div>Segment 3</div><div>Commentary.</div>")
    )
    parsed = analyze_filing(html, FILING_ID)
    assert [s.item_number for s in parsed.sections] == ["1", "15"]
    assert parsed.coverage.report_sections == 0


def test_unmatched_report_entry_is_reported_not_invented() -> None:
    html = wrap(
        item_toc([("1", "Business"), ("8", "Financial Statements")])
        + item("1", "Business", "<div>Widgets.</div>")
        + item("8", "Financial Statements", "<div>See the Financial Section.</div>")
        + "<table><tr><td>Overview</td><td>10</td></tr><tr><td>Results</td><td>11</td></tr>"
        "<tr><td>Liquidity</td><td>12</td></tr><tr><td>Outlook</td><td>13</td></tr>"
        "<tr><td>Hedging Policy</td><td>14</td></tr></table>"
        "<div>OVERVIEW</div><div>One.</div><div>RESULTS</div><div>Two.</div>"
        "<div>LIQUIDITY</div><div>Three.</div><div>OUTLOOK</div><div>Four. Hedging is discussed here.</div>"
    )
    parsed = analyze_filing(html, FILING_ID)
    assert [s.title for s in parsed.sections if s.item_number is None] == ["OVERVIEW", "RESULTS", "LIQUIDITY", "OUTLOOK"]
    assert parsed.coverage.report_toc_unmatched == ("Hedging Policy",)


def test_numbered_heading_without_space_matches_its_report_entry() -> None:
    html = wrap(
        item_toc([("1", "Business"), ("8", "Financial Statements")])
        + item("1", "Business", "<div>Widgets.</div>")
        + item("8", "Financial Statements", "<div>See the Financial Section.</div>")
        + "<table><tr><td>Overview</td><td>10</td></tr><tr><td>1. Accounting Policies</td><td>11</td></tr>"
        "<tr><td>2. Leases</td><td>12</td></tr><tr><td>3. Other Comprehensive Income</td><td>13</td></tr>"
        "<tr><td>4. Debt</td><td>14</td></tr></table>"
        "<div>OVERVIEW</div><div>One.</div><div>1. Accounting Policies</div><div>Two.</div>"
        "<div>2. Leases</div><div>Three.</div><div>3.Other Comprehensive Income</div><div>Four.</div>"
        "<div>4. Debt</div><div>Five.</div>"
    )
    parsed = analyze_filing(html, FILING_ID)
    titles = [s.title for s in parsed.sections if s.item_number is None]
    assert titles == ["OVERVIEW", "1. Accounting Policies", "2. Leases", "3.Other Comprehensive Income", "4. Debt"]
    assert parsed.coverage.report_toc_unmatched == ()


def test_linked_report_entry_without_heading_uses_its_anchor() -> None:
    html = wrap(
        item_toc([("1", "Business"), ("8", "Financial Statements")])
        + item("1", "Business", "<div>Widgets.</div>")
        + item("8", "Financial Statements", "<div>See the report.</div>")
        + "<table><tr><td>Overview</td><td>10</td></tr><tr><td>Results</td><td>11</td></tr>"
        '<tr><td><a href="#fs">Consolidated Financial Statements</a></td><td>12</td></tr>'
        "<tr><td>Outlook</td><td>13</td></tr><tr><td>Glossary</td><td>14</td></tr></table>"
        "<div>OVERVIEW</div><div>One.</div><div>RESULTS</div><div>Two.</div>"
        '<div id="fs"></div><table><tr><td></td><td>2025</td></tr><tr><td>Revenue</td><td>1,000</td></tr></table>'
        "<div>OUTLOOK</div><div>Four.</div><div>GLOSSARY</div><div>Terms.</div>"
    )
    parsed = analyze_filing(html, FILING_ID)
    titles = [s.title for s in parsed.sections if s.item_number is None]
    assert titles == ["OVERVIEW", "RESULTS", "Consolidated Financial Statements", "OUTLOOK", "GLOSSARY"]
    statements = next(s for s in parsed.sections if s.title == "Consolidated Financial Statements")
    assert "Revenue | 2025: 1,000" in statements.text


def test_coverage_counts_table_cells_not_rendered_labels() -> None:
    table = (
        "<table><tr><td></td><td>Year ended 2025</td><td>Year ended 2024</td></tr>"
        "<tr><td>Net sales</td><td>$</td><td>100</td><td>$</td><td>90</td></tr>"
        "<tr><td>Costs</td><td>40</td><td>35</td></tr></table>"
    )
    html = wrap(item_toc([("1", "Business"), ("8", "Statements")])
                + item("1", "Business", "<div>Widgets.</div>") + item("8", "Statements", table))
    parsed = analyze_filing(html, FILING_ID)
    item8 = next(i for i, s in enumerate(parsed.sections) if s.item_number == "8")
    # heading (4 words) + cells: "Year ended 2025", "Year ended 2024", "Net sales", "$100", "$90",
    # "Costs", "40", "35" = 12 words; the rendered text repeats the header labels per value.
    assert parsed.coverage.section_words[item8] == 4 + 12
    assert len(parsed.sections[item8].text.split()) > 16


def test_page_number_artifact_after_boilerplate_is_removed() -> None:
    html = wrap(
        item_toc([("1", "Business"), ("1B", "Unresolved Staff Comments"), ("2", "Properties")])
        + page(1) + item("1", "Business", "<div>Widgets.</div>") + page(2)
        + item("1B", "Unresolved Staff Comments", "<div>None.</div>") + page(28)
        + item("2", "Properties", "<div>Offices.</div>") + page(29)
    )
    item1b = next(s for s in analyze_filing(html, FILING_ID).sections if s.item_number == "1B")
    assert item1b.text == "Item 1B. Unresolved Staff Comments\nNone."


def test_single_row_footer_tables_are_page_furniture() -> None:
    footer = lambda n: f"<table><tr><td>Acme Corp/2025 Form 10-K</td><td>{n}</td></tr></table>"  # noqa: E731
    html = wrap(
        item_toc([("1", "Business"), ("2", "Properties")])
        + "".join(f"{footer(n)}<hr/><div>Page {n} text continues here.</div>" for n in range(1, 5))
        + item("1", "Business", "<div>Widgets.</div>") + footer(5) + "<hr/>"
        + item("2", "Properties", "<div>Offices.</div>") + footer(6) + "<hr/>"
    )
    assert "Form 10-K" not in extract_visible_text(html)
