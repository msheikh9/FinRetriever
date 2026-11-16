"""SEC 10-K filing parser.

Turns a raw EDGAR 10-K document (HTML or inline-XBRL XHTML) into clean,
section-aware text for chunking and retrieval. This is a retrieval parser,
not an accounting parser: XBRL facts are kept only as the visible text they
wrap.

Pipeline:

1. **Clean.** Drop the hidden ``ix:header`` block, ``display:none`` elements,
   scripts/styles and decorative elements (images, rules). Inline XBRL tags
   (``ix:nonFraction``, ``ix:nonNumeric``, ...) are transparent: their
   visible text is kept in place.
2. **Flatten into blocks.** The body becomes an ordered list of paragraph
   blocks, table blocks and page-break markers. Element ``id``s are attached
   to the block they start, so table-of-contents anchors can be resolved.
3. **Strip page artifacts.** Short blocks adjacent to page breaks that repeat
   across many pages once digits are masked (``Apple Inc. | 2025 Form 10-K |
   N``, bare page numbers, "Table of Contents" back-links) are removed, and
   paragraphs split by a page break are rejoined.
4. **Find sections.** Internal TOC links whose row reads ``Item N ...`` are
   resolved to their anchors, and an anchor is accepted only if the block at
   (or just after) it is an ``Item N`` heading for the same item. If fewer
   than two items resolve, a heading-based fallback is used; if that also
   finds fewer than two, no sections are produced rather than guessed ones.
   A section runs from its heading to the next item heading, ``PART`` heading
   or ``SIGNATURES`` heading.

``extract_visible_text`` returns the joined block text, and each ``Section``
is an exact slice of it (``start_char``/``end_char``), so sections never
overlap or duplicate text.

Data tables are rendered one row per line, each value labelled with its
column header and each row with its group context
(``Cost of sales > Products | 2025: $194,116 | 2024: $185,233``), with split
``$``/``%``/``)`` cells re-attached and spacer cells dropped. They are
wrapped in ``TABLE_START``/``TABLE_END`` marker lines so table-derived text
stays distinguishable. Layout tables (bullets, single-cell boxes) become
ordinary narrative lines. Superscripted footnote references render as
``[1]`` so they cannot be mistaken for negative values like ``(1)``.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

from bs4 import BeautifulSoup, CData, NavigableString, Tag, XMLParsedAsHTMLWarning

from finretriever.data.schemas import Section, make_section_id

TABLE_START = "[TABLE]"
TABLE_END = "[/TABLE]"


class FilingParseError(ValueError):
    """No reliable Item sections could be identified in a filing."""


_TEXT_TYPES = (NavigableString, CData)
_SKIP_TAGS = frozenset(
    {"head", "script", "style", "noscript", "template", "title", "meta", "link",
     "img", "svg", "canvas", "object", "embed", "iframe", "video", "audio", "picture"}
)
_BLOCK_TAGS = frozenset(
    {"address", "article", "aside", "blockquote", "body", "center", "dd", "div", "dl",
     "dt", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
     "header", "html", "li", "main", "nav", "ol", "p", "pre", "section", "ul"}
)
_CONTAINER_TAGS = sorted(_BLOCK_TAGS | {"table", "hr"})

_DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.I)
_BREAK_BEFORE_RE = re.compile(r"(?:page-)?break-before\s*:\s*(?:always|page)", re.I)
_BREAK_AFTER_RE = re.compile(r"(?:page-)?break-after\s*:\s*(?:always|page)", re.I)

_ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)(?![a-z0-9])\s*[.:\-–—]*\s*(.*)$", re.I | re.S)
_PART_RE = re.compile(r"^part\s+(iv|i{1,3})\b", re.I)
_SIGNATURES_RE = re.compile(r"^signatures?$", re.I)
_TRAILING_PAGE_RE = re.compile(r"(?:\s+|\s*\|\s*)(?:[A-Z]{1,2}-)?\d{1,3}$")
_MAX_HEADING_CHARS = 250
_MAX_ARTIFACT_CHARS = 150

_CURRENCY_PREFIXES = frozenset({"$", "US$", "€", "£", "¥", "(", "$("})
_VALUE_SUFFIXES = frozenset({"%", ")", ")%", "%)"})
_BULLETS = frozenset({"•", "●", "◦", "▪", "‣", "⁃", "-", "*"})
_VALUE_RE = re.compile(r"^\(?[-–]?[$€£¥]?\s?\(?\d[\d,]*(?:\.\d+)?\)?\s?%?\)?$")
_DASH_VALUES = frozenset({"—", "–", "-"})
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

_CHAR_MAP = str.maketrans(
    {
        **dict.fromkeys(
            "            　\t\f\v",
            " ",
        ),
        **dict.fromkeys("​‌‍⁠﻿­", None),
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
    }
)
_HSPACE_RE = re.compile(r"[ ]+")
_ANY_SPACE_RE = re.compile(r"\s+")


# --- text normalisation ---------------------------------------------------------


def normalize_filing_text(text: str) -> str:
    """Normalise extracted filing text while keeping line structure.

    Non-breaking and other Unicode spaces become plain spaces, zero-width
    characters and soft hyphens are removed, curly quotes become ASCII
    quotes (so ``Company's`` matches either spelling), runs of spaces are
    collapsed, and blank lines are dropped. Dashes, symbols and numbers are
    left untouched. HTML entities are decoded by the HTML parser, not here.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_CHAR_MAP)
    lines = (_HSPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _normalize_line(text: str) -> str:
    return _ANY_SPACE_RE.sub(" ", text.translate(_CHAR_MAP)).strip()


# --- HTML loading and cleaning ----------------------------------------------------


def _load_soup(html: str | bytes) -> BeautifulSoup:
    # Inline-XBRL 10-Ks are XHTML with an XML declaration; the lenient HTML
    # parser is used on purpose, so silence that one warning for this call only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    hidden = [
        tag
        for tag in soup.find_all(True)
        if tag.name == "ix:header"
        or tag.has_attr("hidden")
        or _DISPLAY_NONE_RE.search(tag.get("style", ""))
    ]
    for tag in hidden:
        if not tag.decomposed:
            tag.decompose()
    return soup


def _inline_text(node: Tag, parts: list[str]) -> None:
    """Append the visible text of ``node``; nested tables are flattened inline."""
    for child in node.children:
        if isinstance(child, NavigableString):
            if type(child) in _TEXT_TYPES:
                parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name in _SKIP_TAGS:
                continue
            if child.name == "br":
                parts.append(" ")
            elif marker := _footnote_marker(child):
                parts.append(marker)
            elif child.name == "table":
                parts.append(" " + extract_table_text(child).replace("\n", "; ") + " ")
            elif child.name in _BLOCK_TAGS or child.name in ("td", "th", "tr"):
                parts.append(" ")
                _inline_text(child, parts)
                parts.append(" ")
            else:
                _inline_text(child, parts)


def _text_of(node: Tag) -> str:
    parts: list[str] = []
    _inline_text(node, parts)
    return _normalize_line("".join(parts))


# --- tables -------------------------------------------------------------------------

_FOOTNOTE_RE = re.compile(r"^(?:\((?:\d{1,2}|[a-z])\)|\*{1,3}|[†‡])+$")
_FOOTNOTE_PART_RE = re.compile(r"\((\d{1,2}|[a-z])\)|(\*{1,3}|[†‡])")
_FOOTNOTE_SUFFIX_RE = re.compile(r"(?:\s*\[(?:\d{1,2}|[a-z]|\*{1,3}|[†‡])\])+$")
_SUPER_ALIGN_RE = re.compile(r"vertical-align\s*:\s*super", re.I)
_RELATIVE_RE = re.compile(r"position\s*:\s*relative", re.I)
_RAISED_RE = re.compile(r"(?<![-\w])top\s*:\s*-\s*\d", re.I)
_TOP_BORDER_RE = re.compile(r"border-top(?:-style)?\s*:\s*[^;]*\b(?:solid|double|dashed|dotted)\b", re.I)
_PADDING_LEFT_RE = re.compile(r"padding-left\s*:\s*([^;]+)", re.I)
_PADDING_RE = re.compile(r"(?<![-\w])padding\s*:\s*([^;]+)", re.I)
_MARGIN_LEFT_RE = re.compile(r"margin-left\s*:\s*([^;]+)", re.I)
_LENGTH_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*(pt|px|em|rem|in)?$", re.I)
_LENGTH_UNITS_PT = {"pt": 1.0, "px": 0.75, "em": 12.0, "rem": 12.0, "in": 72.0, None: 0.75}
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_PERIOD_RE = re.compile(
    rf"^(?:(?:fiscal|fy)\s*)?(?:19|20)\d{{2}}$|^{_MONTHS}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}$"
    r"|^q[1-4]\b|^(?:three|six|nine|twelve) months ended\b|^years? ended\b|^change$|^% change$",
    re.I,
)
_MAX_HEADER_CELL_CHARS = 100


_FOOTNOTE_DEFINITION_RE = re.compile(r"^\((\d{1,2})\)(?=[A-Z])")


def _mark_footnote_definition(text: str) -> str:
    """``(1)Services net sales ...`` → ``[1] Services net sales ...``.

    A parenthesised number glued to a capitalised word at the start of a
    block or cell can only be a footnote definition, never a negative value.
    """
    return _FOOTNOTE_DEFINITION_RE.sub(r"[\1] ", text)


def _footnote_marker(tag: Tag) -> str | None:
    """`` [1] `` for a superscripted footnote reference like ``(1)``, else ``None``.

    Only superscript markup (``<sup>``, ``vertical-align:super``, or a raised
    ``position:relative; top:-…`` span) counts, so a parenthesised value such
    as ``(1)`` in an ordinary cell stays a negative number.
    """
    style = tag.get("style", "")
    raised = tag.name == "sup" or bool(_SUPER_ALIGN_RE.search(style)) or (
        bool(_RELATIVE_RE.search(style)) and bool(_RAISED_RE.search(style))
    )
    if not raised:
        return None
    text = _normalize_line(tag.get_text()).replace(" ", "")
    if not _FOOTNOTE_RE.match(text):
        return None
    marks = "".join(f"[{number or symbol}]" for number, symbol in _FOOTNOTE_PART_RE.findall(text))
    return f" {marks} "


@dataclass
class _Cell:
    start: int
    end: int
    text: str


@dataclass
class _Row:
    cells: list[_Cell]
    indent: float = 0.0
    top_border: bool = False
    # Cells from earlier rows whose ``rowspan`` covers this row. They give
    # context (header labels, row label) but their text belongs to their own row.
    spanning: list[_Cell] = field(default_factory=list)


@dataclass
class _Group:
    text: str
    indent: float
    children: int = 0

    @property
    def label(self) -> str:
        return self.text.rstrip(":").strip()


def _span(cell: Tag, attribute: str) -> int:
    match = re.match(r"\s*(\d+)", str(cell.get(attribute, "1")))
    return min(max(int(match.group(1)), 1), 100) if match else 1


def _colspan(cell: Tag) -> int:
    return _span(cell, "colspan")


def _is_value(text: str) -> bool:
    text = _FOOTNOTE_SUFFIX_RE.sub("", text)
    return text in _DASH_VALUES or (bool(_VALUE_RE.match(text)) and not _YEAR_RE.match(text))


def _length_pt(value: str) -> float:
    match = _LENGTH_RE.match(value.strip())
    if not match:
        return 0.0
    return float(match.group(1)) * _LENGTH_UNITS_PT[match.group(2).lower() if match.group(2) else None]


def _left_indent(style: str) -> float:
    """Left padding + margin in points; ``text-indent`` (hanging indents) is ignored."""
    indent = 0.0
    if match := _PADDING_LEFT_RE.search(style):
        indent += _length_pt(match.group(1))
    elif match := _PADDING_RE.search(style):
        parts = match.group(1).split()
        indent += _length_pt(parts[3] if len(parts) == 4 else parts[1] if len(parts) > 1 else parts[0])
    if match := _MARGIN_LEFT_RE.search(style):
        indent += _length_pt(match.group(1))
    return indent


def _table_rows(table: Tag) -> list[_Row]:
    """Rows of ``table`` (not of nested tables), with ``rowspan``-aware columns.

    A cell with ``rowspan="n"`` occupies its columns in the next ``n - 1``
    rows, so those rows' cells start further right; without this, every cell
    after a spanning header would be shifted under the wrong column.
    """
    rows: list[_Row] = []
    # column -> (rows left, spanning cell); empty spanning cells still occupy columns
    carried: dict[int, tuple[int, _Cell | None]] = {}
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        placed: list[tuple[Tag, int, int, str]] = []
        column = 0
        for td in tr.find_all(["td", "th"], recursive=False):
            while column in carried:
                column += 1
            span = _colspan(td)
            placed.append((td, column, span, _mark_footnote_definition(_text_of(td))))
            column += span
        spanning = list({id(c): c for _, c in carried.values() if c is not None}.values())
        rows.append(_build_row(placed, spanning))
        carried = {col: (left - 1, c) for col, (left, c) in carried.items() if left > 1}
        for td, start, span, text in placed:
            extra_rows = _span(td, "rowspan") - 1
            if extra_rows > 0:
                cell = _Cell(start, start + span, text) if text else None
                for col in range(start, start + span):
                    carried[col] = (extra_rows, cell)
    return rows


def _build_row(placed: list[tuple[Tag, int, int, str]], spanning: list[_Cell]) -> _Row:
    """Non-empty cells of one row with column spans, split symbols re-attached."""
    cells: list[_Cell] = []
    label_td: Tag | None = None
    for td, start, span, text in placed:
        if text:
            cells.append(_Cell(start, start + span, text))
            label_td = label_td or td

    merged: list[_Cell] = []
    prefix: _Cell | None = None
    for cell in cells:
        if cell.text in _CURRENCY_PREFIXES:
            prefix = cell if prefix is None else _Cell(prefix.start, cell.end, prefix.text + cell.text)
            continue
        if cell.text in _VALUE_SUFFIXES and merged and prefix is None:
            last = merged[-1]
            merged[-1] = _Cell(last.start, cell.end, last.text + cell.text)
            continue
        if prefix is not None:
            cell = _Cell(prefix.start, cell.end, prefix.text + cell.text)
            prefix = None
        merged.append(cell)
    if prefix is not None:
        merged.append(prefix)

    indent = 0.0
    if label_td is not None:
        indent = _left_indent(label_td.get("style", ""))
        inner = label_td.find(["div", "p"])
        if inner is not None:
            indent += _left_indent(inner.get("style", ""))
    top_border = any(_TOP_BORDER_RE.search(td.get("style", "")) for td, *_ in placed)
    return _Row(merged, indent, top_border, sorted(spanning, key=lambda c: c.start))


def _is_group_row(cells: list[_Cell], label_start: int) -> bool:
    """A lone label ending in ``:`` (``Net sales:``) that introduces child rows."""
    return (
        len(cells) == 1
        and cells[0].start <= label_start
        and cells[0].text.endswith(":")
        and not _is_value(cells[0].text)
    )


def _header_row_count(rows: list[_Row], label_start: int) -> int:
    """Number of leading non-blank rows that are column headers.

    A header row has no values and is either indented past the row-label
    column (``| 2025 | 2024``) or carries a caption in that column followed by
    at least two column headings, or by period headings only (years, dates).
    Only one captioned row is accepted, and it ends the header: the rows
    after it may be text-only data rows (``Common Stock | AAPL | Nasdaq``).
    """
    count = 0
    for row in rows:
        cells = row.cells
        if any(_is_value(c.text) for c in cells) or _is_group_row(cells, label_start):
            break
        if cells[0].start > label_start:
            count += 1
            continue
        rest = cells[1:]
        if (
            rest
            and all(len(c.text) <= _MAX_HEADER_CELL_CHARS for c in rest)
            and (len(rest) >= 2 or all(_PERIOD_RE.match(c.text) for c in rest))
        ):
            count += 1
        break
    return 0 if count == len(rows) else count


def _header_labels(cells: list[_Cell], header_rows: list[list[_Cell]]) -> list[str]:
    """Column label for each cell, or ``""`` where alignment is not unambiguous.

    A header cell labels a body cell only if it is the single header cell in
    its row overlapping that cell's column span and the cell lies entirely
    within it (so a value shifted across a spacer column is not attributed to
    the neighbouring period). In the lowest header row, a header shared by
    two cells of the same body row, or a cell under no header, leaves that
    cell unlabelled rather than half-labelled.
    """
    labels: list[list[str] | None] = [[] for _ in cells]
    for depth, header in enumerate(header_rows):
        matches: list[_Cell | None] = []
        for cell in cells:
            overlapping = [h for h in header if h.start < cell.end and cell.start < h.end]
            contained = (
                len(overlapping) == 1
                and overlapping[0].start <= cell.start
                and cell.end <= overlapping[0].end
            )
            matches.append(overlapping[0] if contained else None)
        if depth == len(header_rows) - 1:
            uses = Counter(id(m) for m in matches if m is not None)
            for i, match in enumerate(matches):
                if match is None or uses[id(match)] > 1:
                    labels[i] = None
        for i, match in enumerate(matches):
            current = labels[i]
            # A rowspan header appears in several header rows; name it once.
            if match is not None and current is not None and current[-1:] != [match.text]:
                current.append(match.text)
    return [" ".join(label) if label else "" for label in labels]


def _value_cells(row: _Row, label_start: int) -> tuple[_Cell | None, list[_Cell]]:
    """Split a row into its label cell (if any) and the remaining cells.

    A row whose label column is covered by a ``rowspan`` label from above
    takes that label, since the HTML states that it applies to this row too.
    """
    first = row.cells[0]
    if first.start <= label_start and not _is_value(first.text):
        return first, row.cells[1:]
    carried = next((c for c in row.spanning if c.start <= label_start < c.end), None)
    if carried is not None and not _is_value(carried.text):
        return carried, row.cells
    return None, row.cells


def _alignment_is_reliable(body: list[_Row], header_rows: list[list[_Cell]], label_start: int) -> bool:
    total = labelled = 0
    for row in body:
        if not row.cells or _is_group_row(row.cells, label_start):
            continue
        _, cells = _value_cells(row, label_start)
        labels = _header_labels(cells, header_rows)
        total += len(labels)
        labelled += sum(bool(label) for label in labels)
    return total == 0 or labelled * 2 >= total


def _render_body(body: list[_Row], header_rows: list[list[_Cell]], label_start: int) -> list[str]:
    """Render body rows, carrying ``Group > Child`` context from group rows.

    Context rules (deliberately conservative):

    * A group row (``Net sales:``) nests inside the current group only if its
      label is indented further; otherwise it replaces it. Without
      indentation in the HTML, groups never nest.
    * A closing row — label starting with ``Total``/``Subtotal``, or a row
      with a top border (the accounting rule line) after at least one child —
      closes the innermost group whose name its label contains (or the
      innermost group), and is labelled with the remaining outer groups only.
    * A row with an empty label column inside a group is labelled with the
      group path (``Lease liabilities | ...``); outside a group it is not.
    * After a blank spacer row, outer groups survive only if the next row is
      a more deeply indented group; any other row starts with no context.
    * A label-only row without a colon resets the context.
    """
    data_rows = [r for r in body if r.cells and not _is_group_row(r.cells, label_start)]
    use_borders = sum(r.top_border for r in data_rows) <= 0.8 * len(data_rows)

    out: list[str | _Group] = []
    stack: list[_Group] = []
    stale = False
    for row in body:
        if not row.cells:
            stale = bool(stack)
            continue
        if _is_group_row(row.cells, label_start):
            group = _Group(row.cells[0].text, row.indent)
            while stack and stack[-1].indent >= group.indent:
                stack.pop()
            stack.append(group)
            stale = False
            out.append(group)
            continue

        label, cells = _value_cells(row, label_start)
        if stale:
            stack, stale = [], False
        if label is not None and not cells:
            stack = []
            out.append(label.text)
            continue
        if label is not None:
            name = label.text.lower()
            closing = name.startswith(("total", "subtotal")) or (
                use_borders and row.top_border and bool(stack) and stack[-1].children > 0
            )
            if closing and stack:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i].label.lower() in name:
                        del stack[i:]
                        break
                else:
                    stack.pop()
        # An unlabelled row inside a group (a continuation line) is labelled
        # with the group path itself; outside a group it stays unlabelled.
        path = [g.label for g in stack] + ([label.text] if label is not None else [])
        for group in stack:
            group.children += 1
        labels = _header_labels(cells, header_rows) if header_rows else [""] * len(cells)
        parts = [f"{lab}: {cell.text}" if lab else cell.text for lab, cell in zip(labels, cells, strict=True)]
        if path:
            parts.insert(0, " > ".join(path))
        out.append(" | ".join(parts))
    return [item if isinstance(item, str) else item.text for item in out if isinstance(item, str) or not item.children]


class _RenderedTable(NamedTuple):
    text: str
    is_data: bool
    source_words: int  # words in the table's own cells, excluding added labels


def _render_table(table: Tag) -> _RenderedTable:
    """Render ``table`` as text; ``is_data`` says whether it is a data table."""
    all_rows = _table_rows(table)
    rows = [r for r in all_rows if r.cells]
    if not rows:
        return _RenderedTable("", False, 0)

    cell_rows = [r.cells for r in rows]
    source_words = sum(len(c.text.split()) for cells in cell_rows for c in cells)
    has_values = any(_is_value(c.text) for cells in cell_rows for c in cells)
    if all(len(cells) == 1 for cells in cell_rows) or all(
        len(cells) <= 2 and cells[0].text in _BULLETS for cells in cell_rows
    ) or (not has_values and len(rows) == 1):
        text = "\n".join(" ".join(c.text for c in cells) for cells in cell_rows)
        return _RenderedTable(text, False, source_words)
    if not has_values:
        text = "\n".join(" | ".join(c.text for c in cells) for cells in cell_rows)
        return _RenderedTable(text, True, source_words)

    label_start = min(cells[0].start for cells in cell_rows)
    header_count = _header_row_count(rows, label_start)
    header_rows = rows[:header_count]
    # Rowspan cells from higher header rows also head the columns below them.
    label_rows = [sorted(r.cells + r.spanning, key=lambda c: c.start) for r in header_rows]
    body = all_rows
    if header_count:
        last_header = next(i for i, r in enumerate(all_rows) if r is header_rows[-1])
        body = all_rows[last_header + 1:]
        if not _alignment_is_reliable(body, label_rows, label_start):
            header_rows, label_rows, body = [], [], all_rows

    # Header rows with a caption in the row-label column ("Periods | ...") are
    # kept as a line, since that caption is not attached to any value.
    lines = [" | ".join(c.text for c in h.cells) for h in header_rows if h.cells[0].start <= label_start]
    lines += _render_body(body, label_rows, label_start)
    return _RenderedTable("\n".join(lines), True, source_words)


def extract_table_text(table_element: Tag) -> str:
    """Render one ``<table>`` as retrieval-friendly text, one row per line.

    * Empty spacer cells are dropped and split ``$`` / ``%`` / ``)`` cells are
      re-attached (``$`` + ``416,161`` → ``$416,161``; ``(4)`` + ``%`` →
      ``(4)%``).
    * Column headers (including multi-row and ``colspan``-spanning ones) are
      prefixed to the values they sit over, when alignment is unambiguous:
      ``Total net sales | 2025: $416,161 | Change: 6%``. Ambiguous cells keep
      their plain value; if most values cannot be aligned, the header rows
      are kept as ordinary rows instead.
    * Group rows such as ``Net sales:`` are carried into their child rows as
      ``Net sales > Products | ...`` (see ``_render_body`` for the rules).
    * Superscripted footnote references render as ``[1]``; parenthesised
      values like ``(56)`` stay negative numbers.

    Cells are separated by `` | `` so values never run into one another.
    """
    return _render_table(table_element).text


# --- block model ----------------------------------------------------------------------


@dataclass
class _Block:
    kind: str  # "text", "table" or "break"
    text: str = ""
    ids: list[str] = field(default_factory=list)
    # Words of filing text this block stands for (table cells, not rendered
    # labels), used to measure section coverage of the original document.
    source_words: int = 0

    @property
    def plain(self) -> str:
        if self.kind != "table":
            return self.text
        body = self.text.removeprefix(TABLE_START).removesuffix(TABLE_END)
        return _normalize_line(body.replace(" | ", " "))


class _Flattener:
    def __init__(self) -> None:
        self.blocks: list[_Block] = []
        self._buffer: list[str] = []
        self._pending_ids: list[str] = []

    def run(self, root: Tag) -> list[_Block]:
        self._walk(root)
        self._flush()
        return self.blocks

    def _emit(self, kind: str, text: str, source_words: int) -> None:
        self.blocks.append(_Block(kind, text, self._pending_ids, source_words))
        self._pending_ids = []

    def _flush(self) -> None:
        text = _mark_footnote_definition(_normalize_line("".join(self._buffer)))
        self._buffer.clear()
        if text:
            self._emit("text", text, len(text.split()))

    def _page_break(self) -> None:
        self._flush()
        self.blocks.append(_Block("break"))

    def _walk(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if type(child) in _TEXT_TYPES:
                    self._buffer.append(str(child))
                continue
            if not isinstance(child, Tag) or child.name in _SKIP_TAGS:
                continue
            if child.get("id"):
                self._pending_ids.append(child["id"])
            if child.name == "br":
                self._buffer.append(" ")
                continue
            if child.name == "hr":
                self._page_break()
                continue
            style = child.get("style", "")
            if _BREAK_BEFORE_RE.search(style):
                self._page_break()
            if child.name == "table":
                self._flush()
                self._pending_ids.extend(t["id"] for t in child.find_all(id=True))
                rendered = _render_table(child)
                if rendered.is_data:
                    self._emit("table", f"{TABLE_START}\n{rendered.text}\n{TABLE_END}", rendered.source_words)
                elif rendered.text:
                    self._emit("text", rendered.text, rendered.source_words)
            elif child.name in _BLOCK_TAGS or child.find(_CONTAINER_TAGS) is not None:
                self._flush()
                self._walk(child)
                self._flush()
            else:
                self._pending_ids.extend(t["id"] for t in child.find_all(id=True))
                if marker := _footnote_marker(child):
                    self._buffer.append(marker)
                else:
                    parts: list[str] = []
                    _inline_text(child, parts)
                    self._buffer.append("".join(parts))
            if _BREAK_AFTER_RE.search(style):
                self._page_break()


_PAGE_NUMBER_SIGNATURE_RE = re.compile(r"^(?:page )?#$")


def _artifact_signature(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def _artifact_candidate(block: _Block) -> str | None:
    """Signature of a block that could be page furniture: short text, or a one-row table."""
    if block.kind == "text":
        text = block.text
    elif block.kind == "table" and block.text.count("\n") == 2:  # [TABLE] / one row / [/TABLE]
        text = block.plain  # e.g. a footer laid out as "Company/2025 Form 10-K | 327"
    else:
        return None
    return _artifact_signature(text) if len(text) <= _MAX_ARTIFACT_CHARS else None


def _strip_page_artifacts(blocks: list[_Block]) -> list[_Block]:
    """Remove repeated page headers/footers and break markers; rejoin split paragraphs.

    A short block next to a page break is page furniture if its digit-masked
    text recurs at one break in five (``Apple Inc. | 2025 Form 10-K | 12``),
    or if it is a bare page number recurring at three or more breaks.
    """
    breaks = [i for i, b in enumerate(blocks) if b.kind == "break"]
    artifacts: set[str] = set()
    if len(breaks) >= 3:
        counts: Counter[str] = Counter()
        for i in breaks:
            neighbours = set()
            for step in (-1, 1):
                j = i + step
                while 0 <= j < len(blocks) and blocks[j].kind == "break":
                    j += step
                if 0 <= j < len(blocks) and (signature := _artifact_candidate(blocks[j])):
                    neighbours.add(signature)
            counts.update(neighbours)
        threshold = max(3, len(breaks) // 5)
        artifacts = {
            sig
            for sig, n in counts.items()
            if n >= threshold or (n >= 3 and _PAGE_NUMBER_SIGNATURE_RE.match(sig))
        }

    kept: list[_Block] = []
    carried_ids: list[str] = []
    removed_since_last = False
    for block in blocks:
        is_artifact = block.kind == "break" or _artifact_candidate(block) in artifacts
        if is_artifact:
            carried_ids.extend(block.ids)
            removed_since_last = True
            continue
        block = _Block(block.kind, block.text, carried_ids + block.ids, block.source_words)
        carried_ids = []
        previous = kept[-1] if kept else None
        if (
            removed_since_last
            and previous is not None
            and previous.kind == block.kind == "text"
            and previous.text[-1] not in '.:;!?")'
            and block.text[0].islower()
        ):
            kept[-1] = _Block(
                "text",
                f"{previous.text} {block.text}",
                previous.ids + block.ids,
                previous.source_words + block.source_words,
            )
        else:
            kept.append(block)
        removed_since_last = False
    return kept


@dataclass
class _Document:
    soup: BeautifulSoup
    blocks: list[_Block]
    offsets: list[int]
    text: str


def _build_document(html: str | bytes) -> _Document:
    soup = _load_soup(html)
    root = soup.body or soup
    blocks = [b for b in _strip_page_artifacts(_Flattener().run(root)) if b.text]
    offsets: list[int] = []
    position = 0
    for block in blocks:
        offsets.append(position)
        position += len(block.text) + 1
    return _Document(soup, blocks, offsets, "\n".join(b.text for b in blocks))


def extract_visible_text(html: str | bytes) -> str:
    """Return the filing's visible text, one block per line.

    Hidden XBRL metadata, ``display:none`` content, page headers/footers and
    page breaks are removed; inline XBRL facts keep their visible values;
    data tables appear between ``TABLE_START``/``TABLE_END`` lines.
    ``Section.start_char``/``end_char`` index into this string.
    """
    return _build_document(html).text


# --- section detection ----------------------------------------------------------------


@dataclass
class _Heading:
    block: int
    item: str
    title: str


def _item_rank(item: str) -> int:
    match = re.match(r"(\d+)([A-C]?)", item)
    assert match is not None
    return int(match.group(1)) * 10 + (ord(match.group(2)) - 64 if match.group(2) else 0)


def _clean_title(raw: str) -> str:
    title = _normalize_line(raw.replace("|", " "))
    title = _TRAILING_PAGE_RE.sub("", title)
    return title.strip(" .:-–—")


def _parse_heading(text: str) -> tuple[str, str] | None:
    """``(item, title)`` if ``text`` reads as an Item heading, not a cross-reference."""
    if len(text) > _MAX_HEADING_CHARS:
        return None
    match = _ITEM_RE.match(text)
    if not match:
        return None
    rest = match.group(2).strip()
    if rest and not (rest[0].isupper() or rest[0] in "[(\"'"):
        return None  # e.g. "Item 1A of this Form 10-K ..."
    return match.group(1).upper(), _clean_title(rest)


def _verify_anchor(blocks: list[_Block], start: int, item: str) -> tuple[int, str] | None:
    """``(block index, title)`` of the ``item`` heading at or just after ``start``."""
    for index in range(start, min(start + 3, len(blocks))):
        if blocks[index].kind != "text":
            continue
        heading = _parse_heading(blocks[index].plain)
        if heading is not None and heading[0] == item:
            return index, heading[1]
    return None


def _toc_link_text(link: Tag) -> str:
    own = _text_of(link)
    if _ITEM_RE.match(own):
        return own
    row = link.find_parent("tr")
    if row is not None:
        return _normalize_line(" ".join(_text_of(c) for c in row.find_all(["td", "th"])))
    block = link.find_parent(sorted(_BLOCK_TAGS - {"body", "html"}))
    text = _text_of(block) if block is not None else ""
    return text if len(text) <= _MAX_HEADING_CHARS else ""


def _toc_headings(doc: _Document) -> list[_Heading]:
    """Item headings reached through internal TOC links and verified at the anchor."""
    id_to_block: dict[str, int] = {}
    for index, block in enumerate(doc.blocks):
        for element_id in block.ids:
            id_to_block.setdefault(element_id, index)

    entries: dict[str, tuple[str, str]] = {}
    for link in doc.soup.find_all("a", href=True):
        href = link["href"].strip()
        target = href[1:]
        if not href.startswith("#") or target not in id_to_block or target in entries:
            continue
        match = _ITEM_RE.match(_toc_link_text(link))
        if match:
            entries[target] = (match.group(1).upper(), _clean_title(match.group(2)))

    headings: dict[str, _Heading] = {}
    for target, (item, toc_title) in entries.items():
        verified = _verify_anchor(doc.blocks, id_to_block[target], item)
        if verified is None:
            continue
        index, body_title = verified
        heading = _Heading(index, item, body_title or toc_title)
        if item not in headings or index < headings[item].block:
            headings[item] = heading
    return sorted(headings.values(), key=lambda h: h.block)


def _longest_increasing(headings: list[_Heading]) -> list[_Heading]:
    """Longest subsequence (in document order) with strictly increasing item rank."""
    if not headings:
        return []
    best = [1] * len(headings)
    previous = [-1] * len(headings)
    for i, current in enumerate(headings):
        for j in range(i):
            if _item_rank(headings[j].item) < _item_rank(current.item) and best[j] + 1 > best[i]:
                best[i], previous[i] = best[j] + 1, j
    i = max(range(len(headings)), key=lambda k: (best[k], k))
    chain: list[_Heading] = []
    while i != -1:
        chain.append(headings[i])
        i = previous[i]
    return chain[::-1]


def _fallback_headings(doc: _Document) -> list[_Heading]:
    """Item headings found from the text alone, when no linked TOC resolves."""
    candidates: list[_Heading] = []
    for index, block in enumerate(doc.blocks):
        if block.kind != "text" or _TRAILING_PAGE_RE.search(block.plain):
            continue  # tables, or TOC lines ending in a page number
        heading = _parse_heading(block.plain)
        if heading is not None:
            candidates.append(_Heading(index, *heading))

    # Three or more headings in consecutive blocks are an unlinked TOC.
    keep: list[_Heading] = []
    run: list[_Heading] = []
    for heading in candidates + [_Heading(-2, "", "")]:
        if run and heading.block != run[-1].block + 1:
            if len(run) < 3:
                keep.extend(run)
            run = []
        run.append(heading)

    last_per_item: dict[str, _Heading] = {}
    for heading in keep:
        last_per_item[heading.item] = heading
    ordered = sorted(last_per_item.values(), key=lambda h: h.block)
    return _longest_increasing(ordered)


def _fill_toc_gaps(toc: list[_Heading], fallback: list[_Heading]) -> list[_Heading]:
    """Add text-found headings for Items the TOC missed, where they fit its order.

    Without this, an Item whose TOC anchor is missing or wrong would be
    silently absorbed into the previous section.
    """
    covered = {h.item for h in toc}
    taken = {h.block for h in toc}
    result = list(toc)
    for extra in fallback:
        if extra.item in covered or extra.block in taken:
            continue
        before = [h for h in toc if h.block < extra.block]
        after = [h for h in toc if h.block > extra.block]
        if before and _item_rank(before[-1].item) >= _item_rank(extra.item):
            continue
        if after and _item_rank(after[0].item) <= _item_rank(extra.item):
            continue
        result.append(extra)
        covered.add(extra.item)
    return sorted(result, key=lambda h: h.block)


# --- appended report sections --------------------------------------------------------
#
# Some 10-Ks keep the Items as short cross-references and put the substance in
# a report appended after the last Item, introduced by that report's own table
# of contents: JPMorgan's annual report follows Item 15, ExxonMobil's
# "Financial Section" follows Item 16. The report TOC is used to find the
# report's real headings, which become sections without an Item number.

_REPORT_PAGE_RE = re.compile(r"^(?:[^|:]{1,20}:\s*)?(\d{1,3})$")
_EXHIBIT_INDEX_RE = re.compile(r"^(?:index to exhibits|exhibit index)$", re.I)
# A body heading may put one of these in front of its TOC entry, e.g.
# "CONSOLIDATED STATEMENT OF INCOME" for "Statement of Income".
_HEADING_PREFIXES = ("", "consolidated ", "note ", "notes to ")
_MIN_REPORT_PAGES = 5
_MAX_REPORT_HEADING_CHARS = 150


@dataclass(frozen=True)
class _ReportEntry:
    title: str
    page: int


@dataclass
class _AppendedReport:
    start: int  # first block of the report: its TOC, or the short heading above it
    # (block, title, anchored): anchored entries start at a TOC link target that
    # is itself content (e.g. a statement table), not at a heading line.
    headings: list[tuple[int, str, bool]]
    exhibit_indexes: list[int]
    entries: int
    unmatched: list[str]


def _heading_key(text: str) -> str:
    text = re.sub(r"\s*[\u2013\u2014-]+\s*", " - ", text.casefold())
    text = re.sub(r"^(\d{1,3}\.)(?=[^\d\s])", r"\1 ", text)  # "4.Other" -> "4. other"
    return re.sub(r"\s+", " ", text).strip(" .:")


def _is_heading_like(block: _Block, max_chars: int = _MAX_REPORT_HEADING_CHARS) -> bool:
    text = block.plain
    return (
        block.kind == "text"
        and 0 < len(text) <= max_chars
        and not text.endswith(".")
        and any(c.isalpha() for c in text)
    )


def _report_toc_entries(block: _Block) -> list[_ReportEntry]:
    """``(title, page)`` pairs of a table-of-contents table, in page order.

    Handles both ``Title | 29`` rows and multi-column ``50 | Title | 168 |
    Title`` layouts; cells ending in ``:`` are group labels, not entries.
    Returns ``[]`` unless page numbers rise down each column, as they do in
    a contents table and rarely do in a table of financial values.
    """
    rows = [line.split(" | ") for line in block.text.split("\n")[1:-1]]

    def page_of(cell: str) -> int | None:
        match = _REPORT_PAGE_RE.match(cell.strip())
        return int(match.group(1)) if match else None

    page_first = sum(len(r) >= 2 and page_of(r[0]) is not None for r in rows)
    page_last = sum(len(r) >= 2 and page_of(r[-1]) is not None for r in rows)
    step = 1 if page_first > page_last else -1
    entries = []
    columns: dict[int, list[int]] = {}
    for row in rows:
        for k, cell in enumerate(row):
            page, j = page_of(cell), k + step
            if page is None or not 0 <= j < len(row):
                continue
            title = _normalize_line(row[j])
            if title and page_of(title) is None and not title.endswith(":"):
                entries.append(_ReportEntry(title, page))
                columns.setdefault(k, []).append(page)
    steps = [b >= a for pages in columns.values() for a, b in zip(pages, pages[1:], strict=False)]
    if not steps or sum(steps) < 0.8 * len(steps):
        return []
    return sorted(entries, key=lambda e: e.page)


def _align_report(
    entries: list[_ReportEntry], by_key: dict[str, list[int]], first: int
) -> list[tuple[int, int]]:
    """Longest in-order matching of TOC entries to heading blocks after block ``first``.

    ``by_key`` maps heading keys to block indices. Returns ``(block, entry
    index)`` pairs increasing in both, so a missing or repeated heading
    cannot pull later entries out of order.
    """
    pairs = sorted(
        (block, e_index)
        for e_index, entry in enumerate(entries)
        for prefix in _HEADING_PREFIXES
        for block in by_key.get(prefix + _heading_key(entry.title), [])
        if block > first
    )
    if not pairs:
        return []
    best, back = [1] * len(pairs), [-1] * len(pairs)
    for i, (c_i, e_i) in enumerate(pairs):
        for j in range(i):
            c_j, e_j = pairs[j]
            if c_j < c_i and e_j < e_i and best[j] + 1 > best[i]:
                best[i], back[i] = best[j] + 1, j
    i = max(range(len(pairs)), key=lambda k: (best[k], -k))
    chain = []
    while i != -1:
        chain.append(pairs[i])
        i = back[i]
    return chain[::-1]


_LINK_PAGES_RE = re.compile(r"(?:\s*\|)?\s+(?:[A-Z]{1,2}-)?\d{1,3}(?:\s*[-–]\s*\d{1,3})?$")
_TRAILING_PARENTHETICAL_RE = re.compile(r"\s*\([^()]*\)$")
# Financial-statement structure that a report TOC often lists only as one
# entry ("Notes to Consolidated Financial Statements | 177").
_NOTE_HEADING_RE = re.compile(r"^note\s+(\d{1,3})\s*[-–—.:]\s*\S", re.I)
_STATEMENT_HEADING_RE = re.compile(r"^consolidated (?:statements? of\b.*|balance sheets?)$", re.I)
_AUDITOR_REPORT_RE = re.compile(r"^report of independent registered public accounting firm", re.I)


def _statement_headings(doc: _Document, start: int, stop: int) -> list[tuple[int, str]]:
    """Note and financial-statement headings in a report, first occurrence of each.

    Notes (``Note 12 – Loans``) must appear in increasing number order;
    statement titles (``Consolidated balance sheets``) count only after the
    auditor's report, where the statements are.
    """
    notes: list[tuple[int, int, str]] = []
    statements: dict[str, tuple[int, str]] = {}
    auditor_seen = False
    for i in range(start, stop):
        block = doc.blocks[i]
        if not _is_heading_like(block):
            continue
        text = block.plain
        if _AUDITOR_REPORT_RE.match(text):
            auditor_seen = True
        elif auditor_seen and len(text) <= 80 and _STATEMENT_HEADING_RE.match(text):
            statements.setdefault(_heading_key(text), (i, text))
        elif match := _NOTE_HEADING_RE.match(text):
            number = int(match.group(1))
            if not notes or number > notes[-1][0]:
                notes.append((number, i, text))
    return sorted([(i, text) for _, i, text in notes] + list(statements.values()))


def _report_link_targets(doc: _Document, after: int, stop: int) -> dict[str, int]:
    """Heading keys of internal links (page numbers stripped) -> target block, within the report."""
    id_to_block: dict[str, int] = {}
    for index, block in enumerate(doc.blocks):
        for element_id in block.ids:
            id_to_block.setdefault(element_id, index)
    targets: dict[str, int] = {}
    for link in doc.soup.find_all("a", href=True):
        href = link["href"].strip()
        target = id_to_block.get(href[1:]) if href.startswith("#") else None
        if target is not None and after < target < stop:
            title = _LINK_PAGES_RE.sub("", _toc_link_text(link))
            targets.setdefault(_heading_key(title), target)
    return targets


def _find_appended_report(doc: _Document, after: int, stop: int) -> _AppendedReport | None:
    """The report appended after block ``after`` (the last Item heading), if any.

    It is found through the first run of consecutive tables between ``after``
    and ``stop`` that together list at least five distinct pages and whose
    entries match, in order, at least half of themselves (and at least
    three) to later heading blocks. Headings match an entry exactly, with
    one of ``_HEADING_PREFIXES`` in front, or with a trailing parenthetical
    such as ``(unaudited)``; a heading is a short text block or the
    upper-case first cell of a table. An entry with no matching heading still gets a
    boundary if the report TOC links it to a block between its matched
    neighbours (e.g. statements that open with a table, not a heading);
    otherwise it is listed in ``unmatched``.
    """
    by_key: dict[str, list[int]] = {}
    titles: dict[int, str] = {}
    for i in range(after + 1, stop):
        block = doc.blocks[i]
        if _is_heading_like(block):
            titles[i] = block.plain
        elif block.kind == "table":
            # Some reports open a subsection with a table whose first cell is
            # the heading: "REVIEW OF 2019 AND 2018 RESULTS | 2019 | 2018".
            first_cell = block.text.split("\n")[1].split(" | ")[0].strip()
            if 0 < len(first_cell) <= _MAX_REPORT_HEADING_CHARS and first_cell.isupper():
                titles[i] = first_cell
        if i in titles:
            key = _heading_key(titles[i])
            for variant in {key, _TRAILING_PARENTHETICAL_RE.sub("", key)}:
                by_key.setdefault(variant, []).append(i)
    toc = after + 1
    while toc < stop:
        if doc.blocks[toc].kind != "table" or not _report_toc_entries(doc.blocks[toc]):
            toc += 1
            continue
        last = toc  # a contents list may continue in the next table(s)
        while last + 1 < stop and doc.blocks[last + 1].kind == "table" and _report_toc_entries(doc.blocks[last + 1]):
            last += 1
        entries = sorted(
            (e for i in range(toc, last + 1) for e in _report_toc_entries(doc.blocks[i])),
            key=lambda e: e.page,
        )
        chain = _align_report(entries, by_key, last) if len({e.page for e in entries}) >= _MIN_REPORT_PAGES else []
        if len(chain) < max(3, (len(entries) + 1) // 2):
            toc = last + 1
            continue
        matched = dict((e_index, block) for block, e_index in chain)
        links = _report_link_targets(doc, last, stop)
        headings = [(block, titles[block], False) for block, _ in chain]
        unmatched = []
        for e_index, entry in enumerate(entries):
            if e_index in matched:
                continue
            before = max((b for k, b in matched.items() if k < e_index), default=last)
            after_next = min((b for k, b in matched.items() if k > e_index), default=stop)
            anchor = links.get(_heading_key(entry.title))
            if anchor is not None and before < anchor < after_next:
                headings.append((anchor, entry.title, True))
            else:
                unmatched.append(entry.title)
        taken = {block for block, *_ in headings}
        structural = [h for h in _statement_headings(doc, min(taken), stop) if h[0] not in taken]
        if structural:
            headings += [(block, title, False) for block, title in structural]
            # Entries such as "Notes to Consolidated Financial Statements" are now
            # resolved by the individual note / statement headings.
            unmatched = [t for t in unmatched if not re.search(r"financial statements", t, re.I)]
        headings.sort()
        above = doc.blocks[toc - 1]
        start = toc - 1 if toc - 1 > after and _is_heading_like(above, 60) else toc
        exhibit_indexes = [
            i
            for i in range(headings[0][0], stop)
            if doc.blocks[i].kind == "text" and _EXHIBIT_INDEX_RE.match(doc.blocks[i].plain)
        ]
        return _AppendedReport(start, headings, exhibit_indexes, len(entries), unmatched)
    return None


# --- section assembly -----------------------------------------------------------------

COVERAGE_CATEGORIES = (
    "sections",
    "front_matter",
    "report_toc",
    "exhibit_index",
    "signatures_and_after",
    "empty_headings",
    "unassigned",
)


@dataclass(frozen=True)
class FilingCoverage:
    """Where a filing's visible source text ended up.

    Counts are whitespace words of the filing's own text after page
    furniture is removed; table blocks count their cell text, never the
    column labels or group paths added when rendering them. Every block falls
    in exactly one of ``COVERAGE_CATEGORIES``: ``sections`` (text in returned
    sections), ``front_matter`` (cover page and TOC before the first
    section), ``report_toc`` (an appended report's table of contents),
    ``exhibit_index``, ``signatures_and_after``, ``empty_headings`` (headings
    with no content) and ``unassigned`` (anything else — the part to
    investigate when coverage is low).
    """

    source_words: int
    words_by_category: dict[str, int]
    section_words: tuple[int, ...]  # per returned section, in order
    report_sections: int = 0
    report_toc_entries: int = 0
    report_toc_unmatched: tuple[str, ...] = ()

    @property
    def coverage(self) -> float:
        return self.words_by_category.get("sections", 0) / self.source_words if self.source_words else 0.0


@dataclass(frozen=True)
class ParsedFiling:
    """Sections of one filing plus an account of the text they cover."""

    sections: list[Section]
    coverage: FilingCoverage


def _part_boundaries(doc: _Document, heading_blocks: set[int]) -> set[int]:
    """``PART II``-style headings that directly precede an Item heading.

    Requiring the Item heading keeps running page headers (a ``Part IV`` at
    the top of every exhibit-list page) from cutting a section short.
    """
    return {
        index
        for index, block in enumerate(doc.blocks)
        if block.kind == "text"
        and len(block.text) <= 60
        and _PART_RE.match(block.text)
        and (index + 1 in heading_blocks or index + 2 in heading_blocks)
    }


def analyze_filing(html: str | bytes, filing_id: str) -> ParsedFiling:
    """Extract sections and account for the filing text they cover.

    Item sections come from linked-TOC headings (gaps filled from body
    headings that fit the TOC's order) or, failing that, the heading
    fallback; fewer than two Items yields no sections. Each runs from its
    heading to the next Item, ``PART`` or ``SIGNATURES`` heading. A report
    appended after the last Item (see ``_find_appended_report``) adds one
    section per matched report heading, titled with that heading and with
    ``item_number=None``; it ends the last Item, and an exhibit index inside
    it is excluded. Sections whose heading has no content are skipped.
    ``section_id`` is ``make_section_id(filing_id, title, index)``.
    """
    doc = _build_document(html)
    n = len(doc.blocks)
    headings = _toc_headings(doc)
    fallback = _fallback_headings(doc)
    headings = _fill_toc_gaps(headings, fallback) if len(headings) >= 2 else fallback
    category = ["unassigned"] * n

    def coverage(sections_words: tuple[int, ...] = (), report: _AppendedReport | None = None) -> FilingCoverage:
        words = Counter(dict.fromkeys(COVERAGE_CATEGORIES, 0))
        for block, name in zip(doc.blocks, category, strict=True):
            words[name] += block.source_words
        return FilingCoverage(
            source_words=sum(b.source_words for b in doc.blocks),
            words_by_category=dict(words),
            section_words=sections_words,
            report_sections=len(report.headings) if report else 0,
            report_toc_entries=report.entries if report else 0,
            report_toc_unmatched=tuple(report.unmatched) if report else (),
        )

    if len(headings) < 2:
        return ParsedFiling([], coverage())

    heading_blocks = {h.block for h in headings}
    boundaries = heading_blocks | _part_boundaries(doc, heading_blocks)
    last_start = headings[-1].block
    signatures = next(
        (i for i in range(last_start + 1, n) if doc.blocks[i].kind == "text" and _SIGNATURES_RE.match(doc.blocks[i].text)),
        None,
    )
    if signatures is not None:
        boundaries.add(signatures)
    report = _find_appended_report(doc, last_start, n if signatures is None else signatures)
    if report is not None:
        boundaries |= {report.start, *(block for block, *_ in report.headings), *report.exhibit_indexes}
    ordered_boundaries = sorted(boundaries) + [n]

    spans: list[tuple[int, str, str | None, bool]] = [
        (h.block, h.title or f"Item {h.item}", h.item, False) for h in headings
    ]
    if report is not None:
        spans += [(block, _normalize_line(text), None, anchored) for block, text, anchored in report.headings]

    sections: list[Section] = []
    section_words: list[int] = []
    for start, title, item, anchored in sorted(spans):
        end = next(b for b in ordered_boundaries if b > start)
        if end - start < (1 if anchored else 2):
            category[start] = "empty_headings"
            continue
        start_char = doc.offsets[start]
        end_char = doc.offsets[end - 1] + len(doc.blocks[end - 1].text)
        sections.append(
            Section(
                section_id=make_section_id(filing_id, title, len(sections)),
                filing_id=filing_id,
                title=title,
                item_number=item,
                text=doc.text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
        )
        category[start:end] = ["sections"] * (end - start)
        section_words.append(sum(b.source_words for b in doc.blocks[start:end]))

    def mark(start: int, end: int, name: str) -> None:
        for i in range(start, end):
            if category[i] == "unassigned":
                category[i] = name

    mark(0, min(start for start, *_ in spans), "front_matter")
    if report is not None:
        mark(report.start, report.headings[0][0], "report_toc")
        for index in report.exhibit_indexes:
            mark(index, next(b for b in ordered_boundaries if b > index), "exhibit_index")
    if signatures is not None:
        mark(signatures, n, "signatures_and_after")
    return ParsedFiling(sections, coverage(tuple(section_words), report))


def extract_filing_sections(html: str | bytes, filing_id: str) -> list[Section]:
    """Extract sections in document order; ``[]`` if none are reliable.

    The sections of ``analyze_filing``: Item sections, then any appended
    report sections (``item_number=None``).
    """
    return analyze_filing(html, filing_id).sections


def parse_filing(html: str | bytes, filing_id: str) -> list[Section]:
    """Parse a 10-K into sections, raising if none can be identified.

    Same output as ``extract_filing_sections``, but an empty result raises
    ``FilingParseError`` so a filing with an unrecognised layout fails
    loudly instead of silently contributing nothing (or guessed sections).
    """
    if not filing_id or not filing_id.strip():
        raise ValueError("filing_id must be non-empty")
    sections = extract_filing_sections(html, filing_id)
    if not sections:
        raise FilingParseError(
            f"{filing_id}: no Item sections found (no verified TOC anchors and fewer "
            "than two reliable Item headings)"
        )
    return sections


__all__ = [
    "COVERAGE_CATEGORIES",
    "TABLE_END",
    "TABLE_START",
    "FilingCoverage",
    "FilingParseError",
    "ParsedFiling",
    "analyze_filing",
    "extract_filing_sections",
    "extract_table_text",
    "extract_visible_text",
    "normalize_filing_text",
    "parse_filing",
]
