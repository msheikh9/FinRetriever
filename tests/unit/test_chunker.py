"""Tests for finretriever.data.chunker.

Chunking is load-bearing for both training and evaluation: a bad chunker
silently corrupts qrels, training pairs, and metrics. These tests pin down
the behaviour that the rest of the pipeline depends on.
"""

from __future__ import annotations

import pytest

from finretriever.data.chunker import (
    CONTINUATION_MARKER,
    chunk_section,
    chunk_sections,
    chunk_text,
    chunk_text_table_aware,
    count_tokens,
    normalize_whitespace,
    simple_tokenize,
)
from finretriever.data.filing_parser import TABLE_END, TABLE_START
from finretriever.data.schemas import Passage, Section, make_passage_id

# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


def test_normalize_whitespace_collapses_spaces_tabs_and_newlines() -> None:
    assert normalize_whitespace("a   b\t\tc\n\nd") == "a b c d"
    assert normalize_whitespace("  leading and trailing   ") == "leading and trailing"
    assert normalize_whitespace("line one\nline two\r\nline three") == (
        "line one line two line three"
    )


def test_normalize_whitespace_returns_empty_for_whitespace_only() -> None:
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   ") == ""
    assert normalize_whitespace("\n\t   \r\n") == ""


def test_normalize_whitespace_preserves_sentence_punctuation() -> None:
    assert (
        normalize_whitespace("The company,  Inc.   reported earnings.")
        == "The company, Inc. reported earnings."
    )


# ---------------------------------------------------------------------------
# simple_tokenize / count_tokens
# ---------------------------------------------------------------------------


def test_simple_tokenize_is_deterministic() -> None:
    text = "Risk factors disclosed under Item 1A include market volatility."
    assert simple_tokenize(text) == simple_tokenize(text)


def test_simple_tokenize_splits_on_whitespace() -> None:
    assert simple_tokenize("alpha beta gamma") == ["alpha", "beta", "gamma"]
    # str.split collapses runs of whitespace automatically.
    assert simple_tokenize("alpha   beta\tgamma\n delta") == [
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]


def test_simple_tokenize_empty_input() -> None:
    assert simple_tokenize("") == []
    assert simple_tokenize("   ") == []


def test_count_tokens_matches_simple_tokenize_length() -> None:
    text = "Apple Inc. (NASDAQ: AAPL) is a publicly traded company."
    assert count_tokens(text) == len(simple_tokenize(text))
    assert count_tokens("") == 0


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_returns_empty_for_empty_or_whitespace_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t") == []


def test_chunk_text_returns_single_chunk_when_text_is_short() -> None:
    chunks = chunk_text(
        "short passage of text", chunk_size_tokens=10, chunk_overlap_tokens=2
    )
    assert chunks == ["short passage of text"]


def test_chunk_text_splits_long_text_into_multiple_chunks() -> None:
    text = " ".join(f"tok{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size_tokens=10, chunk_overlap_tokens=2)
    assert len(chunks) > 1


def test_chunk_text_no_chunk_exceeds_size() -> None:
    text = " ".join(f"tok{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size_tokens=10, chunk_overlap_tokens=2)
    assert chunks  # sanity
    for c in chunks:
        assert count_tokens(c) <= 10


def test_chunk_text_applies_overlap_at_chunk_boundaries() -> None:
    # Use zero-padded distinct tokens so we can compare slices exactly.
    tokens = [f"tok{i:03d}" for i in range(50)]
    text = " ".join(tokens)
    overlap = 3
    chunks = chunk_text(text, chunk_size_tokens=10, chunk_overlap_tokens=overlap)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_tail = prev.split()[-overlap:]
        next_head = nxt.split()[:overlap]
        assert prev_tail == next_head


def test_chunk_text_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", chunk_size_tokens=0, chunk_overlap_tokens=0)
    with pytest.raises(ValueError):
        chunk_text("anything", chunk_size_tokens=-1, chunk_overlap_tokens=0)


def test_chunk_text_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", chunk_size_tokens=10, chunk_overlap_tokens=-1)


def test_chunk_text_rejects_overlap_geq_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", chunk_size_tokens=10, chunk_overlap_tokens=10)
    with pytest.raises(ValueError):
        chunk_text("anything", chunk_size_tokens=10, chunk_overlap_tokens=11)


def test_chunk_text_exact_chunk_size_returns_one_chunk() -> None:
    tokens = [f"tok{i}" for i in range(10)]
    text = " ".join(tokens)
    chunks = chunk_text(text, chunk_size_tokens=10, chunk_overlap_tokens=2)
    assert len(chunks) == 1
    assert simple_tokenize(chunks[0]) == tokens


def test_chunk_text_handles_heavy_punctuation_without_crashing() -> None:
    text = (
        "The company's risk factors include: market volatility, "
        "supply-chain disruption (notably semiconductors), and changes "
        "in tax law — see Item 1A!"
    )
    chunks = chunk_text(text, chunk_size_tokens=8, chunk_overlap_tokens=2)
    assert len(chunks) >= 1
    assert all(c for c in chunks)
    for c in chunks:
        assert count_tokens(c) <= 8


def test_chunk_text_small_window_with_overlap_terminates() -> None:
    # chunk_size=3, overlap=2 implies step=1: the worst-case but still legal
    # configuration. Must still terminate and respect the size cap.
    text = " ".join(f"tok{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size_tokens=3, chunk_overlap_tokens=2)
    assert 0 < len(chunks) < 250  # bounded, finite
    for c in chunks:
        assert count_tokens(c) <= 3


def test_chunk_text_is_deterministic_across_runs() -> None:
    text = " ".join(f"tok{i}" for i in range(75))
    a = chunk_text(text, chunk_size_tokens=12, chunk_overlap_tokens=4)
    b = chunk_text(text, chunk_size_tokens=12, chunk_overlap_tokens=4)
    assert a == b


# ---------------------------------------------------------------------------
# chunk_section
# ---------------------------------------------------------------------------


def _make_section(text: str, *, item_number: str | None = "1A") -> Section:
    return Section(
        section_id="section_filing_x_0001_abc123",
        filing_id="filing_x",
        title="Item 1A — Risk Factors",
        item_number=item_number,
        text=text,
    )


def test_chunk_section_returns_passage_objects() -> None:
    text = " ".join(f"tok{i}" for i in range(120))
    section = _make_section(text)
    passages = chunk_section(
        section,
        ticker="AAPL",
        form_type="10-K",
        filing_date="2023-11-03",
        chunk_size_tokens=20,
        chunk_overlap_tokens=4,
    )
    assert len(passages) > 1
    assert all(isinstance(p, Passage) for p in passages)


def test_chunk_section_populates_required_fields_and_metadata() -> None:
    section = _make_section("Risk factors are discussed below in detail.")
    passages = chunk_section(
        section,
        ticker="AAPL",
        form_type="10-K",
        filing_date="2023-11-03",
        chunk_size_tokens=20,
        chunk_overlap_tokens=4,
    )
    assert len(passages) == 1
    p = passages[0]
    assert p.filing_id == section.filing_id
    assert p.section_id == section.section_id
    assert p.section_title == section.title
    assert p.filing_date == "2023-11-03"
    assert p.token_count == count_tokens(p.text)
    assert p.metadata["chunk_index"] == 0
    assert p.metadata["section_title"] == section.title
    assert p.metadata["item_number"] == "1A"


def test_chunk_section_omits_item_number_when_section_has_none() -> None:
    section = _make_section("body text without an item number", item_number=None)
    passages = chunk_section(
        section, ticker="AAPL", form_type="10-K", filing_date="2023-11-03"
    )
    assert passages
    assert "item_number" not in passages[0].metadata


def test_chunk_section_produces_deterministic_passage_ids() -> None:
    section = _make_section(" ".join(f"tok{i}" for i in range(60)))
    a = chunk_section(
        section,
        ticker="AAPL",
        form_type="10-K",
        filing_date="2023-11-03",
        chunk_size_tokens=20,
        chunk_overlap_tokens=4,
    )
    b = chunk_section(
        section,
        ticker="AAPL",
        form_type="10-K",
        filing_date="2023-11-03",
        chunk_size_tokens=20,
        chunk_overlap_tokens=4,
    )
    assert [p.passage_id for p in a] == [p.passage_id for p in b]
    # And IDs match a direct call to make_passage_id with the same args.
    assert a[0].passage_id == make_passage_id(
        section.filing_id, section.section_id, 0
    )
    assert a[1].passage_id == make_passage_id(
        section.filing_id, section.section_id, 1
    )


def test_chunk_section_passages_have_normalized_ticker_and_form_type() -> None:
    section = _make_section("some words here")
    passages = chunk_section(
        section, ticker="aapl", form_type="10-k", filing_date="2023-11-03"
    )
    assert all(p.ticker == "AAPL" for p in passages)
    assert all(p.form_type == "10-K" for p in passages)


# ---------------------------------------------------------------------------
# chunk_sections
# ---------------------------------------------------------------------------


def test_chunk_sections_preserves_section_and_chunk_order() -> None:
    s1 = Section(
        section_id="section_filing_x_0001_a",
        filing_id="filing_x",
        title="Item 1",
        text=" ".join(f"a{i:02d}" for i in range(50)),
    )
    s2 = Section(
        section_id="section_filing_x_0002_b",
        filing_id="filing_x",
        title="Item 1A",
        text=" ".join(f"b{i:02d}" for i in range(50)),
    )
    passages = chunk_sections(
        [s1, s2],
        ticker="AAPL",
        form_type="10-K",
        filing_date="2023-11-03",
        chunk_size_tokens=15,
        chunk_overlap_tokens=3,
    )
    # All s1 passages must come strictly before all s2 passages.
    section_ids = [p.section_id for p in passages]
    first_s2 = section_ids.index(s2.section_id)
    assert all(sid == s1.section_id for sid in section_ids[:first_s2])
    assert all(sid == s2.section_id for sid in section_ids[first_s2:])
    # Within each section, chunk_index must be strictly increasing.
    s1_indices = [
        p.metadata["chunk_index"] for p in passages if p.section_id == s1.section_id
    ]
    s2_indices = [
        p.metadata["chunk_index"] for p in passages if p.section_id == s2.section_id
    ]
    assert s1_indices == sorted(s1_indices)
    assert s2_indices == sorted(s2_indices)
    assert s1_indices[0] == 0
    assert s2_indices[0] == 0


def test_chunk_sections_returns_flat_list_of_passages() -> None:
    sections = [
        Section(
            section_id=f"section_filing_x_{i:04d}_x",
            filing_id="filing_x",
            title=f"Item {i}",
            text=" ".join(f"tok{j}" for j in range(40)),
        )
        for i in range(3)
    ]
    passages = chunk_sections(
        sections,
        ticker="AAPL",
        form_type="10-K",
        filing_date="2023-11-03",
        chunk_size_tokens=15,
        chunk_overlap_tokens=3,
    )
    assert isinstance(passages, list)
    assert passages
    assert all(isinstance(p, Passage) for p in passages)


def test_chunk_sections_is_deterministic_across_runs() -> None:
    sections = [
        Section(
            section_id="section_filing_x_0001_a",
            filing_id="filing_x",
            title="Item 1A",
            text=" ".join(f"tok{j}" for j in range(80)),
        ),
        Section(
            section_id="section_filing_x_0002_b",
            filing_id="filing_x",
            title="Item 7",
            text=" ".join(f"mda{j}" for j in range(80)),
        ),
    ]
    a = chunk_sections(
        sections, ticker="AAPL", form_type="10-K", filing_date="2023-11-03"
    )
    b = chunk_sections(
        sections, ticker="AAPL", form_type="10-K", filing_date="2023-11-03"
    )
    assert [p.passage_id for p in a] == [p.passage_id for p in b]
    assert [p.text for p in a] == [p.text for p in b]


# ---------------------------------------------------------------------------
# chunk_text_table_aware
# ---------------------------------------------------------------------------


def prose(n: int, word: str = "w") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def table(rows: list[str]) -> str:
    return "\n".join([TABLE_START, *rows, TABLE_END])


def statement_rows() -> list[str]:
    """Income-statement rows as the parser renders them (group > child | col: value)."""
    rows = []
    for group in ("Net sales", "Cost of sales"):
        for year_offset, item in enumerate(("Products", "Services")):
            rows.append(
                f"{group} > {item} | Years ended September 27, 2025: ${300 + year_offset},000 | "
                f"Years ended September 28, 2024: ${290 + year_offset},000"
            )
        rows.append(f"Total {group.lower()} | Years ended September 27, 2025: 400,000")
    return rows


def unique_rows(n: int) -> list[str]:
    return [f"Group {i // 3} > Line {i} | 2025: ${i},000 | Change: {i}% | 2024: ${i},500" for i in range(n)]


def chunk_spans(text: str, chunks: list[str]) -> list[tuple[int, int]]:
    """Token span of each chunk in ``text`` (chunks must be contiguous slices, in order)."""
    tokens = text.split()
    spans, previous = [], -1
    for chunk in chunks:
        part = chunk.split()
        start = next(i for i in range(previous + 1, len(tokens)) if tokens[i:i + len(part)] == part)
        spans.append((start, start + len(part)))
        previous = start
    return spans


def row_spans(text: str) -> list[tuple[int, int]]:
    spans, offset, inside = [], 0, False
    for line in text.split("\n"):
        n = len(line.split())
        if line == TABLE_START:
            inside = True
        elif line == TABLE_END:
            inside = False
        elif inside:
            spans.append((offset, offset + n))
        offset += n
    return spans


def contiguous_windows(text: str, chunks: list[str]) -> None:
    """Chunks are in-order token slices of ``text`` that cover it without gaps."""
    spans = chunk_spans(text, chunks)
    assert spans[0][0] == 0 and spans[-1][1] == len(text.split())
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert start <= end, "gap between consecutive chunks"


@pytest.mark.parametrize("size,overlap", [(512, 80), (50, 10), (7, 0), (13, 12)])
def test_table_aware_matches_fixed_windows_for_prose(size: int, overlap: int) -> None:
    text = prose(700)
    result = chunk_text_table_aware(text, size, overlap)
    assert result.chunks == chunk_text(text, size, overlap)
    assert result.continued_rows == 0


def test_prose_before_a_table_keeps_fixed_windows() -> None:
    text = prose(200) + "\n" + table(statement_rows())
    aware = chunk_text_table_aware(text, 50, 10).chunks
    fixed = chunk_text(prose(200), 50, 10)
    assert aware[:3] == fixed[:3]  # windows that end before the table are unchanged


@pytest.mark.parametrize("size,overlap", [(25, 0), (40, 10), (60, 15), (90, 30), (512, 80)])
def test_table_rows_are_never_split_across_chunks(size: int, overlap: int) -> None:
    rows = unique_rows(40)
    text = prose(37) + "\n" + table(rows) + "\n" + prose(23, "z")
    result = chunk_text_table_aware(text, size, overlap)
    assert result.continued_rows == 0
    for chunk in result.chunks:
        assert count_tokens(chunk) <= size
    for row in rows:
        assert any(f" {row} " in f" {c} " for c in result.chunks), row
    rows_at = row_spans(text)
    for start, end in chunk_spans(text, result.chunks):
        for a, b in rows_at:
            assert not a < start < b and not a < end < b, "chunk edge inside a table row"
    contiguous_windows(text, result.chunks)


def test_closing_marker_never_forms_a_chunk_on_its_own() -> None:
    row = "Matter | " + prose(40, "t")
    text = table([row]) + "\nAfter the table."
    for size in (12, 13, 14, 15, 16):
        chunks = chunk_text_table_aware(text, size, 3).chunks
        assert all(c.split() not in ([TABLE_END], [TABLE_START]) for c in chunks)
        assert all(count_tokens(c) <= size for c in chunks)
        # The marker survives, once: on the last row piece or before "After".
        assert sum(c.split().count(TABLE_END) for c in chunks if "After" in c or "t39" in c) >= 1
    # 52 values fill four 16-token pieces exactly (13 values each); with nothing
    # after the table, the marker is dropped rather than emitted alone.
    table_only = chunk_text_table_aware(table(["Matter | " + prose(52, "t")]), 16, 3).chunks
    assert [count_tokens(c) for c in table_only] == [16, 16, 16, 16]
    assert table_only[-1].split()[-1] == "t51"


def test_repeated_child_labels_keep_their_parent_in_every_chunk() -> None:
    chunks = chunk_text_table_aware(table(statement_rows()), 30, 8).chunks
    products = [c for c in chunks if "Products" in c]
    assert products
    for chunk in products:
        for start in [i for i, t in enumerate(chunk.split()) if t == "Products"]:
            assert chunk.split()[start - 1] == ">"  # always "Net sales >" or "Cost of sales >"


def test_overlap_near_tables_uses_whole_rows_and_stays_within_budget() -> None:
    rows = [f"Row {i} | 2025: {i},000 | 2024: {i},500" for i in range(12)]  # 9 tokens each
    chunks = chunk_text_table_aware(table(rows), 30, 10).chunks
    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = [row for row in rows if row in previous and row in current]
        assert len(shared) == 1  # one whole row (9 tokens <= 10) is carried over
        assert current.startswith(shared[0])


def test_rows_larger_than_overlap_are_not_partially_repeated() -> None:
    rows = [f"Row {i} | " + " | ".join(f"c{k}: {i}{k}" for k in range(5)) for i in range(6)]  # 18 tokens
    chunks = chunk_text_table_aware(table(rows), 40, 10).chunks
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert not any(row in previous and row in current for row in rows)
    assert all(any(row in c for c in chunks) for row in rows)


def test_table_markers_stay_attached_to_rows() -> None:
    text = prose(25) + "\n" + table(statement_rows()) + "\n" + prose(25, "z")
    for size in range(30, 80, 7):
        for chunk in chunk_text_table_aware(text, size, 5).chunks:
            assert not chunk.endswith(TABLE_START)
            assert not chunk.startswith(TABLE_END)


def test_long_row_is_split_into_labelled_continuations() -> None:
    row = "Net sales > Products | " + " | ".join(f"Col {k}: ${k},000" for k in range(30))
    result = chunk_text_table_aware(table([row, "Short row | 2025: 1"]), 24, 6)
    assert result.continued_rows == 1
    pieces = [c for c in result.chunks if "Col" in c]
    assert len(pieces) > 2
    assert all(count_tokens(c) <= 24 for c in result.chunks)
    for piece in pieces[1:]:
        assert piece.startswith(f"Net sales > Products {CONTINUATION_MARKER} | Col")
    values = [t for t in row.split() if t.startswith("$")]
    joined = " ".join(result.chunks)
    assert all(joined.count(v) >= 1 for v in values)
    assert sum(c.count("Col ") for c in pieces) >= 30  # every column survives


def test_long_single_cell_is_split_under_its_label() -> None:
    row = "Description of the Matter | " + prose(60, "t")
    result = chunk_text_table_aware(table([row]), 16, 4)
    assert result.continued_rows == 1
    assert all(c.startswith(("[TABLE] Description of the Matter |", "Description of the Matter (continued) |"))
               for c in result.chunks)
    body = [t for c in result.chunks for t in c.split() if t.startswith("t") and t[1:].isdigit()]
    assert body == prose(60, "t").split()


def test_table_aware_validates_parameters() -> None:
    with pytest.raises(ValueError, match="chunk_overlap_tokens must be <"):
        chunk_text_table_aware(table(["A | 1"]), 10, 10)


def test_chunk_section_uses_table_aware_boundaries_deterministically() -> None:
    text = "Item 8. Financial Statements\n" + prose(20) + "\n" + table(statement_rows() * 3)
    section = Section(section_id="section_f_0008_x", filing_id="f", title="Statements", text=text)
    first = chunk_section(section, "AAPL", "10-K", "2025-10-31", chunk_size_tokens=40, chunk_overlap_tokens=10)
    second = chunk_section(section, "AAPL", "10-K", "2025-10-31", chunk_size_tokens=40, chunk_overlap_tokens=10)
    assert [p.text for p in first] == chunk_text_table_aware(text, 40, 10).chunks
    assert [p.passage_id for p in first] == [p.passage_id for p in second]
    assert [p.passage_id for p in first] == [make_passage_id("f", section.section_id, i) for i in range(len(first))]
