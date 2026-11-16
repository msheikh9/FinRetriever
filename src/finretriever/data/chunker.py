"""Section-aware passage chunker for FinRetriever.

Converts cleaned :class:`finretriever.data.schemas.Section` records into
retrievable :class:`finretriever.data.schemas.Passage` records. v1 stays
deliberately simple: a deterministic, whitespace-tokenised sliding-window
chunker with overlap.

Why a custom tokenizer
----------------------
We do *not* import the BGE tokenizer here. The model-level tokenizer is the
right tool at training and inference time, but it would couple the corpus
build to a specific HuggingFace model and slow it down materially. For chunk
*sizing* a stable whitespace split is enough, and it is deterministic across
machines without any model download.

Determinism
-----------
The same ``(section, ticker, form_type, filing_date, chunk_size, overlap)``
inputs always produce the same passages, with the same IDs (via
``make_passage_id``). This is required for the v1 gold qrels to remain valid
across re-runs.

Tables
------
Sections without tables are chunked by ``chunk_text`` (fixed windows).
Sections containing parser table blocks (``[TABLE]`` ... ``[/TABLE]``, one
self-describing row per line) go through ``chunk_text_table_aware``, which
packs the same tokens but treats each table row as indivisible, so a passage
never starts or ends inside a row. A row longer than the whole budget is
split between its cells, and every continuation piece repeats the row label
followed by ``(continued)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finretriever.data.filing_parser import TABLE_END, TABLE_START
from finretriever.data.schemas import Passage, Section, make_passage_id

CONTINUATION_MARKER = "(continued)"

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip the result.

    Returns an empty string for ``None``-equivalent or whitespace-only input.
    Sentence punctuation is preserved verbatim.
    """
    if not text:
        return ""
    collapsed = _WHITESPACE_RE.sub(" ", text)
    return collapsed.strip()


def simple_tokenize(text: str) -> list[str]:
    """Split ``text`` on whitespace into a list of tokens.

    Intentionally lightweight: not a model tokenizer, but stable, dependency
    free, and good enough for approximate 512-token chunk sizing on English
    SEC text. ``str.split()`` with no argument collapses runs of whitespace.
    """
    if not text:
        return []
    return text.split()


def count_tokens(text: str) -> int:
    """Return the number of whitespace tokens in ``text``."""
    return len(simple_tokenize(text))


def chunk_text(
    text: str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 80,
) -> list[str]:
    """Split ``text`` into overlapping token-bounded chunks.

    Steps:

    1. Validate parameters.
    2. Normalize whitespace; return ``[]`` if the result is empty.
    3. Tokenize, then slide a window of ``chunk_size_tokens`` with step
       ``chunk_size_tokens - chunk_overlap_tokens`` across the token list.
    4. Reconstruct each chunk by ``" ".join``-ing its tokens.

    No chunk exceeds ``chunk_size_tokens`` tokens. The final chunk may be
    shorter. Because ``overlap < chunk_size`` the step is always >= 1, so
    the loop is guaranteed to terminate.
    """
    if chunk_size_tokens <= 0:
        raise ValueError(
            f"chunk_size_tokens must be > 0, got {chunk_size_tokens}"
        )
    if chunk_overlap_tokens < 0:
        raise ValueError(
            f"chunk_overlap_tokens must be >= 0, got {chunk_overlap_tokens}"
        )
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError(
            "chunk_overlap_tokens must be < chunk_size_tokens "
            f"(got overlap={chunk_overlap_tokens}, size={chunk_size_tokens})"
        )

    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    tokens = simple_tokenize(normalized)
    if not tokens:
        return []

    if len(tokens) <= chunk_size_tokens:
        return [" ".join(tokens)]

    step = chunk_size_tokens - chunk_overlap_tokens
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size_tokens
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
        start += step
    return chunks


@dataclass(frozen=True)
class TextChunks:
    """Chunks of one text, plus how many table rows needed continuation pieces."""

    chunks: list[str] = field(default_factory=list)
    continued_rows: int = 0


def _split_long_row(row: list[str], size: int, lead: list[str]) -> list[list[str]]:
    """Split a table row that cannot fit in one chunk into labelled pieces.

    The row ``label | header: value | ...`` is split between cells; every
    piece after the first starts with the label and ``(continued)``. A single
    cell longer than a whole piece is split by tokens, still under the label.
    """
    cells = " ".join(row).split(" | ")
    label, values = (cells[0].split(), cells[1:]) if len(cells) > 1 else ([], cells)
    separator = ["|"] if label else []
    prefix = [*label, CONTINUATION_MARKER]
    if len(lead) + len(label) + 2 > size or len(prefix) + 2 > size:
        # Degenerate label that nearly fills a chunk by itself: plain windows.
        tokens = lead + row
        pieces = [tokens[:size]]
        rest = tokens[size:]
        while rest:
            pieces.append([CONTINUATION_MARKER, *rest[: size - 1]])
            rest = rest[size - 1:]
        return pieces

    pieces: list[list[str]] = []
    current = [*lead, *label]
    has_value = False
    for value in values:
        cell = [*separator, *value.split()]
        if has_value and len(current) + len(cell) > size:
            pieces.append(current)
            current, has_value = list(prefix), False
        # Only a cell longer than a whole piece is split; the guard above
        # leaves at least two tokens of room, so every pass makes progress.
        while len(current) + len(cell) > size:
            room = size - len(current)
            pieces.append(current + cell[:room])
            current, cell = list(prefix), [*separator, *cell[room:]]
        current.extend(cell)
        has_value = True
    pieces.append(current)
    return pieces


def _table_aware_units(text: str, size: int) -> tuple[list[list[str]], int]:
    """Split ``text`` into packing units: single prose tokens and whole table rows.

    The ``[TABLE]``/``[/TABLE]`` markers are attached to the first/last row
    so a chunk never ends with a dangling marker. If the last row already
    fills a whole chunk, ``[/TABLE]`` moves to the front of the next unit
    instead (or is dropped at the end of the text): a unit, and so a chunk,
    never consists of a marker alone.
    """
    units: list[list[str]] = []
    continued = 0
    in_table = pending_end = False
    lead: list[str] = []
    last_row: int | None = None

    def add(new_units: list[list[str]]) -> None:
        nonlocal pending_end
        if pending_end and new_units and len(new_units[0]) < size:
            new_units[0].insert(0, TABLE_END)
        pending_end = False
        units.extend(new_units)

    for raw_line in text.split("\n"):
        tokens = raw_line.split()
        if not tokens:
            continue
        line = " ".join(tokens)
        if line == TABLE_START:
            in_table, lead, last_row = True, [TABLE_START], None
        elif line == TABLE_END:
            if last_row is not None:
                if len(units[last_row]) < size:
                    units[last_row].append(TABLE_END)
                else:
                    pending_end = True
            in_table, lead = False, []
        elif in_table:
            row = lead + tokens
            if len(row) <= size:
                add([row])
            else:
                add(_split_long_row(tokens, size, lead))
                continued += 1
            lead, last_row = [], len(units) - 1
        else:
            add([[token] for token in tokens])
    return units, continued


def _pack(units: list[list[str]], size: int, overlap: int) -> list[list[str]]:
    """Greedily fill chunks with whole units; overlap backs up by whole units.

    With single-token units this reproduces ``chunk_text``'s windows exactly.
    """
    chunks: list[list[str]] = []
    start = 0
    while start < len(units):
        used, end = 0, start
        while end < len(units) and used + len(units[end]) <= size:
            used += len(units[end])
            end += 1
        chunks.append([token for unit in units[start:end] for token in unit])
        if end >= len(units):
            break
        back, next_start = 0, end
        while next_start - 1 > start and back + len(units[next_start - 1]) <= overlap:
            back += len(units[next_start - 1])
            next_start -= 1
        start = next_start
    return chunks


def chunk_text_table_aware(
    text: str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 80,
) -> TextChunks:
    """Chunk ``text`` without ever cutting a parser table row in two.

    Text without a ``[TABLE]`` block is passed to ``chunk_text`` unchanged.
    Otherwise prose tokens and whole table rows are packed greedily into
    chunks of at most ``chunk_size_tokens`` tokens; consecutive chunks
    overlap by at most ``chunk_overlap_tokens`` tokens of whole units (so the
    overlap shrinks, possibly to zero, rather than starting mid-row). Rows
    longer than ``chunk_size_tokens`` are split into labelled continuation
    pieces (see ``_split_long_row``) and counted in ``continued_rows``.
    """
    if TABLE_START not in text:
        return TextChunks(chunk_text(text, chunk_size_tokens, chunk_overlap_tokens))
    chunk_text("", chunk_size_tokens, chunk_overlap_tokens)  # validates the parameters
    units, continued = _table_aware_units(text, chunk_size_tokens)
    packed = _pack(units, chunk_size_tokens, chunk_overlap_tokens)
    return TextChunks([" ".join(chunk) for chunk in packed], continued)


def chunk_section(
    section: Section,
    ticker: str,
    form_type: str,
    filing_date: str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 80,
) -> list[Passage]:
    """Chunk a single :class:`Section` into :class:`Passage` records.

    The Passage ID is produced by ``make_passage_id`` so it is stable across
    runs. ``ticker`` and ``form_type`` are normalised on the Passage itself
    (the model validators uppercase them).

    Sections with parser table blocks are chunked by
    ``chunk_text_table_aware`` so no passage starts or ends inside a table
    row; all other sections use ``chunk_text``.

    Returns ``[]`` if the section text normalises to empty. In practice the
    Section model already rejects empty text, so this is a defensive check.
    """
    chunks = chunk_text_table_aware(
        section.text,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    ).chunks
    if not chunks:
        return []

    passages: list[Passage] = []
    for chunk_index, chunk in enumerate(chunks):
        metadata: dict[str, str | int | float | bool | None] = {
            "chunk_index": chunk_index,
            "section_title": section.title,
        }
        if section.item_number is not None:
            metadata["item_number"] = section.item_number

        passages.append(
            Passage(
                passage_id=make_passage_id(
                    section.filing_id, section.section_id, chunk_index
                ),
                filing_id=section.filing_id,
                section_id=section.section_id,
                ticker=ticker,
                form_type=form_type,
                filing_date=filing_date,
                section_title=section.title,
                text=chunk,
                token_count=count_tokens(chunk),
                metadata=metadata,
            )
        )
    return passages


def chunk_sections(
    sections: list[Section],
    ticker: str,
    form_type: str,
    filing_date: str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 80,
) -> list[Passage]:
    """Flat-map ``chunk_section`` across a list of sections.

    Section order is preserved, and within each section the chunk order
    matches ``chunk_text``.
    """
    out: list[Passage] = []
    for section in sections:
        out.extend(
            chunk_section(
                section,
                ticker=ticker,
                form_type=form_type,
                filing_date=filing_date,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        )
    return out


__all__ = [
    "CONTINUATION_MARKER",
    "TextChunks",
    "chunk_section",
    "chunk_sections",
    "chunk_text",
    "chunk_text_table_aware",
    "count_tokens",
    "normalize_whitespace",
    "simple_tokenize",
]
