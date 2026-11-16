"""Locating an evidence quote in a passage: verbatim, as fragments, or under a rebuilt row label.

Used by validator ``sec-qcheck-v3`` (``finretriever.training.query_generation``) and by the
evidence-reuse checks. Every form is checked against the text itself, so an invented quote, cell
or label still fails:

1. **Verbatim.** The quote, case-folded with quotes and dashes unified, table pipes dropped and
   spaces collapsed, is a substring of the text (validator v2's only form).
2. **Fragments.** The quote is split at ellipses (``...``, ``…``) and table-cell separators
   (``|``). Every fragment holds a letter or digit and is found in the text, in order, and
   together the fragments hold at least ``MIN_FRAGMENT_CHARS`` letters and digits. Each gap must
   be one the quote may skip:
   * after an ellipsis, at most ``MAX_ELLIPSIS_GAP`` characters and no table boundary, and the
     next fragment starts with a word (a value after "..." could belong to any row);
   * after a cell separator, whole cells of the same row only: at most ``MAX_SKIPPED_CELLS``,
     with no table boundary and no row boundary. A row boundary is a value followed by words
     (``304,739 Operating expenses``) or a label marker (``>``), as flattened tables join rows.
3. **Rebuilt row label.** Flattened tables sometimes shorten a row's parent label: "For funded
   pension plans with an accumulated benefit obligation in excess of plan assets > Accumulated
   benefit obligation | ..." is followed by "in excess of plan assets > Fair value of plan assets |
   13,266 | ...". A quote naming the full parent is accepted when its leaf label and cells match
   as in (2), the text's shortened parent is the end of the quoted parent, and the nearest earlier
   row of the same table that states a longer parent ending the same way states the quoted one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MIN_FRAGMENT_CHARS = 15
MAX_ELLIPSIS_GAP = 600
MAX_SKIPPED_CELLS = 6
CELL = "\x00"  # stands for a table pipe in normalised text (not whitespace, unlike \x1c-\x1f)
_SPACE = r"(?:\s|\x00)+"  # a space in a quote may span a cell separator in the text
_TRANSLATE = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\s*\.\s*\.|…)\s*")
_PIECE_TOKEN_RE = re.compile(r"\[/?table\]|>|[a-z][a-z'\-]*|\(?\$?-?\d[\d,]*(?:\.\d+)?\)?%?")
_YEAR_RE = re.compile(r"\(?(?:19|20)\d\d\)?")
_NOT_ROW_START = frozenset(
    "million billion thousand trillion to and or of years year months percent per".split()
)
Form = Literal["verbatim", "fragments", "rebuilt_label"]


def normalise(text: str) -> str:
    """Case-folded text with quotes and dashes unified, pipes as ``CELL`` and spaces collapsed."""
    text = text.casefold().translate(_TRANSLATE).replace("|", f" {CELL} ")
    return re.sub(r"\s+", " ", text).strip()


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(CELL, " ")).strip()


def _alnum(text: str) -> int:
    return sum(c.isalnum() for c in text)


@dataclass(frozen=True)
class QuoteMatch:
    form: Form
    start: int  # in the normalised text
    end: int


def _pattern(fragment: str) -> re.Pattern[str]:
    parts = [re.escape(w) for w in fragment.split(" ") if w]
    return re.compile(_SPACE.join(parts))


def _row_boundary(piece: str) -> bool:
    tokens = _PIECE_TOKEN_RE.findall(piece)
    for i, token in enumerate(tokens):
        if token in (">", "[table]", "[/table]"):
            return True
        if token[0].isalpha() or _YEAR_RE.fullmatch(token):
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if nxt[:1].isalpha() and nxt not in _NOT_ROW_START:
            return True
    return False


def _gap_ok(gap: str, kind: str) -> bool:
    if "[table]" in gap or "[/table]" in gap:
        return False
    if kind == "ellipsis":
        return len(gap) <= MAX_ELLIPSIS_GAP
    if CELL not in gap:
        return False
    pieces = gap.split(CELL)
    if len(pieces) - 2 > MAX_SKIPPED_CELLS:
        return False
    return not any(_row_boundary(p) for p in pieces)


def _split(quote: str) -> list[tuple[str, str]]:
    """``(separator before it, fragment)`` pairs; the first separator is empty."""
    out: list[tuple[str, str]] = []
    for i, part in enumerate(_ELLIPSIS_RE.split(quote)):
        sep = "ellipsis" if i else "cell"
        for cell in part.split(CELL):
            cell = cell.strip()
            if cell:
                out.append((sep if out else "", cell))
                sep = "cell"
    return out


def _match_fragments(
    fragments: list[tuple[str, str]], text: str, start: int = 0
) -> tuple[int, int] | None:
    """Earliest in-order placement of ``fragments`` from ``start`` with legitimate gaps."""
    if not fragments or any(_alnum(f) == 0 for _, f in fragments):
        return None
    # After an ellipsis the quote resumes with words, so a value is never joined to a far-off label.
    if any(sep == "ellipsis" and not f[0].isalpha() for sep, f in fragments):
        return None
    if sum(_alnum(f) for _, f in fragments) < MIN_FRAGMENT_CHARS:
        return None
    first = _pattern(fragments[0][1])
    for m in first.finditer(text, start):
        end, ok = m.end(), True
        for sep, fragment in fragments[1:]:
            nxt = _pattern(fragment).search(text, end)
            if nxt is None or not _gap_ok(text[end : nxt.start()], sep):
                ok = False
                break
            end = nxt.end()
        if ok:
            return m.start(), end
    return None


def _parent_before(text: str, marker: int) -> str:
    """The parent label ending at ``marker`` (the index of a ``>``), back to the row's start."""
    head = text[:marker]
    cut = max(head.rfind(CELL), head.rfind("[table]") + len("[table]") - 1)
    piece = head[cut + 1 :]
    tokens = _PIECE_TOKEN_RE.findall(piece)
    for i in range(len(tokens) - 1, -1, -1):  # after the last value: the row's own label
        token = tokens[i]
        if not token[0].isalpha() and token != ">" and not _YEAR_RE.fullmatch(token):
            tokens = tokens[i + 1 :]
            break
    return " ".join(t for t in tokens if t[0].isalpha() or _YEAR_RE.fullmatch(t))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text)


def _ends_with(longer: list[str], shorter: list[str]) -> bool:
    return len(longer) >= len(shorter) and longer[len(longer) - len(shorter) :] == shorter


def _rebuilt_label(fragments: list[tuple[str, str]], text: str) -> tuple[int, int] | None:
    sep0, label = fragments[0]
    parts = [p.strip() for p in label.split(">")]
    for k in range(1, len(parts)):
        ancestor = _words(" ".join(parts[:k]))
        leaf = [(sep0, " > ".join(parts[k:]))] + fragments[1:]
        found = _match_fragments(leaf, text)
        while found is not None:
            start = found[0]
            table_start = text.rfind("[table]", 0, start)
            head = text[:start].rstrip()
            visible_parent = (
                _words(_parent_before(text, len(head) - 1)) if head.endswith(">") else []
            )
            if _ends_with(ancestor, visible_parent) and table_start >= 0:
                row_start = len(head) - 1 if head.endswith(">") else start
                position = text.rfind(">", table_start, row_start)
                while position > table_start:
                    parent = _words(_parent_before(text, position))
                    if len(parent) > len(visible_parent) and _ends_with(parent, visible_parent):
                        if _ends_with(parent, ancestor):
                            return found
                        break
                    position = text.rfind(">", table_start, position)
            found = _match_fragments(leaf, text, start + 1)
    return None


def locate_quote(quote: str, text: str) -> QuoteMatch | None:
    """Where ``quote`` lies in ``text`` (see the module docstring), or ``None``."""
    nq, nt = normalise(quote), normalise(text)
    flat_q, flat_t = _flat(nq), _flat(nt)
    if flat_q and flat_q in flat_t:
        at = _pattern(flat_q).search(nt)
        if at is not None:
            return QuoteMatch("verbatim", at.start(), at.end())
    fragments = _split(nq)
    found = _match_fragments(fragments, nt)
    if found is not None:
        return QuoteMatch("fragments", *found)
    if fragments and ">" in fragments[0][1]:
        found = _rebuilt_label(fragments, nt)
        if found is not None:
            return QuoteMatch("rebuilt_label", *found)
    return None


def table_label_grams(
    text: str, *, min_words: int = 2, max_words: int = 10
) -> set[tuple[str, ...]]:
    """Every run of 2-10 consecutive words inside a table's labels and titles (no digits)."""
    out: set[tuple[str, ...]] = set()
    for block in re.findall(r"\[table\](.*?)(?:\[/table\]|$)", normalise(text), flags=re.DOTALL):
        run: list[str] = []
        for token in [*_PIECE_TOKEN_RE.findall(block.replace(CELL, " > ")), ">"]:
            if token[0].isalpha():
                run.extend(_words(token))
                continue
            for n in range(min_words, max_words + 1):
                out.update(tuple(run[i : i + n]) for i in range(len(run) - n + 1))
            run = []
    return out


__all__ = [
    "CELL",
    "MAX_ELLIPSIS_GAP",
    "MAX_SKIPPED_CELLS",
    "MIN_FRAGMENT_CHARS",
    "QuoteMatch",
    "locate_quote",
    "normalise",
    "table_label_grams",
]
