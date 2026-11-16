"""Token windows over long passages, so that no passage text is lost to the encoder's length limit.

90.8% of corpus passages are longer than the dense encoders' 512-token window and 27.6% of passage tokens are
never encoded (docs/retrieval_analysis_v1.md). This module derives *window passages* from the corpus, measured in
the encoder's own tokens, without changing any corpus passage:

* :func:`tail_windows` covers only what the encoder cuts off: windows start ``overlap`` tokens before the end of
  the encoded head and continue to the end of the passage. Scored together with the passage's own (head)
  embedding, a passage is represented in full (MaxP: its score is the best of its head and its windows).
* :func:`sliding_windows` covers each passage from the start with shorter windows (for example 256 tokens), for
  the evidence-position problem (a detail deep inside a 512-token passage is diluted in one vector).

Window boundaries fall only between words (never inside a WordPiece word), and each window's text is the exact
substring of the passage text. Every window passage carries ``parent_passage_id``, so results map back to the
source passage and its filing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from finretriever.data.schemas import Passage

WINDOW_SEPARATOR = "#w"


def _token_spans(text: str, tokenizer: Any) -> tuple[list[tuple[int, int]], list[bool]]:
    """Character span of every token and whether a window may start at it (not a ``##`` continuation)."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = [tuple(s) for s in encoded["offset_mapping"]]
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    starts = [not (i > 0 and str(tok).startswith("##") and spans[i][0] == spans[i - 1][1])
              for i, tok in enumerate(tokens)]
    return spans, starts


def _boundary_at_or_before(starts: Sequence[bool], position: int, floor: int) -> int:
    """The last token index ``<= position`` (and ``> floor``) where a window may start or end."""
    for i in range(min(position, len(starts) - 1), floor, -1):
        if starts[i]:
            return i
    return position  # one word longer than a window: cut inside it rather than stall


def _windows(n: int, starts: Sequence[bool], first_start: int, size: int, overlap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = first_start
    while start < n:
        end = n if start + size >= n else _boundary_at_or_before(starts, start + size, start)
        out.append((start, end))
        if end >= n:
            break
        nxt = _boundary_at_or_before(starts, end - overlap, start)
        start = nxt if nxt > start else end
    return out


def _window_passage(parent: Passage, index: int, text: str, span: tuple[int, int], kind: str) -> Passage:
    return parent.model_copy(update={
        "passage_id": f"{parent.passage_id}{WINDOW_SEPARATOR}{index}",
        "text": text,
        "token_count": None,
        "metadata": {**parent.metadata, "parent_passage_id": parent.passage_id, "window_kind": kind,
                     "window_index": index, "window_start_token": span[0], "window_end_token": span[1]},
    })


def tail_windows(passages: Sequence[Passage], tokenizer: Any, *, head_tokens: int, size: int,
                 overlap: int) -> list[Passage]:
    """Windows over the tokens past each passage's first ``head_tokens`` (the part an encoder truncates)."""
    _check(size, overlap)
    out: list[Passage] = []
    for p in passages:
        spans, starts = _token_spans(p.text, tokenizer)
        if len(spans) <= head_tokens:
            continue
        first = _boundary_at_or_before(starts, head_tokens - overlap, 0)
        for index, (s, e) in enumerate(_windows(len(spans), starts, first, size, overlap), start=1):
            out.append(_window_passage(p, index, p.text[spans[s][0]: spans[e - 1][1]], (s, e), "tail"))
    return out


def sliding_windows(passages: Sequence[Passage], tokenizer: Any, *, size: int, overlap: int) -> list[Passage]:
    """Windows of at most ``size`` tokens covering each whole passage (a short passage is one window)."""
    _check(size, overlap)
    out: list[Passage] = []
    for p in passages:
        spans, starts = _token_spans(p.text, tokenizer)
        if not spans:
            continue
        for index, (s, e) in enumerate(_windows(len(spans), starts, 0, size, overlap)):
            out.append(_window_passage(p, index, p.text[spans[s][0]: spans[e - 1][1]], (s, e), "sliding"))
    return out


def _check(size: int, overlap: int) -> None:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(f"window size must be > overlap >= 0, got size={size}, overlap={overlap}")


def parent_id(window_passage_id: str) -> str:
    return window_passage_id.split(WINDOW_SEPARATOR, 1)[0]


def windows_sha256(windows: Sequence[Passage]) -> str:
    """Identity of a derived window corpus (IDs and texts, in order): the key of its embedding cache."""
    h = hashlib.sha256()
    for w in windows:
        h.update(json.dumps([w.passage_id, w.text], ensure_ascii=False).encode("utf-8") + b"\n")
    return h.hexdigest()


__all__ = ["WINDOW_SEPARATOR", "parent_id", "sliding_windows", "tail_windows", "windows_sha256"]
