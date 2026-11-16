"""Evidence reuse below the overlap-group threshold: shared spans, sentences and numeric facts.

Overlap groups (``finretriever.training.overlap``) join passages sharing at least half their
5-word shingles. 10-K evidence is also reused in smaller pieces: consecutive chunks of a section
share 80 words, a risk factor or policy paragraph is carried into the next filing among new
text, and a table's prior-year column reappears in the next filing. Such a passage can sit in
another overlap group, or another split, than the passage it repeats. This module finds the
reuse, deterministically, at the level of evidence:

* **Spans.** Every run of ``span_words`` consecutive case-folded ``[a-z0-9]+`` words, hashed to
  64 bits (BLAKE2b). Shared spans measure how much of a passage another one repeats, including
  sentences edited by a word or two and lists without full sentences (exhibit indexes).
* **Sentence units.** Prose outside ``[TABLE]`` blocks is split into sentences (at ``.``,
  ``!`` or ``?`` before a capital, digit or bullet, and at bullets). A sentence of at least
  ``min_sentence_words`` words is a unit, keyed by its case-folded ``[a-z0-9]+`` word sequence.
* **Fact units.** Every significant number, in prose or tables (at least ``min_fact_digits``
  digits, or at least ``min_scaled_fact_digits`` followed by thousand/million/billion/trillion;
  never a bare year or a percentage), is a unit keyed by the company's ticker, the value
  (separators removed, parentheses as a minus sign, scale kept) and its label: the nearest
  word before it, within ``label_window_tokens`` tokens, that is not a number, a period word
  ("year", "ended", month names) or a stop word. "Total operating expenses | Year Ended December
  31, 2021: 444,943" gives ``(AMZN, 444943, expenses)`` in the FY2021 MD&A and in the FY2023
  income statement alike. A reused table row is a set of facts sharing a label. Facts only
  match within one company.
* **Generic units are ignored.** A span or sentence found in passages of more than
  ``max_text_companies`` companies (standard definitions, legal boilerplate), or any unit held
  by more than ``max_unit_passages`` passages, cannot relate two passages. Terminology on its own
  never forms a unit: a span or sentence needs at least 12 words, and a fact a significant number.

**Substantial reuse.** A passage's visible text substantially reuses a set of other passages
when the spans it shares with them cover at least ``substantial_word_share`` of its visible
words, or when it shares at least ``substantial_facts`` fact units with them and those
are at least ``substantial_fact_share`` of its fact units (most of a table's figures, such as
the prior-year columns of a comparative statement). A few shared figures among many do not
count: the same three totals quoted in a 44-figure note are not that note's evidence.
Consecutive chunks of a section share 80 words by construction (about a fifth of the visible
text); spans only an adjacent chunk shares do not count towards a passage's share, and adjacent
chunks are never linked. That overlap, and any reuse below the thresholds, is still caught for
each training pair's evidence (``EvidenceLocator``).

The index records, for every passage, the units of its tokenizer-visible text (what a training
query can be written from) and of its full text (what an evaluation judgment could rest on).

**One pair's evidence (``EvidenceLocator``, ``LOCATOR_VERSION``).** Fact units key a figure as
written, so "$57,411 million" and the table cell "57,411" are different units. The per-pair
check compares figures by amount instead (``Figure``, ``same_amount``): currency signs,
separators and spacing are dropped; a written unit ("million", "$914M") or the unit the passage
declares ("(In millions)") scales the digits; thousands, millions and billions stay distinct and
no rounding is equated; signs (a minus or accounting parentheses), percentages and dates are
kept apart from amounts. An exact figure match also needs the evidence's label. The index's
units and digest are unchanged by this comparison.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cached_property
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from finretriever.data.schemas import Passage
from finretriever.training.overlap import words
from finretriever.training.quotes import locate_quote

_TABLE_RE = re.compile(r"\[TABLE\].*?(?:\[/TABLE\]|$)", flags=re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9•])|\s*•\s*")
# A number ends in a digit (a trailing comma or full stop is punctuation, so "in 2025," is a year).
_TOKEN_RE = re.compile(r"\[/?TABLE\]|\$?\(?-?\d(?:[\d,]*\d)?(?:\.\d+)?\)?%?|[A-Za-z][A-Za-z'&\-]*")
_YEAR_RE = re.compile(r"(?:19|20)\d\d")
SCALES = {"thousand": "k", "million": "m", "billion": "b", "trillion": "t"}
PERIOD_WORDS = frozenset(
    "year years ended ending end fiscal quarter quarters month months period periods as of at dec december "
    "jan january feb february mar march apr april may jun june jul july aug august sep sept september oct "
    "october nov november through to date balance balances".split()
)
STOP_WORDS = frozenset(
    "the a an and or of in for on by with from was were is are be been being our we its their this that these "
    "those it which approximately respectively including include includes per each us u s".split()
)


class ReuseSettings(BaseModel):
    """Parameters of the evidence-reuse method; recorded with every artifact built from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_sentence_words: int = Field(default=12, ge=5)
    span_words: int = Field(default=12, ge=5)
    max_text_companies: int = Field(default=1, ge=1)
    max_unit_passages: int = Field(default=50, ge=2)
    min_fact_digits: int = Field(default=4, ge=2)
    min_scaled_fact_digits: int = Field(default=2, ge=1)
    label_window_tokens: int = Field(default=40, ge=1)
    substantial_word_share: float = Field(default=0.25, gt=0, le=1)
    substantial_facts: int = Field(default=3, ge=1)
    substantial_fact_share: float = Field(default=0.5, gt=0, le=1)


def _key(kind: str, text: str) -> str:
    return f"{kind}:{hashlib.blake2b(text.encode(), digest_size=8).hexdigest()}"


def sentences(text: str) -> list[str]:
    """The prose sentences of ``text`` (tables removed), in order."""
    prose = _TABLE_RE.sub(" ", text)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(prose) if s and s.strip()]


def sentence_units(text: str, settings: ReuseSettings) -> dict[str, int]:
    """Sentence-unit keys of ``text`` with their word counts."""
    out = {}
    for sentence in sentences(text):
        tokens = words(sentence)
        if len(tokens) >= settings.min_sentence_words:
            out[_key("s", " ".join(tokens))] = len(tokens)
    return out


def _value(token: str, scale: str | None) -> tuple[str, int] | None:
    """A number token's normalised value and digit count, or ``None`` for years and percentages."""
    if token.endswith("%"):
        return None
    bare = token.strip("$()")
    if _YEAR_RE.fullmatch(bare) and "$" not in token and "(" not in token:
        return None
    negative = token.startswith(("(", "$(")) or bare.startswith("-")
    digits = bare.lstrip("-").replace(",", "")
    if "." in digits:
        digits = digits.rstrip("0").rstrip(".")
    count = sum(c.isdigit() for c in digits)
    return ("-" if negative else "") + digits + (SCALES[scale] if scale else ""), count


@dataclass(frozen=True)
class Fact:
    value: str  # normalised: separators removed, a minus for parentheses, the scale's initial
    label: str
    text: str  # readable form: the label and the number as written

    def __str__(self) -> str:
        return self.text


def significant_values(text: str, settings: ReuseSettings) -> set[str]:
    """The normalised significant numbers of ``text`` (as fact values, whatever their label)."""
    return {f.value for f in facts(text, "", settings, need_label=False).values()}


def _label(lowered: Sequence[str], i: int, settings: ReuseSettings) -> str | None:
    """The fact label of the number at token ``i``: the nearest content word before it, or ``None``."""
    for j in range(i - 1, max(-1, i - 1 - settings.label_window_tokens), -1):
        word = lowered[j]
        if word in ("[table]", "[/table]"):
            break
        if not word[0].isalpha() or word in PERIOD_WORDS or word in STOP_WORDS or word in SCALES:
            continue
        return word
    return None


def facts(
    text: str, ticker: str, settings: ReuseSettings, *, need_label: bool = True
) -> dict[str, Fact]:
    """Fact units of ``text`` by key (``need_label=False`` also keeps numbers without a label)."""
    tokens = _TOKEN_RE.findall(text)
    lowered = [t.casefold() for t in tokens]
    out = {}
    for i, token in enumerate(tokens):
        if not token[0].isdigit() and not token.startswith(("$", "(", "-")):
            continue
        scale = lowered[i + 1] if i + 1 < len(tokens) and lowered[i + 1] in SCALES else None
        parsed = _value(token, scale)
        if parsed is None:
            continue
        value, digits = parsed
        if digits < (settings.min_scaled_fact_digits if scale else settings.min_fact_digits):
            continue
        label = _label(lowered, i, settings)
        if label is not None or not need_label:
            out[_key("f", f"{ticker}|{value}|{label}")] = Fact(
                value, label or "", f"{label or ''} {token}".strip()
            )
    return out


@dataclass(frozen=True)
class PassageUnits:
    sentences: Mapping[str, int]  # key -> words
    facts: Mapping[str, Fact]

    def keys(self) -> set[str]:
        return set(self.sentences) | set(self.facts)


def passage_units(text: str, ticker: str, settings: ReuseSettings) -> PassageUnits:
    return PassageUnits(sentence_units(text, settings), facts(text, ticker, settings))


@dataclass(frozen=True)
class Reuse:
    """What one passage's visible text shares with a set of other passages."""

    sentences: tuple[str, ...]  # shared sentence units
    facts: tuple[str, ...]  # shared fact units
    span_words: int  # visible words inside shared spans (an adjacent chunk's overlap excluded)
    visible_words: int
    visible_facts: int
    sources: tuple[str, ...]

    @property
    def word_share(self) -> float:
        return round(self.span_words / self.visible_words, 4) if self.visible_words else 0.0

    @property
    def fact_share(self) -> float:
        return round(len(self.facts) / self.visible_facts, 4) if self.visible_facts else 0.0

    def substantial(self, settings: ReuseSettings) -> bool:
        return self.word_share >= settings.substantial_word_share or (
            len(self.facts) >= settings.substantial_facts
            and self.fact_share >= settings.substantial_fact_share
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_sentences": len(self.sentences),
            "shared_span_words": self.span_words,
            "word_share": self.word_share,
            "shared_facts": len(self.facts),
            "fact_share": self.fact_share,
            "sources": list(self.sources),
        }


def span_hashes(tokens: Sequence[str], n: int) -> np.ndarray:
    """64-bit hashes of every ``n``-word window of ``tokens``, by position."""
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(" ".join(tokens[i : i + n]).encode(), digest_size=8).digest(),
                "little",
            )
            >> 1
            for i in range(len(tokens) - n + 1)
        ),
        dtype=np.int64,
        count=max(0, len(tokens) - n + 1),
    )


def _group_starts(sorted_values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])


@dataclass(frozen=True)
class ReuseIndex:
    """Units and spans of every passage's visible and full text, and which passages hold them."""

    settings: ReuseSettings
    order: tuple[str, ...]
    ticker: Mapping[str, str]
    visible_chars: Mapping[str, int]
    neighbours: Mapping[str, frozenset[str]]  # consecutive chunks of the same section
    visible: Mapping[str, PassageUnits]
    full: Mapping[str, PassageUnits]
    visible_words: Mapping[str, int]
    visible_spans: Mapping[str, np.ndarray]  # span hash at each visible word position
    full_spans: Mapping[str, np.ndarray]  # sorted distinct span hashes of the full text
    holders: Mapping[str, frozenset[str]]  # unit -> passages whose full text holds it
    visible_holders: Mapping[str, frozenset[str]]  # unit -> passages whose visible text holds it
    values: Mapping[tuple[str, str], frozenset[str]]  # (ticker, value) -> passages (full text)
    generic: frozenset[str]  # generic sentence and fact units
    generic_spans: np.ndarray  # sorted generic span hashes
    span_stats: Mapping[str, int]

    @classmethod
    def build(
        cls,
        passages: Sequence[Passage],
        visible_chars: Mapping[str, int],
        settings: ReuseSettings | None = None,
    ) -> ReuseIndex:
        """Index ``passages`` (corpus order); ``visible_chars`` gives each one's visible prefix."""
        settings = settings or ReuseSettings()
        visible, full, counts, vspans, fspans = {}, {}, {}, {}, {}
        holders: dict[str, set[str]] = defaultdict(set)
        visible_holders: dict[str, set[str]] = defaultdict(set)
        values: dict[tuple[str, str], set[str]] = defaultdict(set)
        for p in passages:
            shown = p.text[: visible_chars[p.passage_id]]
            visible[p.passage_id] = passage_units(shown, p.ticker, settings)
            full[p.passage_id] = passage_units(p.text, p.ticker, settings)
            shown_words = words(shown)
            counts[p.passage_id] = len(shown_words)
            vspans[p.passage_id] = span_hashes(shown_words, settings.span_words)
            fspans[p.passage_id] = np.unique(span_hashes(words(p.text), settings.span_words))
            for key in full[p.passage_id].keys():
                holders[key].add(p.passage_id)
            for key in visible[p.passage_id].keys():
                visible_holders[key].add(p.passage_id)
            for value in significant_values(p.text, settings):
                values[(p.ticker, value)].add(p.passage_id)
        ticker = {p.passage_id: p.ticker for p in passages}
        limit = settings.max_unit_passages
        generic = {
            key
            for key, owners in holders.items()
            if len(owners) > limit
            or (
                key.startswith("s:")
                and len({ticker[o] for o in owners}) > settings.max_text_companies
            )
        }
        generic_spans, span_stats = cls._generic_spans(passages, fspans, ticker, settings)
        sections: dict[tuple[str, Any], str] = {}
        for p in passages:
            sections[(p.section_id, p.metadata.get("chunk_index"))] = p.passage_id
        neighbours = {}
        for p in passages:
            chunk = p.metadata.get("chunk_index")
            near = set()
            if isinstance(chunk, int):
                for other in (chunk - 1, chunk + 1):
                    if (pid := sections.get((p.section_id, other))) is not None:
                        near.add(pid)
            neighbours[p.passage_id] = frozenset(near)
        return cls(
            settings,
            tuple(p.passage_id for p in passages),
            ticker,
            {p.passage_id: visible_chars[p.passage_id] for p in passages},
            neighbours,
            visible,
            full,
            counts,
            vspans,
            fspans,
            {k: frozenset(v) for k, v in holders.items()},
            {k: frozenset(v) for k, v in visible_holders.items()},
            {k: frozenset(v) for k, v in values.items()},
            frozenset(generic),
            generic_spans,
            span_stats,
        )

    @staticmethod
    def _generic_spans(
        passages: Sequence[Passage],
        spans: Mapping[str, np.ndarray],
        ticker: Mapping[str, str],
        settings: ReuseSettings,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Spans in more than ``max_unit_passages`` passages or more than ``max_text_companies`` companies."""
        codes = {t: i for i, t in enumerate(sorted(set(ticker.values())))}
        arrays = [spans[p.passage_id] for p in passages]
        if not arrays or sum(a.size for a in arrays) == 0:
            return np.empty(0, dtype=np.int64), {"spans": 0, "shared_spans": 0, "generic_spans": 0}
        hashes = np.concatenate(arrays)
        companies = np.repeat(
            np.array([codes[ticker[p.passage_id]] for p in passages], dtype=np.int64),
            [a.size for a in arrays],
        )
        order = np.lexsort((companies, hashes))
        hashes, companies = hashes[order], companies[order]
        starts = _group_starts(hashes)
        holders = np.diff(np.r_[starts, hashes.size])
        distinct = np.r_[True, (hashes[1:] != hashes[:-1]) | (companies[1:] != companies[:-1])]
        firms = np.add.reduceat(distinct.astype(np.int64), starts)
        unique = hashes[starts]
        generic = unique[
            (holders > settings.max_unit_passages) | (firms > settings.max_text_companies)
        ]
        stats = {
            "spans": int(unique.size),
            "shared_spans": int(
                (
                    (holders > 1)
                    & (holders <= settings.max_unit_passages)
                    & (firms <= settings.max_text_companies)
                ).sum()
            ),
            "generic_spans": int(generic.size),
        }
        return generic, stats

    @property
    def position(self) -> dict[str, int]:
        cached = self.__dict__.get("_position")
        if cached is None:
            cached = {pid: i for i, pid in enumerate(self.order)}
            object.__setattr__(self, "_position", cached)
        return cached

    def _distinct_visible_spans(self, pid: str) -> np.ndarray:
        cache = self.__dict__.setdefault("_distinct_visible", {})
        if pid not in cache:
            cache[pid] = np.unique(self.visible_spans[pid])
        return cache[pid]

    def reuse_against(
        self, candidates: Iterable[str], reference: Iterable[str], *, full_text: bool = True
    ) -> dict[str, Reuse]:
        """What each candidate's visible text shares with the ``reference`` passages' full (or visible) text.

        A passage never counts as its own source, and a shared span counts only if a reference
        passage other than the candidate's adjacent chunks holds it: the overlap consecutive chunks
        share by construction is left to the per-pair evidence check.
        """
        position = self.position
        ref_ids = sorted(set(reference), key=position.__getitem__)
        arrays = [
            self.full_spans[r] if full_text else self._distinct_visible_spans(r) for r in ref_ids
        ]
        hashes = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
        owners = np.repeat(np.arange(len(ref_ids), dtype=np.int64), [a.size for a in arrays])
        keep = ~np.isin(hashes, self.generic_spans)
        hashes, owners = hashes[keep], owners[keep]
        order = np.argsort(hashes, kind="stable")
        hashes, owners = hashes[order], owners[order]
        unit_index = self.holders if full_text else self.visible_holders
        ref_set = set(ref_ids)
        n = self.settings.span_words
        out = {}
        for pid in sorted(set(candidates), key=position.__getitem__):
            excluded = {pid} | self.neighbours.get(pid, frozenset())
            spans = self.visible_spans[pid]
            covered = np.zeros(self.visible_words[pid], dtype=bool)
            sources: set[str] = set()
            if spans.size and hashes.size:
                lo = np.searchsorted(hashes, spans, "left")
                hi = np.searchsorted(hashes, spans, "right")
                for i in np.flatnonzero(hi > lo):
                    found = {ref_ids[o] for o in owners[lo[i] : hi[i]]} - excluded
                    if found:
                        covered[i : i + n] = True
                        sources |= found
            units = self.visible[pid]
            found_s, found_f = [], []
            for key in sorted(units.keys()):
                if key in self.generic:
                    continue
                hit = (unit_index.get(key, frozenset()) & ref_set) - excluded
                if hit:
                    (found_s if key.startswith("s:") else found_f).append(key)
                    sources |= hit
            out[pid] = Reuse(
                tuple(found_s),
                tuple(found_f),
                int(covered.sum()),
                self.visible_words[pid],
                len(units.facts),
                tuple(sorted(sources, key=position.__getitem__)),
            )
        return out

    def shared(self, passage_id: str, others: Iterable[str], *, full_text: bool = True) -> Reuse:
        """What ``passage_id``'s visible text shares with ``others`` (their full or visible text)."""
        return self.reuse_against([passage_id], others, full_text=full_text)[passage_id]

    def links(self, candidates: Iterable[str]) -> dict[tuple[str, str], Reuse]:
        """Substantial reuse between pairs of ``candidates`` (visible text on both sides).

        Candidate pairs share at least one non-generic span or fact unit (consecutive chunks of a
        section excepted); a pair (a, b), a before b in corpus order, is returned when either
        passage substantially reuses the other, with the ``Reuse`` of a direction that does.
        """
        position = self.position
        allowed = sorted(set(candidates), key=position.__getitem__)
        index_of = {pid: i for i, pid in enumerate(allowed)}
        arrays = [self._distinct_visible_spans(pid) for pid in allowed]
        hashes = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
        owners = np.repeat(np.arange(len(allowed), dtype=np.int64), [a.size for a in arrays])
        keep = ~np.isin(hashes, self.generic_spans)
        hashes, owners = hashes[keep], owners[keep]
        order = np.lexsort((owners, hashes))
        hashes, owners = hashes[order], owners[order]
        pairs: set[tuple[int, int]] = set()
        if hashes.size:
            starts = _group_starts(hashes)
            sizes = np.diff(np.r_[starts, hashes.size])
            for size in range(2, self.settings.max_unit_passages + 1):
                group_starts = starts[sizes == size]
                if group_starts.size == 0:
                    continue
                block = owners[group_starts[:, None] + np.arange(size)]
                left, right = np.triu_indices(size, k=1)
                pairs.update(
                    zip(
                        block[:, left].ravel().tolist(),
                        block[:, right].ravel().tolist(),
                        strict=True,
                    )
                )
        for key, owners_of in self.visible_holders.items():
            if key in self.generic or key.startswith("s:"):
                continue
            inside = sorted(index_of[o] for o in owners_of if o in index_of)
            pairs.update((a, b) for i, a in enumerate(inside) for b in inside[i + 1 :])
        out = {}
        for i, j in sorted(pairs):
            a, b = allowed[i], allowed[j]
            if b in self.neighbours.get(a, frozenset()):
                continue
            ra, rb = self._pair_reuse(a, b), self._pair_reuse(b, a)
            if ra.substantial(self.settings) or rb.substantial(self.settings):
                out[(a, b)] = ra if ra.substantial(self.settings) else rb
        return out

    def _pair_reuse(self, pid: str, other: str) -> Reuse:
        """What ``pid``'s visible text shares with ``other``'s visible text."""
        theirs = np.setdiff1d(
            self._distinct_visible_spans(other), self.generic_spans, assume_unique=True
        )
        spans = self.visible_spans[pid]
        covered = np.zeros(self.visible_words[pid], dtype=bool)
        n = self.settings.span_words
        for i in np.flatnonzero(np.isin(spans, theirs)):
            covered[i : i + n] = True
        units, other_units = self.visible[pid], self.visible[other]
        s = tuple(
            sorted(
                k for k in units.sentences if k in other_units.sentences and k not in self.generic
            )
        )
        f = tuple(
            sorted(k for k in units.facts if k in other_units.facts and k not in self.generic)
        )
        return Reuse(s, f, int(covered.sum()), self.visible_words[pid], len(units.facts), (other,))

    def digest(self) -> str:
        """SHA-256 over every passage's units and spans and the generic sets: same inputs, same digest."""
        h = hashlib.sha256()
        for pid in self.order:
            h.update(pid.encode())
            for part in (self.visible[pid], self.full[pid]):
                h.update(",".join(sorted(part.keys())).encode())
            h.update(self.visible_spans[pid].tobytes())
            h.update(self.full_spans[pid].tobytes())
        h.update(",".join(sorted(self.generic)).encode())
        h.update(self.generic_spans.tobytes())
        return h.hexdigest()

    def stats(self) -> dict[str, Any]:
        kinds = Counter(k[0] for k in self.holders)
        generic = Counter(k[0] for k in self.generic)
        shared = Counter(
            k[0] for k, v in self.holders.items() if len(v) > 1 and k not in self.generic
        )
        return {
            "sentence_units": kinds.get("s", 0),
            "fact_units": kinds.get("f", 0),
            "generic_sentence_units": generic.get("s", 0),
            "generic_fact_units": generic.get("f", 0),
            "shared_sentence_units": shared.get("s", 0),
            "shared_fact_units": shared.get("f", 0),
            **self.span_stats,
        }


# --- figures: numbers compared by amount, for the evidence of one training pair --------------------

_SCALE_EXPONENTS = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
_SCALE_WORDS = {**_SCALE_EXPONENTS, **{f"{w}s": e for w, e in _SCALE_EXPONENTS.items()}, "bn": 9}
_GLUED_SCALE_RE = re.compile(r"(MM|BN|bn|K|M|B|T)\b")  # "$914M": only after a dollar figure
_GLUED_EXPONENTS = {"K": 3, "M": 6, "MM": 6, "B": 9, "BN": 9, "bn": 9, "T": 12}
_NUMBER_RE = re.compile(r"(\$?)(\(?)(-?)(\d(?:[\d,]*\d)?(?:\.\d+)?)(\)?)(%?)")  # a _TOKEN_RE number
_DECLARED_RE = re.compile(
    r"\bin\s+(thousands|millions|billions|trillions)\b"
    r"|\(\s*(?:\$\s*)?(thousands|millions|billions|trillions)\b",
    flags=re.IGNORECASE,
)
# A declaration that qualifies counts ("shares, which are reflected in thousands") is not the unit
# of the passage's amounts.
_COUNT_CONTEXT_RE = re.compile(
    r"\b(?:shares?|units?|employees|barrels|users|customers|members|square)\b", flags=re.IGNORECASE
)
_PERCENT_WORDS = frozenset({"percent", "percentage", "pct", "basis", "bps"})
_DATE_PART_RE = re.compile(r"[-/]\d{1,2}[-/]\d{1,4}\b")  # after the year of 2020-09-26, the 9 of 9/26/2020
_DATE_PART_BEFORE_RE = re.compile(r"(?:\d[-/]|/)$")  # the 26 and 2020 of 9/26/2020
_ANY_SCALE = (0, 3, 6, 9, 12)
MIN_PERCENT_DIGITS = 3  # "4%" is too common to identify a fact


def declared_scale(text: str) -> int | None:
    """The one unit a passage declares for figures written without one, as a power of ten.

    "(In millions)", "(dollars in millions)", "(Millions of dollars)" declare 6. ``None`` when the
    passage declares none, declares several, or qualifies a declaration by a count ("number of
    shares, which are reflected in thousands"): its unscaled figures then have an unknown unit.
    """
    found = set()
    for m in _DECLARED_RE.finditer(text):
        if _COUNT_CONTEXT_RE.search(text, max(0, m.start() - 40), m.start()):
            return None
        found.add(_SCALE_WORDS[(m.group(1) or m.group(2)).casefold()])
    return found.pop() if len(found) == 1 else None


@dataclass(frozen=True)
class Figure:
    """A significant number as written: its sign, digits, unit and label (see ``same_amount``)."""

    label: str  # the fact label (see ``facts``); "" when no content word precedes it
    negative: bool  # a minus sign or accounting parentheses
    magnitude: Decimal  # the digits, separators removed
    scale: int | None  # the power of ten written with it ("$57,411 million": 6), or None
    declared: int | None  # its passage's declared unit (``declared_scale``), for unscaled figures
    percent: bool  # a percentage, or basis points: never an amount
    key: str | None  # its fact unit in the reuse index, when it is one
    text: str  # as written

    def exponents(self) -> tuple[int, ...] | None:
        """The figure's unit as a power of ten, if known (written, else declared)."""
        if self.scale is not None:
            return (self.scale,)
        return None if self.declared is None else (self.declared,)

    @cached_property
    def known_amounts(self) -> frozenset[Decimal]:
        exponents = self.exponents()
        return frozenset() if exponents is None else frozenset(
            self.magnitude.scaleb(e) for e in exponents
        )

    @cached_property
    def possible_amounts(self) -> frozenset[Decimal]:
        """Its amount under its known unit, or under every unit when the unit is unknown."""
        if self.exponents() is not None:
            return self.known_amounts
        return frozenset(self.magnitude.scaleb(e) for e in _ANY_SCALE)

    def __str__(self) -> str:
        return self.text


def same_amount(a: Figure, b: Figure, *, signed: bool = True) -> bool:
    """Whether ``a`` and ``b`` state the same amount.

    Units count where they are known (written with the figure, or declared by its passage):
    57,411 million is 57.411 billion, but not 57,411 thousand, and 57.4 billion is not 57,411
    million. A figure of unknown unit takes whichever unit makes it equal, so a bare table cell
    "57,411" matches "$57,411 million"; two figures of unknown unit need the same digits. A
    percentage only ever equals a percentage. ``signed`` compares signs too.
    """
    if a.percent != b.percent or (signed and a.negative != b.negative):
        return False
    known_a, known_b = a.exponents() is not None, b.exponents() is not None
    if a.percent or not (known_a or known_b):
        return a.magnitude == b.magnitude
    return bool(a.possible_amounts & b.possible_amounts)


def figures(
    text: str, ticker: str, settings: ReuseSettings, *, declared: int | None = None
) -> list[Figure]:
    """The significant figures of ``text`` in order, with the labels ``facts`` gives them.

    Significant as for fact units: at least ``min_fact_digits`` digits, or ``min_scaled_fact_digits``
    with a unit written after it ("$550 million", "$914M"); a percentage needs
    ``MIN_PERCENT_DIGITS``. Bare years are never figures, and dates give no figure (their parts
    are too short). ``declared`` is the passage's unit for figures written without one.
    """
    found = [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    lowered = [t.casefold() for t, _, _ in found]
    out = []
    for i, (token, start, end) in enumerate(found):
        parts = _NUMBER_RE.fullmatch(token)
        if parts is None:
            continue
        dollar, opened, minus, number, closed, pct = parts.groups()
        if _DATE_PART_RE.match(text, end) or _DATE_PART_BEFORE_RE.search(text, max(0, start - 2), start):
            continue  # part of 2020-09-26 or 9/26/2020
        if _YEAR_RE.fullmatch(number) and not dollar and not (opened and closed):
            continue
        following = lowered[i + 1] if i + 1 < len(found) else ""
        percent = bool(pct) or following in _PERCENT_WORDS
        scale = None
        if not percent:
            if following in _SCALE_WORDS:
                scale = _SCALE_WORDS[following]
            elif dollar and (glued := _GLUED_SCALE_RE.match(text, end)):
                scale = _GLUED_EXPONENTS[glued.group(1)]
        before = text[max(0, start - 2) : start]
        negative = bool(opened or minus) or (
            before.rstrip().endswith("(") and bool(closed or text[end : end + 1] == ")")
        ) or (bool(dollar) and before.endswith(("-", "−")))
        digits = number.replace(",", "")
        if "." in digits:
            digits = digits.rstrip("0").rstrip(".")
        count = sum(c.isdigit() for c in digits)
        needed = (
            MIN_PERCENT_DIGITS
            if percent
            else settings.min_scaled_fact_digits
            if scale is not None
            else settings.min_fact_digits
        )
        if count < needed:
            continue
        label = _label(lowered, i, settings)
        key = None
        if not percent and label is not None:  # the unit ``facts`` records for this number
            index_scale = following if following in SCALES else None
            parsed = _value(token, index_scale)
            if parsed is not None and parsed[1] >= (
                settings.min_scaled_fact_digits if index_scale else settings.min_fact_digits
            ):
                key = _key("f", f"{ticker}|{parsed[0]}|{label}")
        out.append(
            Figure(
                label or "",
                negative,
                Decimal(number.replace(",", "")),
                scale,
                None if scale is not None else declared,
                percent,
                key,
                text[start:end] if scale is None else " ".join(t for t, _, _ in found[i : i + 2]),
            )
        )
    return out


def refers_to(answer: Figure, evidence: Figure) -> bool:
    """Whether an answer's figure is this evidence figure, as the evidence writes it.

    The evidence's declared unit is not used: "$57,411 million" refers to the cell "57,411" whatever
    the table's heading says, since the answer restates the cell. A negative answer figure needs
    a negative cell; an unsigned one may restate a negative ("a loss of $1,403 million").
    """
    as_written = replace(evidence, declared=None)
    return (not answer.negative or evidence.negative) and same_amount(
        answer, as_written, signed=False
    )


class FigureSet:
    """The figures of one passage, indexed for comparison by label and by amount."""

    def __init__(self, found: Sequence[Figure]) -> None:
        self.by_label: dict[str, list[Figure]] = defaultdict(list)
        for f in found:
            if f.label:
                self.by_label[f.label].append(f)
        amounts = [f for f in found if not f.percent]
        self.percents = {f.magnitude for f in found if f.percent}
        self.magnitudes = {f.magnitude for f in amounts}
        self.known = {a for f in amounts for a in f.known_amounts}
        self.any_amount = self.known | {a for f in amounts for a in f.possible_amounts}

    def holds_fact(self, figure: Figure) -> bool:
        """The same amount, with the same sign, under the same label."""
        return any(same_amount(figure, f) for f in self.by_label.get(figure.label, ()))

    def holds_figure(self, figure: Figure) -> bool:
        """The same digits or amount under the same label, whatever the sign or unit."""
        return any(
            f.percent == figure.percent
            and (f.magnitude == figure.magnitude or same_amount(figure, f, signed=False))
            for f in self.by_label.get(figure.label, ())
        )

    def holds_amount(self, figure: Figure) -> bool:
        """The same digits or amount under any label, whatever the sign."""
        if figure.percent:
            return figure.magnitude in self.percents
        if figure.magnitude in self.magnitudes:
            return True
        if figure.exponents() is None:
            return bool(figure.possible_amounts & self.known)
        return bool(figure.known_amounts & self.any_amount)


def value_strings(figure: Figure) -> set[str]:
    """Every value (as ``_value`` writes it, either sign) under which the reuse index may hold
    ``figure``'s amount: unscaled in any unit, or with any scale word."""
    if figure.percent:
        return set()
    out = set()
    for amount in figure.possible_amounts:
        for exponent in _ANY_SCALE:
            plain = format(amount.scaleb(-exponent).normalize(), "f")
            suffix = {3: "k", 6: "m", 9: "b", 12: "t"}.get(exponent, "")
            out.update({plain, f"-{plain}"})
            if suffix:
                out.update({plain + suffix, f"-{plain}{suffix}"})
    return out


# --- the evidence of one training pair ---------------------------------------------------------

SPAN_WORDS = 8  # an uncertain match: this many consecutive evidence words, but not the whole quote
EXACT_MATCHES = ("quote", "sentence", "fact")
UNCERTAIN_MATCHES = ("other_fact", "value", "span")
LOCATOR_VERSION = "evidence-reuse-v2"  # v2: figures compared by amount (sign, digits and unit)


@dataclass(frozen=True)
class Equivalent:
    """Another passage of the same company that holds a pair's evidence, and how.

    ``match`` describes the passage's full text, ``visible_match`` its tokenizer-visible text
    (``None`` if nothing matches there). Exact matches: the quote itself (``quote``, located as
    validator v3 does), a whole evidence sentence (``sentence``), or the evidence figure the answer
    states, as the same amount with the same sign under the same label (``fact``; ``same_amount``).
    Uncertain: an evidence figure under its label whatever its sign or unit (``other_fact``), the
    answer's figure under any label (``value``), or ``SPAN_WORDS`` consecutive evidence words
    (``span``); such a passage may answer the question, so it is never used as a negative either.
    A figure alone, without its label, is never an exact match.
    """

    passage_id: str
    match: str
    visible_match: str | None

    @property
    def exact(self) -> bool:
        return self.match in EXACT_MATCHES

    @property
    def visible_exact(self) -> bool:
        return self.visible_match in EXACT_MATCHES


class EvidenceLocator:
    """Finds every passage of the same company that holds a training pair's evidence.

    Candidates come from an inverted index of each company's words (the evidence's two rarest
    content words), the holders of the evidence's units and of every value under which the index
    may hold the answer's or the evidence's figures, so the whole corpus is never scanned. Figures
    are compared by amount (``Figure``, ``same_amount``): "$57,411 million" in an answer is the
    cell "57,411" of a table, and a blocked passage holding that cell under the evidence's label
    holds the evidence. Built from corpus text only: no evaluation judgment is read.
    """

    version = LOCATOR_VERSION

    def __init__(self, passages: Mapping[str, Passage], index: ReuseIndex) -> None:
        self.passages = passages
        self.index = index
        self._postings: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for pid in index.order:
            for word in set(words(passages[pid].text)):
                self._postings[index.ticker[pid]][word].add(pid)
        self._declared: dict[str, int | None] = {}
        self._figures: dict[tuple[str, bool], FigureSet] = {}

    def declared(self, passage_id: str) -> int | None:
        """The unit ``passage_id``'s full text declares (``declared_scale``)."""
        if passage_id not in self._declared:
            self._declared[passage_id] = declared_scale(self.passages[passage_id].text)
        return self._declared[passage_id]

    def figures_of(self, passage_id: str, visible: bool) -> FigureSet:
        """The figures of a passage's full or visible text (the full text declares the unit)."""
        key = (passage_id, visible)
        if key not in self._figures:
            text = self.passages[passage_id].text
            if visible:
                text = text[: self.index.visible_chars[passage_id]]
            self._figures[key] = FigureSet(
                figures(
                    text,
                    self.index.ticker[passage_id],
                    self.index.settings,
                    declared=self.declared(passage_id),
                )
            )
        return self._figures[key]

    def amount_holders(self, figure: Figure, ticker: str) -> set[str]:
        """Passages of ``ticker`` that may hold ``figure``'s amount (candidates for ``FigureSet.holds_amount``):
        the reuse index's value postings, or for a percentage, which the index does not hold, the passages
        holding all of its digit groups as words."""
        if not figure.percent:
            return {pid for v in value_strings(figure) for pid in self.index.values.get((ticker, v), ())}
        postings = self._postings.get(ticker, {})
        groups = [g for g in format(figure.magnitude, "f").split(".") if g]
        found: set[str] | None = None
        for group in groups:
            holders = postings.get(group, set())
            found = set(holders) if found is None else found & holders
        return found or set()

    def _candidates(
        self, evidence: str, ticker: str, units: PassageUnits, values: set[str]
    ) -> set[str]:
        postings = self._postings.get(ticker, {})
        content = sorted(
            {w for w in words(evidence) if len(w) > 2 and w not in STOP_WORDS and w in postings},
            key=lambda w: (len(postings[w]), w),
        )
        found: set[str] = set()
        if content:
            found = set(postings[content[0]])
            for word in content[1:2]:
                found &= postings[word]
        for key in units.keys():
            if key not in self.index.generic:
                found |= self.index.holders.get(key, frozenset())
        for value in values:
            found |= self.index.values.get((ticker, value), frozenset())
        return {pid for pid in found if self.index.ticker[pid] == ticker}

    def _match(
        self,
        pid: str,
        visible: bool,
        evidence: str,
        units: PassageUnits,
        figures: _EvidenceFigures,
        spans: set[tuple[str, ...]],
    ) -> str | None:
        text = self.passages[pid].text
        if visible:
            text = text[: self.index.visible_chars[pid]]
        held = (self.index.visible if visible else self.index.full)[pid]
        generic = self.index.generic
        if locate_quote(evidence, text) is not None:
            return "quote"
        if any(k in held.sentences for k in units.sentences if k not in generic):
            return "sentence"
        found = self.figures_of(pid, visible)
        if any(found.holds_fact(f) for f in figures.answer_facts):
            return "fact"
        if any(k in held.facts for k in units.facts if k not in generic) or any(
            found.holds_figure(f) for f in figures.evidence if f.label and f.key not in generic
        ):
            return "other_fact"
        if any(found.holds_amount(f) for f in figures.answer):
            return "value"
        tokens = words(text)
        n = SPAN_WORDS
        if spans and any(tuple(tokens[i : i + n]) in spans for i in range(len(tokens) - n + 1)):
            return "span"
        return None

    def evidence_figures(
        self, evidence: str, answer: str, ticker: str, positive: str | None = None
    ) -> _EvidenceFigures:
        """The figures of a pair's evidence (in its positive's declared unit) and answer, and the
        evidence figures the answer states (``refers_to``) under a label."""
        settings = self.index.settings
        declared = self.declared(positive) if positive in self.passages else None
        found = figures(evidence, ticker, settings, declared=declared)
        stated = figures(answer, ticker, settings)
        facts = tuple(e for e in found if e.label and any(refers_to(a, e) for a in stated))
        return _EvidenceFigures(tuple(found), tuple(stated), facts)

    def locate(self, evidence: str, answer: str, ticker: str, positive: str) -> list[Equivalent]:
        """Every other passage of ``ticker`` holding the evidence, in corpus order."""
        settings = self.index.settings
        units = passage_units(evidence, ticker, settings)
        evidence_figures = self.evidence_figures(evidence, answer, ticker, positive)
        values = {
            v for f in (*evidence_figures.evidence, *evidence_figures.answer) for v in value_strings(f)
        }
        tokens = words(evidence)
        n = SPAN_WORDS
        spans = {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
        out = []
        candidates = self._candidates(evidence, ticker, units, values) - {positive}
        for pid in sorted(candidates, key=self.index.position.__getitem__):
            args = (evidence, units, evidence_figures, spans)
            match = self._match(pid, False, *args)
            if match is not None:
                out.append(Equivalent(pid, match, self._match(pid, True, *args)))
        return out


@dataclass(frozen=True)
class _EvidenceFigures:
    evidence: tuple[Figure, ...]
    answer: tuple[Figure, ...]
    answer_facts: tuple[Figure, ...]  # evidence figures the answer states, under a label


__all__ = [
    "EXACT_MATCHES",
    "LOCATOR_VERSION",
    "MIN_PERCENT_DIGITS",
    "SPAN_WORDS",
    "UNCERTAIN_MATCHES",
    "Equivalent",
    "EvidenceLocator",
    "Fact",
    "Figure",
    "FigureSet",
    "PERIOD_WORDS",
    "STOP_WORDS",
    "PassageUnits",
    "Reuse",
    "ReuseIndex",
    "ReuseSettings",
    "declared_scale",
    "facts",
    "figures",
    "passage_units",
    "refers_to",
    "same_amount",
    "sentence_units",
    "sentences",
    "significant_values",
    "value_strings",
]
