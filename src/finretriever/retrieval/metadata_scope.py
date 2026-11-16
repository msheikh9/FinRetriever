"""Metadata-aware scoping: which passages a question may retrieve, and how fiscal year shapes the ranking.

Built on a :class:`~finretriever.retrieval.query_understanding.QueryScope` and each passage's filing metadata
(ticker; fiscal year = year of the period end, :func:`~finretriever.retrieval.enrichment.passage_fiscal_year`).

**Company** (``company="filter"``): only the named companies' passages are eligible. A question that names no
company is not constrained (the parser never guesses).

**Fiscal year** (``year="filter"`` or ``"boost"``): the reporting years a question names (with parser ``qu-v2``,
not the target, maturity, historical or incidental years it also mentions), widened to the later filings
that repeat them. A 10-K reports its own year and up to two earlier years in its comparative columns, so a
year ``y`` allows filings ``y .. y + after``. A year read from a date or a plain number (not ``FY``/``fiscal``)
may also be a filing-date or forward-looking reference ("holders of record as of January 28, 2026" is in the
FY2025 10-K), so it also allows ``before_implicit`` earlier filings. On the 771 training and validation
questions (never the 100 human-written ones) this keeps the positive passage's filing year for 693 of 699
questions with a year, while allowing 43% of a company's fiscal years (window policy chosen there, recorded in
docs/retrieval_v2_metadata_hybrid.md).

* ``filter`` restricts eligibility to the allowed years; if fewer than ``min_eligible`` passages remain, the year
  constraint is dropped for that question and the fallback is recorded.
* ``boost`` keeps every passage but adds a metadata ranking (exact year, then later filings in the window, then
  the rest) as one more voter in rank fusion (:mod:`finretriever.retrieval.hybrid`).

**Cells** (``cells="interleave"``): a question naming several companies or several years is split into
company-year cells; each cell is ranked separately and the cell rankings are interleaved, so that every cell is
represented near the top instead of one company or year taking the whole top 10.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from finretriever.data.schemas import Passage
from finretriever.retrieval.enrichment import passage_fiscal_year
from finretriever.retrieval.query_understanding import QueryScope, YearMention

COMPANY_MODES = ("none", "filter")
YEAR_MODES = ("none", "filter", "boost")
CELL_MODES = ("off", "interleave")
SCOPE_SOURCES = ("parsed", "oracle")


@dataclass(frozen=True)
class ScopePolicy:
    company: str = "none"
    year: str = "none"
    after: int = 2
    before_explicit: int = 0
    before_implicit: int = 1
    min_eligible: int = 20
    cells: str = "off"
    source: str = "parsed"  # "oracle" uses a question set's annotated scope: a reference, never a system

    def __post_init__(self) -> None:
        for value, allowed, what in ((self.company, COMPANY_MODES, "company"), (self.year, YEAR_MODES, "year"),
                                     (self.cells, CELL_MODES, "cells"), (self.source, SCOPE_SOURCES, "source")):
            if value not in allowed:
                raise ValueError(f"scope {what} must be one of {list(allowed)}, got {value!r}")
        if min(self.after, self.before_explicit, self.before_implicit) < 0 or self.min_eligible < 1:
            raise ValueError("year windows must be >= 0 and min_eligible >= 1")

    @property
    def uses_year(self) -> bool:
        return self.year != "none"

    def describe(self) -> dict[str, Any]:
        return {"company": self.company, "year": self.year, "after": self.after,
                "before_explicit": self.before_explicit, "before_implicit": self.before_implicit,
                "min_eligible": self.min_eligible, "cells": self.cells, "source": self.source}


def mention_window(mention: YearMention, year: int, policy: ScopePolicy) -> set[int]:
    before = policy.before_explicit if mention.explicit_fiscal else policy.before_implicit
    return set(range(year - before, year + policy.after + 1))


def allowed_years(scope: QueryScope, policy: ScopePolicy) -> set[int]:
    """Every fiscal year whose filings may hold what ``scope`` asks about (empty: no year named)."""
    out: set[int] = set()
    for mention in scope.year_mentions:
        if not mention.reporting:  # a target, maturity, historical or incidental year (qu-v2) constrains nothing
            continue
        for year in mention.years:
            out |= mention_window(mention, year, policy)
    if not scope.year_mentions:  # an oracle scope has years but no mentions: treat them as explicit fiscal years
        for year in scope.years:
            out |= set(range(year - policy.before_explicit, year + policy.after + 1))
    return out


def year_windows(scope: QueryScope, policy: ScopePolicy) -> dict[int, set[int]]:
    """Each named year's own window (for per-year cells)."""
    out: dict[int, set[int]] = {}
    for mention in scope.year_mentions:
        if not mention.reporting:
            continue
        for year in mention.years:
            out.setdefault(year, set()).update(mention_window(mention, year, policy))
    for year in scope.years:
        out.setdefault(year, set(range(year - policy.before_explicit, year + policy.after + 1)))
    return dict(sorted(out.items()))


class CorpusMetadata:
    """Ticker and fiscal year of every passage, in corpus order."""

    def __init__(self, passages: Sequence[Passage]) -> None:
        self.passage_ids = [p.passage_id for p in passages]
        self.tickers = [p.ticker for p in passages]
        self.years = [passage_fiscal_year(p) for p in passages]
        self.position = {pid: i for i, pid in enumerate(self.passage_ids)}

    def __len__(self) -> int:
        return len(self.passage_ids)

    def select(self, tickers: Collection[str] | None = None, years: Collection[int] | None = None,
               exclude: Collection[int] = ()) -> list[int]:
        tickers = set(tickers) if tickers else None
        years = set(years) if years is not None else None
        excluded = set(exclude)
        return [i for i in range(len(self)) if i not in excluded
                and (tickers is None or self.tickers[i] in tickers)
                and (years is None or self.years[i] in years)]

    def year_tier(self, position: int, scope: QueryScope, allowed: set[int]) -> int:
        """0: a named fiscal year; 1: another allowed year (a later filing in the window); 2: any other year."""
        year = self.years[position]
        return 0 if year in scope.years else 1 if year in allowed else 2


@dataclass(frozen=True)
class Cell:
    label: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class ScopePlan:
    """What a policy decided for one question: the eligible passages, any cells, and why."""

    eligible: tuple[int, ...]
    cells: tuple[Cell, ...]
    allowed_years: tuple[int, ...]
    decisions: tuple[str, ...]
    company_filtered: bool
    year_filtered: bool

    def describe(self) -> dict[str, Any]:
        return {"eligible": len(self.eligible), "cells": [{"label": c.label, "eligible": len(c.positions)}
                                                          for c in self.cells],
                "allowed_years": list(self.allowed_years), "decisions": list(self.decisions),
                "company_filtered": self.company_filtered, "year_filtered": self.year_filtered}


def plan_scope(scope: QueryScope, policy: ScopePolicy, meta: CorpusMetadata,
               exclude: Collection[int] = ()) -> ScopePlan:
    """Apply ``policy`` to ``scope``: eligible corpus positions (minus ``exclude``), cells and the reasons."""
    decisions: list[str] = []
    tickers = list(scope.tickers) if policy.company == "filter" and scope.tickers else None
    if policy.company == "filter":
        decisions.append(f"company filter {tickers}" if tickers else "no company named: company filter not applied")
    base = meta.select(tickers=tickers, exclude=exclude)
    allowed = allowed_years(scope, policy) if policy.uses_year else set()
    eligible = base
    year_filtered = False
    if policy.year == "filter":
        if not allowed:
            decisions.append("no fiscal year named: year filter not applied")
        else:
            narrowed = [i for i in base if meta.years[i] in allowed]
            if len(narrowed) < policy.min_eligible:
                decisions.append(f"year filter fell back: {len(narrowed)} eligible passages < {policy.min_eligible}")
            else:
                eligible, year_filtered = narrowed, True
                decisions.append(f"year filter {sorted(allowed)}")
    elif policy.year == "boost":
        decisions.append(f"year boost {sorted(allowed)}" if allowed else "no fiscal year named: no year boost")

    cells: list[Cell] = []
    if policy.cells == "interleave":
        companies = list(tickers) if tickers else [None]
        windows = year_windows(scope, policy) if (policy.year == "filter" and year_filtered) else {}
        year_groups: list[tuple[str, set[int] | None]] = [(str(y), w) for y, w in windows.items()] or [("", None)]
        if len(companies) * len(year_groups) >= 2:
            for company in companies:
                for label, window in year_groups:
                    positions = tuple(i for i in eligible
                                      if (company is None or meta.tickers[i] == company)
                                      and (window is None or meta.years[i] in window))
                    if positions:
                        cells.append(Cell("/".join(x for x in (company or "", label) if x), positions))
            if len(cells) >= 2:
                decisions.append(f"interleaving {len(cells)} company-year cells")
            else:
                cells = []
    return ScopePlan(tuple(eligible), tuple(cells), tuple(sorted(allowed)), tuple(decisions),
                     company_filtered=tickers is not None, year_filtered=year_filtered)


def interleave(rankings: Sequence[Sequence[int]], limit: int) -> list[int]:
    """Round-robin merge of ranked lists (first of each, then second of each, ...), without duplicates."""
    out: list[int] = []
    seen: set[int] = set()
    depth = max((len(r) for r in rankings), default=0)
    for i in range(depth):
        for ranking in rankings:
            if i < len(ranking) and ranking[i] not in seen:
                seen.add(ranking[i])
                out.append(ranking[i])
                if len(out) == limit:
                    return out
    return out


__all__ = [
    "CELL_MODES",
    "COMPANY_MODES",
    "YEAR_MODES",
    "Cell",
    "CorpusMetadata",
    "ScopePlan",
    "ScopePolicy",
    "allowed_years",
    "interleave",
    "mention_window",
    "plan_scope",
    "year_windows",
]
