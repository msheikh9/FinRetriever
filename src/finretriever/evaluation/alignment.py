"""Retrieval alignment diagnostic: does a ranking stay inside a question's company and fiscal-year scope?

Label-free. It reads only a question's own scope (its tickers and fiscal years) and each retrieved passage's
filing metadata (ticker, fiscal year from the period end, table flag). It says nothing about relevance: an
in-scope passage can be useless, and an out-of-scope one can hold the answer. In particular a later 10-K
repeats earlier years in its comparative columns (rubric P3), so a later filing within
:data:`COMPARATIVE_WINDOW` years is counted apart, never as an error.

For each retrieved passage (top ``k``):

* ``company``: its ticker is one of the question's tickers; otherwise it is a cross-company retrieval;
* year relation, for a correct-company passage: ``exact`` (its fiscal year is one the question asks about),
  ``later_comparative`` (a later filing, within the window, that may carry the asked year as a prior-year
  column) or ``other_year`` (a cross-year retrieval);
* ``table``: the passage holds a rendered financial table (corpus flag ``contains_table``).

A question's scope cells are its tickers times its fiscal years; a cell is covered when the top ``k`` holds a
passage of that company and exact year. Multi-company and multi-year questions need every cell.

The answer-figure check (:func:`answer_figures`, :func:`figure_in_text`) looks for an answer's figures (for
example ``6,686`` from "$6,686 million") in retrieved text: a label-free sign that the evidence, not only the
topic, was retrieved. Years and short numbers are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

COMPARATIVE_WINDOW = 2  # a 10-K's statements cover the current year and up to two earlier years
YEAR_RELATIONS = ("exact", "later_comparative", "other_year")


@dataclass(frozen=True)
class HitInfo:
    """What the diagnostic reads from a retrieved passage."""

    ticker: str
    fiscal_year: int
    table: bool


@dataclass(frozen=True)
class Scope:
    """A question's company and fiscal-year scope."""

    tickers: frozenset[str]
    years: frozenset[int]

    @classmethod
    def of(cls, tickers: Iterable[str], years: Iterable[int | str] | None) -> Scope:
        tickers = frozenset(str(t).upper() for t in tickers)
        if not tickers:
            raise ValueError("a scope needs at least one ticker")
        return cls(tickers, frozenset(int(y) for y in (years or ())))

    @property
    def cells(self) -> list[tuple[str, int]]:
        return [(t, y) for t in sorted(self.tickers) for y in sorted(self.years)]


def fiscal_year(metadata: Mapping[str, Any], filing_date: str) -> int:
    """A filing's fiscal year: the year of its period end (``report_date``), else of its filing date.

    The same rule as ``eval_set.candidate_generator.fiscal_year_of``, which scoped the gold-set pools."""
    return int(str(metadata.get("report_date") or filing_date)[:4])


def year_relation(year: int, years: Collection[int], window: int = COMPARATIVE_WINDOW) -> str:
    """``exact``, ``later_comparative`` (a later filing within ``window`` years) or ``other_year``."""
    if not years:
        raise ValueError("no fiscal years in scope")
    if year in years:
        return "exact"
    if any(0 < year - y <= window for y in years):
        return "later_comparative"
    return "other_year"


def classify(hit: HitInfo, scope: Scope) -> dict[str, Any]:
    company = hit.ticker.upper() in scope.tickers
    relation = year_relation(hit.fiscal_year, scope.years) if company and scope.years else None
    return {"company": company, "year": relation, "table": hit.table, "in_scope": relation == "exact"}


def _first(flags: Sequence[bool]) -> int | None:
    return next((i for i, flag in enumerate(flags, start=1) if flag), None)


def question_alignment(hits: Sequence[HitInfo], scope: Scope, k: int = 10) -> dict[str, Any]:
    """Shares of the top ``k`` by company, year relation and table flag, plus scope-cell coverage."""
    top = list(hits[:k])
    if not top:
        raise ValueError("no retrieved passages")
    rows = [classify(h, scope) for h in top]
    n = len(rows)

    def share(flags: Iterable[bool]) -> float:
        return sum(flags) / n

    covered = {(h.ticker.upper(), h.fiscal_year) for h, r in zip(top, rows, strict=True) if r["in_scope"]}
    within = {
        (h.ticker.upper(), y)
        for h, r in zip(top, rows, strict=True)
        if r["company"] and r["year"] in ("exact", "later_comparative")
        for y in scope.years
        if 0 <= h.fiscal_year - y <= COMPARATIVE_WINDOW
    }
    cells = scope.cells
    return {
        "k": n,
        "company_share": share(r["company"] for r in rows),
        "cross_company_share": share(not r["company"] for r in rows),
        "exact_year_share": share(r["year"] == "exact" for r in rows),
        "later_comparative_share": share(r["year"] == "later_comparative" for r in rows),
        "cross_year_share": share(r["year"] == "other_year" for r in rows),
        "table_share": share(r["table"] for r in rows),
        "company_hit": any(r["company"] for r in rows),
        "in_scope_hit": any(r["in_scope"] for r in rows),
        "top1_company": rows[0]["company"],
        "top1_in_scope": rows[0]["in_scope"],
        "top1_year": rows[0]["year"],
        "first_company_rank": _first([r["company"] for r in rows]),
        "first_in_scope_rank": _first([r["in_scope"] for r in rows]),
        "companies_covered": len({h.ticker.upper() for h, r in zip(top, rows, strict=True) if r["company"]})
        / len(scope.tickers),
        "cells": len(cells),
        "cells_covered_exact": sum(c in covered for c in cells) / len(cells) if cells else None,
        "all_cells_covered_exact": all(c in covered for c in cells) if cells else None,
        "cells_covered_with_comparatives": sum(c in within for c in cells) / len(cells) if cells else None,
    }


_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|(?<![\w.,])\d+\.\d+(?![\d.])|(?<![\w.,])\d{3,}(?![\d.,])")


def answer_figures(answer: str, min_digits: int = 3) -> list[str]:
    """The figures in an answer worth searching for: at least ``min_digits`` digits, no bare year (1990-2039)."""
    figures = []
    for match in _NUMBER.finditer(answer or ""):
        token = match.group(0)
        digits = re.sub(r"\D", "", token)
        if len(digits) < min_digits:
            continue
        if re.fullmatch(r"(19[9]\d|20[0-3]\d)", token):
            continue
        if token not in figures:
            figures.append(token)
    return figures


def figure_in_text(figure: str, text: str) -> bool:
    """``figure`` as a whole number in ``text`` (not part of a longer number)."""
    return re.search(rf"(?<![\d,.]){re.escape(figure)}(?![\d]|[.,]\d)", text) is not None


__all__ = [
    "COMPARATIVE_WINDOW",
    "YEAR_RELATIONS",
    "HitInfo",
    "Scope",
    "answer_figures",
    "classify",
    "figure_in_text",
    "fiscal_year",
    "question_alignment",
    "year_relation",
]
