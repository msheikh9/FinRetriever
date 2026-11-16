"""Versioned sampling plans: which approved passages a generation pilot asks about, and in what order.

A plan is a JSON file ``<prepared>/plans/<split>.<name>.json`` written once and never changed: a
second write with different content is refused, so a pilot's selection stays auditable after the
selection code or the source-quality filter changes. ``generate --plan <name>`` sends its passages
in plan order; generation records stay where they are, so a passage done under one plan is
skipped under another.

Generation under a plan can be closed (``stop-plan``): a stop record
``<prepared>/plans/<split>.<name>.stopped.json``, written once beside the plan, names the last
position with a generation record and binds the record lines at that moment. ``generate --plan``
refuses a stopped plan before reading any credential, so no request is sent for it.

Methods:

* ``company_round_robin``: the original pilot selection (``query_generation.sample_sources``):
  one passage per company in turn, each company's passages in seeded hash order, no source filter.
* ``company_year_round_robin``: one passage per company in turn; each company's next passage comes
  from the fiscal year it has used least, then the year least used across the plan, then seeded
  year order, with each year's passages in seeded hash order. Only passages the source-quality
  filter keeps are eligible.
* ``balanced`` (``balanced_selection``): company quotas as equal as the sources allow, with no
  company above ``max_company_share`` of the plan; at most one passage per overlap group and per
  split unit (groups joined by evidence reuse), so no two planned passages substantially share
  evidence; overlap groups listed in ``exclude_groups`` (for example those already covered by
  carried-over pairs) are skipped. Within a company each pick takes the passage kind (table or
  narrative) it has used least, then its least-used fiscal year, then the section family
  (``section_family``) least used across the plan relative to ``FAMILY_WEIGHTS`` (statements and
  MD&A 3, risk 2, business and other 1), then the year least used across the plan, then seeded
  order. Picks alternate
  between companies, so a run stopped early is still balanced. Question types are only known after
  generation; passage kinds and section families are what a plan can balance beforehand.
* ``balanced_years``: ``balanced`` with fiscal years first. Within a company each pick takes the
  fiscal year it has used least and, among those, the year its fewest remaining split units still
  reach, so a year only one unit can supply is taken before that unit is spent on another year;
  then passage kind, section family, plan-wide year and seeded order as in ``balanced``. The
  greedy ``balanced`` order can leave a company's years uncovered when one unit spans several
  years (Apple's validation sources: 5 of 7 years under ``balanced``, all 7 here). The plan records
  each company's covered and available years (``coverage``).

Options of the balanced methods, recorded in the plan only when used:

* ``one_per="group"``: at most one passage per overlap group, but several groups of one split unit
  may be planned (the default, ``"unit"``, allows one passage per unit). Groups of one unit are
  linked by evidence reuse, so their questions are less independent; a validation set records the
  equivalents of each question;
* ``scores`` (passage ID -> a score, higher preferred) with ``min_score``: passages scoring below
  ``min_score`` are not eligible, and among passages the balance keys tie on, the higher score is
  taken before seeded order. The validation plans score each passage by the share of its visible
  sentences that no passage outside its split holds (``validation_set.isolated_evidence_share``),
  so questions are likelier to pass the validation set's evidence rule;
* ``max_per_company_year``: at most this many passages per company and fiscal year (one 10-K
  each in this corpus, so per filing), so a few filings with many isolated passages cannot fill a
  company's quota;
* ``max_per_unit`` (with ``one_per="group"``): at most this many passages per split unit, limiting
  questions linked by evidence reuse (in this corpus, mostly one note or table across years).

* ``prior_units`` and ``prior_company_years``: passages already generated under an earlier plan,
  counted toward ``max_per_unit`` and ``max_per_company_year`` (their overlap groups are excluded by
  the caller), so a later plan keeps the caps across plans;
* ``score_priority``: the score, in steps of ``SCORE_PRIORITY_STEP``, ranks right after the fiscal-year
  balance (``balanced_years``) instead of being the last tie-break, so a company's best-scoring
  passages in its least-used year come first and a plan stopped early holds the likeliest yield.

With caps, a company's quota counts only the passages it can take under them (a maximum flow over
its split units, overlap groups and fiscal years); a plan that cannot reach its size is refused,
never shortened.

All are deterministic for a given preparation, filter, seed, size and scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from finretriever.training.query_generation import SourcePassage, sample_sources
from finretriever.training.source_quality import (
    SourceQualityResult,
    SourceQualitySettings,
    item_kind,
)

PLAN_FORMAT_VERSION = 1
SCORE_PRIORITY_STEP = 0.1  # score_priority compares scores in these steps, so kind and section still balance
PLANS_DIRNAME = "plans"
METHODS: tuple[str, ...] = ("company_round_robin", "company_year_round_robin", "balanced", "balanced_years")
Method = Literal["company_round_robin", "company_year_round_robin", "balanced", "balanced_years"]
BALANCED_METHODS = ("balanced", "balanced_years")
_FAMILY_TITLES = (
    ("risk", ("risk",)),
    (
        "statements",
        (
            "financial statements",
            "note ",
            "notes to",
            "consolidated balance",
            "statements of",
            "selected",
            "supplemental information",
            "supplementary",
        ),
    ),
    (
        "mdna",
        (
            "management's discussion",
            "results of operations",
            "liquidity",
            "capital resources",
            "segment",
        ),
    ),
    ("business", ("business", "properties", "human capital")),
)
_FAMILY_ITEMS = {
    "1A": "risk",
    "7A": "risk",
    "7": "mdna",
    "6": "statements",
    "8": "statements",
    "1": "business",
    "2": "business",
}
# How much of a plan each section family should get, relative to the others: financial statements and MD&A
# carry most figures and analysis, risk factors the disclosures, business and other sections the least.
FAMILY_WEIGHTS = {"statements": 3, "mdna": 3, "risk": 2, "business": 1, "other": 1}


class PlanError(ValueError):
    """A plan cannot be made, read or used; nothing is written."""


def _order(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def company_year_round_robin(
    passages: Sequence[SourcePassage], n: int, seed: int
) -> list[SourcePassage]:
    """``n`` passages, one per company in turn, spread over fiscal years.

    Each company's next passage comes from the year it has used least, then the year least used
    across the plan so far, then seeded year order; within a year, seeded passage order.
    """
    tree: dict[str, dict[str, list[SourcePassage]]] = {}
    for p in passages:
        tree.setdefault(p.ticker, {}).setdefault(p.fiscal_year, []).append(p)
    for years in tree.values():
        for queue in years.values():
            queue.sort(key=lambda p: _order(seed, p.passage_id))
    used: dict[str, Counter[str]] = {ticker: Counter() for ticker in tree}
    overall: Counter[str] = Counter()
    chosen: list[SourcePassage] = []
    while len(chosen) < n and any(q for years in tree.values() for q in years.values()):
        for ticker in sorted(tree):
            open_years = [y for y, queue in tree[ticker].items() if queue]
            if len(chosen) >= n or not open_years:
                continue
            year = min(
                open_years,
                key=lambda y, t=ticker: (used[t][y], overall[y], _order(seed, f"{t}:{y}")),
            )
            chosen.append(tree[ticker][year].pop(0))
            used[ticker][year] += 1
            overall[year] += 1
    return chosen


def passage_kind(passage: SourcePassage) -> str:
    """``table`` if the visible text holds a financial table, else ``narrative``."""
    return "table" if "[TABLE]" in passage.visible_text else "narrative"


def section_family(passage: SourcePassage) -> str:
    """``risk`` (1A, 7A), ``mdna`` (7), ``statements`` (6, 8, notes, supplemental financial data),
    ``business`` (1, 2) or ``other``: from the 10-K item number, or from the section title when the
    filing has none (JPMorgan's notes and MD&A sections)."""
    if passage.item in _FAMILY_ITEMS:
        return _FAMILY_ITEMS[passage.item]
    title = passage.section.casefold()
    for family, needles in _FAMILY_TITLES:
        if any(n in title for n in needles):
            return family
    return "other"


def _max_flow(edges: Mapping[tuple[Any, Any], float], source: Any, sink: Any) -> int:
    """The maximum flow through a small directed graph (shortest augmenting paths)."""
    residual: dict[Any, dict[Any, float]] = {}
    for (a, b), capacity in edges.items():
        residual.setdefault(a, {})[b] = residual.get(a, {}).get(b, 0) + capacity
        residual.setdefault(b, {}).setdefault(a, 0)
    flow = 0
    while True:
        parent = {source: None}
        queue = [source]
        while queue and sink not in parent:
            node = queue.pop(0)
            for nxt, capacity in residual.get(node, {}).items():
                if capacity > 0 and nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if sink not in parent:
            return flow
        path = []
        node = sink
        while parent[node] is not None:
            path.append((parent[node], node))
            node = parent[node]
        push = min(residual[a][b] for a, b in path)
        for a, b in path:
            residual[a][b] -= push
            residual[b][a] += push
        flow += int(push)


def _capacity(
    pool: Sequence[SourcePassage],
    slot_of: Mapping[str, str],
    unit_of: Mapping[str, str],
    year_room: Callable[[str], int],
    unit_room: Callable[[str], int],
) -> int:
    """How many passages of ``pool`` (one company's) fit one per slot, with at most ``year_room(year)`` per
    fiscal year and ``unit_room(unit)`` per split unit."""
    edges: dict[tuple[Any, Any], float] = {}
    for p in pool:
        unit, slot = unit_of[p.passage_id], slot_of[p.passage_id]
        edges[("source", ("unit", unit))] = unit_room(unit)
        edges[(("unit", unit), ("slot", slot))] = 1
        edges[(("slot", slot), ("year", p.fiscal_year))] = 1
        edges[(("year", p.fiscal_year), "sink")] = year_room(p.fiscal_year)
    return _max_flow(edges, "source", "sink")


def balanced_selection(
    passages: Sequence[SourcePassage],
    n: int,
    seed: int,
    *,
    unit_of: Mapping[str, str],
    max_company_share: float,
    exclude_groups: Collection[str] = (),
    years_first: bool = False,
    one_per: Literal["unit", "group"] = "unit",
    scores: Mapping[str, float] | None = None,
    max_per_company_year: int | None = None,
    max_per_unit: int | None = None,
    prior_units: Mapping[str, int] | None = None,
    prior_company_years: Mapping[tuple[str, str], int] | None = None,
    score_priority: bool = False,
) -> list[SourcePassage]:
    """``n`` passages under the ``balanced`` method (``balanced_years`` with ``years_first``); see the
    module docstring."""
    if not 0 < max_company_share <= 1:
        raise PlanError("max_company_share must be in (0, 1]")
    if max_per_unit is not None and one_per != "group":
        raise PlanError("max_per_unit applies with one_per='group' (one per unit is the default)")
    if any(c is not None and c < 1 for c in (max_per_company_year, max_per_unit)):
        raise PlanError("max_per_company_year and max_per_unit must be at least 1")
    if score_priority and scores is None:
        raise PlanError("score_priority needs scores")
    prior_unit = Counter(prior_units or {})
    prior_year = Counter(prior_company_years or {})
    split_unit = dict(unit_of)
    if one_per == "group":  # each group is its own slot: groups of one unit may all be planned
        unit_of = {p.passage_id: p.group_id for p in passages}
    cap = int(max_company_share * n)
    pools: dict[str, list[SourcePassage]] = {}
    for p in passages:
        if p.group_id not in exclude_groups:
            pools.setdefault(p.ticker, []).append(p)
    for pool in pools.values():
        pool.sort(key=lambda p: _order(seed, p.passage_id))
    # A group or unit with passages of several companies (shared boilerplate) can serve only one of them, so a
    # company's quota counts only the ones no other company draws from: it never depends on a shared one.
    serves = Counter(slot for pool in pools.values() for slot in {unit_of[p.passage_id] for p in pool})
    # Under a unit cap, a unit several companies draw from has one cap for all of them, so it counts for none.
    unit_serves = Counter(u for pool in pools.values() for u in {split_unit[p.passage_id] for p in pool})
    exclusive = {
        t: [
            p
            for p in pool
            if serves[unit_of[p.passage_id]] == 1
            and (max_per_unit is None or unit_serves[split_unit[p.passage_id]] == 1)
        ]
        for t, pool in pools.items()
    }
    capped = max_per_company_year is not None or max_per_unit is not None
    year_cap, unit_cap = max_per_company_year or len(passages), max_per_unit or len(passages)
    distinct = {
        t: _capacity(
            exclusive[t],
            unit_of,
            split_unit,
            lambda y, t=t: max(year_cap - prior_year[(t, y)], 0),
            lambda u: max(unit_cap - prior_unit[u], 0),
        )
        for t in pools
    }
    caps = "".join(
        (
            f" and at most {max_per_company_year} per company and fiscal year" if max_per_company_year else "",
            f" and at most {max_per_unit} per split unit" if max_per_unit else "",
        )
    )
    tickers = sorted(pools, key=lambda t: _order(seed, t))
    quota = dict.fromkeys(tickers, 0)
    while (
        sum(quota.values()) < n
    ):  # one more for each company in seeded order, within its cap and sources
        open_ = [t for t in tickers if quota[t] < min(cap, distinct[t])]
        if not open_:
            raise PlanError(
                f"only {sum(quota.values())} passages fit {n} with at most {max_company_share:.0%} per company and "
                f"one passage per {'overlap group' if one_per == 'group' else 'split unit'}{caps}"
            )
        for t in open_:
            if sum(quota.values()) < n:
                quota[t] += 1
    used_groups: set[str] = set()
    used_units: set[str] = set()
    per_unit: Counter[str] = Counter(prior_unit)
    kinds: dict[str, Counter[str]] = {t: Counter() for t in tickers}
    years: dict[str, Counter[str]] = {t: Counter() for t in tickers}
    families: Counter[str] = Counter()
    all_years: Counter[str] = Counter()
    chosen: list[SourcePassage] = []

    def leaves_room(t: str, pick: SourcePassage) -> bool:
        """Whether, after ``pick``, the caps still let ``t`` reach its quota from its exclusive passages."""
        need = quota[t] - sum(1 for p in chosen if p.ticker == t) - 1
        rest = [
            p
            for p in exclusive[t]
            if p.group_id not in used_groups | {pick.group_id}
            and unit_of[p.passage_id] not in used_units | {unit_of[pick.passage_id]}
        ]
        return need <= 0 or _capacity(
            rest,
            unit_of,
            split_unit,
            lambda y: max(year_cap - prior_year[(t, y)] - years[t][y] - (y == pick.fiscal_year), 0),
            lambda u: unit_cap - per_unit[u] - (u == split_unit[pick.passage_id]),
        ) >= need

    while len(chosen) < n:
        progressed = False
        for t in sorted(tickers):
            if sum(1 for p in chosen if p.ticker == t) >= quota[t]:
                continue
            candidates = [
                p
                for p in pools[t]
                if p.group_id not in used_groups
                and unit_of[p.passage_id] not in used_units
                and (
                    max_per_company_year is None
                    or prior_year[(t, p.fiscal_year)] + years[t][p.fiscal_year] < max_per_company_year
                )
                and (max_per_unit is None or per_unit[split_unit[p.passage_id]] < max_per_unit)
            ]
            if not candidates:
                raise PlanError(f"{t} ran out of distinct overlap groups and split units{caps}")
            reach: Counter[str] = Counter()  # balanced_years: how many remaining units still reach each year
            if years_first:
                reach.update(year for year, _ in {(p.fiscal_year, unit_of[p.passage_id]) for p in candidates})

            def rank(p: SourcePassage, t: str = t, reach: Counter[str] = reach) -> tuple[Any, ...]:
                return (
                    *((years[t][p.fiscal_year], reach[p.fiscal_year]) if years_first else ()),
                    *((-math.floor(scores[p.passage_id] / SCORE_PRIORITY_STEP),) if score_priority else ()),
                    kinds[t][passage_kind(p)],
                    years[t][p.fiscal_year],
                    families[section_family(p)] / FAMILY_WEIGHTS[section_family(p)],
                    all_years[p.fiscal_year],
                    *((-scores[p.passage_id],) if scores is not None else ()),
                    _order(seed, p.passage_id),
                )

            if capped:  # the best-ranked pick that still leaves the quota reachable: a greedy pick could spend
                # the only group that reaches a year the cap leaves short
                pick = next((p for p in sorted(candidates, key=rank) if leaves_room(t, p)), None)
                if pick is None:
                    raise PlanError(f"{t} cannot reach its quota of {quota[t]}{caps}")
            else:
                pick = min(candidates, key=rank)
            chosen.append(pick)
            used_groups.add(pick.group_id)
            used_units.add(unit_of[pick.passage_id])
            per_unit[split_unit[pick.passage_id]] += 1
            kinds[t][passage_kind(pick)] += 1
            years[t][pick.fiscal_year] += 1
            families[section_family(pick)] += 1
            all_years[pick.fiscal_year] += 1
            progressed = True
        if not progressed:
            break
    return chosen


def year_coverage(
    eligible: Sequence[SourcePassage], chosen: Sequence[SourcePassage], exclude_groups: Collection[str] = ()
) -> dict[str, dict[str, list[str]]]:
    """Each company's fiscal years in the plan (``covered``) and among its eligible sources outside the
    excluded groups (``available``)."""
    available: dict[str, set[str]] = {}
    for p in eligible:
        if p.group_id not in exclude_groups:
            available.setdefault(p.ticker, set()).add(p.fiscal_year)
    covered: dict[str, set[str]] = {}
    for p in chosen:
        covered.setdefault(p.ticker, set()).add(p.fiscal_year)
    return {
        t: {"available": sorted(years), "covered": sorted(covered.get(t, ()))}
        for t, years in sorted(available.items())
    }


def plan_path(directory: Path, split: str, name: str) -> Path:
    return directory / PLANS_DIRNAME / f"{split}.{name}.json"


def make_plan(
    *,
    name: str,
    split: str,
    method: Method,
    seed: int,
    size: int,
    approved: Sequence[SourcePassage],
    quality: SourceQualityResult | None,
    quality_settings: SourceQualitySettings | None,
    prepared_manifest_sha256: str,
    unit_of: Mapping[str, str] | None = None,
    max_company_share: float = 0.25,
    exclude_groups: Collection[str] = (),
    excluded_from: Sequence[str] = (),
    one_per: Literal["unit", "group"] = "unit",
    scores: Mapping[str, float] | None = None,
    min_score: float | None = None,
    score_method: str | None = None,
    max_per_company_year: int | None = None,
    max_per_unit: int | None = None,
    prior_units: Mapping[str, int] | None = None,
    prior_company_years: Mapping[tuple[str, str], int] | None = None,
    score_priority: bool = False,
) -> dict[str, Any]:
    """A plan over ``approved`` (filtered by ``quality`` when given); see the module docstring."""
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise PlanError("plan names use lowercase letters, digits and underscores")
    if method not in METHODS:
        raise PlanError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    if size < 1:
        raise PlanError("a plan needs at least one passage")
    eligible = quality.kept if quality is not None else list(approved)
    options = (
        one_per != "unit",
        scores is not None,
        max_per_company_year is not None,
        max_per_unit is not None,
        bool(prior_units or prior_company_years),
        score_priority,
    )
    if any(options) and method not in BALANCED_METHODS:
        raise PlanError(
            "one_per, scores, score priority and the per-year and per-unit caps apply to the balanced methods only"
        )
    below = 0
    if scores is not None:
        if missing := [p.passage_id for p in eligible if p.passage_id not in scores]:
            raise PlanError(f"no score for {len(missing)} eligible passages, e.g. {missing[0]}")
        if min_score is not None:
            below = sum(1 for p in eligible if scores[p.passage_id] < min_score)
            eligible = [p for p in eligible if scores[p.passage_id] >= min_score]
    if method == "company_round_robin":
        chosen = sample_sources(eligible, size, seed)
    elif method == "company_year_round_robin":
        chosen = company_year_round_robin(eligible, size, seed)
    else:
        if unit_of is None:
            raise PlanError("the balanced method needs each passage's split unit")
        chosen = balanced_selection(
            eligible,
            size,
            seed,
            unit_of=unit_of,
            max_company_share=max_company_share,
            exclude_groups=set(exclude_groups),
            years_first=method == "balanced_years",
            one_per=one_per,
            scores=scores,
            max_per_company_year=max_per_company_year,
            max_per_unit=max_per_unit,
            prior_units=prior_units,
            prior_company_years=prior_company_years,
            score_priority=score_priority,
        )
    if len(chosen) < size:
        raise PlanError(f"only {len(chosen)} eligible passages for a plan of {size}")
    balanced = method in BALANCED_METHODS
    extra: dict[str, Any] = {}
    if method == "balanced_years":
        extra["coverage"] = year_coverage(eligible, chosen, set(exclude_groups))
    if balanced:
        extra = {
            **extra,
            "constraints": {
                "max_company_share": max_company_share,
                "one_per_overlap_group": True,
                "one_per_split_unit": one_per == "unit",
                "excluded_groups": len(set(exclude_groups)),
                "excluded_from": list(excluded_from),
                **({"scores": {"method": score_method, "min_score": min_score, "below_min_score": below}}
                   if scores is not None else {}),
                **({"max_per_company_year": max_per_company_year} if max_per_company_year is not None else {}),
                **({"max_per_split_unit": max_per_unit} if max_per_unit is not None else {}),
                **({"prior_counted": {"units": sum((prior_units or {}).values()),
                                      "company_years": sum((prior_company_years or {}).values())}}
                   if prior_units or prior_company_years else {}),
                **({"score_priority": True} if score_priority else {}),
            },
            "by_kind": dict(sorted(Counter(passage_kind(p) for p in chosen).items())),
            "by_section_family": dict(sorted(Counter(section_family(p) for p in chosen).items())),
        }
    return {
        **extra,
        "format_version": PLAN_FORMAT_VERSION,
        "name": name,
        "split": split,
        "method": method,
        "seed": seed,
        "size": size,
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "source_quality": (
            quality_settings.record() if quality is not None and quality_settings else None
        ),
        "eligible": {
            "approved": len(approved),
            "eligible": len(eligible),
            **({"excluded_by_rule": quality.stats()["excluded_by_rule"]} if quality else {}),
        },
        "passages": [
            {
                "passage_id": p.passage_id,
                "ticker": p.ticker,
                "fiscal_year": p.fiscal_year,
                "section": p.section,
                **(
                    {
                        "group_id": p.group_id,
                        "split_unit": unit_of[p.passage_id],
                        "kind": passage_kind(p),
                        "section_family": section_family(p),
                    }
                    if balanced and unit_of is not None
                    else {}
                ),
                **({"score": scores[p.passage_id]} if scores is not None else {}),
            }
            for p in chosen
        ],
        "by_ticker": dict(sorted(Counter(p.ticker for p in chosen).items())),
        "by_fiscal_year": dict(sorted(Counter(p.fiscal_year for p in chosen).items())),
        "by_item": dict(sorted(Counter(item_kind(p.item) for p in chosen).items())),
    }


def write_plan(directory: Path, plan: Mapping[str, Any]) -> bool:
    """Write the plan; False if an identical copy exists. A different existing plan is refused."""
    path = plan_path(directory, plan["split"], plan["name"])
    text = json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise PlanError(
            f"{path} already holds a different plan; plans are never changed, so choose a new name"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


STOP_FORMAT_VERSION = 1
STOP_NOTICE = (
    "Generation under this plan is closed: generate --plan refuses it, so no further request is sent for it. Positions "
    "after the stop position are never generated under this plan; the plan file itself is unchanged."
)


def stop_path(directory: Path, split: str, name: str) -> Path:
    """Where a plan's stop record lives, beside the plan (which is never changed)."""
    return directory / PLANS_DIRNAME / f"{split}.{name}.stopped.json"


def make_stop_record(
    plan: Mapping[str, Any],
    planned: Sequence[SourcePassage],
    state: Mapping[str, Mapping[str, Any]],
    *,
    plan_sha256: str,
    records: Mapping[str, Any],
    reason: str,
    stopped_at: str,
) -> dict[str, Any]:
    """Close ``plan`` after its last generated position: the longest run of plan passages, from the first, that have a
    generation record (``state``: the latest record per passage). ``records`` binds the record lines at the stop (path,
    lines, SHA-256). Positions after the stop whose passage has a record (records are per passage, so another plan may
    have generated it) are listed, never counted as this plan's."""
    if not reason.strip():
        raise PlanError("a stop record needs a reason")
    position = 0
    while position < len(planned) and planned[position].passage_id in state:
        position += 1
    return {
        "kind": "sampling_plan_stop",
        "format_version": STOP_FORMAT_VERSION,
        "notice": STOP_NOTICE,
        "plan": plan["name"],
        "split": plan["split"],
        "plan_sha256": plan_sha256,
        "passages": len(planned),
        "position": position,
        "statuses": dict(sorted(Counter(str(state[p.passage_id]["status"]) for p in planned[:position]).items())),
        "later_positions_with_records": [
            i for i, p in enumerate(planned[position:], start=position + 1) if p.passage_id in state
        ],
        "records": dict(records),
        "reason": reason.strip(),
        "stopped_at": stopped_at,
    }


def write_stop_record(directory: Path, record: Mapping[str, Any]) -> bool:
    """Write the stop record once; ``False`` if the plan is already stopped at the same position. A plan stopped at
    another position, or a record for another plan file, is refused."""
    path = stop_path(directory, record["split"], record["plan"])
    if path.exists():
        present = json.loads(path.read_text(encoding="utf-8"))
        if (present.get("position"), present.get("plan_sha256")) == (record["position"], record["plan_sha256"]):
            return False
        raise PlanError(
            f"{path} already stops plan {record['plan']} at position {present.get('position')}; stop records are "
            "never changed"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def read_stop_record(directory: Path, split: str, name: str) -> dict[str, Any] | None:
    """The plan's stop record, ``None`` if its generation was never stopped; refused if it does not belong to the
    plan file as it is."""
    path = stop_path(directory, split, name)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("kind"), record.get("format_version"), record.get("plan"), record.get("split")) != (
        "sampling_plan_stop", STOP_FORMAT_VERSION, name, split
    ):
        raise PlanError(f"{path} is not a format-{STOP_FORMAT_VERSION} stop record for {split} plan {name}")
    plan_file = plan_path(directory, split, name)
    digest = hashlib.sha256(plan_file.read_bytes()).hexdigest() if plan_file.is_file() else ""
    if digest != record.get("plan_sha256"):
        raise PlanError(f"{path} was written for another version of {plan_file}")
    return record


def load_plan(
    directory: Path,
    split: str,
    name: str,
    sources: Mapping[str, SourcePassage],
    *,
    prepared_manifest_sha256: str,
) -> tuple[dict[str, Any], list[SourcePassage]]:
    """A written plan and its passages in plan order, checked against the current preparation."""
    path = plan_path(directory, split, name)
    if not path.exists():
        raise PlanError(f"no plan {name!r} for {split} at {path}; create it with the plan command")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("format_version") != PLAN_FORMAT_VERSION or plan.get("split") != split:
        raise PlanError(f"{path} is not a format-{PLAN_FORMAT_VERSION} plan for {split}")
    if plan["prepared_manifest_sha256"] != prepared_manifest_sha256:
        raise PlanError(f"{path} was made from a different preparation")
    missing = [p["passage_id"] for p in plan["passages"] if p["passage_id"] not in sources]
    if missing:
        raise PlanError(
            f"{path} lists passages that are not approved {split} sources: {missing[:3]}"
        )
    return plan, [sources[p["passage_id"]] for p in plan["passages"]]


__all__ = [
    "BALANCED_METHODS",
    "METHODS",
    "PLANS_DIRNAME",
    "PLAN_FORMAT_VERSION",
    "SCORE_PRIORITY_STEP",
    "STOP_FORMAT_VERSION",
    "STOP_NOTICE",
    "FAMILY_WEIGHTS",
    "PlanError",
    "balanced_selection",
    "company_year_round_robin",
    "passage_kind",
    "section_family",
    "load_plan",
    "make_plan",
    "make_stop_record",
    "plan_path",
    "read_stop_record",
    "stop_path",
    "write_plan",
    "write_stop_record",
    "year_coverage",
]
