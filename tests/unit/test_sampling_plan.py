"""Tests for versioned sampling plans: reproducible, balanced, filtered selections that are never rewritten."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from finretriever.training.query_generation import SourcePassage, sample_sources
from finretriever.training.sampling_plan import (
    PlanError,
    balanced_selection,
    company_year_round_robin,
    load_plan,
    make_plan,
    passage_kind,
    plan_path,
    section_family,
    write_plan,
)
from finretriever.training.source_quality import SourceQualitySettings, filter_sources

TICKERS = ("AAA", "BBB", "CCC")
YEARS = ("2021", "2022", "2023", "2024")
SUBSTANTIVE = "Revenue rose 8% to $1,250 million on higher volumes and prices, lifting operating margin and cash flow."
EXHIBITS = ("Exhibit 4.1 | Indenture, incorporated by reference to Exhibit 4.1 to the Current Report on Form 8-K. "
            "Exhibit 31.1 | Certification of the Chief Executive Officer, filed herewith. Exhibit 101 | XBRL.")


def pool() -> list[SourcePassage]:
    out = []
    for t in TICKERS:
        for y in YEARS:
            for i in range(3):
                text = EXHIBITS if i == 0 else SUBSTANTIVE
                out.append(SourcePassage(
                    passage_id=f"{t}_{y}_{i}", split="train", group_id=f"g_{t}_{y}_{i}", ticker=t, fiscal_year=y,
                    company=f"{t} Corp.", form_type="10-K", filing_date=f"{int(y) + 1}-02-20",
                    report_date=f"{y}-12-31", section="Item 7 MD&A" if i else "Item 15 Exhibits", visible_text=text,
                    item="7" if i else "15",
                ))
    return out


def plan(**kw: object) -> dict:
    sources = pool()
    quality = filter_sources(sources, SourceQualitySettings(), {})
    args = dict(name="pilot_v2", split="train", method="company_year_round_robin", seed=7, size=12,
                approved=sources, quality=quality, quality_settings=SourceQualitySettings(),
                prepared_manifest_sha256="a" * 64)
    args.update(kw)
    return make_plan(**args)  # type: ignore[arg-type]


def test_the_original_method_reproduces_the_original_pilot_sample() -> None:
    sources = pool()
    original = plan(name="pilot_v1", method="company_round_robin", quality=None, quality_settings=None, size=7)
    assert [p["passage_id"] for p in original["passages"]] == [p.passage_id for p in sample_sources(sources, 7, 7)]
    assert original["source_quality"] is None and original["eligible"] == {"approved": 36, "eligible": 36}


def test_balanced_plans_spread_over_companies_and_years_and_skip_filtered_passages() -> None:
    p = plan()
    assert p["by_ticker"] == {"AAA": 4, "BBB": 4, "CCC": 4}
    assert p["by_fiscal_year"] == {"2021": 3, "2022": 3, "2023": 3, "2024": 3}
    for t in TICKERS:  # each company's picks come from four different years
        assert len({e["fiscal_year"] for e in p["passages"] if e["ticker"] == t}) == 4
    assert all(not e["passage_id"].endswith("_0") for e in p["passages"])  # exhibit lists are excluded
    assert p["eligible"]["excluded_by_rule"]["exhibit_index"] == 12 and p["source_quality"]["version"]
    assert [e["ticker"] for e in p["passages"][:3]] == ["AAA", "BBB", "CCC"]  # companies take turns


def test_plans_are_deterministic_and_seeded() -> None:
    sources = pool()
    first = company_year_round_robin(sources, 9, seed=3)
    assert first == company_year_round_robin(list(reversed(sources)), 9, seed=3)  # input order is irrelevant
    assert first != company_year_round_robin(sources, 9, seed=4)
    assert Counter(p.ticker for p in first) == {"AAA": 3, "BBB": 3, "CCC": 3}
    assert plan() == plan()
    with pytest.raises(PlanError, match="only 24 eligible"):
        plan(size=25)
    with pytest.raises(PlanError, match="lowercase"):
        plan(name="Pilot V2")


def test_plans_are_written_once_and_checked_when_used(tmp_path: Path) -> None:
    p = plan()
    assert write_plan(tmp_path, p) is True and write_plan(tmp_path, p) is False  # identical copy: unchanged
    with pytest.raises(PlanError, match="never changed"):
        write_plan(tmp_path, plan(seed=8))
    assert json.loads(plan_path(tmp_path, "train", "pilot_v2").read_text()) == p
    sources = {s.passage_id: s for s in pool()}
    loaded, passages = load_plan(tmp_path, "train", "pilot_v2", sources, prepared_manifest_sha256="a" * 64)
    assert loaded == p and [s.passage_id for s in passages] == [e["passage_id"] for e in p["passages"]]
    with pytest.raises(PlanError, match="different preparation"):
        load_plan(tmp_path, "train", "pilot_v2", sources, prepared_manifest_sha256="b" * 64)
    with pytest.raises(PlanError, match="not approved"):
        load_plan(tmp_path, "train", "pilot_v2", {}, prepared_manifest_sha256="a" * 64)
    with pytest.raises(PlanError, match="no plan"):
        load_plan(tmp_path, "train", "pilot_v9", sources, prepared_manifest_sha256="a" * 64)


# --- the balanced method: company quotas with a cap, one passage per group and split unit ---------

SEVEN = ("AAPL", "AMZN", "JPM", "PFE", "TSLA", "WMT", "XOM")
SECTIONS = (("1A", "Item 1A Risk Factors"), ("7", "Item 7 Management's Discussion and Analysis"),
            ("8", "Item 8 Financial Statements"), ("", "Note 12 – Loans"), ("", "CREDIT RISK MANAGEMENT"))


def seven_pool(jpm: int = 40, others: int = 10) -> tuple[list[SourcePassage], dict[str, str]]:
    """JPMorgan-heavy sources; pairs of passages share a group, and groups 0-1 of each company share a unit."""
    out, unit_of = [], {}
    for t in SEVEN:
        for i in range(jpm if t == "JPM" else others):
            item, section = SECTIONS[i % len(SECTIONS)]
            text = (f"[TABLE] Revenue | {t} {i} [/TABLE]" if i % 2 else f"{t} narrative passage {i} about risk.")
            group = f"g_{t}_{i // 2}"
            out.append(SourcePassage(
                passage_id=f"{t}_{i}", split="train", group_id=group, ticker=t, fiscal_year=str(2019 + i % 5),
                company=f"{t} Inc.", form_type="10-K", filing_date="2025-02-01", report_date="2024-12-31",
                section=section, visible_text=text, item=item))
            unit_of[f"{t}_{i}"] = f"u_{t}_0" if i // 2 in (0, 1) else group
    return out, unit_of


def test_balanced_plans_cap_companies_and_take_one_passage_per_group_and_unit() -> None:
    sources, unit_of = seven_pool()
    chosen = balanced_selection(sources, 21, 7, unit_of=unit_of, max_company_share=0.25)
    by_ticker = Counter(p.ticker for p in chosen)
    assert by_ticker == dict.fromkeys(SEVEN, 3)  # equal quotas although JPMorgan has four times the sources
    assert [p.ticker for p in chosen[:7]] == sorted(SEVEN)  # picks alternate, so an early stop stays balanced
    assert len({p.group_id for p in chosen}) == len({unit_of[p.passage_id] for p in chosen}) == 21
    for t in SEVEN:
        picks = [p for p in chosen if p.ticker == t]
        assert {passage_kind(p) for p in picks} == {"table", "narrative"}  # both kinds in every company
        assert len({p.fiscal_year for p in picks}) == 3  # three different years
    assert chosen == balanced_selection(list(reversed(sources)), 21, 7, unit_of=unit_of, max_company_share=0.25)
    assert chosen != balanced_selection(sources, 21, 8, unit_of=unit_of, max_company_share=0.25)
    # Excluded groups (e.g. already covered by carried-over pairs) are never planned.
    banned = {p.group_id for p in chosen[:7]}
    again = balanced_selection(sources, 21, 7, unit_of=unit_of, max_company_share=0.25, exclude_groups=banned)
    assert not {p.group_id for p in again} & banned


def test_the_company_cap_holds_even_when_other_companies_run_short() -> None:
    sources, unit_of = seven_pool(jpm=40, others=2)  # each other company: one group, one unit
    with pytest.raises(PlanError, match="at most 25% per company"):
        balanced_selection(sources, 20, 7, unit_of=unit_of, max_company_share=0.25)
    chosen = balanced_selection(sources, 8, 7, unit_of=unit_of, max_company_share=0.25)
    assert Counter(p.ticker for p in chosen)["JPM"] <= 2  # int(0.25 * 8)
    with pytest.raises(PlanError, match="in \\(0, 1\\]"):
        balanced_selection(sources, 8, 7, unit_of=unit_of, max_company_share=0)


def test_balanced_plans_record_their_constraints_and_spread() -> None:
    sources, unit_of = seven_pool()
    p = make_plan(name="pilot_v3", split="train", method="balanced", seed=7, size=14, approved=sources, quality=None,
                  quality_settings=None, prepared_manifest_sha256="a" * 64, unit_of=unit_of, max_company_share=0.25,
                  exclude_groups={"g_JPM_19"}, excluded_from=["carryover:pilot_v2"])
    assert p["constraints"] == {"max_company_share": 0.25, "one_per_overlap_group": True, "one_per_split_unit": True,
                                "excluded_groups": 1, "excluded_from": ["carryover:pilot_v2"]}
    assert p["by_ticker"] == dict.fromkeys(SEVEN, 2) and set(p["by_kind"]) == {"table", "narrative"}
    assert set(p["passages"][0]) == {"passage_id", "ticker", "fiscal_year", "section", "group_id", "split_unit", "kind",
                                     "section_family"}
    with pytest.raises(PlanError, match="split unit"):
        make_plan(name="x", split="train", method="balanced", seed=7, size=14, approved=sources, quality=None,
                  quality_settings=None, prepared_manifest_sha256="a" * 64)
    families = {s: section_family(SourcePassage(**{**sources[0].__dict__, "item": item, "section": s}))
                for item, s in SECTIONS}
    assert families == {"Item 1A Risk Factors": "risk", "Item 7 Management's Discussion and Analysis": "mdna",
                        "Item 8 Financial Statements": "statements", "Note 12 – Loans": "statements",
                        "CREDIT RISK MANAGEMENT": "risk"}


# --- balanced_years: every fiscal year a company's split units can supply ------------------------

# Apple's validation sources: an Item 2 chain and an Item 7A chain each span several years in one split unit.
APPLE_UNITS = {"u_properties": ("2", "2019 2020 2021 2022 2023 2024 2025"), "u_market_risk": ("7A", "2019 2020 2021 2022"),
               "u_t1": ("8", "2020 2021"), "u_t2": ("8", "2021 2022"), "u_n1": ("8", "2021"), "u_t3": ("8", "2022"),
               "u_t4": ("8", "2023"), "u_t5": ("8", "2025")}


def apple_pool() -> tuple[list[SourcePassage], dict[str, str]]:
    out, unit_of = [], {}
    for unit, (item, years) in APPLE_UNITS.items():
        for year in years.split():
            pid = f"{unit}_{year}"
            text = f"[TABLE] Net sales | {year} [/TABLE]" if unit.startswith("u_t") else f"Apple narrative {unit} {year}."
            out.append(SourcePassage(
                passage_id=pid, split="validation", group_id=f"g_{pid}", ticker="AAPL", fiscal_year=year,
                company="Apple Inc.", form_type="10-K", filing_date=f"{int(year) + 1}-10-30", report_date=f"{year}-09-28",
                section=f"Item {item}", visible_text=text, item=item))
            unit_of[pid] = unit
    return out, unit_of


def test_balanced_years_covers_every_year_the_units_can_supply_where_balanced_may_not() -> None:
    sources, unit_of = apple_pool()
    years = {p.fiscal_year for p in sources}
    short = 0
    for seed in range(30):
        chosen = balanced_selection(sources, 7, seed, unit_of=unit_of, max_company_share=1.0, years_first=True)
        assert {p.fiscal_year for p in chosen} == years  # 7 years from 8 units, one passage per unit
        assert len({unit_of[p.passage_id] for p in chosen}) == len({p.group_id for p in chosen}) == 7
        assert {passage_kind(p) for p in chosen} == {"table", "narrative"}
        short += len({p.fiscal_year for p in balanced_selection(
            sources, 7, seed, unit_of=unit_of, max_company_share=1.0)}) < len(years)
    assert short  # balanced takes kinds first and can spend a multi-year unit on a year others supply

    def years_plan(size: int, excluded: set[str]) -> dict:
        return make_plan(name="val_pilot_v1", split="validation", method="balanced_years", seed=3, size=size,
                         approved=sources, quality=None, quality_settings=None, prepared_manifest_sha256="a" * 64,
                         unit_of=unit_of, max_company_share=1.0, exclude_groups=excluded)

    assert years_plan(7, set())["coverage"] == {"AAPL": {"available": sorted(years), "covered": sorted(years)}}
    # Without the 7A passage, 2019 and 2024 come from the Item 2 unit alone: one of them, as on the real data.
    coverage = years_plan(7, {"g_u_market_risk_2019"})["coverage"]["AAPL"]
    assert coverage["available"] == sorted(years) and len(coverage["covered"]) == 6
    assert set(coverage["available"]) - set(coverage["covered"]) < {"2019", "2024"}
    p = years_plan(6, {"g_u_t4_2023", "g_u_properties_2023"})  # excluded groups are not available
    assert "2023" not in p["coverage"]["AAPL"]["available"]
    assert p["constraints"]["one_per_split_unit"] and p["method"] == "balanced_years"
    assert "coverage" not in plan()  # earlier methods' plans are unchanged
    old = make_plan(name="pilot_v3", split="validation", method="balanced", seed=3, size=7, approved=sources,
                    quality=None, quality_settings=None, prepared_manifest_sha256="a" * 64, unit_of=unit_of,
                    max_company_share=1.0)
    assert "coverage" not in old and [e["passage_id"] for e in old["passages"]] == [
        q.passage_id for q in balanced_selection(sources, 7, 3, unit_of=unit_of, max_company_share=1.0)]


# --- bge_base_ft_v3 options: one passage per overlap group, isolated-evidence scores ---------------


def unit_pool() -> tuple[list[SourcePassage], dict[str, str]]:
    """Two companies; each split unit joins two overlap groups of different years."""
    out, unit_of = [], {}
    for t in ("AAA", "BBB"):
        for i in range(8):
            out.append(SourcePassage(
                passage_id=f"{t}_{i}", split="validation", group_id=f"g_{t}_{i}", ticker=t,
                fiscal_year=str(2019 + i), company=f"{t} Corp.", form_type="10-K", filing_date="2025-02-01",
                report_date="2024-12-31", section="Item 7 MD&A", visible_text=SUBSTANTIVE, item="7"))
            unit_of[f"{t}_{i}"] = f"u_{t}_{i // 2}"
    return out, unit_of


def test_one_per_group_plans_several_groups_of_a_unit_and_scores_prefer_isolated_evidence() -> None:
    sources, unit_of = unit_pool()
    with pytest.raises(PlanError, match="one passage per split unit"):
        balanced_selection(sources, 12, 7, unit_of=unit_of, max_company_share=0.5)
    chosen = balanced_selection(sources, 12, 7, unit_of=unit_of, max_company_share=0.5, one_per="group")
    assert len({p.group_id for p in chosen}) == 12 and len({unit_of[p.passage_id] for p in chosen}) < 12
    scores = {p.passage_id: (0.9 if p.passage_id.endswith(("_1", "_2")) else 0.2) for p in sources}
    args = dict(name="val_b", split="validation", method="balanced", seed=7, approved=sources, quality=None,
                quality_settings=None, prepared_manifest_sha256="a" * 64, unit_of=unit_of, max_company_share=0.5,
                one_per="group", scores=scores, score_method="test")
    strict = make_plan(**args, size=4, min_score=0.5)  # type: ignore[arg-type]
    assert {p["passage_id"] for p in strict["passages"]} == {"AAA_1", "AAA_2", "BBB_1", "BBB_2"}
    assert strict["constraints"]["scores"] == {"method": "test", "min_score": 0.5, "below_min_score": 12}
    assert strict["constraints"]["one_per_split_unit"] is False and strict["passages"][0]["score"] == 0.9
    with pytest.raises(PlanError, match="only 4 passages fit 6"):
        make_plan(**args, size=6, min_score=0.5)  # type: ignore[arg-type]
    loose = make_plan(**args, size=4)  # type: ignore[arg-type]  # all eligible; high scores win the ties
    assert {p["passage_id"] for p in loose["passages"]} <= {p.passage_id for p in sources}
    default = plan(method="balanced", size=9, max_company_share=0.34, unit_of={p.passage_id: p.group_id for p in pool()})
    assert "scores" not in default["constraints"] and default["constraints"]["one_per_split_unit"] is True
    assert all("score" not in p for p in default["passages"])
    with pytest.raises(PlanError, match="balanced methods only"):
        plan(one_per="group")


def test_a_group_shared_by_two_companies_never_leaves_one_of_them_short() -> None:
    """Shared boilerplate can put two companies' passages in one overlap group; quotas count only exclusive ones."""
    out = []
    for t, groups in (("AAA", ("a1", "a2", "shared")), ("BBB", ("b1", "shared"))):
        for g in groups:
            out.append(SourcePassage(
                passage_id=f"{t}_{g}", split="validation", group_id=g, ticker=t, fiscal_year="2024",
                company=f"{t} Corp.", form_type="10-K", filing_date="2025-02-01", report_date="2024-12-31",
                section="Item 7 MD&A", visible_text=SUBSTANTIVE, item="7"))
    unit_of = {p.passage_id: p.group_id for p in out}
    for seed in range(30):  # whichever company reaches the shared group first, the other still has its quota
        chosen = balanced_selection(out, 3, seed, unit_of=unit_of, max_company_share=1.0)
        assert len({p.group_id for p in chosen}) == 3 and Counter(p.ticker for p in chosen) == {"AAA": 2, "BBB": 1}
    with pytest.raises(PlanError, match="only 3 passages fit 4"):
        balanced_selection(out, 4, 7, unit_of=unit_of, max_company_share=1.0)


# --- caps per company and fiscal year and per split unit (alternatives to val_pilot_v2) ------------


def validation_passage(pid: str, group: str, ticker: str, year: str) -> SourcePassage:
    return SourcePassage(
        passage_id=pid, split="validation", group_id=group, ticker=ticker, fiscal_year=year, company=f"{ticker} Corp.",
        form_type="10-K", filing_date="2025-02-01", report_date="2024-12-31", section="Item 7 MD&A",
        visible_text=SUBSTANTIVE, item="7")


def test_caps_per_company_year_and_split_unit_hold_are_recorded_and_a_plan_beyond_them_is_refused() -> None:
    """Each company has six 2019 groups in one split unit and one group in each of 2020 and 2021."""
    out, unit_of = [], {}
    for t in ("AAA", "BBB"):
        for y, n in (("2019", 6), ("2020", 1), ("2021", 1)):
            for i in range(n):
                pid = f"{t}_{y}_{i}"
                out.append(validation_passage(pid, f"g_{pid}", t, y))
                unit_of[pid] = f"u_{t}_2019" if y == "2019" else f"u_{pid}"
    args = dict(name="val_c", split="validation", method="balanced_years", seed=7, approved=out, quality=None,
                quality_settings=None, prepared_manifest_sha256="a" * 64, unit_of=unit_of, max_company_share=0.5,
                one_per="group")
    plain = make_plan(**args, size=16)  # type: ignore[arg-type]
    assert plain["by_fiscal_year"]["2019"] == 12
    assert not {"max_per_company_year", "max_per_split_unit"} & set(plain["constraints"])
    capped = make_plan(**args, size=8, max_per_company_year=3, max_per_unit=2)  # type: ignore[arg-type]
    assert max(Counter((p["ticker"], p["fiscal_year"]) for p in capped["passages"]).values()) <= 3
    assert max(Counter(p["split_unit"] for p in capped["passages"]).values()) == 2  # the 2019 unit gives two
    assert capped["by_ticker"] == {"AAA": 4, "BBB": 4} and capped["by_fiscal_year"] == {"2019": 4, "2020": 2, "2021": 2}
    assert capped["constraints"]["max_per_company_year"] == 3 and capped["constraints"]["max_per_split_unit"] == 2
    with pytest.raises(PlanError, match="only 8 passages fit 10 .* at most 3 per company and fiscal year and at most 2 per"):
        make_plan(**args, size=10, max_per_company_year=3, max_per_unit=2)  # type: ignore[arg-type]
    with pytest.raises(PlanError, match="one_per='group'"):
        make_plan(**{**args, "one_per": "unit"}, size=4, max_per_unit=2)  # type: ignore[arg-type]
    with pytest.raises(PlanError, match="at least 1"):
        make_plan(**args, size=4, max_per_company_year=0)  # type: ignore[arg-type]
    with pytest.raises(PlanError, match="balanced methods only"):
        plan(max_per_company_year=2)


def test_a_capped_plan_keeps_the_only_group_of_a_scarce_year_for_that_year() -> None:
    """The best-ranked pick can spend a group that alone reaches another year; a capped plan takes the next pick that
    still lets the company reach its quota instead of running short."""
    out = [validation_passage("both_2019", "g_both", "AAA", "2019"),
           validation_passage("both_2020", "g_both", "AAA", "2020"),
           validation_passage("only_2019", "g_only", "AAA", "2019")]
    unit_of = {p.passage_id: p.group_id for p in out}
    scores = {"both_2019": 0.9, "both_2020": 0.6, "only_2019": 0.6}  # the best-ranked pick spends g_both on 2019
    chosen = balanced_selection(out, 2, 7, unit_of=unit_of, max_company_share=1.0, one_per="group", scores=scores,
                                max_per_company_year=1)
    assert {p.passage_id for p in chosen} == {"both_2020", "only_2019"}


def test_a_split_unit_shared_by_two_companies_counts_for_neither_quota_under_a_unit_cap() -> None:
    out = [validation_passage(f"{t}_{g}", f"g_{t}_{g}", t, "2024") for t in ("AAA", "BBB") for g in ("shared", "own")]
    unit_of = {p.passage_id: ("u_shared" if p.passage_id.endswith("shared") else p.group_id) for p in out}
    with pytest.raises(PlanError, match="only 2 passages fit 3"):
        balanced_selection(out, 3, 7, unit_of=unit_of, max_company_share=1.0, one_per="group", max_per_unit=1)
    for seed in range(20):  # whichever company takes the shared unit, the other still reaches its quota
        chosen = balanced_selection(out, 2, seed, unit_of=unit_of, max_company_share=0.5, one_per="group",
                                    max_per_unit=1)
        assert Counter(p.ticker for p in chosen) == {"AAA": 1, "BBB": 1}
        assert max(Counter(unit_of[p.passage_id] for p in chosen).values()) == 1


def test_passages_of_earlier_plans_count_toward_the_caps() -> None:
    """A unit of four groups with two passages already generated has room for one more under a cap of 3."""
    out = [validation_passage(f"AAA_{i}", f"g{i}", "AAA", str(2020 + i)) for i in range(4)]
    out += [validation_passage(f"AAA_x{i}", f"x{i}", "AAA", "2019") for i in range(3)]
    unit_of = {p.passage_id: ("u_shared" if p.passage_id.startswith("AAA_") and "x" not in p.passage_id else p.group_id)
               for p in out}
    args = dict(unit_of=unit_of, max_company_share=1.0, one_per="group", max_per_unit=3)
    assert sum(unit_of[p.passage_id] == "u_shared" for p in balanced_selection(out, 6, 7, **args)) == 3
    with pytest.raises(PlanError, match="only 4 passages fit 6"):
        balanced_selection(out, 6, 7, **args, prior_units={"u_shared": 2})
    chosen = balanced_selection(out, 4, 7, **args, prior_units={"u_shared": 2})
    assert sum(unit_of[p.passage_id] == "u_shared" for p in chosen) == 1
    with pytest.raises(PlanError, match="only 5 passages fit 6"):  # 2019 already holds two of its three
        balanced_selection(out, 6, 7, unit_of=unit_of, max_company_share=1.0, one_per="group",
                           max_per_company_year=3, prior_company_years={("AAA", "2019"): 2})
    plan_args = dict(name="val_c", split="validation", method="balanced_years", seed=7, size=4, approved=out,
                     quality=None, quality_settings=None, prepared_manifest_sha256="a" * 64, **args)
    recorded = make_plan(**plan_args, prior_units={"u_shared": 2})  # type: ignore[arg-type]
    assert recorded["constraints"]["prior_counted"] == {"units": 2, "company_years": 0}
    assert "prior_counted" not in make_plan(**plan_args)["constraints"]  # type: ignore[arg-type]


def test_score_priority_ranks_by_score_within_the_year_balance() -> None:
    """Kind balance would take the table second; with score priority the better narrative comes first."""
    out = [
        validation_passage("n_best", "g1", "AAA", "2024"),
        validation_passage("n_good", "g2", "AAA", "2024"),
        SourcePassage(passage_id="t_poor", split="validation", group_id="g3", ticker="AAA", fiscal_year="2024",
                      company="AAA Corp.", form_type="10-K", filing_date="2025-02-01", report_date="2024-12-31",
                      section="Item 8", visible_text="[TABLE] Revenue | 2024: $1,250 [/TABLE]", item="8"),
    ]
    unit_of = {p.passage_id: p.group_id for p in out}
    scores = {"n_best": 0.9, "n_good": 0.85, "t_poor": 0.2}
    args = dict(unit_of=unit_of, max_company_share=1.0, years_first=True, scores=scores)
    assert [p.passage_id for p in balanced_selection(out, 3, 7, **args)] == ["n_best", "t_poor", "n_good"]
    assert [p.passage_id for p in balanced_selection(out, 3, 7, **args, score_priority=True)] == ["n_best", "n_good", "t_poor"]
    with pytest.raises(PlanError, match="score_priority needs scores"):
        balanced_selection(out, 3, 7, unit_of=unit_of, max_company_share=1.0, score_priority=True)
    plan_args = dict(name="val_c", split="validation", method="balanced_years", seed=7, size=3, approved=out,
                     quality=None, quality_settings=None, prepared_manifest_sha256="a" * 64, unit_of=unit_of,
                     max_company_share=1.0, scores=scores, score_method="test")
    assert make_plan(**plan_args, score_priority=True)["constraints"]["score_priority"] is True  # type: ignore[arg-type]
    assert "score_priority" not in make_plan(**plan_args)["constraints"]  # type: ignore[arg-type]
