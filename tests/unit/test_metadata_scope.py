"""Metadata-aware scoping (finretriever.retrieval.metadata_scope)."""

import pytest

from finretriever.data.schemas import Passage
from finretriever.retrieval.metadata_scope import (
    CorpusMetadata,
    ScopePolicy,
    allowed_years,
    interleave,
    plan_scope,
)
from finretriever.retrieval.query_understanding import QueryParser, QueryScope

PARSER = QueryParser.from_companies({"AAPL": "Apple Inc.", "WMT": "Walmart Inc.", "XOM": "EXXON MOBIL CORP"})


def corpus(per_cell: int = 3) -> list[Passage]:
    out = []
    for ticker in ("AAPL", "WMT", "XOM"):
        for year in range(2019, 2026):
            for i in range(per_cell):
                out.append(Passage(passage_id=f"{ticker}-{year}-{i}", filing_id=f"{ticker}-{year}", ticker=ticker,
                                   form_type="10-K", filing_date=f"{year + 1}-02-01", text=f"{ticker} {year} {i}",
                                   metadata={"report_date": f"{year}-12-31"}))
    return out


META = CorpusMetadata(corpus())


def years_of(positions):
    return sorted({META.years[i] for i in positions})


def test_explicit_fiscal_years_allow_the_year_and_two_later_filings():
    policy = ScopePolicy(year="filter")
    assert allowed_years(PARSER.parse("Apple's fiscal 2021 revenue"), policy) == {2021, 2022, 2023}
    # a date may be a filing-date reference: one earlier filing is allowed too
    assert allowed_years(PARSER.parse("Apple's holders of record as of January 28, 2022"), policy) == {
        2021, 2022, 2023, 2024}
    assert allowed_years(PARSER.parse("Apple's risks"), policy) == set()


def test_company_and_year_filters():
    plan = plan_scope(PARSER.parse("Apple's fiscal 2021 revenue"), ScopePolicy("filter", "filter", min_eligible=1),
                      META)
    assert {META.tickers[i] for i in plan.eligible} == {"AAPL"} and years_of(plan.eligible) == [2021, 2022, 2023]
    assert plan.company_filtered and plan.year_filtered


def test_no_named_company_or_year_means_no_constraint():
    plan = plan_scope(PARSER.parse("What are the main risks?"), ScopePolicy("filter", "filter"), META)
    assert len(plan.eligible) == len(META) and not plan.company_filtered and not plan.year_filtered
    assert any("no company named" in d for d in plan.decisions)


def test_year_filter_falls_back_when_too_few_passages_remain():
    scope = PARSER.parse("Apple's revenue in fiscal 2030")  # no FY2030-2032 filings
    plan = plan_scope(scope, ScopePolicy("filter", "filter", min_eligible=1), META)
    assert not plan.year_filtered and {META.tickers[i] for i in plan.eligible} == {"AAPL"}
    assert any("fell back" in d for d in plan.decisions)


def test_exclusions_are_never_eligible():
    excluded = {META.position["AAPL-2021-0"]}
    plan = plan_scope(PARSER.parse("Apple fiscal 2021"), ScopePolicy("filter", "filter", min_eligible=1), META,
                      excluded)
    assert META.position["AAPL-2021-0"] not in plan.eligible


def test_cells_for_cross_company_and_cross_year_questions():
    policy = ScopePolicy("filter", "filter", cells="interleave", min_eligible=1)
    plan = plan_scope(PARSER.parse("Compare Apple and Walmart in fiscal 2021"), policy, META)
    assert [c.label for c in plan.cells] == ["AAPL/2021", "WMT/2021"]
    plan = plan_scope(PARSER.parse("How did Apple change from FY2019 to FY2020?"), policy, META)
    assert [c.label for c in plan.cells] == ["AAPL/2019", "AAPL/2020"]
    assert plan_scope(PARSER.parse("Apple fiscal 2021"), policy, META).cells == ()


def test_year_tiers():
    scope = PARSER.parse("Apple fiscal 2021")
    allowed = allowed_years(scope, ScopePolicy(year="boost"))
    tiers = {META.year_tier(META.position[f"AAPL-{y}-0"], scope, allowed) for y in (2021,)}
    assert tiers == {0}
    assert META.year_tier(META.position["AAPL-2022-0"], scope, allowed) == 1
    assert META.year_tier(META.position["AAPL-2019-0"], scope, allowed) == 2


def test_oracle_scope_without_mentions_uses_explicit_windows():
    oracle = QueryScope(text="q", tickers=("WMT",), years=(2020,), measures=(), comparison="none",
                        comparative=False, asks_for_figure=False)
    assert allowed_years(oracle, ScopePolicy(year="filter", source="oracle")) == {2020, 2021, 2022}


def test_interleave_round_robin_without_duplicates():
    assert interleave([[1, 2, 3], [4, 1, 5]], 10) == [1, 4, 2, 3, 5]
    assert interleave([[1, 2], [3, 4]], 3) == [1, 3, 2]


def test_policy_validation():
    for bad in (dict(company="maybe"), dict(year="strict"), dict(cells="split"), dict(after=-1),
                dict(min_eligible=0), dict(source="gold")):
        with pytest.raises(ValueError):
            ScopePolicy(**bad)
