"""Query understanding (finretriever.retrieval.query_understanding): companies, fiscal years, measures."""

import pytest

from finretriever.retrieval.query_understanding import (
    PARSER_VERSION,
    QueryParser,
    extract_measures,
    extract_years,
)

COMPANIES = {
    "AAPL": "Apple Inc.", "AMZN": "AMAZON COM INC", "JPM": "JPMORGAN CHASE & CO", "PFE": "PFIZER INC",
    "TSLA": "Tesla, Inc.", "WMT": "Walmart Inc.", "XOM": "EXXON MOBIL CORP",
}
EXTRA = {"JPM": ["j.p. morgan", "jp morgan"], "XOM": ["exxon mobil"]}


@pytest.fixture(scope="module")
def parser() -> QueryParser:
    return QueryParser.from_companies(COMPANIES, EXTRA)


@pytest.mark.parametrize(
    ("text", "tickers"),
    [
        ("What was Apple's total net sales in fiscal 2023?", ("AAPL",)),
        ("How did ExxonMobil's capital spending break down in 2022?", ("XOM",)),
        ("What did J.P. Morgan report as its CET1 ratio?", ("JPM",)),
        ("Compare Walmart and Amazon revenue growth in FY2024.", ("AMZN", "WMT")),
        ("What does TSLA say about regulatory credits?", ("TSLA",)),
        ("Which company had the highest margin?", ()),
    ],
)
def test_companies(parser, text, tickers):
    assert parser.parse(text).tickers == tickers


@pytest.mark.parametrize(
    ("text", "years"),
    [
        ("Apple's FY2022 revenue", [2022]),
        ("Apple's FY 22 revenue", [2022]),
        ("in fiscal 2024", [2024]),
        ("for fiscal year 2021", [2021]),
        ("for the year ended January 31, 2024", [2024]),  # Walmart's fiscal 2024
        ("as of December 31, 2021", [2021]),
        ("in 2023 versus 2022", [2022, 2023]),
        ("each year from 2019 through 2023", [2019, 2020, 2021, 2022, 2023]),
        ("over 2019-2021", [2019, 2020, 2021]),
        ("from FY2021 to FY2023", [2021, 2022, 2023]),
        ("between FY2023 and FY2025", [2023, 2025]),  # the years named; a later filing still covers 2024
        ("a $2,023 million charge and 1,999 employees", []),  # amounts are not years
        ("upon adoption of ASU 2016-02 (Topic 842)", []),  # accounting-standard codes are not years
        ("no year here", []),
    ],
)
def test_years(text, years):
    assert extract_years(text)[0] == years


def test_year_mentions_record_the_rule_and_whether_the_year_is_named_as_fiscal():
    _, mentions = extract_years("Walmart's fiscal 2024 sales as of January 31, 2024 and in 2023")
    rules = {m.rule: m for m in mentions}
    assert rules["fiscal"].explicit_fiscal and rules["fiscal"].years == (2024,)
    assert not rules["date"].explicit_fiscal and not rules["year"].explicit_fiscal
    _, ranged = extract_years("from FY2019 through FY2021")
    assert ranged[0].rule == "range" and ranged[0].explicit_fiscal


def test_comparison_type(parser):
    assert parser.parse("How did Tesla and Walmart compare in FY2024?").comparison == "cross_company"
    assert parser.parse("How did Pfizer's revenue change in 2023 versus 2022?").comparison == "cross_year"
    single = parser.parse("What was Pfizer's revenue in 2023?")
    assert single.comparison == "none" and not single.comparative
    assert parser.parse("How did Pfizer's revenue change in 2023?").comparative


def test_fallbacks_are_recorded_when_nothing_is_named(parser):
    scope = parser.parse("What are the main risks of the business?")
    assert not scope.has_company and not scope.has_year and len(scope.fallbacks) == 2
    assert scope.cells == [] and scope.parser_version == PARSER_VERSION


def test_measures_and_figure_intent():
    assert extract_measures("total long-term debt and operating cash flow") == ["cash_flow", "debt"]
    assert "research_and_development" in extract_measures("research and development expense")
    assert extract_measures("supply chain concentration") == []


def test_scope_serialises(parser):
    scope = parser.parse("How much total long-term debt did Apple report at the end of FY2022?")
    d = scope.to_dict()
    assert d["tickers"] == ["AAPL"] and d["years"] == [2022] and d["asks_for_figure"] is True
    assert d["year_mentions"] == [{"rule": "fy", "surface": "FY2022", "years": [2022], "role": "fiscal"}]
    assert scope.cells == [("AAPL", 2022)]


def test_empty_question_is_rejected(parser):
    with pytest.raises(ValueError):
        parser.parse("  ")


# ---- year roles (qu-v2) ----------------------------------------------------------------------------------------
# Realistic analyst questions written for these tests (none is taken from an evaluation set). Each gives the
# reporting years the scope must keep and the role of every year mention, by its surface text.
ROLE_CASES = [
    # fiscal reporting years, including both sides of a cross-year comparison
    ("What was Apple's total net sales in fiscal 2023?", [2023], {"fiscal 2023": "fiscal"}),
    ("How did JPMorgan's net interest income change in 2023 compared with 2022?", [2022, 2023],
     {"2023": "fiscal", "2022": "fiscal"}),
    ("What were Walmart's total assets as of January 31, 2024?", [2024], {"January 31, 2024": "fiscal"}),
    ("How did Pfizer's revenue evolve from 2019 through 2022?", [2019, 2020, 2021, 2022],
     {"from 2019 through 2022": "fiscal"}),
    ("What did Tesla report for automotive regulatory credits in Q4 2024?", [2024], {"2024": "fiscal"}),
    ("How much did Walmart spend on share repurchases in fiscal 2025 under its repurchase program?", [2025],
     {"fiscal 2025": "fiscal"}),
    ("What emissions targets had Walmart established for fiscal 2024 operations?", [2024],
     {"fiscal 2024": "fiscal"}),  # a target named, but the year is the reporting year
    # comparison target years: a goal or forecast year, and the baseline it is measured against
    ("How much does ExxonMobil aim to cut upstream methane intensity by 2030 compared to 2016 levels?", [],
     {"2030": "target", "2016": "historical"}),
    ("According to ExxonMobil's fiscal 2025 Form 10-K, what global energy demand does it project by 2050, and how "
     "does that compare with 2024?", [2025], {"fiscal 2025": "fiscal", "2050": "target", "2024": "historical"}),
    ("What is Walmart's renewable electricity goal for 2035?", [], {"2035": "target"}),
    ("What are Apple's 2030 carbon neutrality goals for its supply chain?", [], {"2030": "target"}),
    ("What capital expenditure outlook for 2026 did ExxonMobil give in its fiscal 2025 report?", [2025],
     {"2026": "target", "fiscal 2025": "fiscal"}),
    ("How many vehicles is Tesla scheduled to deliver from 2026 through 2027 under the contract?", [],
     {"from 2026 through 2027": "target"}),
    # historical reference years
    ("How has Amazon's free cash flow developed since 2015?", [], {"2015": "historical"}),
    ("How does JPMorgan's CET1 ratio at December 31, 2023 compare with its 2019 baseline?", [2023],
     {"December 31, 2023": "fiscal", "2019": "historical"}),
    ("How has Amazon's business mix changed since it was incorporated in 1994?", [], {"1994": "historical"}),
    # debt maturity and expiry years
    ("What was the carrying amount of Apple's notes due 2030 as of September 30, 2023?", [2023],
     {"2030": "maturity", "September 30, 2023": "fiscal"}),
    ("How much of Amazon's long-term debt matures in 2027 and 2028?", [], {"2027": "maturity", "2028": "maturity"}),
    ("What is the outstanding principal of Tesla's 2025 Notes at fiscal 2021 year end?", [2021],
     {"2025": "maturity", "fiscal 2021": "fiscal"}),
    ("When do Pfizer's net operating loss carryforwards expire in 2031?", [], {"2031": "maturity"}),
    ("What interest rate applies to the senior notes due March 1, 2033 that Walmart reported in fiscal 2024?", [2024],
     {"March 1, 2033": "maturity", "fiscal 2024": "fiscal"}),
    # years mentioned incidentally, inside the name of a law, plan, award, programme or event
    ("How did the Tax Cuts and Jobs Act of 2017 affect Apple's effective tax rate in fiscal 2019?", [2019],
     {"2017": "incidental", "fiscal 2019": "fiscal"}),
    ("How many shares remained available under the 2019 Stock Incentive Plan at the end of fiscal 2023?", [2023],
     {"2019": "incidental", "fiscal 2023": "fiscal"}),
    ("What compensation expense did Tesla record in 2021 for the 2018 CEO Performance Award?", [2021],
     {"2021": "fiscal", "2018": "incidental"}),
    ("How did JPMorgan describe lessons from the 2008 financial crisis in its 2022 risk factors?", [2022],
     {"2008": "incidental", "2022": "fiscal"}),
    ("Which forward-looking statements does Amazon's 2020 annual report make under the Private Securities "
     "Litigation Reform Act of 1995?", [2020], {"2020": "fiscal", "1995": "incidental"}),
    ("In fiscal 2024, what emissions targets did Walmart report?", [2024], {"fiscal 2024": "fiscal"}),
    # no cue: the year stays a reporting year (the qu-v1 reading)
    ("How did Pfizer's operating cash flow change by 2023?", [2023], {"2023": "fiscal"}),
    ("What did Amazon spend on technology and infrastructure in 2022?", [2022], {"2022": "fiscal"}),
]


@pytest.mark.parametrize(("text", "years", "roles"), ROLE_CASES, ids=[c[0][:50] for c in ROLE_CASES])
def test_year_roles(parser, text, years, roles):
    scope = parser.parse(text)
    assert list(scope.years) == years
    assert {m.surface: m.role for m in scope.year_mentions} == roles


def test_non_reporting_years_are_recorded_and_fall_back_to_no_year_constraint(parser):
    scope = parser.parse("How much of ExxonMobil's debt matures in 2029, and what is its 2030 emissions goal?")
    assert scope.years == () and not scope.has_year and scope.comparison == "none"
    assert scope.non_reporting_years == {"maturity": (2029,), "target": (2030,)}
    assert any("not reporting periods: no year constraint" in f for f in scope.fallbacks)
    d = scope.to_dict()
    assert [m["role"] for m in d["year_mentions"]] == ["maturity", "target"] and d["parser_version"] == "qu-v2"


def test_roles_only_ever_remove_years_never_add_them(parser):
    legacy = QueryParser.from_companies(COMPANIES, EXTRA, version="qu-v1")
    for text, _, _ in ROLE_CASES:
        assert set(parser.parse(text).years) <= set(legacy.parse(text).years), text


def test_qu_v1_is_unchanged_every_year_is_fiscal_and_no_role_is_serialised():
    legacy = QueryParser.from_companies(COMPANIES, EXTRA, version="qu-v1")
    scope = legacy.parse("What was the balance of Apple's notes due 2030 as of September 30, 2023?")
    assert scope.years == (2023, 2030) and scope.parser_version == "qu-v1"
    assert all(m.role is None and m.reporting for m in scope.year_mentions)
    assert scope.to_dict()["year_mentions"] == [
        {"rule": "date", "surface": "September 30, 2023", "years": [2023]},
        {"rule": "year", "surface": "2030", "years": [2030]},
    ]
    assert legacy.describe()["parser_version"] == "qu-v1"


def test_unknown_parser_version_is_rejected():
    with pytest.raises(ValueError, match="parser version"):
        QueryParser.from_companies(COMPANIES, EXTRA, version="qu-v9")
