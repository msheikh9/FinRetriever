"""The retrieval alignment diagnostic (finretriever.evaluation.alignment)."""

import pytest

from finretriever.evaluation.alignment import (
    HitInfo,
    Scope,
    answer_figures,
    figure_in_text,
    fiscal_year,
    question_alignment,
    year_relation,
)


def test_fiscal_year_uses_the_period_end_then_the_filing_date():
    assert fiscal_year({"report_date": "2025-01-31"}, "2025-03-14") == 2025  # Walmart FY2025 ends in January
    assert fiscal_year({"report_date": "2020-09-26"}, "2020-10-30") == 2020
    assert fiscal_year({}, "2021-02-01") == 2021


@pytest.mark.parametrize(
    ("year", "years", "expected"),
    [
        (2020, {2020}, "exact"),
        (2021, {2020}, "later_comparative"),
        (2022, {2020}, "later_comparative"),
        (2023, {2020}, "other_year"),
        (2019, {2020}, "other_year"),  # an earlier filing never reports a later year
        (2024, {2022, 2024}, "exact"),
        (2023, {2022, 2025}, "later_comparative"),
    ],
)
def test_year_relation(year, years, expected):
    assert year_relation(year, years) == expected


def test_year_relation_needs_years():
    with pytest.raises(ValueError):
        year_relation(2020, set())


def test_single_scope_shares_and_ranks():
    scope = Scope.of(["aapl"], [2020])
    hits = [
        HitInfo("MSFT", 2020, False),  # cross-company
        HitInfo("AAPL", 2021, True),  # later filing, comparative window
        HitInfo("AAPL", 2020, True),  # in scope
        HitInfo("AAPL", 2017, False),  # cross-year
    ]
    out = question_alignment(hits, scope, k=10)
    assert out["k"] == 4
    assert out["company_share"] == 0.75 and out["cross_company_share"] == 0.25
    assert out["exact_year_share"] == 0.25
    assert out["later_comparative_share"] == 0.25
    assert out["cross_year_share"] == 0.25
    assert out["table_share"] == 0.5
    assert out["first_company_rank"] == 2 and out["first_in_scope_rank"] == 3
    assert out["top1_company"] is False and out["top1_year"] is None
    assert out["cells"] == 1 and out["all_cells_covered_exact"] is True


def test_cross_year_cells_need_each_year():
    scope = Scope.of(["JPM"], [2021, 2023])
    one_year = [HitInfo("JPM", 2023, True)] * 3
    out = question_alignment(one_year, scope)
    assert out["cells"] == 2
    assert out["cells_covered_exact"] == 0.5 and out["all_cells_covered_exact"] is False
    # the 2023 filing may carry 2021 as a comparative column (within two years)
    assert out["cells_covered_with_comparatives"] == 1.0


def test_cross_company_coverage():
    scope = Scope.of(["XOM", "PFE"], [2022])
    out = question_alignment([HitInfo("XOM", 2022, False), HitInfo("XOM", 2022, False)], scope)
    assert out["companies_covered"] == 0.5
    assert out["cells_covered_exact"] == 0.5
    out = question_alignment([HitInfo("XOM", 2022, False), HitInfo("PFE", 2022, True)], scope)
    assert out["companies_covered"] == 1.0 and out["all_cells_covered_exact"] is True


def test_only_the_top_k_counts():
    scope = Scope.of(["TSLA"], [2019])
    hits = [HitInfo("WMT", 2019, False)] * 10 + [HitInfo("TSLA", 2019, False)]
    out = question_alignment(hits, scope, k=10)
    assert out["company_hit"] is False and out["first_in_scope_rank"] is None


def test_scope_needs_a_ticker_and_hits_are_required():
    with pytest.raises(ValueError):
        Scope.of([], [2020])
    with pytest.raises(ValueError):
        question_alignment([], Scope.of(["AAPL"], [2020]))


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("$6,686 million (benefit)", ["6,686"]),
        ("$18.7 billion in 2022 and 2023", ["18.7"]),  # years are not figures; 18.7 has three digits
        ("approximately 1,200,000 employees", ["1,200,000"]),
        ("about 154,000 people", ["154,000"]),
        ("2.5%", []),  # too short to be distinctive
        ("revenue of 3456 million", ["3456"]),
        ("no figure here", []),
    ],
)
def test_answer_figures(answer, expected):
    assert answer_figures(answer) == expected


def test_figure_in_text_matches_whole_numbers_only():
    text = "U.S. Federal > Deferred | 2022: (6,686) | 2021: 155 | total 16,686 and 6,686.5"
    assert figure_in_text("6,686", text)
    assert not figure_in_text("6,686", "total 16,686 and 6,686.5")
    assert figure_in_text("18.7", "rose to $18.7 billion.")
    assert not figure_in_text("18.7", "rose to $118.75 billion")
