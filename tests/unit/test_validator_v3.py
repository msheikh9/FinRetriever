"""Tests for validator sec-qcheck-v3: quotes over table cells, ellipses and shortened row labels,
standard terminology, and the checks that stay strict (offline)."""

from __future__ import annotations

import pytest

from finretriever.training.query_generation import (
    VALIDATOR_VERSION,
    CompanyAliases,
    GenerationSettings,
    SourcePassage,
    check_candidate,
)
from finretriever.training.quotes import locate_quote, table_label_grams

SETTINGS = GenerationSettings(model="test-model")
ALIASES = CompanyAliases.from_companies(
    {"AMZN": "Amazon.com Inc.", "XOM": "Exxon Mobil Corp", "PFE": "Pfizer Inc", "TSLA": "Tesla, Inc."}
)
# Flattened as the corpus flattens 10-K tables: rows run on without a separator after their last cell.
OPEX = (
    "Operating expenses > Cost of sales | Year Ended December 31, 2021: 272,344 | Year Ended December 31, 2022: "
    "288,831 | Year Ended December 31, 2023: 304,739 Operating expenses > Fulfillment | Year Ended December 31, "
    "2021: 75,111 | Year Ended December 31, 2022: 84,299 | Year Ended December 31, 2023: 90,619 Total operating "
    "expenses | Year Ended December 31, 2021: 444,943 | Year Ended December 31, 2022: 501,735 | Year Ended "
    "December 31, 2023: 537,933"
)
PENSION = (
    "[TABLE] Pension Benefits U.S. | Non-U.S. 2021 | 2020 | 2021 | 2020 (millions of dollars) For funded pension "
    "plans with an accumulated benefit obligation in excess of plan assets > Accumulated benefit obligation | "
    "14,511 | 16,129 | 3,108 | 4,602 in excess of plan assets > Fair value of plan assets | 13,266 | 15,300 | "
    "1,711 | 2,652 For funded pension plans with a projected benefit obligation in excess of plan assets > "
    "Projected benefit obligation | 16,836 | 19,456 | 4,840 | 13,836 [/TABLE]"
)
DEALERS = (
    "In certain locations, decisions by regulators permitting us to sell vehicles have been and may be challenged "
    "by dealer associations and others. In some states, there have also been regulatory and legislative efforts "
    "by dealer associations to propose laws that, if enacted, would prevent us from obtaining dealer licenses."
)
TAXES = (
    "[TABLE] Cash paid for income taxes, net of refunds, consisted of | Year Ended December 31, (MILLIONS) | 2024 | "
    "2023 United States | $2,593 | $1,923 International | 1,012 | 1,224 Total | $3,605 | $3,147 [/TABLE] We "
    "continue to evaluate the effect of new tax legislation on our future cash payments and estimated liabilities."
)


def source(ticker: str, company: str, text: str, year: str = "2023") -> SourcePassage:
    return SourcePassage(
        passage_id=f"{ticker.lower()}_{year}", split="train", group_id="grp_00001", ticker=ticker,
        fiscal_year=year, company=company, form_type="10-K", filing_date=f"{int(year) + 1}-02-02",
        report_date=f"{year}-12-31", section="Item 8 Financial Statements", visible_text=text,
    )


AMZN = source("AMZN", "Amazon.com Inc.", OPEX)
XOM = source("XOM", "Exxon Mobil Corp", PENSION, "2021")
TSLA = source("TSLA", "Tesla, Inc.", DEALERS, "2020")
PFE = source("PFE", "Pfizer Inc", TAXES, "2024")


def check(passage: SourcePassage, question: str, answer: str, evidence: str, kind: str = "financial_table") -> str | None:
    return check_candidate({"question": question, "type": kind, "answer": answer, "evidence": evidence}, passage,
                           SETTINGS, ALIASES)[1]


TOTAL_Q = "What was the total operating expense reported by Amazon.com Inc. for the fiscal year ended December 31, 2023?"


def test_a_correct_quote_may_skip_cells_of_its_own_row() -> None:
    assert VALIDATOR_VERSION == "sec-qcheck-v3"
    evidence = "Total operating expenses | Year Ended December 31, 2023: 537,933"
    assert locate_quote(evidence, OPEX).form == "fragments"
    assert check(AMZN, TOTAL_Q, "537,933", evidence) is None
    fulfillment = "Operating expenses > Fulfillment | Year Ended December 31, 2023: 90,619"
    assert check(AMZN, "How much did Amazon.com Inc. spend on fulfillment in 2023?", "90,619", fulfillment) is None


@pytest.mark.parametrize(
    ("evidence", "answer"),
    [
        ("Total operating expenses | Year Ended December 31, 2023: 537,934", "537,934"),  # an invented cell
        ("Operating expenses > Fulfillment | Year Ended December 31, 2023: 537,933", "537,933"),  # another row's value
        ("Total operating expenses ... 537,933", "537,933"),  # a value after an ellipsis could belong to any row
        ("Total operating expenses | - | - | - | -", "0"),  # empty cells are no evidence
    ],
)
def test_invented_or_stitched_quotes_are_still_rejected(evidence: str, answer: str) -> None:
    assert locate_quote(evidence, OPEX) is None
    assert check(AMZN, TOTAL_Q, answer, evidence) in ("evidence_not_in_visible_text", "evidence_length",
                                                      "number_not_in_passage")


def test_ellipsis_fragments_must_each_be_verbatim() -> None:
    q = "According to Tesla, Inc.'s 2020 annual report, how have dealer associations challenged its direct sales?"
    joined = ("decisions by regulators permitting us to sell vehicles have been and may be challenged by dealer "
              "associations... regulatory and legislative efforts by dealer associations to propose laws")
    assert check(TSLA, q, "By challenging regulator decisions and proposing laws", joined, "risk") is None
    edited = joined.replace("regulatory and legislative", "regulatory and judicial")
    assert check(TSLA, q, "By challenging regulator decisions and proposing laws", edited, "risk") == (
        "evidence_not_in_visible_text")


def test_a_row_label_may_restate_the_parent_the_flattened_table_shortened() -> None:
    q = ("What was the fair value of plan assets for Exxon Mobil Corp's U.S. funded pension plans where the "
         "accumulated benefit obligation exceeded plan assets in 2021?")
    right = ("For funded pension plans with an accumulated benefit obligation in excess of plan assets > Fair value "
             "of plan assets | 13,266 | 15,300 | 1,711 | 2,652")
    assert locate_quote(right, PENSION).form == "rebuilt_label"
    assert check(XOM, q, "13,266", right) is None
    wrong = right.replace("an accumulated benefit obligation", "a projected benefit obligation")
    assert locate_quote(wrong, PENSION) is None  # the nearest full parent of that row is the other one
    assert check(XOM, q, "13,266", wrong) == "evidence_not_in_visible_text"


def test_standard_terms_and_table_titles_are_not_copying_but_prose_still_is() -> None:
    assert ("cash", "paid", "for", "income", "taxes", "net", "of", "refunds") in table_label_grams(TAXES)
    q = "What was the total cash paid for income taxes, net of refunds, by Pfizer Inc for the fiscal year ended December 31, 2024?"
    assert check(PFE, q, "$3,605 million", "Total | $3,605 | $3,147") is None  # v2: copied_from_passage
    copied = ("Does Pfizer Inc continue to evaluate the effect of new tax legislation on our future cash payments "
              "in 2024?")
    assert check(PFE, copied, "yes", "We continue to evaluate the effect of new tax legislation on our future cash "
                 "payments", "other") == "copied_from_passage"


def test_invented_figures_dates_and_companies_are_still_rejected() -> None:
    evidence = "Total operating expenses | Year Ended December 31, 2023: 537,933"
    assert check(AMZN, TOTAL_Q, "540,000", evidence) == "number_not_in_passage"
    assert check(AMZN, TOTAL_Q.replace("2023?", "2023, after the $9,999 restructuring charge?"), "537,933",
                 evidence) == "number_not_in_passage"
    assert check(AMZN, TOTAL_Q.replace("December 31, 2023", "March 15, 2023"), "537,933", evidence) in (
        "date_not_in_passage", "number_not_in_passage")
    assert check(AMZN, TOTAL_Q.replace("Amazon.com Inc.", "Tesla, Inc."), "537,933", evidence) in (
        "missing_company", "other_company")
    hidden = source("AMZN", "Amazon.com Inc.", OPEX[: OPEX.index("Total operating")])  # the row is not visible
    assert check(hidden, TOTAL_Q, "537,933", evidence) == "evidence_not_in_visible_text"
