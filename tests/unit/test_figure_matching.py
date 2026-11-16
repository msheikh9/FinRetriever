"""Tests for figure matching in the per-pair evidence check (offline).

The six regression cases reproduce, with short excerpts of the public filings, the pilot_v3 pairs the
first locator missed because their answers state a unit the table cell does not
(docs/training_pilot_v3_quality_report.md, section 6).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from finretriever.data.schemas import Passage
from finretriever.training.evidence_reuse import (
    LOCATOR_VERSION,
    EvidenceLocator,
    Figure,
    ReuseIndex,
    ReuseSettings,
    declared_scale,
    figures,
    refers_to,
    same_amount,
    value_strings,
)
from finretriever.training.pair_contract import EQUIVALENCE_METHOD
from finretriever.training.pair_quality import equivalent_ids, evidence_leak
from tests.unit.training_fixtures import make_passage

SETTINGS = ReuseSettings()


def fig(text: str, declared: int | None = None) -> Figure:
    (found,) = figures(text, "AAA", SETTINGS, declared=declared)
    return found


# --- normalisation ---------------------------------------------------------------------------------


def test_currency_separators_spacing_and_compatible_units_are_one_amount() -> None:
    forms = ["57,411", "$57,411", "$ 57,411", "$57,411 million", "57,411 million", "57,411 millions",
             "$57.411 billion", "$57,411M"]
    found = [fig(f) for f in forms]
    assert all(f.magnitude in (Decimal("57411"), Decimal("57.411")) and not f.negative for f in found)
    assert all(same_amount(a, b) for a in found for b in found if a is not b)
    assert fig("$57,411 million").scale == 6 and fig("$57.411 billion").scale == 9 and fig("57,411").scale is None
    assert fig("$57,411M").scale == 6  # a unit glued to a dollar figure
    assert {"57411", "57411m", "57.411b"} <= value_strings(fig("$57,411 million"))  # how the index writes it


def test_thousands_millions_and_billions_stay_distinct_and_no_rounding_is_equated() -> None:
    millions = fig("$57,411 million")
    assert not same_amount(millions, fig("$57,411 thousand"))
    assert not same_amount(millions, fig("$57,411 billion"))
    assert not same_amount(millions, fig("$57.4 billion"))  # a rounding is a different amount
    assert same_amount(fig("$1.2 billion"), fig("$1,200 million"))
    # A table cell takes its passage's declared unit; without one, its unit is unknown.
    assert same_amount(fig("57,411", declared=6), millions)
    assert not same_amount(fig("57,411", declared=3), millions)
    assert not same_amount(fig("57,411", declared=3), fig("57,411", declared=6))
    assert same_amount(fig("57,411", declared=6), fig("57,411"))
    assert not same_amount(fig("57,411"), fig("57.411"))  # two unknown units: the digits must agree
    assert not same_amount(fig("57,411"), fig("57,412 million"))


def test_signs_and_accounting_parentheses_are_kept() -> None:
    negatives = ["(16,117)", "$(16,117)", "($16,117)", "-16,117", "-$16,117 million", "$(16,117) million"]
    assert all(fig(f).negative for f in negatives)
    assert all(same_amount(fig(a), fig(b)) for a in negatives for b in negatives)
    assert not same_amount(fig("(16,117)"), fig("16,117")) and same_amount(fig("(16,117)"), fig("16,117"), signed=False)
    # An unsigned answer may state a negative cell in words; a negative answer needs a negative cell.
    assert refers_to(fig("A loss of $1,403 million"), fig("(1,403)"))
    assert not refers_to(fig("$(16,117) million"), fig("16,117"))


def test_percentages_dates_and_years_are_never_amounts() -> None:
    assert fig("a margin of 12.5%").percent and fig("125 basis points").percent
    assert not same_amount(fig("12.5%"), fig("$12.5 million"))
    assert same_amount(fig("12.5%"), fig("12.5 percent"))
    assert figures("On September 26, 2020 (2020-09-26, or 9/26/2020), in fiscal 2024: 4%.", "AAA", SETTINGS) == []
    assert fig("$2,020 million").magnitude == Decimal("2020")  # a figure that looks like a year


def test_declared_units() -> None:
    assert declared_scale("The components were as follows (In millions): [TABLE] Net income | 2020: 57,411") == 6
    assert declared_scale("[TABLE] (MILLIONS OF DOLLARS) | Revenue") == 6
    assert declared_scale("Selected metrics (in billions, except ratios)") == 9
    assert declared_scale("(dollars in thousands) Revenue | 9,104") == 3
    # A declaration about counts, several units, or none: the unit stays unknown.
    assert declared_scale("(in millions, except number of shares, which are reflected in thousands)") is None
    assert declared_scale("(in millions) ... (in billions)") is None
    assert declared_scale("Net income | 2020: 57,411") is None


# --- the six pilot_v3 leaks ------------------------------------------------------------------------

FILLER = "This sentence keeps the passage long enough and shares nothing with the evidence quoted above."

CASES = {
    # pair: (ticker, evidence, answer, other passage's text, its split, the expected rejection)
    "002": ("AAPL",
            "Operating activities > Net income | Years ended September 26, 2020: 57,411 | Years ended September 28, "
            "2019: 55,256 | Years ended September 29, 2018: 59,531",
            "$57,411 million",
            "[TABLE] Net sales | Years ended September 26, 2020: $274,515 Net income | Years ended September 26, "
            "2020: $57,411 | Years ended September 28, 2019: $55,256 [/TABLE]",
            "blocked", "evaluation_evidence"),
    "052": ("WMT",
            "Net cash used in financing activities | Fiscal Years Ended January 31, 2021: $(16,117)",
            "$(16,117) million",
            "[TABLE] Net cash used in financing activities | Fiscal Years Ended January 31, 2023: (17,039) | Fiscal "
            "Years Ended January 31, 2022: (22,828) | Fiscal Years Ended January 31, 2021: (16,117) [/TABLE]",
            "blocked", "evaluation_evidence"),
    "006": ("AAPL",
            "Gross property, plant and equipment | 2022: 114,457 | 2021: 109,723",
            "$114,457 million",
            "[TABLE] Leasehold improvements | 2023: 12,839 | 2022: 11,271 Gross property, plant and equipment | "
            "2023: 114,599 | 2022: 114,457 [/TABLE]",
            "validation", "cross_split_evidence"),
    "007": ("AAPL",
            "Interest and dividend income | 2022: $2,825 | 2021: $2,843 | 2020: $3,763",
            "$2,825 million",
            "Other income/(expense), net for 2023, 2022 and 2021 (in millions): [TABLE] Interest and dividend "
            "income | 2023: $3,750 | 2022: $2,825 | 2021: $2,843 [/TABLE]",
            "validation", "cross_split_evidence"),
    "034": ("PFE",
            "Oncology | $12,132 | $12,333 | $10,867",
            "$12,132 million",
            "[TABLE] All other Specialty Care | Various | 2,244 | 2,350 | 2,830 Oncology | $11,627 | $12,132 | "
            "$12,333 [/TABLE]",
            "validation", "cross_split_evidence"),
    "035": ("PFE",
            "Ibrance | HR-positive/HER2-negative metastatic breast cancer | 5,120 | 5,437 | 5,392",
            "5,120 million",
            "[TABLE] Oncology | $11,627 | $12,132 | $12,333 Ibrance | HR-positive/HER2-negative metastatic breast "
            "cancer | 4,753 | 5,120 | 5,437 [/TABLE]",
            "validation", "cross_split_evidence"),
}


def case_corpus(ticker: str, evidence: str, other: str, split: str, *, positive_heading: str = "",
                ) -> tuple[dict[str, Passage], dict[str, dict[str, str]], EvidenceLocator]:
    texts = {"positive": (f"{positive_heading}[TABLE] {evidence} [/TABLE] {FILLER}", "train"),
             "other": (f"{other} {FILLER}", split),
             "unrelated": ("[TABLE] Units shipped | 2020: 9,999 [/TABLE] " + FILLER, "train")}
    ps, rows = {}, {}
    for i, (name, (text, where)) in enumerate(texts.items()):
        p = make_passage(name, ticker, f"{ticker} Inc.", "2022-12-31", text, section_index=i, chunk=0,
                         accession=f"0000000009-23-00000{i}")
        ps[name], rows[p.passage_id] = p, {"split": where, "group_id": f"grp_{i:05d}"}
    index = ReuseIndex.build(list(ps.values()), {p.passage_id: len(p.text) for p in ps.values()}, SETTINGS)
    return ps, rows, EvidenceLocator({p.passage_id: p for p in ps.values()}, index)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_six_pilot_v3_leaks_are_rejected(case: str) -> None:
    ticker, evidence, answer, other, split, reason = CASES[case]
    ps, rows, locator = case_corpus(ticker, evidence, other, split)
    eq = locator.locate(evidence, answer, ticker, ps["positive"].passage_id)
    held = next(e for e in eq if e.passage_id == ps["other"].passage_id)
    assert (held.match, held.visible_match) == ("fact", "fact")  # the answer's figure, under its label
    assert evidence_leak(eq, rows, "train") == reason
    # The same figure with no unit in the answer was always caught; the unit no longer hides it.
    bare = answer.replace(" million", "")
    assert evidence_leak(locator.locate(evidence, bare, ticker, ps["positive"].passage_id), rows, "train") == reason


def test_a_figure_without_its_label_or_in_other_units_is_never_an_exact_match() -> None:
    ticker, evidence, answer, _, _, _ = CASES["002"]
    others = {
        "coincidence": ("[TABLE] Units shipped | 2020: 57,411 [/TABLE]", "value"),  # same digits, another label
        "thousands": ("(dollars in thousands) [TABLE] Net income | 2020: 57,411 [/TABLE]", "other_fact"),
        "rounded": ("Net income was $57.4 billion in fiscal 2020.", None),
        "sign": ("[TABLE] Net income | 2020: (57,411) [/TABLE]", "other_fact"),
        "same_amount": ("Net income was $57.411 billion in fiscal 2020.", "fact"),
    }
    for name, (text, expected) in others.items():
        ps, rows, locator = case_corpus(ticker, evidence, text, "blocked", positive_heading="(In millions) ")
        eq = locator.locate(evidence, answer, ticker, ps["positive"].passage_id)
        held = next((e.match for e in eq if e.passage_id == ps["other"].passage_id), None)
        assert held == expected, name
        assert (evidence_leak(eq, rows, "train") is not None) == (expected == "fact"), name
        # Uncertain matches are never negatives (equivalent_ids keeps them for the pair).
        if expected not in (None, "fact"):
            rows[ps["other"].passage_id]["split"] = "train"
            assert equivalent_ids(eq, rows, "train")["uncertain"] == [ps["other"].passage_id]


def test_pairs_checked_by_the_first_locator_are_stale() -> None:
    assert LOCATOR_VERSION == EQUIVALENCE_METHOD == "evidence-reuse-v2"
