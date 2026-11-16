"""Tests for the source-quality filter: generic rules on synthetic 10-K-style text (no real filing text)."""

from __future__ import annotations

from typing import Any

import pytest

from finretriever.training.query_generation import SourcePassage
from finretriever.training.source_quality import (
    RULES,
    SourceQualitySettings,
    filings_per_group,
    filter_sources,
    matching_rules,
    source_features,
)

SETTINGS = SourceQualitySettings()

EXHIBIT_INDEX = (
    "4.1 | Indenture, dated as of May 3, 2013, between the Registrant and Example Trust Company, as trustee, is "
    "incorporated by reference to Exhibit 4.1 to the Registrant's Current Report on Form 8-K filed May 3, 2013. "
    "4.2 | Officer's Certificate of the Registrant, dated as of May 3, 2013, including forms of global notes, is "
    "incorporated by reference from the Registration Statement on Form S-3. 10.1 | Amended and Restated Credit "
    "Agreement, incorporated by reference to Exhibit 10.1 to the Quarterly Report on Form 10-Q. 31.1 | "
    "Certification of the Chief Executive Officer, filed herewith. 101.INS | XBRL Instance Document."
)
CROSS_REFERENCE = (
    "Item 11. Executive Compensation. The information required by this Item is incorporated herein by reference to "
    "the sections captioned Executive Compensation and Director Compensation in the Proxy Statement for the 2025 "
    "Annual Meeting of Shareholders. See Part III, Item 10 for information about the executive officers of the "
    "registrant, which is included in Part I of this report under the caption Executive Officers."
)
CONTROLS = (
    "Item 9A. Controls and Procedures. Evaluation of Disclosure Controls and Procedures. Our principal executive "
    "officer and principal financial officer evaluated the effectiveness of our disclosure controls and procedures "
    "under Rule 13a-15. Management assessed internal control over financial reporting using the framework of the "
    "Committee of Sponsoring Organizations of the Treadway Commission and concluded that internal control over "
    "financial reporting was effective. The certifications required by Section 1350 are filed as exhibits."
)
CULTURE = (
    "Our people make the difference. We foster a culture where every colleague is encouraged to learn, grow and "
    "volunteer in the communities where we live and work, and we celebrate the diverse perspectives our teams "
    "bring. Through mentoring circles, leadership programs and wellbeing initiatives we help colleagues build "
    "rewarding careers and feel connected to our purpose every day."
)
RISK = (
    "Adverse economic conditions, including inflation, higher interest rates and slower growth, could reduce demand "
    "for our products, pressure our gross margin and impair our ability to obtain credit on acceptable terms, which "
    "could materially adversely affect our results of operations and financial condition."
)
REVENUE_TABLE = (
    "[TABLE] Net sales by segment | 2024: $41,250 | 2023: $38,910 Services | 2024: $12,404 | 2023: 11.2% "
    "[/TABLE] Net sales increased 6% as higher volumes offset lower average selling prices; see Note 5 for the "
    "segment reconciliation."
)
PRODUCTION_TABLE = (
    "[TABLE] Daily production | Region A 2024: 121 | 2023: 118 Region B | 2024: 64 | 2023: 59 Region C | 2024: 33 "
    "| 2023: 35 Region D | 2024: 17 | 2023: 12 [/TABLE] Thousands of units per day for each operating region."
)


def passage(text: str, *, item: str = "1", group: str = "g1", pid: str = "p1") -> SourcePassage:
    return SourcePassage(
        passage_id=pid, split="train", group_id=group, ticker="DELTA", fiscal_year="2024",
        company="Delta Holdings Inc.", form_type="10-K", filing_date="2025-02-20", report_date="2024-12-31",
        section=f"Item {item} Example", visible_text=text, item=item,
    )


def rules(text: str, *, item: str = "1", repeats: int = 1, **settings: Any) -> tuple[str, ...]:
    s = SETTINGS.model_copy(update=settings)
    return matching_rules(source_features(text, repeats), item, s)


def test_exhibit_lists_cross_references_and_repeated_boilerplate_are_excluded() -> None:
    assert rules(EXHIBIT_INDEX, item="15")[0] == "exhibit_index"
    assert rules(EXHIBIT_INDEX, item="")[0] == "exhibit_index"  # by its text, not its section
    assert rules(CROSS_REFERENCE, item="11")[0] == "cross_reference"
    assert rules(CONTROLS, item="9A", repeats=7) == ("repetitive_administrative",)
    assert rules(CONTROLS, item="9A", repeats=1) == ()  # boilerplate that does not repeat is kept


def test_substantive_risk_mdna_and_financial_tables_are_kept() -> None:
    for text, item in ((RISK, "1A"), (RISK, ""), (REVENUE_TABLE, "7"), (REVENUE_TABLE, "8"), (PRODUCTION_TABLE, "2")):
        assert rules(text, item=item) == ()
    # A table full of figures is not a cross reference just because it says "see Note 5".
    assert source_features(REVENUE_TABLE, 1).cross_reference_share > 0
    assert "cross_reference" not in rules(REVENUE_TABLE, cross_reference_share=0.01)


def test_low_financial_content_spares_protected_items_and_numeric_tables() -> None:
    assert rules(CULTURE, item="1") == ("low_financial_content",)
    assert rules(CULTURE, item="") == ("low_financial_content",)
    assert rules(CULTURE, item="1A") == ()  # risk factors, MD&A, market risk, statements: never low content
    assert rules(CULTURE, item="8") == ()
    assert rules(CULTURE, item="1", protected_items=["1"]) == ()
    counts_only = "[TABLE] Plants | 2024: 14 | 2023: 12 Lines | 2024: 51 | 2023: 47 Sites | 2024: 9 | 2023: 8 [/TABLE]"
    assert rules(CULTURE + " " + counts_only) == ()
    roster = "[TABLE] Name | Age | Position Jane Roe | Age: 58 | Position: Chief Legal Officer [/TABLE] " + CULTURE
    assert source_features(roster, 1).table_numbers_per_100_words == 0  # ages are not figures


def test_filter_reports_first_and_any_matching_rule_and_keeps_order() -> None:
    sources = [
        passage(RISK, item="1A", pid="risk"),
        passage(EXHIBIT_INDEX, item="15", pid="exhibits"),
        passage(CONTROLS, item="9A", group="boiler", pid="controls"),
        passage(CULTURE, item="1", pid="culture"),
        passage(REVENUE_TABLE, item="8", pid="table"),
    ]
    rows = {
        f"r{i}": {"group_id": "boiler", "ticker": "DELTA", "fiscal_year": str(2019 + i)} for i in range(4)
    } | {"r9": {"group_id": "g1", "ticker": "DELTA", "fiscal_year": "2024"}}
    groups = filings_per_group(rows)
    assert groups == {"boiler": 4, "g1": 1}
    result = filter_sources(sources, SETTINGS, groups)
    assert [p.passage_id for p in result.kept] == ["risk", "table"]
    assert result.excluded_ids() == {
        "exhibits": ("exhibit_index",),
        "controls": ("repetitive_administrative",),
        "culture": ("low_financial_content",),
    }
    stats = result.stats()
    assert stats["approved"] == 5 and stats["kept"] == 2 and set(stats["excluded_by_rule"]) == set(RULES)
    assert stats["excluded_by_rule"]["exhibit_index"] == 1 and stats["excluded_by_item"]["Item 15"] == 1
    disabled = filter_sources(sources, SETTINGS.model_copy(update={"enabled": False}), groups)
    assert len(disabled.kept) == 5 and disabled.excluded == []


def test_settings_are_validated_and_recorded() -> None:
    record = SETTINGS.record()
    assert record["version"] == "source-quality-v1" and record["settings"]["protected_items"] == ["1A", "7", "7A", "8"]
    with pytest.raises(ValueError):
        SourceQualitySettings(cross_reference_share=1.5)
    with pytest.raises(ValueError):
        SourceQualitySettings(unknown_rule=True)


# --- context rules (bge_base_ft_v3): openings that hide what a figure refers to -------------------

HEADLESS_TABLE = ("Future net cash flows | United States: 131,855 | Canada: 81,031 | Total: 287,559 [/TABLE] The "
                  "standardized measure is computed with year-end prices, as our reserves rules require.")
ROWS_ONLY = "3.095% notes due 2049 | 1,500 | 1,500 4.327% notes due 2050 | 2,750 | 2,750 Total long-term debt | 43,428"
DATED_TABLE = ("Leasehold improvements | 2023: 12,839 | 2022: 11,271 Total | 2023: $43,715 | 2022: $42,117 [/TABLE] "
               "Depreciation expense was $8.5 billion in 2023.")
FRAGMENT = ("locations and by lower fuel sales. Segment eCommerce sales contributed approximately 2.2% to comparable "
            "sales. Gross profit rate decreased for fiscal 2022 and 2021.")
FRAGMENT_IN_CONTEXT = ("are subject to U.S. taxation. Our federal tax provision for 2022 included a full election. "
                       "The provision was $2,175 million.")
FRAGMENT_QUALITATIVE = "its business practices are contrary to policy. Physical risks could harm our clients and us."


def test_context_rules_exclude_openings_that_hide_the_period_or_entity_only_when_enabled() -> None:
    on = {"context_rules": True}
    assert rules(HEADLESS_TABLE, item="8", **on) == ("headless_table_start",)  # caption and year cut off
    assert rules(ROWS_ONLY, item="8", **on) == ("headless_table_start",)  # rows only, no heading at all
    assert rules(DATED_TABLE, item="8", **on) == ()  # mid-table, but every column names its year
    assert rules(FRAGMENT, item="7", **on) == ("fragment_opening",)  # its first figure's year was in the last chunk
    assert rules(FRAGMENT_IN_CONTEXT, item="8", **on) == ()  # period and entity come before the first figure
    assert rules(FRAGMENT_QUALITATIVE, item="1A", **on) == ()  # no figure to misattribute
    assert rules(REVENUE_TABLE, item="7", **on) == ()  # a whole table
    for text in (HEADLESS_TABLE, ROWS_ONLY, FRAGMENT):  # off by default, so earlier plans reproduce
        assert rules(text, item="8") == ()
    assert SETTINGS.rules() == RULES and "context_rules" not in SETTINGS.record()["settings"]
    assert SourceQualitySettings(context_rules=True).record()["settings"]["context_rules"] is True
    off = filter_sources([passage(FRAGMENT, item="7")], SETTINGS, {})
    assert list(off.stats()["excluded_by_rule"]) == list(RULES) and off.kept
    on_result = filter_sources([passage(FRAGMENT, item="7")], SourceQualitySettings(context_rules=True), {})
    assert on_result.stats()["excluded_by_rule"]["fragment_opening"] == 1 and not on_result.kept


# --- administrative disclosures (insider trading, trading plans, cybersecurity governance) ----------

TRADING_PLAN = (
    "Item 9B. Other Information. Insider Trading Arrangements. On August 30, 2024, the Company's Chief Operating "
    "Officer entered into a trading plan intended to satisfy the affirmative defense conditions of Rule 10b5-1(c) "
    "under the Exchange Act. The plan provides for the sale of shares vesting under certain equity awards and will "
    "expire on December 15, 2025, subject to early termination."
)
CYBER_GOVERNANCE = (
    "Item 1C. Cybersecurity. We have processes for assessing, identifying and managing material risks from "
    "cybersecurity threats. Our Chief Information Security Officer leads the team responsible for cybersecurity and "
    "reports to the Audit Committee, which oversees our cybersecurity program and receives quarterly updates."
)
CYBER_RISK = (
    "A cybersecurity incident could disrupt our operations. Cybersecurity attacks on our systems, or on our "
    "vendors' systems, could expose customer data, lead to litigation and regulatory penalties, and materially "
    "adversely affect our revenue, results of operations and financial condition; our cybersecurity insurance may "
    "not cover every loss."
)
INSIDER_POLICY_WITH_FIGURES = (
    "We repurchased $4,210 million of common stock in 2024 and $3,950 million in 2023 under the program; purchases "
    "follow our insider trading policy, Rule 10b5-1 trading plans and blackout periods, and averaged $182.40 per "
    "share against $171.15 a year earlier, while dividends rose 4.5% to $2,905 million."
)


def test_the_administrative_rule_excludes_governance_disclosures_only_when_enabled() -> None:
    on = {"administrative_rules": True}
    assert rules(TRADING_PLAN, item="9B", **on) == ("administrative_disclosure",)
    assert rules(CYBER_GOVERNANCE, item="1C", **on) == ("administrative_disclosure",)
    assert rules(CYBER_GOVERNANCE, item="", **on) == ("administrative_disclosure",)  # its markers, not its item
    assert rules(CYBER_RISK, item="1A", **on) == ()  # a risk factor is protected
    assert rules(INSIDER_POLICY_WITH_FIGURES, item="5", **on) == ()  # mostly financial figures
    assert rules(REVENUE_TABLE, item="9B", **on) == ()  # no governance marker
    for text, item in ((TRADING_PLAN, "9B"), (CYBER_GOVERNANCE, "1C")):  # off by default, so earlier plans reproduce
        assert "administrative_disclosure" not in rules(text, item=item)
    assert not {"administrative_rules", "administrative_min_markers", "administrative_max_figures"} & set(
        SETTINGS.record()["settings"]
    )
    enabled = SourceQualitySettings(administrative_rules=True)
    assert enabled.rules()[-1] == "administrative_disclosure" and enabled.record()["settings"]["administrative_rules"]
    result = filter_sources([passage(TRADING_PLAN, item="9B"), passage(RISK, item="1A", pid="p2")], enabled, {})
    assert [p.passage_id for p in result.kept] == ["p2"] and result.stats()["excluded_by_rule"]["administrative_disclosure"] == 1
