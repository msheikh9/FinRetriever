"""Source-passage quality filter: which approved passages are worth asking an LLM about.

The rules are generic and text-based (no company names or company-specific lists) and look only at
the tokenizer-visible text the generator would see, the passage's 10-K item number and how often
its text repeats across filings (its overlap group). They change nothing in the preparation: the
split and blocklist stay as they are, and a filtered passage simply gives no synthetic queries.

A passage is excluded by the first rule that matches, in this order (every match is counted):

* ``exhibit_index``: exhibit-list markers (exhibit numbers and descriptions, "incorporated by
  reference to/from", "filed/furnished herewith", certifications, indentures, XBRL documents,
  references to other SEC forms and agreements) at ``exhibit_marker_density`` or more per 100 words;
* ``cross_reference``: prose that only points elsewhere. At least ``cross_reference_share`` of the
  words outside tables sit in sentences that refer to another document or section ("incorporated
  by reference", "see Note 5", "information required by this Item", "is included in"), and the
  passage has fewer than ``cross_reference_max_figures`` financial figures per 100 words;
* ``repetitive_administrative``: administrative boilerplate (controls and procedures,
  certifications, signatures, "not applicable" items, executive-officer rosters) at
  ``administrative_marker_density`` or more per 100 words, repeated nearly verbatim in at least
  ``administrative_min_filings`` filings;
* ``low_financial_content``: finance terms below ``min_finance_term_share`` of the words, fewer
  than ``min_figures_per_100_words`` financial figures and fewer than ``min_table_numbers_per_100_words``
  numbers in tables (so operating tables of plain counts, or patent-expiry years, are kept). Never
  applied to ``protected_items`` (risk factors, MD&A, market risk and financial statements by
  default), whose legal and qualitative passages are kept.

Financial figures are dollar amounts, percentages, numbers with thousands separators and decimals;
bare years do not count. Table numbers are every number inside ``[TABLE] ... [/TABLE]`` except
ages in officer rosters.

With ``context_rules`` (off by default, so earlier plans reproduce), two more rules follow, for
passages whose opening cannot tell a reader what a figure or statement refers to (the main
source of factual errors in the validation pilot, docs/validation_pilot_v1_quality_report.md):

* ``headless_table_start``: the visible text starts inside a table (rows before any ``[TABLE]``, or
  rows only, so the table's caption and header row are cut off) and no column heading of those
  rows names a period (a year, or a month and day), so neither the period nor the column a
  figure belongs to can be read;
* ``fragment_opening``: the visible text starts mid-sentence and the text before its first
  financial figure names no period (a year, or a month and day) or no entity (the company's name
  or ticker, "we", "our", "the Company", "the Firm", "the Corporation"): the context of the first
  figure was in the previous chunk. A passage with no financial figure has none to misattribute.

With ``administrative_rules`` (off by default, so earlier plans reproduce), one more rule follows, for
disclosures that are governance rather than finance (none of the four in ``val_pilot_v2e``'s first batch
gave a usable question, docs/validation_pilot_v2e_batch1_audit.md):

* ``administrative_disclosure``: insider-trading policies, Rule 10b5-1 trading plans and arrangements,
  and cybersecurity governance. The passage has fewer than ``administrative_max_figures`` financial
  figures per 100 words and holds at least ``administrative_min_markers`` of their markers, or at least
  one in the items that hold them (1C cybersecurity, 9B other information, 10 governance). Never applied
  to ``protected_items``: a cybersecurity risk factor, an accounting policy or MD&A stays.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from finretriever.training.query_generation import SourcePassage

SOURCE_QUALITY_VERSION = "source-quality-v1"
RULES: tuple[str, ...] = (
    "exhibit_index",
    "cross_reference",
    "repetitive_administrative",
    "low_financial_content",
)
CONTEXT_RULES: tuple[str, ...] = ("headless_table_start", "fragment_opening")
ADMINISTRATIVE_RULES: tuple[str, ...] = ("administrative_disclosure",)
ADMINISTRATIVE_ITEMS = ("1C", "9B", "10")
_ADMINISTRATIVE_SETTINGS = ("administrative_rules", "administrative_min_markers", "administrative_max_figures")


class SourceQualitySettings(BaseModel):
    """``source_quality`` in the data-preparation config; thresholds are per 100 visible words."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    exhibit_marker_density: float = Field(default=2.5, gt=0)
    cross_reference_share: float = Field(default=0.5, gt=0, le=1)
    cross_reference_max_figures: float = Field(default=2.0, ge=0)
    administrative_marker_density: float = Field(default=1.5, gt=0)
    administrative_min_filings: int = Field(default=3, ge=1)
    min_finance_term_share: float = Field(default=0.005, ge=0, le=1)
    min_figures_per_100_words: float = Field(default=1.0, ge=0)
    min_table_numbers_per_100_words: float = Field(default=3.0, ge=0)
    protected_items: list[str] = Field(default_factory=lambda: ["1A", "7", "7A", "8"])
    context_rules: bool = False  # headless_table_start and fragment_opening
    administrative_rules: bool = False  # administrative_disclosure
    administrative_min_markers: int = Field(default=3, ge=1)
    administrative_max_figures: float = Field(default=2.0, ge=0)

    def rules(self) -> tuple[str, ...]:
        """The rules in force, in the order they exclude."""
        return (
            *RULES,
            *(CONTEXT_RULES if self.context_rules else ()),
            *(ADMINISTRATIVE_RULES if self.administrative_rules else ()),
        )

    def record(self) -> dict[str, Any]:
        """What a plan records about the filter: the rule version and every setting (the context and
        administrative rules' only when they are on, so earlier plans reproduce)."""
        settings = self.model_dump(mode="json")
        if not self.context_rules:
            settings.pop("context_rules")
        if not self.administrative_rules:
            for name in _ADMINISTRATIVE_SETTINGS:
                settings.pop(name)
        return {"version": SOURCE_QUALITY_VERSION, "settings": settings}


_WORD_RE = re.compile(r"[a-z][a-z'-]*|\d[\d,.]*")
_TABLE_RE = re.compile(r"\[table\].*?(?:\[/table\]|$)", flags=re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.;])\s+")
_NUMBER_RE = re.compile(r"\d[\d,.]*")
_AGE_RE = re.compile(r"\bage(?: \([^)]*\))?: \d{2}\b")
_FIGURE_RE = re.compile(r"\$\s?\d|\d[\d,]*\.?\d*\s?%|\b\d{1,3}(?:,\d{3})+\b|\b\d+\.\d+\b")
_EXHIBIT_RE = re.compile(
    r"exhibit number|exhibit description|\bexhibits? \d+(?:\.\d+)?|incorporated (?:herein )?by reference "
    r"(?:from|to)|filed herewith|furnished herewith|certifications? of|\(file no\.|officer.s certificate|"
    r"\bindentures?\b|\bxbrl\b|amended and restated|\bform (?:8-k|10-q|10-k|s-\d)|\bs-\d/a\b|registration "
    r"statement|current report on|quarterly report on|annual report on form|\bpower of attorney|\bbylaws\b|"
    r"by-laws|certificate of incorporation|\bagreement\b"
)
_CROSS_REFERENCE_RE = re.compile(
    r"incorporated (?:herein )?by reference|(?:is|are|will be) (?:set forth|included|contained|presented|"
    r"provided|discussed|described) (?:in|under|on)|\bsee (?:part|item|note|page|the (?:information|section|"
    r"discussion|caption))|\brefer to\b|information required by (?:this )?item|\bpages? \d+"
)
_ADMINISTRATIVE_RE = re.compile(
    r"check mark|pursuant to the requirements|/s/|power of attorney|\bcertif(?:y|ies|ied|ication)|rule "
    r"13a-1[45]|section 1350|sarbanes|mine safety|not applicable|unresolved staff comments|disclosure "
    r"controls and procedures|internal control over financial reporting|treadway|accelerated filer|"
    r"commission file|\bage: \d{2}\b|\bage \((?:at|as of)\b|held current title|executive officers? of the registrant|"
    r"family relationships|principal executive officer|principal financial officer"
)
_GOVERNANCE_RE = re.compile(
    r"insider trading|10b5-1|trading arrangements?|trading plans?|cybersecurity|"
    r"information security officer|\bciso\b"
)
_PERIOD_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+\d{1,2}\b"
)
_ENTITY_RE = re.compile(r"\b(?:we|our|us|the company|the firm|the corporation|the registrant)\b")
FINANCE_TERMS = frozenset(
    """revenue revenues sales income earnings profit profits loss losses margin margins cash liquidity
    debt borrowings borrowing capital assets asset liabilities liability equity expense expenses cost
    costs tax taxes interest dividend dividends shares repurchase repurchases credit loan loans deposit
    deposits investment investments securities derivative derivatives hedge hedging fair value
    impairment goodwill risk risks market markets price prices pricing demand competition competitive
    regulation regulatory financial operating operations segment segments growth decline increase
    increased decrease decreased billion million percent net gross reserves production volume volumes
    customers customer adverse adversely results condition fund funding rate rates exposure portfolio
    allowance provision obligations lease leases payable receivable receivables inventory inventories
    depreciation amortization acquisition acquisitions patent patents exclusivity expiration expire
    expires barrels crude oil gas capacity""".split()
)


@dataclass(frozen=True)
class SourceFeatures:
    """What the rules measure on a passage's visible text."""

    words: int
    exhibit_markers_per_100_words: float
    cross_reference_share: float
    administrative_markers_per_100_words: float
    finance_term_share: float
    figures_per_100_words: float
    table_numbers_per_100_words: float
    repeat_filings: int
    headless_table_start: bool = False
    fragment_without_context: bool = False
    governance_markers: int = 0  # insider-trading, trading-plan and cybersecurity-governance markers


def opening_context(text: str, company: str, ticker: str) -> tuple[bool, bool]:
    """The two context rules' tests on visible text: (it starts inside a table whose visible column
    headings name no period; it starts mid-sentence and names no period or no entity before its first
    financial figure)."""
    t = text.lstrip()
    opening, closing = t.find("[TABLE]"), t.find("[/TABLE]")
    inside = closing != -1 and (opening == -1 or closing < opening)
    rows_only = opening == closing == -1 and " | " in t[:300]
    if (inside or rows_only) and not t.startswith("[TABLE]"):
        fragment = t[:closing] if inside else t
        if " | " in fragment:
            cells = re.split(r" \| |\n", fragment)
            headings = " ".join(c.rsplit(":", 1)[0] for c in cells if ":" in c).casefold()
            return not _PERIOD_RE.search(headings), False
    if not t or not (t[0].islower() or t[0] in ",;:)]%"):
        return False, False
    folded = t.casefold()
    figure = _FIGURE_RE.search(folded)
    if figure is None:
        return False, False
    before = folded[: figure.start()]
    names = {ticker.casefold(), *(w for w in re.findall(r"[a-z]+", company.casefold())[:1] if len(w) > 2)}
    entity = bool(_ENTITY_RE.search(before)) or any(re.search(rf"\b{re.escape(n)}", before) for n in names if n)
    return False, not (_PERIOD_RE.search(before) and entity)


def source_features(
    text: str, repeat_filings: int, *, company: str = "", ticker: str = ""
) -> SourceFeatures:
    t = text.casefold()
    words = _WORD_RE.findall(t)
    n = max(len(words), 1)
    prose = _TABLE_RE.sub(" ", t)
    tables = " ".join(m.group(0) for m in _TABLE_RE.finditer(t))
    table_numbers = len(_NUMBER_RE.findall(_AGE_RE.sub(" ", tables)))
    prose_words = max(len(_WORD_RE.findall(prose)), 1)
    reference_words = sum(
        len(_WORD_RE.findall(s)) for s in _SENTENCE_RE.split(prose) if _CROSS_REFERENCE_RE.search(s)
    )
    return SourceFeatures(
        words=len(words),
        exhibit_markers_per_100_words=100 * len(_EXHIBIT_RE.findall(t)) / n,
        cross_reference_share=min(reference_words / prose_words, 1.0),
        administrative_markers_per_100_words=100 * len(_ADMINISTRATIVE_RE.findall(t)) / n,
        finance_term_share=sum(w in FINANCE_TERMS for w in words) / n,
        figures_per_100_words=100 * len(_FIGURE_RE.findall(t)) / n,
        table_numbers_per_100_words=100 * table_numbers / n,
        repeat_filings=repeat_filings,
        governance_markers=len(_GOVERNANCE_RE.findall(t)),
        **dict(zip(("headless_table_start", "fragment_without_context"),
                   opening_context(text, company, ticker), strict=True)),
    )


def matching_rules(
    features: SourceFeatures, item: str, settings: SourceQualitySettings
) -> tuple[str, ...]:
    """Every rule the passage matches, in rule order (the first one excludes it)."""
    f, s = features, settings
    matched = {
        "exhibit_index": f.exhibit_markers_per_100_words >= s.exhibit_marker_density,
        "cross_reference": f.cross_reference_share >= s.cross_reference_share
        and f.figures_per_100_words < s.cross_reference_max_figures,
        "repetitive_administrative": f.administrative_markers_per_100_words
        >= s.administrative_marker_density
        and f.repeat_filings >= s.administrative_min_filings,
        "low_financial_content": item not in s.protected_items
        and f.finance_term_share < s.min_finance_term_share
        and f.figures_per_100_words < s.min_figures_per_100_words
        and f.table_numbers_per_100_words < s.min_table_numbers_per_100_words,
        "headless_table_start": f.headless_table_start,
        "fragment_opening": f.fragment_without_context,
        "administrative_disclosure": item not in s.protected_items
        and f.figures_per_100_words < s.administrative_max_figures
        and (
            f.governance_markers >= s.administrative_min_markers
            or (item in ADMINISTRATIVE_ITEMS and f.governance_markers >= 1)
        ),
    }
    return tuple(rule for rule in s.rules() if matched[rule])


def filings_per_group(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """For each overlap group, how many filings (company and fiscal year) its passages come from."""
    filings: dict[str, set[tuple[str, str]]] = {}
    for row in rows.values():
        filings.setdefault(row["group_id"], set()).add((row["ticker"], row["fiscal_year"]))
    return {group: len(found) for group, found in filings.items()}


def item_kind(item: str) -> str:
    """A coarse section label for reports: the 10-K item, or "other" when the parser found none."""
    return f"Item {item}" if item else "other"


@dataclass(frozen=True)
class SourceQualityResult:
    kept: list[SourcePassage]
    excluded: list[tuple[SourcePassage, tuple[str, ...]]]  # with every rule each matched
    rules: tuple[str, ...] = RULES  # the rules in force

    def excluded_ids(self) -> dict[str, tuple[str, ...]]:
        return {p.passage_id: rules for p, rules in self.excluded}

    def stats(self) -> dict[str, Any]:
        """Counts per rule (first match and any match), and kept/excluded by section and company."""
        first = Counter(rules[0] for _, rules in self.excluded)
        any_match = Counter(rule for _, rules in self.excluded for rule in rules)
        excluded_items = Counter(item_kind(p.item) for p, _ in self.excluded)
        return {
            "approved": len(self.kept) + len(self.excluded),
            "kept": len(self.kept),
            "excluded": len(self.excluded),
            "excluded_by_rule": {rule: first[rule] for rule in self.rules},
            "matched_by_rule": {rule: any_match[rule] for rule in self.rules},
            "kept_by_ticker": dict(sorted(Counter(p.ticker for p in self.kept).items())),
            "kept_by_item": dict(sorted(Counter(item_kind(p.item) for p in self.kept).items())),
            "excluded_by_item": dict(sorted(excluded_items.items())),
        }


def filter_sources(
    sources: Sequence[SourcePassage],
    settings: SourceQualitySettings,
    group_filings: Mapping[str, int],
) -> SourceQualityResult:
    """Approved passages split into kept and excluded (with the rules each matched)."""
    if not settings.enabled:
        return SourceQualityResult(list(sources), [], settings.rules())
    kept, excluded = [], []
    for passage in sources:
        features = source_features(
            passage.visible_text, group_filings.get(passage.group_id, 1), company=passage.company,
            ticker=passage.ticker,
        )
        rules = matching_rules(features, passage.item, settings)
        if rules:
            excluded.append((passage, rules))
        else:
            kept.append(passage)
    return SourceQualityResult(kept, excluded, settings.rules())


__all__ = [
    "ADMINISTRATIVE_ITEMS",
    "ADMINISTRATIVE_RULES",
    "CONTEXT_RULES",
    "FINANCE_TERMS",
    "RULES",
    "SOURCE_QUALITY_VERSION",
    "SourceFeatures",
    "SourceQualityResult",
    "SourceQualitySettings",
    "filings_per_group",
    "filter_sources",
    "item_kind",
    "matching_rules",
    "opening_context",
    "source_features",
]
