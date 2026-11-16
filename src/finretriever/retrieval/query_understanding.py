"""Query understanding: the companies, fiscal years and financial measures a question asks about.

A deterministic, rule-based parser (no model, no API). It reads only the question text and the corpus's own
company list, and returns a :class:`QueryScope` that says what it found and how it found it, so that a
retrieval strategy can decide whether a constraint is justified:

* **Companies.** Each corpus company is recognised by its ticker, the first word of its registered name and any
  configured extra aliases (the same alias rule the training-data checks use,
  :class:`finretriever.training.query_generation.CompanyAliases`). Several companies are all kept.
* **Fiscal years.** ``FY2024`` / ``FY 24``, ``fiscal 2024`` / ``fiscal year 2024``, period-end dates (``year ended
  January 31, 2024`` is Walmart's fiscal 2024, the year of the period end, as in the corpus), and plain years.
  ``from 2019 through 2023`` and ``2019-2023`` are inclusive ranges; ``between 2023 and 2025`` and ``2023 versus
  2022`` keep the years named. A corpus fiscal year is the year of the filing's period end
  (:func:`finretriever.evaluation.alignment.fiscal_year`), so these are directly comparable.
* **Financial measures.** A small lexicon (revenue, net income, cash flow, debt, ...), recorded for diagnostics
  and for deciding whether a question asks for a figure; it does not constrain retrieval.
* **Comparison.** ``cross_company`` with two or more companies, else ``cross_year`` with two or more years, else
  ``none``; ``comparative`` flags comparison language ("compare", "versus", "change", ...).

* **Year roles** (version ``qu-v2``). Not every year in a question is a reporting period. Each year mention gets a
  role from the words around it (:func:`classify_year_roles`):

  ===============  =====================================================================  ==================
  role             examples                                                               reporting period?
  ===============  =====================================================================  ==================
  ``fiscal``       "in fiscal 2023", "as of December 31, 2022", "2023 versus 2022"        yes
  ``target``       "reduce emissions by 2030", "2025 goals", "outlook for 2026",          no
                   "committed to deliver from 2026 through 2028"
  ``historical``   "since 2015", "compared to 2016 levels", the baseline of a target      no
                   ("by 2025 compared to 2016")
  ``maturity``     "notes due 2030", "matures in 2027", "2031 Notes", "expire in 2028"    no
  ``incidental``   "Tax Cuts and Jobs Act of 2017", "the 2019 Stock Incentive Plan",      no
                   "the 2018 CEO Performance Award", "the 2008 financial crisis"
  ===============  =====================================================================  ==================

  Only ``fiscal`` years become :attr:`QueryScope.years` and so a year constraint; the others are recorded with
  their role. A year with no cue keeps the ``qu-v1`` reading (fiscal): roles only ever *remove* a year from the
  constraint, never add one, and a question whose years are all non-reporting gets no year constraint (the
  fallback is recorded). Version ``qu-v1`` (no roles) is kept unchanged so that the published v2 experiments
  re-run byte for byte.

The parser never guesses: a question that names no company gets no company constraint, one that names no year
gets no year constraint (``fallback`` reasons are recorded). Versions: :data:`PARSER_VERSIONS`, default
:data:`PARSER_VERSION`; a change to the rules must add a version.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from finretriever.data.schemas import Passage
from finretriever.training.query_generation import CompanyAliases

PARSER_VERSIONS = ("qu-v1", "qu-v2")  # qu-v1: every year is a fiscal year (v2 experiments); qu-v2: year roles
PARSER_VERSION = "qu-v2"
YEAR_ROLES = ("fiscal", "target", "historical", "maturity", "incidental")
YEAR_MIN, YEAR_MAX = 1990, 2039  # plausible fiscal years; other four-digit numbers are not years
MAX_RANGE = 12  # an inclusive range longer than this is treated as two separate years

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_YEAR4 = r"(?:19|20)\d\d"
_FY = rf"(?:fy\s?'?(?:{_YEAR4}|\d\d)|fiscal(?:\s+year)?\s+{_YEAR4})"
_ANY_YEAR = rf"(?:{_FY}|{_YEAR4})"
_DASH = r"\s*(?:-|–|—)\s*"

# inclusive ranges: "from 2019 through 2023", "2019-2023", "FY2021-FY2023", "fiscal 2019 to 2021"
_RANGE_RES = (
    re.compile(rf"\bfrom\s+(?P<a>{_ANY_YEAR})\s+(?:to|through|until|thru)\s+(?P<b>{_ANY_YEAR})\b", re.I),
    re.compile(rf"(?<![\d$.,])(?P<a>{_ANY_YEAR}){_DASH}(?P<b>{_ANY_YEAR}|\d\d)(?![\d,.]\d)", re.I),
    re.compile(rf"\b(?P<a>{_FY})\s+(?:to|through|thru)\s+(?P<b>{_ANY_YEAR})\b", re.I),
)
# accounting-standard and similar codes whose digits are not fiscal years ("ASU 2016-02", "ASC 842", "No. 2023-07")
_CODE_RE = re.compile(r"\b(?:ASU|ASC|SFAS|FAS|FASB|IFRS|Update|No\.)\s*(?:No\.\s*)?\d{4}(?:-\d{1,3})?", re.I)
_FY_RE = re.compile(rf"\bfy\s?'?(?P<y>{_YEAR4}|\d\d)\b", re.I)
_FISCAL_RE = re.compile(rf"\bfiscal(?:\s+years?)?\s+(?P<y>{_YEAR4})\b", re.I)
_DATE_RE = re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+(?P<y>{_YEAR4})\b", re.I)
_BARE_RE = re.compile(rf"(?<![\d$.,/])(?P<y>{_YEAR4})(?![\d]|[,.]\d)")
_COMPARATIVE_RE = re.compile(
    r"\b(compar\w*|versus|vs\.?|relative to|change[ds]?|chang\w+|evolv\w+|trend\w*|grow\w*|grew|increas\w+|"
    r"decreas\w+|declin\w+|differ\w*|between|over the|each year|year[- ]over[- ]year)\b",
    re.I,
)
_FIGURE_RE = re.compile(
    r"\b(how much|how many|what (?:was|were|is|are) the (?:total|amount|value|number|percentage|ratio)|"
    r"in (?:millions|billions|thousands)|percent(?:age)?|ratio|margin)\b|%",
    re.I,
)

# canonical measure -> pattern (lower case, word-bounded); diagnostic only
MEASURES: dict[str, str] = {
    "revenue": r"revenues?|net sales|sales",
    "net_income": r"net income|net earnings|net loss",
    "operating_income": r"operating income|operating profit|income from operations",
    "margin": r"(?:gross|operating|profit|net) margins?|margins?",
    "eps": r"earnings per share|\beps\b",
    "cash_flow": r"cash flows?|cash provided by|free cash flow",
    "capital_expenditure": r"capital expenditures?|capital spending|capex|purchases of property",
    "debt": r"long-term debt|debt|borrowings?|credit facilit(?:y|ies)|notes payable|commercial paper",
    "assets": r"total assets|current assets|assets",
    "liabilities": r"liabilit(?:y|ies)",
    "equity": r"(?:shareholders|stockholders)['’]? equity|equity",
    "research_and_development": r"research and development|r&d",
    "tax": r"income tax(?:es)?|tax rate|deferred tax|taxes",
    "dividends": r"dividends?",
    "share_repurchases": r"repurchases?|buybacks?",
    "employees": r"employees|headcount|workforce",
    "segment": r"segments?",
    "liquidity_capital": r"liquidity|capital ratio|cet1|tier 1",
    "inventory": r"inventor(?:y|ies)",
    "goodwill_intangibles": r"goodwill|intangible",
    "leases": r"leases?",
    "pension": r"pensions?|postretirement",
    "impairment_restructuring": r"impairments?|restructuring",
}
_MEASURE_RES = {name: re.compile(rf"(?<![a-z]){pattern}(?![a-z])", re.I) for name, pattern in MEASURES.items()}


@dataclass(frozen=True)
class QueryScope:
    """What a question asks about, as read from its text by :class:`QueryParser`."""

    text: str
    tickers: tuple[str, ...]
    years: tuple[int, ...]
    measures: tuple[str, ...]
    comparison: str  # none | cross_year | cross_company
    comparative: bool
    asks_for_figure: bool
    company_mentions: tuple[tuple[str, str], ...] = ()  # (ticker, surface text)
    year_mentions: tuple[YearMention, ...] = ()
    fallbacks: tuple[str, ...] = field(default=())
    parser_version: str = PARSER_VERSION

    @property
    def has_company(self) -> bool:
        return bool(self.tickers)

    @property
    def has_year(self) -> bool:
        return bool(self.years)

    @property
    def non_reporting_years(self) -> dict[str, tuple[int, ...]]:
        """Years named in a role other than ``fiscal`` (qu-v2), by role: recorded, never a year constraint."""
        out: dict[str, set[int]] = {}
        for m in self.year_mentions:
            if not m.reporting:
                out.setdefault(str(m.role), set()).update(m.years)
        return {role: tuple(sorted(years)) for role, years in sorted(out.items())}

    @property
    def cells(self) -> list[tuple[str, int]]:
        """Company-year cells (every company times every year); empty unless both are known."""
        return [(t, y) for t in self.tickers for y in self.years]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tickers": list(self.tickers),
            "years": list(self.years),
            "measures": list(self.measures),
            "comparison": self.comparison,
            "comparative": self.comparative,
            "asks_for_figure": self.asks_for_figure,
            "company_mentions": [list(m) for m in self.company_mentions],
            "year_mentions": [m.to_dict() for m in self.year_mentions],
            "fallbacks": list(self.fallbacks),
            "parser_version": self.parser_version,
        }


def _year(token: str, *, base: int | None = None) -> int | None:
    """A year from ``2024``, ``FY24``, ``fiscal 2024``; a two-digit year is 20xx (or ``base``'s century)."""
    digits = re.findall(r"\d+", token)
    if not digits:
        return None
    raw = digits[-1]
    if len(raw) == 2:
        century = (base // 100 * 100) if base else 2000
        value = century + int(raw)
    elif len(raw) == 4:
        value = int(raw)
    else:
        return None
    return value if YEAR_MIN <= value <= YEAR_MAX else None


@dataclass(frozen=True)
class YearMention:
    """A year expression found in a question: the rule that matched, its text and the years it names."""

    rule: str  # range | fy | fiscal | date | year; qu-v2 also "future" (2040-2100, never a fiscal year)
    surface: str
    years: tuple[int, ...]
    role: str | None = None  # qu-v2: one of YEAR_ROLES; None under qu-v1 (every year is read as fiscal)
    span: tuple[int, int] = field(default=(-1, -1), compare=False)  # character offsets in the question

    @property
    def reporting(self) -> bool:
        """Whether this mention names a reporting period (and so may constrain retrieval)."""
        return self.role in (None, "fiscal")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"rule": self.rule, "surface": self.surface, "years": list(self.years)}
        if self.role is not None:
            out["role"] = self.role
        return out

    @property
    def explicit_fiscal(self) -> bool:
        """``FY2024`` / ``fiscal 2024`` (or a range of them) name a fiscal year outright; a date or a plain
        year may also be a filing-date or forward-looking reference (a FY2025 10-K gives "holders of record as
        of January 28, 2026")."""
        return self.rule in ("fy", "fiscal") or (self.rule == "range" and bool(re.search(r"fy|fiscal", self.surface, re.I)))


def extract_years(text: str) -> tuple[list[int], list[YearMention]]:
    """The fiscal years a question names, sorted, and every mention that named them."""
    years: set[int] = set()
    mentions: list[YearMention] = []
    covered: list[tuple[int, int]] = [m.span() for m in _CODE_RE.finditer(text)]  # never years

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in covered)

    for pattern in _RANGE_RES:
        for m in pattern.finditer(text):
            if not free(m.span()):
                continue
            a = _year(m.group("a"))
            b = _year(m.group("b"), base=a)
            if a is None or b is None or b <= a:  # "2016-02" is not a range of years
                continue
            named = tuple(range(a, b + 1)) if b - a <= MAX_RANGE else (a, b)
            years.update(named)
            mentions.append(YearMention("range", m.group(0), named, span=m.span()))
            covered.append(m.span())
    for rule, pattern in (("fy", _FY_RE), ("fiscal", _FISCAL_RE), ("date", _DATE_RE), ("year", _BARE_RE)):
        for m in pattern.finditer(text):
            if not free(m.span()):
                continue
            y = _year(m.group("y"))
            if y is None:
                continue
            years.add(y)
            mentions.append(YearMention(rule, m.group(0), (y,), span=m.span()))
            covered.append(m.span())
    return sorted(years), mentions


def extract_measures(text: str) -> list[str]:
    return [name for name, pattern in _MEASURE_RES.items() if pattern.search(text)]


# ---- year roles (qu-v2) --------------------------------------------------------------------------------------
# "Before" patterns end where the mention starts; "after" patterns start where it ends. A cue must sit next to
# the year: a role is assigned only on explicit local evidence, otherwise the year stays fiscal (as in qu-v1).
_FP = r"(?:fiscal\s+(?:years?\s+)?|fy\s?)?"
_INCIDENTAL_AFTER_NAME = re.compile(  # "the 2018 CEO Performance Award", "the 1969 Petroleum Act" (capitalised)
    r"\s+(?:[A-Z][\w&'.-]*\s+){0,4}(?:Act|Plan|Award|Agreement|Program|Programme|Facility|Settlement|Accord)\b")
_INCIDENTAL_AFTER = re.compile(
    r"\s+(?:(?:(?:stock|equity|omnibus|incentive|employee|share|performance|long[- ]term|compensation)\s+){1,3}"
    r"(?:plan|program|award)s?\b|(?:share\s+|stock\s+)?(?:repurchase|buyback)\s+(?:program|plan|authori[sz]ation)\b"
    r"|(?:tax\s+)?act\b|(?:financial\s+|banking\s+)?crisis\b|pandemic\b|recession\b|hurricane\b|winter\s+storm\b"
    r"|tax\s+(?:reform|legislation)\b)",
    re.I,
)
_INCIDENTAL_BEFORE = re.compile(r"\b(?:act|law|code|plan|regulation|directive|amendment)\s+of\s+$", re.I)
_MATURITY_BEFORE = re.compile(
    r"\b(?:due|matur(?:e|es|ed|ing|ity|ities)|expir(?:e|es|ed|ing|ation|y)|redeemable|callable|repayable|payable)"
    rf"\s+(?:(?:in|on|by|after|before|through|from|between|beginning|starting|date|of)\s+){{0,2}}{_FP}$",
    re.I,
)
_MATURITY_AFTER = re.compile(
    r"\s+(?:(?:senior|subordinated|convertible|fixed[- ]rate|floating[- ]rate|unsecured|secured)\s+)*"
    r"(?:notes?(?!\s+to\b)|bonds?|debentures?|term\s+loans?|maturities|debt\s+maturities)\b",
    re.I,
)
_HISTORICAL_BEFORE = re.compile(
    r"\b(?:since|founded\s+in|incorporated\s+in|inception\s+in|went\s+public\s+in|ipo\s+in|"
    rf"dating\s+back\s+to)\s+(?:(?:the\s+)?(?:end|start|beginning)\s+of\s+)?{_FP}$",
    re.I,
)
_HISTORICAL_AFTER = re.compile(r"\s*(?:levels?|baselines?|base[- ]year)\b", re.I)
_BASELINE_BEFORE = re.compile(r"\b(?:baselines?|base[- ]year)\s+(?:(?:of|year|in|from)\s+)?$", re.I)
_COMPARED_BEFORE = re.compile(  # the baseline of a target: "by 2025 compared to 2016", "by 2050 ... from 2024"
    rf"\b(?:compared?\s+(?:to|with)|relative\s+to|versus|vs\.?|from|against)\s+(?:(?:the|its|a|their)\s+)?{_FP}$",
    re.I,
)
_TARGET_BY = re.compile(rf"\bby\s+(?:the\s+end\s+of\s+)?{_FP}$", re.I)
_FORWARD_CUE = re.compile(  # "by <year>" is a target only in forward-looking questions
    r"\b(?:goals?|targets?|targeted|aims?|aiming|commit\w*|pledg\w*|ambitions?|aspir\w*|net[- ]zero|carbon|"
    r"neutral\w*|renewable|emissions?|project(?:s|ed|ions?)?|forecast\w*|outlook|expect\w*|anticipat\w*|"
    r"planned|intends?)\b",
    re.I,
)
_TARGET_BEFORE = re.compile(
    r"\b(?:goals?|targets?|ambitions?|commitments?|aspirations?|outlook|guidance|forecasts?|projections?|plans?)"
    rf"\s+(?:for|of|by|in|through|to)\s+{_FP}$",
    re.I,
)
_TARGET_NOUNS = r"(?:goals?|targets?|ambitions?|commitments?|aspirations?|outlook|guidance|forecasts?|projections?)\b"
_TARGET_AFTER = re.compile(rf"\s*{_TARGET_NOUNS}", re.I)
_TARGET_AFTER_GAP = re.compile(  # "2030 carbon neutrality goals": up to two content words between (plain years only)
    r"\s+(?:(?!(?:the|a|an|what|which|how|who|when|did|does|do|was|were|is|are|its|their|and|or|in|for|of|to|by)\b)"
    rf"[a-z][\w-]*\s+){{1,2}}{_TARGET_NOUNS}",
    re.I,
)
_FORWARD_VERB_BEFORE = re.compile(  # "committed to deliver worldwide from 2026 through 2028"
    r"\b(?:expects?|expected|anticipates?|anticipated|projects?|projected|forecasts?|forecasted|"
    r"intends?\s+to|committed\s+to|scheduled\s+to)(?:\s+[\w'’.-]+){0,6}?\s+"
    rf"(?:in|for|by|during|from|through|beginning\s+in|starting\s+in)\s+{_FP}$",
    re.I,
)
_FAR_FUTURE_RE = re.compile(r"(?<![\d$.,/])(?P<y>20[4-9]\d|2100)(?![\d]|[,.]\d)")  # beyond any fiscal year
_LEADING_PREPOSITION = re.compile(r"(?:from|between)\s+", re.I)  # part of a range mention's own text
_LIST_GAP = re.compile(r"[\s,]*(?:(?:and|or|to|through|thru|until|&|–|—|-)\s*)?", re.I)


def _own_role(text: str, mention: YearMention) -> str | None:
    """The role the words next to one mention give it, or None (no local cue)."""
    start, end = mention.span
    lead = _LEADING_PREPOSITION.match(mention.surface)
    before = text[:start] + (lead.group(0) if lead else "")
    after = text[end:]
    if _INCIDENTAL_AFTER_NAME.match(after) or _INCIDENTAL_AFTER.match(after) or _INCIDENTAL_BEFORE.search(before):
        return "incidental"
    if _MATURITY_BEFORE.search(before) or _MATURITY_AFTER.match(after):
        return "maturity"
    if _HISTORICAL_BEFORE.search(before) or _HISTORICAL_AFTER.match(after) or _BASELINE_BEFORE.search(before):
        return "historical"
    if ((_TARGET_BY.search(before) and _FORWARD_CUE.search(text)) or _TARGET_BEFORE.search(before)
            or _TARGET_AFTER.match(after) or _FORWARD_VERB_BEFORE.search(before)
            or (not mention.explicit_fiscal and _TARGET_AFTER_GAP.match(after))):
        return "target"
    return None


def classify_year_roles(text: str, mentions: Sequence[YearMention]) -> list[YearMention]:
    """``mentions`` (from :func:`extract_years` on ``text``) with a role each, in the same order.

    0. Years 2040-2100, which :func:`extract_years` ignores, are added as mentions: no filing reports them, so
       they are targets unless a cue says otherwise ("notes due 2050" is a maturity).
    1. A local cue gives a mention its own role (incidental, maturity, historical or target).
    2. A mention without one, listed right after a non-fiscal mention ("notes due 2025 and 2030"), shares its role.
    3. In a question with a target year, a year it is compared with ("by 2025 compared to 2016") is its
       historical baseline.
    4. Everything else is fiscal: without explicit evidence a year is read as a reporting period, as in qu-v1.
    """
    if any(m.span[0] < 0 for m in mentions):
        raise ValueError("year roles need mention spans (use extract_years)")
    taken = [m.span for m in mentions]
    mentions = [*mentions, *(  # 2040-2100: never a filing's fiscal year (extract_years ignores them), often a target
        YearMention("future", m.group(0), (int(m.group("y")),), span=m.span()) for m in _FAR_FUTURE_RE.finditer(text)
        if all(m.end() <= a or m.start() >= b for a, b in taken))]
    order = sorted(range(len(mentions)), key=lambda i: mentions[i].span)
    roles: dict[int, str | None] = {i: _own_role(text, mentions[i]) for i in order}
    previous: int | None = None
    for i in order:
        if roles[i] is None and previous is not None and roles[previous] not in (None, "fiscal"):
            gap = text[mentions[previous].span[1]:mentions[i].span[0]]
            if _LIST_GAP.fullmatch(gap):
                roles[i] = roles[previous]
        previous = i
    if any(r == "target" for r in roles.values()) or any(m.rule == "future" for m in mentions):
        for i in order:
            if roles[i] is None:
                start = mentions[i].span[0]
                lead = _LEADING_PREPOSITION.match(mentions[i].surface)
                if _COMPARED_BEFORE.search(text[:start] + (lead.group(0) if lead else "")):
                    roles[i] = "historical"
    default = {i: "target" if m.rule == "future" else "fiscal" for i, m in enumerate(mentions)}
    return [YearMention(m.rule, m.surface, m.years, roles[i] or default[i], m.span) for i, m in enumerate(mentions)]


class QueryParser:
    """Parses questions against a fixed set of companies (normally the corpus's)."""

    def __init__(self, aliases: CompanyAliases, version: str = PARSER_VERSION) -> None:
        if version not in PARSER_VERSIONS:
            raise ValueError(f"parser version must be one of {list(PARSER_VERSIONS)}, got {version!r}")
        self.aliases = aliases
        self.version = version
        self._company_res = {
            ticker: [re.compile(rf"(?<![a-z0-9]){re.escape(alias)}", re.I) for alias in names]
            for ticker, names in sorted(aliases.by_ticker.items())
        }

    @classmethod
    def from_companies(
        cls, companies: Mapping[str, str], extra_aliases: Mapping[str, Sequence[str]] | None = None,
        version: str = PARSER_VERSION,
    ) -> QueryParser:
        """``companies`` maps ticker to registered name (for example ``{"AAPL": "Apple Inc."}``)."""
        return cls(CompanyAliases.from_companies(companies, extra_aliases), version)

    @classmethod
    def from_passages(
        cls, passages: Iterable[Passage], extra_aliases: Mapping[str, Sequence[str]] | None = None,
        version: str = PARSER_VERSION,
    ) -> QueryParser:
        companies: dict[str, str] = {}
        for p in passages:
            companies.setdefault(p.ticker, str(p.metadata.get("company_name") or p.ticker))
        return cls.from_companies(companies, extra_aliases, version)

    def describe(self) -> dict[str, Any]:
        return {"parser_version": self.version,
                "aliases": {t: list(a) for t, a in sorted(self.aliases.by_ticker.items())}}

    def parse(self, text: str) -> QueryScope:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("a question must be a non-empty string")
        company_mentions: list[tuple[str, str]] = []
        for ticker, patterns in self._company_res.items():
            for pattern in patterns:
                if m := pattern.search(text):
                    company_mentions.append((ticker, m.group(0)))
                    break
        tickers = tuple(sorted({t for t, _ in company_mentions}))
        years, year_mentions = extract_years(text)
        if self.version != "qu-v1":
            year_mentions = classify_year_roles(text, year_mentions)
            years = sorted({y for m in year_mentions if m.reporting for y in m.years})
        fallbacks = []
        if not tickers:
            fallbacks.append("no company named: no company constraint")
        if not years and year_mentions:
            roles = sorted({str(m.role) for m in year_mentions})
            fallbacks.append(f"years named only as {'/'.join(roles)} years, not reporting periods: no year constraint")
        elif not years:
            fallbacks.append("no fiscal year named: no year constraint")
        if len(tickers) >= 2:
            comparison = "cross_company"
        elif len(years) >= 2:
            comparison = "cross_year"
        else:
            comparison = "none"
        return QueryScope(
            text=text,
            tickers=tickers,
            years=tuple(years),
            measures=tuple(extract_measures(text)),
            comparison=comparison,
            comparative=bool(_COMPARATIVE_RE.search(text)) or comparison != "none",
            asks_for_figure=bool(_FIGURE_RE.search(text)),
            company_mentions=tuple(company_mentions),
            year_mentions=tuple(year_mentions),
            fallbacks=tuple(fallbacks),
            parser_version=self.version,
        )


__all__ = [
    "MEASURES",
    "PARSER_VERSION",
    "PARSER_VERSIONS",
    "YEAR_ROLES",
    "QueryParser",
    "QueryScope",
    "YearMention",
    "classify_year_roles",
    "extract_measures",
    "extract_years",
]
