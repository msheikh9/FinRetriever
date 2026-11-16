"""Passage views: the text a retriever indexes for a passage, without changing the passage.

The corpus is never modified. A *view* renders the text that a retriever encodes (dense) or tokenizes (BM25)
for each passage; hits always carry the original passage ID and metadata.

* ``text`` (default): the passage text as stored. Every existing run and embedding cache uses this view.
* ``header_v1``: a one-line metadata header (company, ticker, form, fiscal year, section) followed by the text.
  59% of passages never name their company (docs/retrieval_analysis_v1.md, S1); the header puts the filing's
  identity into the indexed text. The header is built from the passage's own metadata only.

A view other than ``text`` changes the dense embedding cache key (``encoded_field``), so enriched and plain
embeddings can never be confused.
"""

from __future__ import annotations

from finretriever.data.schemas import Passage

PASSAGE_VIEWS = ("text", "header_v1")


def passage_fiscal_year(passage: Passage) -> int:
    """A passage's fiscal year: the year of its filing's period end (``report_date``), else of its filing date.

    The same rule as :func:`finretriever.evaluation.alignment.fiscal_year`."""
    return int(str(passage.metadata.get("report_date") or passage.filing_date)[:4])


def section_label(passage: Passage) -> str:
    item = passage.metadata.get("item_number")
    title = passage.section_title or str(passage.metadata.get("section_title") or "").strip() or "Unknown section"
    return f"Item {item}. {title}" if item else title


def header_v1(passage: Passage) -> str:
    company = str(passage.metadata.get("company_name") or passage.ticker)
    return (
        f"Company: {company} ({passage.ticker}). Filing: {passage.form_type} for fiscal year "
        f"{passage_fiscal_year(passage)}. Section: {section_label(passage)}."
    )


def render_passage(passage: Passage, view: str = "text") -> str:
    """The text a retriever indexes for ``passage`` under ``view``."""
    if view == "text":
        return passage.text
    if view == "header_v1":
        return f"{header_v1(passage)}\n{passage.text}"
    raise ValueError(f"unknown passage view {view!r}; use one of {list(PASSAGE_VIEWS)}")


def encoded_field(view: str) -> str:
    """The ``encoded_field`` an embedding cache key records for ``view`` (``passage.text`` for the default)."""
    if view not in PASSAGE_VIEWS:
        raise ValueError(f"unknown passage view {view!r}; use one of {list(PASSAGE_VIEWS)}")
    return "passage.text" if view == "text" else f"{view}+passage.text"


__all__ = ["PASSAGE_VIEWS", "encoded_field", "header_v1", "passage_fiscal_year", "render_passage", "section_label"]
