"""Pydantic v2 data contracts for FinRetriever.

These models are the single source of truth for every JSONL record that flows
between pipeline stages: filings on disk, parsed sections, retrieval passages,
training/eval queries, qrels, retrieval hits, and benchmark run results.

Schemas are intentionally explicit and ``extra="forbid"``: silent field drift
would invalidate the gold set and benchmark outputs that this project commits
to git. Add fields here first, then update the pipeline — never the other way
around.

This module also exposes deterministic ID factories (``make_*_id``) so that
the same filing/section/passage/query always hashes to the same string across
runs and machines, which is required for stable qrels.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MetadataValue = str | int | float | bool | None


_STRICT = ConfigDict(extra="forbid")


class Filing(BaseModel):
    """A single SEC filing as downloaded from EDGAR (one accession number)."""

    model_config = _STRICT

    filing_id: str
    ticker: str
    company_name: str | None = None
    form_type: str
    accession_number: str
    filing_date: str
    fiscal_year: int | None = None
    source_url: str | None = None
    local_path: str | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("filing_date")
    @classmethod
    def _non_empty_filing_date(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("filing_date must be non-empty")
        return v

    @field_validator("accession_number")
    @classmethod
    def _non_empty_accession(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("accession_number must be non-empty")
        return v


class Section(BaseModel):
    """A logical section of a filing (e.g. Item 1A, Item 7 MD&A)."""

    model_config = _STRICT

    section_id: str
    filing_id: str
    title: str
    item_number: str | None = None
    text: str
    start_char: int | None = None
    end_char: int | None = None

    @field_validator("title", "text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v

    @field_validator("start_char", "end_char")
    @classmethod
    def _non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("character offset must be >= 0")
        return v

    @model_validator(mode="after")
    def _ordered_span(self) -> Section:
        if self.start_char is not None and self.end_char is not None:
            if self.end_char < self.start_char:
                raise ValueError("end_char must be >= start_char")
        return self


class Passage(BaseModel):
    """The retrieval unit: a chunk of text inside a single filing/section."""

    model_config = _STRICT

    passage_id: str
    filing_id: str
    section_id: str | None = None
    ticker: str
    form_type: str
    filing_date: str
    section_title: str | None = None
    text: str
    token_count: int | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be non-empty")
        return v

    @field_validator("token_count")
    @classmethod
    def _positive_token_count(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("token_count must be > 0")
        return v


class Query(BaseModel):
    """A retrieval query (synthetic, gold, manual, etc.)."""

    model_config = _STRICT

    query_id: str
    text: str
    source: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must be non-empty")
        return v

    @field_validator("source")
    @classmethod
    def _non_empty_source(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source must be non-empty")
        return v


class Qrel(BaseModel):
    """A TREC-style relevance judgement for a (query, passage) pair."""

    model_config = _STRICT

    query_id: str
    passage_id: str
    relevance: int

    @field_validator("relevance")
    @classmethod
    def _non_negative_relevance(cls, v: int) -> int:
        if v < 0:
            raise ValueError("relevance must be >= 0")
        return v


class SearchHit(BaseModel):
    """A single retrieved (passage, score) result for a query."""

    model_config = _STRICT

    query_id: str
    passage_id: str
    rank: int
    score: float
    retriever: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("rank")
    @classmethod
    def _rank_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rank must be >= 1")
        return v

    @field_validator("retriever")
    @classmethod
    def _non_empty_retriever(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("retriever must be non-empty")
        return v


class RunResult(BaseModel):
    """Output of one retriever over the full gold set: hits + headline metrics."""

    model_config = _STRICT

    run_name: str
    metrics: dict[str, float]
    hits: list[SearchHit] = Field(default_factory=list)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("run_name")
    @classmethod
    def _non_empty_run_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("run_name must be non-empty")
        return v


# ---------------------------------------------------------------------------
# Deterministic ID factories
# ---------------------------------------------------------------------------
#
# Every ID is `prefix_<readable bits>_<sha1[:12]>`. The hash is computed over
# the *normalised* inputs so case differences in tickers/form types don't
# produce different IDs for the same underlying record. Python's built-in
# ``hash()`` is process-salted and must not be used here.

_HASH_LEN = 12
_SEP = "\x1f"  # ASCII unit separator — never appears in any sensible input.


def _sha1_short(*parts: str) -> str:
    payload = _SEP.join(parts).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:_HASH_LEN]


def make_filing_id(ticker: str, form_type: str, accession_number: str) -> str:
    t = ticker.strip().lower()
    f = form_type.strip().lower()
    a = accession_number.strip()
    return f"filing_{t}_{f}_{_sha1_short(t, f, a)}"


def make_section_id(filing_id: str, title: str, index: int) -> str:
    digest = _sha1_short(filing_id, title.strip().lower(), str(index))
    return f"section_{filing_id}_{index:04d}_{digest}"


def make_passage_id(filing_id: str, section_id: str | None, chunk_index: int) -> str:
    digest = _sha1_short(filing_id, section_id or "", str(chunk_index))
    return f"passage_{filing_id}_{chunk_index:06d}_{digest}"


def make_query_id(text: str, source: str) -> str:
    s = source.strip().lower()
    t = text.strip()
    return f"query_{s}_{_sha1_short(s, t)}"


__all__ = [
    "Filing",
    "Passage",
    "Qrel",
    "Query",
    "RunResult",
    "SearchHit",
    "Section",
    "make_filing_id",
    "make_passage_id",
    "make_query_id",
    "make_section_id",
]
