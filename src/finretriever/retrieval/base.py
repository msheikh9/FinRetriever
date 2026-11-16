"""Retriever interface shared by every FinRetriever baseline and model.

This is a :class:`typing.Protocol`, not an abstract base class. The benchmark
harness only cares that an object exposes ``index``, ``search``, and
``batch_search`` with compatible signatures — it does not care about an
inheritance hierarchy.

Company-constrained retrieval is shared by every retriever through
:func:`eligible_indices` and :func:`rank_top_k`: a retriever scores passages
with its usual scoring function, keeps only the passages of the allowed
companies, and ranks those. It never takes a global top-k and filters it
afterwards, so a constrained run always returns up to ``top_k`` eligible
passages.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, runtime_checkable

from finretriever.data.schemas import Passage, SearchHit


@runtime_checkable
class BaseRetriever(Protocol):
    """Minimal contract every FinRetriever retriever satisfies.

    Implementations may accept extra keyword arguments (e.g. a per-query
    ``query_id``, or ``tickers`` for company-constrained retrieval) as long as
    they remain callable through the signatures declared here.
    """

    def index(self, passages: Sequence[Passage]) -> None:
        """Build any necessary in-memory indices from the given passages."""
        ...

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Return up to ``top_k`` :class:`SearchHit` records for ``query``."""
        ...

    def batch_search(
        self, queries: Mapping[str, str], top_k: int = 10
    ) -> list[SearchHit]:
        """Run ``search`` for every (query_id, query_text) pair in order."""
        ...


def eligible_indices(
    passage_tickers: Sequence[str], tickers: Collection[str] | None
) -> list[int] | None:
    """Corpus positions a query may retrieve: ``None`` (every passage) without a restriction.

    Raises ``ValueError`` for a ticker that no passage carries, so a typo can
    never silently turn into an empty or unrestricted search.
    """
    if not tickers:
        return None
    allowed = set(tickers)
    if unknown := sorted(allowed - set(passage_tickers)):
        raise ValueError(f"no passages for ticker(s) {unknown}")
    return [i for i, ticker in enumerate(passage_tickers) if ticker in allowed]


def hit_metadata(passage: Passage) -> dict[str, str | int | float | bool | None]:
    """The passage fields every retriever copies onto its hits."""
    metadata: dict[str, str | int | float | bool | None] = {
        "ticker": passage.ticker,
        "form_type": passage.form_type,
        "filing_date": passage.filing_date,
    }
    if passage.section_title is not None:
        metadata["section_title"] = passage.section_title
    return metadata


def rank_top_k(
    scores: Sequence[float], top_k: int, eligible: Sequence[int] | None = None
) -> list[int]:
    """Positions of the ``top_k`` highest scores among ``eligible`` (all if ``None``).

    Ties are broken by corpus position, so the same scores always give the same
    ranking on any host.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}.")
    candidates = range(len(scores)) if eligible is None else eligible
    order = sorted(candidates, key=lambda i: (-float(scores[i]), i))
    return order[:top_k]


__all__ = ["BaseRetriever", "eligible_indices", "hit_metadata", "rank_top_k"]
