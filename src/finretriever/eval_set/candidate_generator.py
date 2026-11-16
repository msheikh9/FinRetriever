"""Candidate passages for a manually written evaluation question.

Annotators judge only what they are shown, so the pool decides which relevant
passages can ever be found. Four sources are available, and each candidate
records which one proposed it (``CandidateSource``) so the pool is auditable:

* ``bm25`` — the existing BM25 retriever over the whole frozen corpus, with
  punctuation-insensitive tokens, run on the question and on any
  reformulations the annotator writes; optionally restricted to the
  question's companies.
* ``manual`` — passage IDs the annotator found by browsing or searching.
* ``stratified`` — a seeded sample from every in-scope filing, so filings that
  BM25 ranks low (other years, other companies) are still looked at.
* ``random`` — a seeded uniform sample, a rough check on what BM25 misses.

Dense retrievers are not a source. The fine-tuned FinRetriever model must
never become one: a pool built by the system under test favours that system.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from finretriever.data.schemas import Passage
from finretriever.eval_set.schemas import Candidate, CandidateSource, GoldQuestion
from finretriever.retrieval.bm25 import WORD_TOKEN_RE, BM25Retriever, _PurePythonBM25

_POOL_TOKEN_RE = WORD_TOKEN_RE


class _PoolBM25(BM25Retriever):
    """``BM25Retriever`` as used for pooling: fixed backend, punctuation-insensitive tokens.

    The baseline splits on whitespace, so "supplier?" in a question never
    matches "supplier." in a passage; the pool needs that recall (numbers such
    as "1,200" stay whole). The project's pure-Python backend (non-negative
    Lucene IDF) is always used, so a pool does not depend on whether
    ``rank_bm25`` happens to be installed. The benchmark's BM25 baseline is
    unchanged.
    """

    def _tokenize(self, text: str) -> list[str]:
        return _POOL_TOKEN_RE.findall(text.lower())

    def index(self, passages: Sequence[Passage]) -> None:
        self.passages = list(passages)
        self.passage_ids = [p.passage_id for p in self.passages]
        self.tokenized_corpus = [self._tokenize(p.text) for p in self.passages]
        self._backend = _PurePythonBM25(self.tokenized_corpus)
        self._indexed = True


def fiscal_year_of(passage: Passage) -> int:
    """Fiscal year of a passage's filing: the year of its period-of-report date.

    Companies name fiscal years by the year they end in (Walmart's year ending
    January 2025 is "fiscal 2025"); the filing date is the fallback when the
    report date is missing.
    """
    report_date = passage.metadata.get("report_date")
    return int(str(report_date or passage.filing_date)[:4])


@dataclass(frozen=True)
class Scope:
    """Which passages a candidate source may propose. Empty fields mean "any"."""

    tickers: frozenset[str] = frozenset()
    fiscal_years: frozenset[int] = frozenset()

    @classmethod
    def of(cls, question: GoldQuestion, *, use_years: bool = False) -> Scope:
        # Years are opt-in: later filings restate earlier years' figures, so a
        # year filter can hide relevant comparative passages.
        return cls(frozenset(question.tickers), frozenset(question.fiscal_years if use_years else ()))

    @property
    def restricted(self) -> bool:
        return bool(self.tickers or self.fiscal_years)

    def admits(self, passage: Passage) -> bool:
        return (not self.tickers or passage.ticker in self.tickers) and (
            not self.fiscal_years or fiscal_year_of(passage) in self.fiscal_years
        )

    def describe(self) -> str:
        parts = []
        if self.tickers:
            parts.append(f"tickers={','.join(sorted(self.tickers))}")
        if self.fiscal_years:
            parts.append(f"fiscal_years={','.join(map(str, sorted(self.fiscal_years)))}")
        return "; ".join(parts) or "whole corpus"


WHOLE_CORPUS = Scope()


class CandidateGenerator:
    """Proposes candidates from a corpus; it never judges relevance."""

    def __init__(self, passages: Sequence[Passage] | Mapping[str, Passage]) -> None:
        self._passages = list(passages.values() if isinstance(passages, Mapping) else passages)
        if not self._passages:
            raise ValueError("cannot generate candidates from an empty corpus")
        self._by_id = {p.passage_id: p for p in self._passages}
        if len(self._by_id) != len(self._passages):
            raise ValueError("duplicate passage IDs in the corpus")
        self._bm25: _PoolBM25 | None = None

    def passage(self, passage_id: str) -> Passage:
        return self._by_id[passage_id]

    def _retriever(self) -> _PoolBM25:
        if self._bm25 is None:
            self._bm25 = _PoolBM25(retriever_name="bm25_pool")
            self._bm25.index(self._passages)
        return self._bm25

    def bm25(self, text: str, *, depth: int, scope: Scope = WHOLE_CORPUS) -> list[Candidate]:
        """The top ``depth`` in-scope passages for ``text``; zero-score matches are dropped."""
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        retriever = self._retriever()
        hits = retriever.search(text, top_k=len(self._passages) if scope.restricted else depth)
        kept = [h for h in hits if h.score > 0 and scope.admits(self._by_id[h.passage_id])][:depth]
        detail = f"query={text.strip()!r}; {scope.describe()}; depth={depth}"
        return [
            Candidate(
                passage_id=hit.passage_id,
                source=CandidateSource(method="bm25", rank=rank, score=round(hit.score, 4), detail=detail),
            )
            for rank, hit in enumerate(kept, start=1)
        ]

    def manual(self, passage_ids: Iterable[str], *, note: str = "") -> list[Candidate]:
        """Candidates for passage IDs an annotator looked up; unknown IDs raise ``ValueError``."""
        ids = list(dict.fromkeys(passage_ids))
        unknown = [pid for pid in ids if pid not in self._by_id]
        if unknown:
            raise ValueError(f"passage IDs not in the corpus: {unknown}")
        source = CandidateSource(method="manual", detail=note or "manual lookup")
        return [Candidate(passage_id=pid, source=source) for pid in ids]

    def stratified(self, per_filing: int, *, seed: int | str, scope: Scope = WHOLE_CORPUS) -> list[Candidate]:
        """Up to ``per_filing`` seeded random passages from every in-scope filing."""
        by_filing: dict[str, list[Passage]] = {}
        for passage in self._passages:
            if scope.admits(passage):
                by_filing.setdefault(passage.filing_id, []).append(passage)
        candidates: list[Candidate] = []
        for filing_id, passages in by_filing.items():
            chosen = random.Random(f"{seed}:{filing_id}").sample(passages, min(per_filing, len(passages)))
            first = passages[0]
            detail = (
                f"seed={seed}; filing={first.ticker} {first.filing_date}; "
                f"{len(chosen)} of {len(passages)}; {scope.describe()}"
            )
            order = {p.passage_id: i for i, p in enumerate(passages)}
            for passage in sorted(chosen, key=lambda p: order[p.passage_id]):
                candidates.append(
                    Candidate(passage_id=passage.passage_id, source=CandidateSource(method="stratified", detail=detail))
                )
        return candidates

    def random(
        self, count: int, *, seed: int | str, scope: Scope = WHOLE_CORPUS, exclude: Collection[str] = ()
    ) -> list[Candidate]:
        """``count`` seeded uniform samples from the in-scope passages not in ``exclude``."""
        eligible = [p for p in self._passages if scope.admits(p) and p.passage_id not in exclude]
        chosen = random.Random(seed).sample(eligible, min(count, len(eligible)))
        detail = f"seed={seed}; {len(chosen)} of {len(eligible)}; {scope.describe()}"
        return [
            Candidate(passage_id=p.passage_id, source=CandidateSource(method="random", detail=detail))
            for p in chosen
        ]


def build_pool(
    generator: CandidateGenerator,
    question: GoldQuestion,
    *,
    bm25_depth: int,
    reformulations: Sequence[str] = (),
    manual_ids: Sequence[str] = (),
    manual_note: str = "",
    stratified_per_filing: int = 0,
    random_count: int = 0,
    seed: int | str = 0,
    use_years: bool = False,
) -> list[Candidate]:
    """Candidates for ``question`` from every requested source, in display order.

    BM25 runs on the question and then on each annotator-written
    reformulation; samples are seeded by ``seed`` and the query ID, so the
    same request always proposes the same passages. Duplicates are kept here
    (each with its own source) and merged when they join the pool.
    """
    scope = Scope.of(question, use_years=use_years)
    candidates = generator.bm25(question.text, depth=bm25_depth, scope=scope)
    for text in reformulations:
        candidates += generator.bm25(text, depth=bm25_depth, scope=scope)
    query_seed = f"{seed}:{question.query_id}"
    if stratified_per_filing:
        candidates += generator.stratified(stratified_per_filing, seed=query_seed, scope=scope)
    if random_count:
        exclude = {c.passage_id for c in candidates}
        candidates += generator.random(random_count, seed=query_seed, scope=scope, exclude=exclude)
    if manual_ids:
        candidates += generator.manual(manual_ids, note=manual_note)
    return candidates


__all__ = ["WHOLE_CORPUS", "CandidateGenerator", "Scope", "build_pool", "fiscal_year_of"]
