"""BM25 baseline retriever for FinRetriever.

BM25 is the first baseline in the v1 benchmark: a strong, parameter-free
lexical retriever that every dense model needs to beat to justify its
existence on SEC text.

This module:

* Prefers ``rank_bm25.BM25Okapi`` when installed (the de facto reference).
* Falls back to a small pure-Python BM25 implementation when it is not, so
  the benchmark remains runnable on minimal environments. The fallback uses
  a Lucene-style non-negative IDF (``log(1 + (N - df + 0.5) / (df + 0.5))``)
  so a term that appears in many small-corpus documents still contributes a
  small positive signal instead of penalising matching docs — which is the
  behaviour FinRetriever's baseline needs on a 50-filing corpus.
* Wraps either backend behind a single deterministic :class:`BM25Retriever`
  that produces :class:`finretriever.data.schemas.SearchHit` records.

Two settings are configurable; their defaults keep the original behaviour:

* ``tokenizer``: ``"whitespace"`` (default) splits on whitespace, so
  punctuation stays attached ("OPEC," never matches "OPEC"); ``"word"`` keeps
  runs of word characters, with numbers such as "1,200" whole — the analyzer
  the candidate pools were built with.
* ``backend``: ``"auto"`` (default) uses ``rank_bm25`` when installed;
  ``"python"`` always uses the pure-Python engine, so results do not depend
  on the environment; ``"rank_bm25"`` requires the package.

Determinism is non-negotiable. Score ties are broken by the corpus insertion
order so the same ``(corpus, query, top_k)`` always yields the same ranking
on any host.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from importlib import metadata as importlib_metadata

from finretriever.data.chunker import simple_tokenize
from finretriever.data.schemas import Passage, SearchHit
from finretriever.retrieval.base import eligible_indices, hit_metadata, rank_top_k
from finretriever.retrieval.enrichment import PASSAGE_VIEWS, render_passage

try:  # pragma: no cover - exercised by import side effects
    from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]

    _HAS_RANK_BM25 = True
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment,misc]
    _HAS_RANK_BM25 = False


_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75

TOKENIZERS = ("whitespace", "word")
BACKENDS = ("auto", "python", "rank_bm25")
# Runs of word characters; a number keeps its separators ("1,200", "3.5").
WORD_TOKEN_RE = re.compile(r"\w+(?:[.,]\d+)*")


class _PurePythonBM25:
    """Minimal BM25 implementation used when ``rank_bm25`` is absent.

    The IDF formula is the Lucene-style non-negative variant
    ``log(1 + (N - df + 0.5) / (df + 0.5))``. This deliberately differs from
    ``rank_bm25.BM25Okapi``'s ``log(N - df + 0.5) - log(df + 0.5)``, which
    can go negative for very common terms and cause matching documents to
    rank *below* documents with zero overlap on a small corpus — pathological
    behaviour for FinRetriever's baseline. The chosen formula keeps every
    term contribution >= 0, so adding query tokens never penalises a doc
    that contains them.
    """

    def __init__(
        self,
        tokenized_corpus: list[list[str]],
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        total_tokens = sum(self.doc_lengths)
        self.avgdl = (
            total_tokens / self.corpus_size if self.corpus_size > 0 else 0.0
        )

        self.term_freqs: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for doc in tokenized_corpus:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.term_freqs.append(tf)
            for term in tf:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        self.idf: dict[str, float] = {}
        for term, df in document_frequency.items():
            # Lucene-style non-negative IDF: log(1 + (N - df + 0.5) / (df + 0.5)).
            # Always >= 0 because the inner ratio is > 0 for any valid df.
            self.idf[term] = math.log(
                1.0 + (self.corpus_size - df + 0.5) / (df + 0.5)
            )

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for doc_index in range(self.corpus_size):
            doc_len = self.doc_lengths[doc_index]
            tf_dict = self.term_freqs[doc_index]
            for term in query_tokens:
                idf = self.idf.get(term)
                if idf is None:
                    continue
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue
                if self.avgdl == 0:
                    denom = float(tf)
                else:
                    denom = tf + self.k1 * (
                        1.0 - self.b + self.b * doc_len / self.avgdl
                    )
                scores[doc_index] += idf * tf * (self.k1 + 1) / denom
        return scores


class BM25Retriever:
    """BM25 baseline that indexes :class:`Passage` records and returns hits."""

    def __init__(
        self,
        retriever_name: str = "bm25",
        lowercase: bool = True,
        tokenizer: str = "whitespace",
        backend: str = "auto",
        passage_view: str = "text",
    ) -> None:
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {TOKENIZERS}, got {tokenizer!r}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if backend == "rank_bm25" and not _HAS_RANK_BM25:
            raise ValueError("backend 'rank_bm25' requested but rank_bm25 is not installed")
        if passage_view not in PASSAGE_VIEWS:
            raise ValueError(f"passage_view must be one of {list(PASSAGE_VIEWS)}, got {passage_view!r}")
        self.passage_view = passage_view  # what is tokenized for each passage (finretriever.retrieval.enrichment)
        self.retriever_name = retriever_name
        self.lowercase = lowercase
        self.tokenizer = tokenizer
        self.backend = backend
        self.passages: list[Passage] = []
        self.passage_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self._backend: BM25Okapi | _PurePythonBM25 | None = None
        self._indexed = False

    @property
    def effective_backend(self) -> str:
        """The engine ``index`` uses: ``"rank_bm25"`` or ``"python"``."""
        if self.backend == "auto":
            return "rank_bm25" if _HAS_RANK_BM25 else "python"
        return self.backend

    def describe(self) -> dict[str, str | float | bool | None]:
        """Settings that determine this retriever's ranking, for run provenance."""
        engine = self.effective_backend
        rank_bm25 = engine == "rank_bm25"
        idf = "BM25Okapi (epsilon floor)" if rank_bm25 else "log(1 + (N - df + 0.5) / (df + 0.5))"
        return {
            "name": self.retriever_name,
            "type": "bm25",
            "tokenizer": self.tokenizer,
            "lowercase": self.lowercase,
            "backend": engine,
            "idf": idf,
            "k1": _DEFAULT_K1,
            "b": _DEFAULT_B,
            "backend_version": importlib_metadata.version("rank-bm25") if rank_bm25 else None,
            **({"passage_view": self.passage_view} if self.passage_view != "text" else {}),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    def __len__(self) -> int:
        return len(self.passages)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        if self.tokenizer == "word":
            return WORD_TOKEN_RE.findall(text)
        return simple_tokenize(text)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, passages: Sequence[Passage]) -> None:
        if len(passages) == 0:
            raise ValueError("Cannot index an empty passage list.")

        seen: set[str] = set()
        for passage in passages:
            if passage.passage_id in seen:
                raise ValueError(f"Duplicate passage_id in corpus: {passage.passage_id!r}")
            seen.add(passage.passage_id)

        # Copy the list (not the Passage objects) so callers can mutate their
        # own list reference without affecting the retriever, and so we never
        # mutate the input list.
        self.passages = list(passages)
        self.passage_ids = [p.passage_id for p in self.passages]
        self.tokenized_corpus = [self._tokenize(render_passage(p, self.passage_view)) for p in self.passages]

        if self.effective_backend == "rank_bm25":
            self._backend = BM25Okapi(self.tokenized_corpus)
        else:
            self._backend = _PurePythonBM25(self.tokenized_corpus)

        self._indexed = True

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def scores(self, query: str) -> list[float]:
        """The BM25 score of every indexed passage for ``query`` (corpus order; all zero without query tokens)."""
        if not self._indexed or self._backend is None:
            raise RuntimeError("BM25Retriever.scores() called before index().")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [0.0] * len(self.passages)
        return [float(x) for x in self._backend.get_scores(query_tokens)]

    def search(
        self,
        query: str,
        top_k: int = 10,
        query_id: str = "query",
        tickers: Collection[str] | None = None,
    ) -> list[SearchHit]:
        """Top ``top_k`` passages; with ``tickers``, only those companies' passages are ranked."""
        if not self._indexed or self._backend is None:
            raise RuntimeError("BM25Retriever.search() called before index().")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}.")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            raw_scores: list[float] = [0.0] * len(self.passages)
        else:
            raw_scores = list(self._backend.get_scores(query_tokens))

        # Sort by score descending; tie-break by original corpus index so the
        # output is deterministic even when every score is zero. A company
        # restriction removes ineligible passages before ranking.
        eligible = eligible_indices([p.ticker for p in self.passages], tickers)
        order = rank_top_k(raw_scores, top_k, eligible)

        hits: list[SearchHit] = []
        for rank, doc_index in enumerate(order, start=1):
            passage = self.passages[doc_index]
            hits.append(
                SearchHit(
                    query_id=query_id,
                    passage_id=passage.passage_id,
                    rank=rank,
                    score=float(raw_scores[doc_index]),
                    retriever=self.retriever_name,
                    metadata=hit_metadata(passage),
                )
            )
        return hits

    def batch_search(
        self,
        queries: Mapping[str, str],
        top_k: int = 10,
        tickers: Mapping[str, Collection[str]] | None = None,
    ) -> list[SearchHit]:
        """``search`` for every query; ``tickers`` maps a query ID to its allowed companies."""
        restrictions = tickers or {}
        all_hits: list[SearchHit] = []
        for query_id, query_text in queries.items():
            all_hits.extend(
                self.search(
                    query_text, top_k=top_k, query_id=query_id, tickers=restrictions.get(query_id)
                )
            )
        return all_hits


__all__ = ["BACKENDS", "TOKENIZERS", "WORD_TOKEN_RE", "BM25Retriever"]
