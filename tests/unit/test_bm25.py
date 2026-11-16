"""Tests for finretriever.retrieval.bm25.

The BM25 baseline anchors the benchmark, so these tests pin down the
behaviour future retrievers and the benchmark harness depend on:

* indexing / re-indexing semantics and error handling;
* deterministic ranking and tie-breaking;
* SearchHit shape and metadata population;
* tokenisation behaviour (lowercase, punctuation);
* the pure-Python fallback engine, exercised directly.

Tests are agnostic to which backend (``rank_bm25`` or fallback) is active
unless the test specifically targets the fallback class.
"""

from __future__ import annotations

import math

import pytest

from finretriever.data.schemas import Passage, SearchHit
from finretriever.retrieval.bm25 import BM25Retriever, _PurePythonBM25


def _make_passage(
    passage_id: str,
    ticker: str,
    text: str,
    section_title: str | None = None,
) -> Passage:
    return Passage(
        passage_id=passage_id,
        filing_id=f"filing_{ticker.lower()}_2023",
        ticker=ticker,
        form_type="10-K",
        filing_date="2023-11-03",
        section_title=section_title,
        text=text,
    )


@pytest.fixture
def corpus() -> list[Passage]:
    return [
        _make_passage(
            "p1",
            "AAPL",
            "Apple supply chain risk in semiconductors and components",
            section_title="Item 1A",
        ),
        _make_passage(
            "p2",
            "JPM",
            "JPMorgan credit risk and loan loss provisions",
            section_title="Item 7",
        ),
        _make_passage(
            "p3",
            "TSLA",
            "Tesla manufacturing capacity at Fremont and Shanghai",
        ),
        _make_passage(
            "p4",
            "PFE",
            "Healthcare regulatory risk and FDA approval processes",
            section_title="Item 1A",
        ),
    ]


# ---------------------------------------------------------------------------
# Lifecycle / indexing
# ---------------------------------------------------------------------------


def test_retriever_starts_unindexed() -> None:
    r = BM25Retriever()
    assert r.is_indexed is False
    assert len(r) == 0


def test_index_stores_passages_and_len(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    assert r.is_indexed is True
    assert len(r) == len(corpus)
    assert r.passage_ids == [p.passage_id for p in corpus]


def test_index_rejects_empty_passage_list() -> None:
    r = BM25Retriever()
    with pytest.raises(ValueError, match="empty passage list"):
        r.index([])


def test_index_rejects_duplicate_passage_ids() -> None:
    r = BM25Retriever()
    dupes = [
        _make_passage("p1", "AAPL", "first text"),
        _make_passage("p1", "MSFT", "second text"),
    ]
    with pytest.raises(ValueError, match="Duplicate passage_id"):
        r.index(dupes)


# ---------------------------------------------------------------------------
# Search input validation
# ---------------------------------------------------------------------------


def test_search_before_index_raises_runtime_error() -> None:
    r = BM25Retriever()
    with pytest.raises(RuntimeError, match="before index"):
        r.search("anything")


def test_search_rejects_empty_query(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    with pytest.raises(ValueError, match="non-empty"):
        r.search("")
    with pytest.raises(ValueError, match="non-empty"):
        r.search("    ")


def test_search_rejects_non_positive_top_k(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    with pytest.raises(ValueError, match="top_k must be"):
        r.search("Apple", top_k=0)
    with pytest.raises(ValueError, match="top_k must be"):
        r.search("Apple", top_k=-1)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_search_returns_search_hit_objects(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    hits = r.search("Apple supply chain", top_k=3)
    assert hits
    assert all(isinstance(h, SearchHit) for h in hits)


def test_search_returns_at_most_top_k_hits(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    assert len(r.search("risk", top_k=2)) == 2
    assert len(r.search("risk", top_k=1)) == 1


def test_top_k_larger_than_corpus_returns_corpus_size(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    hits = r.search("risk", top_k=100)
    assert len(hits) == len(corpus)


def test_ranks_start_at_one_and_are_contiguous(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    hits = r.search("risk supply chain", top_k=4)
    assert [h.rank for h in hits] == [1, 2, 3, 4]


def test_search_hits_validate_through_pydantic(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    hits = r.search("Apple supply chain risk", top_k=2)
    # A clean round-trip through the schema proves the hit is Pydantic-valid.
    for h in hits:
        SearchHit.model_validate_json(h.model_dump_json())


# ---------------------------------------------------------------------------
# Ranking behaviour
# ---------------------------------------------------------------------------


def test_exact_match_outranks_unrelated(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    hits = r.search("Apple supply chain", top_k=4)
    assert hits[0].passage_id == "p1"
    # Tesla manufacturing capacity has no overlap with the query.
    apple_score = hits[0].score
    tesla_hit = next(h for h in hits if h.passage_id == "p3")
    assert apple_score > tesla_hit.score


def test_repeated_query_terms_scale_score_linearly(
    corpus: list[Passage],
) -> None:
    r = BM25Retriever()
    r.index(corpus)
    # "Apple" is unique to p1, so repeating it should scale that doc's score
    # linearly without changing which doc wins.
    single = r.search("Apple", top_k=1)
    triple = r.search("Apple Apple Apple", top_k=1)
    assert single[0].passage_id == "p1"
    assert triple[0].passage_id == "p1"
    # BM25 sums independently over query tokens, so the triple-query score
    # is exactly 3x the single-query score for the matching doc.
    assert math.isclose(triple[0].score, 3 * single[0].score, rel_tol=1e-9)


def test_common_query_term_still_ranks_matching_docs_above_unrelated(
    corpus: list[Passage],
) -> None:
    r = BM25Retriever()
    r.index(corpus)
    # "risk" appears in three of four docs (p1, p2, p4); p3 has zero overlap.
    # On a small corpus a naive Okapi IDF can go negative for such a common
    # term and push matching docs *below* p3, which would be wrong. The
    # non-negative IDF formula must keep p3 last.
    hits = r.search("risk risk risk", top_k=4)
    assert hits[3].passage_id == "p3"
    assert {h.passage_id for h in hits[:3]} == {"p1", "p2", "p4"}
    # And the matching docs must score strictly higher than the unrelated one.
    assert all(h.score > hits[3].score for h in hits[:3])


def test_tie_scores_break_deterministically_by_corpus_order(
    corpus: list[Passage],
) -> None:
    r = BM25Retriever()
    r.index(corpus)
    # Two corpus orderings differing only in their internal order should
    # produce different tie-broken results.
    reversed_corpus = list(reversed(corpus))
    r2 = BM25Retriever()
    r2.index(reversed_corpus)

    # A query whose only token is in no document yields all-zero scores.
    hits1 = [h.passage_id for h in r.search("zzznotaword", top_k=4)]
    hits2 = [h.passage_id for h in r2.search("zzznotaword", top_k=4)]
    assert hits1 == ["p1", "p2", "p3", "p4"]
    assert hits2 == ["p4", "p3", "p2", "p1"]


def test_all_zero_scores_return_deterministic_corpus_order(
    corpus: list[Passage],
) -> None:
    r = BM25Retriever()
    r.index(corpus)
    hits = r.search("xyzqwerty12345", top_k=4)
    assert len(hits) == 4
    assert [h.passage_id for h in hits] == ["p1", "p2", "p3", "p4"]
    assert all(h.score == 0.0 for h in hits)


# ---------------------------------------------------------------------------
# Metadata + retriever name
# ---------------------------------------------------------------------------


def test_metadata_includes_ticker_form_type_filing_date(
    corpus: list[Passage],
) -> None:
    r = BM25Retriever()
    r.index(corpus)
    hits = r.search("Apple", top_k=1)
    md = hits[0].metadata
    assert md["ticker"] == "AAPL"
    assert md["form_type"] == "10-K"
    assert md["filing_date"] == "2023-11-03"


def test_section_title_is_included_when_present(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    # p1 has section_title "Item 1A"; p3 has none.
    hits_with = r.search("Apple supply chain", top_k=1)
    assert hits_with[0].passage_id == "p1"
    assert hits_with[0].metadata.get("section_title") == "Item 1A"

    hits_without = r.search("Tesla manufacturing", top_k=1)
    assert hits_without[0].passage_id == "p3"
    assert "section_title" not in hits_without[0].metadata


def test_retriever_name_propagates_to_search_hit(corpus: list[Passage]) -> None:
    r = BM25Retriever(retriever_name="bm25_v1")
    r.index(corpus)
    hits = r.search("Apple", top_k=1)
    assert hits[0].retriever == "bm25_v1"


# ---------------------------------------------------------------------------
# batch_search
# ---------------------------------------------------------------------------


def test_batch_search_returns_flat_list(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    queries = {"q1": "Apple supply chain", "q2": "JPMorgan credit"}
    hits = r.batch_search(queries, top_k=2)
    assert isinstance(hits, list)
    assert all(isinstance(h, SearchHit) for h in hits)
    assert len(hits) == 4  # 2 queries * top_k 2


def test_batch_search_preserves_query_order_then_rank_order(
    corpus: list[Passage],
) -> None:
    r = BM25Retriever()
    r.index(corpus)
    queries = {"q1": "Apple supply chain", "q2": "JPMorgan credit"}
    hits = r.batch_search(queries, top_k=2)

    q1_hits = [h for h in hits if h.query_id == "q1"]
    q2_hits = [h for h in hits if h.query_id == "q2"]
    # All q1 hits appear before any q2 hits in the flat list.
    first_q2_index = next(i for i, h in enumerate(hits) if h.query_id == "q2")
    assert all(h.query_id == "q1" for h in hits[:first_q2_index])
    # Within each query, ranks are 1, 2, ...
    assert [h.rank for h in q1_hits] == [1, 2]
    assert [h.rank for h in q2_hits] == [1, 2]


def test_batch_search_assigns_query_id_from_mapping(corpus: list[Passage]) -> None:
    r = BM25Retriever()
    r.index(corpus)
    queries = {"alpha": "Apple", "beta": "JPMorgan"}
    hits = r.batch_search(queries, top_k=1)
    qids = {h.query_id for h in hits}
    assert qids == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Tokenisation behaviour
# ---------------------------------------------------------------------------


def test_lowercase_true_is_case_insensitive(corpus: list[Passage]) -> None:
    r = BM25Retriever(lowercase=True)
    r.index(corpus)
    hits = r.search("APPLE", top_k=1)
    assert hits[0].passage_id == "p1"
    assert hits[0].score > 0.0


def test_lowercase_false_preserves_case_sensitive_matching(
    corpus: list[Passage],
) -> None:
    r = BM25Retriever(lowercase=False)
    r.index(corpus)
    # "Apple" (capital A) is in the corpus; "apple" (lower) is not.
    cased = r.search("Apple", top_k=1)
    uncased = r.search("apple", top_k=1)
    assert cased[0].score > 0.0
    assert uncased[0].score == 0.0


def test_tokenization_handles_punctuation_without_crashing(
    corpus: list[Passage],
) -> None:
    r = BM25Retriever()
    r.index(corpus)
    hits = r.search("Apple's risk, factor — supply-chain!", top_k=4)
    assert len(hits) == 4
    assert all(isinstance(h, SearchHit) for h in hits)


# ---------------------------------------------------------------------------
# Non-mutation
# ---------------------------------------------------------------------------


def test_index_and_search_do_not_mutate_input_passages(
    corpus: list[Passage],
) -> None:
    snapshots = [p.model_dump() for p in corpus]
    r = BM25Retriever()
    r.index(corpus)
    r.search("Apple supply chain risk", top_k=4)
    assert [p.model_dump() for p in corpus] == snapshots


# ---------------------------------------------------------------------------
# Pure-Python fallback (exercised directly)
# ---------------------------------------------------------------------------


def test_pure_python_fallback_scores_relevant_docs_higher() -> None:
    bm25 = _PurePythonBM25(
        [
            ["apple", "supply", "chain", "risk"],
            ["tesla", "manufacturing", "capacity"],
            ["healthcare", "regulatory", "risk"],
        ]
    )
    scores = bm25.get_scores(["apple"])
    assert len(scores) == 3
    # Only doc 0 contains "apple"; the rest must score strictly lower.
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[1] == 0.0
    assert scores[2] == 0.0


def test_pure_python_fallback_returns_zeros_for_unknown_terms() -> None:
    bm25 = _PurePythonBM25([["apple", "supply"], ["tesla"]])
    scores = bm25.get_scores(["unknown-term-xyz"])
    assert scores == [0.0, 0.0]


def test_pure_python_fallback_handles_empty_document() -> None:
    bm25 = _PurePythonBM25([["risk", "factors"], []])
    scores = bm25.get_scores(["risk"])
    assert len(scores) == 2
    # The empty document contributes no term frequencies, so its score is 0.
    assert scores[1] == 0.0
    # The non-empty document with the matching term scores higher than the
    # empty one.
    assert scores[0] >= scores[1]


def test_pure_python_fallback_idf_values_are_non_negative() -> None:
    # A term ("risk") appears in every document — the worst case for the
    # naive Okapi formula, which would produce a negative IDF here. The
    # Lucene-style formula used by _PurePythonBM25 must keep it >= 0.
    bm25 = _PurePythonBM25(
        [
            ["risk", "apple", "supply"],
            ["risk", "jpmorgan", "credit"],
            ["risk", "healthcare", "regulatory"],
        ]
    )
    assert bm25.idf, "expected a non-empty IDF table"
    assert all(value >= 0.0 for value in bm25.idf.values())
    # And the all-docs term still contributes a strictly positive signal,
    # so matching docs cannot be penalised relative to non-matching ones.
    assert bm25.idf["risk"] > 0.0


def test_pure_python_fallback_common_term_does_not_invert_ranking() -> None:
    # Doc 0 contains the common term; doc 1 does not. Even though "risk"
    # appears in 2 of 3 docs, doc 0 must score strictly higher than doc 1.
    bm25 = _PurePythonBM25(
        [
            ["risk", "apple"],
            ["tesla", "manufacturing"],
            ["risk", "regulatory"],
        ]
    )
    scores = bm25.get_scores(["risk"])
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


# ---------------------------------------------------------------------------
# Configurable tokenizer and backend (defaults keep the original behaviour)
# ---------------------------------------------------------------------------


def test_defaults_are_the_original_whitespace_tokenizer_and_auto_backend() -> None:
    r = BM25Retriever()
    assert (r.tokenizer, r.backend) == ("whitespace", "auto")
    assert r._tokenize("OPEC, credit (LCR) 1,790.") == ["opec,", "credit", "(lcr)", "1,790."]


def test_word_tokenizer_matches_across_punctuation() -> None:
    passages = [
        _make_passage("p1", "XOM", "Decisions by OPEC, and other producers affect prices."),
        _make_passage("p2", "XOM", "Refining margins depend on regional demand."),
    ]
    whitespace = BM25Retriever(backend="python")
    whitespace.index(passages)
    assert whitespace.search("OPEC", top_k=1)[0].score == 0.0  # "opec," never matches "opec"
    word = BM25Retriever(tokenizer="word", backend="python")
    word.index(passages)
    top = word.search("OPEC?", top_k=1)[0]
    assert (top.passage_id, top.score > 0.0) == ("p1", True)
    assert word._tokenize("Revenue of $1,790 million in 2023.") == ["revenue", "of", "1,790", "million", "in", "2023"]


def test_backend_and_tokenizer_are_validated_and_described(corpus: list[Passage]) -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        BM25Retriever(tokenizer="spacy")
    with pytest.raises(ValueError, match="backend"):
        BM25Retriever(backend="lucene")
    r = BM25Retriever(retriever_name="bm25", tokenizer="word", backend="python")
    r.index(corpus)
    assert r.effective_backend == "python"
    assert r.describe() == {
        "name": "bm25", "type": "bm25", "tokenizer": "word", "lowercase": True, "backend": "python",
        "idf": "log(1 + (N - df + 0.5) / (df + 0.5))", "k1": 1.5, "b": 0.75, "backend_version": None,
    }


def test_python_backend_ranking_does_not_depend_on_rank_bm25(corpus: list[Passage]) -> None:
    pytest.importorskip("rank_bm25")
    python = BM25Retriever(backend="python")
    python.index(corpus)
    reference = _PurePythonBM25([p.text.lower().split() for p in corpus])
    query = "risk supply chain"
    hits = python.search(query, top_k=len(corpus))
    scores = reference.get_scores(query.split())
    assert [h.score for h in hits] == sorted(scores, reverse=True)
    rank = BM25Retriever(backend="rank_bm25")
    assert rank.effective_backend == "rank_bm25" and rank.describe()["backend_version"]
