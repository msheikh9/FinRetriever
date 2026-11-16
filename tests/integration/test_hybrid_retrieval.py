"""End to end: query understanding, scoping, BM25 + dense fusion and MaxP windows on a small corpus.

Uses the real BM25 retriever, the real dense retriever with a fake (bag-of-words) encoder, and the real parser.
"""

import numpy as np
import pytest

from finretriever.data.schemas import Passage
from finretriever.retrieval.biencoder import BiEncoderRetriever
from finretriever.retrieval.bm25 import BM25Retriever
from finretriever.retrieval.fusion import FusionConfig, reciprocal_rank_fusion
from finretriever.retrieval.hybrid import CachedScorer, HybridConfig, HybridRetriever, MaxPScorer
from finretriever.retrieval.metadata_scope import ScopePolicy
from finretriever.retrieval.query_understanding import QueryParser
from tests.unit.dense_fixtures import REVISION, FakeEncoder

BGE = "BAAI/bge-base-en-v1.5"
COMPANIES = {"AAPL": "Apple Inc.", "WMT": "Walmart Inc."}


def corpus() -> list[Passage]:
    out = []
    for ticker in ("AAPL", "WMT"):
        for year in (2020, 2021, 2022, 2023):
            texts = [f"long-term debt totaled {year * 10} million at year end",
                     f"risk factors supply chain manufacturing {ticker.lower()} {year}",
                     f"net sales and revenue by segment for {year}"]
            for i, text in enumerate(texts):
                out.append(Passage(passage_id=f"{ticker}-{year}-{i}", filing_id=f"{ticker}-{year}", ticker=ticker,
                                   form_type="10-K", filing_date=f"{year + 1}-02-01", section_title="Item 8",
                                   text=text, metadata={"company_name": COMPANIES[ticker],
                                                        "report_date": f"{year}-12-31"}))
    return out


@pytest.fixture(scope="module")
def setup():
    passages = corpus()
    bm25 = BM25Retriever(tokenizer="word", backend="python")
    bm25.index(passages)
    dense = BiEncoderRetriever(BGE, model_revision=REVISION, encoder=FakeEncoder(dim=32))
    dense.index(passages)
    scorers = {"bm25": CachedScorer(bm25), "dense": dense}
    return passages, scorers, QueryParser.from_companies(COMPANIES)


def build(setup, name, components, fusion=None, policy=None, **kw):
    passages, scorers, parser = setup
    config = HybridConfig(name, tuple(components), fusion or FusionConfig(), policy or ScopePolicy(), **kw)
    return HybridRetriever(config, scorers, passages, parser)


QUESTION = "How much long-term debt did Apple report in fiscal 2021?"


def test_single_component_without_scope_reproduces_the_dense_ranking(setup):
    retriever = build(setup, "dense", ["dense"])
    hits, trace = retriever.search(QUESTION, top_k=5)
    direct = setup[1]["dense"].search(QUESTION, top_k=5)
    assert [h.passage_id for h in hits] == [h.passage_id for h in direct]
    assert trace["scope"]["tickers"] == ["AAPL"] and trace["plan"]["eligible"] == 24


def test_company_and_year_filters_keep_the_comparative_window(setup):
    retriever = build(setup, "scoped", ["dense"], policy=ScopePolicy("filter", "filter", min_eligible=1))
    hits, trace = retriever.search(QUESTION, top_k=12)
    assert {h.metadata["ticker"] for h in hits} == {"AAPL"}
    assert {h.metadata["fiscal_year"] for h in hits} == {2021, 2022, 2023}  # FY2021 and two later filings
    assert trace["plan"]["year_filtered"] and len(hits) == 9


def test_rrf_hybrid_contains_both_components_top_results(setup):
    retriever = build(setup, "rrf", ["bm25", "dense"], fusion=FusionConfig("rrf", k=60))
    hits, _ = retriever.search(QUESTION, top_k=24)
    ids = [h.passage_id for h in hits]
    assert setup[1]["bm25"].scores(QUESTION).shape == (24,)
    assert len(ids) == len(set(ids)) == 24
    # the same fusion computed independently from the two components' own rankings
    position = {p.passage_id: i for i, p in enumerate(setup[0])}
    bm25_rank = [position[h.passage_id] for h in setup[1]["bm25"].scorer.search(QUESTION, top_k=24)]
    dense_rank = [position[h.passage_id] for h in setup[1]["dense"].search(QUESTION, top_k=24)]
    expected = reciprocal_rank_fusion({"bm25": bm25_rank, "dense": dense_rank}, k=60)
    assert ids == [setup[0][p].passage_id for p, _ in expected]
    assert [h.score for h in hits] == [s for _, s in expected]


def test_year_boost_moves_named_year_passages_up_without_removing_others(setup):
    plain = build(setup, "rrf", ["bm25", "dense"], fusion=FusionConfig("rrf"),
                  policy=ScopePolicy("filter"))
    boosted = build(setup, "rrf+boost", ["bm25", "dense"], fusion=FusionConfig("rrf"),
                    policy=ScopePolicy("filter", "boost"))
    before, _ = plain.search(QUESTION, top_k=12)
    after, _ = boosted.search(QUESTION, top_k=12)
    assert {h.passage_id for h in before} == {h.passage_id for h in after}  # nothing removed
    years_after = [h.metadata["fiscal_year"] for h in after]
    assert sum(y == 2021 for y in years_after[:3]) >= sum(h.metadata["fiscal_year"] == 2021 for h in before[:3])


def test_cells_cover_every_company_of_a_cross_company_question(setup):
    retriever = build(setup, "cells", ["dense"], policy=ScopePolicy("filter", "filter", cells="interleave",
                                                                    min_eligible=1))
    hits, trace = retriever.search("Compare Apple and Walmart long-term debt in fiscal 2022", top_k=4)
    assert {h.metadata["ticker"] for h in hits[:2]} == {"AAPL", "WMT"}
    assert [c["label"] for c in trace["plan"]["cells"]] == ["AAPL/2022", "WMT/2022"]


def test_exclusions_and_oracle_scope(setup):
    retriever = build(setup, "dense", ["dense"])
    top = retriever.search(QUESTION, top_k=1)[0][0].passage_id
    hits, _ = retriever.search(QUESTION, top_k=24, exclude={top})
    assert top not in {h.passage_id for h in hits}
    oracle = build(setup, "oracle", ["dense"], policy=ScopePolicy("filter", source="oracle"))
    with pytest.raises(ValueError, match="annotated scope"):
        oracle.search(QUESTION)
    scope = setup[2].parse("Walmart")
    hits, _ = oracle.search(QUESTION, top_k=5, oracle_scope=scope)
    assert {h.metadata["ticker"] for h in hits} == {"WMT"}


def test_maxp_scores_a_passage_by_its_best_window(setup):
    passages, scorers, _ = setup

    class Fixed:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def scores(self, query):
            return self.values

    head = Fixed(np.zeros(len(passages)))
    windows = Fixed([0.9, 0.2, 0.5])
    maxp = MaxPScorer(head, windows, parents=[3, 3, 7])
    out = maxp.scores("q")
    assert out[3] == 0.9 and out[7] == 0.5 and out.sum() == pytest.approx(1.4)
    assert head.values.sum() == 0  # the head scores are not modified in place


def test_searches_are_deterministic(setup):
    retriever = build(setup, "all", ["bm25", "dense"], fusion=FusionConfig("rrf"),
                      policy=ScopePolicy("filter", "boost", cells="interleave"))
    first = retriever.search(QUESTION, top_k=10)
    second = retriever.search(QUESTION, top_k=10)
    assert first == second


def test_explain_reports_each_components_own_rank_and_the_year_tier(setup):
    retriever = build(setup, "e5", ["bm25", "dense"], fusion=FusionConfig("rrf", k=60),
                      policy=ScopePolicy("filter", "boost"))
    hits, trace = retriever.search(QUESTION, top_k=12)
    why = retriever.explain(QUESTION, [h.passage_id for h in hits])
    passages, scorers, _ = setup
    position = {p.passage_id: i for i, p in enumerate(passages)}
    apple = sorted(i for i, p in enumerate(passages) if p.ticker == "AAPL")
    for name in ("bm25", "dense"):  # each component's own order over the in-scope (Apple) passages
        scores = np.asarray(scorers[name].scores(QUESTION))
        order = sorted(apple, key=lambda i: (-scores[i], i))
        for h in hits:
            assert why[h.passage_id]["component_ranks"][name] == order.index(position[h.passage_id]) + 1
    for h in hits:
        info = why[h.passage_id]
        assert info["eligible_passages"] == 12 and info["company_in_scope"] is True
        year = h.metadata["fiscal_year"]
        assert info["fiscal_year"] == year and info["year_in_scope"] == (year == 2021)
        assert info["year_tier"] == {2021: "requested year", 2022: "later filing", 2023: "later filing"}.get(
            year, "other year")
    assert hits[0].metadata["fiscal_year"] == 2021  # the boost puts the requested year first
    assert trace == retriever.search(QUESTION, top_k=12)[1]  # explain changes nothing


def test_explain_without_a_named_year_or_company_reports_no_scope(setup):
    retriever = build(setup, "plain", ["dense"])
    hits, _ = retriever.search("supply chain risk factors", top_k=3)
    why = retriever.explain("supply chain risk factors", [h.passage_id for h in hits])
    assert all(w["company_in_scope"] is None and w["year_in_scope"] is None and w["year_tier"] is None
               for w in why.values())
