"""Unit tests for finretriever.eval_set.candidate_generator (synthetic corpus, offline)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from finretriever.eval_set.candidate_generator import (
    CandidateGenerator,
    Scope,
    build_pool,
    fiscal_year_of,
)
from finretriever.eval_set.schemas import CandidateSource, GoldQuestion
from tests.unit.eval_set_fixtures import fixture_passages, passage_ids

IDS = passage_ids()


@pytest.fixture
def generator() -> CandidateGenerator:
    return CandidateGenerator(fixture_passages())


def ids(candidates) -> list[str]:
    return [c.passage_id for c in candidates]


def question(**overrides) -> GoldQuestion:
    values = {
        "query_id": "q_0001",
        "text": "How dependent is Acme on one semiconductor supplier?",
        "topic": "risk_factors",
        "question_type": "semantic",
        "tickers": ["ACME"],
        "authoring": "independent",
        "author": "T1",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    return GoldQuestion(**{**values, **overrides})


def test_bm25_ranks_matches_and_records_provenance(generator: CandidateGenerator) -> None:
    candidates = generator.bm25("Which supplier provides semiconductor components?", depth=3)
    assert set(ids(candidates)[:2]) == {IDS["acme24_supplier"], IDS["acme23_supplier"]}
    assert [c.source.rank for c in candidates] == [1, 2, 3]
    source = candidates[0].source
    assert source.method == "bm25" and source.score and source.score > 0
    assert source.detail == "query='Which supplier provides semiconductor components?'; whole corpus; depth=3"


def test_bm25_matches_words_regardless_of_punctuation(generator: CandidateGenerator) -> None:
    # The passages end "...net sales." and the question "...supplier?".
    assert IDS["acme23_supplier"] in ids(generator.bm25("supplier?", depth=5))
    assert IDS["acme23_table"] in ids(generator.bm25("net sales $952", depth=12))


def test_bm25_drops_passages_without_a_matching_term(generator: CandidateGenerator) -> None:
    assert generator.bm25("zebra", depth=5) == []
    assert ids(generator.bm25("Widget Pro", depth=10)) == [IDS["acme24_sales"]]


def test_scope_restricts_companies_and_years(generator: CandidateGenerator) -> None:
    beta = generator.bm25("credit facility", depth=10, scope=Scope(tickers=frozenset({"BETA"})))
    assert ids(beta)[0] == IDS["beta24_liquidity"]
    assert all(generator.passage(pid).ticker == "BETA" for pid in ids(beta))
    assert "tickers=BETA" in beta[0].source.detail
    assert IDS["acme24_liquidity"] in ids(generator.bm25("credit facility", depth=10))

    fy2023 = Scope(tickers=frozenset({"ACME"}), fiscal_years=frozenset({2023}))
    assert set(ids(generator.bm25("net sales", depth=10, scope=fy2023))) == {
        IDS["acme23_supplier"], IDS["acme23_sales"], IDS["acme23_table"],
    }


def test_fiscal_year_comes_from_the_report_date() -> None:
    passage = fixture_passages()[0]  # FY2024 10-K filed 2025-02-20
    assert fiscal_year_of(passage) == 2024
    assert fiscal_year_of(passage.model_copy(update={"metadata": {}})) == 2025


def test_manual_candidates_validate_ids(generator: CandidateGenerator) -> None:
    candidates = generator.manual([IDS["beta24_credit"], IDS["beta24_credit"]], note="found via search")
    assert ids(candidates) == [IDS["beta24_credit"]]
    assert candidates[0].source == CandidateSource(method="manual", detail="found via search")
    with pytest.raises(ValueError, match="not in the corpus"):
        generator.manual(["passage_missing"])


def test_samples_are_seeded_and_scoped(generator: CandidateGenerator) -> None:
    first = generator.random(4, seed="7:q_0001")
    assert first == generator.random(4, seed="7:q_0001")
    assert len(set(ids(first))) == 4 and first[0].source.method == "random"
    assert first[0].source.detail.startswith("seed=7:q_0001; 4 of 12")
    excluded = set(ids(first))
    rest = generator.random(20, seed=1, exclude=excluded)
    assert len(rest) == 8 and not excluded & set(ids(rest))

    per_filing = generator.stratified(1, seed=3)
    assert len({generator.passage(pid).filing_id for pid in ids(per_filing)}) == 3 == len(per_filing)
    acme = generator.stratified(2, seed=3, scope=Scope(tickers=frozenset({"ACME"})))
    assert len(acme) == 4 and all(generator.passage(pid).ticker == "ACME" for pid in ids(acme))
    assert acme == generator.stratified(2, seed=3, scope=Scope(tickers=frozenset({"ACME"})))


def test_build_pool_orders_sources_and_stays_in_scope(generator: CandidateGenerator) -> None:
    pool = build_pool(
        generator,
        question(),
        bm25_depth=2,
        reformulations=["supplier disruption"],
        manual_ids=[IDS["beta24_business"]],
        stratified_per_filing=1,
        random_count=1,
        seed=9,
    )
    assert [c.source.method for c in pool] == ["bm25"] * 4 + ["stratified"] * 2 + ["random", "manual"]
    assert all(generator.passage(c.passage_id).ticker == "ACME" for c in pool[:-1])
    assert pool[6].passage_id not in ids(pool[:6])  # random never re-proposes
    assert pool == build_pool(
        generator, question(), bm25_depth=2, reformulations=["supplier disruption"],
        manual_ids=[IDS["beta24_business"]], stratified_per_filing=1, random_count=1, seed=9,
    )


def test_years_restrict_the_pool_only_when_asked(generator: CandidateGenerator) -> None:
    q = question(text="What were Acme's net sales?", fiscal_years=[2023])
    assert IDS["acme24_table"] in ids(build_pool(generator, q, bm25_depth=10))
    assert IDS["acme24_table"] not in ids(build_pool(generator, q, bm25_depth=10, use_years=True))


@pytest.mark.parametrize("method", ["dense", "bge", "finretriever"])
def test_model_based_pool_sources_are_rejected(method: str) -> None:
    with pytest.raises(ValidationError):
        CandidateSource(method=method)
