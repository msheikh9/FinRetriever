"""Tests for finretriever.data.schemas and finretriever.utils.io.

These tests are the contract-pinning layer. Anything that quietly changes the
shape of a Passage, the validation of a Qrel, or the determinism of an ID
should fail here before it ever lands in a benchmark run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from finretriever.data.schemas import (
    Filing,
    Passage,
    Qrel,
    Query,
    RunResult,
    SearchHit,
    Section,
    make_filing_id,
    make_passage_id,
    make_query_id,
    make_section_id,
)
from finretriever.utils.io import (
    ensure_parent_dir,
    load_config,
    read_jsonl,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Filing
# ---------------------------------------------------------------------------


def _filing_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        filing_id="filing_aapl_10-k_abc123",
        ticker="AAPL",
        form_type="10-K",
        accession_number="0000320193-23-000106",
        filing_date="2023-11-03",
    )
    base.update(overrides)
    return base


def test_filing_normalizes_ticker_and_form_type() -> None:
    f = Filing(**_filing_kwargs(ticker="aapl", form_type="10-k"))
    assert f.ticker == "AAPL"
    assert f.form_type == "10-K"


def test_filing_rejects_empty_filing_date() -> None:
    with pytest.raises(ValidationError):
        Filing(**_filing_kwargs(filing_date=""))


def test_filing_rejects_whitespace_filing_date() -> None:
    with pytest.raises(ValidationError):
        Filing(**_filing_kwargs(filing_date="   "))


def test_filing_rejects_empty_accession_number() -> None:
    with pytest.raises(ValidationError):
        Filing(**_filing_kwargs(accession_number=""))


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------


def _section_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        section_id="section_x_0001_abc",
        filing_id="filing_x",
        title="Item 1A — Risk Factors",
        text="A material discussion of risk factors.",
    )
    base.update(overrides)
    return base


def test_section_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Section(**_section_kwargs(text=""))


def test_section_rejects_empty_title() -> None:
    with pytest.raises(ValidationError):
        Section(**_section_kwargs(title=""))


def test_section_rejects_negative_char_offsets() -> None:
    with pytest.raises(ValidationError):
        Section(**_section_kwargs(start_char=-1))
    with pytest.raises(ValidationError):
        Section(**_section_kwargs(end_char=-5))


def test_section_rejects_inverted_span() -> None:
    with pytest.raises(ValidationError):
        Section(**_section_kwargs(start_char=200, end_char=100))


def test_section_accepts_equal_endpoints() -> None:
    s = Section(**_section_kwargs(start_char=10, end_char=10))
    assert s.start_char == s.end_char == 10


# ---------------------------------------------------------------------------
# Passage
# ---------------------------------------------------------------------------


def _passage_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        passage_id="passage_x_000001_abc",
        filing_id="filing_x",
        ticker="AAPL",
        form_type="10-K",
        filing_date="2023-11-03",
        text="During fiscal 2023 the company repurchased ...",
    )
    base.update(overrides)
    return base


def test_passage_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Passage(**_passage_kwargs(text=""))


def test_passage_rejects_token_count_zero() -> None:
    with pytest.raises(ValidationError):
        Passage(**_passage_kwargs(token_count=0))


def test_passage_rejects_token_count_negative() -> None:
    with pytest.raises(ValidationError):
        Passage(**_passage_kwargs(token_count=-3))


def test_passage_normalizes_ticker_and_form_type() -> None:
    p = Passage(**_passage_kwargs(ticker="aapl", form_type="10-k"))
    assert p.ticker == "AAPL"
    assert p.form_type == "10-K"


def test_passage_metadata_default_is_per_instance() -> None:
    a = Passage(**_passage_kwargs(passage_id="a"))
    b = Passage(**_passage_kwargs(passage_id="b"))
    a.metadata["item"] = "1A"
    assert b.metadata == {}


# ---------------------------------------------------------------------------
# Query / Qrel / SearchHit / RunResult
# ---------------------------------------------------------------------------


def test_query_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Query(query_id="q1", text="", source="gold")


def test_query_rejects_empty_source() -> None:
    with pytest.raises(ValidationError):
        Query(query_id="q1", text="What were buybacks?", source="")


def test_query_accepts_arbitrary_source_labels() -> None:
    # Sources are intentionally not enumerated — synthetic, gold, manual, etc.
    for src in ("gold", "synthetic", "manual", "generated", "imported"):
        q = Query(query_id="q", text="anything", source=src)
        assert q.source == src


def test_qrel_rejects_negative_relevance() -> None:
    with pytest.raises(ValidationError):
        Qrel(query_id="q1", passage_id="p1", relevance=-1)


def test_qrel_accepts_zero_relevance() -> None:
    assert Qrel(query_id="q1", passage_id="p1", relevance=0).relevance == 0


def test_searchhit_rejects_rank_zero() -> None:
    with pytest.raises(ValidationError):
        SearchHit(query_id="q1", passage_id="p1", rank=0, score=0.9, retriever="bm25")


def test_searchhit_rejects_empty_retriever() -> None:
    with pytest.raises(ValidationError):
        SearchHit(query_id="q1", passage_id="p1", rank=1, score=0.9, retriever="")


def test_runresult_accepts_hits_and_metrics() -> None:
    hits = [
        SearchHit(query_id="q1", passage_id="p1", rank=1, score=0.91, retriever="bm25"),
        SearchHit(query_id="q1", passage_id="p2", rank=2, score=0.74, retriever="bm25"),
    ]
    r = RunResult(
        run_name="bm25_v1",
        metrics={"ndcg@10": 0.512, "mrr@10": 0.61, "recall@100": 0.78},
        hits=hits,
        metadata={"corpus_version": "v1", "top_k": 100},
    )
    assert r.run_name == "bm25_v1"
    assert r.metrics["ndcg@10"] == pytest.approx(0.512)
    assert len(r.hits) == 2
    assert r.hits[0].retriever == "bm25"


def test_runresult_defaults_are_isolated() -> None:
    a = RunResult(run_name="a", metrics={})
    b = RunResult(run_name="b", metrics={})
    a.hits.append(
        SearchHit(query_id="q1", passage_id="p1", rank=1, score=1.0, retriever="bm25")
    )
    assert b.hits == []
    a.metadata["k"] = "v"
    assert b.metadata == {}


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_passage_json_roundtrip() -> None:
    p = Passage(
        **_passage_kwargs(
            metadata={"item": "1A", "fy": 2023, "audited": True, "note": None},
            token_count=312,
        )
    )
    raw = p.model_dump_json()
    back = Passage.model_validate_json(raw)
    assert back == p


def test_runresult_json_roundtrip_preserves_hits() -> None:
    r = RunResult(
        run_name="finretriever_v1",
        metrics={"ndcg@10": 0.63},
        hits=[
            SearchHit(query_id="q1", passage_id="p1", rank=1, score=0.9, retriever="dense"),
        ],
    )
    back = RunResult.model_validate_json(r.model_dump_json())
    assert back == r


# ---------------------------------------------------------------------------
# Deterministic ID factories
# ---------------------------------------------------------------------------


def test_make_filing_id_is_deterministic() -> None:
    a = make_filing_id("AAPL", "10-K", "0000320193-23-000106")
    b = make_filing_id("AAPL", "10-K", "0000320193-23-000106")
    assert a == b
    assert a.startswith("filing_aapl_10-k_")


def test_make_filing_id_normalizes_case() -> None:
    upper = make_filing_id("AAPL", "10-K", "0000320193-23-000106")
    lower = make_filing_id("aapl", "10-k", "0000320193-23-000106")
    assert upper == lower


def test_make_filing_id_differs_for_different_inputs() -> None:
    ids = {
        make_filing_id("AAPL", "10-K", "0000320193-23-000106"),
        make_filing_id("MSFT", "10-K", "0000320193-23-000106"),
        make_filing_id("AAPL", "10-Q", "0000320193-23-000106"),
        make_filing_id("AAPL", "10-K", "0000999999-99-999999"),
    }
    assert len(ids) == 4


def test_make_section_id_is_deterministic_and_indexed() -> None:
    s1 = make_section_id("filing_x", "Item 1A", 3)
    s2 = make_section_id("filing_x", "Item 1A", 3)
    s3 = make_section_id("filing_x", "Item 1A", 4)
    assert s1 == s2
    assert s1 != s3
    assert s1.startswith("section_filing_x_0003_")


def test_make_passage_id_is_deterministic_and_uses_chunk_index() -> None:
    p1 = make_passage_id("filing_x", "section_y", 12)
    p2 = make_passage_id("filing_x", "section_y", 12)
    p3 = make_passage_id("filing_x", "section_y", 13)
    p4 = make_passage_id("filing_x", None, 12)
    assert p1 == p2
    assert p1 != p3
    assert p1 != p4
    assert "_000012_" in p1


def test_make_query_id_is_deterministic_and_includes_source() -> None:
    q1 = make_query_id("What were stock buybacks in 2023?", "gold")
    q2 = make_query_id("What were stock buybacks in 2023?", "gold")
    q3 = make_query_id("What were stock buybacks in 2023?", "synthetic")
    q4 = make_query_id("What was capex?", "gold")
    assert q1 == q2
    assert q1 != q3
    assert q1 != q4
    assert q1.startswith("query_gold_")


# ---------------------------------------------------------------------------
# IO: JSONL + YAML
# ---------------------------------------------------------------------------


def test_write_then_read_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "passages.jsonl"
    rows: list = [
        Passage(**_passage_kwargs(passage_id="p1", text="alpha")),
        {
            "passage_id": "p2",
            "filing_id": "filing_x",
            "ticker": "AAPL",
            "form_type": "10-K",
            "filing_date": "2023-11-03",
            "text": "beta",
            "metadata": {},
        },
    ]
    write_jsonl(path, rows)
    assert path.exists()
    back = read_jsonl(path)
    assert len(back) == 2
    assert back[0]["passage_id"] == "p1"
    assert back[1]["text"] == "beta"


def test_read_jsonl_validates_into_pydantic_models(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [
            Query(query_id="q1", text="What were buybacks?", source="gold"),
            Query(query_id="q2", text="Capex in 2023?", source="gold"),
        ],
    )
    queries = read_jsonl(path, model=Query)
    assert len(queries) == 2
    assert all(isinstance(q, Query) for q in queries)
    assert queries[0].text == "What were buybacks?"


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "with_blanks.jsonl"
    path.write_text(
        '{"a": 1}\n'
        "\n"
        "   \n"
        '{"a": 2}\n'
        "\t\n",
        encoding="utf-8",
    )
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_raises_with_path_and_line_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"a": 1}\n{not valid json\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        read_jsonl(path)
    msg = str(exc_info.value)
    assert str(path) in msg
    assert ":2" in msg


def test_read_jsonl_raises_with_path_and_line_on_validation_failure(tmp_path: Path) -> None:
    path = tmp_path / "bad_validation.jsonl"
    # Missing required `text` and `source` for Query.
    path.write_text('{"query_id": "q1"}\n', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        read_jsonl(path, model=Query)
    msg = str(exc_info.value)
    assert str(path) in msg
    assert ":1" in msg
    assert "Query" in msg


def test_write_jsonl_is_atomic_by_default(tmp_path: Path) -> None:
    path = tmp_path / "atomic.jsonl"
    # Pre-existing file should be replaced wholesale by the next successful write.
    path.write_text('{"old": true}\n', encoding="utf-8")
    write_jsonl(path, [{"new": True}])
    contents = path.read_text(encoding="utf-8").strip().splitlines()
    assert contents == ['{"new":true}'] or contents == ['{"new": true}']
    # And no stray temp files should be left behind on success.
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_load_config_validates_into_pydantic_model(tmp_path: Path) -> None:
    class _TinyConfig(BaseModel):
        name: str
        batch_size: int
        learning_rate: float

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "name: bge_test\nbatch_size: 32\nlearning_rate: 2.0e-5\n",
        encoding="utf-8",
    )
    cfg = load_config(path, _TinyConfig)
    assert isinstance(cfg, _TinyConfig)
    assert cfg.batch_size == 32
    assert cfg.learning_rate == pytest.approx(2e-5)


def test_load_config_without_schema_returns_dict(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("a: 1\nb: two\n", encoding="utf-8")
    assert load_config(path) == {"a": 1, "b": "two"}


def test_ensure_parent_dir_creates_missing_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "out.jsonl"
    returned = ensure_parent_dir(target)
    assert returned == target
    assert target.parent.is_dir()


# ---------------------------------------------------------------------------
# Schema field-drift guards
# ---------------------------------------------------------------------------
#
# These tests fail loudly if anyone adds, removes or renames a field on the
# core retrieval contracts. A drift here would invalidate every committed
# JSONL artefact in data/eval/ and results/, so it is worth catching in CI.


def test_passage_field_names_do_not_drift() -> None:
    assert set(Passage.model_fields.keys()) == {
        "passage_id",
        "filing_id",
        "section_id",
        "ticker",
        "form_type",
        "filing_date",
        "section_title",
        "text",
        "token_count",
        "metadata",
    }


def test_searchhit_field_names_do_not_drift() -> None:
    assert set(SearchHit.model_fields.keys()) == {
        "query_id",
        "passage_id",
        "rank",
        "score",
        "retriever",
        "metadata",
    }
