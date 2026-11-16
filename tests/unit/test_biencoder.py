"""Tests for the dense bi-encoder retriever, with fake encoders (no model is downloaded or loaded)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from finretriever.data.schemas import Passage
from finretriever.retrieval import biencoder
from finretriever.retrieval.base import BaseRetriever
from finretriever.retrieval.biencoder import (
    MODEL_FORMATS,
    BiEncoderRetriever,
    DenseModelError,
    SentenceTransformerEncoder,
    load_sentence_transformer,
    truncation_stats,
)
from finretriever.retrieval.bm25 import BM25Retriever
from finretriever.retrieval.embedding_cache import EmbeddingCacheError
from tests.unit.dense_fixtures import CORPUS_SHA256, REVISION, FakeEncoder, TableEncoder
from tests.unit.eval_set_fixtures import fixture_passages

BGE = "BAAI/bge-base-en-v1.5"
E5 = "intfloat/e5-base-v2"
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def dense(model: str = BGE, encoder: FakeEncoder | None = None, **kwargs: Any) -> BiEncoderRetriever:
    return BiEncoderRetriever(model, model_revision=REVISION, encoder=encoder or FakeEncoder(), **kwargs)


def passage(passage_id: str, ticker: str, text: str) -> Passage:
    return Passage(
        passage_id=passage_id, filing_id=f"filing_{ticker.lower()}", ticker=ticker, form_type="10-K",
        filing_date="2025-02-20", section_title="Risk Factors", text=text,
    )


# --- interface and formats ---------------------------------------------------------------


def test_follows_the_retriever_interface_like_bm25() -> None:
    passages = fixture_passages()
    retriever = dense(retriever_name="bge_zero_shot")
    bm25 = BM25Retriever(tokenizer="word", backend="python")
    assert isinstance(retriever, BaseRetriever) and isinstance(bm25, BaseRetriever)
    assert not retriever.is_indexed
    retriever.index(passages)
    bm25.index(passages)
    assert retriever.is_indexed and len(retriever) == len(passages)

    hits = retriever.search("single supplier for semiconductor components", top_k=3, query_id="q1")
    assert [h.rank for h in hits] == [1, 2, 3]
    assert {h.query_id for h in hits} == {"q1"} and {h.retriever for h in hits} == {"bge_zero_shot"}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    lexical = bm25.search("single supplier for semiconductor components", top_k=3)
    assert [set(h.metadata) for h in hits] == [set(h.metadata) for h in lexical]  # same hit metadata

    batch = retriever.batch_search({"a": "credit facility", "b": "net sales growth"}, top_k=2)
    assert [(h.query_id, h.rank) for h in batch] == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]


@pytest.mark.parametrize(
    ("model", "query_prefix", "passage_prefix"),
    [(BGE, BGE_INSTRUCTION, ""), (E5, "query: ", "passage: ")],
)
def test_each_model_gets_its_own_input_format(model: str, query_prefix: str, passage_prefix: str) -> None:
    passages = fixture_passages()
    encoder = FakeEncoder()
    retriever = dense(model, encoder)
    retriever.index(passages)
    retriever.search("Which supplier does Acme depend on?")
    assert encoder.calls[0] == [passage_prefix + p.text for p in passages]
    assert encoder.calls[-1] == [query_prefix + "Which supplier does Acme depend on?"]
    assert (retriever.format.query_prefix, retriever.format.passage_prefix) == (query_prefix, passage_prefix)
    described = retriever.describe()
    assert (described["query_prefix"], described["passage_prefix"]) == (query_prefix, passage_prefix)
    assert described["format_source"] == MODEL_FORMATS[model].source


def test_one_models_formatting_rules_are_never_used_for_another() -> None:
    with pytest.raises(ValueError, match="must not be used for another"):
        dense(BGE, query_prefix="query: ")
    with pytest.raises(ValueError, match="must not be used for another"):
        dense(BGE, passage_prefix="passage: ")
    with pytest.raises(ValueError, match="must not be used for another"):
        dense(E5, query_prefix=BGE_INSTRUCTION)
    with pytest.raises(ValueError, match="must not be used for another"):
        dense(E5, passage_prefix="")
    with pytest.raises(ValueError, match="no built-in input format"):
        dense("someone/other-model")
    dense(E5, query_prefix="query: ", passage_prefix="passage: ")  # restating the right format is fine
    other = dense("someone/other-model", query_prefix="", passage_prefix="")
    assert other.format.source == "configured explicitly"


def test_constructor_validates_revision_device_batch_size_and_cache() -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        BiEncoderRetriever(BGE, model_revision="main", encoder=FakeEncoder())
    with pytest.raises(ValueError, match="device"):
        dense(device="gpu")
    with pytest.raises(ValueError, match="batch_size"):
        dense(batch_size=0)
    with pytest.raises(ValueError, match="max_seq_length"):
        dense(max_seq_length=-1)
    with pytest.raises(ValueError, match="corpus SHA-256"):
        dense(cache_dir="cache")
    for device in ("auto", "cpu", "mps", "cuda", "cuda:1"):
        dense(device=device)


# --- embeddings and scores ------------------------------------------------------------------


def test_embeddings_are_unit_length_and_scores_are_cosine_similarities() -> None:
    passages = fixture_passages()
    encoder = FakeEncoder()
    retriever = dense(encoder=encoder)
    retriever.index(passages)
    assert retriever.embeddings is not None and retriever.embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(retriever.embeddings, axis=1), 1.0, atol=1e-6)
    query = "credit losses and the allowance"
    hits = retriever.search(query, top_k=len(passages))
    q = encoder.vector(BGE_INSTRUCTION + query).astype(np.float64)
    by_id = {p.passage_id: p for p in passages}
    for hit in hits:
        v = encoder.vector(by_id[hit.passage_id].text).astype(np.float64)
        assert hit.score == pytest.approx(float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q))), abs=1e-6)
    described = retriever.describe()
    assert described["embedding_dim"] == 16 and described["normalization"].startswith("L2")
    assert described["embeddings_sha256"] and described["model"] == encoder.describe()


def test_zero_vectors_wrong_shapes_and_dimensions_are_rejected() -> None:
    passages = fixture_passages()

    class Zero(FakeEncoder):
        def vector(self, text: str) -> np.ndarray:
            return np.zeros(self.embedding_dim, dtype=np.float32)

    class Short(FakeEncoder):
        def encode(self, texts: Any, batch_size: int) -> np.ndarray:
            return super().encode(texts, batch_size)[:-1]

    with pytest.raises(DenseModelError, match="zero or non-finite"):
        dense(encoder=Zero()).index(passages)
    with pytest.raises(DenseModelError, match="shape"):
        dense(encoder=Short()).index(passages)
    with pytest.raises(DenseModelError, match="expects 768"):
        dense(embedding_dim=768).index(passages)


def test_passage_ids_and_queries_are_validated() -> None:
    passages = fixture_passages()
    with pytest.raises(ValueError, match="Duplicate passage_id"):
        dense().index([*passages, passages[0]])
    with pytest.raises(ValueError, match="empty"):
        dense().index([])
    retriever = dense()
    with pytest.raises(RuntimeError, match="before index"):
        retriever.search("anything")
    retriever.index(passages)
    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("  ")
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("anything", top_k=0)


def test_ties_are_broken_by_corpus_position_and_rankings_repeat() -> None:
    same = "Identical boilerplate about forward-looking statements."
    passages = [passage(f"p{i}", "ACME" if i % 2 else "BETA", same) for i in range(6)]
    first = dense()
    first.index(passages)
    ranked = first.search("forward-looking statements", top_k=6)
    assert [h.passage_id for h in ranked] == [f"p{i}" for i in range(6)]
    assert len({h.score for h in ranked}) == 1
    again = dense()
    again.index(passages)
    rerun = again.search("forward-looking statements", top_k=6)
    assert [(h.passage_id, h.rank, h.score) for h in rerun] == [(h.passage_id, h.rank, h.score) for h in ranked]
    assert first.search("forward-looking statements", top_k=6) == ranked  # cached query vector, same hits


# --- company-constrained and global retrieval -------------------------------------------


def company_corpus() -> tuple[list[Passage], TableEncoder]:
    """Four ACME passages close to the query and three BETA passages far from it, each in order."""
    passages = [passage(f"acme{i}", "ACME", f"acme text {i}") for i in range(4)]
    passages += [passage(f"beta{i}", "BETA", f"beta text {i}") for i in range(3)]
    vectors = {f"acme text {i}": [1.0, 0.1 * (i + 1), 0.0] for i in range(4)}
    vectors |= {f"beta text {i}": [0.1 * (3 - i), 0.0, 1.0] for i in range(3)}
    vectors[BGE_INSTRUCTION + "question"] = [1.0, 0.0, 0.0]
    return passages, TableEncoder(vectors, default=[0.0, 1.0, 0.0])


def test_company_mode_ranks_only_eligible_passages_instead_of_filtering_a_global_list() -> None:
    passages, encoder = company_corpus()
    retriever = dense(encoder=encoder)
    retriever.index(passages)
    global_top3 = [h.passage_id for h in retriever.search("question", top_k=3)]
    assert global_top3 == ["acme0", "acme1", "acme2"]
    beta = retriever.search("question", top_k=3, tickers={"BETA"})
    # None of BETA's passages is in the global top 3, yet the constrained run returns three of them.
    assert [h.passage_id for h in beta] == ["beta0", "beta1", "beta2"]
    assert [h.rank for h in beta] == [1, 2, 3] and {h.metadata["ticker"] for h in beta} == {"BETA"}
    both = retriever.search("question", top_k=7, tickers={"ACME", "BETA"})
    assert both == retriever.search("question", top_k=7)  # every company allowed = global ranking
    with pytest.raises(ValueError, match="no passages for ticker"):
        retriever.search("question", tickers={"GAMMA"})
    batch = retriever.batch_search({"q1": "question", "q2": "question"}, top_k=2, tickers={"q2": ["BETA"]})
    assert [h.passage_id for h in batch] == ["acme0", "acme1", "beta0", "beta1"]


def test_bm25_applies_the_same_company_restriction() -> None:
    passages = [passage(f"acme{i}", "ACME", "supplier risk " * (4 - i)) for i in range(3)]
    passages += [passage(f"beta{i}", "BETA", f"supplier note {i} " + "filler " * i) for i in range(3)]
    bm25 = BM25Retriever(tokenizer="word", backend="python")
    bm25.index(passages)
    assert [h.passage_id for h in bm25.search("supplier risk", top_k=3)] == ["acme0", "acme1", "acme2"]
    beta = bm25.search("supplier risk", top_k=3, tickers={"BETA"})
    assert len(beta) == 3 and {h.metadata["ticker"] for h in beta} == {"BETA"}
    with pytest.raises(ValueError, match="no passages for ticker"):
        bm25.search("supplier risk", tickers=["GAMMA"])


def test_global_mode_ranks_every_company() -> None:
    passages, encoder = company_corpus()
    retriever = dense(encoder=encoder)
    retriever.index(passages)
    hits = retriever.search("question", top_k=7)
    assert [h.passage_id for h in hits][:4] == ["acme0", "acme1", "acme2", "acme3"]
    assert {h.metadata["ticker"] for h in hits} == {"ACME", "BETA"}


# --- cache ------------------------------------------------------------------------------------


def cached(tmp_path: Path, encoder: FakeEncoder, **kwargs: Any) -> BiEncoderRetriever:
    return dense(encoder=encoder, cache_dir=tmp_path / "cache", corpus_sha256=CORPUS_SHA256, **kwargs)


def test_cache_is_written_once_and_reused_without_encoding_passages(tmp_path: Path) -> None:
    passages = fixture_passages()
    first_encoder = FakeEncoder()
    first = cached(tmp_path, first_encoder)
    first.index(passages)
    assert first_encoder.encoded == [p.text for p in passages]
    cache_info = first.describe()["cache"]
    entry = Path(cache_info["dir"])
    assert sorted(p.name for p in entry.iterdir()) == ["embeddings.npy", "meta.json", "passage_ids.json"]
    meta = json.loads((entry / "meta.json").read_text())
    assert meta["key"]["corpus_sha256"] == CORPUS_SHA256 and meta["key"]["model_revision"] == REVISION
    assert meta["key"]["encoding"]["passage_prefix"] == "" and meta["key"]["encoding"]["batch_size"] == 32

    second_encoder = FakeEncoder()
    second = cached(tmp_path, second_encoder)
    second.index(passages)
    assert second_encoder.calls == []  # embeddings came from the cache
    assert np.array_equal(first.embeddings, second.embeddings)
    assert second.describe()["cache"] == cache_info
    assert second.describe()["embeddings_sha256"] == first.describe()["embeddings_sha256"]
    assert second.search("supplier risk", top_k=5) == first.search("supplier risk", top_k=5)


def test_other_encoding_settings_get_their_own_cache_entry(tmp_path: Path) -> None:
    passages = fixture_passages()
    base = cached(tmp_path, FakeEncoder())
    base.index(passages)
    encoder = FakeEncoder()
    other_batch = cached(tmp_path, encoder, batch_size=4)
    other_batch.index(passages)
    assert encoder.encoded == [p.text for p in passages]  # encoded again, in batches of 4
    assert [len(call) for call in encoder.calls] == [4, 4, 4]
    assert other_batch.describe()["cache"]["dir"] != base.describe()["cache"]["dir"]
    e5 = dense(E5, encoder=FakeEncoder(), cache_dir=tmp_path / "cache", corpus_sha256=CORPUS_SHA256)
    e5.index(passages)
    assert e5.describe()["cache"]["key"]["encoding"]["passage_prefix"] == "passage: "
    assert len(list((tmp_path / "cache").rglob("meta.json"))) == 3


def test_cache_for_a_different_passage_list_under_the_same_corpus_hash_is_rejected(tmp_path: Path) -> None:
    passages = fixture_passages()
    cached(tmp_path, FakeEncoder()).index(passages)
    with pytest.raises(EmbeddingCacheError, match="passage order differs"):
        cached(tmp_path, FakeEncoder()).index(list(reversed(passages)))


# --- truncation and model loading -----------------------------------------------------------


def test_truncation_is_measured_for_passages_and_queries() -> None:
    passages = fixture_passages()
    retriever = dense(encoder=FakeEncoder(max_seq_length=20))
    retriever.index(passages)
    retriever.search("short question")
    counts = [len(p.text.split()) + 2 for p in passages]
    assert retriever.describe()["passage_truncation"] == truncation_stats(counts, 20)
    assert retriever.describe()["passage_truncation"]["truncated"] == sum(c > 20 for c in counts)
    assert retriever.describe()["query_truncation"]["truncated"] == 0
    assert truncation_stats([10, 30, 50], 20) == {
        "max_seq_length": 20, "texts": 3, "truncated": 2, "truncated_share": 0.6667, "tokens_median": 30,
        "tokens_mean": 30.0, "tokens_max": 50, "tokens_beyond_limit_share": 0.4444,
    }


def test_load_errors_are_dense_model_errors_with_the_reason() -> None:
    # tests/conftest.py makes importing the model backend fail, as on a machine without it.
    with pytest.raises(DenseModelError, match="importing torch / sentence-transformers failed"):
        load_sentence_transformer(BGE, revision=REVISION)
    retriever = BiEncoderRetriever(BGE, model_revision=REVISION)  # no injected encoder: loads lazily
    with pytest.raises(DenseModelError, match="Dense retrieval needs the full requirements"):
        retriever.index(fixture_passages())


class _Cuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _FakeTorch:
    def __init__(self, cuda: bool = False, mps: bool = False) -> None:
        self.cuda = _Cuda(cuda)
        self.backends = type("Backends", (), {"mps": _Cuda(mps)})()

    def get_num_threads(self) -> int:
        return 3


def _module(name: str) -> Any:
    return type(name, (), {"get_pooling_mode_str": lambda self: "cls"})()


class _StubModel:
    """Just enough of a SentenceTransformer for the wrapper."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.max_seq_length = 512
        self.prompts = {"query": "do not use me: "}
        self.similarity_fn_name = "cosine"
        self.tokenizer = _StubTokenizer()
        self.modules = [_module("Transformer"), _module("Pooling"), _module("Normalize")]
        self.encode_calls: list[dict[str, Any]] = []

    def __iter__(self) -> Any:
        return iter(self.modules)

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append({"texts": texts, **kwargs})
        return np.ones((len(texts), 4), dtype=np.float32)


class _StubTokenizer:
    model_max_length = 512
    vocab_size = 100
    do_lower_case = True
    truncation_side = "right"

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        assert kwargs == {"add_special_tokens": True, "truncation": False, "verbose": False}
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


def test_loader_pins_the_revision_and_wraps_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[_StubModel] = []

    def construct(name: str, **kwargs: Any) -> _StubModel:
        built.append(_StubModel(name, **kwargs))
        return built[-1]

    monkeypatch.setattr(biencoder, "_import_backend", lambda: (_FakeTorch(), construct))
    encoder = load_sentence_transformer(BGE, revision=REVISION, local_files_only=True, max_seq_length=256)
    assert isinstance(encoder, SentenceTransformerEncoder)
    assert built[0].kwargs == {"revision": REVISION, "device": "cpu", "local_files_only": True, "trust_remote_code": False}
    assert (encoder.device, encoder.embedding_dim, encoder.max_seq_length) == ("cpu", 4, 256)
    out = encoder.encode(["passage one", "passage two"], 8)
    assert out.dtype == np.float32 and out.shape == (2, 4)
    call = built[0].encode_calls[0]
    assert call["prompt"] == "" and call["normalize_embeddings"] is False and call["batch_size"] == 8
    assert encoder.token_counts(["a b c", "d"]) == [5, 3]
    described = encoder.describe()
    assert described["pooling"] == "cls" and described["model_normalize_module"] is True
    assert described["modules"] == ["Transformer", "Pooling", "Normalize"] and described["torch_threads"] == 3
    assert described["model_prompts_not_applied"] == {"query": "do not use me: "}


def test_loader_reports_unavailable_devices_and_failed_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str, **kwargs: Any) -> Any:
        raise OSError("not found in the local cache")

    monkeypatch.setattr(biencoder, "_import_backend", lambda: (_FakeTorch(), fail))
    with pytest.raises(DenseModelError, match="CUDA is not available"):
        load_sentence_transformer(BGE, revision=REVISION, device="cuda")
    with pytest.raises(DenseModelError, match="MPS is not available"):
        load_sentence_transformer(BGE, revision=REVISION, device="mps")
    with pytest.raises(DenseModelError, match=r"could not load .* local_files_only is set"):
        load_sentence_transformer(BGE, revision=REVISION, local_files_only=True)
    monkeypatch.setattr(biencoder, "_import_backend", lambda: (_FakeTorch(cuda=True), fail))
    with pytest.raises(DenseModelError, match="access to huggingface.co"):
        load_sentence_transformer(BGE, revision=REVISION, device="auto")
