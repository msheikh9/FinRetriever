"""Bi-encoder dense retriever.

Wraps a ``sentence-transformers`` model so one class powers every dense
configuration: the zero-shot baselines ``BAAI/bge-base-en-v1.5`` and
``intfloat/e5-base-v2``, and local fine-tuned checkpoints through the
subclass in :mod:`finretriever.retrieval.checkpoint`.

Input formats are per model and never shared (:data:`MODEL_FORMATS`):

* BGE prepends a retrieval instruction to queries only. Its model card
  recommends the instruction for short-query-to-passage retrieval and says
  v1.5 also works without it, slightly worse; the evaluation questions are
  short queries over long passages, so the instruction is used.
* E5 needs ``"query: "`` before every query and ``"passage: "`` before every
  passage; it degrades without them.

A configured prefix that differs from the model's format is an error, and a
model without a built-in format must state both prefixes explicitly. The
model's own configured prompts are never applied on top.

Embeddings are L2-normalised float32 vectors, so the dot product used for
ranking (computed in float64) is the cosine similarity. Passages are encoded
in batches of ``batch_size`` and each batch is moved to CPU memory before the
next one is encoded, so the accelerator only ever holds one batch. Queries are
encoded one per forward pass, so a query's vector never depends on which other
queries are in the run. Ties are broken by corpus position (see
:mod:`finretriever.retrieval.base`).

Passage embeddings can be cached on disk (:mod:`.embedding_cache`) under a key
of corpus SHA-256, model, revision and encoding configuration.

``torch`` and ``sentence_transformers`` are imported only when a model is
loaded, so this module (and the benchmark harness) imports without them.
"""

from __future__ import annotations

import hashlib
import logging
import re
import statistics
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from finretriever.data.schemas import Passage, SearchHit
from finretriever.retrieval.base import eligible_indices, hit_metadata, rank_top_k
from finretriever.retrieval.embedding_cache import (
    EmbeddingCache,
    EmbeddingCacheKey,
    check_embeddings,
)
from finretriever.retrieval.enrichment import PASSAGE_VIEWS, encoded_field, render_passage

logger = logging.getLogger(__name__)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")
# Library versions that change the numbers a model produces; part of the cache key.
_KEY_LIBRARIES = ("sentence-transformers", "transformers", "torch")
_RECORDED_LIBRARIES = (*_KEY_LIBRARIES, "tokenizers", "huggingface-hub", "numpy")


class DenseModelError(RuntimeError):
    """A dense model could not be loaded, or it does not match its configuration."""


@dataclass(frozen=True)
class ModelFormat:
    """The text a model expects before every query and every passage, and where the rule comes from."""

    query_prefix: str
    passage_prefix: str
    source: str


MODEL_FORMATS: dict[str, ModelFormat] = {
    "BAAI/bge-base-en-v1.5": ModelFormat(
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
        source=(
            "BGE v1.5 model card: for short-query-to-passage retrieval add the instruction to "
            "queries only; passages take no instruction. v1.5 also works without it, slightly "
            "worse. The questions are short queries over long passages, so it is used."
        ),
    ),
    "intfloat/e5-base-v2": ModelFormat(
        query_prefix="query: ",
        passage_prefix="passage: ",
        source=(
            "E5 model card: every input needs a prefix, 'query: ' for queries and 'passage: ' "
            "for passages in asymmetric retrieval; performance degrades without them."
        ),
    ),
}


def resolve_format(
    model_name: str, query_prefix: str | None = None, passage_prefix: str | None = None
) -> ModelFormat:
    """The input format for ``model_name``; configured prefixes must agree with a known model's format."""
    known = MODEL_FORMATS.get(model_name)
    if known is None:
        if query_prefix is None or passage_prefix is None:
            raise ValueError(
                f"{model_name} has no built-in input format: set query_prefix and passage_prefix "
                "explicitly (an empty string means no prefix)"
            )
        return ModelFormat(query_prefix, passage_prefix, source="configured explicitly")
    for field_name, configured, expected in (
        ("query_prefix", query_prefix, known.query_prefix),
        ("passage_prefix", passage_prefix, known.passage_prefix),
    ):
        if configured is not None and configured != expected:
            raise ValueError(
                f"{field_name} {configured!r} does not match the format of {model_name} "
                f"({expected!r}); one model's formatting rules must not be used for another"
            )
    return known


class TextEncoder(Protocol):
    """What :class:`BiEncoderRetriever` needs from a model.

    :class:`SentenceTransformerEncoder` wraps a real model; tests use fakes.
    """

    embedding_dim: int
    max_seq_length: int | None
    device: str

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """One embedding row per text; ``texts`` already carry their prefix."""
        ...

    def token_counts(self, texts: Sequence[str]) -> list[int] | None:
        """Tokens per text, special tokens included, before truncation; ``None`` if unknown."""
        ...

    def describe(self) -> dict[str, Any]:
        """Model details for provenance (pooling, modules, tokenizer, library versions)."""
        ...


def _library_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class SentenceTransformerEncoder:
    """:class:`TextEncoder` over a loaded ``SentenceTransformer``."""

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        revision: str,
        device: str,
        torch_threads: int | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.torch_threads = torch_threads
        self.embedding_dim = int(model.get_sentence_embedding_dimension())
        self.max_seq_length = int(model.max_seq_length)

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        # sentence-transformers runs the forward pass without autograd and moves each
        # batch to CPU memory before encoding the next.
        embeddings = self.model.encode(
            list(texts),
            prompt="",  # the prefixes are already applied; never add the model's own prompts
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalised by the retriever, the same for every model
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        tokenizer = self.model.tokenizer
        counts: list[int] = []
        for start in range(0, len(texts), 512):
            encoded = tokenizer(
                list(texts[start : start + 512]),
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
            counts.extend(len(ids) for ids in encoded["input_ids"])
        return counts

    def describe(self) -> dict[str, Any]:
        modules = [type(module).__name__ for module in self.model]
        pooling = next((m for m in self.model if type(m).__name__ == "Pooling"), None)
        tokenizer = self.model.tokenizer
        return {
            "modules": modules,
            "pooling": pooling.get_pooling_mode_str() if pooling is not None else None,
            "model_normalize_module": "Normalize" in modules,
            "similarity_fn": getattr(self.model, "similarity_fn_name", None),
            "model_prompts_not_applied": dict(getattr(self.model, "prompts", None) or {}),
            "tokenizer": {
                "class": type(tokenizer).__name__,
                "do_lower_case": getattr(tokenizer, "do_lower_case", None),
                "model_max_length": int(tokenizer.model_max_length),
                "vocab_size": int(tokenizer.vocab_size),
                "truncation_side": getattr(tokenizer, "truncation_side", None),
            },
            "torch_threads": self.torch_threads,
            "libraries": _library_versions(_RECORDED_LIBRARIES),
        }


def _import_backend() -> tuple[Any, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    return torch, SentenceTransformer


def resolve_device(torch: Any, device: str) -> str:
    """``auto`` is CUDA when available, else CPU. MPS is used only when asked for by name."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise DenseModelError(f"device {device!r} was requested but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise DenseModelError("device 'mps' was requested but MPS is not available")
    return device


def load_sentence_transformer(
    model_name: str,
    *,
    revision: str,
    device: str = "auto",
    local_files_only: bool = False,
    max_seq_length: int | None = None,
) -> SentenceTransformerEncoder:
    """Load ``model_name`` at ``revision``; every failure is a :class:`DenseModelError` saying why."""
    try:
        torch, sentence_transformer = _import_backend()
    except Exception as exc:  # ImportError, or a broken binary dependency failing at import
        raise DenseModelError(
            f"cannot load {model_name}: importing torch / sentence-transformers failed "
            f"({type(exc).__name__}: {exc}). Dense retrieval needs the full requirements.txt."
        ) from exc
    resolved = resolve_device(torch, device)
    try:
        model = sentence_transformer(
            model_name,
            revision=revision,
            device=resolved,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except Exception as exc:
        hint = (
            "it is not in the local Hugging Face cache and local_files_only is set"
            if local_files_only
            else "check the model name, the revision and access to huggingface.co"
        )
        raise DenseModelError(
            f"could not load {model_name} at revision {revision} ({type(exc).__name__}: {exc}); "
            f"{hint}"
        ) from exc
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return SentenceTransformerEncoder(
        model,
        model_name=model_name,
        revision=revision,
        device=resolved,
        torch_threads=torch.get_num_threads(),
    )


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (float32); a zero or non-finite row is an error."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise DenseModelError(f"expected a 2-D embedding matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise DenseModelError("the model produced a zero or non-finite embedding")
    return (matrix / norms).astype(np.float32)


def truncation_stats(
    token_counts: Sequence[int] | None, max_seq_length: int | None
) -> dict[str, Any] | None:
    """How many texts exceed ``max_seq_length`` tokens, and how much text the model never sees."""
    if token_counts is None or max_seq_length is None or not token_counts:
        return None
    total = sum(token_counts)
    truncated = [c for c in token_counts if c > max_seq_length]
    return {
        "max_seq_length": max_seq_length,
        "texts": len(token_counts),
        "truncated": len(truncated),
        "truncated_share": round(len(truncated) / len(token_counts), 4),
        "tokens_median": statistics.median(token_counts),
        "tokens_mean": round(total / len(token_counts), 1),
        "tokens_max": max(token_counts),
        "tokens_beyond_limit_share": round(sum(c - max_seq_length for c in truncated) / total, 4),
    }


class BiEncoderRetriever:
    """Dense retriever over L2-normalised embeddings from one model at a pinned revision."""

    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str,
        retriever_name: str | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        batch_size: int = 32,
        device: str = "auto",
        max_seq_length: int | None = None,
        embedding_dim: int | None = None,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        corpus_sha256: str | None = None,
        encoder: TextEncoder | None = None,
        passage_view: str = "text",
    ) -> None:
        if not isinstance(model_revision, str) or not _COMMIT_RE.match(model_revision):
            raise ValueError(
                f"model_revision must be a full 40-character commit SHA, got {model_revision!r}"
            )
        self._configure(
            model_name,
            model_revision,
            resolve_format(model_name, query_prefix, passage_prefix),
            retriever_name=retriever_name,
            batch_size=batch_size,
            device=device,
            max_seq_length=max_seq_length,
            embedding_dim=embedding_dim,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            corpus_sha256=corpus_sha256,
            encoder=encoder,
            passage_view=passage_view,
        )

    def _configure(
        self,
        model_name: str,
        model_revision: str,
        input_format: ModelFormat,
        *,
        retriever_name: str | None,
        batch_size: int,
        device: str,
        max_seq_length: int | None,
        embedding_dim: int | None,
        local_files_only: bool,
        cache_dir: str | Path | None,
        corpus_sha256: str | None,
        encoder: TextEncoder | None,
        passage_view: str = "text",
    ) -> None:
        """Validate the settings every dense retriever shares; ``model_revision`` was checked by the caller."""
        if passage_view not in PASSAGE_VIEWS:
            raise ValueError(f"passage_view must be one of {list(PASSAGE_VIEWS)}, got {passage_view!r}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if not _DEVICE_RE.match(device):
            raise ValueError(f"device must be auto, cpu, mps, cuda or cuda:N, got {device!r}")
        if max_seq_length is not None and (
            not isinstance(max_seq_length, int) or max_seq_length <= 0
        ):
            raise ValueError(f"max_seq_length must be a positive integer, got {max_seq_length!r}")
        if cache_dir is not None and (corpus_sha256 is None or not _SHA256_RE.match(corpus_sha256)):
            raise ValueError("an embedding cache needs the corpus SHA-256 (corpus_sha256)")
        self.format = input_format
        self.model_name = model_name
        self.model_revision = model_revision
        self.retriever_name = retriever_name or model_name
        self.batch_size = batch_size
        self.device = device
        self.max_seq_length = max_seq_length
        self.expected_dim = embedding_dim
        self.local_files_only = local_files_only
        self.cache = EmbeddingCache(cache_dir) if cache_dir is not None else None
        self.corpus_sha256 = corpus_sha256
        self.passage_view = passage_view  # what is encoded for each passage (finretriever.retrieval.enrichment)
        self._encoder = encoder
        self.passages: list[Passage] = []
        self.passage_ids: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._scoring: np.ndarray | None = None
        self._embeddings_sha256: str | None = None
        self._passage_tickers: list[str] = []
        self._passage_truncation: dict[str, Any] | None = None
        self._cache_info: dict[str, Any] | None = None
        self._query_vectors: dict[str, np.ndarray] = {}
        self._query_tokens: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_encoder(self) -> TextEncoder:
        return load_sentence_transformer(
            self.model_name,
            revision=self.model_revision,
            device=self.device,
            local_files_only=self.local_files_only,
            max_seq_length=self.max_seq_length,
        )

    @property
    def encoder(self) -> TextEncoder:
        """The model, loaded on first use."""
        if self._encoder is None:
            self._encoder = self._load_encoder()
        encoder = self._encoder
        if self.expected_dim is not None and encoder.embedding_dim != self.expected_dim:
            raise DenseModelError(
                f"{self.model_name} produces {encoder.embedding_dim}-dimensional embeddings, "
                f"but the configuration expects {self.expected_dim}"
            )
        return encoder

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        encoder = self.encoder
        raw = encoder.encode(texts, self.batch_size)
        if not isinstance(raw, np.ndarray) or raw.shape != (len(texts), encoder.embedding_dim):
            shape = getattr(raw, "shape", None)
            raise DenseModelError(
                f"{self.model_name} returned embeddings of shape {shape} for {len(texts)} texts; "
                f"expected ({len(texts)}, {encoder.embedding_dim})"
            )
        return l2_normalize(raw)

    def _encoding_config(self) -> dict[str, Any]:
        """Everything besides corpus, model and revision that determines the passage embeddings."""
        encoder = self.encoder
        info = encoder.describe()
        libraries = info.get("libraries") or {}
        return {
            "passage_prefix": self.format.passage_prefix,
            "encoded_field": encoded_field(self.passage_view),
            "max_seq_length": encoder.max_seq_length,
            "pooling": info.get("pooling"),
            "normalization": "l2",
            "dtype": "float32",
            "device_type": encoder.device.split(":")[0],
            "batch_size": self.batch_size,
            "libraries": {name: libraries.get(name) for name in _KEY_LIBRARIES},
        }

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @property
    def is_indexed(self) -> bool:
        return self._scoring is not None

    @property
    def embeddings(self) -> np.ndarray | None:
        """The passage embeddings (float32, unit length, corpus order) once indexed."""
        return self._embeddings

    def __len__(self) -> int:
        return len(self.passages)

    def index(self, passages: Sequence[Passage]) -> None:
        """Embed ``passages`` (or reuse their cached embeddings); loads the model if needed."""
        if len(passages) == 0:
            raise ValueError("Cannot index an empty passage list.")
        seen: set[str] = set()
        for passage in passages:
            if passage.passage_id in seen:
                raise ValueError(f"Duplicate passage_id in corpus: {passage.passage_id!r}")
            seen.add(passage.passage_id)
        passages = list(passages)
        passage_ids = [p.passage_id for p in passages]
        texts = [self.format.passage_prefix + render_passage(p, self.passage_view) for p in passages]
        encoder = self.encoder
        truncation = truncation_stats(encoder.token_counts(texts), encoder.max_seq_length)

        key: EmbeddingCacheKey | None = None
        embeddings: np.ndarray | None = None
        if self.cache is not None:
            key = EmbeddingCacheKey(
                corpus_sha256=str(self.corpus_sha256),
                model_name=self.model_name,
                model_revision=self.model_revision,
                encoding=self._encoding_config(),
            )
            cached = self.cache.load(key, passage_ids)
            if cached is not None:
                embeddings = cached.embeddings
                logger.info(
                    "%s: reusing cached embeddings from %s", self.retriever_name, cached.path
                )
        if embeddings is None:
            started = time.perf_counter()
            embeddings = self._encode_passages(texts)
            if self.cache is not None and key is not None:
                path = self.cache.save(
                    key,
                    passage_ids,
                    embeddings,
                    provenance={
                        "retriever": self.retriever_name,
                        "encode_seconds": round(time.perf_counter() - started, 1),
                        "device": encoder.device,
                        "passage_truncation": truncation,
                        "model": encoder.describe(),
                    },
                )
                logger.info("%s: cached the embeddings in %s", self.retriever_name, path)
        if embeddings.shape[1] != encoder.embedding_dim:
            raise DenseModelError(
                f"cached embeddings have {embeddings.shape[1]} dimensions; {self.model_name} "
                f"produces {encoder.embedding_dim}"
            )
        check_embeddings(embeddings, len(passages))

        self.passages = passages
        self.passage_ids = passage_ids
        self._passage_tickers = [p.ticker for p in passages]
        self._embeddings = embeddings
        self._scoring = embeddings.astype(np.float64)
        self._embeddings_sha256 = hashlib.sha256(embeddings.tobytes()).hexdigest()
        self._passage_truncation = truncation
        self._cache_info = (
            None
            if self.cache is None or key is None
            else {
                "dir": str(self.cache.entry_dir(key)),
                "key": key.as_dict(),
                "key_sha256": key.digest(),
            }
        )

    def _encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        total = len(texts)
        batches = range(0, total, self.batch_size)
        log_every = max(1, len(batches) // 20)
        started = time.perf_counter()
        logger.info(
            "%s: encoding %d passages in batches of %d on %s",
            self.retriever_name,
            total,
            self.batch_size,
            self.encoder.device,
        )
        chunks: list[np.ndarray] = []
        for number, start in enumerate(batches, start=1):
            chunks.append(self._embed(texts[start : start + self.batch_size]))
            if number % log_every == 0 or number == len(batches):
                done = min(start + self.batch_size, total)
                elapsed = time.perf_counter() - started
                logger.info(
                    "%s: %d/%d passages (%.1f/s, about %.0f s left)",
                    self.retriever_name,
                    done,
                    total,
                    done / elapsed,
                    (total - done) * elapsed / done,
                )
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def _query_vector(self, query: str) -> np.ndarray:
        text = self.format.query_prefix + query
        if text not in self._query_vectors:
            self._query_vectors[text] = self._embed([text])[0].astype(np.float64)
            counts = self.encoder.token_counts([text])
            if counts is not None:
                self._query_tokens[text] = counts[0]
        return self._query_vectors[text]

    def scores(self, query: str) -> np.ndarray:
        """The cosine score of every indexed passage for ``query`` (float64, corpus order)."""
        if self._scoring is None:
            raise RuntimeError("BiEncoderRetriever.scores() called before index().")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        return self._scoring @ self._query_vector(query)

    def search(
        self,
        query: str,
        top_k: int = 10,
        query_id: str = "query",
        tickers: Collection[str] | None = None,
    ) -> list[SearchHit]:
        """Top ``top_k`` passages by cosine; with ``tickers``, only those companies' passages are ranked."""
        if self._scoring is None:
            raise RuntimeError("BiEncoderRetriever.search() called before index().")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}.")
        scores = self._scoring @ self._query_vector(query)
        eligible = eligible_indices(self._passage_tickers, tickers)
        order = rank_top_k(scores, top_k, eligible)
        return [
            SearchHit(
                query_id=query_id,
                passage_id=self.passages[doc_index].passage_id,
                rank=rank,
                score=float(scores[doc_index]),
                retriever=self.retriever_name,
                metadata=hit_metadata(self.passages[doc_index]),
            )
            for rank, doc_index in enumerate(order, start=1)
        ]

    def batch_search(
        self,
        queries: Mapping[str, str],
        top_k: int = 10,
        tickers: Mapping[str, Collection[str]] | None = None,
    ) -> list[SearchHit]:
        """``search`` for every query; ``tickers`` maps a query ID to its allowed companies."""
        restrictions = tickers or {}
        hits: list[SearchHit] = []
        for query_id, query_text in queries.items():
            hits.extend(
                self.search(
                    query_text, top_k=top_k, query_id=query_id, tickers=restrictions.get(query_id)
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Settings and model details that determine this retriever's ranking, for run provenance."""
        encoder = self._encoder
        query_tokens = list(self._query_tokens.values())
        limit = encoder.max_seq_length if encoder is not None else self.max_seq_length
        return {
            "name": self.retriever_name,
            "type": "dense",
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "query_prefix": self.format.query_prefix,
            "passage_prefix": self.format.passage_prefix,
            "passage_view": self.passage_view,
            "format_source": self.format.source,
            "embedding_dim": None if encoder is None else encoder.embedding_dim,
            "normalization": "L2 to unit length after the model's own modules (float32)",
            "similarity": "dot product of unit vectors (cosine), computed in float64",
            "query_encoding": "one query per forward pass",
            "batch_size": self.batch_size,
            "device": None if encoder is None else encoder.device,
            "max_seq_length": limit,
            "model": None if encoder is None else encoder.describe(),
            "passage_truncation": self._passage_truncation,
            "query_truncation": truncation_stats(query_tokens, limit),
            "embeddings_sha256": self._embeddings_sha256,
            "cache": self._cache_info,
        }


__all__ = [
    "MODEL_FORMATS",
    "BiEncoderRetriever",
    "DenseModelError",
    "ModelFormat",
    "SentenceTransformerEncoder",
    "TextEncoder",
    "l2_normalize",
    "load_sentence_transformer",
    "resolve_device",
    "resolve_format",
    "truncation_stats",
]
