"""Reproducible on-disk cache of passage embeddings.

One cache entry is one directory, ``<root>/<model slug>/<key digest>/``:

* ``embeddings.npy`` — float32 array (passages x dimensions); row ``i`` is
  passage ``i`` of the corpus in file order;
* ``passage_ids.json`` — the passage IDs in that order;
* ``meta.json`` — written last: the full key, the shape, and the SHA-256 of
  the two files above, plus provenance that is not part of the key.

The key is the SHA-256 of a canonical JSON object holding the corpus SHA-256,
the model name, the resolved model revision and the encoding configuration
(prefix, maximum sequence length, pooling, normalisation, dtype, device type,
batch size). An entry is reused only when its metadata matches the key and
the current passage order exactly and its files match their recorded hashes.
Anything else is rejected with an error; it is never silently rebuilt. A
directory without ``meta.json`` is an interrupted write and is replaced.

Every file is written to a temporary name and moved into place, so a crash
never leaves a half-written file under its final name.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CACHE_FORMAT_VERSION = 1
EMBEDDINGS_FILE = "embeddings.npy"
PASSAGE_IDS_FILE = "passage_ids.json"
META_FILE = "meta.json"
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


class EmbeddingCacheError(ValueError):
    """A cache entry is incompatible with the current corpus or configuration, or is corrupt."""


@dataclass(frozen=True)
class EmbeddingCacheKey:
    """Everything that determines the cached embeddings."""

    corpus_sha256: str
    model_name: str
    model_revision: str
    encoding: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_version": CACHE_FORMAT_VERSION,
            "corpus_sha256": self.corpus_sha256,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "encoding": dict(sorted(self.encoding.items())),
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedEmbeddings:
    embeddings: np.ndarray
    meta: dict[str, Any]
    path: Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _ids_bytes(passage_ids: Sequence[str]) -> bytes:
    return (json.dumps(list(passage_ids), ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, write: Callable[[Any], None]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def check_embeddings(embeddings: np.ndarray, rows: int, *, normalized: bool = True) -> None:
    """Raise unless ``embeddings`` is a finite float32 matrix with ``rows`` rows (unit-norm if ``normalized``)."""
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise EmbeddingCacheError("embeddings must be a 2-D array")
    if embeddings.dtype != np.float32:
        raise EmbeddingCacheError(f"embeddings must be float32, got {embeddings.dtype}")
    if embeddings.shape[0] != rows:
        raise EmbeddingCacheError(f"{embeddings.shape[0]} embeddings for {rows} passages")
    if not np.isfinite(embeddings).all():
        raise EmbeddingCacheError("embeddings contain NaN or infinite values")
    if normalized:
        norms = np.linalg.norm(embeddings, axis=1)
        if np.abs(norms - 1.0).max(initial=0.0) > 1e-3:
            raise EmbeddingCacheError("embeddings are not L2-normalised")


class EmbeddingCache:
    """Passage embeddings on disk under ``root``, one directory per key."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def entry_dir(self, key: EmbeddingCacheKey) -> Path:
        slug = _SLUG_RE.sub("_", key.model_name.replace("/", "__")).strip("_") or "model"
        return self.root / slug / key.digest()[:16]

    def load(self, key: EmbeddingCacheKey, passage_ids: Sequence[str]) -> CachedEmbeddings | None:
        """The cached embeddings for ``key``, ``None`` if there are none; raises if the entry is incompatible."""
        directory = self.entry_dir(key)
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EmbeddingCacheError(
                f"{meta_path} is not valid JSON: {exc}; delete {directory} to rebuild"
            ) from None

        def reject(reason: str) -> EmbeddingCacheError:
            return EmbeddingCacheError(
                f"incompatible embedding cache {directory}: {reason}; delete it to rebuild"
            )

        if meta.get("format_version") != CACHE_FORMAT_VERSION:
            raise reject(
                f"format version {meta.get('format_version')!r}, expected {CACHE_FORMAT_VERSION}"
            )
        if meta.get("key") != key.as_dict() or meta.get("key_sha256") != key.digest():
            raise reject(
                "its key does not match the corpus, model, revision or encoding configuration"
            )
        ids_path = directory / PASSAGE_IDS_FILE
        if not ids_path.is_file() or _sha256_file(ids_path) != meta.get("passage_ids_sha256"):
            raise reject(f"{PASSAGE_IDS_FILE} is missing or does not match its recorded SHA-256")
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if cached_ids != list(passage_ids):
            raise reject("the passage order differs from the corpus being indexed")
        emb_path = directory / EMBEDDINGS_FILE
        if not emb_path.is_file() or _sha256_file(emb_path) != meta.get("embeddings_sha256"):
            raise reject(f"{EMBEDDINGS_FILE} is missing or does not match its recorded SHA-256")
        embeddings = np.load(emb_path, allow_pickle=False)
        if list(embeddings.shape) != [len(cached_ids), meta.get("embedding_dim")]:
            raise reject(f"shape {list(embeddings.shape)} does not match the metadata")
        try:
            check_embeddings(embeddings, len(cached_ids), normalized=meta.get("normalized", True))
        except EmbeddingCacheError as exc:
            raise reject(str(exc)) from None
        return CachedEmbeddings(embeddings, meta, directory)

    def save(
        self,
        key: EmbeddingCacheKey,
        passage_ids: Sequence[str],
        embeddings: np.ndarray,
        *,
        normalized: bool = True,
        provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write a new entry for ``key``; refuses to replace a complete entry."""
        check_embeddings(embeddings, len(passage_ids), normalized=normalized)
        directory = self.entry_dir(key)
        if (directory / META_FILE).exists():
            raise EmbeddingCacheError(
                f"{directory} already holds a cache entry; it is never overwritten"
            )
        directory.mkdir(parents=True, exist_ok=True)
        ids_bytes = _ids_bytes(passage_ids)
        _atomic_write(
            directory / EMBEDDINGS_FILE, lambda fh: np.save(fh, embeddings, allow_pickle=False)
        )
        _atomic_write(directory / PASSAGE_IDS_FILE, lambda fh: fh.write(ids_bytes))
        meta = {
            "format_version": CACHE_FORMAT_VERSION,
            "key": key.as_dict(),
            "key_sha256": key.digest(),
            "passages": len(passage_ids),
            "embedding_dim": int(embeddings.shape[1]),
            "dtype": "float32",
            "normalized": normalized,
            "passage_ids_sha256": _sha256_bytes(ids_bytes),
            "embeddings_sha256": _sha256_file(directory / EMBEDDINGS_FILE),
            "bytes": (directory / EMBEDDINGS_FILE).stat().st_size,
            "provenance": dict(provenance or {}),
        }
        payload = (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        _atomic_write(directory / META_FILE, lambda fh: fh.write(payload))
        return directory


__all__ = [
    "CACHE_FORMAT_VERSION",
    "EMBEDDINGS_FILE",
    "META_FILE",
    "PASSAGE_IDS_FILE",
    "CachedEmbeddings",
    "EmbeddingCache",
    "EmbeddingCacheError",
    "EmbeddingCacheKey",
    "check_embeddings",
]
