"""Tests for the on-disk passage embedding cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from finretriever.retrieval.embedding_cache import (
    EMBEDDINGS_FILE,
    META_FILE,
    PASSAGE_IDS_FILE,
    EmbeddingCache,
    EmbeddingCacheError,
    EmbeddingCacheKey,
)

KEY = EmbeddingCacheKey(
    corpus_sha256="a" * 64,
    model_name="org/model",
    model_revision="b" * 40,
    encoding={"passage_prefix": "passage: ", "batch_size": 2, "max_seq_length": 512},
)
IDS = ["p1", "p2", "p3"]


def unit_rows(rows: int = 3, dim: int = 4, seed: int = 0) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def test_save_then_load_round_trips_with_metadata(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    embeddings = unit_rows()
    directory = cache.save(KEY, IDS, embeddings, provenance={"encode_seconds": 1.5})
    assert directory == cache.entry_dir(KEY) and directory.parent.name == "org__model"
    assert sorted(p.name for p in directory.iterdir()) == [EMBEDDINGS_FILE, META_FILE, PASSAGE_IDS_FILE]
    meta = json.loads((directory / META_FILE).read_text())
    assert meta["key"] == KEY.as_dict() and meta["key_sha256"] == KEY.digest()
    assert (meta["passages"], meta["embedding_dim"], meta["dtype"], meta["normalized"]) == (3, 4, "float32", True)
    assert meta["provenance"] == {"encode_seconds": 1.5}
    assert json.loads((directory / PASSAGE_IDS_FILE).read_text()) == IDS

    loaded = cache.load(KEY, IDS)
    assert loaded is not None and loaded.meta == meta and loaded.path == directory
    assert loaded.embeddings.dtype == np.float32 and np.array_equal(loaded.embeddings, embeddings)
    assert not list(tmp_path.rglob("*.tmp"))  # atomic writes leave no temporary files


def test_missing_entry_is_none_and_an_interrupted_write_is_replaced(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    assert cache.load(KEY, IDS) is None
    directory = cache.entry_dir(KEY)
    directory.mkdir(parents=True)
    (directory / EMBEDDINGS_FILE).write_bytes(b"half-written")  # no meta.json: the write never finished
    assert cache.load(KEY, IDS) is None
    cache.save(KEY, IDS, unit_rows())
    loaded = cache.load(KEY, IDS)
    assert loaded is not None and np.array_equal(loaded.embeddings, unit_rows())


def test_key_covers_corpus_model_revision_and_encoding() -> None:
    variants = [
        EmbeddingCacheKey("d" * 64, KEY.model_name, KEY.model_revision, KEY.encoding),
        EmbeddingCacheKey(KEY.corpus_sha256, "org/other", KEY.model_revision, KEY.encoding),
        EmbeddingCacheKey(KEY.corpus_sha256, KEY.model_name, "e" * 40, KEY.encoding),
        EmbeddingCacheKey(KEY.corpus_sha256, KEY.model_name, KEY.model_revision, {**KEY.encoding, "batch_size": 4}),
        EmbeddingCacheKey(KEY.corpus_sha256, KEY.model_name, KEY.model_revision, {**KEY.encoding, "passage_prefix": ""}),
    ]
    digests = {KEY.digest(), *(v.digest() for v in variants)}
    assert len(digests) == 6
    reordered = EmbeddingCacheKey(KEY.corpus_sha256, KEY.model_name, KEY.model_revision, dict(reversed(list(KEY.encoding.items()))))
    assert reordered.digest() == KEY.digest()


def test_an_entry_for_other_passages_or_another_key_is_rejected(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    directory = cache.save(KEY, IDS, unit_rows())
    with pytest.raises(EmbeddingCacheError, match="passage order differs"):
        cache.load(KEY, list(reversed(IDS)))
    with pytest.raises(EmbeddingCacheError, match="passage order differs"):
        cache.load(KEY, IDS[:2])
    meta = json.loads((directory / META_FILE).read_text())
    meta["key"]["model_revision"] = "f" * 40
    (directory / META_FILE).write_text(json.dumps(meta))
    with pytest.raises(EmbeddingCacheError, match="key does not match"):
        cache.load(KEY, IDS)


@pytest.mark.parametrize("damage", ["embeddings", "passage_ids", "meta_json", "format_version"])
def test_corrupt_entries_are_rejected(tmp_path: Path, damage: str) -> None:
    cache = EmbeddingCache(tmp_path)
    directory = cache.save(KEY, IDS, unit_rows())
    if damage == "embeddings":
        np.save(directory / EMBEDDINGS_FILE, unit_rows(seed=1), allow_pickle=False)
        expected = "does not match its recorded SHA-256"
    elif damage == "passage_ids":
        (directory / PASSAGE_IDS_FILE).write_text(json.dumps(["p1", "p3", "p2"]) + "\n")
        expected = "does not match its recorded SHA-256"
    elif damage == "meta_json":
        (directory / META_FILE).write_text("{not json")
        expected = "not valid JSON"
    else:
        meta = json.loads((directory / META_FILE).read_text())
        (directory / META_FILE).write_text(json.dumps({**meta, "format_version": 99}))
        expected = "format version"
    with pytest.raises(EmbeddingCacheError, match=expected):
        cache.load(KEY, IDS)


def test_save_validates_embeddings_and_never_overwrites(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    with pytest.raises(EmbeddingCacheError, match="normalised"):
        cache.save(KEY, IDS, unit_rows() * 2)
    with pytest.raises(EmbeddingCacheError, match="float32"):
        cache.save(KEY, IDS, unit_rows().astype(np.float64))
    with pytest.raises(EmbeddingCacheError, match="for 3 passages"):
        cache.save(KEY, IDS, unit_rows(rows=2))
    bad = unit_rows()
    bad[1, 0] = np.nan
    with pytest.raises(EmbeddingCacheError, match="NaN"):
        cache.save(KEY, IDS, bad)
    assert not cache.entry_dir(KEY).exists()
    cache.save(KEY, IDS, unit_rows())
    with pytest.raises(EmbeddingCacheError, match="never overwritten"):
        cache.save(KEY, IDS, unit_rows(seed=2))
