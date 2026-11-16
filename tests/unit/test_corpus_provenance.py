"""Unit tests for finretriever.evaluation.provenance.

Corpora are built with the real corpus builder from synthetic raw filings
(see ``tests.unit.test_corpus_builder``), then damaged one way at a time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from finretriever.data.corpus_builder import CORPUS_BUILDER_VERSION, build_corpus, write_corpus
from finretriever.evaluation.provenance import (
    CorpusIdentity,
    CorpusIntegrityError,
    CorpusMismatchError,
    EvaluationMetadata,
    EvaluationMetadataError,
    check_annotations_match_corpus,
    evaluation_metadata_for,
    load_corpus_identity,
    load_evaluation_metadata,
    sha256_file,
    verify_evaluation_corpus,
)
from finretriever.utils.io import read_jsonl
from tests.unit.test_corpus_builder import make_config, write_raw_filing


def build(tmp_path: Path, name: str = "corpus", **overrides: Any) -> tuple[Path, dict[str, Any]]:
    raw = tmp_path / "raw"
    if not raw.exists():
        write_raw_filing(raw)
    config = make_config(tmp_path, output_path=tmp_path / "out" / f"{name}.jsonl", **overrides)
    return config.output_path, write_corpus(build_corpus(config))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return build(tmp_path)[0]


def write_metadata(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- corpus integrity ---------------------------------------------------------------


def test_matching_corpus_and_manifest(corpus: Path) -> None:
    identity = load_corpus_identity(corpus)
    manifest = json.loads(corpus.with_name("corpus.manifest.json").read_text())
    assert identity.corpus_sha256 == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert identity.corpus_sha256 == manifest["corpus"]["sha256"]
    assert identity.corpus_version == manifest["corpus_version"] == f"test_corpus@{CORPUS_BUILDER_VERSION}"
    assert identity.passages == len(read_jsonl(corpus))
    assert identity.manifest_path == corpus.with_name("corpus.manifest.json")


def test_explicit_manifest_path(tmp_path: Path, corpus: Path) -> None:
    moved = tmp_path / "elsewhere.json"
    corpus.with_name("corpus.manifest.json").rename(moved)
    with pytest.raises(CorpusIntegrityError, match="corpus manifest not found"):
        load_corpus_identity(corpus)
    assert load_corpus_identity(corpus, moved).manifest_path == moved


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda data: data[:-1] + (b"X" if data[-1:] != b"X" else b"Y"),  # one byte changed
        lambda data: data[: len(data) // 2],  # truncated mid-record
        lambda data: data + data.splitlines(keepends=True)[0],  # duplicated passage appended
        lambda data: b"",  # emptied
    ],
    ids=["byte-flip", "truncated", "appended", "emptied"],
)
def test_corrupted_corpus_is_rejected(corpus: Path, corrupt) -> None:
    corpus.write_bytes(corrupt(corpus.read_bytes()))
    with pytest.raises(CorpusIntegrityError, match="hashes to .* but its manifest records"):
        load_corpus_identity(corpus)


def test_missing_corpus_is_rejected(corpus: Path) -> None:
    corpus.unlink()
    with pytest.raises(CorpusIntegrityError, match="corpus file not found"):
        load_corpus_identity(corpus)


@pytest.mark.parametrize(
    "manifest_text,match",
    [
        ("{not json", "unreadable"),
        (json.dumps({"corpus_version": "v"}), "lacks corpus.sha256"),
        (json.dumps({"corpus": {"sha256": "abc", "passages": 1}, "corpus_version": "v"}), "invalid corpus.sha256"),
        (json.dumps({"corpus": {"sha256": "0" * 64, "passages": 1}}), "corpus_version"),
        (json.dumps(["not", "an", "object"]), "unreadable or lacks"),
    ],
)
def test_malformed_manifest_is_rejected(corpus: Path, manifest_text: str, match: str) -> None:
    corpus.with_name("corpus.manifest.json").write_text(manifest_text, encoding="utf-8")
    with pytest.raises(CorpusIntegrityError, match=match):
        load_corpus_identity(corpus)


def test_manifest_from_another_corpus_is_rejected(tmp_path: Path, corpus: Path) -> None:
    other, _ = build(tmp_path, name="other", chunk_size_tokens=8, chunk_overlap_tokens=2)
    with pytest.raises(CorpusIntegrityError, match="does not belong to this manifest"):
        load_corpus_identity(corpus, other.with_name("other.manifest.json"))


def test_sha256_file_streams_in_blocks(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(b"abc" * 1000)
    assert sha256_file(path, block_size=7) == hashlib.sha256(b"abc" * 1000).hexdigest()


# --- evaluation metadata contract -------------------------------------------------


def test_annotations_round_trip_against_their_corpus(tmp_path: Path, corpus: Path) -> None:
    metadata = evaluation_metadata_for(load_corpus_identity(corpus), annotation_version="gold-v1")
    path = write_metadata(tmp_path / "gold_metadata.json", metadata.model_dump())
    assert load_evaluation_metadata(path) == metadata
    identity = verify_evaluation_corpus(path, corpus)
    assert identity.corpus_sha256 == metadata.corpus_sha256


def test_rebuilt_corpus_with_reused_passage_ids_is_rejected(tmp_path: Path, corpus: Path) -> None:
    gold = evaluation_metadata_for(load_corpus_identity(corpus), annotation_version="gold-v1")
    rebuilt, _ = build(tmp_path, name="rebuilt", chunk_size_tokens=8, chunk_overlap_tokens=2)
    old = {p["passage_id"]: p["text"] for p in read_jsonl(corpus)}
    new = {p["passage_id"]: p["text"] for p in read_jsonl(rebuilt)}
    shared = old.keys() & new.keys()
    assert shared and any(old[i] != new[i] for i in shared)  # same IDs, different text: the hazard
    with pytest.raises(CorpusMismatchError, match="passage IDs may point at different text"):
        check_annotations_match_corpus(gold, load_corpus_identity(rebuilt))


def test_version_label_mismatch_is_rejected(corpus: Path) -> None:
    identity = load_corpus_identity(corpus)
    metadata = EvaluationMetadata(
        corpus_sha256=identity.corpus_sha256, corpus_version="corpus_v0@1.0", annotation_version="gold-v1"
    )
    with pytest.raises(CorpusMismatchError, match="corpus_version"):
        check_annotations_match_corpus(metadata, identity)


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"corpus_version": "c@1.1", "annotation_version": "g1"}, "corpus_sha256: Field required"),
        ({"corpus_sha256": "0" * 64, "annotation_version": "g1"}, "corpus_version: Field required"),
        ({"corpus_sha256": "0" * 64, "corpus_version": "c@1.1"}, "annotation_version: Field required"),
        ({"corpus_sha256": "ABC", "corpus_version": "c", "annotation_version": "g"}, "not a lowercase hex SHA-256"),
        ({"corpus_sha256": "0" * 64, "corpus_version": " ", "annotation_version": "g"}, "must be non-empty"),
        ({"corpus_sha256": "0" * 64, "corpus_version": "c", "annotation_version": "g", "x": 1}, "x: Extra inputs"),
    ],
)
def test_invalid_metadata_is_rejected(tmp_path: Path, payload: dict[str, Any], match: str) -> None:
    with pytest.raises(EvaluationMetadataError, match=match):
        load_evaluation_metadata(write_metadata(tmp_path / "meta.json", payload))


def test_missing_metadata_file_is_rejected(tmp_path: Path, corpus: Path) -> None:
    with pytest.raises(EvaluationMetadataError, match="evaluation metadata not found"):
        verify_evaluation_corpus(tmp_path / "absent.json", corpus)


def test_corrupt_corpus_fails_before_metadata_is_trusted(tmp_path: Path, corpus: Path) -> None:
    metadata = evaluation_metadata_for(load_corpus_identity(corpus), annotation_version="gold-v1")
    path = write_metadata(tmp_path / "meta.json", metadata.model_dump())
    corpus.write_bytes(corpus.read_bytes().replace(b"widgets", b"gadgets"))
    with pytest.raises(CorpusIntegrityError):
        verify_evaluation_corpus(path, corpus)


def test_corpus_identity_is_immutable(corpus: Path) -> None:
    identity = load_corpus_identity(corpus)
    assert isinstance(identity, CorpusIdentity)
    with pytest.raises(ValidationError, match="frozen"):
        identity.corpus_sha256 = "0" * 64  # type: ignore[misc]
