"""Corpus identity checks that keep gold annotations tied to their corpus.

Gold queries and qrels refer to passages by ID, and a passage ID is
``(filing, section, chunk index)`` — it is not derived from the passage text.
Re-parsing or re-chunking can therefore change the text behind an existing
ID without changing the ID itself, and evaluating old qrels against such a
corpus would silently score the wrong passages.

The contract is deliberately small:

* A built corpus is identified by the SHA-256 of its JSONL file, recorded
  in the manifest the corpus builder writes next to it
  (``corpus.sha256``, ``corpus_version``). ``load_corpus_identity`` re-hashes
  the file and refuses a corpus that no longer matches its manifest.
* Annotations carry ``EvaluationMetadata`` — ``corpus_sha256``,
  ``corpus_version`` and their own ``annotation_version``.
  ``check_annotations_match_corpus`` refuses to pair them with any other
  corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from finretriever.data.corpus_builder import manifest_path_for

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CorpusIntegrityError(ValueError):
    """A corpus file or its manifest is missing, unreadable, or inconsistent."""


class EvaluationMetadataError(ValueError):
    """Annotation metadata is missing or does not follow the contract."""


class CorpusMismatchError(ValueError):
    """Annotations were made against a different corpus than the one supplied."""


def _check_sha256(value: str) -> str:
    if not _SHA256_RE.match(value):
        raise ValueError(f"not a lowercase hex SHA-256: {value!r}")
    return value


class CorpusIdentity(BaseModel):
    """A corpus file whose bytes were verified against its manifest."""

    model_config = ConfigDict(frozen=True)

    corpus_path: Path
    manifest_path: Path
    corpus_sha256: str
    corpus_version: str
    passages: int


class EvaluationMetadata(BaseModel):
    """What a set of gold annotations must record about the corpus it labels."""

    model_config = ConfigDict(extra="forbid")

    corpus_sha256: str
    corpus_version: str
    annotation_version: str

    @field_validator("corpus_sha256")
    @classmethod
    def _sha256(cls, v: str) -> str:
        return _check_sha256(v)

    @field_validator("corpus_version", "annotation_version")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in blocks so large corpora are not held in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_corpus_identity(
    corpus_path: str | Path, manifest_path: str | Path | None = None
) -> CorpusIdentity:
    """Verify ``corpus_path`` against its manifest and return its identity.

    ``manifest_path`` defaults to ``<stem>.manifest.json`` beside the corpus,
    as written by the corpus builder. Raises ``CorpusIntegrityError`` if
    either file is missing or unreadable, the manifest lacks the required
    fields, or the corpus bytes no longer hash to the recorded SHA-256.
    """
    corpus = Path(corpus_path)
    manifest_file = Path(manifest_path) if manifest_path is not None else manifest_path_for(corpus)
    if not manifest_file.is_file():
        raise CorpusIntegrityError(f"corpus manifest not found: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        recorded = manifest["corpus"]["sha256"]
        version = manifest["corpus_version"]
        passages = manifest["corpus"]["passages"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CorpusIntegrityError(
            f"manifest {manifest_file} is unreadable or lacks corpus.sha256 / corpus_version / "
            f"corpus.passages: {exc!r}"
        ) from None
    if not isinstance(recorded, str) or not _SHA256_RE.match(recorded):
        raise CorpusIntegrityError(f"manifest {manifest_file} has an invalid corpus.sha256: {recorded!r}")
    if not corpus.is_file():
        raise CorpusIntegrityError(f"corpus file not found: {corpus}")
    actual = sha256_file(corpus)
    if actual != recorded:
        raise CorpusIntegrityError(
            f"corpus {corpus} hashes to {actual[:12]}... but its manifest records {recorded[:12]}...; "
            "the corpus was modified or does not belong to this manifest — rebuild it"
        )
    return CorpusIdentity(
        corpus_path=corpus,
        manifest_path=manifest_file,
        corpus_sha256=actual,
        corpus_version=str(version),
        passages=int(passages),
    )


def evaluation_metadata_for(corpus: CorpusIdentity, annotation_version: str) -> EvaluationMetadata:
    """The metadata new annotations of ``corpus`` should be saved with."""
    return EvaluationMetadata(
        corpus_sha256=corpus.corpus_sha256,
        corpus_version=corpus.corpus_version,
        annotation_version=annotation_version,
    )


def load_evaluation_metadata(path: str | Path) -> EvaluationMetadata:
    """Read annotation metadata (a JSON object) or raise ``EvaluationMetadataError``."""
    file = Path(path)
    if not file.is_file():
        raise EvaluationMetadataError(f"evaluation metadata not found: {file}")
    try:
        return EvaluationMetadata.model_validate_json(file.read_bytes())
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise EvaluationMetadataError(f"invalid evaluation metadata {file}: {problems}") from None


def check_annotations_match_corpus(metadata: EvaluationMetadata, corpus: CorpusIdentity) -> None:
    """Raise ``CorpusMismatchError`` unless ``metadata`` was made against ``corpus``."""
    if metadata.corpus_sha256 != corpus.corpus_sha256:
        raise CorpusMismatchError(
            f"annotations {metadata.annotation_version!r} label corpus {metadata.corpus_version} "
            f"(sha256 {metadata.corpus_sha256[:12]}...), but {corpus.corpus_path} is "
            f"{corpus.corpus_version} (sha256 {corpus.corpus_sha256[:12]}...); their passage IDs may "
            "point at different text — rebuild the annotated corpus or re-annotate"
        )
    if metadata.corpus_version != corpus.corpus_version:
        raise CorpusMismatchError(
            f"annotations {metadata.annotation_version!r} record corpus_version "
            f"{metadata.corpus_version!r} but the manifest says {corpus.corpus_version!r}"
        )


def verify_evaluation_corpus(
    metadata_path: str | Path,
    corpus_path: str | Path,
    manifest_path: str | Path | None = None,
) -> CorpusIdentity:
    """Check corpus integrity and that the annotations belong to it; return its identity."""
    corpus = load_corpus_identity(corpus_path, manifest_path)
    check_annotations_match_corpus(load_evaluation_metadata(metadata_path), corpus)
    return corpus


__all__ = [
    "CorpusIdentity",
    "CorpusIntegrityError",
    "CorpusMismatchError",
    "EvaluationMetadata",
    "EvaluationMetadataError",
    "check_annotations_match_corpus",
    "evaluation_metadata_for",
    "load_corpus_identity",
    "load_evaluation_metadata",
    "sha256_file",
    "verify_evaluation_corpus",
]
