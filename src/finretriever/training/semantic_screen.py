"""Semantic leakage screening: training questions against the permitted evaluation question texts, by meaning.

The lexical check (word-set Jaccard, ``pair_quality``) misses a paraphrase of an evaluation question. This screen
embeds every training question and every evaluation question with the zero-shot encoder the preparation pins
(``visible_text``: BGE-base at its revision), without the query instruction (question-to-question similarity),
and flags a training question whose cosine similarity with any evaluation question reaches ``THRESHOLD`` (0.9, as
docs/training_plan_v1.md set it). Evaluation passages are kept out of training separately, by the blocklist; this
screen is about question wording only.

**What it reads.** Only the question files the preparation config names as leakage sources
(``leakage.evaluation_question_files``: the annotation workspace's questions and the gold questions), and from each
row only ``query_id`` and ``text``. A file named like a qrels file, or a row carrying a passage, a relevance grade, a
judgment or a pool (``FORBIDDEN_FIELDS``), is refused before anything is encoded. No qrels, relevance labels,
judged passages, pools or audits are opened.

**Isolation.** The screen runs offline, before candidates are built. Only the candidate builder consults its
record (``candidates.semantic_screening``: a flagged pair, or one not screened, is excluded). The trainer, the
validation evaluator and checkpoint selection never import or read it, and the candidate loader does not open it.

**The record** (``<preparation>/screening/<name>/``, written once): ``screening.jsonl``, one row per distinct
training question (its SHA-256, the source pair, the highest similarity and the ID of the nearest evaluation
question, never that question's text), and ``screening.json``, the provenance: the question files and their
SHA-256, the fields read, the encoder, its revision and library versions, the threshold and the subjects' sources.
An identical rebuild writes nothing; a different one is refused.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from finretriever.evaluation.provenance import sha256_file
from finretriever.training.carryover import question_sha256
from finretriever.training.data_prep import DataPrepConfig, PreparedData

SCREEN_KIND = "semantic_leakage_screening"
SCREEN_FORMAT = 1
SCREENING_DIRNAME = "screening"
MANIFEST_FILE = "screening.json"
ROWS_FILE = "screening.jsonl"
THRESHOLD = 0.9
FIELDS_READ = ("query_id", "text")
FORBIDDEN_FIELDS = frozenset({
    "passage_id", "passage_ids", "relevance", "relevant", "grade", "grades", "judgment", "judgments", "judged",
    "qrels", "label", "labels", "positives", "pool", "candidates",
})
METHOD = (
    "cosine similarity of L2-normalised zero-shot embeddings (the preparation's pinned encoder, no query instruction) "
    f"between each training question and each permitted evaluation question; flagged at {THRESHOLD} or above"
)
NOTICE = (
    "A leakage screen on question wording, made from permitted evaluation question texts only (query_id and text). "
    "No qrels, relevance labels, judged passages, pools or audits were read. Used only to exclude training "
    "candidates; never by training, validation or checkpoint selection."
)


class ScreeningError(ValueError):
    """The screen cannot run safely, or its record does not match what it names."""


class QuestionEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """One L2-normalised row per text."""
        ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Subject:
    """A training question to screen: where it comes from and its text."""

    source: str  # e.g. "candidates:train_candidates_v2" or "pairs:bge_base_ft_v3"
    pair_id: str
    question: str


def load_permitted_questions(config: DataPrepConfig) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """``(query_id, text)`` of every permitted evaluation question, and each file's provenance; see the module
    docstring for what is refused."""
    questions: list[tuple[str, str]] = []
    files = []
    for raw in config.leakage.evaluation_question_files:
        path = Path(raw)
        if "qrels" in path.name.lower():
            raise ScreeningError(f"{path} is named like a qrels file; only question texts may be screened against")
        if not path.is_file():
            raise ScreeningError(f"evaluation question file {path} is missing; the screen fails closed")
        count = 0
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or (forbidden := sorted(FORBIDDEN_FIELDS & set(row))):
                raise ScreeningError(
                    f"{path}:{line_no} carries {forbidden or 'no question'}: not a question row; relevance labels and "
                    "judged passages are never read"
                )
            questions.append((f"{path}:{row['query_id']}", str(row["text"])))
            count += 1
        files.append({"path": str(path), "sha256": sha256_file(path), "questions": count})
    return questions, files


def screen(
    subjects: Sequence[Subject],
    evaluation: Sequence[tuple[str, str]],
    encoder: QuestionEncoder,
    *,
    threshold: float = THRESHOLD,
) -> list[dict[str, Any]]:
    """One row per distinct training question (the first subject asking it), in subject order."""
    distinct: dict[str, Subject] = {}
    for subject in subjects:
        distinct.setdefault(question_sha256(subject.question), subject)
    rows = []
    if not distinct:
        return rows
    ids = [qid for qid, _ in evaluation]
    queries = list(distinct.values())
    if evaluation:
        similarity = encoder.encode([q.question for q in queries]) @ encoder.encode([t for _, t in evaluation]).T
    for i, (key, subject) in enumerate(distinct.items()):
        best = int(np.argmax(similarity[i])) if evaluation else None
        value = float(similarity[i, best]) if best is not None else 0.0
        rows.append({
            "question_sha256": key,
            "subject": f"{subject.source}:{subject.pair_id}",
            "max_similarity": round(value, 6),
            "nearest_evaluation_question": ids[best] if best is not None else None,
            "flagged": value >= threshold,
        })
    return rows


def screening_dir(prepared: PreparedData, name: str) -> Path:
    return prepared.directory / SCREENING_DIRNAME / name


def build_screening(
    name: str,
    *,
    prepared: PreparedData,
    config: DataPrepConfig,
    subjects: Sequence[Subject],
    sources: Sequence[Mapping[str, Any]],
    encoder: QuestionEncoder,
    threshold: float = THRESHOLD,
) -> dict[str, bytes]:
    """The record's files (not yet written); see the module docstring."""
    evaluation, files = load_permitted_questions(config)
    rows = screen(subjects, evaluation, encoder, threshold=threshold)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    manifest = {
        "kind": SCREEN_KIND,
        "format_version": SCREEN_FORMAT,
        "name": name,
        "notice": NOTICE,
        "method": METHOD,
        "threshold": threshold,
        "encoder": encoder.describe(),
        "evaluation_questions": {
            "files": files,
            "fields_read": list(FIELDS_READ),
            "never_read": ["qrels", "relevance labels", "judged passages", "pools", "audits"],
        },
        "preparation": {"version": prepared.manifest["version"], "manifest_sha256": prepared.manifest_sha256},
        "subjects": {"sources": list(sources), "questions": len(rows)},
        "stats": {"flagged": sum(r["flagged"] for r in rows),
                  "max_similarity": max((r["max_similarity"] for r in rows), default=None)},
        "files": {ROWS_FILE: hashlib.sha256(data).hexdigest()},
    }
    return {ROWS_FILE: data, MANIFEST_FILE: (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()}


def write_screening(prepared: PreparedData, name: str, files: Mapping[str, bytes]) -> bool:
    """Write the record once; ``False`` if an identical one is there, an error if a different one is."""
    directory = screening_dir(prepared, name)
    if directory.exists():
        present = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        if present == dict(files):
            return False
        raise ScreeningError(f"{directory} already holds a different screening; it is never written over")
    tmp = directory.with_name(f".{name}.tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for file_name, data in files.items():
        (tmp / file_name).write_bytes(data)
    tmp.rename(directory)
    return True


def read_screening(prepared: PreparedData, name: str) -> dict[str, Any]:
    """A written record, checked against its own hashes and this preparation: its manifest, the manifest's
    SHA-256 and the rows by question SHA-256."""
    directory = screening_dir(prepared, name)
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ScreeningError(f"no semantic screening {name!r} at {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("kind"), manifest.get("format_version")) != (SCREEN_KIND, SCREEN_FORMAT):
        raise ScreeningError(f"{manifest_path} is not a format-{SCREEN_FORMAT} screening record")
    if manifest["preparation"]["manifest_sha256"] != prepared.manifest_sha256:
        raise ScreeningError(f"screening {name!r} was made for another preparation")
    rows_path = directory / ROWS_FILE
    if sha256_file(rows_path) != manifest["files"][ROWS_FILE]:
        raise ScreeningError(f"{rows_path} does not match {manifest_path}")
    for item in manifest["evaluation_questions"]["files"]:
        if not Path(item["path"]).is_file() or sha256_file(item["path"]) != item["sha256"]:
            raise ScreeningError(f"evaluation question file {item['path']} changed since screening {name!r}")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    return {"manifest": manifest, "manifest_sha256": sha256_file(manifest_path),
            "rows": {r["question_sha256"]: r for r in rows}}


class SentenceTransformerQuestionEncoder:
    """The preparation's pinned zero-shot encoder (sentence-transformers), questions without any prefix."""

    def __init__(self, model_name: str, revision: str, *, local_files_only: bool = True, device: str = "cpu") -> None:
        from finretriever.retrieval.biencoder import _import_backend

        _, sentence_transformer = _import_backend()
        self.model_name, self.revision, self.device = model_name, revision, device
        self.model = sentence_transformer(model_name, revision=revision, device=device,
                                          local_files_only=local_files_only)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                            convert_to_numpy=True), dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        versions: dict[str, str | None] = {"python": platform.python_version()}
        for package in ("sentence-transformers", "transformers", "torch"):
            try:
                versions[package] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                versions[package] = None
        return {"model": self.model_name, "revision": self.revision, "query_prefix": "", "normalize": True,
                "device": self.device, "versions": versions}


def default_encoder(config: DataPrepConfig, *, local_files_only: bool = True) -> SentenceTransformerQuestionEncoder:
    visible = config.visible_text
    return SentenceTransformerQuestionEncoder(visible.tokenizer, visible.revision, local_files_only=local_files_only)


EncoderLoader = Callable[[DataPrepConfig], QuestionEncoder]

__all__ = [
    "FIELDS_READ",
    "FORBIDDEN_FIELDS",
    "METHOD",
    "SCREEN_KIND",
    "THRESHOLD",
    "EncoderLoader",
    "QuestionEncoder",
    "ScreeningError",
    "SentenceTransformerQuestionEncoder",
    "Subject",
    "build_screening",
    "default_encoder",
    "load_permitted_questions",
    "read_screening",
    "screen",
    "screening_dir",
    "write_screening",
]
