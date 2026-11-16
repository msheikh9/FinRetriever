"""The annotation workspace: questions, candidate pools and the judgment log.

``AnnotationStore`` is the only writer of a workspace (see
``finretriever.eval_set.schemas`` for its files). Opening one checks that

1. the corpus matches its manifest and, via ``verify_corpus``, the frozen
   SHA-256 and version the gold set is pinned to;
2. the workspace was created for exactly that corpus;
3. every question, pool entry and history event is well formed, every passage
   ID exists in the corpus, and replaying the history gives a consistent state.

Any failure raises before anything can be annotated. Changes are written
immediately (questions and pools atomically, events appended and fsynced), so
a session can stop at any point and resume later. A store refuses to write
when ``history.jsonl`` changed under it, so two sessions cannot interleave
conflicting judgments.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from finretriever.data.schemas import Passage
from finretriever.eval_set.schemas import (
    SKIPPED,
    AnnotationEvent,
    Candidate,
    GoldQuestion,
    PoolEntry,
    QuestionStatus,
    WorkspaceInfo,
    check_grade,
    normalize_question,
)
from finretriever.evaluation.provenance import (
    CorpusIdentity,
    CorpusIntegrityError,
    CorpusMismatchError,
    check_annotations_match_corpus,
    evaluation_metadata_for,
    load_corpus_identity,
)
from finretriever.utils.io import iter_jsonl, write_jsonl

WORKSPACE_FILE = "workspace.json"
QUESTIONS_FILE = "questions.jsonl"
POOL_FILE = "pool.jsonl"
HISTORY_FILE = "history.jsonl"

T = TypeVar("T", bound=BaseModel)


class AnnotationError(ValueError):
    """An annotation action or workspace file breaks the gold-set rules."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _explain(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors())


def verify_corpus(
    corpus_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    expected_sha256: str | None = None,
    expected_version: str | None = None,
) -> CorpusIdentity:
    """Check a corpus against its manifest and the frozen identity the gold set is pinned to."""
    corpus = load_corpus_identity(corpus_path, manifest_path)
    if expected_sha256 is not None and corpus.corpus_sha256 != expected_sha256:
        raise CorpusMismatchError(
            f"corpus {corpus.corpus_path} has sha256 {corpus.corpus_sha256[:12]}..., but the gold set "
            f"is frozen to {expected_sha256[:12]}...; annotate only the frozen corpus"
        )
    if expected_version is not None and corpus.corpus_version != expected_version:
        raise CorpusMismatchError(
            f"corpus {corpus.corpus_path} is {corpus.corpus_version}, but the gold set is frozen to "
            f"{expected_version}"
        )
    return corpus


def load_corpus_passages(corpus: CorpusIdentity) -> dict[str, Passage]:
    """The verified corpus as ``{passage_id: Passage}``, in corpus order."""
    passages: dict[str, Passage] = {}
    for passage in iter_jsonl(corpus.corpus_path, model=Passage):
        if passage.passage_id in passages:
            raise CorpusIntegrityError(f"duplicate passage_id {passage.passage_id} in {corpus.corpus_path}")
        passages[passage.passage_id] = passage
    if len(passages) != corpus.passages:
        raise CorpusIntegrityError(
            f"{corpus.corpus_path} holds {len(passages)} passages but its manifest records {corpus.passages}"
        )
    return passages


@dataclass(frozen=True)
class Judgment:
    """Current judgment of a pooled passage; ``grade is None`` means skipped (unjudged)."""

    query_id: str
    passage_id: str
    grade: int | None
    annotator: str
    at: str
    note: str = ""

    @property
    def state(self) -> str:
        return SKIPPED if self.grade is None else str(self.grade)


def _read_records(path: Path, model: type[T]) -> Iterator[tuple[int, T]]:
    if not path.is_file():
        raise AnnotationError(f"missing workspace file {path}")
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                yield line_no, model.model_validate_json(raw)
            except ValidationError as exc:
                raise AnnotationError(f"{path.name}:{line_no}: {_explain(exc)}") from None


def _write_json_atomic(path: Path, payload: object) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class AnnotationStore:
    """Questions, candidate pools and judgments of one workspace, bound to one corpus."""

    def __init__(
        self,
        directory: Path,
        info: WorkspaceInfo,
        corpus: CorpusIdentity,
        passages: dict[str, Passage],
        *,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.directory = directory
        self.info = info
        self.corpus = corpus
        self.passages = passages
        self.tickers = frozenset(p.ticker for p in passages.values())
        self._clock = clock
        self._questions: dict[str, GoldQuestion] = {}
        self._question_keys: dict[str, str] = {}  # normalized text -> query_id
        self._pools: dict[str, list[PoolEntry]] = {}
        self._pooled: dict[tuple[str, str], PoolEntry] = {}
        self._judgments: dict[tuple[str, str], Judgment] = {}
        self._events: list[AnnotationEvent] = []
        self._history_size = 0

    # --- opening ---------------------------------------------------------------

    @classmethod
    def create(
        cls,
        directory: str | Path,
        corpus: CorpusIdentity,
        *,
        annotation_version: str,
        created_by: str,
        clock: Callable[[], str] = utc_now,
    ) -> AnnotationStore:
        """Start an empty workspace bound to ``corpus``; refuses to overwrite one."""
        directory = Path(directory)
        existing = [n for n in (WORKSPACE_FILE, QUESTIONS_FILE, POOL_FILE, HISTORY_FILE) if (directory / n).exists()]
        if existing:
            raise AnnotationError(f"{directory} already holds annotation files {existing}; open it instead")
        passages = load_corpus_passages(corpus)
        try:
            info = WorkspaceInfo(
                evaluation=evaluation_metadata_for(corpus, annotation_version),
                created_at=clock(),
                created_by=created_by,
            )
        except ValidationError as exc:
            raise AnnotationError(_explain(exc)) from None
        directory.mkdir(parents=True, exist_ok=True)
        for name in (QUESTIONS_FILE, POOL_FILE, HISTORY_FILE):
            (directory / name).touch()
        _write_json_atomic(directory / WORKSPACE_FILE, info.model_dump(mode="json"))
        return cls(directory, info, corpus, passages, clock=clock)

    @classmethod
    def open(
        cls, directory: str | Path, corpus: CorpusIdentity, *, clock: Callable[[], str] = utc_now
    ) -> AnnotationStore:
        """Open a workspace after checking it was made for ``corpus`` and is consistent."""
        directory = Path(directory)
        info_path = directory / WORKSPACE_FILE
        if not info_path.is_file():
            raise AnnotationError(f"no annotation workspace at {directory} (run init first)")
        try:
            info = WorkspaceInfo.model_validate_json(info_path.read_bytes())
        except ValidationError as exc:
            raise AnnotationError(f"{info_path}: {_explain(exc)}") from None
        check_annotations_match_corpus(info.evaluation, corpus)
        store = cls(directory, info, corpus, load_corpus_passages(corpus), clock=clock)
        store._load()
        return store

    def _load(self) -> None:
        for line_no, question in _read_records(self.directory / QUESTIONS_FILE, GoldQuestion):
            where = f"{QUESTIONS_FILE}:{line_no}"
            if question.query_id in self._questions:
                raise AnnotationError(f"{where}: duplicate query_id {question.query_id}")
            key = normalize_question(question.text)
            if key in self._question_keys:
                raise AnnotationError(f"{where}: {question.query_id} repeats {self._question_keys[key]}")
            if unknown := sorted(set(question.tickers) - self.tickers):
                raise AnnotationError(f"{where}: tickers not in the corpus: {unknown}")
            self._questions[question.query_id] = question
            self._question_keys[key] = question.query_id

        for line_no, entry in _read_records(self.directory / POOL_FILE, PoolEntry):
            where = f"{POOL_FILE}:{line_no}"
            if entry.query_id not in self._questions:
                raise AnnotationError(f"{where}: unknown query_id {entry.query_id}")
            if entry.passage_id not in self.passages:
                raise AnnotationError(f"{where}: passage {entry.passage_id} is not in the corpus")
            if (entry.query_id, entry.passage_id) in self._pooled:
                raise AnnotationError(f"{where}: {entry.passage_id} is pooled twice for {entry.query_id}")
            pool = self._pools.setdefault(entry.query_id, [])
            if entry.position != len(pool) + 1:
                raise AnnotationError(f"{where}: expected position {len(pool) + 1}, got {entry.position}")
            pool.append(entry)
            self._pooled[entry.query_id, entry.passage_id] = entry

        statuses: dict[str, str] = dict.fromkeys(self._questions, "open")
        history = self.directory / HISTORY_FILE
        for line_no, event in _read_records(history, AnnotationEvent):
            where = f"{HISTORY_FILE}:{line_no}"
            if event.seq != len(self._events) + 1:
                raise AnnotationError(f"{where}: expected event {len(self._events) + 1}, got {event.seq}")
            if event.query_id not in self._questions:
                raise AnnotationError(f"{where}: unknown query_id {event.query_id}")
            if event.kind == "question_status":
                if event.previous != statuses[event.query_id]:
                    raise AnnotationError(
                        f"{where}: status change from {event.previous!r}, but {event.query_id} was "
                        f"{statuses[event.query_id]!r}"
                    )
                statuses[event.query_id] = event.status  # type: ignore[assignment]
            elif event.kind in ("grade", "skip", "clear"):
                self._replay_judgment(event, where)
            self._events.append(event)
        for query_id, question in self._questions.items():
            if question.status != statuses[query_id]:
                raise AnnotationError(
                    f"{QUESTIONS_FILE} has {query_id} {question.status!r} but {HISTORY_FILE} leaves it "
                    f"{statuses[query_id]!r}"
                )
        self._history_size = history.stat().st_size

    def _replay_judgment(self, event: AnnotationEvent, where: str) -> None:
        key = (event.query_id, event.passage_id or "")
        if key not in self._pooled:
            raise AnnotationError(f"{where}: {event.passage_id} was never pooled for {event.query_id}")
        current = self._judgments.get(key)
        state = current.state if current else None
        if event.previous != state:
            raise AnnotationError(
                f"{where}: event replaces judgment {event.previous!r}, but the history so far gives "
                f"{state!r} — conflicting or out-of-order edits"
            )
        self._apply(event)

    def _apply(self, event: AnnotationEvent) -> None:
        key = (event.query_id, event.passage_id or "")
        if event.kind == "clear":
            self._judgments.pop(key, None)
        else:
            self._judgments[key] = Judgment(
                event.query_id, key[1], event.grade, event.annotator, event.at, event.note
            )

    # --- reading -----------------------------------------------------------------

    def questions(self) -> list[GoldQuestion]:
        return list(self._questions.values())

    def question(self, query_id: str) -> GoldQuestion:
        try:
            return self._questions[query_id]
        except KeyError:
            raise AnnotationError(f"unknown query_id {query_id}") from None

    def pool(self, query_id: str) -> list[PoolEntry]:
        self.question(query_id)
        return list(self._pools.get(query_id, []))

    def judgment(self, query_id: str, passage_id: str) -> Judgment | None:
        return self._judgments.get((query_id, passage_id))

    def judgments(self, query_id: str) -> list[Judgment]:
        """Judgments (grades and skips) of one question, in pool order."""
        return [j for e in self.pool(query_id) if (j := self.judgment(query_id, e.passage_id))]

    def pending(self, query_id: str) -> list[PoolEntry]:
        """Pool entries nobody has graded or skipped yet."""
        return [e for e in self.pool(query_id) if self.judgment(query_id, e.passage_id) is None]

    def history(self, query_id: str | None = None, passage_id: str | None = None) -> list[AnnotationEvent]:
        return [
            e
            for e in self._events
            if (query_id is None or e.query_id == query_id) and (passage_id is None or e.passage_id == passage_id)
        ]

    # --- writing -----------------------------------------------------------------

    def _check_unchanged(self) -> None:
        try:
            size = (self.directory / HISTORY_FILE).stat().st_size
        except FileNotFoundError:
            raise AnnotationError(f"{HISTORY_FILE} disappeared from {self.directory}") from None
        if size != self._history_size:
            raise AnnotationError(
                f"{HISTORY_FILE} changed since this workspace was opened (another session?); reopen it"
            )

    def _append(self, kind: str, query_id: str, *, annotator: str, **fields: object) -> AnnotationEvent:
        try:
            event = AnnotationEvent(
                seq=len(self._events) + 1, at=self._clock(), annotator=annotator, kind=kind, query_id=query_id, **fields
            )
        except ValidationError as exc:
            raise AnnotationError(_explain(exc)) from None
        path = self.directory / HISTORY_FILE
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._history_size = path.stat().st_size
        self._events.append(event)
        return event

    def _require_open(self, query_id: str) -> GoldQuestion:
        question = self.question(query_id)
        if question.status != "open":
            raise AnnotationError(f"{query_id} is {question.status}; reopen it before changing it")
        return question

    def _save_questions(self) -> None:
        write_jsonl(self.directory / QUESTIONS_FILE, self._questions.values())

    def _save_pools(self) -> None:
        write_jsonl(self.directory / POOL_FILE, (e for pool in self._pools.values() for e in pool))

    def add_question(
        self,
        text: str,
        *,
        topic: str,
        question_type: str,
        authoring: str,
        author: str,
        comparison: str = "none",
        tickers: Iterable[str] = (),
        fiscal_years: Iterable[int] = (),
        notes: str = "",
        ai_assisted: bool = False,
        reviewed_by: str = "",
    ) -> GoldQuestion:
        """Register a question under the next free ``q_NNNN`` ID.

        AI-drafted questions pass ``ai_assisted=True`` and the approving
        reviewer; see ``GoldQuestion``.
        """
        self._check_unchanged()
        key = normalize_question(text)
        if key in self._question_keys:
            raise AnnotationError(f"question already exists as {self._question_keys[key]}")
        if unknown := sorted({t.strip().upper() for t in tickers} - self.tickers):
            raise AnnotationError(f"tickers not in the corpus: {unknown}")
        number = len(self._questions) + 1
        while f"q_{number:04d}" in self._questions:
            number += 1
        try:
            question = GoldQuestion(
                query_id=f"q_{number:04d}",
                text=text.strip(),
                topic=topic,
                question_type=question_type,
                comparison=comparison,
                tickers=list(tickers),
                fiscal_years=list(fiscal_years),
                authoring=authoring,
                author=author,
                ai_assisted=ai_assisted,
                reviewed_by=reviewed_by,
                created_at=self._clock(),
                notes=notes,
            )
        except ValidationError as exc:
            raise AnnotationError(_explain(exc)) from None
        self._questions[question.query_id] = question
        self._question_keys[key] = question.query_id
        self._save_questions()
        self._append("question_added", question.query_id, annotator=author, note=question.text)
        return question

    def set_status(self, query_id: str, status: QuestionStatus, *, annotator: str, note: str = "") -> GoldQuestion:
        """Mark a question complete, withdrawn or open again.

        ``complete`` requires a non-empty pool in which every passage is graded
        or skipped; a complete question is frozen until reopened.
        """
        self._check_unchanged()
        question = self.question(query_id)
        if question.status == status:
            return question
        if status == "complete":
            if not self._pools.get(query_id):
                raise AnnotationError(f"{query_id} has no candidates to judge")
            if pending := self.pending(query_id):
                raise AnnotationError(
                    f"{query_id} has {len(pending)} candidates without a grade or skip; judge or skip them first"
                )
        updated = question.model_copy(update={"status": status})
        self._questions[query_id] = updated
        self._save_questions()
        self._append("question_status", query_id, annotator=annotator, status=status, previous=question.status, note=note)
        return updated

    def add_candidates(self, query_id: str, candidates: Iterable[Candidate], *, added_by: str) -> list[PoolEntry]:
        """Add candidates to a question's pool; returns the entries that are new.

        A passage proposed again by another source keeps its position and gains
        that source, so the pool records every reason it was shown.
        """
        self._check_unchanged()
        self._require_open(query_id)
        candidates = list(candidates)
        if unknown := sorted({c.passage_id for c in candidates} - self.passages.keys()):
            raise AnnotationError(f"passage IDs not in the corpus: {unknown}")
        pool = self._pools.setdefault(query_id, [])
        now = self._clock()
        new: list[PoolEntry] = []
        updated = 0
        for candidate in candidates:
            key = (query_id, candidate.passage_id)
            entry = self._pooled.get(key)
            if entry is None:
                entry = PoolEntry(
                    query_id=query_id,
                    passage_id=candidate.passage_id,
                    position=len(pool) + 1,
                    sources=[candidate.source],
                    added_at=now,
                    added_by=added_by,
                )
                pool.append(entry)
                new.append(entry)
            elif candidate.source not in entry.sources:
                entry = entry.model_copy(update={"sources": [*entry.sources, candidate.source]})
                pool[entry.position - 1] = entry
                updated += 1
            else:
                continue
            self._pooled[key] = entry
        if new or updated:
            self._save_pools()
            methods = Counter(c.source.method for c in candidates)
            summary = ", ".join(f"{m}: {n}" for m, n in sorted(methods.items()))
            self._append(
                "candidates_added", query_id, annotator=added_by,
                note=f"{len(new)} new, {updated} with another source ({summary})",
            )
        return new

    def judge(self, query_id: str, passage_id: str, grade: int, *, annotator: str, note: str = "") -> Judgment:
        """Grade a pooled passage 0, 1 or 2; re-grading records a new revision."""
        try:
            check_grade(grade)
        except ValueError as exc:
            raise AnnotationError(str(exc)) from None
        judgment = self._record("grade", query_id, passage_id, annotator=annotator, grade=grade, note=note)
        assert judgment is not None
        return judgment

    def skip(self, query_id: str, passage_id: str, *, annotator: str, note: str = "") -> Judgment:
        """Leave a pooled passage explicitly unjudged (e.g. uncertain); it is not a grade 0."""
        judgment = self._record("skip", query_id, passage_id, annotator=annotator, note=note)
        assert judgment is not None
        return judgment

    def clear(self, query_id: str, passage_id: str, *, annotator: str, note: str = "") -> None:
        """Withdraw a grade or skip, returning the passage to pending."""
        self._record("clear", query_id, passage_id, annotator=annotator, note=note)

    def _record(
        self, kind: str, query_id: str, passage_id: str, *, annotator: str, grade: int | None = None, note: str = ""
    ) -> Judgment | None:
        self._check_unchanged()
        self._require_open(query_id)
        if passage_id not in self.passages:
            raise AnnotationError(f"passage {passage_id} is not in the corpus")
        if (query_id, passage_id) not in self._pooled:
            raise AnnotationError(
                f"passage {passage_id} is not in the candidate pool of {query_id}; add it as a manual candidate first"
            )
        current = self.judgment(query_id, passage_id)
        state = current.state if current else None
        target = None if kind == "clear" else SKIPPED if kind == "skip" else str(grade)
        if target == state:
            return current
        event = self._append(
            kind, query_id, annotator=annotator, passage_id=passage_id, grade=grade, previous=state, note=note
        )
        self._apply(event)
        return self.judgment(query_id, passage_id)


__all__ = [
    "HISTORY_FILE",
    "POOL_FILE",
    "QUESTIONS_FILE",
    "WORKSPACE_FILE",
    "AnnotationError",
    "AnnotationStore",
    "Judgment",
    "load_corpus_passages",
    "utc_now",
    "verify_corpus",
]
