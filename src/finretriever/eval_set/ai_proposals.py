"""AI-proposed relevance labels: development data kept apart from human judgments.

A proposal set lives in its own directory (``data/eval/dev_ai/<name>/``):

* ``proposals.meta.json`` — ``ProposalSetInfo``: the labelled workspace and
  its corpus binding, the model, the method, and a notice that these are not
  human judgments.
* ``proposals.jsonl`` — one ``AIProposal`` per pooled passage: grade,
  confidence, a short evidence-based explanation, the model and a timestamp.

Proposals only cover pooled passages that no human has judged, never replace
a human judgment, and are never read by the gold export. The blind audit
(``finretriever.eval_set.audit``) measures how far they can be trusted.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from finretriever.eval_set.annotation_store import AnnotationError, AnnotationStore, utc_now
from finretriever.eval_set.schemas import (
    CONFIDENCE_LEVELS,
    RELEVANCE_GRADES,
    AIProposal,
    ProposalSetInfo,
)

PROPOSALS_FILE = "proposals.jsonl"
PROPOSALS_INFO_FILE = "proposals.meta.json"


class ProposalError(ValueError):
    """An AI proposal or proposal set breaks the separation rules."""


def _explain(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors())


class ProposalSet:
    """AI proposals for one annotation workspace, validated against it on every load."""

    def __init__(self, directory: Path, info: ProposalSetInfo, store: AnnotationStore) -> None:
        self.directory = directory
        self.info = info
        self.store = store
        self._proposals: dict[tuple[str, str], AIProposal] = {}

    @property
    def path(self) -> Path:
        return self.directory / PROPOSALS_FILE

    @classmethod
    def create(
        cls,
        directory: str | Path,
        store: AnnotationStore,
        *,
        model: str,
        method: str,
        method_description: str,
        created_by: str,
        clock: Callable[[], str] = utc_now,
    ) -> ProposalSet:
        """Start an empty proposal set for ``store``; refuses to overwrite one."""
        directory = Path(directory)
        if (directory / PROPOSALS_INFO_FILE).exists() or (directory / PROPOSALS_FILE).exists():
            raise ProposalError(f"{directory} already holds a proposal set; open it instead")
        if directory.resolve() == store.directory.resolve():
            raise ProposalError("AI proposals must not live in the human annotation workspace")
        try:
            info = ProposalSetInfo(
                evaluation=store.info.evaluation,
                source_workspace=str(store.directory),
                model=model,
                method=method,
                method_description=method_description,
                created_at=clock(),
                created_by=created_by,
            )
        except ValidationError as exc:
            raise ProposalError(_explain(exc)) from None
        directory.mkdir(parents=True, exist_ok=True)
        (directory / PROPOSALS_INFO_FILE).write_text(
            json.dumps(info.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (directory / PROPOSALS_FILE).touch()
        return cls(directory, info, store)

    @classmethod
    def open(cls, directory: str | Path, store: AnnotationStore) -> ProposalSet:
        """Open a proposal set after checking it labels ``store`` and its corpus."""
        directory = Path(directory)
        info_path = directory / PROPOSALS_INFO_FILE
        if not info_path.is_file():
            raise ProposalError(f"no proposal set at {directory}")
        try:
            info = ProposalSetInfo.model_validate_json(info_path.read_bytes())
        except ValidationError as exc:
            raise ProposalError(f"{info_path}: {_explain(exc)}") from None
        if info.evaluation != store.info.evaluation:
            raise ProposalError(
                f"proposals label {info.evaluation.model_dump()} but the workspace is "
                f"{store.info.evaluation.model_dump()}"
            )
        proposals = cls(directory, info, store)
        with proposals.path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                if raw.strip():
                    try:
                        proposal = AIProposal.model_validate_json(raw)
                    except ValidationError as exc:
                        raise ProposalError(f"{PROPOSALS_FILE}:{line_no}: {_explain(exc)}") from None
                    proposals._check(proposal, where=f"{PROPOSALS_FILE}:{line_no}")
                    proposals._proposals[proposal.query_id, proposal.passage_id] = proposal
        return proposals

    def _check(
        self, proposal: AIProposal, *, where: str, existing: Mapping[tuple[str, str], AIProposal] | None = None
    ) -> None:
        existing = self._proposals if existing is None else existing
        key = (proposal.query_id, proposal.passage_id)
        try:
            pooled = {e.passage_id for e in self.store.pool(proposal.query_id)}
        except AnnotationError as exc:
            raise ProposalError(f"{where}: {exc}") from None
        if proposal.passage_id not in pooled:
            raise ProposalError(f"{where}: {proposal.passage_id} is not pooled for {proposal.query_id}")
        if key in existing:
            raise ProposalError(f"{where}: a proposal for {key} already exists; proposals are not revised")
        judgment = self.store.judgment(*key)
        if judgment is not None and judgment.grade is not None:
            raise ProposalError(f"{where}: {key} already has a human grade; AI proposals never replace it")

    def add(self, rows: Iterable[AIProposal | Mapping[str, Any]]) -> list[AIProposal]:
        """Validate and append proposals; nothing is written if any row is invalid."""
        batch: list[AIProposal] = []
        seen = dict(self._proposals)
        for index, row in enumerate(rows):
            try:
                proposal = row if isinstance(row, AIProposal) else AIProposal.model_validate(row)
            except ValidationError as exc:
                raise ProposalError(f"row {index}: {_explain(exc)}") from None
            self._check(proposal, where=f"row {index}", existing=seen)
            seen[proposal.query_id, proposal.passage_id] = proposal
            batch.append(proposal)
        with self.path.open("a", encoding="utf-8") as fh:
            for proposal in batch:
                fh.write(json.dumps(proposal.model_dump(mode="json"), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._proposals = seen
        return batch

    def proposals(self, query_id: str | None = None) -> list[AIProposal]:
        return [p for p in self._proposals.values() if query_id is None or p.query_id == query_id]

    def proposal(self, query_id: str, passage_id: str) -> AIProposal | None:
        return self._proposals.get((query_id, passage_id))

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def coverage(self) -> dict[str, Any]:
        """Per question: pool size, human-graded, proposed and unlabelled passages, grades, confidence."""
        rows = {}
        for question in self.store.questions():
            pool = self.store.pool(question.query_id)
            human = sum(
                1 for e in pool if (j := self.store.judgment(question.query_id, e.passage_id)) and j.grade is not None
            )
            proposed = self.proposals(question.query_id)
            grades = Counter(p.grade for p in proposed)
            confidence = Counter(p.confidence for p in proposed)
            rows[question.query_id] = {
                "pool": len(pool),
                "human_graded": human,
                "ai_proposed": len(proposed),
                "unlabelled": len(pool) - human - len(proposed),
                "grades": {str(g): grades.get(g, 0) for g in RELEVANCE_GRADES},
                "confidence": {c: confidence.get(c, 0) for c in CONFIDENCE_LEVELS},
            }
        return rows


__all__ = ["PROPOSALS_FILE", "PROPOSALS_INFO_FILE", "ProposalError", "ProposalSet"]
