"""Blind human audit of AI-proposed relevance labels.

``create_audit`` draws a reproducible sample from a proposal set, stratified
by question and proposed grade, and copies the questions and sampled passages
into a separate annotation workspace. That workspace and its ``audit.json``
contain no AI grades, explanations or confidence, so the auditor grades blind
(the ``audit`` command). ``audit_agreement`` compares human and AI grades only
after every sampled passage has a human grade or skip, and never changes
either side.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from finretriever.eval_set.ai_proposals import ProposalSet
from finretriever.eval_set.annotation_store import AnnotationStore, Judgment, utc_now
from finretriever.eval_set.schemas import (
    CONFIDENCE_LEVELS,
    QUESTION_TYPES,
    RELEVANCE_GRADES,
    AIProposal,
    AuditInfo,
    AuditItem,
    Candidate,
    CandidateSource,
    GoldQuestion,
)
from finretriever.evaluation.provenance import CorpusIdentity

AUDIT_FILE = "audit.json"
REPORT_FILE = "agreement_report.json"
SELECTION = (
    "Equal share of the sample per question (remainder to the questions with the most proposals); "
    "within a question, slots go round-robin over proposed grades 2, 1, 0 so every proposed class "
    "present is represented; passages within a stratum are drawn with random.Random(f'{seed}:{query_id}:"
    "{grade}'); display order is shuffled with random.Random(f'{seed}:order')."
)


class AuditError(ValueError):
    """An audit cannot be created, opened or scored."""


def select_audit_sample(proposals: Sequence[AIProposal], *, size: int, seed: int | str) -> list[AIProposal]:
    """A reproducible sample of ``size`` proposals, stratified by question and proposed grade.

    Returned in display order. See ``SELECTION`` for the procedure.
    """
    strata: dict[str, dict[int, list[AIProposal]]] = defaultdict(lambda: defaultdict(list))
    for proposal in proposals:
        strata[proposal.query_id][proposal.grade].append(proposal)
    if not strata:
        raise AuditError("no proposals to sample")
    if size > len(proposals):
        raise AuditError(f"sample size {size} exceeds the {len(proposals)} proposals")
    questions = sorted(strata)
    quota = dict.fromkeys(questions, size // len(questions))
    by_size = sorted(questions, key=lambda q: (-sum(map(len, strata[q].values())), q))
    for query_id in by_size[: size % len(questions)]:
        quota[query_id] += 1

    chosen: list[AIProposal] = []
    for query_id in questions:
        grades = sorted(strata[query_id], reverse=True)
        counts = dict.fromkeys(grades, 0)
        remaining = quota[query_id]
        while remaining and any(counts[g] < len(strata[query_id][g]) for g in grades):
            for grade in grades:
                if remaining and counts[grade] < len(strata[query_id][grade]):
                    counts[grade] += 1
                    remaining -= 1
        if remaining:
            raise AuditError(f"{query_id} has fewer proposals than its share of the sample")
        for grade in grades:
            pool = sorted(strata[query_id][grade], key=lambda p: p.passage_id)
            chosen += random.Random(f"{seed}:{query_id}:{grade}").sample(pool, counts[grade])
    random.Random(f"{seed}:order").shuffle(chosen)
    return chosen


class AuditWorkspace:
    """An audit's annotation workspace plus its ``audit.json``."""

    def __init__(self, directory: Path, info: AuditInfo, store: AnnotationStore) -> None:
        self.directory = directory
        self.info = info
        self.store = store

    @classmethod
    def open(cls, directory: str | Path, corpus: CorpusIdentity) -> AuditWorkspace:
        directory = Path(directory)
        info_path = directory / AUDIT_FILE
        if not info_path.is_file():
            raise AuditError(f"no audit at {directory} (run audit-init first)")
        try:
            info = AuditInfo.model_validate_json(info_path.read_bytes())
        except ValidationError as exc:
            raise AuditError(f"{info_path}: {exc}") from None
        store = AnnotationStore.open(directory, corpus)
        listed = {(i.query_id, i.passage_id) for i in info.items}
        pooled = {(q.query_id, e.passage_id) for q in store.questions() for e in store.pool(q.query_id)}
        if listed != pooled or len(listed) != len(info.items):
            raise AuditError(f"{info_path} items do not match the audit workspace's pools")
        return cls(directory, info, store)

    def items(self) -> list[AuditItem]:
        return list(self.info.items)

    def judgment(self, item: AuditItem) -> Judgment | None:
        return self.store.judgment(item.query_id, item.passage_id)

    def pending(self) -> list[AuditItem]:
        return [item for item in self.info.items if self.judgment(item) is None]

    def progress(self) -> dict[str, int]:
        judgments = [self.judgment(item) for item in self.info.items]
        skipped = sum(1 for j in judgments if j is not None and j.grade is None)
        graded = sum(1 for j in judgments if j is not None and j.grade is not None)
        return {"items": len(judgments), "graded": graded, "skipped": skipped, "pending": len(judgments) - graded - skipped}


def create_audit(
    directory: str | Path,
    store: AnnotationStore,
    proposals: ProposalSet,
    *,
    size: int,
    seed: int,
    created_by: str,
    annotation_version: str,
    clock: Callable[[], str] = utc_now,
) -> AuditWorkspace:
    """Draw the sample and set up a blind audit workspace; refuses to overwrite one."""
    directory = Path(directory)
    if directory.resolve() in (store.directory.resolve(), proposals.directory.resolve()):
        raise AuditError("the audit needs its own directory, apart from the workspace and the proposals")
    if (directory / AUDIT_FILE).exists():
        raise AuditError(f"{directory} already holds an audit")
    sample = select_audit_sample(proposals.proposals(), size=size, seed=seed)
    return write_audit_workspace(
        directory,
        store.corpus,
        store.questions(),
        [(p.query_id, p.passage_id) for p in sample],
        source_workspace=str(store.directory),
        proposals_dir=str(proposals.directory),
        proposals_sha256=proposals.sha256(),
        seed=seed,
        selection=SELECTION,
        created_by=created_by,
        annotation_version=annotation_version,
        clock=clock,
    )


def write_audit_workspace(
    directory: str | Path,
    corpus: CorpusIdentity,
    questions: Sequence[GoldQuestion],
    sample: Sequence[tuple[str, str]],
    *,
    source_workspace: str,
    proposals_dir: str,
    proposals_sha256: str,
    seed: int,
    selection: str,
    created_by: str,
    annotation_version: str,
    clock: Callable[[], str] = utc_now,
) -> AuditWorkspace:
    """Write a blind audit of ``sample`` ((question, passage) pairs, in display order).

    The workspace holds the questions and the sampled passages under a neutral source, and
    ``audit.json`` records where the sample came from; neither holds an AI grade, confidence,
    explanation or the reason a passage was sampled. Refuses a directory that holds an audit.
    """
    directory = Path(directory)
    if (directory / AUDIT_FILE).exists():
        raise AuditError(f"{directory} already holds an audit")
    if len(set(sample)) != len(sample) or not sample:
        raise AuditError("an audit sample needs distinct (question, passage) pairs")
    if unknown := sorted({qid for qid, _ in sample} - {q.query_id for q in questions}):
        raise AuditError(f"audit sample names unknown questions {unknown}")
    audit_store = AnnotationStore.create(
        directory, corpus, annotation_version=annotation_version, created_by=created_by, clock=clock
    )
    for question in questions:
        copy = audit_store.add_question(
            question.text,
            topic=question.topic,
            question_type=question.question_type,
            comparison=question.comparison,
            tickers=question.tickers,
            fiscal_years=question.fiscal_years,
            authoring=question.authoring,
            author=question.author,
            notes=question.notes,
            ai_assisted=question.ai_assisted,
            reviewed_by=question.reviewed_by,
        )
        if copy.query_id != question.query_id:
            raise AuditError(f"audit copy of {question.query_id} got ID {copy.query_id}")
    source = CandidateSource(method="stratified", detail=f"blind audit sample (seed={seed})")
    for question in questions:
        sampled = [Candidate(passage_id=pid, source=source) for qid, pid in sample if qid == question.query_id]
        if sampled:
            audit_store.add_candidates(question.query_id, sampled, added_by=created_by)
    info = AuditInfo(
        source_workspace=source_workspace,
        proposals_dir=proposals_dir,
        proposals_sha256=proposals_sha256,
        seed=seed,
        sample_size=len(sample),
        selection=selection,
        items=[AuditItem(item=i, query_id=qid, passage_id=pid) for i, (qid, pid) in enumerate(sample, 1)],
        created_at=clock(),
        created_by=created_by,
    )
    (directory / AUDIT_FILE).write_text(
        json.dumps(info.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return AuditWorkspace(directory, info, audit_store)


# --- agreement ------------------------------------------------------------------------


@dataclass(frozen=True)
class AgreementReport:
    """Human audit grades against AI proposals for the same passages.

    ``confusion[human][ai]`` counts graded items. ``weighted_raw_agreement``
    re-weights each (question, proposed grade) stratum by its share of all
    proposals, estimating agreement over the whole proposal set rather than
    over the deliberately balanced sample.
    """

    items: int
    graded: int
    skipped: int
    raw_agreement: float | None
    binary_agreement: float | None  # relevant (grade >= 1) vs not relevant
    cohen_kappa: float | None
    weighted_raw_agreement: float | None
    confusion: dict[str, dict[str, int]]
    by_question_type: dict[str, dict[str, int]]
    by_human_grade: dict[str, dict[str, int]]
    by_ai_grade: dict[str, dict[str, int]]
    by_ai_confidence: dict[str, dict[str, int]]
    by_question: dict[str, dict[str, int]]
    disagreements: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cohen_kappa(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Unweighted Cohen's kappa for (human, ai) grade pairs; ``None`` if undefined."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(h == a for h, a in pairs) / n
    human, ai = Counter(h for h, _ in pairs), Counter(a for _, a in pairs)
    expected = sum(human[g] * ai[g] for g in RELEVANCE_GRADES) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _tally(rows: list[tuple[str, bool]], keys: Sequence[str]) -> dict[str, dict[str, int]]:
    out = {key: {"items": 0, "agree": 0, "disagree": 0} for key in keys}
    for key, agree in rows:
        cell = out.setdefault(key, {"items": 0, "agree": 0, "disagree": 0})
        cell["items"] += 1
        cell["agree" if agree else "disagree"] += 1
    return out


def audit_agreement(audit: AuditWorkspace, proposals: ProposalSet) -> AgreementReport:
    """Compare the finished audit with the proposals it was drawn from.

    Refuses while any item is pending (AI labels stay hidden until then) or
    if the proposals changed after the sample was drawn. Skipped items are
    counted but excluded from the statistics.
    """
    if proposals.sha256() != audit.info.proposals_sha256:
        raise AuditError("the proposals changed after the audit sample was drawn")
    if pending := audit.pending():
        raise AuditError(
            f"{len(pending)} of {len(audit.items())} audit items still need a human grade or skip; "
            "AI labels stay hidden until the audit is complete"
        )
    pairs: list[tuple[AuditItem, int, AIProposal]] = []
    skipped = 0
    for item in audit.items():
        judgment = audit.judgment(item)
        proposal = proposals.proposal(item.query_id, item.passage_id)
        if proposal is None:
            raise AuditError(f"no proposal for audit item {item.item}")
        if judgment is None or judgment.grade is None:
            skipped += 1
            continue
        pairs.append((item, judgment.grade, proposal))

    grades = [str(g) for g in RELEVANCE_GRADES]
    confusion = {h: dict.fromkeys(grades, 0) for h in grades}
    for _, human, proposal in pairs:
        confusion[str(human)][str(proposal.grade)] += 1
    n = len(pairs)
    agree = [human == p.grade for _, human, p in pairs]

    population = Counter((p.query_id, p.grade) for p in proposals.proposals())
    sampled = Counter((p.query_id, p.grade) for _, _, p in pairs)
    agreed = Counter((p.query_id, p.grade) for _, h, p in pairs if h == p.grade)
    covered = sum(population[s] for s in sampled)
    weighted = sum(population[s] * agreed[s] / sampled[s] for s in sampled) / covered if covered else None

    questions = {q.query_id: q for q in audit.store.questions()}
    disagreements = [
        {
            "item": item.item,
            "query_id": item.query_id,
            "question_type": questions[item.query_id].question_type,
            "passage_id": item.passage_id,
            "human_grade": human,
            "human_note": (audit.judgment(item).note if audit.judgment(item) else ""),
            "ai_grade": proposal.grade,
            "ai_confidence": proposal.confidence,
            "ai_explanation": proposal.explanation,
        }
        for item, human, proposal in pairs
        if human != proposal.grade
    ]
    return AgreementReport(
        items=len(audit.items()),
        graded=n,
        skipped=skipped,
        raw_agreement=sum(agree) / n if n else None,
        binary_agreement=sum((h > 0) == (p.grade > 0) for _, h, p in pairs) / n if n else None,
        cohen_kappa=cohen_kappa([(h, p.grade) for _, h, p in pairs]),
        weighted_raw_agreement=weighted,
        confusion=confusion,
        by_question_type=_tally(
            [(questions[i.query_id].question_type, a) for (i, _, _), a in zip(pairs, agree, strict=True)],
            QUESTION_TYPES,
        ),
        by_human_grade=_tally([(str(h), a) for (_, h, _), a in zip(pairs, agree, strict=True)], grades),
        by_ai_grade=_tally([(str(p.grade), a) for (_, _, p), a in zip(pairs, agree, strict=True)], grades),
        by_ai_confidence=_tally([(p.confidence, a) for (_, _, p), a in zip(pairs, agree, strict=True)], CONFIDENCE_LEVELS),
        by_question=_tally([(i.query_id, a) for (i, _, _), a in zip(pairs, agree, strict=True)], sorted(questions)),
        disagreements=disagreements,
    )


__all__ = [
    "AUDIT_FILE",
    "REPORT_FILE",
    "SELECTION",
    "AgreementReport",
    "AuditError",
    "AuditWorkspace",
    "audit_agreement",
    "cohen_kappa",
    "create_audit",
    "select_audit_sample",
    "write_audit_workspace",
]
