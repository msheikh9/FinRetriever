"""Blind human re-grading for inter-annotator agreement (gold set v1; docs/gold_set_v1_protocol.md §2.4).

**The workspace.** ``create_regrade`` copies the annotation workspace's questions into a separate workspace, which
reuses the blind-audit layout (``audit.json``). It pools there every passage of a seeded sample of complete,
human-written questions. It copies no grade, no note, no pool source and no rank.

**Grading.** The second person grades with the existing ``audit`` command: the question, the filing and the passage
only, in an order shuffled with the seed.

**The report.** ``regrade_agreement`` compares the two people's grades. It refuses in four cases:
- the re-grade is incomplete;
- one person graded both sides;
- the first annotator's grades changed after the sample was drawn (``history.jsonl`` is pinned by SHA-256 in
  ``audit.json``);
- a report is already present: it is written once.

It never changes either workspace's grades. The first annotator's grades stay the gold set's, unless a rubric error
is found and corrected in the workspace with a note.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from finretriever.eval_set.annotation_store import HISTORY_FILE, AnnotationStore, utc_now
from finretriever.eval_set.audit import (
    AUDIT_FILE,
    AuditError,
    AuditWorkspace,
    cohen_kappa,
    write_audit_workspace,
)

REGRADE_MARKER = "human re-grade (no AI proposals)"
REGRADE_REPORT_FILE = "regrade_report.json"
GRADES = (0, 1, 2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_regrade_questions(store: AnnotationStore, *, count: int, seed: int) -> list[str]:
    """``count`` complete, human-written questions, drawn with ``random.Random(f"{seed}:questions")``."""
    eligible = sorted(q.query_id for q in store.questions() if q.status == "complete" and not q.ai_assisted)
    if count < 1 or count > len(eligible):
        raise AuditError(f"cannot draw {count} questions from {len(eligible)} complete human-written questions")
    return sorted(random.Random(f"{seed}:questions").sample(eligible, count))


def create_regrade(directory: str | Path, store: AnnotationStore, *, count: int, seed: int, created_by: str,
                   clock: Callable[[], str] = utc_now) -> AuditWorkspace:
    """Set up the blind re-grade workspace; refuses the annotation workspace's own directory or an existing one."""
    directory = Path(directory)
    if directory.resolve() == store.directory.resolve():
        raise AuditError("the re-grade needs its own directory, apart from the annotation workspace")
    if (directory / AUDIT_FILE).exists():
        raise AuditError(f"{directory} already holds an audit or a re-grade")
    chosen = select_regrade_questions(store, count=count, seed=seed)
    sample = [(qid, entry.passage_id) for qid in chosen for entry in store.pool(qid)]
    random.Random(f"{seed}:order").shuffle(sample)
    return write_audit_workspace(
        directory,
        store.corpus,
        store.questions(),
        sample,
        source_workspace=str(store.directory),
        proposals_dir=REGRADE_MARKER,
        proposals_sha256=_sha256(store.directory / HISTORY_FILE),
        seed=seed,
        selection=(f"{REGRADE_MARKER}: every pooled passage of {count} complete human-written questions drawn with "
                   f"random.Random('{seed}:questions'), shown in the order of random.Random('{seed}:order')"),
        created_by=created_by,
        annotation_version=f"{store.info.evaluation.annotation_version}-regrade",
        clock=clock,
    )


def weighted_kappa(pairs: Sequence[tuple[int, int]], grades: Sequence[int] = GRADES) -> float | None:
    """Linear-weighted Cohen's kappa for ordinal grades; ``None`` if undefined."""
    if not pairs:
        return None
    n, k = len(pairs), len(grades)
    index = {g: i for i, g in enumerate(grades)}

    def weight(a: int, b: int) -> float:
        return 1 - abs(index[a] - index[b]) / (k - 1)

    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = sum(weight(a, b) for a, b in pairs) / n
    expected = sum(weight(a, b) * first[a] * second[b] for a in grades for b in grades) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def regrade_agreement(regrade: AuditWorkspace, source: AnnotationStore) -> dict[str, Any]:
    """Agreement between the first annotator's grades and the blind re-grade (see the module docstring)."""
    if regrade.info.proposals_dir != REGRADE_MARKER:
        raise AuditError(f"{regrade.directory} is an AI-label audit, not a human re-grade")
    if _sha256(source.directory / HISTORY_FILE) != regrade.info.proposals_sha256:
        raise AuditError("the first annotator's grades changed after the re-grade sample was drawn")
    if pending := regrade.pending():
        raise AuditError(f"the re-grade is incomplete: {len(pending)} passage(s) pending")
    pairs: list[tuple[int, int]] = []
    disagreements: list[dict[str, Any]] = []
    skipped = {"first": 0, "second": 0}
    per_question: dict[str, dict[str, int]] = {}
    for item in regrade.items():
        first = source.judgment(item.query_id, item.passage_id)
        second = regrade.judgment(item)
        if first is None:
            raise AuditError(f"{item.query_id} {item.passage_id} has no first grade")
        if second is not None and first.annotator.casefold() == second.annotator.casefold():
            raise AuditError(f"{first.annotator} graded both sides of {item.query_id}; the re-grade needs another person")
        cell = per_question.setdefault(item.query_id, {"compared": 0, "agree": 0})
        if first.grade is None or second is None or second.grade is None:
            skipped["first" if first.grade is None else "second"] += 1
            continue
        pairs.append((first.grade, second.grade))
        cell["compared"] += 1
        cell["agree"] += first.grade == second.grade
        if first.grade != second.grade:
            disagreements.append({"query_id": item.query_id, "passage_id": item.passage_id, "first": first.grade,
                                  "second": second.grade, "first_note": first.note, "second_note": second.note})
    n = len(pairs)
    confusion = {str(a): {str(b): sum(1 for x, y in pairs if (x, y) == (a, b)) for b in GRADES} for a in GRADES}
    return {
        "kind": "finretriever_gold_regrade_agreement",
        "regrade": str(regrade.directory),
        "source_workspace": regrade.info.source_workspace,
        "source_history_sha256": regrade.info.proposals_sha256,
        "questions": len(per_question),
        "items": len(regrade.items()),
        "compared": n,
        "skipped": skipped,
        "raw_agreement": None if not n else sum(a == b for a, b in pairs) / n,
        "relevant_vs_not_agreement": None if not n else sum((a > 0) == (b > 0) for a, b in pairs) / n,
        "cohen_kappa": cohen_kappa(pairs),
        "linear_weighted_kappa": weighted_kappa(pairs),
        "confusion": confusion,
        "confusion_axes": "rows: the first annotator's grade; columns: the re-grade",
        "per_question": per_question,
        "disagreements": disagreements,
        "notice": "Agreement is reported, never deciding; the first annotator's grades remain the gold set's. Each "
                  "disagreement is reviewed; a grade changes only for a rubric error, in the workspace, with a note.",
    }


def write_regrade_report(regrade: AuditWorkspace, report: dict[str, Any]) -> Path:
    """Write the agreement report once."""
    path = regrade.directory / REGRADE_REPORT_FILE
    if path.exists():
        raise AuditError(f"{path} already exists; the agreement is reported once")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "REGRADE_MARKER",
    "REGRADE_REPORT_FILE",
    "create_regrade",
    "regrade_agreement",
    "select_regrade_questions",
    "weighted_kappa",
    "write_regrade_report",
]
