"""Blind priority audit of the AI proposals for an expanded pool's new entries.

The ``pilot_v1`` audit drew a balanced, stratified sample to estimate overall agreement. A
priority audit is purposive instead: it puts in front of a person the labels a comparison
depends on most. Every eligible proposal (by default only passages from the companies the
question asks about) gets a tier and review features:

* tier 1: proposed relevant (grade >= 1) or proposed with low confidence; tier 2: the rest;
* ``table``: the passage holds a financial table;
* ``other_year``: the filing's report year is not one the question asks about (temporal
  rules decide such passages: comparatives and repeated text);
* ``rank_disagreement``: a dense run ranked it in its top 10 while the lexical run ranked it
  below ``lexical_disagreement_rank`` (by default: not in its stored top 100 at all).

``select_priority_sample`` fills the sample one slot at a time with the candidate that
minimises (tier, passages already chosen for its question, confidence (low first), -number
of features, seeded random), so tier 1 comes first, questions stay balanced, and less
confident and feature-rich cases win ties.
The display order is shuffled with a seed. The audit workspace (``write_audit_workspace``)
holds no AI grade, confidence, explanation, tier or feature, so the auditor grades blind;
the features can be recomputed from the inputs, which ``audit.json`` pins by SHA-256.
Agreement from a purposive sample describes these cases, not the whole proposal set.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finretriever.data.schemas import Passage
from finretriever.eval_set.annotation_store import utc_now
from finretriever.eval_set.audit import AuditError, AuditWorkspace, write_audit_workspace
from finretriever.eval_set.dev_labels import NonEmptyStr
from finretriever.eval_set.dev_pool import DevPool, PoolProposalSet, RunRef
from finretriever.eval_set.schemas import GoldQuestion
from finretriever.evaluation.provenance import CorpusIdentity, sha256_file
from finretriever.evaluation.stored_runs import StoredRun
from finretriever.utils.io import read_yaml

FEATURES = ("table", "other_year", "rank_disagreement")
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
PRIORITY_SELECTION = (
    "Purposive priority sample of new pool entries{scope}. Tier 1: proposed relevant (grade >= 1) or "
    "low confidence; tier 2: the rest. Review features: financial table, report year not asked about, "
    "dense top 10 but lexical run {lexical} below rank {depth} or absent. Slots are filled one at a "
    "time with the candidate minimising (tier, passages already chosen for its question, confidence "
    "(low, medium, high), -features, random.Random(f'{{seed}}:{{query_id}}:{{passage_id}}')); display "
    "order random.Random(f'{{seed}}:order'). Agreement describes these cases, not the whole proposal set."
)

_STRICT = ConfigDict(extra="forbid")


class PoolAuditSpec(BaseModel):
    """How a priority audit of a pool's proposals is drawn (``configs/eval/audit_<version>.yaml``)."""

    model_config = _STRICT

    audit_version: NonEmptyStr
    labels_dir: NonEmptyStr  # the label set derived from the proposals: questions snapshot
    pool_dir: NonEmptyStr
    proposals_dir: NonEmptyStr
    output_dir: NonEmptyStr
    size: int = Field(ge=1)
    seed: int
    in_scope_only: bool = True
    lexical_run: RunRef
    lexical_disagreement_rank: int = Field(default=100, ge=1)


def load_pool_audit_spec(path: str | Path) -> tuple[PoolAuditSpec, str]:
    file = Path(path)
    if not file.is_file():
        raise AuditError(f"audit spec not found: {file}")
    try:
        return PoolAuditSpec.model_validate(read_yaml(file)), sha256_file(file)
    except ValidationError as exc:
        raise AuditError(f"{file}: {exc}") from None


@dataclass(frozen=True)
class AuditCandidate:
    """One proposal that may be audited, with the reasons it would be reviewed first."""

    query_id: str
    passage_id: str
    tier: int
    confidence: str
    features: tuple[str, ...]


def audit_candidates(
    pool: DevPool,
    proposals: PoolProposalSet,
    questions: Sequence[GoldQuestion],
    passages: Mapping[str, Passage],
    lexical: StoredRun,
    *,
    lexical_disagreement_rank: int,
    in_scope_only: bool = True,
) -> list[AuditCandidate]:
    """Every proposal for the pool's new entries, with its tier and review features."""
    years = {q.query_id: set(q.fiscal_years) for q in questions}
    lexical_rank = {(h.query_id, h.passage_id): h.rank for h in lexical.hits}
    candidates = []
    for proposal in sorted(proposals.proposals(), key=lambda p: (p.query_id, p.passage_id)):
        key = (proposal.query_id, proposal.passage_id)
        entry = pool.entry(*key)
        if entry is None:
            raise AuditError(f"proposal {key} is not in the pool")
        if in_scope_only and not entry.in_scope:
            continue
        passage = passages[proposal.passage_id]
        report_year = str(passage.metadata.get("report_date") or "")[:4]
        dense_top = any(s.run != lexical.run and s.rank <= 10 for s in entry.run_sources)
        rank = lexical_rank.get(key)
        features = {
            "table": "[TABLE]" in passage.text,
            "other_year": bool(report_year and years[proposal.query_id]) and int(report_year) not in years[proposal.query_id],
            "rank_disagreement": dense_top and (rank is None or rank > lexical_disagreement_rank),
        }
        tier = 1 if proposal.grade >= 1 or proposal.confidence == "low" else 2
        chosen = tuple(f for f in FEATURES if features[f])
        candidates.append(AuditCandidate(key[0], key[1], tier, proposal.confidence, chosen))
    return candidates


def select_priority_sample(candidates: Sequence[AuditCandidate], *, size: int, seed: int) -> list[AuditCandidate]:
    """``size`` candidates in display order; see the module docstring for the rule."""
    if size > len(candidates):
        raise AuditError(f"sample size {size} exceeds the {len(candidates)} eligible proposals")
    jitter = {(c.query_id, c.passage_id): random.Random(f"{seed}:{c.query_id}:{c.passage_id}").random() for c in candidates}
    remaining = list(candidates)
    chosen: list[AuditCandidate] = []
    per_question: Counter[str] = Counter()
    while len(chosen) < size:
        best = min(
            remaining,
            key=lambda c: (
                c.tier, per_question[c.query_id], CONFIDENCE_ORDER[c.confidence], -len(c.features),
                jitter[c.query_id, c.passage_id],
            ),
        )
        remaining.remove(best)
        chosen.append(best)
        per_question[best.query_id] += 1
    random.Random(f"{seed}:order").shuffle(chosen)
    return chosen


def create_pool_audit(
    spec: PoolAuditSpec,
    *,
    corpus: CorpusIdentity,
    questions: Sequence[GoldQuestion],
    pool: DevPool,
    proposals: PoolProposalSet,
    lexical: StoredRun,
    passages: Mapping[str, Passage],
    created_by: str,
    clock: Callable[[], str] = utc_now,
) -> AuditWorkspace:
    """Draw the priority sample and write its blind audit workspace; refuses to overwrite one."""
    target = Path(spec.output_dir).resolve()
    for name, guard in (("the pool", pool.directory), ("the proposals", proposals.directory), ("the labels", spec.labels_dir)):
        if target == Path(guard).resolve():
            raise AuditError(f"the audit needs its own directory, not {name} ({guard})")
    candidates = audit_candidates(
        pool, proposals, questions, passages, lexical,
        lexical_disagreement_rank=spec.lexical_disagreement_rank, in_scope_only=spec.in_scope_only,
    )
    sample = select_priority_sample(candidates, size=spec.size, seed=spec.seed)
    selection = PRIORITY_SELECTION.format(
        scope=" from the companies each question asks about" if spec.in_scope_only else "",
        lexical=lexical.run_id, depth=spec.lexical_disagreement_rank,
    )
    return write_audit_workspace(
        spec.output_dir,
        corpus,
        questions,
        [(c.query_id, c.passage_id) for c in sample],
        source_workspace=str(pool.directory),
        proposals_dir=str(proposals.directory),
        proposals_sha256=proposals.sha256(),
        seed=spec.seed,
        selection=selection,
        created_by=created_by,
        annotation_version=f"{pool.info.evaluation.annotation_version}-audit",
        clock=clock,
    )


__all__ = [
    "CONFIDENCE_ORDER",
    "FEATURES",
    "PRIORITY_SELECTION",
    "AuditCandidate",
    "PoolAuditSpec",
    "audit_candidates",
    "create_pool_audit",
    "load_pool_audit_spec",
    "select_priority_sample",
]
