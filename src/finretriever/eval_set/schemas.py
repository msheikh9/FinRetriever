"""Records of the hand-annotated gold evaluation set.

An annotation workspace (``data/eval/annotation/`` by default) holds:

* ``workspace.json`` — ``WorkspaceInfo``: the corpus the annotations label
  (``EvaluationMetadata``: corpus SHA-256, corpus version, annotation version).
* ``questions.jsonl`` — ``GoldQuestion`` records, written by people.
* ``pool.jsonl`` — ``PoolEntry`` records: the candidate passages shown for each
  question, and how each one was selected.
* ``history.jsonl`` — the append-only ``AnnotationEvent`` log. Judgments live
  nowhere else: the current state is the log replayed, so every grade keeps
  its revision history.

A passage with no judgment is *unjudged*, never "not relevant": grade 0 is a
decision someone made. Skipping a passage records that it was looked at but
left unjudged.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from finretriever.evaluation.provenance import EvaluationMetadata

RELEVANCE_GRADES: dict[int, str] = {
    0: "not relevant",
    1: "partially relevant",
    2: "directly relevant",
}

Topic = Literal[
    "risk_factors",
    "mdna",
    "financial_statements",
    "debt_liquidity",
    "revenue_profitability",
    "other",
]
QuestionType = Literal["exact_fact", "semantic", "table", "multi_passage"]
Comparison = Literal["none", "cross_year", "cross_company"]
# "independent": written from an information need before looking at any passage.
# "passage_inspired": written after reading a passage — allowed, but tracked,
# because such questions tend to reuse the passage's wording.
Authoring = Literal["independent", "passage_inspired"]
QuestionStatus = Literal["open", "complete", "withdrawn"]
# Pool sources are deliberately limited: dense retrievers (and above all the
# model being evaluated) never select candidates. See docs/eval_methodology.md.
CandidateMethod = Literal["bm25", "manual", "random", "stratified"]
EventKind = Literal["question_added", "question_status", "candidates_added", "grade", "skip", "clear"]

TOPICS: tuple[str, ...] = Topic.__args__  # type: ignore[attr-defined]
QUESTION_TYPES: tuple[str, ...] = QuestionType.__args__  # type: ignore[attr-defined]
COMPARISONS: tuple[str, ...] = Comparison.__args__  # type: ignore[attr-defined]
AUTHORING: tuple[str, ...] = Authoring.__args__  # type: ignore[attr-defined]
CANDIDATE_METHODS: tuple[str, ...] = CandidateMethod.__args__  # type: ignore[attr-defined]

QUERY_ID_RE = re.compile(r"^q_\d{4,}$")
SKIPPED = "skipped"

_STRICT = ConfigDict(extra="forbid")


def _non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("must be non-empty")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]


def check_grade(value: object) -> int:
    """Return ``value`` if it is a relevance grade (0, 1 or 2), else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in RELEVANCE_GRADES:
        raise ValueError(f"relevance grade must be one of {sorted(RELEVANCE_GRADES)}, got {value!r}")
    return value


def normalize_question(text: str) -> str:
    """Comparison key for question texts: case, spacing and end punctuation ignored."""
    return re.sub(r"\s+", " ", text.casefold()).strip().rstrip("?.! ")


class WorkspaceInfo(BaseModel):
    """``workspace.json``: binds an annotation workspace to one corpus."""

    model_config = _STRICT

    format_version: Literal[1] = 1
    evaluation: EvaluationMetadata
    created_at: str
    created_by: NonEmptyStr


class GoldQuestion(BaseModel):
    """An evaluation question, written by a person or AI-drafted and approved by one.

    An AI-drafted question records the model as ``author``, sets
    ``ai_assisted`` and names the person who approved it in ``reviewed_by``;
    it can never be recorded as ``independent``.
    """

    model_config = _STRICT

    query_id: str
    text: NonEmptyStr
    topic: Topic
    question_type: QuestionType
    comparison: Comparison = "none"
    tickers: list[str] = Field(default_factory=list)  # companies asked about; [] = any
    fiscal_years: list[int] = Field(default_factory=list)
    authoring: Authoring
    author: NonEmptyStr
    ai_assisted: bool = False
    reviewed_by: str = ""
    created_at: str
    status: QuestionStatus = "open"
    notes: str = ""

    @field_validator("query_id")
    @classmethod
    def _query_id(cls, v: str) -> str:
        if not QUERY_ID_RE.match(v):
            raise ValueError(f"query_id must look like q_0001, got {v!r}")
        return v

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, v: list[str]) -> list[str]:
        return sorted({t.strip().upper() for t in v if t.strip()})

    @field_validator("fiscal_years")
    @classmethod
    def _years(cls, v: list[int]) -> list[int]:
        if any(not 1990 <= year <= 2100 for year in v):
            raise ValueError(f"fiscal years must be four-digit years, got {v}")
        return sorted(set(v))

    @model_validator(mode="after")
    def _ai_provenance(self) -> GoldQuestion:
        if self.ai_assisted:
            if self.authoring == "independent":
                raise ValueError("AI-assisted questions cannot be recorded as independently authored")
            if not self.reviewed_by.strip():
                raise ValueError("AI-assisted questions need reviewed_by: the person who approved them")
        return self


class CandidateSource(BaseModel):
    """How one candidate was selected: method, rank within it, and its inputs."""

    model_config = _STRICT

    method: CandidateMethod
    rank: int | None = Field(default=None, ge=1)
    score: float | None = None
    detail: str = ""  # e.g. the BM25 search text and filters, or the sampling seed


class Candidate(BaseModel):
    """A passage proposed for judgment, before it joins a question's pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str
    source: CandidateSource


class PoolEntry(BaseModel):
    """A candidate passage shown for one question, with every way it was selected."""

    model_config = _STRICT

    query_id: str
    passage_id: str
    position: int = Field(ge=1)  # order in which the annotator sees it
    sources: list[CandidateSource] = Field(min_length=1)
    added_at: str
    added_by: NonEmptyStr


class AnnotationEvent(BaseModel):
    """One line of ``history.jsonl``.

    ``previous`` is the judgment state the event replaced (``"0"``, ``"1"``,
    ``"2"``, ``"skipped"`` or ``None`` for unjudged; the old status for
    ``question_status``). Replaying the log checks it, so out-of-order or
    conflicting edits are caught instead of silently winning.
    """

    model_config = _STRICT

    seq: int = Field(ge=1)
    at: str
    annotator: NonEmptyStr
    kind: EventKind
    query_id: str
    passage_id: str | None = None
    grade: int | None = None
    status: QuestionStatus | None = None
    previous: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _fields_for_kind(self) -> AnnotationEvent:
        if self.kind in ("grade", "skip", "clear") and self.passage_id is None:
            raise ValueError(f"{self.kind} events need a passage_id")
        if self.kind == "grade":
            if self.grade is None:
                raise ValueError("grade events need a grade")
            check_grade(self.grade)
        elif self.grade is not None:
            raise ValueError(f"{self.kind} events carry no grade")
        if (self.kind == "question_status") != (self.status is not None):
            raise ValueError("status is required for question_status events and only for them")
        return self


Confidence = Literal["low", "medium", "high"]
CONFIDENCE_LEVELS: tuple[str, ...] = Confidence.__args__  # type: ignore[attr-defined]
AI_NOTICE = (
    "AI-generated provisional relevance labels for development use only. They are not human "
    "judgments and must never be exported or reported as gold qrels."
)


class AIProposal(BaseModel):
    """An AI-proposed grade for one pooled passage: development data, never gold."""

    model_config = _STRICT

    query_id: str
    passage_id: str
    grade: int
    confidence: Confidence
    explanation: NonEmptyStr = Field(max_length=600)
    model: NonEmptyStr  # the model that read the passage, e.g. "claude-opus-5-5"
    method: NonEmptyStr  # how it was asked to judge (see ProposalSetInfo.method_description)
    created_at: str

    @field_validator("grade")
    @classmethod
    def _grade(cls, v: int) -> int:
        return check_grade(v)


class ProposalSetInfo(BaseModel):
    """``proposals.meta.json``: what a set of AI proposals labels and how it was made."""

    model_config = _STRICT

    format_version: Literal[1] = 1
    label_source: Literal["ai_proposal"] = "ai_proposal"
    evaluation: EvaluationMetadata  # corpus and annotation version of the labelled workspace
    source_workspace: str
    model: NonEmptyStr
    method: NonEmptyStr
    method_description: NonEmptyStr
    created_at: str
    created_by: NonEmptyStr
    notice: str = AI_NOTICE


class AuditItem(BaseModel):
    """One passage of a blind audit, in the order the auditor sees it."""

    model_config = _STRICT

    item: int = Field(ge=1)
    query_id: str
    passage_id: str


class AuditInfo(BaseModel):
    """``audit.json``: a blind human audit drawn from a proposal set.

    It holds no AI grades: the auditor's workspace never contains them. The
    sample can be redrawn from the proposals with ``seed`` and
    ``sample_size``, and ``proposals_sha256`` pins the proposals it was drawn
    from.
    """

    model_config = _STRICT

    format_version: Literal[1] = 1
    kind: Literal["blind_human_audit"] = "blind_human_audit"
    source_workspace: str
    proposals_dir: str
    proposals_sha256: str
    seed: int
    sample_size: int = Field(ge=1)
    selection: str
    items: list[AuditItem] = Field(min_length=1)
    created_at: str
    created_by: NonEmptyStr


class ApprovedPoolSnapshot(BaseModel):
    """A pool snapshot version and its SHA-256, as approved by the reviewer before grading."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(f"not a lowercase hex SHA-256: {v!r}")
        return v


class GoldSetConfig(BaseModel):
    """``configs/eval/gold_set_v1.yaml``: the frozen corpus, pooling and audit defaults."""

    model_config = _STRICT

    name: str
    annotation_version: str
    corpus_path: str
    corpus_version: str
    corpus_sha256: str
    workspace_dir: str
    export_dir: str
    target_queries: int = Field(ge=1)
    bm25_depth: int = Field(default=30, ge=1)
    stratified_per_filing: int = Field(default=0, ge=0)
    random_candidates: int = Field(default=0, ge=0)
    seed: int = 0
    small_pool_below: int = Field(default=10, ge=1)
    ai_proposals_dir: str = "data/eval/dev_ai/pilot_v1"
    audit_dir: str = "data/eval/audit/pilot_v1"
    audit_sample_size: int = Field(default=30, ge=1)
    audit_seed: int = 0
    # Identities that are AI systems: their judgments can never be exported as gold.
    ai_annotators: list[str] = Field(default_factory=lambda: ["claude"])
    # Gold set v1's question stage (eval_set.question_stage): the query plan, where the worksheet, the leakage record
    # and the frozen question export live (pooling a slot question needs that export), and blind grading enforced.
    query_plan: str | None = None
    question_stage_dir: str | None = None
    require_blind: bool = False
    # Slots kept provisionally, with the condition to check during pooling (printed by `pool`).
    provisional_slots: dict[str, str] = Field(default_factory=dict)
    # The reviewer-approved pool snapshot. Once set, pooling the approved questions is closed, and grading and the
    # re-grade start only if the workspace pools still equal this snapshot (eval_set.rewording.verify_approved_pool).
    approved_pool_snapshot: ApprovedPoolSnapshot | None = None
    # The registered primary annotator (a person's initials). Once set, only they grade the approved questions.
    primary_annotator: str | None = None
    # Or several primary annotators, each grading the approved questions assigned to them (eval_set.text_assignment).
    primary_annotators: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_registration(self) -> GoldSetConfig:
        if self.primary_annotator and self.primary_annotators:
            raise ValueError("set primary_annotator or primary_annotators, not both")
        names = [n.casefold() for n in self.primary_annotators]
        if len(names) != len(set(names)) or any(not n.strip() for n in names):
            raise ValueError("primary_annotators must be distinct, non-empty initials")
        if ai := [n for n in self.registered_annotators if n.casefold() in {x.casefold() for x in self.ai_annotators}]:
            raise ValueError(f"an AI identity cannot be a primary annotator: {ai}")
        return self

    @property
    def registered_annotators(self) -> list[str]:
        """The registered primary annotators, from either field."""
        return list(self.primary_annotators) or ([self.primary_annotator] if self.primary_annotator else [])

    @field_validator("corpus_sha256")
    @classmethod
    def _sha256(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(f"not a lowercase hex SHA-256: {v!r}")
        return v


__all__ = [
    "AI_NOTICE",
    "AIProposal",
    "ApprovedPoolSnapshot",
    "AUTHORING",
    "AnnotationEvent",
    "AuditInfo",
    "AuditItem",
    "CONFIDENCE_LEVELS",
    "CANDIDATE_METHODS",
    "COMPARISONS",
    "Candidate",
    "CandidateSource",
    "GoldQuestion",
    "GoldSetConfig",
    "PoolEntry",
    "ProposalSetInfo",
    "QUERY_ID_RE",
    "QUESTION_TYPES",
    "RELEVANCE_GRADES",
    "SKIPPED",
    "TOPICS",
    "WorkspaceInfo",
    "check_grade",
    "normalize_question",
]
