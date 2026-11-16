"""Versioned development labels: AI proposals revised under an adopted rubric.

A development label set (``data/eval/dev_ai/<version>/``) is derived, never
edited by hand. It starts from an AI proposal set and the human judgments in
the annotation workspace, applies the deterministic rules of a spec
(``configs/eval/dev_labels_<version>.yaml``) and writes:

* ``labels.meta.json`` — ``DevLabelSetInfo``: the corpus binding, every input
  with its SHA-256, the rubric version and decisions, counts, how each audit
  disagreement was handled, and the SHA-256 of every file below;
* ``questions.jsonl`` — the questions as they were when the set was derived;
* ``labels.jsonl`` — one ``DevLabel`` per labelled pool entry: its grade, its
  source (AI proposal or human judgment) and whether it differs from its base;
* ``changes.jsonl`` — one ``GradeChange`` per changed grade: previous and new
  grade, rule, reason, the check that triggered it and the evidence;
* ``flags.jsonl`` — ``LabelFlag``s for labels a rule may change only with a
  reader's judgment. A flag never changes a grade.

Only deterministic rules change grades, and only AI proposals: a human grade
is never rewritten. Audit grades are evidence attached to labels, never labels
(an audit made with AI assistance cannot validate AI labels on its own). Pool
entries without a human grade or an AI proposal stay unlabelled. The same
inputs and spec always give byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
)

from finretriever.data.schemas import Passage
from finretriever.eval_set.ai_proposals import PROPOSALS_FILE, PROPOSALS_INFO_FILE, ProposalSet
from finretriever.eval_set.annotation_store import HISTORY_FILE, AnnotationStore
from finretriever.eval_set.audit import AUDIT_FILE, REPORT_FILE, AuditWorkspace
from finretriever.eval_set.schemas import GoldQuestion, check_grade
from finretriever.evaluation.provenance import (
    CorpusIdentity,
    EvaluationMetadata,
    check_annotations_match_corpus,
    sha256_file,
)
from finretriever.utils.io import read_yaml

LABELS_INFO_FILE = "labels.meta.json"
QUESTIONS_SNAPSHOT_FILE = "questions.jsonl"
LABELS_FILE = "labels.jsonl"
CHANGES_FILE = "changes.jsonl"
FLAGS_FILE = "flags.jsonl"
DATA_FILES = (QUESTIONS_SNAPSHOT_FILE, LABELS_FILE, CHANGES_FILE, FLAGS_FILE)

DEV_STATUS = "provisional_development"
DEV_NOTICE = (
    "Provisional development labels. Every label is AI-generated unless label_source is human_judgment; "
    "they are never gold qrels, never training data, and every metric computed with them is a "
    "provisional development metric."
)

RuleId = Literal["P1", "P2", "P3", "P4", "P5", "P6"]
# A flag cites a rubric rule, or ``low_confidence``: the labelling model said it was unsure.
FlagRule = Literal["P1", "P2", "P3", "P4", "P5", "P6", "low_confidence"]
LabelSource = Literal["ai_proposal", "human_judgment"]
# Assistance behind a label: AI labels are AI-generated; a human grade's assistance is known only if recorded.
Assistance = Literal["not_applicable", "not_recorded", "none", "ai_assisted"]
AuditIndependence = Literal["independent", "ai_assisted"]

_STRICT = ConfigDict(extra="forbid")
_VERSION_RE = re.compile(r"^[a-z0-9_]+$")


def _non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("must be non-empty")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
Grade = Annotated[int, AfterValidator(check_grade)]


class DevLabelError(ValueError):
    """A development label set or its spec breaks the derivation rules."""


def _explain(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors())


# --- spec -----------------------------------------------------------------------------


class RubricDecisions(BaseModel):
    """The rubric version the labels follow and the decision taken on each proposal."""

    model_config = _STRICT

    version: NonEmptyStr
    adopted: NonEmptyStr
    decisions: dict[RuleId, NonEmptyStr]


class BasisRule(BaseModel):
    """P2 check for one figure question: a grade-2 label must show the requested period or basis."""

    model_config = _STRICT

    query_id: NonEmptyStr
    requires_any: list[NonEmptyStr] = Field(min_length=1)  # regular expressions; one match is enough
    why: NonEmptyStr

    @field_validator("requires_any")
    @classmethod
    def _patterns(cls, v: list[str]) -> list[str]:
        for pattern in v:
            re.compile(pattern)
        return v


class SpecFlag(BaseModel):
    """A label a reader should review under ``rule``; it keeps its grade until then."""

    model_config = _STRICT

    query_id: NonEmptyStr
    passage_id: NonEmptyStr
    rule: RuleId
    candidate_grade: Grade | None = None
    reason: NonEmptyStr


class KeptDisagreement(BaseModel):
    """An audit item whose human grade differs from the label, kept under an adopted rule."""

    model_config = _STRICT

    item: int = Field(ge=1)
    rule: RuleId
    reason: NonEmptyStr


class DevLabelSpec(BaseModel):
    """How a development label version is derived (``configs/eval/dev_labels_<version>.yaml``)."""

    model_config = _STRICT

    label_version: NonEmptyStr
    annotation_version: NonEmptyStr
    description: NonEmptyStr
    base_proposals_dir: NonEmptyStr
    audit_dir: NonEmptyStr
    audit_independence: AuditIndependence
    audit_independence_note: NonEmptyStr
    output_dir: NonEmptyStr
    rubric: RubricDecisions
    p2_basis: list[BasisRule] = Field(default_factory=list)
    p1_open_questions: list[NonEmptyStr] = Field(default_factory=list)
    flags: list[SpecFlag] = Field(default_factory=list)
    kept_audit_disagreements: list[KeptDisagreement] = Field(default_factory=list)

    @field_validator("label_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not _VERSION_RE.match(v):
            raise ValueError("use lowercase letters, digits and underscores (e.g. pilot_v2)")
        return v


def load_dev_label_spec(path: str | Path) -> tuple[DevLabelSpec, str]:
    """The spec at ``path`` and the SHA-256 of its bytes (recorded with the labels it derives)."""
    file = Path(path)
    if not file.is_file():
        raise DevLabelError(f"development label spec not found: {file}")
    try:
        return DevLabelSpec.model_validate(read_yaml(file)), sha256_file(file)
    except ValidationError as exc:
        raise DevLabelError(f"{file}: {_explain(exc)}") from None


# --- records ----------------------------------------------------------------------------


class AuditEvidence(BaseModel):
    """An audit grade for a labelled passage: evidence for review, not a label."""

    model_config = _STRICT

    item: int = Field(ge=1)
    human_grade: Grade | None  # None: the auditor skipped it
    annotator: NonEmptyStr
    independence: AuditIndependence
    agrees_with_base: bool | None


class DevLabel(BaseModel):
    """The development grade of one pooled passage and where it came from."""

    model_config = _STRICT

    query_id: NonEmptyStr
    passage_id: NonEmptyStr
    grade: Grade
    label_source: LabelSource
    ai_generated: bool
    assistance: Assistance
    labelled_by: NonEmptyStr  # the model for AI proposals, the person for human judgments
    base_grade: Grade
    changed: bool
    flags: list[str] = Field(default_factory=list)
    audit: AuditEvidence | None = None
    # The label version that first held this label, recorded by versions derived from another
    # label set (e.g. pilot_v3 from pilot_v2). Omitted when unset, so earlier sets render unchanged.
    introduced_in: str | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_origin(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if data.get("introduced_in") is None:
            data.pop("introduced_in", None)
        return data


class GradeChange(BaseModel):
    """One grade that differs from its base label, with everything needed to review it."""

    model_config = _STRICT

    change_id: NonEmptyStr
    query_id: NonEmptyStr
    passage_id: NonEmptyStr
    previous_grade: Grade
    new_grade: Grade
    rule: RuleId
    reason: NonEmptyStr
    check: dict[str, Any]
    evidence: list[str] = Field(default_factory=list)
    previous_label_source: LabelSource
    previous_explanation: str
    applied_by: NonEmptyStr


class LabelFlag(BaseModel):
    """A label that ``rule`` may change, but only with a reader's judgment."""

    model_config = _STRICT

    flag_id: NonEmptyStr
    query_id: NonEmptyStr
    passage_id: NonEmptyStr
    rule: FlagRule
    current_grade: Grade
    candidate_grade: Grade | None
    origin: Literal["rule", "spec"]
    reason: NonEmptyStr
    evidence: list[str] = Field(default_factory=list)


class AuditDisposition(BaseModel):
    """How the label set handles one audit item whose human grade differs from its base label."""

    model_config = _STRICT

    item: int = Field(ge=1)
    query_id: NonEmptyStr
    passage_id: NonEmptyStr
    human_grade: Grade
    base_grade: Grade
    grade: Grade
    disposition: Literal["changed", "flagged", "kept"]
    rule: RuleId | None
    reason: NonEmptyStr


class DevLabelSetInfo(BaseModel):
    """``labels.meta.json``: what a development label set is, where it came from, and its files."""

    model_config = _STRICT

    format_version: Literal[1] = 1
    kind: Literal["derived_development_labels"] = "derived_development_labels"
    status: Literal["provisional_development"] = DEV_STATUS
    label_version: NonEmptyStr
    description: NonEmptyStr
    evaluation: EvaluationMetadata
    notice: NonEmptyStr = DEV_NOTICE
    derived_from: dict[str, str]
    audit_independence: AuditIndependence
    audit_independence_note: NonEmptyStr
    rubric: RubricDecisions
    spec: dict[str, str]
    rules_applied: list[str]
    counts: dict[str, Any]
    audit_dispositions: list[AuditDisposition]
    files: dict[str, str]


# --- derivation -------------------------------------------------------------------------


def _dumps_line(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "\n"


def _render_jsonl(rows: Iterable[BaseModel]) -> bytes:
    return "".join(_dumps_line(row) for row in rows).encode("utf-8")


def _render_json(model: BaseModel) -> bytes:
    return (json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _evidence_text(audit: AuditEvidence) -> str:
    grade = "skipped" if audit.human_grade is None else f"human grade {audit.human_grade}"
    return f"audit item {audit.item}: {grade} by {audit.annotator} ({audit.independence} audit; evidence, not validation)"


P1_FLAG_REASON = (
    "P1: open question. Grade 2 if the passage explicitly states one element of the answer for the "
    "requested company and report, even as a small part of the passage; otherwise it stays 1. "
    "Deciding that needs a reader."
)


def p2_change(
    label: DevLabel, *, explanation: str, rule: BasisRule, text: str, change_id: str, applied_by: str
) -> GradeChange | None:
    """The P2 change for an AI grade-2 label whose passage shows none of ``rule``'s patterns, else ``None``."""
    if label.label_source != "ai_proposal" or label.grade != 2:
        return None
    matched = [pattern for pattern in rule.requires_any if re.search(pattern, text)]
    if matched:
        return None
    return GradeChange(
        change_id=change_id,
        query_id=label.query_id,
        passage_id=label.passage_id,
        previous_grade=2,
        new_grade=1,
        rule="P2",
        reason=(
            f"P2: the passage states the figure but not the requested basis ({rule.why}); a figure "
            "whose period or basis is defined only outside the passage is grade 1."
        ),
        check={"type": "p2_basis", "requires_any": list(rule.requires_any), "matched": matched},
        evidence=[] if label.audit is None else [_evidence_text(label.audit)],
        previous_label_source=label.label_source,
        previous_explanation=explanation,
        applied_by=applied_by,
    )


def make_flag(
    label: DevLabel, *, flag_id: str, rule: str, candidate: int | None, origin: str, reason: str
) -> LabelFlag:
    """A flag on ``label``; refuses a candidate grade the label already has."""
    if candidate is not None and candidate == label.grade:
        raise DevLabelError(f"flag on {(label.query_id, label.passage_id)} proposes grade {candidate}, which it already has")
    return LabelFlag(
        flag_id=flag_id,
        query_id=label.query_id, passage_id=label.passage_id, rule=rule, current_grade=label.grade,
        candidate_grade=candidate, origin=origin, reason=reason,
        evidence=[] if label.audit is None else [_evidence_text(label.audit)],
    )


# Files a derived label set records by SHA-256, by the derived_from keys that name them (the pool
# file names are those of finretriever.eval_set.dev_pool, which imports this module):
# {hash key: (directory key, file name)}. ``verify_inputs`` re-checks every one present.
_RECORDED_INPUTS: dict[str, tuple[str, str]] = {
    "proposals_sha256": ("proposals_dir", PROPOSALS_FILE),
    "proposals_meta_sha256": ("proposals_dir", PROPOSALS_INFO_FILE),
    "audit_sha256": ("audit_dir", AUDIT_FILE),
    "audit_history_sha256": ("audit_dir", HISTORY_FILE),
    "audit_report_sha256": ("audit_dir", REPORT_FILE),
    "base_labels_meta_sha256": ("base_labels_dir", LABELS_INFO_FILE),
    "pool_sha256": ("pool_dir", "pool.jsonl"),
    "pool_meta_sha256": ("pool_dir", "pool.meta.json"),
}


@dataclass(frozen=True)
class DevLabelSet:
    """A development label set in memory: its info and the records of its four data files."""

    info: DevLabelSetInfo
    questions: tuple[GoldQuestion, ...]
    labels: tuple[DevLabel, ...]
    changes: tuple[GradeChange, ...]
    flags: tuple[LabelFlag, ...]

    def rendered(self) -> dict[str, bytes]:
        """Exact bytes of every file, ``labels.meta.json`` included."""
        files = {
            QUESTIONS_SNAPSHOT_FILE: _render_jsonl(self.questions),
            LABELS_FILE: _render_jsonl(self.labels),
            CHANGES_FILE: _render_jsonl(self.changes),
            FLAGS_FILE: _render_jsonl(self.flags),
        }
        return {**files, LABELS_INFO_FILE: _render_json(self.info)}

    def label(self, query_id: str, passage_id: str) -> DevLabel | None:
        return next((x for x in self.labels if x.query_id == query_id and x.passage_id == passage_id), None)

    def write(self, directory: str | Path) -> bool:
        """Write the set; ``False`` if an identical copy is already there. Never overwrites a different set."""
        directory = Path(directory)
        files = self.rendered()
        existing = {name: directory / name for name in files if (directory / name).exists()}
        if existing:
            if len(existing) == len(files) and all(path.read_bytes() == files[name] for name, path in existing.items()):
                return False
            raise DevLabelError(
                f"{directory} already holds a different development label set; label versions are "
                "immutable, so derive a new version instead"
            )
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (directory / name).write_bytes(data)
        return True

    # --- reading -----------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        directory: str | Path,
        corpus: CorpusIdentity,
        passages: Mapping[str, Passage],
    ) -> DevLabelSet:
        """Load a label set after checking its corpus binding, file hashes and internal consistency."""
        directory = Path(directory)
        info_path = directory / LABELS_INFO_FILE
        if not info_path.is_file():
            raise DevLabelError(f"no development label set at {directory}")
        try:
            info = DevLabelSetInfo.model_validate_json(info_path.read_bytes())
        except ValidationError as exc:
            raise DevLabelError(f"{info_path}: {_explain(exc)}") from None
        check_annotations_match_corpus(info.evaluation, corpus)
        if sorted(info.files) != sorted(DATA_FILES):
            raise DevLabelError(f"{info_path} must list exactly {sorted(DATA_FILES)}")
        raw: dict[str, bytes] = {}
        for name, recorded in info.files.items():
            path = directory / name
            if not path.is_file():
                raise DevLabelError(f"missing label-set file {path}")
            raw[name] = path.read_bytes()
            if _sha256(raw[name]) != recorded:
                raise DevLabelError(f"{path} does not match the SHA-256 in {LABELS_INFO_FILE}; it was modified")

        def parse(name: str, model: type[BaseModel]) -> list[Any]:
            rows = []
            for line_no, line in enumerate(raw[name].decode("utf-8").splitlines(), start=1):
                try:
                    rows.append(model.model_validate_json(line))
                except ValidationError as exc:
                    raise DevLabelError(f"{name}:{line_no}: {_explain(exc)}") from None
            return rows

        label_set = cls(
            info,
            tuple(parse(QUESTIONS_SNAPSHOT_FILE, GoldQuestion)),
            tuple(parse(LABELS_FILE, DevLabel)),
            tuple(parse(CHANGES_FILE, GradeChange)),
            tuple(parse(FLAGS_FILE, LabelFlag)),
        )
        validate_label_set(label_set, passages)
        return label_set

    def verify_inputs(self) -> None:
        """Raise unless every recorded input is byte-identical to when the set was derived.

        That is the proposals and the audit, and for a set derived from another label set also
        the base labels, the pool, and (from the base's own record) the base's inputs.
        """
        _verify_recorded_inputs(self.info.derived_from, self.info.label_version)

    def input_dirs(self) -> list[str]:
        """The input directories recorded in ``derived_from``: never write into them."""
        recorded = self.info.derived_from
        keys = dict.fromkeys(directory for directory, _ in _RECORDED_INPUTS.values())
        return [recorded[key] for key in keys if key in recorded]


def _verify_recorded_inputs(recorded: Mapping[str, str], label_version: str) -> None:
    if "proposals_sha256" not in recorded:
        raise DevLabelError(f"{label_version} records no proposals; it was not derived by this tool")
    for key, (directory_key, name) in _RECORDED_INPUTS.items():
        if key not in recorded:
            continue
        path = Path(recorded[directory_key]) / name
        if not path.is_file():
            raise DevLabelError(f"input {path} of {label_version} is missing")
        if sha256_file(path) != recorded[key]:
            raise DevLabelError(
                f"{path} changed after {label_version} was derived; its inputs (proposals, audit, base "
                "labels, pool) must stay immutable"
            )
    if "base_labels_dir" in recorded:
        info_path = Path(recorded["base_labels_dir"]) / LABELS_INFO_FILE
        base = DevLabelSetInfo.model_validate_json(info_path.read_bytes())
        _verify_recorded_inputs(base.derived_from, base.label_version)


def validate_label_set(label_set: DevLabelSet, passages: Mapping[str, Passage]) -> None:
    """Raise ``DevLabelError`` unless labels, changes and flags are consistent and point at real passages."""
    questions = {q.query_id for q in label_set.questions}
    if len(questions) != len(label_set.questions):
        raise DevLabelError("duplicate question in the snapshot")
    keys = Counter((x.query_id, x.passage_id) for x in label_set.labels)
    if conflicts := sorted(key for key, n in keys.items() if n > 1):
        raise DevLabelError(f"conflicting labels: more than one label for {conflicts[:5]}")
    if unknown_q := sorted({x.query_id for x in label_set.labels} - questions):
        raise DevLabelError(f"labels for questions not in the snapshot: {unknown_q}")
    if unknown_p := sorted({x.passage_id for x in label_set.labels} - passages.keys()):
        raise DevLabelError(f"labels for passages not in the corpus: {unknown_p[:5]}")
    by_key = {(x.query_id, x.passage_id): x for x in label_set.labels}
    for label in label_set.labels:
        if label.changed != (label.grade != label.base_grade):
            raise DevLabelError(f"label {label.query_id}/{label.passage_id}: 'changed' disagrees with its grades")
        if label.ai_generated != (label.label_source == "ai_proposal"):
            raise DevLabelError(f"label {label.query_id}/{label.passage_id}: ai_generated disagrees with its source")
    change_ids = Counter(c.change_id for c in label_set.changes)
    if duplicates := sorted(cid for cid, n in change_ids.items() if n > 1):
        raise DevLabelError(f"duplicate change IDs: {duplicates}")
    changed = set()
    for change in label_set.changes:
        key = (change.query_id, change.passage_id)
        label = by_key.get(key)
        if label is None:
            raise DevLabelError(f"change {change.change_id} refers to an unlabelled passage {key}")
        if (change.previous_grade, change.new_grade) != (label.base_grade, label.grade):
            raise DevLabelError(f"change {change.change_id} does not match the label it changed")
        if change.previous_grade == change.new_grade:
            raise DevLabelError(f"change {change.change_id} records no change")
        if label.label_source != "ai_proposal":
            raise DevLabelError(f"change {change.change_id} rewrites a human grade; only AI proposals may change")
        if key in changed:
            raise DevLabelError(f"more than one change for {key}")
        changed.add(key)
    if silent := sorted(key for key, x in by_key.items() if x.changed and key not in changed):
        raise DevLabelError(f"grades changed without a change record: {silent[:5]}")
    flag_ids = Counter(f.flag_id for f in label_set.flags)
    if duplicates := sorted(fid for fid, n in flag_ids.items() if n > 1):
        raise DevLabelError(f"duplicate flag IDs: {duplicates}")
    flagged: dict[tuple[str, str], list[str]] = {}
    for flag in label_set.flags:
        key = (flag.query_id, flag.passage_id)
        label = by_key.get(key)
        if label is None:
            raise DevLabelError(f"flag {flag.flag_id} refers to an unlabelled passage {key}")
        if flag.current_grade != label.grade:
            raise DevLabelError(f"flag {flag.flag_id} records grade {flag.current_grade}, the label is {label.grade}")
        flagged.setdefault(key, []).append(flag.flag_id)
    for key, label in by_key.items():
        if sorted(label.flags) != sorted(flagged.get(key, [])):
            raise DevLabelError(f"label {key} lists flags {label.flags}, flags.jsonl has {flagged.get(key, [])}")


def derive_dev_labels(
    spec: DevLabelSpec,
    *,
    spec_path: str,
    spec_sha256: str,
    store: AnnotationStore,
    proposals: ProposalSet,
    audit: AuditWorkspace,
) -> DevLabelSet:
    """Derive the label set ``spec`` describes from the workspace, the proposals and the audit."""
    if audit.info.proposals_sha256 != proposals.sha256():
        raise DevLabelError(
            f"the audit at {audit.directory} was drawn from different proposals than {proposals.directory}"
        )
    report = audit.directory / REPORT_FILE
    if not report.is_file():
        raise DevLabelError(f"{report} is missing: finish the audit and run audit-report first")
    questions = sorted(store.questions(), key=lambda q: q.query_id)
    question_ids = {q.query_id for q in questions}
    for query_id in [r.query_id for r in spec.p2_basis] + list(spec.p1_open_questions):
        if query_id not in question_ids:
            raise DevLabelError(f"spec names unknown question {query_id}")

    audit_by_key: dict[tuple[str, str], AuditEvidence] = {}
    for item in audit.items():
        judgment = audit.judgment(item)
        if judgment is None:
            raise DevLabelError(f"audit item {item.item} is still pending")
        audit_by_key[item.query_id, item.passage_id] = AuditEvidence(
            item=item.item,
            human_grade=judgment.grade,
            annotator=judgment.annotator,
            independence=spec.audit_independence,
            agrees_with_base=None,
        )

    base: list[tuple[DevLabel, str]] = []  # (label before rules, the AI explanation if any)
    unlabelled: dict[str, int] = {}
    pool_sizes: dict[str, int] = {}
    for question in questions:
        pool = store.pool(question.query_id)
        pool_sizes[question.query_id] = len(pool)
        missing = 0
        for entry in pool:
            key = (question.query_id, entry.passage_id)
            human = store.judgment(*key)
            proposal = proposals.proposal(*key)
            human_grade = None if human is None else human.grade
            if human_grade is not None and proposal is not None:
                raise DevLabelError(f"conflicting labels for {key}: human grade {human_grade} and AI proposal")
            evidence = audit_by_key.get(key)
            if human_grade is not None:
                label = DevLabel(
                    query_id=key[0], passage_id=key[1], grade=human_grade, label_source="human_judgment",
                    ai_generated=False, assistance="not_recorded", labelled_by=human.annotator,
                    base_grade=human_grade, changed=False,
                )
                explanation = ""
            elif proposal is not None:
                label = DevLabel(
                    query_id=key[0], passage_id=key[1], grade=proposal.grade, label_source="ai_proposal",
                    ai_generated=True, assistance="not_applicable", labelled_by=proposal.model,
                    base_grade=proposal.grade, changed=False,
                )
                explanation = proposal.explanation
            else:
                missing += 1  # never invented: an unlabelled pool entry stays unjudged
                continue
            if evidence is not None:
                agrees = None if evidence.human_grade is None else evidence.human_grade == label.base_grade
                label = label.model_copy(update={"audit": evidence.model_copy(update={"agrees_with_base": agrees})})
            base.append((label, explanation))
        unlabelled[question.query_id] = missing
    if stray := sorted(set(audit_by_key) - {(x.query_id, x.passage_id) for x, _ in base}):
        raise DevLabelError(f"audit items without a label: {stray[:5]}")

    texts = store.passages
    basis = {rule.query_id: rule for rule in spec.p2_basis}
    applied_by = f"deterministic rule in {spec_path} (sha256 {spec_sha256[:12]})"
    labels: list[DevLabel] = []
    changes: list[GradeChange] = []
    for label, explanation in base:
        rule = basis.get(label.query_id)
        change = None if rule is None else p2_change(
            label, explanation=explanation, rule=rule, text=texts[label.passage_id].text,
            change_id=f"chg_{len(changes) + 1:03d}", applied_by=applied_by,
        )
        if change is not None:
            changes.append(change)
            label = label.model_copy(update={"grade": 1, "changed": True})
        labels.append(label)

    by_key = {(x.query_id, x.passage_id): x for x in labels}
    flags: list[LabelFlag] = []

    def add_flag(key: tuple[str, str], rule: str, candidate: int | None, origin: str, reason: str) -> None:
        label = by_key[key]
        flag = make_flag(
            label, flag_id=f"flg_{len(flags) + 1:03d}", rule=rule, candidate=candidate, origin=origin, reason=reason
        )
        flags.append(flag)
        by_key[key] = label.model_copy(update={"flags": [*label.flags, flag.flag_id]})

    open_questions = set(spec.p1_open_questions)
    for label in labels:
        if label.query_id in open_questions and label.label_source == "ai_proposal" and label.grade == 1:
            add_flag((label.query_id, label.passage_id), "P1", 2, "rule", P1_FLAG_REASON)
    for spec_flag in spec.flags:
        key = (spec_flag.query_id, spec_flag.passage_id)
        if key not in by_key:
            raise DevLabelError(f"spec flag refers to an unlabelled passage {key}")
        add_flag(key, spec_flag.rule, spec_flag.candidate_grade, "spec", spec_flag.reason)
    labels = [by_key[x.query_id, x.passage_id] for x in labels]

    # Every audited label whose human grade differs from its base, and every audited label a rule
    # changed, gets an explicit disposition, so audit evidence is never ignored silently.
    kept = {k.item: k for k in spec.kept_audit_disagreements}
    dispositions: list[AuditDisposition] = []
    for label in labels:
        evidence = label.audit
        if evidence is None or evidence.human_grade is None:
            continue
        if evidence.human_grade == label.base_grade and not label.changed:
            continue
        common = {
            "item": evidence.item, "query_id": label.query_id, "passage_id": label.passage_id,
            "human_grade": evidence.human_grade, "base_grade": label.base_grade, "grade": label.grade,
        }
        if label.changed:
            key = (label.query_id, label.passage_id)
            change = next(c for c in changes if (c.query_id, c.passage_id) == key)
            disposition = AuditDisposition(**common, disposition="changed", rule=change.rule, reason=change.reason)
        elif label.flags:
            flag = next(f for f in flags if f.flag_id == label.flags[0])
            disposition = AuditDisposition(**common, disposition="flagged", rule=flag.rule, reason=flag.reason)
        elif evidence.item in kept:
            entry = kept.pop(evidence.item)
            disposition = AuditDisposition(**common, disposition="kept", rule=entry.rule, reason=entry.reason)
        else:
            raise DevLabelError(
                f"audit item {evidence.item} ({label.query_id}) has human grade {evidence.human_grade} but the "
                f"label keeps {label.grade} with no rule, flag or kept_audit_disagreements entry; handle it "
                "explicitly"
            )
        dispositions.append(disposition)
    if kept:
        raise DevLabelError(f"kept_audit_disagreements lists items that are not open disagreements: {sorted(kept)}")
    dispositions.sort(key=lambda d: d.item)

    grades = Counter(x.grade for x in labels)
    per_question: dict[str, dict[str, Any]] = {}
    for question in questions:
        rows = [x for x in labels if x.query_id == question.query_id]
        per_question[question.query_id] = {
            "pool": pool_sizes[question.query_id],
            "labelled": len(rows),
            "unlabelled": unlabelled[question.query_id],
            "grades": {str(g): sum(1 for x in rows if x.grade == g) for g in (0, 1, 2)},
            "positives": sum(1 for x in rows if x.grade > 0),
            "changed": sum(1 for x in rows if x.changed),
            "flagged": sum(1 for x in rows if x.flags),
        }
    counts = {
        "labels": len(labels),
        "by_source": dict(sorted(Counter(x.label_source for x in labels).items())),
        "grades": {str(g): grades.get(g, 0) for g in (0, 1, 2)},
        "changes": dict(sorted(Counter(c.rule for c in changes).items())),
        "flags": dict(sorted(Counter(f.rule for f in flags).items())),
        "questions": per_question,
        "questions_without_positive": sorted(q for q, row in per_question.items() if row["positives"] == 0),
    }
    rules_applied = [
        f"P2 (deterministic): AI grade-2 labels for {', '.join(sorted(basis))} must match the period/basis "
        "patterns in the spec; otherwise grade 1",
        f"P1 (flag only): AI grade-1 labels for open questions {', '.join(sorted(open_questions))} are flagged "
        "for review; grades unchanged",
        "Curated spec flags (P2 explanation grounding, P5 evidence units): flagged for review; grades unchanged",
        "P3, P4 (option A) and P5 (option A): already reflected in the base labels; no grade changes",
        "P6: every change and flag records its rule and reason; label assistance is recorded per label",
        "Human judgments in the workspace are taken as they are and never changed",
    ]
    proposals_info = proposals.info
    derived_from = {
        "proposals_dir": str(proposals.directory),
        "proposals_sha256": proposals.sha256(),
        "proposals_meta_sha256": sha256_file(proposals.directory / PROPOSALS_INFO_FILE),
        "proposal_model": proposals_info.model,
        "proposal_method": proposals_info.method,
        "workspace_dir": str(store.directory),
        "workspace_history_sha256": sha256_file(store.directory / HISTORY_FILE),
        "audit_dir": str(audit.directory),
        "audit_sha256": sha256_file(audit.directory / AUDIT_FILE),
        "audit_history_sha256": sha256_file(audit.directory / HISTORY_FILE),
        "audit_report_sha256": sha256_file(audit.directory / REPORT_FILE),
    }
    draft = DevLabelSet(
        info=DevLabelSetInfo(
            label_version=spec.label_version,
            description=spec.description,
            evaluation=EvaluationMetadata(
                corpus_sha256=store.info.evaluation.corpus_sha256,
                corpus_version=store.info.evaluation.corpus_version,
                annotation_version=spec.annotation_version,
            ),
            derived_from=derived_from,
            audit_independence=spec.audit_independence,
            audit_independence_note=spec.audit_independence_note,
            rubric=spec.rubric,
            spec={"path": spec_path, "sha256": spec_sha256},
            rules_applied=rules_applied,
            counts=counts,
            audit_dispositions=dispositions,
            files={},
        ),
        questions=tuple(questions),
        labels=tuple(labels),
        changes=tuple(changes),
        flags=tuple(flags),
    )
    rendered = draft.rendered()
    files = {name: _sha256(rendered[name]) for name in DATA_FILES}
    label_set = DevLabelSet(
        draft.info.model_copy(update={"files": files}), draft.questions, draft.labels, draft.changes, draft.flags
    )
    validate_label_set(label_set, store.passages)
    return label_set


__all__ = [
    "CHANGES_FILE",
    "DATA_FILES",
    "DEV_NOTICE",
    "DEV_STATUS",
    "FLAGS_FILE",
    "LABELS_FILE",
    "LABELS_INFO_FILE",
    "P1_FLAG_REASON",
    "QUESTIONS_SNAPSHOT_FILE",
    "AuditDisposition",
    "AuditEvidence",
    "DevLabel",
    "DevLabelError",
    "DevLabelSet",
    "DevLabelSetInfo",
    "DevLabelSpec",
    "GradeChange",
    "LabelFlag",
    "derive_dev_labels",
    "load_dev_label_spec",
    "make_flag",
    "p2_change",
    "validate_label_set",
]
