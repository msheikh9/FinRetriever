"""Validation pairs: a separate, versioned, write-once set for choosing checkpoints.

A validation set (``build_validation_set``, configured by ``ValidationSetConfig``) takes the
generation records of one validation sampling plan, one question per passage, and includes a
question only if all of these hold (every reason it fails is logged):

* **provenance**: the plan is the written one, and the question is the candidate the generator
  accepted in its passage's latest record, under the split's recorded run (prompt, model,
  parameters and preparation);
* **quality**: one AI review decision (``load_validation_decisions``), bound to the passage and the
  SHA-256 of the question, accepts it, and none of the reviewed aspects (``REVIEW_ASPECTS``) failed;
* the current candidate checks (``VALIDATOR_VERSION``) pass;
* **the positive is validation's alone**: a validation passage, not blocked, whose overlap group
  and split unit lie wholly in validation, and which no training pair of this preparation or of
  the named training-candidate datasets uses as a positive or an equivalent;
* **evidence** (``LOCATOR_VERSION``): no passage outside the validation split holds it, by any
  match the locator reports, exact or uncertain, in its full text. Training pairs are rejected for
  exact matches only (``pair_quality.evidence_leak``: ``evaluation_evidence``,
  ``cross_split_evidence``); here any other match also excludes the question, as
  ``possible_evaluation_evidence`` (a blocked passage) or ``possible_training_evidence`` (a train
  passage), so no validation question can be answered by a passage a model may train on or by an
  evaluation passage;
* **questions**: none is lexically an evaluation question or a training question of this
  preparation (word-set Jaccard at the leakage threshold; training questions are every question
  of the train generation records, of the built train pairs and of the named candidate
  datasets), and none repeats within the set.

**Plan prefix and decisions** (``plan_positions``, ``decisions``): a set may take only the plan's first N passages, where
its generation ended; later positions are neither checked nor logged. A plan whose generation was stopped
(``sampling_plan.read_stop_record``) must be cut at or before its stop position, and the manifest binds the stop record.
The decisions may come in several files (one per audited batch), read together; a question decided in two of them is
an error.

**Carry-over** (``carry_over``): the pairs of written validation sets, of an earlier preparation
or this one, are re-checked here one by one, with every check above against this preparation and
its training data; their AI decisions come with them. A carried question that fails a check is
logged with its reasons and left out, never repaired or moved. Carried questions are checked
before the plan's, so a new question repeating one is the duplicate. A set may consist of carried
questions only.

``isolated_evidence_share`` scores a passage before any question exists: the share of its visible
sentences that no passage outside its split holds. Validation plans prefer high scores, so their
questions are likelier to pass the evidence rule.

The set is written once to ``<preparation>/validation_sets/<version>/``: ``validation_pairs.jsonl``
(contract pairs of the validation split), ``validation_log.jsonl`` (every passage of the plan,
included or not, with its reasons and every passage outside validation that holds its evidence)
and ``manifest.json`` (inputs and their SHA-256). The records file is bound by the lines it held,
so later validation runs may append to it. Trainers never read this directory: training
candidates live under ``candidates/`` and are opened by kind (``candidates.load_candidates``).

``load_validation_set`` reopens a set and fails closed (``TrainingDataError``) unless its files and
inputs are those recorded, every pair is still its record's accepted candidate and passes
``training_data.verify_pairs`` and the evidence rule again, and, given the training pairs a run
uses, none shares a passage or a question with them (``isolation_problems``). It reads no
evaluation file: the lexical evaluation check ran at build time and is bound by the manifest.

These are synthetic questions accepted by an AI review: not human-validated, not gold labels,
never training or evaluation data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finretriever.data.schemas import Passage, Query
from finretriever.eval_set.qrels_builder import find_gold_overlap
from finretriever.evaluation.provenance import sha256_file
from finretriever.training.candidates import (
    evaluation_questions_in,
    load_candidates,
    read_candidates_manifest,
)
from finretriever.training.carryover import CarryOverError, load_quality_decisions, question_sha256
from finretriever.training.data_prep import BLOCKLIST_FILE, DataPrepConfig, PreparedData
from finretriever.training.evidence_reuse import (
    LOCATOR_VERSION,
    Equivalent,
    EvidenceLocator,
    sentences,
)
from finretriever.training.pair_contract import (
    PAIR_CONTRACT_VERSION,
    TrainingPair,
    contract_problems,
    make_training_pair,
)
from finretriever.training.pair_quality import (
    GENERATION_DIRNAME,
    PAIRS_DIRNAME,
    PAIRS_META_FILE,
    PairQualityError,
    _max_similarity,
    aliases_for,
    equivalent_ids,
    evidence_locator,
    read_pairs,
)
from finretriever.training.query_generation import (
    VALIDATOR_VERSION,
    GenerationError,
    SourcePassage,
    check_candidate,
    check_run,
    continuations_path,
    latest_records,
    prompt_sha256,
    question_key,
    read_records,
    records_path,
    run_path,
)
from finretriever.training.sampling_plan import (
    PlanError,
    load_plan,
    passage_kind,
    plan_path,
    read_stop_record,
    stop_path,
)
from finretriever.training.training_data import TrainingDataError, verify_pairs

VALIDATION_DIRNAME = "validation_sets"
PAIRS_FILE = "validation_pairs.jsonl"
LOG_FILE = "validation_log.jsonl"
MANIFEST_FILE = "manifest.json"
KIND = "validation_pairs"
FORMAT_VERSION = 1
SPLIT = "validation"
REVIEWER = "ai_review"
REVIEW_ASPECTS = ("factual_accuracy", "financial_relevance", "company", "period", "figures", "units")
ASPECT_OUTCOMES = ("pass", "minor", "fail", "n/a")  # minor: imprecise, but the passage still answers the question
NOTICE = (
    "Synthetic validation questions written by an LLM from validation passages only, accepted by an AI quality "
    "review and re-checked against this preparation. They are for choosing checkpoints: not human-validated, not "
    "gold labels, never training data and never evaluation data."
)
EVIDENCE_RULE = (
    "no passage outside the validation split may hold the evidence: any exact or uncertain match of the evidence "
    "locator in the full text of a train or blocked passage excludes the question"
)
_NAME = r"^[a-z0-9][a-z0-9_.-]*$"
# Measured in the validation pilot val_pilot_v1 (docs/validation_pilot_v1_quality_report.md), used only for
# estimates: the generator accepted 43 of 50 questions, and the AI review accepted 16 of the 19 that passed the
# evidence rule. A planned passage's expected usable questions are about these two times its isolated share.
PILOT_GENERATOR_ACCEPTANCE = 43 / 50
PILOT_REVIEW_ACCEPTANCE = 16 / 19
ISOLATION_METHOD = (
    "share of a passage's visible sentences (6+ words, first 300 characters) that no passage outside its split holds "
    "by any evidence-locator match"
)


class ValidationSetError(ValueError):
    """Inputs, decisions or provenance are inconsistent, or an output would be overwritten."""


class CarriedSet(BaseModel):
    """A written validation set to carry over: re-checked question by question against this preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_prep_config: str  # the preparation it was built in (it may be this one)
    version: str = Field(pattern=_NAME)


class ValidationSetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(pattern=_NAME)
    data_prep_config: str
    plan: str | None = Field(default=None, pattern=_NAME)  # a validation sampling plan of the preparation
    plan_positions: int | None = Field(default=None, ge=1)  # only the plan's first N passages (where generation ended)
    decisions: str | list[str] | None = None  # AI quality decisions (JSONL) for the plan's questions: one file or several
    carry_over: list[CarriedSet] = Field(default_factory=list)  # earlier validation sets, re-checked here
    training_candidates: list[str] = Field(default_factory=list)  # candidate datasets it must stay apart from

    @model_validator(mode="after")
    def _sources(self) -> ValidationSetConfig:
        if (self.plan is None) != (self.decisions is None):
            raise ValueError("a plan needs its decisions, and decisions need their plan")
        if self.plan_positions is not None and self.plan is None:
            raise ValueError("plan_positions needs a plan")
        if isinstance(self.decisions, list) and (not self.decisions or len(set(self.decisions)) != len(self.decisions)):
            raise ValueError("decisions must list at least one file, each once")
        if self.plan is None and not self.carry_over:
            raise ValueError("a validation set needs a plan or a validation set to carry over")
        if len({(c.data_prep_config, c.version) for c in self.carry_over}) != len(self.carry_over):
            raise ValueError("a carried validation set is named twice")
        return self

    @property
    def decision_files(self) -> list[str]:
        if self.decisions is None:
            return []
        return [self.decisions] if isinstance(self.decisions, str) else list(self.decisions)


def load_validation_config(path: str | Path) -> tuple[ValidationSetConfig, str]:
    path = Path(path)
    try:
        return ValidationSetConfig.model_validate(yaml.safe_load(path.read_text("utf-8"))), sha256_file(path)
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        raise ValidationSetError(f"{path} is not a validation-set config ({exc})") from None


def validation_dir(prepared: PreparedData, version: str) -> Path:
    return prepared.directory / VALIDATION_DIRNAME / version


def pair_id_for(position: int) -> str:
    """The decision ID of the question for the plan's ``position``-th passage (1-based): fixed before any check,
    so a rejected question never renumbers the others."""
    return f"{SPLIT}_{position:06d}"


def load_validation_decisions(path: str | Path) -> dict[str, dict[str, Any]]:
    """AI quality decisions by pair ID (the ``carryover.load_quality_decisions`` format), each with an outcome for
    every reviewed aspect: an accepted question fails none, a revised or rejected one fails at least one."""
    try:
        decisions = load_quality_decisions(path)
    except CarryOverError as exc:
        raise ValidationSetError(str(exc)) from None
    for pair_id, row in decisions.items():
        where = f"{path}: {pair_id}"
        if row["reviewer"] != REVIEWER:
            raise ValidationSetError(
                f"{where}: reviewer {row['reviewer']!r}; these decisions are an AI review ({REVIEWER})"
            )
        aspects = row.get("aspects")
        if not isinstance(aspects, dict) or sorted(aspects) != sorted(REVIEW_ASPECTS):
            raise ValidationSetError(f"{where}: needs one outcome for each of {', '.join(REVIEW_ASPECTS)}")
        if unknown := sorted({str(v) for v in aspects.values()} - set(ASPECT_OUTCOMES)):
            raise ValidationSetError(f"{where}: unknown aspect outcomes {unknown}")
        failed = [a for a in REVIEW_ASPECTS if aspects[a] == "fail"]
        if row["decision"] == "accept" and failed:
            raise ValidationSetError(f"{where}: accepted although {', '.join(failed)} failed")
        if row["decision"] != "accept" and not failed:
            raise ValidationSetError(f"{where}: a {row['decision']} decision names no failed aspect")
    return decisions


def load_decision_files(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """The decisions of several files, each read by ``load_validation_decisions``; a question decided in two files is an
    error, so a later file can only add decisions, never replace one."""
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for pair_id, row in load_validation_decisions(path).items():
            if pair_id in merged:
                raise ValidationSetError(f"{path}: {pair_id} is decided in an earlier decisions file too")
            merged[pair_id] = row
    return merged


def passage_ids_of(pair: TrainingPair) -> set[str]:
    """The pair's positive and every equivalent, exact or uncertain."""
    return {pair.positive_passage_id, *pair.equivalents.exact, *pair.equivalents.uncertain}


@dataclass(frozen=True)
class TrainingSide:
    """What a validation set must stay apart from: a preparation's training questions and pair passages."""

    questions: list[Query]
    passages: frozenset[str]  # positives and equivalents of the built train pairs and the candidate datasets
    sources: dict[str, Any]  # what was read, with SHA-256


def training_side(
    prepared: PreparedData, passages: Mapping[str, Passage], config: DataPrepConfig, candidates: Sequence[str]
) -> TrainingSide:
    """Every training question of ``prepared`` (train generation records, built train pairs, the train candidates of
    each named dataset, re-verified) and every passage those pairs use."""
    texts: dict[str, str] = {}
    used: set[str] = set()
    records = records_path(prepared.directory / GENERATION_DIRNAME, "train")
    for record in read_records(records):
        for c in (*record.get("candidates", ()), *record.get("accepted", ()), *record.get("rejected", ())):
            if isinstance(c, Mapping) and isinstance(c.get("question"), str) and c["question"].strip():
                texts.setdefault(question_key(c["question"]), c["question"])
    sources: dict[str, Any] = {
        "generation_records": {"path": str(records), "sha256": sha256_file(records) if records.is_file() else ""},
        "pairs_meta_sha256": "",
        "candidates": [],
    }
    if (prepared.directory / PAIRS_DIRNAME / PAIRS_META_FILE).is_file():
        try:
            built = read_pairs(prepared)["train"]
        except PairQualityError as exc:
            raise ValidationSetError(str(exc)) from None
        for row in built:
            texts.setdefault(question_key(row["query"]), row["query"])
            eq = row.get("equivalents") or {}
            used.update([row["positive_passage_id"], *eq.get("exact", []), *eq.get("uncertain", [])])
        sources["pairs_meta_sha256"] = sha256_file(prepared.directory / PAIRS_DIRNAME / PAIRS_META_FILE)
    for version in candidates:
        try:
            _, digest = read_candidates_manifest(prepared, version)
            pairs = load_candidates(prepared, passages, config, version, "train")
        except TrainingDataError as exc:
            raise ValidationSetError(f"training candidates {version!r}: {exc}") from None
        for pair in pairs:
            texts.setdefault(question_key(pair.query), pair.query)
            used.update(passage_ids_of(pair))
        sources["candidates"].append({"version": version, "manifest_sha256": digest, "pairs": len(pairs)})
    questions = [Query(query_id=f"training:{i}", text=t, source="training") for i, t in enumerate(texts.values())]
    return TrainingSide(questions, frozenset(used), sources)


def outside_evidence_reasons(equivalents: Sequence[Equivalent], rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Why the passages outside the validation split that hold a question's evidence exclude it (empty if none do):
    the training-pair names for exact matches (``pair_quality.evidence_leak``), ``possible_*`` for the rest."""
    outside = [e for e in equivalents if rows[e.passage_id]["split"] != SPLIT]
    blocked = [e for e in outside if rows[e.passage_id]["split"] == "blocked"]
    train = [e for e in outside if rows[e.passage_id]["split"] != "blocked"]
    reasons = []
    if blocked:
        reasons.append("evaluation_evidence" if any(e.exact for e in blocked) else "possible_evaluation_evidence")
    if train:
        exact = any(e.visible_exact for e in train)
        reasons.append("cross_split_evidence" if exact else "possible_training_evidence")
    return reasons


def isolated_evidence_share(
    sources: Sequence[SourcePassage], locator: EvidenceLocator, rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    """For each passage, the share of its visible sentences (6 or more words, first 300 characters each) that no
    passage outside its own split holds, by any match of the evidence locator (``ISOLATION_METHOD``): how likely a
    question quoting one of them passes the validation set's evidence rule. In ``val_pilot_v1`` the plan's mean
    share was 41% and 44% of its accepted questions passed. Corpus text only; no evaluation label is read."""
    out = {}
    for p in sources:
        units = [s[:300] for s in sentences(p.visible_text) if len(s.split()) >= 6]
        clean = sum(
            1 for u in units
            if not any(rows[e.passage_id]["split"] != p.split for e in locator.locate(u, "", p.ticker, p.passage_id))
        )
        out[p.passage_id] = round(clean / len(units), 4) if units else 0.0
    return out


def isolation_problems(
    validation: Sequence[TrainingPair], training: Sequence[TrainingPair], *, threshold: float
) -> list[str]:
    """Every way ``validation`` is not apart from ``training``: a pair of the wrong split, a passage both use (as a
    positive or an equivalent), or a question repeated or lexically copied (word-set Jaccard >= ``threshold``)."""
    problems = [f"{t.pair_id}: a {t.split} pair among the training pairs" for t in training if t.split != "train"]
    problems += [f"{v.pair_id}: a {v.split} pair among the validation pairs" for v in validation if v.split != SPLIT]
    used = {pid: t.pair_id for t in training for pid in sorted(passage_ids_of(t))}
    keys = {question_key(t.query): t.pair_id for t in training}
    questions = [Query(query_id=t.pair_id, text=t.query, source="training") for t in training]
    for v in validation:
        if shared := sorted(passage_ids_of(v) & used.keys()):
            problems.append(f"{v.pair_id}: shares passage {shared[0]} with training pair {used[shared[0]]}")
        if question_key(v.query) in keys:
            problems.append(f"{v.pair_id}: repeats the question of training pair {keys[question_key(v.query)]}")
        elif overlap := find_gold_overlap(questions, [v.query], threshold=threshold):
            problems.append(f"{v.pair_id}: its question overlaps training pair {overlap[0].query_id}'s")
    return problems


def records_prefix(path: Path, lines: int | None = None) -> tuple[int, str]:
    """The number of complete lines of ``path`` (or ``lines`` of them) and the SHA-256 of those lines; ``""`` if the
    file holds fewer."""
    complete = (path.read_bytes() if path.is_file() else b"").split(b"\n")[:-1]
    n = len(complete) if lines is None else lines
    if n > len(complete):
        return len(complete), ""
    return n, hashlib.sha256(b"".join(line + b"\n" for line in complete[:n])).hexdigest()


def _locator(prepared: PreparedData, passages: Mapping[str, Passage]) -> EvidenceLocator:
    try:
        return evidence_locator(prepared, passages)
    except PairQualityError as exc:
        raise ValidationSetError(str(exc)) from None


def _block_reasons(prepared: PreparedData) -> dict[str, list[str]]:
    """Why each blocked passage is blocked, by reason kind (an evaluation pool, its overlap group, evidence reuse)."""
    out = {}
    for line in (prepared.directory / BLOCKLIST_FILE).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row["passage_id"]] = sorted({reason.split(":")[0] for reason in row["reasons"]})
    return out


@dataclass(frozen=True)
class ValidationSet:
    pairs: list[dict[str, Any]]
    log: list[dict[str, Any]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class CarriedSource:
    """A written validation set to carry into the preparation being built (``ValidationSetConfig.carry_over``)."""

    entry: CarriedSet
    prepared: PreparedData  # where the set was built
    config: DataPrepConfig
    config_sha256: str


def build_validation_set(
    config: ValidationSetConfig,
    *,
    config_path: str,
    config_sha256: str,
    prepared: PreparedData,
    prep_config: DataPrepConfig,
    prep_config_sha256: str,
    passages: Mapping[str, Passage],
    evaluation_questions: Sequence[Query],
    carried: Sequence[CarriedSource] = (),
) -> ValidationSet:
    """Check every carried question and every question of ``config.plan``; see the module docstring. Nothing is
    written."""
    locator = _locator(prepared, passages)  # before anything else: fail closed
    settings = prep_config.generation
    aliases = aliases_for(passages, settings.company_aliases)
    rows = prepared.rows
    if [c.entry for c in carried] != list(config.carry_over):
        raise ValidationSetError("the carried sources are not the ones the config names")
    training = training_side(prepared, passages, prep_config, config.training_candidates)
    blocked_for = _block_reasons(prepared)
    unit_of = prepared.split_units()
    sides: dict[str, set[str]] = {}
    for pid, row in rows.items():
        sides.setdefault(f"group:{row['group_id']}", set()).add(row["split"])
        sides.setdefault(f"unit:{unit_of[pid]}", set()).add(row["split"])
    threshold = prep_config.leakage.lexical_overlap_threshold
    pairs: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    seen: set[str] = set()

    def assess(raw: Mapping[str, str], pid: str, reasons: list[str]) -> dict[str, Any]:
        """The checks every question gets, appended to ``reasons``; what a pair and its log row need."""
        row = rows[pid]
        candidate = None
        if row["split"] != SPLIT:
            reasons.append("positive_not_validation")
        else:
            candidate, problem = check_candidate(raw, prepared.source_passage(pid, passages), settings, aliases)
            if problem is not None:
                reasons.append(f"validator_{problem}")
        if sides[f"group:{row['group_id']}"] != {SPLIT} or sides[f"unit:{unit_of[pid]}"] != {SPLIT}:
            reasons.append("positive_unit_crosses_split")
        if pid in training.passages:
            reasons.append("positive_used_in_training")
        equivalents = locator.locate(raw["evidence"], raw["answer"], row["ticker"], pid)
        reasons.extend(outside_evidence_reasons(equivalents, rows))
        evaluation_similarity = _max_similarity(evaluation_questions, raw["question"])
        if evaluation_similarity >= threshold:
            reasons.append("evaluation_overlap")
        training_similarity = _max_similarity(training.questions, raw["question"])
        if training_similarity >= threshold:
            reasons.append("training_query_overlap")
        if question_key(raw["question"]) in seen:
            reasons.append("duplicate_query")
        ids = equivalent_ids(equivalents, rows, SPLIT)
        return {
            "candidate": candidate,
            "ids": ids,
            "evaluation_similarity": evaluation_similarity,
            "log": {
                "evidence": {
                    "method": LOCATOR_VERSION,
                    "outside_validation": [
                        {"passage_id": e.passage_id, "split": rows[e.passage_id]["split"], "match": e.match,
                         "visible_match": e.visible_match,
                         **({"blocked_for": blocked_for.get(e.passage_id, [])}
                            if rows[e.passage_id]["split"] == "blocked" else {})}
                        for e in equivalents
                        if rows[e.passage_id]["split"] != SPLIT
                    ],
                    "exact": len(ids["exact"]),
                    "uncertain": len(ids["uncertain"]),
                },
                "max_evaluation_similarity": evaluation_similarity,
                "max_training_similarity": training_similarity,
            },
        }

    def include(pair: dict[str, Any], question: str) -> None:
        if problems := contract_problems(pair, prepared, passages):
            raise ValidationSetError(f"{pair['pair_id']}: {problems[:3]}")  # a bug, not a data issue
        if any(p["pair_id"] == pair["pair_id"] for p in pairs):
            raise ValidationSetError(f"pair ID {pair['pair_id']} repeats; carried sets and plans must not share IDs")
        seen.add(question_key(question))
        pairs.append(pair)

    carried_info = []
    for source in carried:  # existing questions first, so a new question repeating one is the duplicate
        entry = source.entry
        try:
            source_manifest, source_manifest_sha256 = read_validation_manifest(source.prepared, entry.version)
            source_pairs = load_validation_set(source.prepared, passages, source.config, entry.version)
        except TrainingDataError as exc:
            raise ValidationSetError(f"carried validation set {entry.version!r}: {exc}") from None
        source_log = {r["pair_id"]: r for r in _read_jsonl(validation_dir(source.prepared, entry.version) / LOG_FILE)}
        for p in source_pairs:
            raw = {"question": p.query, "type": p.question_type, "answer": p.answer, "evidence": p.evidence}
            reasons: list[str] = []
            checked = assess(raw, p.positive_passage_id, reasons)
            if not reasons and checked["candidate"] is not None:
                prov = p.provenance
                include(make_training_pair(
                    pair_id=p.pair_id,
                    split=SPLIT,
                    candidate=checked["candidate"].to_dict(),
                    positive=p.positive_passage_id,
                    prepared=prepared,
                    passages=passages,
                    similarity=checked["evaluation_similarity"],
                    equivalents=checked["ids"],
                    provider=prov.provider,
                    model=prov.model,
                    prompt_version=prov.prompt_version,
                    prompt_sha=prov.prompt_sha256,
                    validator_version=VALIDATOR_VERSION,
                    params_sha256=prov.generation_params_sha256,
                    origin=f"carryover:{entry.version}",
                ), p.query)
            log.append({
                "pair_id": p.pair_id,
                "carried_from": entry.version,
                "positive_passage_id": p.positive_passage_id,
                "ticker": p.ticker,
                "fiscal_year": p.fiscal_year,
                "passage_kind": source_log[p.pair_id]["passage_kind"],
                "question": p.query,
                "question_sha256": question_sha256(p.query),
                "question_type": p.question_type,
                "included": not reasons,
                "reasons": reasons,
                "quality": source_log[p.pair_id]["quality"],
                **checked["log"],
            })
        carried_info.append({
            "version": entry.version,
            "data_prep_config": {"path": entry.data_prep_config, "sha256": source.config_sha256},
            "preparation": {
                "directory": str(source.prepared.directory),
                "version": source.prepared.manifest["version"],
                "manifest_sha256": source.prepared.manifest_sha256,
            },
            "manifest": {"path": str(validation_dir(source.prepared, entry.version) / MANIFEST_FILE),
                         "sha256": source_manifest_sha256},
            "decisions": source_manifest.get("decisions"),
            "pairs": len(source_pairs),
        })

    plan_sections: dict[str, Any] = {}
    if config.plan is not None:
        plan_sections = _plan_questions(
            config, prepared, prep_config, passages, settings, aliases, assess, include, log
        )
    question_files = [
        {"path": path, "sha256": sha256_file(path), "questions": len(evaluation_questions_in(path))}
        for path in prep_config.leakage.evaluation_question_files
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "version": config.version,
        "notice": NOTICE,
        "labels": "AI quality review (accept only); not human-validated; not gold",
        "use": "choosing checkpoints and early stopping; never training data, never evaluation data",
        "config": {"path": config_path, "sha256": config_sha256},
        "corpus": prepared.manifest["corpus"],
        "preparation": {
            "data_prep_config": {"path": config.data_prep_config, "sha256": prep_config_sha256},
            "directory": str(prepared.directory),
            "version": prepared.manifest["version"],
            "manifest_sha256": prepared.manifest_sha256,
            "evidence_index_digest": locator.index.digest(),
            "visible_text": prepared.manifest["visible_text"]["identity"],
        },
        **plan_sections,
        **({"carry_over": carried_info} if carried_info else {}),
        "training": {**training.sources, "questions": len(training.questions), "passages": len(training.passages)},
        "checks": {
            "pair_contract_version": PAIR_CONTRACT_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "equivalence_method": LOCATOR_VERSION,
            "evidence_rule": EVIDENCE_RULE,
            "evaluation_overlap": {
                "method": "word-set Jaccard", "threshold": threshold, "question_files": question_files
            },
            "training_overlap": {"method": "word-set Jaccard", "threshold": threshold},
        },
        "stats": _stats(log),
    }
    return ValidationSet(pairs, log, manifest)


def _plan_questions(
    config: ValidationSetConfig,
    prepared: PreparedData,
    prep_config: DataPrepConfig,
    passages: Mapping[str, Passage],
    settings: Any,
    aliases: Any,
    assess: Any,
    include: Any,
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check the question of every passage of ``config.plan`` (appending to the log and, through ``include``, the
    pairs); the manifest's plan, generation and decisions sections."""
    assert config.plan is not None and config.decisions is not None
    approved = prepared.source_passages(SPLIT, passages, min_words=prep_config.min_source_words)
    sources = {p.passage_id: p for p in approved}
    try:
        plan, planned = load_plan(
            prepared.directory, SPLIT, config.plan, sources, prepared_manifest_sha256=prepared.manifest_sha256
        )
        run = check_run(prepared.directory / GENERATION_DIRNAME, SPLIT, settings, prepared.manifest_sha256)
        stop = read_stop_record(prepared.directory, SPLIT, config.plan)
    except (PlanError, GenerationError) as exc:
        raise ValidationSetError(str(exc)) from None
    plan_size = len(planned)
    if config.plan_positions is not None:
        if config.plan_positions > plan_size:
            raise ValidationSetError(
                f"plan {config.plan} has {plan_size} passages, fewer than plan_positions {config.plan_positions}"
            )
        planned = planned[: config.plan_positions]  # later positions are not part of this set: not checked, not logged
    if stop is not None and len(planned) > stop["position"]:
        raise ValidationSetError(
            f"plan {config.plan} was stopped at position {stop['position']}; set plan_positions to {stop['position']} "
            "or fewer"
        )
    generation_dir = prepared.directory / GENERATION_DIRNAME
    if run.state == "new":
        raise ValidationSetError(
            f"no validation generation run in {generation_dir}; generate the plan's questions first"
        )
    bound = [b["params_sha256"] for b in run.bindings]  # the run, then each continuation
    records_file = records_path(generation_dir, SPLIT)
    lines, records_sha256 = records_prefix(records_file)
    state = latest_records(read_records(records_file))
    decisions = load_decision_files(config.decision_files)
    planned_ids = {pair_id_for(i): p.passage_id for i, p in enumerate(planned, start=1)}
    if unknown := sorted(set(decisions) - set(planned_ids)):
        raise ValidationSetError(
            f"{', '.join(config.decision_files)} decide {unknown[:3]}, which plan {config.plan} does not have"
            + ("" if config.plan_positions is None else f" in its first {config.plan_positions} positions")
        )
    for position, source in enumerate(planned, start=1):
        pid, pair_id = source.passage_id, pair_id_for(position)
        entry: dict[str, Any] = {
            "pair_id": f"{config.plan}:{pair_id}",
            "plan_position": position,
            "positive_passage_id": pid,
            "ticker": source.ticker,
            "fiscal_year": source.fiscal_year,
            "passage_kind": passage_kind(source),
        }
        record = state.get(pid)
        if record is None or record["status"] != "done":
            log.append({**entry, "included": False,
                        "reasons": ["no_generation_record" if record is None else "generation_failed"]})
            continue
        if len(record["accepted"]) > 1:
            raise ValidationSetError(
                f"the record for {pid} accepted {len(record['accepted'])} questions; a validation set takes one "
                "question per passage (queries_per_passage.validation: 1)"
            )
        if not record["accepted"]:
            rejected = record["rejected"]
            log.append({**entry, "question": rejected[0].get("question") if rejected else None, "included": False,
                        "reasons": [f"generator_{r['reason']}" for r in rejected] or ["generator_no_question"]})
            continue
        raw = record["accepted"][0]
        question_hash = question_sha256(raw["question"])
        decision = decisions.get(pair_id)
        if decision is not None and (decision["passage_id"] != pid or decision["question_sha256"] != question_hash):
            raise ValidationSetError(
                f"the decision for {pair_id} names another passage or question; decisions never move between "
                "questions"
            )
        reasons: list[str] = []
        params = record.get("params_sha256", bound[0])  # records before format 3 predate any continuation
        if params not in bound:
            reasons.append("provenance_parameters_differ")
        version = record.get("prompt_version", settings.prompt_version)
        if version != settings.prompt_version:
            reasons.append("provenance_prompt_differs")
        if decision is None:
            reasons.append("no_quality_decision")
        elif decision["decision"] != "accept":
            reasons.append(f"quality_{decision['decision']}")
        checked = assess(raw, pid, reasons)
        if not reasons and checked["candidate"] is not None:
            include(make_training_pair(
                pair_id=entry["pair_id"],
                split=SPLIT,
                candidate=checked["candidate"].to_dict(),
                positive=pid,
                prepared=prepared,
                passages=passages,
                similarity=checked["evaluation_similarity"],
                equivalents=checked["ids"],
                provider=settings.provider,
                model=record.get("model", settings.model),
                prompt_version=version,
                prompt_sha=prompt_sha256(version),
                validator_version=VALIDATOR_VERSION,
                params_sha256=params,
                origin=f"validation-plan:{config.plan}",
            ), raw["question"])
        log.append({
            **entry,
            "question": raw["question"],
            "question_sha256": question_hash,
            "question_type": raw["type"],
            "included": not reasons,
            "reasons": reasons,
            "quality": None if decision is None else {
                "decision": decision["decision"],
                "issues": list(decision["issues"]),
                "aspects": dict(decision["aspects"]),
                "reviewer": decision["reviewer"],
                "gold": False,
            },
            **checked["log"],
        })
    plan_file = plan_path(prepared.directory, SPLIT, config.plan)
    run_file, continuations = run_path(generation_dir, SPLIT), continuations_path(generation_dir, SPLIT)
    plan_section: dict[str, Any] = {"name": config.plan, "path": str(plan_file), "sha256": sha256_file(plan_file),
                                    "method": plan["method"], "seed": plan["seed"], "passages": plan_size}
    if config.plan_positions is not None:
        plan_section["positions_used"] = len(planned)
    if stop is not None:
        stop_file = stop_path(prepared.directory, SPLIT, config.plan)
        plan_section["stop"] = {"path": str(stop_file), "sha256": sha256_file(stop_file), "position": stop["position"]}
    files = config.decision_files
    decisions_section: dict[str, Any] = (
        {"path": files[0], "sha256": sha256_file(files[0])}
        if isinstance(config.decisions, str)
        else {"files": [{"path": p, "sha256": sha256_file(p)} for p in files]}
    )
    return {
        "plan": plan_section,
        "generation": {
            "provider": settings.provider,
            "model": settings.model,
            "prompt_version": settings.prompt_version,
            "prompt_sha256": prompt_sha256(settings.prompt_version),
            "params_sha256": bound,
            "run": {"path": str(run_file), "sha256": sha256_file(run_file)},
            "continuations": {"path": str(continuations),
                              "sha256": sha256_file(continuations) if continuations.exists() else ""},
            "records": {"path": str(records_file), "lines": lines, "sha256": records_sha256},
        },
        "decisions": {**decisions_section, "reviewer": REVIEWER, "decided": len(decisions)},
    }


def _stats(log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(r[field]) for r in rows).items()))

    not_generated = ("generator_", "generation_", "no_generation")
    planned = [r for r in log if "carried_from" not in r]
    carried = [r for r in log if "carried_from" in r]
    generated = [r for r in planned if not any(x.startswith(not_generated) for x in r["reasons"])]
    included = [r for r in log if r["included"]]
    tickers = Counter(r["ticker"] for r in included)
    return {
        "plan_passages": len(planned),
        "accepted_by_generator": len(generated),
        **({"carried": {"pairs": len(carried), "included": sum(r["included"] for r in carried)}} if carried else {}),
        "included": len(included),
        "excluded_by_reason": dict(sorted(Counter(x for r in log for x in r["reasons"]).items())),
        "included_by_ticker": counts(included, "ticker"),
        "included_by_fiscal_year": counts(included, "fiscal_year"),
        "included_by_question_type": counts(included, "question_type"),
        "included_by_passage_kind": counts(included, "passage_kind"),
        "largest_company_share": round(max(tickers.values()) / len(included), 4) if included else 0.0,
        "included_with_exact_equivalents": sum(r["evidence"]["exact"] > 0 for r in included),
        "included_with_uncertain_equivalents": sum(r["evidence"]["uncertain"] > 0 for r in included),
        "max_evaluation_similarity": max((r["max_evaluation_similarity"] for r in included), default=0.0),
        "max_training_similarity": max((r["max_training_similarity"] for r in included), default=0.0),
    }


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows
    ).encode()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_validation_set(prepared: PreparedData, built: ValidationSet) -> tuple[bool, Path]:
    """Write the set once; ``False`` if an identical copy exists. A different one is refused, and nothing is
    written."""
    directory = validation_dir(prepared, built.manifest["version"])
    data = {PAIRS_FILE: _jsonl(built.pairs), LOG_FILE: _jsonl(built.log)}
    files = {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(data.items())}
    manifest = {**built.manifest, "files": files}
    data[MANIFEST_FILE] = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    if directory.exists():
        present = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        if present == data:
            return False, directory
        raise ValidationSetError(
            f"{directory} already holds a different validation set; written sets are never replaced, so nothing was "
            "written; use a new version"
        )
    directory.mkdir(parents=True)
    for name in sorted(n for n in data if n != MANIFEST_FILE):
        (directory / name).write_bytes(data[name])
    (directory / MANIFEST_FILE).write_bytes(data[MANIFEST_FILE])  # last: a manifest means a complete set
    return True, directory


def read_validation_manifest(prepared: PreparedData, version: str) -> tuple[dict[str, Any], str]:
    """The manifest of a written set and its SHA-256, after checking its files and every input it is bound to."""
    directory = validation_dir(prepared, version)
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise TrainingDataError(f"no validation set {version!r} at {directory}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != KIND or manifest.get("format_version") != FORMAT_VERSION:
        raise TrainingDataError(f"{path} is not a format-{FORMAT_VERSION} validation-set manifest")
    for name, digest in manifest["files"].items():
        if not (directory / name).is_file() or sha256_file(directory / name) != digest:
            raise TrainingDataError(f"{directory / name} is missing or does not match {MANIFEST_FILE}")
    prep = manifest["preparation"]
    if prep["manifest_sha256"] != prepared.manifest_sha256 or manifest["corpus"] != prepared.manifest["corpus"]:
        raise TrainingDataError(f"validation set {version!r} was built against another preparation or corpus")
    if prepared.manifest.get("reuse", {}).get("index", {}).get("digest") != prep["evidence_index_digest"]:
        raise TrainingDataError(f"validation set {version!r} was checked under another evidence-reuse index")
    checks = manifest["checks"]
    if (checks["equivalence_method"], checks["validator_version"]) != (LOCATOR_VERSION, VALIDATOR_VERSION):
        raise TrainingDataError(
            f"validation set {version!r} was checked by {checks['equivalence_method']} and "
            f"{checks['validator_version']}, not {LOCATOR_VERSION} and {VALIDATOR_VERSION}; build a new version"
        )
    recorded = [c["manifest"] for c in manifest.get("carry_over", ())]  # each carried set's manifest
    if "plan" in manifest:
        generation, decisions = manifest["generation"], manifest["decisions"]
        recorded += [manifest["plan"], *decisions.get("files", [decisions]), generation["run"], generation["continuations"]]
        if "stop" in manifest["plan"]:
            recorded.append(manifest["plan"]["stop"])
        records = generation["records"]
        if records_prefix(Path(records["path"]), records["lines"]) != (records["lines"], records["sha256"]):
            raise TrainingDataError(
                f"validation set {version!r}: the first {records['lines']} lines of {records['path']} changed since "
                "it was built; generation records are append-only"
            )
    for item in recorded:
        file = Path(item["path"])
        if (sha256_file(file) if file.is_file() else "") != item["sha256"]:
            raise TrainingDataError(
                f"validation set {version!r}: {file} changed since it was built; build a new version"
            )
    return manifest, sha256_file(path)


def _provenance_problems(
    manifest: Mapping[str, Any], log: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Each pair must be the accepted candidate of its plan passage's record among the bound record lines, or the
    same question, on the same passage, as a pair of the carried set it came from."""
    problems = []
    included = {r["pair_id"]: r for r in log if r["included"]}
    if sorted(included) != sorted(r["pair_id"] for r in records):
        problems.append("the pair file and the log disagree on which questions were included")
    fields = ("query", "question_type", "answer", "evidence", "positive_passage_id")
    carried: dict[str, dict[str, Any]] = {}
    for c in manifest.get("carry_over", ()):
        source = json.loads(Path(c["manifest"]["path"]).read_text(encoding="utf-8"))  # its SHA-256 is checked
        pairs_file = Path(c["manifest"]["path"]).parent / PAIRS_FILE
        if not pairs_file.is_file() or sha256_file(pairs_file) != source["files"][PAIRS_FILE]:
            problems.append(f"carried set {c['version']}: {pairs_file} does not match its manifest")
            continue
        carried[c["version"]] = {r["pair_id"]: r for r in _read_jsonl(pairs_file)}
    plan = generation = state = None
    if "plan" in manifest:
        plan = json.loads(Path(manifest["plan"]["path"]).read_text(encoding="utf-8"))["passages"]
        generation = manifest["generation"]["records"]
        lines = Path(generation["path"]).read_text(encoding="utf-8").splitlines()[: generation["lines"]]
        state = latest_records([json.loads(line) for line in lines])
    for pair in records:
        entry = included.get(pair["pair_id"])
        if entry is None:
            continue
        if "carried_from" in entry:
            original = carried.get(entry["carried_from"], {}).get(pair["pair_id"])
            if original is None or any(original[k] != pair[k] for k in fields):
                problems.append(f"{pair['pair_id']}: not the pair of the carried set {entry['carried_from']}")
            continue
        if plan is None or state is None:
            problems.append(f"{pair['pair_id']}: a planned question in a set without a plan")
            continue
        pid = plan[entry["plan_position"] - 1]["passage_id"]
        record = state.get(pid)
        wanted = tuple(" ".join(str(pair[k]).split()) for k in ("query", "question_type", "answer", "evidence"))
        accepted = {
            tuple(" ".join(str(c.get(k, "")).split()) for k in ("question", "type", "answer", "evidence"))
            for c in (record or {}).get("accepted", ())
        }
        if pid != pair["positive_passage_id"] or wanted not in accepted:
            problems.append(f"{pair['pair_id']}: not the accepted candidate of its plan passage's generation record")
    return problems


def load_validation_set(
    prepared: PreparedData,
    passages: Mapping[str, Passage],
    config: DataPrepConfig,
    version: str,
    *,
    training: Sequence[TrainingPair] | None = None,
) -> list[TrainingPair]:
    """The pairs of a written set, re-verified (and checked apart from ``training``); reads no evaluation file."""
    manifest, _ = read_validation_manifest(prepared, version)
    directory = validation_dir(prepared, version)
    records = _read_jsonl(directory / PAIRS_FILE)
    problems = _provenance_problems(manifest, _read_jsonl(directory / LOG_FILE), records)
    try:
        locator = evidence_locator(prepared, passages)
    except PairQualityError as exc:
        raise TrainingDataError(str(exc)) from None
    pairs = verify_pairs(records, prepared, passages, config, SPLIT, evaluation_questions=None, locator=locator)
    for pair in pairs:
        equivalents = locator.locate(pair.evidence, pair.answer, pair.ticker, pair.positive_passage_id)
        if reasons := outside_evidence_reasons(equivalents, prepared.rows):
            problems.append(f"{pair.pair_id}: {', '.join(reasons)}")
    if training is not None:
        problems += isolation_problems(pairs, training, threshold=config.leakage.lexical_overlap_threshold)
    if problems:
        raise TrainingDataError(f"{len(problems)} problems, no validation pairs returned: {problems[:5]}")
    return pairs


__all__ = [
    "ASPECT_OUTCOMES",
    "EVIDENCE_RULE",
    "ISOLATION_METHOD",
    "KIND",
    "LOG_FILE",
    "MANIFEST_FILE",
    "NOTICE",
    "PAIRS_FILE",
    "PILOT_GENERATOR_ACCEPTANCE",
    "PILOT_REVIEW_ACCEPTANCE",
    "REVIEWER",
    "REVIEW_ASPECTS",
    "VALIDATION_DIRNAME",
    "CarriedSet",
    "CarriedSource",
    "TrainingSide",
    "ValidationSet",
    "ValidationSetConfig",
    "ValidationSetError",
    "build_validation_set",
    "isolated_evidence_share",
    "isolation_problems",
    "load_decision_files",
    "load_validation_config",
    "load_validation_decisions",
    "load_validation_set",
    "outside_evidence_reasons",
    "pair_id_for",
    "passage_ids_of",
    "read_validation_manifest",
    "records_prefix",
    "training_side",
    "validation_dir",
    "write_validation_set",
]
