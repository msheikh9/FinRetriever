"""The training-pair contract: what every pair record carries, and its checks against a preparation.

A pair (``TrainingPair``, ``PAIR_CONTRACT_VERSION``) is one query and the passage that answers
it, with everything a trainer needs to use it safely and everything an audit needs to trace it:

* the query, its answer and evidence quote, and its question type;
* the positive passage, its source metadata (company, filing, fiscal year, section, visible
  length), its overlap group and its split unit (groups joined by evidence reuse);
* its evidence equivalents: same-split passages whose visible text holds the evidence exactly
  (additional positives) or may answer the question (uncertain), with the evidence-reuse method
  and the index digest they were computed under;
* its provenance: corpus, preparation (version and manifest SHA-256), generation model,
  parameters, prompt version and prompt SHA-256, validator version, and where the pair came from
  (``build-pairs`` or a carry-over).

``contract_problems`` checks a record against the preparation it names and returns every
problem: a positive outside the pair's split (so validation pairs come only from the validation
split, and train pairs only from train), metadata or groups that differ from the preparation,
equivalents that are blocked, in another split or missing, an index digest or manifest that is
not this preparation's, a prompt SHA-256 that does not match its version, or a validator other
than the current one (the pair must be re-checked). Nothing is repaired.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finretriever.data.schemas import Passage
from finretriever.training.data_prep import PreparedData
from finretriever.training.evidence_reuse import LOCATOR_VERSION
from finretriever.training.query_generation import (
    PROMPTS,
    QUESTION_TYPES,
    VALIDATOR_VERSION,
    SourcePassage,
    prompt_sha256,
)

PAIR_CONTRACT_VERSION = 1
# finretriever.training.evidence_reuse.EvidenceLocator; v1 compared figures as written, so pairs it
# checked are stale (contract_problems) and must be re-checked.
EQUIVALENCE_METHOD = LOCATOR_VERSION
EQUIVALENCE_METHODS = ("evidence-reuse-v1", "evidence-reuse-v2")
_SHA256 = r"^[0-9a-f]{64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceInfo(_Strict):
    ticker: str
    company: str
    fiscal_year: str
    form_type: str
    filing_date: str
    report_date: str
    section: str
    item: str
    visible_chars: int = Field(ge=1)


class Equivalents(_Strict):
    exact: list[str]  # also positives: their visible text holds the evidence itself
    uncertain: list[str]  # may answer the question: never negatives, not positives
    method: Literal["evidence-reuse-v1", "evidence-reuse-v2"]
    index_digest: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _disjoint(self) -> Equivalents:
        if len(set(self.exact)) != len(self.exact) or len(set(self.uncertain)) != len(
            self.uncertain
        ):
            raise ValueError("equivalents repeat a passage")
        if set(self.exact) & set(self.uncertain):
            raise ValueError("a passage cannot be both an exact and an uncertain equivalent")
        return self


class Provenance(_Strict):
    origin: str  # "build-pairs" or "carryover:<name>"
    corpus_version: str
    corpus_sha256: str = Field(pattern=_SHA256)
    preparation_version: str
    preparation_manifest_sha256: str = Field(pattern=_SHA256)
    provider: str
    model: str
    prompt_version: str
    prompt_sha256: str = Field(pattern=_SHA256)
    validator_version: str
    generation_params_sha256: str = Field(pattern=_SHA256)


class TrainingPair(_Strict):
    contract_version: Literal[1]
    pair_id: str = Field(min_length=1)
    split: Literal["train", "validation"]
    query: str = Field(min_length=1)
    positive_passage_id: str
    group_id: str
    split_unit: str
    question_type: str
    answer: str
    evidence: str
    max_evaluation_similarity: float = Field(ge=0, le=1)
    ticker: str
    fiscal_year: str
    source: SourceInfo
    equivalents: Equivalents
    provenance: Provenance

    @property
    def positive_ids(self) -> tuple[str, ...]:
        """The positive and every exact equivalent: all hold the evidence."""
        return (self.positive_passage_id, *self.equivalents.exact)


def source_info(source: SourcePassage, visible_chars: int) -> dict[str, Any]:
    return {
        "ticker": source.ticker,
        "company": source.company,
        "fiscal_year": source.fiscal_year,
        "form_type": source.form_type,
        "filing_date": source.filing_date,
        "report_date": source.report_date,
        "section": source.section,
        "item": source.item,
        "visible_chars": visible_chars,
    }


def make_training_pair(
    *,
    pair_id: str,
    split: str,
    candidate: Mapping[str, str],
    positive: str,
    prepared: PreparedData,
    passages: Mapping[str, Passage],
    similarity: float,
    equivalents: Mapping[str, list[str]],
    provider: str,
    model: str,
    prompt_version: str,
    prompt_sha: str,
    validator_version: str,
    params_sha256: str,
    origin: str,
) -> dict[str, Any]:
    """A contract record (as a dict) for ``candidate`` on ``positive`` in ``prepared``."""
    row = prepared.rows[positive]
    manifest = prepared.manifest
    record = {
        "contract_version": PAIR_CONTRACT_VERSION,
        "pair_id": pair_id,
        "split": split,
        "query": candidate["question"],
        "positive_passage_id": positive,
        "group_id": row["group_id"],
        "split_unit": prepared.split_units()[positive],
        "question_type": candidate["type"],
        "answer": candidate["answer"],
        "evidence": candidate["evidence"],
        "max_evaluation_similarity": similarity,
        "ticker": row["ticker"],
        "fiscal_year": row["fiscal_year"],
        "source": source_info(prepared.source_passage(positive, passages), row["visible_chars"]),
        "equivalents": {
            "exact": list(equivalents["exact"]),
            "uncertain": list(equivalents["uncertain"]),
            "method": EQUIVALENCE_METHOD,
            "index_digest": manifest["reuse"]["index"]["digest"],
        },
        "provenance": {
            "origin": origin,
            "corpus_version": manifest["corpus"]["version"],
            "corpus_sha256": manifest["corpus"]["sha256"],
            "preparation_version": manifest["version"],
            "preparation_manifest_sha256": prepared.manifest_sha256,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha,
            "validator_version": validator_version,
            "generation_params_sha256": params_sha256,
        },
    }
    return TrainingPair.model_validate(record).model_dump(mode="json")


def contract_problems(
    record: Mapping[str, Any], prepared: PreparedData, passages: Mapping[str, Passage]
) -> list[str]:
    """Every way ``record`` breaks the contract or disagrees with ``prepared`` (empty if none)."""
    where = str(record.get("pair_id", "?"))
    try:
        pair = TrainingPair.model_validate(record)
    except ValidationError as exc:
        fields = sorted({".".join(str(x) for x in e["loc"]) for e in exc.errors()})
        return [f"{where}: not a contract-{PAIR_CONTRACT_VERSION} pair ({', '.join(fields[:6])})"]
    problems: list[str] = []
    rows, manifest = prepared.rows, prepared.manifest
    row = rows.get(pair.positive_passage_id)
    if row is None or pair.positive_passage_id not in passages:
        return [f"{where}: positive {pair.positive_passage_id} is not in the corpus"]
    if row["split"] != pair.split:
        problems.append(f"{where}: a {pair.split} pair's positive is in {row['split']}")
    if pair.group_id != row["group_id"]:
        problems.append(
            f"{where}: overlap group {pair.group_id} is not the positive's ({row['group_id']})"
        )
    if pair.split_unit != prepared.split_units()[pair.positive_passage_id]:
        problems.append(f"{where}: split unit {pair.split_unit} is not the positive's")
    if (pair.ticker, pair.fiscal_year) != (row["ticker"], row["fiscal_year"]):
        problems.append(f"{where}: company or fiscal year differs from the positive's")
    if row["split"] in ("train", "validation"):
        expected = source_info(
            prepared.source_passage(pair.positive_passage_id, passages), row["visible_chars"]
        )
        if pair.source.model_dump() != expected:
            problems.append(f"{where}: source metadata differs from the preparation")
    if pair.question_type not in QUESTION_TYPES:
        problems.append(f"{where}: unknown question type {pair.question_type!r}")
    eq = pair.equivalents
    reuse = manifest.get("reuse")
    if reuse is None:
        problems.append(
            f"{where}: the preparation has no evidence-reuse index, so equivalents cannot be trusted"
        )
    elif eq.index_digest != reuse["index"]["digest"]:
        problems.append(f"{where}: equivalents were computed under another evidence-reuse index")
    if eq.method != EQUIVALENCE_METHOD:
        problems.append(
            f"{where}: equivalents were found by {eq.method}, not {EQUIVALENCE_METHOD}; re-check the pair"
        )
    for pid in (*eq.exact, *eq.uncertain):
        other = rows.get(pid)
        if other is None:
            problems.append(f"{where}: equivalent {pid} is not in the corpus")
        elif pid == pair.positive_passage_id:
            problems.append(f"{where}: the positive is listed as its own equivalent")
        elif other["split"] != pair.split:
            problems.append(f"{where}: equivalent {pid} is in {other['split']}, not {pair.split}")
    prov = pair.provenance
    if (prov.corpus_version, prov.corpus_sha256) != (
        manifest["corpus"]["version"],
        manifest["corpus"]["sha256"],
    ):
        problems.append(f"{where}: built from another corpus")
    if (prov.preparation_version, prov.preparation_manifest_sha256) != (
        manifest["version"],
        prepared.manifest_sha256,
    ):
        problems.append(f"{where}: built against another preparation")
    if prov.prompt_version not in PROMPTS:
        problems.append(f"{where}: unknown prompt version {prov.prompt_version!r}")
    elif prov.prompt_sha256 != prompt_sha256(prov.prompt_version):
        problems.append(f"{where}: prompt SHA-256 does not match {prov.prompt_version}")
    if prov.validator_version != VALIDATOR_VERSION:
        problems.append(
            f"{where}: checked by {prov.validator_version}, not {VALIDATOR_VERSION}; rebuild or re-check it"
        )
    return problems


__all__ = [
    "EQUIVALENCE_METHOD",
    "EQUIVALENCE_METHODS",
    "PAIR_CONTRACT_VERSION",
    "Equivalents",
    "Provenance",
    "SourceInfo",
    "TrainingPair",
    "contract_problems",
    "make_training_pair",
    "source_info",
]
