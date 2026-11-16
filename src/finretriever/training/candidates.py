"""Training candidates: pairs accepted by an AI quality review, re-checked, in a versioned manifest.

A candidate dataset (``build_candidates``, configured by ``CandidatesConfig``) gathers the pairs of
named sources, each with a quality-decision file (``carryover.load_quality_decisions``): the
pairs this preparation built (``build-pairs``), or those of an earlier preparation, re-checked
here. Every source pair is logged with its outcome, and a pair is included only if all of these
hold (every reason it fails is recorded):

* **provenance**: its pair file matches its pairs manifest; the generation run, continuations and
  records that manifest names still have their recorded SHA-256 (they are read, never written);
  the pair's question, type, answer and evidence are an accepted candidate of its passage's latest
  generation record, under the parameters and prompt the pair names;
* **quality**: one decision, bound to its passage and the SHA-256 of its question, accepts it.
  ``revise`` and ``reject`` exclude it, and so does a missing decision;
* the current candidate checks (``VALIDATOR_VERSION``) pass on its positive, which is still a
  passage of its split;
* **evidence** (``LOCATOR_VERSION``): no blocked passage holds it and no other-split passage's
  visible text holds it; its equivalents are recomputed;
* its query does not overlap an evaluation question (lexical) and repeats no included query;
* with ``semantic_screening`` (a record of ``semantic_screen`` in this preparation), its question was
  screened and is not flagged as a paraphrase of a permitted evaluation question text. Without it,
  nothing changes: the datasets built before the screen existed rebuild identically.

Included pairs are contract records (``pair_contract``) carrying their source's generation
provenance, written once to ``<preparation>/candidates/<version>/``: ``<split>_candidates.jsonl``,
``candidates_log.jsonl`` (every source pair, included or not, with its reasons) and
``manifest.json`` (inputs and their SHA-256, counts by company, fiscal year, question type and
passage kind).

These are synthetic pairs accepted by an AI review: not human-validated, not gold labels, and
never evaluation data.

``load_candidates`` reopens a dataset for a trainer and fails closed (``TrainingDataError``)
unless its files, its preparation and evidence index, its sources and decision files are those
recorded, and every pair passes ``training_data.verify_pairs`` again. It reads no evaluation
file: the lexical evaluation check ran at build time and is bound by the manifest.
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
from finretriever.evaluation.provenance import sha256_file
from finretriever.training.carryover import load_quality_decisions, question_sha256
from finretriever.training.data_prep import DataPrepConfig, PreparedData
from finretriever.training.evidence_reuse import LOCATOR_VERSION
from finretriever.training.pair_contract import (
    PAIR_CONTRACT_VERSION,
    TrainingPair,
    contract_problems,
    make_training_pair,
)
from finretriever.training.pair_quality import (
    GENERATION_DIRNAME,
    PAIR_FILES,
    PAIRS_DIRNAME,
    PAIRS_META_FILE,
    PairQualityError,
    _max_similarity,
    aliases_for,
    equivalent_ids,
    evidence_leak,
    evidence_locator,
    read_pairs,
)
from finretriever.training.query_generation import (
    VALIDATOR_VERSION,
    check_candidate,
    continuations_path,
    latest_records,
    prompt_sha256,
    question_key,
    read_records,
    records_path,
    run_path,
)
from finretriever.training.sampling_plan import passage_kind
from finretriever.training.training_data import TrainingDataError, verify_pairs

CANDIDATES_DIRNAME = "candidates"
CANDIDATE_FILES = {"train": "train_candidates.jsonl", "validation": "validation_candidates.jsonl"}
LOG_FILE = "candidates_log.jsonl"
MANIFEST_FILE = "manifest.json"
FORMAT_VERSION = 1
NOTICE = (
    "Synthetic query-passage pairs written by an LLM, accepted by an AI quality review and re-checked against "
    "this preparation. They are training candidates: not human-validated, not gold labels, never evaluation data."
)
_NAME = r"^[a-z0-9][a-z0-9_.-]*$"


class CandidateError(ValueError):
    """Sources, decisions or provenance are inconsistent, or an output would be overwritten."""


class CandidateSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=_NAME)  # prefixes its pair IDs: "<name>:<pair_id>"
    decisions: str  # quality decisions (JSONL) for the source's pairs
    data_prep_config: str | None = None  # an earlier preparation; None: this one's own pairs


class CandidatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(pattern=_NAME)
    data_prep_config: str
    sources: list[CandidateSource] = Field(min_length=1)
    semantic_screening: str | None = Field(default=None, pattern=_NAME)  # a screening record of this preparation

    @model_validator(mode="after")
    def _unique(self) -> CandidatesConfig:
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names repeat")
        return self


def load_candidates_config(path: str | Path) -> tuple[CandidatesConfig, str]:
    path = Path(path)
    try:
        return CandidatesConfig.model_validate(yaml.safe_load(path.read_text("utf-8"))), sha256_file(
            path
        )
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        raise CandidateError(f"{path} is not a candidates config ({exc})") from None


def candidates_dir(prepared: PreparedData, version: str) -> Path:
    return prepared.directory / CANDIDATES_DIRNAME / version


@dataclass(frozen=True)
class SourcePreparation:
    source: CandidateSource
    prepared: PreparedData  # where its pairs were built


@dataclass(frozen=True)
class CandidateSet:
    pairs: dict[str, list[dict[str, Any]]]
    log: list[dict[str, Any]]
    manifest: dict[str, Any]


def _generation(preparation_dir: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Latest generation records by passage of every split the pairs manifest names, after
    checking the run, continuations and records files against their recorded SHA-256."""
    directory = preparation_dir / GENERATION_DIRNAME
    out: dict[str, Any] = {}
    for split, run in meta["generation"]["runs"].items():
        files = {
            "run_sha256": run_path(directory, split),
            "records_sha256": records_path(directory, split),
            "continuations_sha256": continuations_path(directory, split),
        }
        for field, path in files.items():
            actual = sha256_file(path) if path.is_file() else ""
            if actual != run[field]:
                raise CandidateError(
                    f"{path} is not the file its pairs were built from; the provenance cannot be verified"
                )
        out[split] = {
            "records": latest_records(read_records(files["records_sha256"])),
            "params": list(run["params_sha256"]),
        }
    return out


def _clean(text: Any) -> str:
    return " ".join(str(text).split())


def _generation_fields(pair: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    """The generation provenance a source pair names (contract pairs, or the flat format-1 fields)."""
    prov = pair.get("provenance")
    gen = meta["generation"]
    if prov:
        return {
            "provider": prov["provider"],
            "model": prov["model"],
            "prompt_version": prov["prompt_version"],
            "prompt_sha256": prov["prompt_sha256"],
            "params_sha256": prov["generation_params_sha256"],
        }
    return {
        "provider": pair.get("provider", gen["provider"]),
        "model": pair.get("model", gen["model"]),
        "prompt_version": pair.get("prompt_version", gen["prompt_version"]),
        "prompt_sha256": gen["prompt_sha256"],
        "params_sha256": pair.get("generation_params_sha256"),
    }


def _provenance_problem(
    pair: Mapping[str, Any], fields: Mapping[str, Any], generation: Mapping[str, Any]
) -> str | None:
    split = generation.get(pair["split"])
    if split is None:
        return "provenance_no_generation_run"
    record = split["records"].get(pair["positive_passage_id"])
    if record is None or record["status"] != "done":
        return "provenance_no_generation_record"
    if fields["params_sha256"] not in split["params"] or record.get(
        "params_sha256", split["params"][0]
    ) != fields["params_sha256"]:
        return "provenance_parameters_differ"
    if record.get("prompt_version", fields["prompt_version"]) != fields["prompt_version"]:
        return "provenance_prompt_differs"
    if prompt_sha256(fields["prompt_version"]) != fields["prompt_sha256"]:
        return "provenance_prompt_sha256_differs"
    wanted = tuple(
        _clean(pair[k]) for k in ("query", "question_type", "answer", "evidence")
    )
    accepted = {
        tuple(_clean(c.get(k, "")) for k in ("question", "type", "answer", "evidence"))
        for c in record["accepted"]
    }
    return None if wanted in accepted else "provenance_not_an_accepted_candidate"


def build_candidates(
    config: CandidatesConfig,
    *,
    config_path: str,
    config_sha256: str,
    target: PreparedData,
    target_config: DataPrepConfig,
    target_config_sha256: str,
    sources: Sequence[SourcePreparation],
    passages: Mapping[str, Passage],
    evaluation_questions: Sequence[Query],
    screening: Mapping[str, Any] | None = None,
) -> CandidateSet:
    """Check every pair of every source against ``target``; see the module docstring. ``screening`` is the
    ``semantic_screen.read_screening`` record ``config.semantic_screening`` names."""
    if (screening is None) != (config.semantic_screening is None):
        raise CandidateError("a semantic screening is used if and only if the candidates config names one")
    try:
        locator = evidence_locator(target, passages)
    except PairQualityError as exc:
        raise CandidateError(str(exc)) from None
    settings = target_config.generation
    aliases = aliases_for(passages, settings.company_aliases)
    approved = {
        split: {p.passage_id: p for p in target.source_passages(split, passages, min_words=1)}
        for split in PAIR_FILES
    }
    threshold = target_config.leakage.lexical_overlap_threshold
    pairs: dict[str, list[dict[str, Any]]] = {split: [] for split in PAIR_FILES}
    log: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_info = []
    for item in sources:
        src, prepared = item.source, item.prepared
        if src.data_prep_config is None and prepared.directory.resolve() != target.directory.resolve():
            raise CandidateError(f"source {src.name} must be this preparation's own pairs")
        try:
            earlier = read_pairs(prepared)  # checked against its pairs manifest
        except PairQualityError as exc:
            raise CandidateError(f"source {src.name}: {exc}") from None
        pairs_dir = prepared.directory / PAIRS_DIRNAME
        meta = json.loads((pairs_dir / PAIRS_META_FILE).read_text(encoding="utf-8"))
        generation = _generation(prepared.directory, meta)
        decisions = load_quality_decisions(src.decisions)
        listed = {p["pair_id"] for split in PAIR_FILES for p in earlier[split]}
        if unknown := sorted(set(decisions) - listed):
            raise CandidateError(
                f"{src.decisions} decides pairs {unknown[:3]} that {pairs_dir} does not hold"
            )
        own = prepared.directory.resolve() == target.directory.resolve()
        for split in PAIR_FILES:
            for pair in earlier[split]:
                pid, pair_id = pair["positive_passage_id"], pair["pair_id"]
                question_hash = question_sha256(pair["query"])
                decision = decisions.get(pair_id)
                if decision is not None and (
                    decision["passage_id"] != pid or decision["question_sha256"] != question_hash
                ):
                    raise CandidateError(
                        f"the decision for {src.name}:{pair_id} names another passage or question; decisions "
                        "never move between questions"
                    )
                reasons: list[str] = []
                fields = _generation_fields(pair, meta)
                if problem := _provenance_problem(pair, fields, generation):
                    reasons.append(problem)
                if decision is None:
                    reasons.append("no_quality_decision")
                elif decision["decision"] != "accept":
                    reasons.append(f"quality_{decision['decision']}")
                now = target.rows[pid]["split"] if pid in target.rows else "missing"
                if now != split:
                    reasons.append(f"positive_now_{now}")
                raw = {
                    "question": pair["query"],
                    "type": pair["question_type"],
                    "answer": pair["answer"],
                    "evidence": pair["evidence"],
                }
                source_passage = approved.get(now, {}).get(pid)
                if source_passage is not None:
                    _, reason = check_candidate(raw, source_passage, settings, aliases)
                    if reason is not None:
                        reasons.append(f"validator_{reason}")
                equivalents = locator.locate(pair["evidence"], pair["answer"], pair["ticker"], pid)
                leak = evidence_leak(equivalents, target.rows, split)
                if leak:
                    reasons.append(leak)
                similarity = _max_similarity(evaluation_questions, pair["query"])
                if similarity >= threshold:
                    reasons.append("evaluation_overlap")
                screened = None
                if screening is not None:
                    screened = screening["rows"].get(question_hash)
                    if screened is None:
                        reasons.append("not_semantically_screened")
                    elif screened["flagged"]:
                        reasons.append("semantic_evaluation_overlap")
                key = question_key(pair["query"])
                if key in seen:
                    reasons.append("duplicate_query")
                ids = (
                    equivalent_ids(equivalents, target.rows, split)
                    if now == split
                    else {"exact": [], "uncertain": []}
                )
                candidate_id = f"{src.name}:{pair_id}"
                record = None
                if not reasons:
                    try:
                        record = make_training_pair(
                            pair_id=candidate_id,
                            split=split,
                            candidate=raw,
                            positive=pid,
                            prepared=target,
                            passages=passages,
                            similarity=similarity,
                            equivalents=ids,
                            provider=fields["provider"],
                            model=fields["model"],
                            prompt_version=fields["prompt_version"],
                            prompt_sha=fields["prompt_sha256"],
                            validator_version=VALIDATOR_VERSION,
                            params_sha256=fields["params_sha256"],
                            origin="build-pairs" if own else f"carryover:{src.name}",
                        )
                    except (KeyError, ValueError) as exc:
                        reasons.append(f"not_a_contract_pair ({type(exc).__name__})")
                if record is not None and (problems := contract_problems(record, target, passages)):
                    raise CandidateError(f"{candidate_id}: {problems[:3]}")  # a bug, not a data issue
                if record is not None:
                    seen.add(key)
                    pairs[split].append(record)
                previous = pair.get("equivalents") or {}
                where_leaks = sorted(
                    e.passage_id
                    for e in equivalents
                    if (e.exact and target.rows[e.passage_id]["split"] == "blocked")
                    or (e.visible_exact and target.rows[e.passage_id]["split"] not in (split, "blocked"))
                )
                log.append(
                    {
                        "candidate_id": candidate_id,
                        "source": src.name,
                        "source_pair_id": pair_id,
                        "split": split,
                        "positive_passage_id": pid,
                        "question_sha256": question_hash,
                        "ticker": pair["ticker"],
                        "fiscal_year": pair["fiscal_year"],
                        "question_type": pair["question_type"],
                        "passage_kind": passage_kind(source_passage) if source_passage else None,
                        "quality": None
                        if decision is None
                        else {
                            "decision": decision["decision"],
                            "issues": list(decision["issues"]),
                            "reviewer": decision["reviewer"],
                            "gold": False,
                        },
                        "included": record is not None,
                        "reasons": reasons,
                        "evidence": {
                            "method": LOCATOR_VERSION,
                            "leak_passages": where_leaks,
                            "exact": len(ids["exact"]),
                            "uncertain": len(ids["uncertain"]),
                            "previous": None
                            if not previous
                            else {
                                "method": previous.get("method", "evidence-reuse-v1"),
                                "exact": len(previous.get("exact", [])),
                                "uncertain": len(previous.get("uncertain", [])),
                            },
                        },
                        "max_evaluation_similarity": similarity,
                        **({} if screening is None else {
                            "max_semantic_similarity": None if screened is None else screened["max_similarity"]}),
                    }
                )
        source_info.append(
            {
                "name": src.name,
                "data_prep_config": src.data_prep_config,
                "preparation": {
                    "directory": str(prepared.directory),
                    "version": prepared.manifest["version"],
                    "manifest_sha256": prepared.manifest_sha256,
                },
                "pairs_meta_sha256": sha256_file(pairs_dir / PAIRS_META_FILE),
                "pair_files_sha256": {n: sha256_file(pairs_dir / n) for n in PAIR_FILES.values()},
                "generation_runs": meta["generation"]["runs"],
                "decisions": {"path": src.decisions, "sha256": sha256_file(src.decisions)},
            }
        )
    question_files = [
        {
            "path": path,
            "sha256": sha256_file(path),
            "questions": len(evaluation_questions_in(path)),
        }
        for path in target_config.leakage.evaluation_question_files
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": "training_candidates",
        "version": config.version,
        "notice": NOTICE,
        "labels": "AI quality review (accept only); not human-validated; not gold",
        "config": {"path": config_path, "sha256": config_sha256},
        "corpus": target.manifest["corpus"],
        "preparation": {
            "data_prep_config": {"path": config.data_prep_config, "sha256": target_config_sha256},
            "directory": str(target.directory),
            "version": target.manifest["version"],
            "manifest_sha256": target.manifest_sha256,
            "evidence_index_digest": locator.index.digest(),
            "visible_text": target.manifest["visible_text"]["identity"],
        },
        "checks": {
            "pair_contract_version": PAIR_CONTRACT_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "equivalence_method": LOCATOR_VERSION,
            "evaluation_overlap": {"method": "word-set Jaccard", "threshold": threshold,
                                   "question_files": question_files},
            **({} if screening is None else {"semantic_overlap": {
                "screening": config.semantic_screening,
                "manifest_sha256": screening["manifest_sha256"],
                "method": screening["manifest"]["method"],
                "threshold": screening["manifest"]["threshold"],
                "encoder": screening["manifest"]["encoder"],
                "note": "consulted only here, when the candidates are built; the trainer never reads it",
            }}),
        },
        "sources": source_info,
        "stats": _stats(log),
    }
    return CandidateSet(pairs, log, manifest)


def evaluation_questions_in(path: str) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _stats(log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(r[field]) for r in rows).items()))

    included = [r for r in log if r["included"]]
    by_source = {}
    for name in dict.fromkeys(r["source"] for r in log):
        rows = [r for r in log if r["source"] == name]
        by_source[name] = {
            "pairs": len(rows),
            "included": sum(r["included"] for r in rows),
            "excluded_by_reason": dict(
                sorted(Counter(x for r in rows for x in r["reasons"]).items())
            ),
        }
    tickers = Counter(r["ticker"] for r in included)
    return {
        "source_pairs": len(log),
        "included": len(included),
        "included_by_split": counts(included, "split"),
        "by_source": by_source,
        "included_by_ticker": counts(included, "ticker"),
        "included_by_fiscal_year": counts(included, "fiscal_year"),
        "included_by_question_type": counts(included, "question_type"),
        "included_by_passage_kind": counts(included, "passage_kind"),
        "largest_company_share": round(max(tickers.values()) / len(included), 4) if included else 0.0,
        "included_with_exact_equivalents": sum(r["evidence"]["exact"] > 0 for r in included),
        "included_with_uncertain_equivalents": sum(r["evidence"]["uncertain"] > 0 for r in included),
    }


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows
    ).encode()


def write_candidates(target: PreparedData, candidates: CandidateSet) -> tuple[bool, Path]:
    """Write the dataset once; ``False`` if an identical copy exists. A different one is refused."""
    directory = candidates_dir(target, candidates.manifest["version"])
    data = {CANDIDATE_FILES[s]: _jsonl(candidates.pairs[s]) for s in CANDIDATE_FILES}
    data[LOG_FILE] = _jsonl(candidates.log)
    manifest = {
        **candidates.manifest,
        "files": {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(data.items())},
    }
    data[MANIFEST_FILE] = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    if directory.exists():
        present = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        if present == data:
            return False, directory
        raise CandidateError(f"{directory} already holds a different candidate dataset; use a new version")
    directory.mkdir(parents=True)
    for name in sorted(n for n in data if n != MANIFEST_FILE):
        (directory / name).write_bytes(data[name])
    (directory / MANIFEST_FILE).write_bytes(data[MANIFEST_FILE])  # last: a manifest means a complete set
    return True, directory


def read_candidates_manifest(prepared: PreparedData, version: str) -> tuple[dict[str, Any], str]:
    """The manifest of a written dataset and its SHA-256, after checking its files and bindings."""
    directory = candidates_dir(prepared, version)
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise TrainingDataError(f"no candidate dataset {version!r} at {directory}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "training_candidates" or manifest.get("format_version") != FORMAT_VERSION:
        raise TrainingDataError(f"{path} is not a format-{FORMAT_VERSION} candidate manifest")
    for name, digest in manifest["files"].items():
        if not (directory / name).is_file() or sha256_file(directory / name) != digest:
            raise TrainingDataError(f"{directory / name} is missing or does not match {MANIFEST_FILE}")
    prep = manifest["preparation"]
    if prep["manifest_sha256"] != prepared.manifest_sha256 or manifest["corpus"] != prepared.manifest["corpus"]:
        raise TrainingDataError(f"candidate dataset {version!r} was built against another preparation or corpus")
    if prepared.manifest.get("reuse", {}).get("index", {}).get("digest") != prep["evidence_index_digest"]:
        raise TrainingDataError(f"candidate dataset {version!r} was checked under another evidence-reuse index")
    checks = manifest["checks"]
    if (checks["equivalence_method"], checks["validator_version"]) != (LOCATOR_VERSION, VALIDATOR_VERSION):
        raise TrainingDataError(
            f"candidate dataset {version!r} was checked by {checks['equivalence_method']} and "
            f"{checks['validator_version']}, not {LOCATOR_VERSION} and {VALIDATOR_VERSION}; build a new version"
        )
    for source in manifest["sources"]:
        pairs_dir = Path(source["preparation"]["directory"]) / PAIRS_DIRNAME
        recorded = {
            pairs_dir / PAIRS_META_FILE: source["pairs_meta_sha256"],
            Path(source["decisions"]["path"]): source["decisions"]["sha256"],
            **{pairs_dir / n: d for n, d in source["pair_files_sha256"].items()},
        }
        for file, digest in recorded.items():
            if not file.is_file() or sha256_file(file) != digest:
                raise TrainingDataError(
                    f"candidate dataset {version!r}: {file} changed since it was built; build a new version"
                )
    return manifest, sha256_file(path)


def verify_candidate_provenance(prepared: PreparedData, version: str) -> dict[str, Any]:
    """Trace every candidate of a written dataset back to where it came from, again.

    After ``read_candidates_manifest``: each candidate must be its source pair (same query, type,
    answer, evidence, positive and split), accepted by a decision bound to that passage and
    question, under the generation provider, model, prompt and parameters its source records, and
    an accepted candidate of its passage's latest generation record, whose run, continuations and
    records files must still have their recorded SHA-256. The log must list exactly the included
    candidates. Raises ``CandidateError`` with every problem; returns counts.
    """
    manifest, _ = read_candidates_manifest(prepared, version)
    directory = candidates_dir(prepared, version)
    records = [
        json.loads(line)
        for name in CANDIDATE_FILES.values()
        for line in (directory / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log = [
        json.loads(line)
        for line in (directory / LOG_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = {}
    for source in manifest["sources"]:
        preparation_dir = Path(source["preparation"]["directory"])
        pairs_dir = preparation_dir / PAIRS_DIRNAME
        meta = json.loads((pairs_dir / PAIRS_META_FILE).read_text(encoding="utf-8"))
        if meta["generation"]["runs"] != source["generation_runs"]:
            raise CandidateError(f"source {source['name']}: its generation runs differ from the manifest's")
        pairs = {
            p["pair_id"]: p
            for name in PAIR_FILES.values()
            for p in map(json.loads, (pairs_dir / name).read_text(encoding="utf-8").splitlines())
        }
        sources[source["name"]] = (
            meta,
            pairs,
            _generation(preparation_dir, meta),
            load_quality_decisions(source["decisions"]["path"]),
        )
    problems: list[str] = []
    included = sorted(r["candidate_id"] for r in log if r["included"])
    if sorted(r["pair_id"] for r in records) != included:
        problems.append("the candidate files and the log disagree on which pairs were included")
    counts: Counter[str] = Counter()
    for record in records:
        where = record["pair_id"]
        name, _, source_pair_id = where.partition(":")
        if name not in sources:
            problems.append(f"{where}: no source {name!r} in the manifest")
            continue
        meta, pairs, generation, decisions = sources[name]
        pair = pairs.get(source_pair_id)
        if pair is None:
            problems.append(f"{where}: its source pair {source_pair_id} is not in the source pair files")
            continue
        for field in ("query", "question_type", "answer", "evidence", "positive_passage_id", "split"):
            if record[field] != pair[field]:
                problems.append(f"{where}: {field} differs from its source pair")
        decision = decisions.get(source_pair_id)
        if (
            decision is None
            or decision["decision"] != "accept"
            or decision["passage_id"] != pair["positive_passage_id"]
            or decision["question_sha256"] != question_sha256(pair["query"])
        ):
            problems.append(f"{where}: no accepting decision bound to its passage and question")
        fields = _generation_fields(pair, meta)
        prov = record["provenance"]
        recorded = {
            "provider": prov["provider"],
            "model": prov["model"],
            "prompt_version": prov["prompt_version"],
            "prompt_sha256": prov["prompt_sha256"],
            "params_sha256": prov["generation_params_sha256"],
        }
        if recorded != fields:
            problems.append(f"{where}: its generation provenance differs from its source pair's")
        if problem := _provenance_problem(pair, fields, generation):
            problems.append(f"{where}: {problem}")
        counts[name] += 1
    if problems:
        raise CandidateError(f"{len(problems)} provenance problems: {problems[:5]}")
    return {"candidates": len(records), "by_source": dict(sorted(counts.items()))}


def load_candidates(
    prepared: PreparedData,
    passages: Mapping[str, Passage],
    config: DataPrepConfig,
    version: str,
    split: str,
) -> list[TrainingPair]:
    """The candidates of ``split`` a trainer may use, re-verified; reads no evaluation file."""
    read_candidates_manifest(prepared, version)
    path = candidates_dir(prepared, version) / CANDIDATE_FILES[split]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return verify_pairs(records, prepared, passages, config, split, evaluation_questions=None)


__all__ = [
    "CANDIDATES_DIRNAME",
    "CANDIDATE_FILES",
    "LOG_FILE",
    "MANIFEST_FILE",
    "NOTICE",
    "CandidateError",
    "CandidateSet",
    "CandidateSource",
    "CandidatesConfig",
    "SourcePreparation",
    "build_candidates",
    "candidates_dir",
    "load_candidates",
    "load_candidates_config",
    "read_candidates_manifest",
    "verify_candidate_provenance",
    "write_candidates",
]
