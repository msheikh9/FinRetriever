"""Evaluation isolation of a fine-tuned checkpoint, shown from files before any model is loaded.

A trainer checkpoint records what it was trained on (``metadata.json`` ``inputs``): the corpus,
the preparation (its config and manifest SHA-256) and the training-candidate dataset (version and
manifest SHA-256). :func:`checkpoint_isolation` opens those records, checks each against the
SHA-256 the checkpoint recorded, and fails closed (:class:`IsolationError`) unless

* the checkpoint was trained on the corpus being evaluated (SHA-256 and version), so a passage ID
  names the same text in training and in evaluation;
* every passage the evaluation set judges is on the preparation's evaluation blocklist, so no
  training pair could use it as a positive, an equivalent or an in-batch negative;
* no candidate of the recorded dataset (every pair a run could draw from, a superset of those it
  used) has a judged passage as its positive or as an exact or uncertain equivalent;
* no training question repeats an evaluation question (word-set Jaccard at the preparation's
  leakage threshold);
* for a checkpoint chosen on a validation set (its ``inputs`` name one), that set was the one
  recorded and shares no judged passage and no question with the evaluation set, since its
  questions selected the checkpoint (``validation`` in the record, with ``VALIDATION_METHOD``).

Only passage IDs, file hashes and question texts are read. Grades matter only as far as which
passages are judged, so no evaluation result selects or shapes a checkpoint.

**Isolation modes** (``mode``; decision D8 of 2026-11-12, docs/gold_set_v1_protocol.md §4):

* ``strict`` (the default) is everything above. An evaluation set built after the checkpoint's preparation can never
  pass it: the blocklist was frozen before that set's passages existed.
* ``post_training_eligible``, for a set built after training that ranks only the eligible corpus (the training and
  validation passages removed before ranking). The blocklist requirement is replaced by a recorded count. Any judged
  passage that a training or validation pair uses is still refused, and so is a checkpoint trained with mined hard
  negatives: with in-batch negatives only, the candidates' positives and equivalents are every passage training
  showed the model.
* ``post_training_full_corpus``, for the same set ranked on the whole corpus, as a sensitivity analysis. Judged
  passages that a training or validation pair uses are counted, not refused, and the record is marked
  ``potentially_contaminated``.

In every mode a repeated question (training or validation against evaluation) and a different corpus are refused.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from finretriever.data.schemas import Qrel, Query
from finretriever.eval_set.qrels_builder import find_gold_overlap
from finretriever.evaluation.provenance import CorpusIdentity, sha256_file
from finretriever.retrieval.checkpoint import LocalCheckpoint

ISOLATION_METHOD = (
    "From the checkpoint's recorded inputs, each checked against its recorded SHA-256: the training corpus is the "
    "evaluated corpus; every judged passage is on the preparation's evaluation blocklist; no candidate of the "
    "recorded training-candidate dataset uses a judged passage as positive or equivalent; no training question "
    "repeats an evaluation question (word-set Jaccard at the preparation's leakage threshold)."
)
VALIDATION_METHOD = (
    "The validation set that chose the checkpoint, checked against its recorded SHA-256: no validation question's "
    "positive or equivalent is a judged passage, and no validation question repeats an evaluation question "
    "(word-set Jaccard at the preparation's leakage threshold)."
)
_DEFAULT_THRESHOLD = 0.8
ISOLATION_STRICT = "strict"
ISOLATION_POST_TRAINING_ELIGIBLE = "post_training_eligible"
ISOLATION_POST_TRAINING_FULL = "post_training_full_corpus"
ISOLATION_MODES = (ISOLATION_STRICT, ISOLATION_POST_TRAINING_ELIGIBLE, ISOLATION_POST_TRAINING_FULL)


class IsolationError(ValueError):
    """A checkpoint's training data cannot be shown to be apart from the evaluation set."""


def _check(path: Path, recorded: str | None, what: str, problems: list[str]) -> bool:
    if not path.is_file():
        problems.append(f"{what} {path} is missing")
        return False
    if recorded is not None and sha256_file(path) != recorded:
        problems.append(f"{what} {path} does not match the SHA-256 the checkpoint recorded; it changed since training")
        return False
    return True


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint_isolation(
    checkpoint: LocalCheckpoint,
    *,
    corpus: CorpusIdentity,
    queries: Sequence[Query],
    qrels: Sequence[Qrel],
    mode: str = ISOLATION_STRICT,
) -> dict[str, Any]:
    """The isolation record of ``checkpoint`` against an evaluation set; raises :class:`IsolationError` with every
    problem found (see the module docstring, including the isolation modes)."""
    if mode not in ISOLATION_MODES:
        raise ValueError(f"unknown isolation mode {mode!r}; use one of {list(ISOLATION_MODES)}")
    post_training = mode != ISOLATION_STRICT
    problems: list[str] = []
    inputs = checkpoint.metadata.get("inputs") or {}
    trained = inputs.get("corpus") or {}
    if (trained.get("sha256"), trained.get("version")) != (corpus.corpus_sha256, corpus.corpus_version):
        problems.append(
            f"trained on corpus {trained.get('version')} ({str(trained.get('sha256'))[:12]}...), evaluated on "
            f"{corpus.corpus_version} ({corpus.corpus_sha256[:12]}...)"
        )
    preparation = inputs.get("preparation") or {}
    prep_config = preparation.get("data_prep_config") or {}
    candidates = inputs.get("candidates") or {}
    record: dict[str, Any] = {"method": ISOLATION_METHOD, "checkpoint": checkpoint.model_id}
    if post_training:  # the strict record is unchanged; the post-training modes say which they are
        record["mode"] = mode
        objective = checkpoint.metadata.get("objective") or {}
        if mined := sorted(k for k, v in objective.items() if "hard_negative" in k and v):
            problems.append(f"the checkpoint was trained with mined hard negatives ({mined}); post-training isolation "
                            "assumes in-batch negatives only")
    judged = {q.passage_id for q in qrels}
    record["judged_passages"] = len(judged)
    config_path = Path(str(prep_config.get("path")))
    if not prep_config or not _check(config_path, prep_config.get("sha256"), "preparation config", problems):
        raise IsolationError(f"{checkpoint.model_id}: {'; '.join(problems) or 'no preparation recorded'}")
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    threshold = float((settings.get("leakage") or {}).get("lexical_overlap_threshold", _DEFAULT_THRESHOLD))
    prepared = Path(str(settings.get("output_dir")))
    manifest_path = prepared / "manifest.json"
    if _check(manifest_path, preparation.get("manifest_sha256"), "preparation manifest", problems):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record["preparation"] = {
            "version": manifest.get("version"),
            "config": {"path": str(config_path), "sha256": prep_config.get("sha256")},
            "manifest": {"path": str(manifest_path), "sha256": preparation.get("manifest_sha256")},
            "blocklist_sources": [s.get("path") for s in (manifest.get("blocklist") or {}).get("sources", [])],
        }
        blocklist = prepared / "blocklist.jsonl"
        if _check(blocklist, (manifest.get("files") or {}).get("blocklist.jsonl"), "blocklist", problems):
            blocked = {row["passage_id"] for row in _jsonl(blocklist)}
            unblocked = sorted(judged - blocked)
            record["blocklist"] = {"path": str(blocklist), "passages": len(blocked),
                                   "judged_passages_blocked": len(judged) - len(unblocked)}
            if unblocked and not post_training:
                problems.append(
                    f"{len(unblocked)} judged passages are not on the training blocklist (e.g. {unblocked[0]}): the "
                    "checkpoint may have trained on them"
                )
    if not candidates.get("version"):
        problems.append("the checkpoint records no training-candidate dataset")
    else:
        directory = prepared / "candidates" / str(candidates["version"])
        cand_manifest = directory / "manifest.json"
        if _check(cand_manifest, candidates.get("manifest_sha256"), "candidate manifest", problems):
            files = json.loads(cand_manifest.read_text(encoding="utf-8")).get("files") or {}
            train_file = directory / "train_candidates.jsonl"
            if _check(train_file, files.get("train_candidates.jsonl"), "training candidates", problems):
                rows = _jsonl(train_file)
                used: dict[str, str] = {}
                for row in rows:
                    eq = row.get("equivalents") or {}
                    for pid in (row["positive_passage_id"], *eq.get("exact", ()), *eq.get("uncertain", ())):
                        used.setdefault(pid, row["pair_id"])
                if (shared := sorted(set(used) & judged)) and mode != ISOLATION_POST_TRAINING_FULL:
                    problems.append(
                        f"training pair {used[shared[0]]} uses judged passage {shared[0]} ({len(shared)} judged "
                        "passages used in all)"
                    )
                questions = [row["query"] for row in rows]
                overlaps = find_gold_overlap(list(queries), questions, threshold=0.0)
                top = max((o.similarity for o in overlaps), default=0.0)
                if repeated := [o for o in overlaps if o.similarity >= threshold]:
                    problems.append(
                        f"training question {rows[repeated[0].index]['pair_id']} repeats evaluation question "
                        f"{repeated[0].query_id} (Jaccard {repeated[0].similarity})"
                    )
                record["candidates"] = {
                    "version": candidates["version"],
                    "manifest": {"path": str(cand_manifest), "sha256": candidates.get("manifest_sha256")},
                    "pairs_available": len(rows),
                    "pairs_trained": (inputs.get("pairs") or {}).get("count"),
                    "passages_used": len(used),
                    "passages_used_and_judged": len(set(used) & judged),
                    "max_question_similarity": top,
                    "threshold": threshold,
                }
    if validation := inputs.get("validation"):
        record["validation"] = _validation_isolation(prepared, validation, judged, queries, threshold, problems,
                                                     record_shared=mode == ISOLATION_POST_TRAINING_FULL)
        record["validation_method"] = VALIDATION_METHOD
    if mode == ISOLATION_POST_TRAINING_FULL:
        trained = (record.get("candidates") or {}).get("passages_used_and_judged") or 0
        validated = (record.get("validation") or {}).get("passages_used_and_judged") or 0
        record["potentially_contaminated"] = bool(trained or validated)
        record["contamination"] = {"judged_training_passages": trained, "judged_validation_passages": validated}
    if problems:
        raise IsolationError(f"{checkpoint.model_id} is not isolated from the evaluation set: {'; '.join(problems)}")
    record["isolated"] = True
    return record


def _validation_isolation(
    prepared: Path,
    validation: Mapping[str, Any],
    judged: set[str],
    queries: Sequence[Query],
    threshold: float,
    problems: list[str],
    *,
    record_shared: bool = False,
) -> dict[str, Any] | None:
    """The selecting validation set against the evaluation set (see the module docstring)."""
    directory = prepared / "validation_sets" / str(validation.get("version"))
    manifest = directory / "manifest.json"
    if not _check(manifest, validation.get("manifest_sha256"), "validation manifest", problems):
        return None
    files = json.loads(manifest.read_text(encoding="utf-8")).get("files") or {}
    pairs_file = directory / "validation_pairs.jsonl"
    if not _check(pairs_file, files.get("validation_pairs.jsonl"), "validation pairs", problems):
        return None
    rows = _jsonl(pairs_file)
    used: dict[str, str] = {}
    for row in rows:
        eq = row.get("equivalents") or {}
        for pid in (row["positive_passage_id"], *eq.get("exact", ()), *eq.get("uncertain", ())):
            used.setdefault(pid, row["pair_id"])
    if (shared := sorted(set(used) & judged)) and not record_shared:
        problems.append(
            f"validation question {used[shared[0]]} uses judged passage {shared[0]} ({len(shared)} in all): the "
            "checkpoint was selected with an evaluation passage"
        )
    overlaps = find_gold_overlap(list(queries), [row["query"] for row in rows], threshold=0.0)
    if repeated := [o for o in overlaps if o.similarity >= threshold]:
        problems.append(
            f"validation question {rows[repeated[0].index]['pair_id']} repeats evaluation question "
            f"{repeated[0].query_id} (Jaccard {repeated[0].similarity})"
        )
    return {
        "version": validation.get("version"),
        "manifest": {"path": str(manifest), "sha256": validation.get("manifest_sha256")},
        "questions": len(rows),
        "passages_used": len(used),
        "passages_used_and_judged": len(set(used) & judged),
        "max_question_similarity": max((o.similarity for o in overlaps), default=0.0),
        "threshold": threshold,
    }


def isolation_summary(record: Mapping[str, Any]) -> str:
    """One line for a report."""
    blocklist, cand = record.get("blocklist") or {}, record.get("candidates") or {}
    line = (
        f"{blocklist.get('judged_passages_blocked')} of {record.get('judged_passages')} judged passages on the "
        f"training blocklist; {cand.get('passages_used_and_judged')} of {cand.get('passages_used')} passages of the "
        f"{cand.get('pairs_available')} training candidates ({cand.get('version')}) judged; highest training-question "
        f"similarity to an evaluation question {cand.get('max_question_similarity')} (threshold {cand.get('threshold')})"
    )
    if validation := record.get("validation"):
        line += (
            f"; selecting validation set {validation['version']}: {validation['passages_used_and_judged']} of "
            f"{validation['passages_used']} passages of its {validation['questions']} questions judged, highest "
            f"question similarity {validation['max_question_similarity']}"
        )
    return line


__all__ = [
    "ISOLATION_METHOD",
    "ISOLATION_MODES",
    "ISOLATION_POST_TRAINING_ELIGIBLE",
    "ISOLATION_POST_TRAINING_FULL",
    "ISOLATION_STRICT",
    "VALIDATION_METHOD",
    "IsolationError",
    "checkpoint_isolation",
    "isolation_summary",
]
