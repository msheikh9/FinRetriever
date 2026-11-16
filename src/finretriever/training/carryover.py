"""Carrying pairs from an earlier preparation into a newer one, explicitly and after every check.

A newer preparation can block more passages or split them differently, so pairs built against an
earlier one are never silently rebound to it. ``carry_over`` re-checks each earlier pair against
the newer preparation and writes the outcome beside it, under ``<preparation>/carryover/``, as
development training candidates; it never adds them to the preparation's pairs.

A pair is eligible only if all of these hold:

* a quality decision (``load_quality_decisions``) accepts it. Decisions are bound to the pair by
  its passage and the SHA-256 of its exact question, so a decision cannot drift onto another
  question; any mismatch stops the carry-over. Questions are never edited;
* its positive is in the same split in the newer preparation (not blocked, not moved);
* the current candidate checks pass (``VALIDATOR_VERSION``);
* its evidence is held by no blocked passage and by no other-split passage's visible text
  (``finretriever.training.pair_quality.evidence_leak``), and its question does not overlap an
  evaluation question.

Each row also records the pair's equivalent passages in the newer preparation and how many
fiscal years repeat its evidence (a repeated annual disclosure when more than one).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finretriever.data.schemas import Passage, Query
from finretriever.evaluation.provenance import sha256_file
from finretriever.training.data_prep import DataPrepConfig, PreparedData
from finretriever.training.pair_quality import (
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
from finretriever.training.query_generation import VALIDATOR_VERSION, check_candidate

CARRYOVER_DIRNAME = "carryover"
QUALITY_DECISIONS = ("accept", "revise", "reject")
QUALITY_ISSUES = (
    "boilerplate",
    "misinterpreted_table_value",
    "incorrect_period",
    "weak_financial_relevance",
    "inaccurate_wording",
)
NOTICE = (
    "Development training candidates carried from an earlier preparation after the current leakage, split and "
    "candidate checks. They are not part of this preparation's pairs and are never evaluation data. Quality "
    "decisions come from an AI review, not human annotation, and are not gold labels."
)


class CarryOverError(ValueError):
    """Decisions or pairs are inconsistent, or an output would be overwritten; nothing is written."""


def question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def load_quality_decisions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Quality decisions by pair ID, validated; a malformed or duplicate row stops the load."""
    path = Path(path)
    if not path.is_file():
        raise CarryOverError(f"quality decisions {path} do not exist")
    out: dict[str, dict[str, Any]] = {}
    required = {
        "pair_id",
        "passage_id",
        "question_sha256",
        "decision",
        "issues",
        "reviewer",
        "gold",
    }
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        where = f"{path}:{line_no}"
        if missing := sorted(required - set(row)):
            raise CarryOverError(f"{where} lacks {missing}")
        if row["decision"] not in QUALITY_DECISIONS:
            raise CarryOverError(f"{where}: unknown decision {row['decision']!r}")
        if unknown := sorted(set(row["issues"]) - set(QUALITY_ISSUES)):
            raise CarryOverError(f"{where}: unknown issues {unknown}")
        if row["gold"] is not False:
            raise CarryOverError(f"{where}: review decisions are never gold labels")
        if row["decision"] != "accept" and not row.get("reason"):
            raise CarryOverError(f"{where}: a {row['decision']} decision needs a reason")
        if row["pair_id"] in out:
            raise CarryOverError(f"{where}: a second decision for {row['pair_id']}")
        out[row["pair_id"]] = row
    return out


@dataclass(frozen=True)
class CarryOver:
    rows: list[dict[str, Any]]
    meta: dict[str, Any]


def carry_over(
    *,
    name: str,
    source: PreparedData,
    target: PreparedData,
    target_config: DataPrepConfig,
    passages: Mapping[str, Passage],
    decisions: Mapping[str, Mapping[str, Any]],
    decisions_path: str | Path,
    evaluation_questions: Sequence[Query],
) -> CarryOver:
    """Re-check every pair of ``source`` against ``target``; see the module docstring."""
    if source.directory.resolve() == target.directory.resolve():
        raise CarryOverError("source and target are the same preparation")
    try:
        locator = evidence_locator(target, passages)
        earlier = read_pairs(source)
    except PairQualityError as exc:
        raise CarryOverError(str(exc)) from None
    settings = target_config.generation
    aliases = aliases_for(passages, settings.company_aliases)
    sources = {
        split: {p.passage_id: p for p in target.source_passages(split, passages, min_words=1)}
        for split in ("train", "validation")
    }
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        for pair in earlier[split]:
            pid, pair_id = pair["positive_passage_id"], pair["pair_id"]
            reasons: list[str] = []
            decision = decisions.get(pair_id)
            if decision is None:
                reasons.append("no_quality_decision")
            elif decision["passage_id"] != pid or decision["question_sha256"] != question_sha256(
                pair["query"]
            ):
                raise CarryOverError(
                    f"the decision for {pair_id} names another passage or question; decisions never move "
                    "between questions"
                )
            elif decision["decision"] != "accept":
                reasons.append(f"quality_{decision['decision']}")
            now = target.rows[pid]["split"]
            if now != split:
                reasons.append(f"positive_now_{now}")
            raw = {
                "question": pair["query"],
                "type": pair["question_type"],
                "answer": pair["answer"],
                "evidence": pair["evidence"],
            }
            if now in sources and (passage := sources[now].get(pid)) is not None:
                _, reason = check_candidate(raw, passage, settings, aliases)
                if reason is not None:
                    reasons.append(f"validator_{reason}")
            equivalents = locator.locate(pair["evidence"], pair["answer"], pair["ticker"], pid)
            if leak := evidence_leak(equivalents, target.rows, split):
                reasons.append(leak)
            similarity = _max_similarity(evaluation_questions, pair["query"])
            if similarity >= target_config.leakage.lexical_overlap_threshold:
                reasons.append("evaluation_overlap")
            years = {target.rows[pid]["fiscal_year"]} | {
                target.rows[e.passage_id]["fiscal_year"] for e in equivalents if e.exact
            }
            # Fields that describe the earlier preparation are not carried; its provenance is kept apart.
            carried = {
                k: v
                for k, v in pair.items()
                if k not in ("contract_version", "split_unit", "source", "provenance")
            }
            if "provenance" in pair:
                carried["source_provenance"] = pair["provenance"]
            rows.append(
                {
                    **carried,
                    "pair_id": f"{name}:{pair_id}",
                    "source_pair_id": pair_id,
                    "split": split,
                    "group_id": target.rows[pid]["group_id"],
                    "equivalents": (
                        equivalent_ids(equivalents, target.rows, split)
                        if now == split
                        else {"exact": [], "uncertain": []}
                    ),
                    "max_evaluation_similarity": similarity,
                    "quality": (
                        None
                        if decision is None
                        else {
                            "decision": decision["decision"],
                            "issues": list(decision["issues"]),
                            "reviewer": decision["reviewer"],
                        }
                    ),
                    "evidence_fiscal_years": len(years),
                    "validator_version": VALIDATOR_VERSION,
                    "eligible": not reasons,
                    "reasons": reasons,
                }
            )
    eligible = [r for r in rows if r["eligible"]]
    source_pairs = source.directory / PAIRS_DIRNAME
    meta = {
        "format_version": 1,
        "kind": "pair_carryover",
        "name": name,
        "notice": NOTICE,
        "source": {
            "directory": str(source.directory),
            "manifest_sha256": source.manifest_sha256,
            "pairs_meta_sha256": sha256_file(source_pairs / PAIRS_META_FILE),
            "pair_files_sha256": {n: sha256_file(source_pairs / n) for n in PAIR_FILES.values()},
        },
        "target": {
            "directory": str(target.directory),
            "manifest_sha256": target.manifest_sha256,
            "evidence_index_digest": locator.index.digest(),
        },
        "decisions": {"path": str(decisions_path), "sha256": sha256_file(decisions_path)},
        "validator_version": VALIDATOR_VERSION,
        "stats": {
            "pairs": len(rows),
            "eligible": len(eligible),
            "ineligible_by_reason": dict(
                sorted(Counter(r for row in rows for r in row["reasons"]).items())
            ),
            "eligible_by_ticker": dict(sorted(Counter(r["ticker"] for r in eligible).items())),
            "eligible_by_fiscal_year": dict(
                sorted(Counter(r["fiscal_year"] for r in eligible).items())
            ),
            "eligible_with_repeated_evidence": sum(
                r["evidence_fiscal_years"] > 1 for r in eligible
            ),
        },
    }
    return CarryOver(rows, meta)


def write_carry_over(target: PreparedData, carried: CarryOver) -> tuple[bool, Path]:
    """Write ``<target>/carryover/<name>.jsonl`` and its meta once; ``False`` if identical files exist."""
    directory = target.directory / CARRYOVER_DIRNAME
    name = carried.meta["name"]
    data = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for r in carried.rows
    ).encode()
    meta = {**carried.meta, "files": {f"{name}.jsonl": hashlib.sha256(data).hexdigest()}}
    meta_bytes = (json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    rows_path, meta_path = directory / f"{name}.jsonl", directory / f"{name}.meta.json"
    if rows_path.exists() or meta_path.exists():
        if (
            rows_path.exists()
            and meta_path.exists()
            and (
                rows_path.read_bytes(),
                meta_path.read_bytes(),
            )
            == (data, meta_bytes)
        ):
            return False, rows_path
        raise CarryOverError(f"{rows_path} already holds a different carry-over; use a new name")
    directory.mkdir(parents=True, exist_ok=True)
    rows_path.write_bytes(data)
    meta_path.write_bytes(meta_bytes)
    return True, rows_path


__all__ = [
    "CARRYOVER_DIRNAME",
    "QUALITY_DECISIONS",
    "QUALITY_ISSUES",
    "CarryOver",
    "CarryOverError",
    "carry_over",
    "load_quality_decisions",
    "question_sha256",
    "write_carry_over",
]
