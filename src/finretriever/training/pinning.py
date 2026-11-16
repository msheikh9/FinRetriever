"""Building split pins from an earlier preparation (``split_pins`` describes the file and its rules)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from finretriever.data.schemas import Passage
from finretriever.evaluation.provenance import sha256_file
from finretriever.training.candidates import load_candidates, read_candidates_manifest
from finretriever.training.data_prep import DataPrepConfig, PreparedData
from finretriever.training.evidence_reuse import LOCATOR_VERSION
from finretriever.training.pair_quality import (
    PAIRS_DIRNAME,
    PAIRS_META_FILE,
    PairQualityError,
    evidence_locator,
    read_pairs,
)
from finretriever.training.sampling_plan import PlanError, load_plan, plan_path
from finretriever.training.split_pins import FORMAT_VERSION, KIND, NOTICE, SplitPinsError
from finretriever.training.training_data import TrainingDataError


def build_split_pins(
    prepared: PreparedData,
    passages: Mapping[str, Passage],
    config: DataPrepConfig,
    *,
    config_path: str,
    candidates: Sequence[str] = (),
    pairs: bool = False,
    plans: Sequence[str] = (),
    keep_validation: bool = False,
) -> dict[str, Any]:
    """The pins document for ``prepared`` (the rules are in ``split_pins``). Nothing is written."""
    rows = prepared.rows
    reasons: dict[str, set[str]] = defaultdict(set)
    try:
        locator = evidence_locator(prepared, passages)
    except PairQualityError as exc:
        raise SplitPinsError(str(exc)) from None

    def pin_pair(origin: str, positive: str, evidence: str, answer: str, ticker: str) -> None:
        reasons[positive].add(f"positive:{origin}")
        for e in locator.locate(evidence, answer, ticker, positive):
            if rows[e.passage_id]["split"] == "train":
                reasons[e.passage_id].add(f"holds_evidence:{origin}")

    source: dict[str, Any] = {
        "data_prep_config": {"path": config_path, "sha256": sha256_file(config_path)},
        "preparation": {"version": prepared.manifest["version"], "manifest_sha256": prepared.manifest_sha256},
        "evidence_method": LOCATOR_VERSION,
        "candidates": [],
        "pairs": None,
        "plans": [],
        "validation": "all" if keep_validation else None,
    }
    for version in candidates:
        try:
            _, digest = read_candidates_manifest(prepared, version)
            found = load_candidates(prepared, passages, config, version, "train")
        except TrainingDataError as exc:
            raise SplitPinsError(f"candidates {version!r}: {exc}") from None
        for pair in found:
            pin_pair(f"{version}:{pair.pair_id}", pair.positive_passage_id, pair.evidence, pair.answer, pair.ticker)
        source["candidates"].append({"version": version, "manifest_sha256": digest, "pairs": len(found)})
    if pairs:
        try:
            built = read_pairs(prepared)["train"]
        except PairQualityError as exc:
            raise SplitPinsError(str(exc)) from None
        for pair in built:
            pin_pair(f"pairs:{pair['pair_id']}", pair["positive_passage_id"], pair["evidence"], pair["answer"],
                     pair["ticker"])
        source["pairs"] = {
            "pairs_meta_sha256": sha256_file(prepared.directory / PAIRS_DIRNAME / PAIRS_META_FILE),
            "pairs": len(built),
        }
    approved = {p.passage_id: p for p in prepared.source_passages("train", passages, min_words=1)}
    for name in plans:
        try:
            _, planned = load_plan(prepared.directory, "train", name, approved,
                                   prepared_manifest_sha256=prepared.manifest_sha256)
        except PlanError as exc:
            raise SplitPinsError(str(exc)) from None
        for p in planned:
            reasons[p.passage_id].add(f"plan:train.{name}")
        source["plans"].append({"split": "train", "name": name, "passages": len(planned),
                                "sha256": sha256_file(plan_path(prepared.directory, "train", name))})
    train = set(reasons)
    if wrong := sorted(pid for pid in train if rows[pid]["split"] != "train"):
        raise SplitPinsError(f"training data names passages that are not train passages: {wrong[:3]}")
    side = dict.fromkeys(train, "train")
    if keep_validation:
        for pid, row in rows.items():
            if row["split"] == "validation":
                side[pid] = "validation"
                reasons[pid].add(f"validation:{prepared.manifest['version']}")
    pins = [
        {"passage_id": pid, "side": side[pid], "reasons": sorted(reasons[pid])}
        for pid in rows  # corpus order
        if pid in side
    ]
    return {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "notice": NOTICE,
        "corpus": {"sha256": prepared.manifest["corpus"]["sha256"], "version": prepared.manifest["corpus"]["version"]},
        "source": source,
        "counts": dict(sorted(Counter(p["side"] for p in pins).items())),
        "pins": pins,
    }


__all__ = ["build_split_pins"]
