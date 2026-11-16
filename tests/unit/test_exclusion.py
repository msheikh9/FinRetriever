"""The frozen exclusion sets (``finretriever.evaluation.exclusion``): recomputed from pinned sources, checked against
their size and SHA-256, and unable to change silently."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from finretriever.evaluation.exclusion import (
    A1_TRAINING_AND_VALIDATION,
    EXCLUSIONS,
    ExclusionError,
    collection_record,
    exclusion_ids,
    ids_sha256,
)

REPO = Path(__file__).resolve().parents[2]


def write_source(root: Path, name: str, rows: list[dict[str, Any]], file: str = "pairs.jsonl") -> dict[str, str]:
    directory = root / name
    directory.mkdir(parents=True)
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (directory / file).write_bytes(data)
    manifest = json.dumps({"files": {file: hashlib.sha256(data).hexdigest()}}).encode()
    (directory / "manifest.json").write_bytes(manifest)
    return {"directory": name, "manifest_sha256": hashlib.sha256(manifest).hexdigest(), "file": file}


@pytest.fixture
def spec(tmp_path: Path) -> dict[str, Any]:
    sources = {
        "train": write_source(tmp_path, "train", [
            {"positive_passage_id": "p1", "equivalents": {"exact": ["p2"], "uncertain": ["p3"]}},
            {"positive_passage_id": "p1", "equivalents": {}}]),
        "val": write_source(tmp_path, "val", [{"positive_passage_id": "p4"}]),
    }
    return {"description": "fixture", "sources": sources, "count": 4, "sha256": ids_sha256({"p1", "p2", "p3", "p4"})}


def test_the_registered_set_is_the_frozen_d8_one() -> None:
    frozen = EXCLUSIONS[A1_TRAINING_AND_VALIDATION]
    assert frozen["count"] == 1745 and frozen["sha256"].startswith("9c408e00")
    assert {k: v["manifest_sha256"][:8] for k, v in frozen["sources"].items()} == {
        "train_candidates_v3": "0586c7ca", "validation_pairs_v4": "40238cb0"}
    assert "blocklist" not in json.dumps(frozen)  # derived from the training data, not from the evaluation blocklist


def test_positives_and_both_kinds_of_equivalent_are_excluded(tmp_path: Path, spec: dict[str, Any]) -> None:
    assert exclusion_ids("fixture", tmp_path, spec=spec) == {"p1", "p2", "p3", "p4"}


def test_any_change_to_the_sources_or_the_set_is_refused(tmp_path: Path, spec: dict[str, Any]) -> None:
    with pytest.raises(ExclusionError, match="not the frozen 5"):
        exclusion_ids("fixture", tmp_path, spec={**spec, "count": 5})
    with pytest.raises(ExclusionError, match="not the frozen"):
        exclusion_ids("fixture", tmp_path, spec={**spec, "sha256": "0" * 64})
    (tmp_path / "val" / "pairs.jsonl").write_text(json.dumps({"positive_passage_id": "p9"}) + "\n")
    with pytest.raises(ExclusionError, match="differs from its manifest"):
        exclusion_ids("fixture", tmp_path, spec=spec)
    (tmp_path / "val" / "manifest.json").write_text("{}")
    with pytest.raises(ExclusionError, match="not the pinned val manifest"):
        exclusion_ids("fixture", tmp_path, spec=spec)
    with pytest.raises(ExclusionError, match="unknown exclusion set"):
        exclusion_ids("nonexistent", tmp_path)


def test_the_collection_record_gives_the_corpus_sizes() -> None:
    excluded = frozenset({"p1", "p2"})
    record = collection_record("unregistered", excluded, ["p1", "p2", "p3", "p4", "p5"])
    assert (record["corpus_passages"], record["excluded_passages"], record["eligible_passages"]) == (5, 2, 3)
    assert record["exclusion_sha256"] == ids_sha256(excluded) and "before ranking" in record["applied"]
    with pytest.raises(ExclusionError, match="not in the corpus"):
        collection_record("unregistered", frozenset({"p9"}), ["p1"])


@pytest.mark.skipif(not (REPO / EXCLUSIONS[A1_TRAINING_AND_VALIDATION]["sources"]["train_candidates_v3"]["directory"])
                    .is_dir(), reason="the frozen training data is not in this checkout (it is not under git)")
def test_the_real_frozen_set_reproduces_from_the_pinned_training_data() -> None:
    excluded = exclusion_ids(A1_TRAINING_AND_VALIDATION, REPO)
    assert len(excluded) == 1745 and ids_sha256(excluded) == EXCLUSIONS[A1_TRAINING_AND_VALIDATION]["sha256"]
