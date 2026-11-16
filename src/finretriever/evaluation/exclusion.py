"""Passages removed from an evaluation's ranking corpus before ranking, frozen by name and SHA-256.

Gold set v1's primary analysis ranks only the **eligible corpus** (decision D8, 2026-11-12; docs/gold_set_v1_protocol.md
§4). That is the corpus minus every passage the A1 runs used as a training or validation positive, or as an exact or
uncertain equivalent of one.

**Where the set comes from.** It is derived only from the two datasets that every official A1 run recorded in its
``run.json`` and checkpoint metadata:
- ``train_candidates_v3``, manifest ``0586c7ca…``;
- ``validation_pairs_v4``, manifest ``40238cb0…``.

Both were frozen on 2026-11-09, before any training. The A1 objective uses in-batch negatives only (no mined hard
negatives), so these are all the passages its training showed the model.

**Not the blocklist.** The preparation's evaluation blocklist is the opposite list: evaluation-pool passages kept
out of training. It is not this set and is not read here.

**Frozen.** ``exclusion_ids`` recomputes the set from the pinned files. Each manifest must match its pin, each file
must match its manifest, and the set's size and SHA-256 must match ``EXCLUSIONS``. Otherwise it refuses, so the set
cannot change after annotation begins.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

A1_TRAINING_AND_VALIDATION = "a1_training_and_validation"
EXCLUSIONS: dict[str, dict[str, Any]] = {
    A1_TRAINING_AND_VALIDATION: {
        "description": "every passage the A1 runs used as a training or validation positive, or as an exact or "
                       "uncertain equivalent of one (train_candidates_v3 and validation_pairs_v4)",
        "sources": {
            "train_candidates_v3": {
                "directory": "data/processed/train/bge_base_ft_v3/candidates/train_candidates_v3",
                "manifest_sha256": "0586c7ca6982fbc1d9a15a952d483fbecb9f4589779d86bbb99bfba62b065284",
                "file": "train_candidates.jsonl",
            },
            "validation_pairs_v4": {
                "directory": "data/processed/train/bge_base_ft_v3/validation_sets/validation_pairs_v4",
                "manifest_sha256": "40238cb06d73287821115dbe1b5c33a2dbbd294c7dd5b3c9fdc569c475e24b0e",
                "file": "validation_pairs.jsonl",
            },
        },
        "count": 1745,
        "sha256": "9c408e005267eb66c7a7ede7c4785e2e91ccdce9757b1fafe08136f4383995c3",
    },
}


class ExclusionError(ValueError):
    """An exclusion set cannot be reproduced exactly as frozen."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha256(ids: Iterable[str]) -> str:
    """The SHA-256 of a set of passage IDs: sorted, joined by newlines."""
    return hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()


def exclusion_ids(name: str, root: str | Path = ".", *, spec: Mapping[str, Any] | None = None) -> frozenset[str]:
    """The frozen exclusion set ``name``, recomputed from its pinned sources and checked against its size and SHA-256.
    ``spec`` replaces the registered definition (tests only)."""
    definition = spec if spec is not None else EXCLUSIONS.get(name)
    if definition is None:
        raise ExclusionError(f"unknown exclusion set {name!r}; the frozen ones are {sorted(EXCLUSIONS)}")
    found: set[str] = set()
    for source, pin in definition["sources"].items():
        directory = Path(root) / pin["directory"]
        manifest = directory / "manifest.json"
        if not manifest.is_file() or _sha256_file(manifest) != pin["manifest_sha256"]:
            raise ExclusionError(f"{name}: {manifest} is missing or not the pinned {source} manifest")
        listed = (json.loads(manifest.read_text(encoding="utf-8")).get("files") or {}).get(pin["file"])
        data = directory / pin["file"]
        if not data.is_file() or _sha256_file(data) != listed:
            raise ExclusionError(f"{name}: {data} is missing or differs from its manifest")
        for line in data.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                equivalents = row.get("equivalents") or {}
                found.update([row["positive_passage_id"], *equivalents.get("exact", ()),
                              *equivalents.get("uncertain", ())])
    if (len(found), ids_sha256(found)) != (definition["count"], definition["sha256"]):
        raise ExclusionError(f"{name}: {len(found)} passages ({ids_sha256(found)[:12]}), not the frozen "
                             f"{definition['count']} ({definition['sha256'][:12]})")
    return frozenset(found)


def collection_record(name: str, excluded: frozenset[str], corpus_ids: Iterable[str]) -> dict[str, Any]:
    """What an evaluation on the eligible corpus records: the set's identity and the corpus sizes."""
    corpus = set(corpus_ids)
    definition = EXCLUSIONS.get(name, {})
    if outside := sorted(excluded - corpus):
        raise ExclusionError(f"{name}: {len(outside)} excluded passages are not in the corpus (e.g. {outside[0]})")
    return {
        "exclusion": name,
        "description": definition.get("description"),
        "exclusion_sha256": ids_sha256(excluded),
        "sources": {k: v["manifest_sha256"] for k, v in (definition.get("sources") or {}).items()},
        "corpus_passages": len(corpus),
        "excluded_passages": len(excluded),
        "eligible_passages": len(corpus - excluded),
        "applied": "before ranking: every retriever indexes only the eligible passages",
    }


__all__ = [
    "A1_TRAINING_AND_VALIDATION",
    "EXCLUSIONS",
    "ExclusionError",
    "collection_record",
    "exclusion_ids",
    "ids_sha256",
]
