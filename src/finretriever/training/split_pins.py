"""Split pins: passages whose split units a new preparation must keep on the side they are on.

A pins file is built once from an earlier preparation (``pinning.build_split_pins``, CLI
``pin-split``)
and read by ``prepare`` through ``split.pins`` (``splits.assign_splits``). It lists every pinned
passage with its side and every reason:

* ``train``: the positive of every train pair of the named training-candidate datasets and of the
  built train pairs, every passage of the named train plans, and every train passage that holds
  any of their evidence, exactly or possibly (``evidence_reuse.EvidenceLocator``). So no training
  example, and no passage that may answer one, can land in validation;
* ``validation`` (``keep_validation``): every validation passage of the earlier preparation, so
  the new validation split contains the old one and its validation sets carry over.

A pinned passage takes its whole split unit with it. The file names its sources with their
SHA-256 and is written once; the new preparation's manifest records the file's SHA-256. Nothing
moves silently: a carried pair whose positive would change sides cannot arise, because its unit
is pinned, and carried pairs are still re-checked one by one.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from finretriever.data.schemas import Passage
from finretriever.evaluation.provenance import CorpusIdentity, sha256_file

KIND = "split_pins"
FORMAT_VERSION = 1
SIDES = ("train", "validation")
NOTICE = (
    "Passages whose split units a new preparation keeps where they are: training examples and every train passage "
    "that may answer one stay train, and every earlier validation passage stays validation. Built from passage IDs "
    "and corpus text only; no evaluation label was read."
)


class SplitPinsError(ValueError):
    """A pins file cannot be built, read or trusted; nothing is written."""


def _text(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_split_pins(path: str | Path, doc: Mapping[str, Any]) -> bool:
    """Write the pins file once; ``False`` if an identical copy exists. A different one is refused."""
    path = Path(path)
    text = _text(doc)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise SplitPinsError(f"{path} already holds different pins; pins are never changed, so use a new file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def load_split_pins(
    path: str | Path, corpus: CorpusIdentity, passages: Mapping[str, Passage]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Passage ID -> side, and what a manifest records about the file; fails closed on any inconsistency."""
    path = Path(path)
    if not path.is_file():
        raise SplitPinsError(f"split pins {path} do not exist (build them with pin-split)")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SplitPinsError(f"{path} is not JSON ({exc})") from None
    if doc.get("kind") != KIND or doc.get("format_version") != FORMAT_VERSION:
        raise SplitPinsError(f"{path} is not a format-{FORMAT_VERSION} split-pins file")
    if (doc["corpus"]["sha256"], doc["corpus"]["version"]) != (corpus.corpus_sha256, corpus.corpus_version):
        raise SplitPinsError(f"{path} was built from another corpus")
    pins: dict[str, str] = {}
    for row in doc["pins"]:
        pid, side = row["passage_id"], row["side"]
        if pid not in passages:
            raise SplitPinsError(f"{path}: {pid} is not in the corpus")
        if side not in SIDES:
            raise SplitPinsError(f"{path}: {pid} has side {side!r}")
        if not row.get("reasons"):
            raise SplitPinsError(f"{path}: {pid} is pinned without a reason")
        if pid in pins:
            raise SplitPinsError(f"{path}: {pid} is listed twice")
        pins[pid] = side
    if dict(sorted(Counter(pins.values()).items())) != doc["counts"]:
        raise SplitPinsError(f"{path}: its counts do not match its pins")
    record = {"path": str(path), "sha256": sha256_file(path), "source": doc["source"], "counts": doc["counts"]}
    return pins, record


__all__ = [
    "FORMAT_VERSION",
    "KIND",
    "NOTICE",
    "SIDES",
    "SplitPinsError",
    "load_split_pins",
    "write_split_pins",
]
