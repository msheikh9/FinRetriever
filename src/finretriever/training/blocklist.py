"""The evaluation blocklist: passages that may never become training examples or negatives.

Built from passage IDs only:

* every candidate passage of the development evaluation: the pools of the annotation
  workspaces, of the expanded development pools and of the audit workspaces;
* every passage in the gold qrels, if any exist;
* every passage in the overlap group of any of the above (``finretriever.training.overlap``),
  because a year-to-year copy of an evaluation passage is the same evidence;
* optionally (``expand_by_evidence_reuse``), every passage whose visible text substantially
  reuses the sentences or figures of a blocked passage although it shares less than half its
  shingles (``finretriever.training.evidence_reuse``), with its overlap group. The expansion is
  one step: it starts from the evaluation passages and their groups, never from passages it
  added itself.

Grades, judgments and question texts are never read, so no label can shape the training data;
blocking every pooled passage, relevant or not, keeps it that way. Each source is opened with
its own corpus-binding checks (``DevPool.open``, ``AnnotationStore.open``,
``AuditWorkspace.open``, gold metadata), and the build fails closed: a missing source, a source
bound to another corpus, a passage outside the corpus, or gold qrels without metadata stop it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from finretriever.data.schemas import Passage
from finretriever.eval_set.annotation_store import POOL_FILE, WORKSPACE_FILE, AnnotationStore
from finretriever.eval_set.audit import AUDIT_FILE, AuditWorkspace
from finretriever.eval_set.dev_pool import POOL_INFO_FILE, DevPool
from finretriever.evaluation.provenance import (
    CorpusIdentity,
    check_annotations_match_corpus,
    load_evaluation_metadata,
    sha256_file,
)
from finretriever.training.evidence_reuse import Reuse, ReuseIndex
from finretriever.training.overlap import OverlapGroups


class BlocklistError(ValueError):
    """An evaluation source is missing, unreadable or bound to another corpus."""


class BlocklistSources(BaseModel):
    """Where evaluation passages live; every listed source must exist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation_workspaces: list[str] = Field(default_factory=list)
    dev_pools: list[str] = Field(default_factory=list)
    audit_workspaces: list[str] = Field(default_factory=list)
    gold_qrels: str | None = None
    gold_metadata: str | None = None


@dataclass(frozen=True)
class Blocklist:
    """Blocked passages (ID -> reasons), the sources read, and statistics."""

    reasons: dict[str, tuple[str, ...]]
    sources: tuple[dict[str, Any], ...]
    stats: dict[str, Any]

    def __contains__(self, passage_id: object) -> bool:
        return passage_id in self.reasons

    def rows(self, order: list[str]) -> list[dict[str, Any]]:
        """One row per blocked passage, in ``order`` (corpus order)."""
        return [
            {"passage_id": pid, "reasons": list(self.reasons[pid])}
            for pid in order
            if pid in self.reasons
        ]


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise BlocklistError(f"{what} {path} does not exist; the blocklist fails closed")
    return path


def _gold_passages(
    sources: BlocklistSources, corpus: CorpusIdentity
) -> tuple[set[str], dict[str, Any] | None]:
    if sources.gold_qrels is None:
        return set(), None
    qrels = _require(Path(sources.gold_qrels), "gold qrels")
    rows = [
        json.loads(line) for line in qrels.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    metadata = Path(sources.gold_metadata) if sources.gold_metadata else None
    if metadata is not None and metadata.exists():
        check_annotations_match_corpus(load_evaluation_metadata(metadata), corpus)
    elif rows:
        raise BlocklistError(
            f"{qrels} holds {len(rows)} qrels but its metadata ({sources.gold_metadata}) is missing, so its corpus "
            "cannot be verified"
        )
    try:
        passage_ids = {str(row["passage_id"]) for row in rows}
    except (KeyError, TypeError) as exc:
        raise BlocklistError(f"{qrels} has a row without passage_id ({exc!r})") from None
    record = {
        "kind": "gold_qrels",
        "path": str(qrels),
        "sha256": sha256_file(qrels),
        "rows": len(rows),
        "metadata": None if metadata is None or not metadata.exists() else str(metadata),
    }
    return passage_ids, record


def build_blocklist(
    sources: BlocklistSources,
    *,
    corpus: CorpusIdentity,
    passages: Mapping[str, Passage],
    groups: OverlapGroups,
) -> Blocklist:
    """Every evaluation candidate passage and its overlap group; see the module docstring."""
    direct: dict[str, set[str]] = {}
    records: list[dict[str, Any]] = []

    def add(passage_ids: set[str], reason: str, record: dict[str, Any]) -> None:
        if unknown := sorted(passage_ids - passages.keys()):
            raise BlocklistError(
                f"{record['path']} names passages outside the corpus: {unknown[:3]}"
            )
        for pid in passage_ids:
            direct.setdefault(pid, set()).add(reason)
        records.append({**record, "passages": len(passage_ids)})

    for directory in map(Path, sources.annotation_workspaces):
        store = AnnotationStore.open(_require(directory, "annotation workspace"), corpus)
        ids = {e.passage_id for q in store.questions() for e in store.pool(q.query_id)}
        add(
            ids,
            f"annotation_pool:{directory}",
            {
                "kind": "annotation_pool",
                "path": str(directory),
                "pool_sha256": sha256_file(directory / POOL_FILE),
                "workspace_sha256": sha256_file(directory / WORKSPACE_FILE),
            },
        )
    for directory in map(Path, sources.dev_pools):
        pool = DevPool.open(_require(directory, "development pool"), corpus, passages)
        add(
            {e.passage_id for e in pool.entries},
            f"dev_pool:{directory}",
            {
                "kind": "dev_pool",
                "path": str(directory),
                "pool_meta_sha256": sha256_file(directory / POOL_INFO_FILE),
                "pool_version": pool.info.pool_version,
            },
        )
    for directory in map(Path, sources.audit_workspaces):
        audit = AuditWorkspace.open(_require(directory, "audit workspace"), corpus)
        add(
            {i.passage_id for i in audit.items()},
            f"audit_sample:{directory}",
            {
                "kind": "audit_sample",
                "path": str(directory),
                "audit_sha256": sha256_file(directory / AUDIT_FILE),
            },
        )
    gold, record = _gold_passages(sources, corpus)
    if record is not None:
        add(gold, "gold_qrels", record)
    if not records:
        raise BlocklistError("no evaluation source is configured; the blocklist fails closed")

    reasons: dict[str, set[str]] = {pid: set(r) for pid, r in direct.items()}
    for pid in sorted(direct):
        group = groups.group_of[pid]
        for member in groups.members[group]:
            if member not in direct:
                reasons.setdefault(member, set()).add(f"overlap_group:{group}")
    ordered = {pid: tuple(sorted(r)) for pid, r in sorted(reasons.items())}
    blocked_groups = {groups.group_of[pid] for pid in ordered}
    if partial := sorted(
        g for g in blocked_groups if any(m not in ordered for m in groups.members[g])
    ):
        raise BlocklistError(f"overlap groups only partly blocked: {partial[:3]}")
    stats = {
        "direct": len(direct),
        "via_overlap_group": len(ordered) - len(direct),
        "blocked": len(ordered),
        "blocked_groups": len(blocked_groups),
        "by_source": dict(sorted(Counter(r for rs in direct.values() for r in rs).items())),
        "by_ticker": dict(sorted(Counter(passages[pid].ticker for pid in ordered).items())),
    }
    return Blocklist(ordered, tuple(records), stats)


def expand_by_evidence_reuse(
    blocklist: Blocklist,
    index: ReuseIndex,
    groups: OverlapGroups,
    passages: Mapping[str, Passage],
) -> tuple[Blocklist, dict[str, Reuse]]:
    """``blocklist`` plus every passage substantially reusing blocked evidence, and its group.

    Returns the expanded blocklist and, for every passage that was not blocked, what its visible
    text shares with the blocked passages' full text (empty ``Reuse`` values included), so the
    decision for each passage can be audited.
    """
    blocked = set(blocklist.reasons)
    reuse = index.reuse_against([pid for pid in index.order if pid not in blocked], blocked)
    direct = [pid for pid, r in reuse.items() if r.substantial(index.settings)]
    reasons: dict[str, set[str]] = {pid: set(r) for pid, r in blocklist.reasons.items()}
    for pid in direct:
        reasons.setdefault(pid, set()).add("evidence_reuse")
    added_by_group = 0
    for pid in direct:
        group = groups.group_of[pid]
        for member in groups.members[group]:
            if member not in reasons:
                reasons[member] = {f"overlap_group:{group}"}
                added_by_group += 1
    ordered = {pid: tuple(sorted(r)) for pid, r in sorted(reasons.items())}
    blocked_groups = {groups.group_of[pid] for pid in ordered}
    if partial := sorted(
        g for g in blocked_groups if any(m not in ordered for m in groups.members[g])
    ):
        raise BlocklistError(f"overlap groups only partly blocked: {partial[:3]}")
    stats = {
        **blocklist.stats,
        "before_evidence_reuse": len(blocklist.reasons),
        "evidence_reuse": len(direct),
        "evidence_reuse_by_rule": dict(
            sorted(
                Counter(
                    "+".join(
                        rule
                        for rule, hit in (
                            (
                                "spans",
                                reuse[pid].word_share >= index.settings.substantial_word_share,
                            ),
                            (
                                "facts",
                                len(reuse[pid].facts) >= index.settings.substantial_facts
                                and reuse[pid].fact_share >= index.settings.substantial_fact_share,
                            ),
                        )
                        if hit
                    )
                    for pid in direct
                ).items()
            )
        ),
        "via_evidence_reuse_group": added_by_group,
        "blocked": len(ordered),
        "blocked_groups": len(blocked_groups),
        "by_ticker": dict(sorted(Counter(passages[pid].ticker for pid in ordered).items())),
    }
    return Blocklist(ordered, blocklist.sources, stats), reuse


__all__ = [
    "Blocklist",
    "BlocklistError",
    "BlocklistSources",
    "build_blocklist",
    "expand_by_evidence_reuse",
]
