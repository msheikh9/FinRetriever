"""Groups of substantially overlapping passages, so copies never straddle a train/validation split.

10-K filings repeat risk factors, policies and tables from year to year, and chunk boundaries
shift, so one paragraph appears in several passages. Method (deterministic, documented in
docs/training_data_v1.md):

1. **Shingles.** A passage's text is case-folded and split into ``[a-z0-9]+`` words (punctuation
   and table pipes ignored); its shingles are all runs of ``shingle_words`` consecutive words,
   each hashed to 64 bits with BLAKE2b. A passage with fewer words has one shingle of all of them.
2. **Exact duplicates.** Passages whose normalised word sequences are identical are linked
   directly (SHA-256 of the sequence), whatever their length.
3. **Candidates without all-pairs comparison.** An inverted index maps each shingle to the
   passages containing it. Shingles in more than ``max_shingle_df`` passages (boilerplate) are
   skipped; every other shingle votes for each pair of its passages. Pairs with at least
   ``min_shared_shingles`` votes are candidates. Work grows with how often shingles repeat, not
   with the square of the corpus size.
4. **Exact verification.** For each candidate pair the full shingle sets give Jaccard
   ``|A∩B| / |A∪B|`` and containment (overlap coefficient) ``|A∩B| / min(|A|, |B|)``. A pair is
   an edge if containment >= ``containment_threshold``; it is ``exact`` if the word sequences
   are identical, ``near_duplicate`` if Jaccard >= ``near_duplicate_threshold``, else ``overlap``.
5. **Groups.** Connected components of the edges (union-find), so a chain A~B~C is one group
   even when A and C do not overlap. A group is named after its first member in corpus order.

A pair can be missed only if almost all the shingles it shares are boilerplate repeated in more
than ``max_shingle_df`` passages; exact duplicates are never missed.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from finretriever.data.schemas import Passage

_WORD_RE = re.compile(r"[a-z0-9]+")
EdgeKind = Literal["exact", "near_duplicate", "overlap"]


class OverlapSettings(BaseModel):
    """Parameters of the overlap method; recorded with every artifact built from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shingle_words: int = Field(default=5, ge=1)
    max_shingle_df: int = Field(default=50, ge=2)
    min_shared_shingles: int = Field(default=10, ge=1)
    containment_threshold: float = Field(default=0.5, gt=0, le=1)
    near_duplicate_threshold: float = Field(default=0.8, gt=0, le=1)


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.casefold())


def shingle_hashes(text: str, k: int) -> frozenset[int]:
    """The 64-bit hashes of ``text``'s ``k``-word shingles (one shingle if it is shorter)."""
    tokens = words(text)
    if not tokens:
        return frozenset()
    grams = [" ".join(tokens[i : i + k]) for i in range(max(1, len(tokens) - k + 1))]
    return frozenset(
        int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") >> 1
        for g in grams
    )


@dataclass(frozen=True)
class OverlapEdge:
    a: str
    b: str
    kind: EdgeKind
    jaccard: float
    containment: float
    shared_shingles: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_a": self.a,
            "passage_b": self.b,
            "kind": self.kind,
            "jaccard": self.jaccard,
            "containment": self.containment,
            "shared_shingles": self.shared_shingles,
        }


@dataclass(frozen=True)
class OverlapGroups:
    """Every passage's group (singletons included), the edges behind them, and statistics."""

    settings: OverlapSettings
    group_of: dict[str, str]
    members: dict[str, tuple[str, ...]]
    edges: tuple[OverlapEdge, ...]
    stats: dict[str, Any] = field(default_factory=dict)

    def group_rows(self) -> list[dict[str, Any]]:
        """One row per passage, in corpus order: its group and its strongest overlap."""
        best: dict[str, tuple[float, float, int]] = {}
        for e in self.edges:
            for pid in (e.a, e.b):
                j, c, n = best.get(pid, (0.0, 0.0, 0))
                best[pid] = (max(j, e.jaccard), max(c, e.containment), n + 1)
        rows = []
        for pid, gid in self.group_of.items():
            j, c, n = best.get(pid, (0.0, 0.0, 0))
            rows.append(
                {
                    "passage_id": pid,
                    "group_id": gid,
                    "group_size": len(self.members[gid]),
                    "edges": n,
                    "max_jaccard": j,
                    "max_containment": c,
                }
            )
        return rows


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.parent[max(a, b)] = min(a, b)  # the root is always the earliest member


def _candidate_pairs(
    shingles: Sequence[frozenset[int]], settings: OverlapSettings
) -> dict[tuple[int, int], int]:
    """Pairs of passage indices sharing at least ``min_shared_shingles`` non-boilerplate shingles."""
    n = len(shingles)
    hashes = np.fromiter(
        (h for s in shingles for h in s), dtype=np.int64, count=sum(len(s) for s in shingles)
    )
    owners = np.repeat(np.arange(n, dtype=np.int64), [len(s) for s in shingles])
    if hashes.size == 0:
        return {}
    order = np.lexsort((owners, hashes))
    hashes, owners = hashes[order], owners[order]
    starts = np.flatnonzero(np.r_[True, hashes[1:] != hashes[:-1]])
    sizes = np.diff(np.r_[starts, hashes.size])
    codes: list[np.ndarray] = []
    for size in range(2, settings.max_shingle_df + 1):
        group_starts = starts[sizes == size]
        if group_starts.size == 0:
            continue
        block = owners[group_starts[:, None] + np.arange(size)]  # members, ascending within a row
        left, right = np.triu_indices(size, k=1)
        codes.append(block[:, left].ravel() * n + block[:, right].ravel())
    if not codes:
        return {}
    pair_codes, counts = np.unique(np.concatenate(codes), return_counts=True)
    keep = counts >= settings.min_shared_shingles
    return {
        (int(c // n), int(c % n)): int(k)
        for c, k in zip(pair_codes[keep], counts[keep], strict=True)
    }


def build_overlap_groups(
    passages: Sequence[Passage], settings: OverlapSettings | None = None
) -> OverlapGroups:
    """Overlap groups of ``passages`` (in corpus order) under ``settings``; see the module docstring."""
    settings = settings or OverlapSettings()
    ids = [p.passage_id for p in passages]
    if len(set(ids)) != len(ids):
        raise ValueError("passage IDs must be unique")
    shingles = [shingle_hashes(p.text, settings.shingle_words) for p in passages]
    sequence_hash = [hashlib.sha256(" ".join(words(p.text)).encode()).hexdigest() for p in passages]

    candidates = _candidate_pairs(shingles, settings)
    by_text: dict[str, list[int]] = defaultdict(list)
    for i, digest in enumerate(sequence_hash):
        by_text[digest].append(i)
    for group in by_text.values():
        for j in group[1:]:
            candidates.setdefault((group[0], j), 0)

    edges: list[OverlapEdge] = []
    union = _UnionFind(len(passages))
    for i, j in sorted(candidates):
        a, b = shingles[i], shingles[j]
        shared = len(a & b)
        exact = sequence_hash[i] == sequence_hash[j]
        jaccard = 1.0 if exact else (shared / len(a | b) if a or b else 0.0)
        containment = 1.0 if exact else (shared / min(len(a), len(b)) if a and b else 0.0)
        if not exact and containment < settings.containment_threshold:
            continue
        kind: EdgeKind = (
            "exact"
            if exact
            else ("near_duplicate" if jaccard >= settings.near_duplicate_threshold else "overlap")
        )
        edges.append(
            OverlapEdge(ids[i], ids[j], kind, round(jaccard, 4), round(containment, 4), shared)
        )
        union.union(i, j)

    roots = [union.find(i) for i in range(len(passages))]
    group_of = {ids[i]: f"grp_{root:05d}" for i, root in enumerate(roots)}
    members: dict[str, list[str]] = defaultdict(list)
    for pid, gid in group_of.items():
        members[gid].append(pid)
    sizes = Counter(len(m) for m in members.values())
    index = {pid: i for i, pid in enumerate(ids)}
    direct = {(min(index[e.a], index[e.b]), max(index[e.a], index[e.b])) for e in edges}
    transitive = 0
    for group in members.values():
        if len(group) > 2:
            positions = sorted(index[pid] for pid in group)
            pairs = len(positions) * (len(positions) - 1) // 2
            linked = sum(
                1
                for x in range(len(positions))
                for y in range(x + 1, len(positions))
                if (positions[x], positions[y]) in direct
            )
            transitive += linked < pairs
    filing = {p.passage_id: p.filing_id for p in passages}
    ticker = {p.passage_id: p.ticker for p in passages}
    stats = {
        "passages": len(passages),
        "candidate_pairs_verified": len(candidates),
        "edges": len(edges),
        "edges_by_kind": dict(sorted(Counter(e.kind for e in edges).items())),
        "cross_filing_edges": sum(filing[e.a] != filing[e.b] for e in edges),
        "cross_company_edges": sum(ticker[e.a] != ticker[e.b] for e in edges),
        "groups": len(members),
        "singleton_groups": sizes.get(1, 0),
        "passages_in_multi_member_groups": sum(n * k for n, k in sizes.items() if n > 1),
        "largest_group": max(sizes) if sizes else 0,
        "groups_linked_only_transitively": transitive,
        "group_size_histogram": {str(n): k for n, k in sorted(sizes.items())},
    }
    return OverlapGroups(
        settings, group_of, {g: tuple(m) for g, m in members.items()}, tuple(edges), stats
    )


__all__ = [
    "OverlapEdge",
    "OverlapGroups",
    "OverlapSettings",
    "build_overlap_groups",
    "shingle_hashes",
    "words",
]
