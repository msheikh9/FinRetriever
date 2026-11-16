"""Rank fusion: combine the rankings of several retrievers into one.

Reciprocal Rank Fusion (Cormack, Clarke and Büttcher, 2009): a passage's fused score is
``sum_r weight_r / (k + rank_r(passage))`` over the input rankings ``r`` that contain it, with ranks starting at 1.
It needs no score calibration, so BM25 scores and cosine similarities combine without normalisation. Each input
ranking is cut at ``depth``; a passage outside a ranking's top ``depth`` gets nothing from it.

Determinism: equal fused scores are ordered by the best (smallest) input rank, then by corpus position, so the
same inputs always give the same ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusionConfig:
    """How component rankings are combined.

    ``method``: ``single`` (one component, its own order) or ``rrf``. ``weights`` maps a component name to its
    RRF weight (missing components weigh 1.0). ``depth``: how many passages of each component ranking are fused.
    """

    method: str = "single"
    k: int = DEFAULT_RRF_K
    weights: Mapping[str, float] = field(default_factory=dict)
    depth: int = 100

    def __post_init__(self) -> None:
        if self.method not in ("single", "rrf"):
            raise ValueError(f"fusion method must be 'single' or 'rrf', got {self.method!r}")
        if self.k < 0:
            raise ValueError(f"RRF k must be >= 0, got {self.k}")
        if self.depth <= 0:
            raise ValueError(f"fusion depth must be > 0, got {self.depth}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("RRF weights must be >= 0")

    def describe(self) -> dict[str, Any]:
        return {"method": self.method, "k": self.k, "weights": dict(sorted(self.weights.items())),
                "depth": self.depth}


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    depth: int | None = None,
) -> list[tuple[int, float]]:
    """Fuse ``rankings`` (name -> corpus positions, best first) into ``[(position, fused score), ...]``, best first.

    Every position in any (cut) ranking is returned once. Ties: best input rank, then corpus position.
    """
    weights = weights or {}
    fused: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for name in sorted(rankings):  # fixed summation order: identical floats on every run
        weight = float(weights.get(name, 1.0))
        ranking = rankings[name] if depth is None else rankings[name][:depth]
        seen: set[int] = set()
        for rank, position in enumerate(ranking, start=1):
            if position in seen:
                raise ValueError(f"ranking {name!r} lists corpus position {position} twice")
            seen.add(position)
            fused[position] = fused.get(position, 0.0) + weight / (k + rank)
            best_rank[position] = min(best_rank.get(position, rank), rank)
    order = sorted(fused, key=lambda p: (-fused[p], best_rank[p], p))
    return [(p, fused[p]) for p in order]


__all__ = ["DEFAULT_RRF_K", "FusionConfig", "reciprocal_rank_fusion"]
