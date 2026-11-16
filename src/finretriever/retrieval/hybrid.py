"""Metadata-aware hybrid retrieval: query understanding, scoping, rank fusion and long-passage scoring.

A :class:`HybridRetriever` composes existing pieces; it adds no new scoring model:

1. **Query understanding** (:class:`~finretriever.retrieval.query_understanding.QueryParser`): the companies and
   fiscal years the question names (or, for a reference run only, a question set's annotated scope).
2. **Scoping** (:func:`~finretriever.retrieval.metadata_scope.plan_scope`): the eligible passages under a
   :class:`~finretriever.retrieval.metadata_scope.ScopePolicy` (company filter, fiscal-year filter or boost,
   per-cell interleaving), with every decision and fallback recorded.
3. **Component scores**: any object with ``scores(query) -> per-passage scores in corpus order`` -- the existing
   :class:`~finretriever.retrieval.bm25.BM25Retriever` and dense retrievers, a :class:`MaxPScorer` over long-passage
   windows, or a :class:`CachedScorer` around either.
4. **Fusion** (:mod:`~finretriever.retrieval.fusion`): a single component's own order, or Reciprocal Rank Fusion
   of several; with a fiscal-year ``boost`` a metadata ranking (named year, then later filings in the window,
   then the rest) joins the fusion as one more voter.

Every hit carries the original passage ID and filing metadata. ``search`` returns the hits and a trace of what
was parsed and decided for the question.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from finretriever.data.schemas import Passage, SearchHit
from finretriever.retrieval.base import hit_metadata
from finretriever.retrieval.fusion import FusionConfig, reciprocal_rank_fusion
from finretriever.retrieval.metadata_scope import (
    CorpusMetadata,
    ScopePolicy,
    interleave,
    plan_scope,
)
from finretriever.retrieval.query_understanding import QueryParser, QueryScope

METADATA_VOTER = "metadata"


class Scorer(Protocol):
    def scores(self, query: str) -> Sequence[float]: ...


class CachedScorer:
    """Memoises a scorer's score vector per query text (BM25 scoring is the slow part of an experiment)."""

    def __init__(self, scorer: Scorer) -> None:
        self.scorer = scorer
        self._cache: dict[str, np.ndarray] = {}

    def scores(self, query: str) -> np.ndarray:
        if query not in self._cache:
            self._cache[query] = np.asarray(self.scorer.scores(query), dtype=np.float64)
        return self._cache[query]


class MaxPScorer:
    """A passage's score is the best of its own (head) score and its windows' scores.

    ``parents[j]`` is the corpus position of window ``j``'s source passage."""

    def __init__(self, head: Scorer, windows: Scorer, parents: Sequence[int]) -> None:
        self.head = head
        self.windows = windows
        self.parents = np.asarray(parents, dtype=np.int64)

    def scores(self, query: str) -> np.ndarray:
        out = np.array(self.head.scores(query), dtype=np.float64, copy=True)
        window_scores = np.asarray(self.windows.scores(query), dtype=np.float64)
        if window_scores.shape != self.parents.shape:
            raise ValueError("window scores and window parents differ in length")
        np.maximum.at(out, self.parents, window_scores)
        return out


@dataclass(frozen=True)
class HybridConfig:
    name: str
    components: tuple[str, ...]
    fusion: FusionConfig = field(default_factory=FusionConfig)
    policy: ScopePolicy = field(default_factory=ScopePolicy)
    metadata_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError(f"{self.name}: at least one component is needed")
        if self.fusion.method == "single" and len(self.components) != 1:
            raise ValueError(f"{self.name}: fusion 'single' takes exactly one component")
        if METADATA_VOTER in self.components:
            raise ValueError(f"{self.name}: {METADATA_VOTER!r} is reserved for the fiscal-year boost")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "components": list(self.components), "fusion": self.fusion.describe(),
                "policy": self.policy.describe(), "metadata_weight": self.metadata_weight}


def _ranked(scores: np.ndarray, eligible: np.ndarray, depth: int) -> list[int]:
    """The ``depth`` best eligible positions by score; ties by corpus position."""
    if eligible.size == 0:
        return []
    order = np.lexsort((eligible, -scores[eligible]))
    return [int(eligible[i]) for i in order[:depth]]


class HybridRetriever:
    def __init__(self, config: HybridConfig, scorers: Mapping[str, Scorer], passages: Sequence[Passage],
                 parser: QueryParser) -> None:
        if missing := sorted(set(config.components) - set(scorers)):
            raise ValueError(f"{config.name}: no scorer for components {missing}")
        self.config = config
        self.scorers = {name: scorers[name] for name in config.components}
        self.passages = list(passages)
        self.meta = CorpusMetadata(self.passages)
        self.parser = parser

    def describe(self) -> dict[str, Any]:
        return {**self.config.describe(), "parser": self.parser.describe()["parser_version"]}

    def _fuse(self, component_scores: Mapping[str, np.ndarray], eligible: Sequence[int], depth: int,
              scope: QueryScope, allowed: set[int]) -> list[tuple[int, float]]:
        cfg = self.config
        pool = np.asarray(sorted(eligible), dtype=np.int64)
        fusion_depth = max(depth, cfg.fusion.depth)
        rankings = {name: _ranked(component_scores[name], pool, fusion_depth) for name in cfg.components}
        if cfg.fusion.method == "single":
            name = cfg.components[0]
            fused = [(p, float(component_scores[name][p])) for p in rankings[name]]
        else:
            fused = reciprocal_rank_fusion(rankings, k=cfg.fusion.k, weights=cfg.fusion.weights,
                                           depth=cfg.fusion.depth)
        if cfg.policy.year == "boost" and allowed:
            prior = [p for p, _ in fused]
            tiered = sorted(range(len(prior)), key=lambda i: (self.meta.year_tier(prior[i], scope, allowed), i))
            voters = dict(rankings) if cfg.fusion.method == "rrf" else {cfg.components[0]: prior}
            voters[METADATA_VOTER] = [prior[i] for i in tiered]
            weights = {**cfg.fusion.weights, METADATA_VOTER: cfg.metadata_weight}
            fused = reciprocal_rank_fusion(voters, k=cfg.fusion.k, weights=weights, depth=fusion_depth)
        return fused[:depth]

    def search(self, query: str, top_k: int = 10, query_id: str = "query", exclude: Collection[str] = (),
               oracle_scope: QueryScope | None = None) -> tuple[list[SearchHit], dict[str, Any]]:
        """The top ``top_k`` passages for ``query`` (never one in ``exclude``) and a trace of the decisions."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}.")
        policy = self.config.policy
        if policy.source == "oracle":
            if oracle_scope is None:
                raise ValueError(f"{self.config.name}: an oracle-scope run needs the question's annotated scope")
            scope = oracle_scope
        else:
            scope = self.parser.parse(query)
        excluded = {self.meta.position[pid] for pid in exclude if pid in self.meta.position}
        plan = plan_scope(scope, policy, self.meta, excluded)
        component_scores = {name: np.asarray(s.scores(query), dtype=np.float64) for name, s in self.scorers.items()}
        for name, vector in component_scores.items():
            if vector.shape != (len(self.passages),):
                raise ValueError(f"component {name!r} scored {vector.shape} passages; the corpus has {len(self.passages)}")
        allowed = set(plan.allowed_years)
        if plan.cells:
            per_cell = [self._fuse(component_scores, cell.positions, top_k, scope, allowed) for cell in plan.cells]
            score_of = {p: s for ranking in per_cell for p, s in ranking}
            order = interleave([[p for p, _ in ranking] for ranking in per_cell], top_k)
            if len(order) < top_k:  # fewer cell passages than top_k: fill from the whole eligible set
                taken = set(order)
                whole = self._fuse(component_scores, plan.eligible, top_k, scope, allowed)
                rest = [(p, s) for p, s in whole if p not in taken][: top_k - len(order)]
                order += [p for p, _ in rest]
                score_of.update(rest)
            ranked = [(p, score_of[p]) for p in order]
        else:
            ranked = self._fuse(component_scores, plan.eligible, top_k, scope, allowed)
        hits = []
        for rank, (position, score) in enumerate(ranked, start=1):
            passage = self.passages[position]
            hits.append(SearchHit(
                query_id=query_id, passage_id=passage.passage_id, rank=rank, score=float(score),
                retriever=self.config.name,
                metadata={**hit_metadata(passage), "fiscal_year": self.meta.years[position]},
            ))
        return hits, {"scope": scope.to_dict(), "plan": plan.describe()}

    def explain(self, query: str, passage_ids: Sequence[str], scope: QueryScope | None = None,
                exclude: Collection[str] = ()) -> dict[str, dict[str, Any]]:
        """Why each passage ranks where it does: its rank in every component's own ranking of the passages the
        scope allows (1 = best), whether its company and fiscal year match the question's scope, and its fiscal-year
        tier (``requested year``, ``later filing`` that repeats it, or ``other year``). Read-only; :meth:`search`
        does not use it."""
        if scope is None:
            if self.config.policy.source == "oracle":
                raise ValueError(f"{self.config.name}: an oracle-scope run needs the question's annotated scope")
            scope = self.parser.parse(query)
        excluded = {self.meta.position[pid] for pid in exclude if pid in self.meta.position}
        plan = plan_scope(scope, self.config.policy, self.meta, excluded)
        pool = np.asarray(sorted(plan.eligible), dtype=np.int64)
        ranks: dict[str, dict[int, int]] = {}
        for name, scorer in self.scorers.items():
            ranking = _ranked(np.asarray(scorer.scores(query), dtype=np.float64), pool, len(pool))
            ranks[name] = {position: rank for rank, position in enumerate(ranking, start=1)}
        allowed = set(plan.allowed_years)
        tiers = ("requested year", "later filing", "other year")
        out: dict[str, dict[str, Any]] = {}
        for pid in passage_ids:
            position = self.meta.position[pid]
            year = self.meta.years[position]
            out[pid] = {
                "component_ranks": {name: r.get(position) for name, r in ranks.items()},
                "eligible_passages": len(pool),
                "company_in_scope": None if not scope.tickers else self.meta.tickers[position] in scope.tickers,
                "fiscal_year": year,
                "year_in_scope": None if not scope.years else year in scope.years,
                "year_tier": tiers[self.meta.year_tier(position, scope, allowed)] if allowed else None,
            }
        return out


__all__ = ["METADATA_VOTER", "CachedScorer", "HybridConfig", "HybridRetriever", "MaxPScorer", "Scorer"]
