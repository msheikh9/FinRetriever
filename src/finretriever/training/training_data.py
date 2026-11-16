"""What a contrastive trainer must use: validated pairs, false-negative-safe batches and negatives.

No training loop lives here; this is the interface the future trainer is bound to.

**Loading (``load_training_pairs``).** Pairs come from ``build-pairs`` (``pairs/<split>_pairs.jsonl``)
and, optionally, from carry-overs of earlier preparations (eligible rows of
``carryover/<name>.jsonl``, converted to the same contract). Loading fails closed
(``TrainingDataError``) unless every pair:

* satisfies the contract (``finretriever.training.pair_contract``) against this preparation, with
  equivalence metadata present: a pair without it, from another preparation or evidence-reuse
  index, or checked by an older validator is refused, never repaired;
* belongs to the requested split, so validation pairs come only from the validation split;
* still passes the candidate checks, holds no evidence a blocked or other-split passage holds, and
  has exactly the equivalents the evidence locator finds now;
* has a query no other loaded pair has.

**Protection (``NegativeGuard``).** A pair's positive equivalence set is its positive and every
exact equivalent (each holds the evidence; a trainer may use them as extra positives). Never a
negative for the pair: that set, the overlap groups of its members, every uncertain equivalent,
and every passage directly linked to the positive by substantial evidence reuse. Blocked passages
and passages of the other split are never negatives either.

**Batches (``plan_batches``).** In-batch negatives are the other pairs' positives, so two pairs may
share a batch only if neither's positives are excluded for the other. Batches are planned
deterministically from a seed and verified pair by pair before they are returned, by one of two
methods (``BATCHING_METHODS``): ``first_fit`` (``false_negatives.conflict_free_batches``, the
default, which every historical run used) packs batches full in a hashed order and leaves the
members of large conflict cliques in a tail of small batches; ``balanced``
(``false_negatives.balanced_conflict_free_batches``) fills the fewest batches the batch size
allows, of near-equal size, under the same conflict relation.

**Hard negatives (``hard_negative_candidates``).** Mined candidates in rank order, minus every
excluded passage.

**What remains.** Equivalence is found from shared text and figures (compared by amount since
``evidence-reuse-v2``: "$14.6 billion" is the cell "14,600" under the same label). Evidence restated
in other words, a rounded figure ("$14.6 billion" against "14,612"), a figure under another label
or with its sign in words, a table row whose label changed between filings, or an answer that only
a reader would infer from several passages is not detected as exact: such a passage is at most an
uncertain equivalent, and if nothing matches it can still be sampled as a negative. Uncertain equivalents are excluded but
not used as positives, so a passage that truly answers the question may be ignored rather than
learnt from. Passages beyond another passage's visible window are judged on visible text only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finretriever.data.schemas import Passage, Query
from finretriever.evaluation.provenance import sha256_file
from finretriever.training.carryover import CARRYOVER_DIRNAME
from finretriever.training.data_prep import DataPrepConfig, PreparedData
from finretriever.training.evidence_reuse import EvidenceLocator
from finretriever.training.false_negatives import (
    balanced_conflict_free_batches,
    conflict_free_batches,
)
from finretriever.training.pair_contract import (
    TrainingPair,
    contract_problems,
    make_training_pair,
)
from finretriever.training.pair_quality import (
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
from finretriever.training.query_generation import check_candidate, question_key


class TrainingDataError(ValueError):
    """Pairs, equivalence metadata or a batch plan cannot be trusted; nothing is returned."""


def _carryover_records(
    prepared: PreparedData, passages: Mapping[str, Passage], name: str, split: str
) -> list[dict[str, Any]]:
    """Eligible rows of a carry-over, as contract records, after checking the carry-over's files."""
    directory = prepared.directory / CARRYOVER_DIRNAME
    rows_path, meta_path = directory / f"{name}.jsonl", directory / f"{name}.meta.json"
    if not rows_path.is_file() or not meta_path.is_file():
        raise TrainingDataError(f"no carry-over {name!r} in {directory}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["files"].get(rows_path.name) != sha256_file(rows_path):
        raise TrainingDataError(f"{rows_path} does not match {meta_path.name}")
    if meta["target"]["manifest_sha256"] != prepared.manifest_sha256:
        raise TrainingDataError(f"carry-over {name!r} was checked against another preparation")
    source_meta_file = Path(meta["source"]["directory"]) / PAIRS_DIRNAME / PAIRS_META_FILE
    if (
        not source_meta_file.is_file()
        or sha256_file(source_meta_file) != meta["source"]["pairs_meta_sha256"]
    ):
        raise TrainingDataError(
            f"carry-over {name!r}: its source pairs manifest is missing or changed"
        )
    source_generation = json.loads(source_meta_file.read_text(encoding="utf-8"))["generation"]
    out = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["eligible"] or row["split"] != split:
            continue
        earlier = (
            row.get("source_provenance") or {}
        )  # contract-format sources; flat fields otherwise
        version = earlier.get("prompt_version", row.get("prompt_version"))
        prompt_sha = earlier.get("prompt_sha256", source_generation["prompt_sha256"])
        if not earlier and version != source_generation["prompt_version"]:
            raise TrainingDataError(
                f"{row['pair_id']}: prompt version differs from its source run's"
            )
        try:
            out.append(
                make_training_pair(
                    pair_id=row["pair_id"],
                    split=row["split"],
                    candidate={
                        "question": row["query"],
                        "type": row["question_type"],
                        "answer": row["answer"],
                        "evidence": row["evidence"],
                    },
                    positive=row["positive_passage_id"],
                    prepared=prepared,
                    passages=passages,
                    similarity=row["max_evaluation_similarity"],
                    equivalents=row["equivalents"],
                    provider=earlier.get("provider", row.get("provider")),
                    model=earlier.get("model", row.get("model")),
                    prompt_version=version,
                    prompt_sha=prompt_sha,
                    validator_version=row["validator_version"],
                    params_sha256=earlier.get(
                        "generation_params_sha256", row.get("generation_params_sha256")
                    ),
                    origin=f"carryover:{name}",
                )
            )
        except (KeyError, ValueError) as exc:
            raise TrainingDataError(
                f"{row.get('pair_id')}: cannot form a contract record ({exc})"
            ) from None
    return out


def load_training_pairs(
    prepared: PreparedData,
    passages: Mapping[str, Passage],
    config: DataPrepConfig,
    split: str,
    *,
    evaluation_questions: Sequence[Query],
    carryovers: Sequence[str] = (),
    include_built: bool = True,
) -> list[TrainingPair]:
    """Every pair of ``split`` a trainer may use, re-verified; see the module docstring."""
    if split not in ("train", "validation"):
        raise TrainingDataError(f"no training pairs in the {split!r} split")
    records: list[dict[str, Any]] = []
    if include_built and (prepared.directory / PAIRS_DIRNAME).exists():
        try:
            records.extend(read_pairs(prepared)[split])
        except PairQualityError as exc:
            raise TrainingDataError(str(exc)) from None
    for name in carryovers:
        records.extend(_carryover_records(prepared, passages, name, split))
    return verify_pairs(
        records, prepared, passages, config, split, evaluation_questions=evaluation_questions
    )


def verify_pairs(
    records: Sequence[Mapping[str, Any]],
    prepared: PreparedData,
    passages: Mapping[str, Passage],
    config: DataPrepConfig,
    split: str,
    *,
    evaluation_questions: Sequence[Query] | None,
    locator: EvidenceLocator | None = None,
) -> list[TrainingPair]:
    """``records`` as contract pairs of ``split``, after every check of ``load_training_pairs``.

    ``evaluation_questions=None`` skips only the lexical evaluation check, for callers that must
    not read evaluation files and verify instead that the check was recorded (``candidates``).
    """
    if locator is None:
        try:
            locator = evidence_locator(prepared, passages)
        except PairQualityError as exc:
            raise TrainingDataError(str(exc)) from None
    settings = config.generation
    aliases = aliases_for(passages, settings.company_aliases)
    problems: list[str] = []
    pairs: list[TrainingPair] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for record in records:
        found = contract_problems(record, prepared, passages)
        if found:
            problems.extend(found)
            continue
        pair = TrainingPair.model_validate(record)
        where = pair.pair_id
        if pair.split != split:
            problems.append(f"{where}: a {pair.split} pair among {split} pairs")
        if pair.pair_id in seen_ids:
            problems.append(f"{where}: duplicate pair_id")
        key = question_key(pair.query)
        if key in seen_queries:
            problems.append(f"{where}: its query repeats another pair's")
        seen_ids.add(pair.pair_id)
        seen_queries.add(key)
        raw = {
            "question": pair.query,
            "type": pair.question_type,
            "answer": pair.answer,
            "evidence": pair.evidence,
        }
        _, reason = check_candidate(
            raw, prepared.source_passage(pair.positive_passage_id, passages), settings, aliases
        )
        if reason is not None:
            problems.append(f"{where}: fails the candidate check ({reason})")
        equivalents = locator.locate(
            pair.evidence, pair.answer, pair.ticker, pair.positive_passage_id
        )
        if leak := evidence_leak(equivalents, prepared.rows, pair.split):
            problems.append(f"{where}: {leak.replace('_', ' ')}")
        if equivalent_ids(equivalents, prepared.rows, pair.split) != {
            "exact": pair.equivalents.exact,
            "uncertain": pair.equivalents.uncertain,
        }:
            problems.append(f"{where}: its equivalents are not the ones its evidence has now")
        if (
            evaluation_questions is not None
            and _max_similarity(evaluation_questions, pair.query)
            >= config.leakage.lexical_overlap_threshold
        ):
            problems.append(f"{where}: its query overlaps an evaluation question")
        pairs.append(pair)
    if problems:
        raise TrainingDataError(f"{len(problems)} problems, no pairs returned: {problems[:5]}")
    return pairs


class NegativeGuard:
    """Which passages may be negatives for which pair, in one preparation; see the module docstring."""

    def __init__(self, prepared: PreparedData) -> None:
        self.prepared = prepared
        self.split_of = {pid: row["split"] for pid, row in prepared.rows.items()}
        members: dict[str, list[str]] = {}
        for pid, row in prepared.rows.items():
            members.setdefault(row["group_id"], []).append(pid)
        self.group_members = {g: tuple(m) for g, m in members.items()}
        self.links = prepared.reuse_links()

    def _check(self, pair: TrainingPair) -> None:
        if pair.provenance.preparation_manifest_sha256 != self.prepared.manifest_sha256:
            raise TrainingDataError(f"{pair.pair_id} belongs to another preparation")

    def positives(self, pair: TrainingPair) -> frozenset[str]:
        """The pair's positive equivalence set: its positive and every exact equivalent."""
        self._check(pair)
        return frozenset(pair.positive_ids)

    def exclusions(self, pair: TrainingPair) -> frozenset[str]:
        """Passages that must never be a negative for ``pair``."""
        out = set(self.positives(pair)) | set(pair.equivalents.uncertain)
        for pid in pair.positive_ids:
            out |= set(self.group_members[self.prepared.rows[pid]["group_id"]])
        out |= self.links.get(pair.positive_passage_id, frozenset())
        return frozenset(out)

    def allowed_negative(self, pair: TrainingPair, passage_id: str) -> bool:
        return self.split_of.get(passage_id) == pair.split and passage_id not in self.exclusions(
            pair
        )

    def conflict(self, a: TrainingPair, b: TrainingPair) -> bool:
        """Whether ``a`` and ``b`` may not share a batch (one's positives are excluded for the other)."""
        return bool(
            self.positives(b) & self.exclusions(a) or self.positives(a) & self.exclusions(b)
        )


@dataclass(frozen=True)
class BatchPlan:
    seed: int
    batch_size: int
    batches: tuple[tuple[str, ...], ...]  # pair IDs; each tuple is one in-batch-negative group
    stats: Mapping[str, Any]


BATCHING_METHODS = {"first_fit": conflict_free_batches, "balanced": balanced_conflict_free_batches}


def plan_batches(
    pairs: Sequence[TrainingPair], guard: NegativeGuard, *, batch_size: int, seed: int, method: str = "first_fit"
) -> BatchPlan:
    """Deterministic batches in which no pair's positives are another pair's excluded negatives (``method``: one of
    ``BATCHING_METHODS``; see the module docstring)."""
    if method not in BATCHING_METHODS:
        raise TrainingDataError(f"unknown batching method {method!r}; one of {list(BATCHING_METHODS)}")
    if not pairs:
        raise TrainingDataError("no pairs to plan")
    if len({p.split for p in pairs}) != 1:
        raise TrainingDataError("a batch plan holds pairs of one split only")
    if len({p.pair_id for p in pairs}) != len(pairs):
        raise TrainingDataError("pair IDs repeat")
    by_id = {p.pair_id: p for p in pairs}
    batches = BATCHING_METHODS[method](
        pairs,
        batch_size,
        seed,
        guard.group_members,
        pair_id=lambda p: p.pair_id,
        positives=guard.positives,
        exclusions=guard.exclusions,
    )
    for batch in batches:  # verify, independently of how the batches were made
        members = [by_id[i] for i in batch]
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if guard.conflict(a, b):
                    raise TrainingDataError(f"{a.pair_id} and {b.pair_id} would be false negatives")
    if sorted(i for b in batches for i in b) != sorted(by_id):
        raise TrainingDataError("the plan does not place every pair exactly once")
    sizes = [len(b) for b in batches]
    stats = {
        "pairs": len(pairs),
        "batches": len(batches),
        "full_batches": sum(s == batch_size for s in sizes),
        "smallest_batch": min(sizes),
        "mean_fill": round(sum(sizes) / (len(sizes) * batch_size), 4),
    }
    return BatchPlan(seed, batch_size, tuple(tuple(b) for b in batches), stats)


def hard_negative_candidates(
    pair: TrainingPair, ranked: Sequence[str], guard: NegativeGuard, *, limit: int | None = None
) -> list[str]:
    """Mined candidates (in rank order) a trainer may use as hard negatives for ``pair``."""
    allowed = [pid for pid in ranked if guard.allowed_negative(pair, pid)]
    return allowed if limit is None else allowed[:limit]


__all__ = [
    "BATCHING_METHODS",
    "BatchPlan",
    "NegativeGuard",
    "TrainingDataError",
    "hard_negative_candidates",
    "load_training_pairs",
    "plan_batches",
    "verify_pairs",
]
