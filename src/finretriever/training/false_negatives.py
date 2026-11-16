"""False-negative protection: passages that answer a training question are never its negatives.

A contrastive bi-encoder treats every other passage in a batch, and every mined hard negative,
as wrong for a query. In 10-K filings that is often false: next year's filing repeats a risk
factor, a comparative table repeats last year's figures, and consecutive chunks share text. Each
pair built by ``finretriever.training.pair_quality`` records its *positive equivalence group*:

* its overlap group (``group_id``, whole-passage copies);
* ``equivalents.exact``: same-split passages whose visible text holds the evidence itself (the
  quote, a whole evidence sentence, or the answer's figure under the evidence's label);
* ``equivalents.uncertain``: same-split passages that may answer it (another figure of the
  evidence, the answer's figure under another label, or 8 consecutive evidence words).

Both kinds are excluded here, uncertain ones included: wrongly treating a passage as negative
teaches the model the opposite of the truth, while excluding one too many only costs a
negative. The groups are built from corpus text alone; no evaluation judgment is read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def negative_exclusions(
    pair: Mapping[str, Any], group_members: Mapping[str, Sequence[str]]
) -> frozenset[str]:
    """Passages that must never be a negative for ``pair``: its positive equivalence group."""
    equivalents = pair.get("equivalents") or {}
    return frozenset(
        {
            pair["positive_passage_id"],
            *group_members.get(pair["group_id"], ()),
            *equivalents.get("exact", ()),
            *equivalents.get("uncertain", ()),
        }
    )


def filter_negatives(
    pair: Mapping[str, Any],
    candidates: Iterable[str],
    group_members: Mapping[str, Sequence[str]],
    blocked: Iterable[str] = (),
) -> list[str]:
    """Hard-negative candidates for ``pair`` (in their order) without its equivalence group or
    blocked passages."""
    excluded = negative_exclusions(pair, group_members) | set(blocked)
    return [pid for pid in candidates if pid not in excluded]


def conflict(
    a: Mapping[str, Any], b: Mapping[str, Any], group_members: Mapping[str, Sequence[str]]
) -> bool:
    """Whether ``a`` and ``b`` may not share a batch: either positive answers the other's query."""
    return b["positive_passage_id"] in negative_exclusions(a, group_members) or a[
        "positive_passage_id"
    ] in negative_exclusions(b, group_members)


def conflict_free_batches(
    pairs: Sequence[Any],
    batch_size: int,
    seed: int,
    group_members: Mapping[str, Sequence[str]],
    *,
    pair_id: Callable[[Any], str] = lambda p: p["pair_id"],
    positives: Callable[[Any], frozenset[str]] | None = None,
    exclusions: Callable[[Any], frozenset[str]] | None = None,
) -> list[list[str]]:
    """Pair IDs in batches of at most ``batch_size`` with no two conflicting pairs.

    Two pairs conflict when a positive of one (``positives``: by default its positive passage) is
    excluded as a negative for the other (``exclusions``: by default ``negative_exclusions``). Pairs
    are taken in the order of SHA-256 of ``f"{seed}:{pair_id}"`` and each goes into the first open
    batch it does not conflict with, else a new batch. The same pairs and seed always give the same
    batches; every pair is placed exactly once.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    positive_of = positives or (lambda p: frozenset({p["positive_passage_id"]}))
    excluded_of = exclusions or (lambda p: negative_exclusions(p, group_members))
    order = sorted(pairs, key=lambda p: hashlib.sha256(f"{seed}:{pair_id(p)}".encode()).hexdigest())
    batches: list[list[Any]] = []
    held: list[set[str]] = []  # positives in each batch
    barred: list[set[str]] = []  # passages excluded as negatives by some pair in each batch
    for pair in order:
        own_positives, own_excluded = positive_of(pair), excluded_of(pair)
        for i, batch in enumerate(batches):
            if (
                len(batch) < batch_size
                and not own_positives & barred[i]
                and not own_excluded & held[i]
            ):
                batch.append(pair)
                held[i] |= own_positives
                barred[i] |= own_excluded
                break
        else:
            batches.append([pair])
            held.append(set(own_positives))
            barred.append(set(own_excluded))
    return [[pair_id(p) for p in batch] for batch in batches]


def balanced_conflict_free_batches(
    pairs: Sequence[Any],
    batch_size: int,
    seed: int,
    group_members: Mapping[str, Sequence[str]],
    *,
    pair_id: Callable[[Any], str] = lambda p: p["pair_id"],
    positives: Callable[[Any], frozenset[str]] | None = None,
    exclusions: Callable[[Any], frozenset[str]] | None = None,
) -> list[list[str]]:
    """Pair IDs in the fewest batches ``batch_size`` allows, of near-equal size, with no two conflicting pairs.

    The conflict relation is ``conflict_free_batches``'. Unlike its first-fit packing, which leaves the members of
    large conflict cliques for a tail of tiny batches, this fills K = ceil(N / batch_size) batches of at most
    ceil(N / K) pairs: pairs are taken most-constrained first (by how many pairs they conflict with, ties by SHA-256
    of ``f"{seed}:{pair_id}"``), each into the smallest batch it does not conflict with (ties: the lower index). If a
    pair fits nowhere, K grows by one and the placement starts again (K = N always succeeds). The batches are then
    ordered by SHA-256 of ``f"{seed}:"`` and their sorted pair IDs. The same pairs and seed always give the same
    batches, whatever their input order; every pair is placed exactly once.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    positive_of = positives or (lambda p: frozenset({p["positive_passage_id"]}))
    excluded_of = exclusions or (lambda p: negative_exclusions(p, group_members))
    ids = [pair_id(p) for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("pair IDs repeat")
    holders: dict[str, set[str]] = {}  # passage -> pairs holding it as a positive
    for p, i in zip(pairs, ids, strict=True):
        for passage in positive_of(p):
            holders.setdefault(passage, set()).add(i)
    adjacent: dict[str, set[str]] = {i: set() for i in ids}
    for p, i in zip(pairs, ids, strict=True):  # a conflicts with b when one's positive is excluded for the other
        for passage in excluded_of(p):
            for j in holders.get(passage, ()):
                if j != i:
                    adjacent[i].add(j)
                    adjacent[j].add(i)

    def key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    order = sorted(ids, key=lambda i: (-len(adjacent[i]), key(f"{seed}:{i}")))
    count = max(1, -(-len(ids) // batch_size))
    while True:
        capacity = -(-len(ids) // count)
        batches: list[list[str]] = [[] for _ in range(count)]
        members: list[set[str]] = [set() for _ in range(count)]
        for i in order:
            fits = [b for b in range(count) if len(batches[b]) < capacity and not adjacent[i] & members[b]]
            if not fits:
                break
            b = min(fits, key=lambda b: (len(batches[b]), b))
            batches[b].append(i)
            members[b].add(i)
        else:
            placed = [b for b in batches if b]
            return sorted(placed, key=lambda b: key(f"{seed}:" + ",".join(sorted(b))))
        count += 1


__all__ = [
    "balanced_conflict_free_batches",
    "conflict",
    "conflict_free_batches",
    "filter_negatives",
    "negative_exclusions",
]
