"""Train/validation split by overlap group, after removing the evaluation blocklist.

Whole overlap groups are assigned, so no copy of a passage can sit on both sides. Every group
gets a stratum, the (ticker, fiscal year) most of its members belong to (ties: the smallest).
Within each stratum, groups are ordered by SHA-256 of ``f"{seed}:{group_id}"`` and moved to
validation one at a time while that brings the stratum's validation passage count closer to
``validation_share`` of its passages; the rest are training groups. The same corpus, groups,
blocklist and seed always give the same split. A group with any blocked member is blocked
whole (the blocklist expands to groups); a partly blocked group is refused.

With ``links`` (pairs of passages that substantially reuse each other's evidence although they
sit in different overlap groups, ``finretriever.training.evidence_reuse``), linked groups form
one split unit, assigned whole and named after its first group; the procedure is otherwise the
same, unit by unit. Without links every group is its own unit, so earlier splits reproduce.

Two optional settings change the targets, never the rule that units move whole:

* ``validation_share_by_ticker``: a company's strata use its own share instead of
  ``validation_share`` (more validation for companies with few units, less for one with many);
* ``pins`` (passage ID -> ``train`` or ``validation``, from a split-pins file,
  ``finretriever.training.split_pins``): each pinned passage's whole unit goes to that side before
  any other unit, and counts toward its stratum's target. A unit pinned to both sides, or a pin on
  a blocked passage, is refused. Pins keep existing training and validation data where it is
  instead of letting a new split move it.

Without either, the split is exactly the earlier one.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finretriever.data.schemas import Passage
from finretriever.training.blocklist import Blocklist
from finretriever.training.overlap import OverlapGroups

Split = Literal["train", "validation", "blocked"]
SPLITS: tuple[str, ...] = ("train", "validation", "blocked")


class SplitError(ValueError):
    """A split would leak across train and validation or include blocked passages."""


class SplitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    validation_share: float = Field(default=0.10, gt=0, lt=0.5)
    strategy: Literal["overlap_group_stratified"] = "overlap_group_stratified"
    stratify_by: tuple[Literal["ticker", "fiscal_year"], ...] = ("ticker", "fiscal_year")
    validation_share_by_ticker: dict[str, float] = Field(default_factory=dict)  # a company's own share
    pins: str | None = None  # a split-pins file whose passages' units are assigned first

    @model_validator(mode="after")
    def _shares(self) -> SplitSettings:
        if bad := sorted(t for t, s in self.validation_share_by_ticker.items() if not 0 < s < 0.5):
            raise ValueError(f"validation shares must be in (0, 0.5): {bad}")
        if self.validation_share_by_ticker and "ticker" not in self.stratify_by:
            raise ValueError("validation_share_by_ticker needs ticker in stratify_by")
        return self

    def share_for(self, stratum: tuple[str, ...]) -> float:
        """The target validation share of a stratum (its company's own share, if set)."""
        if not self.validation_share_by_ticker:
            return self.validation_share
        ticker = stratum[self.stratify_by.index("ticker")]
        return self.validation_share_by_ticker.get(ticker, self.validation_share)

    def manifest_settings(self) -> dict[str, Any]:
        """The settings a manifest records; the optional ones only when set, so earlier manifests reproduce."""
        out = self.model_dump(mode="json")
        if not self.validation_share_by_ticker:
            out.pop("validation_share_by_ticker")
        if self.pins is None:
            out.pop("pins")
        return out


def fiscal_year(passage: Passage) -> str:
    """The report year (fiscal year end) of a passage's filing, else its filing year."""
    return str(passage.metadata.get("report_date") or passage.filing_date)[:4]


@dataclass(frozen=True)
class SplitResult:
    split_of: dict[str, Split]  # every passage, in corpus order
    group_split: dict[str, Split]
    stats: dict[str, Any]


def _stratum(
    members: tuple[str, ...], passages: Mapping[str, Passage], keys: tuple[str, ...]
) -> tuple[str, ...]:
    values = Counter(
        tuple(passages[m].ticker if k == "ticker" else fiscal_year(passages[m]) for k in keys)
        for m in members
    )
    top = max(values.values())
    return min(v for v, n in values.items() if n == top)


def split_units(
    groups: OverlapGroups, links: Iterable[tuple[str, str]]
) -> dict[str, tuple[str, ...]]:
    """Split units (named after their first group) and their groups, joining linked groups."""
    parent = {gid: gid for gid in groups.members}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for a, b in links:
        ra, rb = find(groups.group_of[a]), find(groups.group_of[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # group IDs are zero-padded corpus positions
    units: dict[str, list[str]] = defaultdict(list)
    for gid in groups.members:
        units[find(gid)].append(gid)
    return {uid: tuple(gids) for uid, gids in units.items()}


def assign_splits(
    passages: Mapping[str, Passage],
    groups: OverlapGroups,
    blocklist: Blocklist,
    settings: SplitSettings,
    links: Iterable[tuple[str, str]] | None = None,
    pins: Mapping[str, str] | None = None,
) -> SplitResult:
    """Assign every passage to train, validation or blocked; see the module docstring."""
    link_list = list(links or ())
    units = split_units(groups, link_list)
    group_split: dict[str, Split] = {}
    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    unit_members: dict[str, tuple[str, ...]] = {}
    for uid, gids in units.items():
        members = tuple(m for g in gids for m in groups.members[g])
        blocked = [m in blocklist for m in members]
        if any(blocked):
            if not all(blocked):
                what = f"group {uid}" if len(gids) == 1 else f"split unit {uid}"
                raise SplitError(f"{what} is only partly blocked")
            for g in gids:
                group_split[g] = "blocked"
            continue
        unit_members[uid] = members
        strata[_stratum(members, passages, settings.stratify_by)].append(uid)
    pinned = _pinned_units(pins or {}, groups, units, blocklist)

    for stratum in sorted(strata):
        uids = sorted(
            strata[stratum],
            key=lambda u: hashlib.sha256(f"{settings.seed}:{u}".encode()).hexdigest(),
        )
        total = sum(len(unit_members[u]) for u in uids)
        target = settings.share_for(stratum) * total
        in_validation = sum(len(unit_members[u]) for u in uids if pinned.get(u) == "validation")
        for uid in uids:
            size = len(unit_members[uid])
            side: Split = "train"
            if uid in pinned:
                side = pinned[uid]
            elif abs(in_validation + size - target) < abs(in_validation - target):
                side = "validation"
                in_validation += size
            for g in units[uid]:
                group_split[g] = side

    split_of: dict[str, Split] = {pid: group_split[groups.group_of[pid]] for pid in passages}
    validate_split(split_of, groups, blocklist, link_list)
    stats = split_statistics(passages, split_of, group_split)
    if links is not None:
        sizes = [len(unit_members[u]) for u in unit_members]
        stats["split_units"] = {
            "links": len(link_list),
            "units": len(unit_members),
            "units_joining_groups": sum(1 for u in unit_members if len(units[u]) > 1),
            "largest_unit": max(sizes, default=0),
        }
    if pins:
        stats["pins"] = {
            "passages": dict(sorted(Counter(pins.values()).items())),
            "units": dict(sorted(Counter(pinned.values()).items())),
            "unit_passages": dict(sorted(Counter(
                side for uid, side in pinned.items() for _ in unit_members[uid]).items())),
        }
    return SplitResult(split_of, group_split, stats)


def _pinned_units(
    pins: Mapping[str, str],
    groups: OverlapGroups,
    units: Mapping[str, tuple[str, ...]],
    blocklist: Blocklist,
) -> dict[str, Split]:
    """The side of every unit holding a pinned passage; a unit pinned both ways or a blocked pin is refused."""
    unit_of_group = {g: uid for uid, gids in units.items() for g in gids}
    pinned: dict[str, Split] = {}
    by: dict[str, str] = {}
    for pid, side in pins.items():
        if side not in ("train", "validation"):
            raise SplitError(f"{pid} is pinned to {side!r}; pins are train or validation")
        if pid not in groups.group_of:
            raise SplitError(f"pinned passage {pid} is not in the corpus")
        if pid in blocklist:
            raise SplitError(f"pinned passage {pid} is on the evaluation blocklist")
        uid = unit_of_group[groups.group_of[pid]]
        if pinned.setdefault(uid, side) != side:  # type: ignore[arg-type]
            raise SplitError(
                f"split unit {uid} is pinned to both {pinned[uid]} (by {by[uid]}) and {side} (by {pid})"
            )
        by.setdefault(uid, pid)
    return pinned


def validate_split(
    split_of: Mapping[str, str],
    groups: OverlapGroups,
    blocklist: Blocklist,
    links: Iterable[tuple[str, str]] = (),
) -> None:
    """Raise ``SplitError`` unless groups and linked passages stay together, splits are disjoint
    and blocked passages are excluded."""
    for gid, members in groups.members.items():
        sides = {split_of[m] for m in members}
        if len(sides) != 1:
            raise SplitError(f"overlap group {gid} spans {sorted(sides)}")
    for a, b in links:
        if split_of[a] != split_of[b]:
            raise SplitError(
                f"{a} and {b} reuse each other's evidence but sit in {split_of[a]} and {split_of[b]}"
            )
    for pid, split in split_of.items():
        if split not in SPLITS:
            raise SplitError(f"{pid} has unknown split {split!r}")
        if (pid in blocklist) != (split == "blocked"):
            raise SplitError(
                f"{pid} is {split} but {'is' if pid in blocklist else 'is not'} on the blocklist"
            )


def split_statistics(
    passages: Mapping[str, Passage], split_of: Mapping[str, str], group_split: Mapping[str, str]
) -> dict[str, Any]:
    by_ticker: dict[str, Counter[str]] = defaultdict(Counter)
    by_year: dict[str, Counter[str]] = defaultdict(Counter)
    for pid, split in split_of.items():
        by_ticker[split][passages[pid].ticker] += 1
        by_year[split][fiscal_year(passages[pid])] += 1
    counts = Counter(split_of.values())
    usable = counts["train"] + counts["validation"]
    return {
        "passages": {s: counts.get(s, 0) for s in SPLITS},
        "groups": {s: sum(1 for v in group_split.values() if v == s) for s in SPLITS},
        "validation_share_of_usable": round(counts["validation"] / usable, 4) if usable else 0.0,
        "by_ticker": {s: dict(sorted(by_ticker[s].items())) for s in SPLITS},
        "by_fiscal_year": {s: dict(sorted(by_year[s].items())) for s in SPLITS},
    }


__all__ = [
    "SPLITS",
    "SplitError",
    "SplitResult",
    "SplitSettings",
    "assign_splits",
    "fiscal_year",
    "split_units",
    "split_statistics",
    "validate_split",
]
