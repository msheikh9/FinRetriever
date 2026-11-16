"""Tests for overlap groups of passages (exact, near-duplicate and transitive overlap; offline)."""

from __future__ import annotations

import hashlib

import pytest

from finretriever.training.overlap import OverlapSettings, build_overlap_groups, shingle_hashes
from tests.unit.training_fixtures import block, chain_texts, make_passage


def passage(name: str, text: str, year: str = "2024", ticker: str = "ACME"):
    chunk = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16)  # unique, stable passage IDs
    return make_passage(name, ticker, f"{ticker.title()} Inc.", f"{year}-12-31", text, chunk=chunk)


def ids(*passages):
    return [p.passage_id for p in passages]


def test_exact_duplicates_are_found_whatever_their_length_or_punctuation() -> None:
    text = "Liquidity remained strong. " + block(1)
    a = passage("a", text)
    b = make_passage("b", "ACME", "Acme Inc.", "2023-12-31", text.upper().replace(".", " ;"), section_index=1)
    short_a = make_passage("short_a", "ACME", "Acme Inc.", "2022-12-31", "None.", section_index=2)
    short_b = make_passage("short_b", "BETA", "Beta Inc.", "2022-12-31", "NONE", section_index=3)
    groups = build_overlap_groups([a, b, short_a, short_b])
    kinds = {(e.a, e.b): e.kind for e in groups.edges}
    assert kinds == {(a.passage_id, b.passage_id): "exact", (short_a.passage_id, short_b.passage_id): "exact"}
    assert groups.group_of[a.passage_id] == groups.group_of[b.passage_id] != groups.group_of[short_a.passage_id]
    assert groups.stats["cross_company_edges"] == 1 and groups.stats["edges_by_kind"] == {"exact": 2}


def test_near_duplicates_and_partial_overlaps_are_classified() -> None:
    base = " ".join(block(i) for i in range(1, 6))
    edited = base.replace("blk3word5", "changed")  # one word in 60: Jaccard stays above 0.8
    shifted = " ".join(block(i) for i in (3, 4, 5, 6, 7))  # 3 of 5 blocks shared
    p1, p2, p3 = passage("p1", base), passage("p2", edited, "2023"), passage("p3", shifted, "2022")
    groups = build_overlap_groups([p1, p2, p3])
    edges = {frozenset((e.a, e.b)): e for e in groups.edges}
    near = edges[frozenset(ids(p1, p2))]
    assert near.kind == "near_duplicate" and near.jaccard >= 0.8 and near.containment >= near.jaccard
    partial = edges[frozenset(ids(p1, p3))]
    assert partial.kind == "overlap" and 0.5 <= partial.containment < 1 and partial.jaccard < 0.8
    assert len(set(groups.group_of.values())) == 1


def test_transitive_overlap_puts_a_chain_in_one_group() -> None:
    a_text, b_text, c_text = chain_texts()
    a, b, c = passage("a", a_text), passage("b", b_text, "2023"), passage("c", c_text, "2022")
    other = passage("other", block(40) + " " + block(41), "2021")
    groups = build_overlap_groups([a, b, c, other])
    pairs = {frozenset((e.a, e.b)) for e in groups.edges}
    assert pairs == {frozenset(ids(a, b)), frozenset(ids(b, c))}  # a and c do not overlap directly
    shared = len(shingle_hashes(a_text, 5) & shingle_hashes(c_text, 5)) / len(shingle_hashes(a_text, 5))
    assert shared < 0.5
    group = groups.group_of[a.passage_id]
    assert groups.group_of[b.passage_id] == groups.group_of[c.passage_id] == group == f"grp_{0:05d}"
    assert groups.members[group] == tuple(ids(a, b, c)) and groups.group_of[other.passage_id] != group
    assert groups.stats["groups_linked_only_transitively"] == 1 and groups.stats["largest_group"] == 3
    rows = {r["passage_id"]: r for r in groups.group_rows()}
    assert rows[b.passage_id]["edges"] == 2 and rows[other.passage_id] == {
        "passage_id": other.passage_id, "group_id": groups.group_of[other.passage_id], "group_size": 1,
        "edges": 0, "max_jaccard": 0.0, "max_containment": 0.0,
    }


def test_boilerplate_shingles_give_no_candidates_and_unrelated_passages_are_not_compared() -> None:
    boilerplate = "this annual report should be read together with the notes to the financial statements"
    passages = [passage(f"p{i}", f"{boilerplate} {block(10 + i, 4)}", str(2010 + i)) for i in range(4)]
    loose = OverlapSettings(max_shingle_df=3, min_shared_shingles=1)
    groups = build_overlap_groups(passages, loose)
    assert groups.stats["candidate_pairs_verified"] == 0 and groups.stats["groups"] == 4  # in 4 passages > max_df
    counted = build_overlap_groups(passages, OverlapSettings(max_shingle_df=4, min_shared_shingles=1))
    assert counted.stats["candidate_pairs_verified"] == 6 and counted.stats["groups"] == 1  # 12/16 shingles shared
    unrelated = [passage(f"u{i}", block(100 + i), str(2000 + i)) for i in range(20)]
    assert build_overlap_groups(unrelated).stats["candidate_pairs_verified"] == 0  # no all-pairs comparison


def test_groups_are_deterministic_and_reject_duplicate_ids() -> None:
    a_text, b_text, c_text = chain_texts()
    passages = [passage("a", a_text), passage("b", b_text, "2023"), passage("c", c_text, "2022")]
    first, second = build_overlap_groups(passages), build_overlap_groups(passages)
    assert first.edges == second.edges and first.group_of == second.group_of and first.stats == second.stats
    with pytest.raises(ValueError, match="unique"):
        build_overlap_groups([passages[0], passages[0]])
    with pytest.raises(ValueError):
        OverlapSettings(containment_threshold=0)
