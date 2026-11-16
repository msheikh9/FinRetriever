"""Tests for sentence- and fact-level evidence reuse, leakage checks and false-negative protection (offline)."""

from __future__ import annotations

from typing import Any

import pytest

from finretriever.data.schemas import Passage
from finretriever.training.blocklist import Blocklist, expand_by_evidence_reuse
from finretriever.training.evidence_reuse import (
    EvidenceLocator,
    ReuseIndex,
    ReuseSettings,
    facts,
    sentence_units,
    significant_values,
)
from finretriever.training.false_negatives import (
    balanced_conflict_free_batches,
    conflict,
    conflict_free_batches,
    filter_negatives,
    negative_exclusions,
)
from finretriever.training.overlap import build_overlap_groups
from finretriever.training.pair_quality import equivalent_ids, evidence_leak
from finretriever.training.splits import SplitError, SplitSettings, assign_splits, validate_split
from tests.unit.training_fixtures import make_passage

SETTINGS = ReuseSettings()  # the defaults are the data_prep_v2 settings
RISK = (
    "Our manufacturing depends on a small number of outsourcing partners located primarily in Asia. "
    "A disruption at any of these partners could delay product shipments for several quarters. "
    "Trade restrictions between the United States and China could raise the cost of our components."
)
SHARED_ELSEWHERE = "Trade restrictions between the United States and China could raise the cost of our components."
TABLE_2022 = (
    "[TABLE] Total operating expenses | Year Ended December 31, 2020: 363,165 | Year Ended December 31, 2021: "
    "444,943 | Year Ended December 31, 2022: 501,735 Operating income | Year Ended December 31, 2020: 22,899 | "
    "Year Ended December 31, 2021: 24,879 | Year Ended December 31, 2022: 12,248 [/TABLE]"
)
TABLE_2023 = (
    "[TABLE] Total operating expenses | Year Ended December 31, 2021: 444,943 | Year Ended December 31, 2022: "
    "501,735 | Year Ended December 31, 2023: 537,933 Operating income | Year Ended December 31, 2021: 24,879 | "
    "Year Ended December 31, 2022: 12,248 | Year Ended December 31, 2023: 36,852 [/TABLE]"
)


def filler(tag: str, sentences: int) -> str:
    """Prose sharing no word with any other filler (each word carries the tag)."""
    return " ".join(
        f"Topic{tag}s{i} covers {tag}w{i}a {tag}w{i}b {tag}w{i}c {tag}w{i}d {tag}w{i}e {tag}w{i}f {tag}w{i}g "
        f"{tag}w{i}h {tag}w{i}i {tag}w{i}j {tag}w{i}k {tag}w{i}l."
        for i in range(sentences)
    )


def passage(name: str, text: str, ticker: str = "AAA", year: str = "2024", chunk: int = 0) -> Passage:
    company = {"AAA": 1, "BBB": 2}[ticker]
    return make_passage(name, ticker, f"{ticker} Corp.", f"{year}-12-31", text, section_index=chunk % 7,
                        chunk=chunk, accession=f"000000000{company}-{year[2:]}-00000{chunk % 10}")


def index_of(passages: list[Passage], visible: dict[str, int] | None = None) -> ReuseIndex:
    return ReuseIndex.build(passages, visible or {p.passage_id: len(p.text) for p in passages}, SETTINGS)


def test_sentences_and_figures_are_units_but_terminology_alone_is_not() -> None:
    assert len(sentence_units(RISK, SETTINGS)) == 3
    assert sentence_units("The allowance for credit losses rose.", SETTINGS) == {}  # a term is never a unit
    found = {str(f) for f in facts(TABLE_2022, "AAA", SETTINGS).values()}
    assert found == {"expenses 363,165", "expenses 444,943", "expenses 501,735", "income 22,899", "income 24,879",
                     "income 12,248"}  # years and the day 31 are never figures
    assert facts("Revenue grew 12% in 2025, and margins held.", "AAA", SETTINGS) == {}
    assert significant_values("Cash rose to $4.3 billion; interest expense was (1,809).", SETTINGS) == {"4.3b", "-1809"}
    # The same figure under the same label is one unit whatever the layout of its row.
    other_layout = facts("Total operating expenses | 2020: $363,165 | 2021: $444,943", "AAA", SETTINGS)
    assert set(other_layout) <= set(facts(TABLE_2022, "AAA", SETTINGS))
    assert not set(facts(TABLE_2022, "BBB", SETTINGS)) & set(facts(TABLE_2022, "AAA", SETTINGS))  # one company only


def reuse_corpus() -> dict[str, Passage]:
    ps = [
        passage("eval", RISK + " " + filler("e", 5), year="2023", chunk=1),
        passage("reuser", RISK + " " + filler("p", 4), year="2024", chunk=2),
        # Shares most of its words with "reuser" (one overlap group) but no evaluation sentence of its own.
        passage("reuser_copy", filler("p", 4) + " " + SHARED_ELSEWHERE, year="2025", chunk=3),
        passage("one_sentence", RISK.split(". ")[0] + ". " + filler("q", 8), year="2025", chunk=4),
        passage("terms", "The allowance for credit losses and the effective tax rate are discussed in Note "
                "four of this report. " + filler("t", 6), chunk=5),
        passage("other_company", SHARED_ELSEWHERE + " " + filler("r", 6), ticker="BBB", chunk=6),
    ]
    return {p.metadata["fixture_name"]: p for p in ps}


def test_cross_year_reuse_below_the_overlap_threshold_expands_the_blocklist_with_whole_groups() -> None:
    by_name = reuse_corpus()
    ps = list(by_name.values())
    ids = {n: p.passage_id for n, p in by_name.items()}
    groups = build_overlap_groups(ps)
    assert groups.group_of[ids["reuser"]] != groups.group_of[ids["eval"]]  # below half the shingles
    assert groups.group_of[ids["reuser"]] == groups.group_of[ids["reuser_copy"]]  # most words shared
    index = index_of(ps)
    assert any(k.startswith("s:") for k in index.generic)  # the sentence two companies use is generic
    blocklist = Blocklist({ids["eval"]: ("annotation_pool:test",)}, (), {"direct": 1})
    expanded, reuse = expand_by_evidence_reuse(blocklist, index, groups, {p.passage_id: p for p in ps})
    assert expanded.reasons[ids["reuser"]] == ("evidence_reuse",)
    assert expanded.reasons[ids["reuser_copy"]] == (f"overlap_group:{groups.group_of[ids['reuser']]}",)
    for name in ("one_sentence", "terms", "other_company"):
        assert ids[name] not in expanded
    shared = reuse[ids["reuser"]]
    assert len(shared.sentences) == 2 and shared.word_share >= SETTINGS.substantial_word_share  # generic one ignored
    one = reuse[ids["one_sentence"]]
    assert len(one.sentences) == 1 and not one.substantial(SETTINGS)  # caught per pair, not per passage
    assert not reuse[ids["terms"]].sentences and not reuse[ids["other_company"]].sentences
    stats = expanded.stats
    assert (stats["evidence_reuse"], stats["via_evidence_reuse_group"], stats["blocked"]) == (1, 1, 3)
    assert expand_by_evidence_reuse(blocklist, index, groups, {p.passage_id: p for p in ps})[0] == expanded


def table_corpus() -> list[Passage]:
    ps = [passage("t22", TABLE_2022 + " " + filler("a", 8), year="2022", chunk=1),
          passage("t23", TABLE_2023 + " " + filler("b", 8), year="2023", chunk=2)]
    for year in ("2022", "2023"):
        ps += [passage(f"f{year}_{i}", filler(f"x{year}{i}", 6), year=year, chunk=10 + i) for i in range(6)]
    return ps


def test_repeated_table_figures_link_comparative_tables_and_keep_them_in_one_split() -> None:
    ps = table_corpus()
    ids = {p.metadata["fixture_name"]: p.passage_id for p in ps}
    groups = build_overlap_groups(ps)
    assert groups.group_of[ids["t22"]] != groups.group_of[ids["t23"]]
    index = index_of(ps)
    links = index.links(ids.values())
    shared = links[(ids["t22"], ids["t23"])]
    assert len(shared.facts) == 4 and shared.fact_share == pytest.approx(4 / 6, abs=1e-4)  # two prior-year columns
    assert len(links) == 1
    passages = {p.passage_id: p for p in ps}
    empty = Blocklist({}, (), {})
    apart = [assign_splits(passages, groups, empty, SplitSettings(seed=s, validation_share=0.3)).split_of
             for s in range(40)]
    assert any(s[ids["t22"]] != s[ids["t23"]] for s in apart)  # without the link they can be split apart
    for seed in range(40):
        result = assign_splits(passages, groups, empty, SplitSettings(seed=seed, validation_share=0.3),
                               links=list(links))
        assert result.split_of[ids["t22"]] == result.split_of[ids["t23"]]
        assert result.stats["split_units"]["links"] == 1
    crossing = dict(apart[next(i for i, s in enumerate(apart) if s[ids["t22"]] != s[ids["t23"]])])
    with pytest.raises(SplitError, match="reuse each other's evidence"):
        validate_split(crossing, groups, empty, list(links))


EVIDENCE = "Total operating expenses | Year Ended December 31, 2023: 537,933"


def leak_corpus(*extra: tuple[str, str, str]) -> tuple[dict[str, Passage], dict[str, dict[str, Any]], EvidenceLocator]:
    base = [("positive", TABLE_2023 + " " + filler("pos", 4), "train"),
            ("same_fact", "[TABLE] Total operating expenses | 2023: $537,933 | 2024: $568,415 [/TABLE] " + filler("sf", 6), "train"),
            ("same_value", "[TABLE] Units shipped | 2023: 537,933 [/TABLE] " + filler("sv", 6), "train"),
            ("unrelated", filler("un", 8), "train"),
            ("other_company", TABLE_2023, "train")]
    ps, rows = {}, {}
    for i, (name, text, split) in enumerate([*base, *extra]):
        p = passage(name, text, ticker="BBB" if name == "other_company" else "AAA", year="2024", chunk=i)
        ps[name] = p
        rows[p.passage_id] = {"split": split, "group_id": f"grp_{i:05d}"}
    visible = {p.passage_id: len(p.text) for p in ps.values()}
    if "hidden" in ps:  # its copy of the evidence lies beyond the tokenizer-visible prefix
        visible[ps["hidden"].passage_id] = len(filler("hd", 8))
    index = ReuseIndex.build(list(ps.values()), visible, SETTINGS)
    return ps, rows, EvidenceLocator({p.passage_id: p for p in ps.values()}, index)


def test_a_shared_figure_in_another_group_is_an_equivalent_positive_not_a_negative() -> None:
    ps, rows, locator = leak_corpus()
    eq = locator.locate(EVIDENCE, "$537,933", "AAA", ps["positive"].passage_id)
    by_name = {n: next((e for e in eq if e.passage_id == p.passage_id), None) for n, p in ps.items()}
    assert by_name["same_fact"].match == "fact" and by_name["same_fact"].exact  # exact: label and figure
    assert by_name["same_value"].match == "value" and not by_name["same_value"].exact  # uncertain: other label
    assert by_name["unrelated"] is None and by_name["other_company"] is None  # other companies never match
    ids = equivalent_ids(eq, rows, "train")
    assert ids == {"exact": [ps["same_fact"].passage_id], "uncertain": [ps["same_value"].passage_id]}
    assert evidence_leak(eq, rows, "train") is None
    pair = {"pair_id": "train_000001", "positive_passage_id": ps["positive"].passage_id,
            "group_id": rows[ps["positive"].passage_id]["group_id"], "equivalents": ids}
    members = {r["group_id"]: [pid] for pid, r in rows.items()}
    assert ps["same_fact"].passage_id in negative_exclusions(pair, members)
    candidates = [ps[n].passage_id for n in ("same_fact", "same_value", "unrelated")]
    assert filter_negatives(pair, candidates, members) == [ps["unrelated"].passage_id]


def test_evidence_held_by_an_evaluation_or_other_split_passage_is_a_leak() -> None:
    ps, rows, locator = leak_corpus(("blocked", "Results " + EVIDENCE + " " + filler("bl", 4), "blocked"))
    eq = locator.locate(EVIDENCE, "$537,933", "AAA", ps["positive"].passage_id)
    assert evidence_leak(eq, rows, "train") == "evaluation_evidence"
    ps, rows, locator = leak_corpus(("validation", EVIDENCE + " " + filler("va", 4), "validation"))
    eq = locator.locate(EVIDENCE, "$537,933", "AAA", ps["positive"].passage_id)
    assert evidence_leak(eq, rows, "train") == "cross_split_evidence"
    # Beyond the visible prefix, another split's copy is not seen by the encoder or a query writer.
    ps, rows, locator = leak_corpus(("hidden", filler("hd", 8) + " " + EVIDENCE, "validation"))
    eq = locator.locate(EVIDENCE, "$537,933", "AAA", ps["positive"].passage_id)
    hidden = next(e for e in eq if e.passage_id == ps["hidden"].passage_id)
    assert hidden.exact and not hidden.visible_exact and evidence_leak(eq, rows, "train") is None


def test_batches_never_hold_a_pair_beside_a_passage_that_answers_it() -> None:
    members = {"gA": ["A", "C"], "gB": ["B"], "gD": ["D"], "gE": ["E"], "gF": ["F"]}
    pairs = [
        {"pair_id": "a", "positive_passage_id": "A", "group_id": "gA", "equivalents": {"exact": ["B"], "uncertain": []}},
        {"pair_id": "b", "positive_passage_id": "B", "group_id": "gB", "equivalents": {"exact": [], "uncertain": []}},
        {"pair_id": "c", "positive_passage_id": "C", "group_id": "gA", "equivalents": {"exact": [], "uncertain": []}},
        {"pair_id": "d", "positive_passage_id": "D", "group_id": "gD", "equivalents": {"exact": [], "uncertain": ["E"]}},
        {"pair_id": "e", "positive_passage_id": "E", "group_id": "gE", "equivalents": {"exact": [], "uncertain": []}},
        {"pair_id": "f", "positive_passage_id": "F", "group_id": "gF", "equivalents": {"exact": [], "uncertain": []}},
    ]
    by_id = {p["pair_id"]: p for p in pairs}
    assert conflict(by_id["a"], by_id["b"], members) and conflict(by_id["b"], by_id["a"], members)  # either direction
    assert conflict(by_id["a"], by_id["c"], members) and conflict(by_id["d"], by_id["e"], members)
    assert not conflict(by_id["a"], by_id["f"], members)
    for seed in range(20):
        batches = conflict_free_batches(pairs, 3, seed, members)
        assert sorted(p for b in batches for p in b) == sorted(by_id) and all(len(b) <= 3 for b in batches)
        for batch in batches:
            assert not any(conflict(by_id[x], by_id[y], members) for x in batch for y in batch if x < y)
        assert batches == conflict_free_batches(list(reversed(pairs)), 3, seed, members)  # deterministic
    with pytest.raises(ValueError, match="at least 1"):
        conflict_free_batches(pairs, 0, 1, members)
    for seed in range(20):  # the balanced planner keeps the same conflicts apart, in equal batches
        batches = balanced_conflict_free_batches(pairs, 3, seed, members)
        assert sorted(p for b in batches for p in b) == sorted(by_id) and sorted(len(b) for b in batches) == [3, 3]
        for batch in batches:
            assert not any(conflict(by_id[x], by_id[y], members) for x in batch for y in batch if x < y)
        assert batches == balanced_conflict_free_batches(list(reversed(pairs)), 3, seed, members)
    with pytest.raises(ValueError, match="at least 1"):
        balanced_conflict_free_batches(pairs, 0, 1, members)
    with pytest.raises(ValueError, match="repeat"):
        balanced_conflict_free_batches([*pairs, pairs[0]], 3, 1, members)


def test_the_reuse_index_is_deterministic() -> None:
    ps = list(reuse_corpus().values()) + table_corpus()
    first, second = index_of(ps), index_of(ps)
    assert first.digest() == second.digest() and first.stats() == second.stats()
    assert first.links(first.order) == second.links(second.order)
    changed = [*ps[:-1], passage("changed", ps[-1].text + " One more figure: 9,999.", year="2023", chunk=99)]
    assert index_of(changed).digest() != first.digest()
