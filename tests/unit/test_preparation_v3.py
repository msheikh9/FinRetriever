"""A new preparation that keeps existing data in place: per-company validation shares, split pins, and the migration
of training candidates and validation questions into it (offline, synthetic corpus)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from finretriever.training.blocklist import Blocklist
from finretriever.training.candidates import load_candidates
from finretriever.training.data_prep import (
    load_data_prep_config,
    open_prepared,
    verify_frozen_corpus,
)
from finretriever.training.overlap import OverlapGroups, OverlapSettings
from finretriever.training.split_pins import SplitPinsError, load_split_pins, write_split_pins
from finretriever.training.splits import SplitError, SplitSettings, assign_splits
from finretriever.training.validation_set import load_validation_set
from tests.unit.training_fixtures import make_passage, write_prep_config
from tests.unit.validation_fixtures import EXTRA, World, block_network, build_world, cli, tree


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    block_network(monkeypatch)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return build_world(tmp_path, monkeypatch)

# --- the split: per-company shares and pins -------------------------------------------------------


def toy() -> tuple[dict[str, Any], OverlapGroups, Blocklist, list[tuple[str, str]]]:
    """Two companies of 20 one-passage groups; AAA's first two are linked into one split unit; BBB's last is blocked."""
    passages = {}
    for t, accession in (("AAA", "0000000001-25-000001"), ("BBB", "0000000002-25-000001")):
        for i in range(20):
            p = make_passage(f"{t}{i}", t, f"{t} Corp.", "2024-12-31", f"{t} passage {i}", chunk=i, accession=accession)
            passages[f"{t}{i}"] = p
    by_id = {p.passage_id: p for p in passages.values()}
    groups = OverlapGroups(settings=OverlapSettings(), group_of={pid: f"g_{pid}" for pid in by_id},
                           members={f"g_{pid}": (pid,) for pid in by_id}, edges=())
    blocklist = Blocklist(reasons={passages["BBB19"].passage_id: ("dev_pool:test",)}, sources=(), stats={})
    return passages, groups, blocklist, [(passages["AAA0"].passage_id, passages["AAA1"].passage_id)]


def split(settings: SplitSettings, pins: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    passages, groups, blocklist, links = toy()
    by_id = {p.passage_id: p for p in passages.values()}
    result = assign_splits(by_id, groups, blocklist, settings, links=links, pins=pins)
    return passages, {name: result.split_of[p.passage_id] for name, p in passages.items()}


def count(sides: dict[str, str], ticker: str) -> int:
    return sum(1 for name, side in sides.items() if name.startswith(ticker) and side == "validation")


def test_company_shares_and_pins_change_the_targets_but_units_still_move_whole() -> None:
    passages, plain = split(SplitSettings(seed=3, validation_share=0.25))
    _, same = split(SplitSettings(seed=3, validation_share=0.25, validation_share_by_ticker={}), pins={})
    assert plain == same  # no shares and no pins: the earlier split exactly
    _, tuned = split(SplitSettings(seed=3, validation_share=0.25, validation_share_by_ticker={"AAA": 0.45, "BBB": 0.1}))
    assert count(tuned, "AAA") in (8, 9, 10) and count(tuned, "BBB") in (1, 2, 3)
    assert tuned["AAA0"] == tuned["AAA1"] and tuned["BBB19"] == "blocked"

    pid = {name: p.passage_id for name, p in passages.items()}
    pins = {pid["AAA1"]: "validation", **{pid[f"BBB{i}"]: "validation" for i in range(5)}, pid["AAA5"]: "train"}
    _, pinned = split(SplitSettings(seed=3, validation_share=0.25), pins=pins)
    assert pinned["AAA1"] == pinned["AAA0"] == "validation"  # a pinned passage takes its whole unit
    assert pinned["AAA5"] == "train" and all(pinned[f"BBB{i}"] == "validation" for i in range(5))
    assert count(pinned, "BBB") == 5  # pinned passages count toward the target (a quarter of 19)

    for bad, message in (({pid["AAA0"]: "train", pid["AAA1"]: "validation"}, "pinned to both"),
                         ({pid["BBB19"]: "validation"}, "evaluation blocklist"),
                         ({"passage_unknown": "train"}, "not in the corpus"),
                         ({pid["AAA2"]: "blocked"}, "train or validation")):
        with pytest.raises(SplitError, match=message):
            split(SplitSettings(seed=3, validation_share=0.25), pins=bad)


def test_split_settings_record_new_fields_only_when_set() -> None:
    plain = SplitSettings(seed=1)
    assert set(plain.manifest_settings()) == {"seed", "validation_share", "strategy", "stratify_by"}
    tuned = SplitSettings(seed=1, validation_share_by_ticker={"AAA": 0.4}, pins="pins.json")
    assert tuned.manifest_settings()["validation_share_by_ticker"] == {"AAA": 0.4}
    assert tuned.manifest_settings()["pins"] == "pins.json" and tuned.share_for(("AAA", "2024")) == 0.4
    assert tuned.share_for(("BBB", "2024")) == 0.10
    with pytest.raises(ValidationError, match="validation shares must be in"):
        SplitSettings(seed=1, validation_share_by_ticker={"AAA": 0.6})
    with pytest.raises(ValidationError, match="needs ticker"):
        SplitSettings(seed=1, stratify_by=("fiscal_year",), validation_share_by_ticker={"AAA": 0.3})


# --- the migration: pins from an earlier preparation, re-checked data, a new validation plan ------


def migrate(world: World, name: str, pins: Path, **split: Any) -> tuple[Path, Any]:
    config_path = write_prep_config(
        world.root, world.root / f"{name}.yaml", version=name,
        output_dir=str(world.root / "data" / "processed" / "train" / name),
        split={"pins": str(pins), **split}, source_quality={"context_rules": True},
    )
    code, out, err = cli(config_path, "prepare")
    assert code == 0, err
    config = load_data_prep_config(config_path)[0]
    return config_path, open_prepared(config.output_dir, _corpus(world), world.passages)


def _corpus(world: World) -> Any:
    return verify_frozen_corpus(world.config)


def test_a_new_preparation_keeps_training_and_validation_data_where_they_are(world: World) -> None:
    config_v1 = world.validation_config("val_v1", world.review("val_decisions"))
    assert cli(world.config_path, "build-validation", "--validation", str(config_v1))[0] == 0
    earlier = {name: tree(world.directory / name) for name in ("pairs", "candidates", "generation", "plans",
                                                               "validation_sets")}
    pins = world.root / "pins.json"
    argv = ("pin-split", "--candidates", "cand_v1", "--pairs", "--keep-validation", "--out", str(pins))
    code, out, err = cli(world.config_path, *argv)
    assert code == 0 and "wrote" in out, err
    assert cli(world.config_path, *argv)[1].count("unchanged (identical pins present)") == 1  # written once
    doc = json.loads(pins.read_text())
    v1_rows = world.prepared.rows
    assert doc["counts"]["validation"] == sum(r["split"] == "validation" for r in v1_rows.values())
    assert all(p["reasons"] for p in doc["pins"])

    new_path, new = migrate(world, "test_v2", pins, validation_share_by_ticker={"GAMMA": 0.45})
    for name in ("overlap_groups.jsonl", "overlap_edges.jsonl", "blocklist.jsonl", "reuse_links.jsonl"):
        assert (world.directory / name).read_bytes() == (new.directory / name).read_bytes()
    moves = {(v1_rows[p]["split"], new.rows[p]["split"]) for p in v1_rows}
    assert moves <= {("train", "train"), ("validation", "validation"), ("blocked", "blocked"), ("train", "validation")}
    assert all(new.rows[p["passage_id"]]["split"] == p["side"] for p in doc["pins"])
    assert new.manifest["split"]["pins"]["sha256"] and new.manifest["split"]["settings"]["pins"] == str(pins)

    # The training candidates, re-checked in the new preparation: the same pairs, the same outcomes.
    (world.root / "cand_v2.yaml").write_text(yaml.safe_dump({
        "version": "cand_v2", "data_prep_config": str(new_path),
        "sources": [{"name": "fixture", "data_prep_config": str(world.config_path),
                     "decisions": str(world.root / "decisions.jsonl")}]}))
    assert cli(new_path, "build-candidates", "--candidates", str(world.root / "cand_v2.yaml"))[0] == 0
    config_new = load_data_prep_config(new_path)[0]
    old_c = load_candidates(world.prepared, world.passages, world.config, "cand_v1", "train")
    new_c = load_candidates(new, world.passages, config_new, "cand_v2", "train")
    assert [(c.query, c.positive_passage_id) for c in old_c] == [(c.query, c.positive_passage_id) for c in new_c]

    # The validation questions, carried over and re-checked: every outcome is logged.
    (world.root / "val_v2.yaml").write_text(yaml.safe_dump({
        "version": "val_v2", "data_prep_config": str(new_path),
        "carry_over": [{"data_prep_config": str(world.config_path), "version": "val_v1"}],
        "training_candidates": ["cand_v2"]}))
    code, out, err = cli(new_path, "build-validation", "--validation", str(world.root / "val_v2.yaml"))
    assert code == 0 and "carried val_v1: 2 of 2 questions included" in out, err
    carried = load_validation_set(new, world.passages, config_new, "val_v2", training=new_c)
    old_v = load_validation_set(world.prepared, world.passages, world.config, "val_v1")
    assert [(v.pair_id, v.query) for v in carried] == [(v.pair_id, v.query) for v in old_v]
    assert {v.provenance.origin for v in carried} == {"carryover:val_v1"}
    log = [json.loads(line) for line in (new.directory / "validation_sets/val_v2/validation_log.jsonl").open()]
    assert [r["carried_from"] for r in log] == ["val_v1", "val_v1"] and all(r["quality"]["gold"] is False for r in log)

    # A new plan skips every unit the earlier validation set touched and records its options.
    plan_argv = ("plan", "--split", "validation", "--name", "val_c", "--size", "1", "--method", "balanced_years",
                 "--max-company-share", "1.0", "--one-per", "group", "--min-isolated-share", "0.0",
                 "--exclude-validation-set", f"{world.config_path}:val_v1")
    code, out, err = cli(new_path, *plan_argv)
    assert code == 0 and "estimated usable questions" in out, err
    plan = json.loads((new.directory / "plans" / "validation.val_c.json").read_text())
    newly_validation = {p for p in v1_rows if v1_rows[p]["split"] == "train" and new.rows[p]["split"] == "validation"}
    assert {p["passage_id"] for p in plan["passages"]} <= newly_validation  # the earlier set's units are skipped
    assert plan["constraints"]["one_per_split_unit"] is False and plan["constraints"]["scores"]["min_score"] == 0.0
    assert {name: tree(world.directory / name) for name in earlier} == earlier  # the earlier preparation is untouched


def test_a_carried_question_whose_passage_left_validation_is_logged_and_left_out(world: World) -> None:
    config_v1 = world.validation_config("val_v1", world.review("val_decisions"))
    assert cli(world.config_path, "build-validation", "--validation", str(config_v1))[0] == 0
    pins = world.root / "pins_train_only.json"
    assert cli(world.config_path, "pin-split", "--candidates", "cand_v1", "--pairs", "--out", str(pins))[0] == 0
    new_path, new = migrate(world, "test_v2b", pins, validation_share=0.01)  # no validation pins, tiny share
    moved = [pid for pid in (EXTRA["gamma24_debt"], EXTRA["gamma22_liquidity"]) if new.rows[pid]["split"] != "validation"]
    assert moved  # the earlier validation passages were not pinned, so the new split moved them
    (world.root / "val_v2b.yaml").write_text(yaml.safe_dump({
        "version": "val_v2b", "data_prep_config": str(new_path),
        "carry_over": [{"data_prep_config": str(world.config_path), "version": "val_v1"}]}))
    code, out, err = cli(new_path, "build-validation", "--validation", str(world.root / "val_v2b.yaml"))
    assert code == 0, err
    log = {r["positive_passage_id"]: r for r in map(
        json.loads, (new.directory / "validation_sets/val_v2b/validation_log.jsonl").open())}
    for pid in moved:
        assert not log[pid]["included"] and "positive_not_validation" in log[pid]["reasons"]
    assert "excluded: positive_not_validation" in out


def test_pins_files_are_written_once_and_checked_when_read(world: World, tmp_path: Path) -> None:
    corpus = _corpus(world)
    pid = EXTRA["gamma24_debt"]
    doc = {"format_version": 1, "kind": "split_pins", "notice": "test",
           "corpus": {"sha256": corpus.corpus_sha256, "version": corpus.corpus_version}, "source": {},
           "counts": {"validation": 1}, "pins": [{"passage_id": pid, "side": "validation", "reasons": ["test"]}]}
    path = tmp_path / "pins.json"
    assert write_split_pins(path, doc) and not write_split_pins(path, doc)
    with pytest.raises(SplitPinsError, match="never changed"):
        write_split_pins(path, {**doc, "notice": "other"})
    assert load_split_pins(path, corpus, world.passages)[0] == {pid: "validation"}
    cases = [({"corpus": {"sha256": "0" * 64, "version": "x"}}, "another corpus"),
             ({"pins": [{"passage_id": "nope", "side": "train", "reasons": ["t"]}], "counts": {"train": 1}}, "not in the corpus"),
             ({"pins": [{"passage_id": pid, "side": "validation", "reasons": []}]}, "without a reason"),
             ({"pins": [doc["pins"][0], doc["pins"][0]], "counts": {"validation": 2}}, "listed twice"),
             ({"counts": {"validation": 2}}, "counts do not match"),
             ({"kind": "other"}, "not a format-1")]
    for i, (change, message) in enumerate(cases):
        bad = tmp_path / f"bad{i}.json"
        bad.write_text(json.dumps({**doc, **change}))
        with pytest.raises(SplitPinsError, match=message):
            load_split_pins(bad, corpus, world.passages)
