"""The semantic leakage screen: permitted evaluation question texts only, write-once records with provenance, use
by the candidate builder only (offline; a fake encoder stands in for the zero-shot model)."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from finretriever.training.candidates import candidates_dir, load_candidates
from finretriever.training.semantic_screen import (
    FORBIDDEN_FIELDS,
    ScreeningError,
    load_permitted_questions,
    read_screening,
    screen,
    screening_dir,
)
from tests.unit.validation_fixtures import World, block_network, build_world, cli, tree

DIM = 64


class FakeEncoder:
    """Hashed bag of words, L2-normalised; texts listed in ``twins`` are embedded as their twin is."""

    def __init__(self, twins: dict[str, str] | None = None) -> None:
        self.twins = twins or {}
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            vector = np.zeros(DIM, dtype=np.float32)
            for word in self.twins.get(text, text).casefold().split():
                vector[zlib.crc32(word.encode()) % DIM] += 1.0
            rows.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
        return np.stack(rows)

    def describe(self) -> dict[str, Any]:
        return {"model": "fake/bag-of-words", "revision": "0" * 40, "query_prefix": "", "normalize": True}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    block_network(monkeypatch)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return build_world(tmp_path, monkeypatch)


def screen_cli(world: World, *argv: str, encoder: FakeEncoder) -> tuple[int, str, str]:
    from finretriever.training.data_cli import main

    out, err = io.StringIO(), io.StringIO()
    code = main(["screen-questions", "--config", str(world.config_path), *argv], out=out, err=err,
                screen_encoder=lambda config: encoder)
    return code, out.getvalue(), err.getvalue()


def evaluation_texts(world: World) -> list[tuple[str, str]]:
    return load_permitted_questions(world.config)[0]


def first_candidate(world: World) -> dict[str, Any]:
    path = candidates_dir(world.prepared, "cand_v1") / "train_candidates.jsonl"
    return json.loads(path.read_text().splitlines()[0])


def test_the_screen_reads_permitted_question_texts_only(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    real_open = io.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.path.realpath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    code, out, err = screen_cli(world, "--name", "screen_v1", "--candidates", "cand_v1", "--pairs",
                                encoder=FakeEncoder())
    assert code == 0, err
    root = os.path.realpath(world.root)
    permitted = {os.path.realpath(p) for p in world.config.leakage.evaluation_question_files}
    evaluation_dirs = [os.path.join(root, d) for d in ("annotation", "dev_ai_v3", "audit")]
    touched = {f for f in opened if any(f.startswith(d + os.sep) for d in evaluation_dirs) or "gold_" in f}
    assert touched == permitted & touched and touched  # only the question files, of all evaluation data
    assert not [f for f in opened if "qrels" in os.path.basename(f) or f"{os.sep}pool" in f or "audit" in f]
    assert "query_id and text only" in out and "no qrels, relevance labels, judged passages" in out


def test_the_record_flags_a_paraphrase_and_states_its_provenance(world: World) -> None:
    question = evaluation_texts(world)[0]
    pair = first_candidate(world)
    encoder = FakeEncoder(twins={pair["query"]: question[1]})  # a paraphrase the lexical check cannot see
    code, out, err = screen_cli(world, "--name", "screen_v1", "--candidates", "cand_v1", encoder=encoder)
    assert code == 0, err
    record = read_screening(world.prepared, "screen_v1")
    manifest, rows = record["manifest"], record["rows"]
    flagged = rows[hashlib.sha256(pair["query"].encode()).hexdigest()]
    assert flagged["flagged"] and flagged["max_similarity"] == pytest.approx(1.0)
    assert flagged["nearest_evaluation_question"] == question[0] and flagged["subject"] == f"candidates:cand_v1:{pair['pair_id']}"
    assert sum(r["flagged"] for r in rows.values()) == manifest["stats"]["flagged"] == 1
    assert manifest["threshold"] == 0.9 and manifest["evaluation_questions"]["fields_read"] == ["query_id", "text"]
    assert {f["path"] for f in manifest["evaluation_questions"]["files"]} == set(world.config.leakage.evaluation_question_files)
    assert all(len(f["sha256"]) == 64 for f in manifest["evaluation_questions"]["files"])
    assert manifest["encoder"]["model"] == "fake/bag-of-words" and "never by training" in manifest["notice"]
    assert manifest["subjects"]["sources"][0]["version"] == "cand_v1"
    text = (screening_dir(world.prepared, "screen_v1") / "screening.jsonl").read_text()
    assert question[1] not in text  # the evaluation question's text is never written
    # Written once.
    written = tree(screening_dir(world.prepared, "screen_v1"))
    assert screen_cli(world, "--name", "screen_v1", "--candidates", "cand_v1", encoder=encoder)[0] == 0
    code, _, err = screen_cli(world, "--name", "screen_v1", "--candidates", "cand_v1", encoder=FakeEncoder())
    assert code == 1 and "never written over" in err
    assert tree(screening_dir(world.prepared, "screen_v1")) == written
    assert screen_cli(world, "--name", "nothing", encoder=encoder)[0] == 1  # nothing named to screen


def test_a_file_with_labels_or_judged_passages_is_refused_before_encoding(world: World, tmp_path: Path) -> None:
    for name, row in (("questions_with_labels.jsonl", {"query_id": "q1", "text": "Q?", "passage_id": "p", "relevance": 2}),
                      ("dev_qrels.jsonl", {"query_id": "q1", "text": "Q?"})):
        path = tmp_path / name
        path.write_text(json.dumps(row) + "\n")
        config = world.config.model_copy(update={"leakage": world.config.leakage.model_copy(
            update={"evaluation_question_files": [str(path)]})})
        with pytest.raises(ScreeningError, match="relevance labels and judged passages are never read|qrels file"):
            load_permitted_questions(config)
    assert {"passage_id", "relevance", "qrels", "judged", "pool"} <= FORBIDDEN_FIELDS
    encoder = FakeEncoder()
    assert screen([], [("q", "Q?")], encoder) == [] and encoder.calls == []


def test_candidates_built_with_the_screen_exclude_flagged_and_unscreened_pairs(world: World) -> None:
    pair = first_candidate(world)
    question = evaluation_texts(world)[0]
    assert screen_cli(world, "--name", "screen_v1", "--pairs", encoder=FakeEncoder(twins={pair["query"]: question[1]}))[0] == 0
    decisions = Path(json.loads((candidates_dir(world.prepared, "cand_v1") / "manifest.json").read_text())
                     ["sources"][0]["decisions"]["path"])
    config = world.root / "cand_screened.yaml"
    config.write_text(yaml.safe_dump({"version": "cand_screened", "data_prep_config": str(world.config_path),
                                      "sources": [{"name": "fixture", "decisions": str(decisions)}],
                                      "semantic_screening": "screen_v1"}))
    code, _, err = cli(world.config_path, "build-candidates", "--candidates", str(config))
    assert code == 0, err
    directory = candidates_dir(world.prepared, "cand_screened")
    log = {r["candidate_id"]: r for r in map(json.loads, (directory / "candidates_log.jsonl").read_text().splitlines())}
    flagged = log[pair["pair_id"]]
    assert not flagged["included"] and "semantic_evaluation_overlap" in flagged["reasons"]
    assert flagged["max_semantic_similarity"] == pytest.approx(1.0)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["checks"]["semantic_overlap"]["screening"] == "screen_v1"
    assert manifest["checks"]["semantic_overlap"]["manifest_sha256"] == read_screening(world.prepared, "screen_v1")["manifest_sha256"]
    loaded = load_candidates(world.prepared, world.passages, world.config, "cand_screened", "train")
    assert pair["pair_id"] not in {p.pair_id for p in loaded} and loaded
    # The dataset built without a screen is unchanged by this feature: no semantic field anywhere.
    original = json.loads((candidates_dir(world.prepared, "cand_v1") / "manifest.json").read_text())
    assert "semantic_overlap" not in original["checks"]
    assert "max_semantic_similarity" not in (candidates_dir(world.prepared, "cand_v1") / "candidates_log.jsonl").read_text()
    # A screen that misses a question excludes that pair rather than letting it through unscreened.
    assert screen_cli(world, "--name", "screen_partial", "--candidates", "cand_screened", encoder=FakeEncoder())[0] == 0
    partial = world.root / "cand_partial.yaml"
    partial.write_text(config.read_text().replace("cand_screened", "cand_partial").replace("screen_v1", "screen_partial"))
    assert cli(world.config_path, "build-candidates", "--candidates", str(partial))[0] == 0
    log = {r["candidate_id"]: r for r in map(json.loads, (candidates_dir(world.prepared, "cand_partial")
                                                          / "candidates_log.jsonl").read_text().splitlines())}
    assert "not_semantically_screened" in log[pair["pair_id"]]["reasons"]


def test_loading_candidates_for_training_never_opens_the_screen(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    pair = first_candidate(world)
    assert screen_cli(world, "--name", "screen_v1", "--pairs", encoder=FakeEncoder())[0] == 0
    decisions = Path(json.loads((candidates_dir(world.prepared, "cand_v1") / "manifest.json").read_text())
                     ["sources"][0]["decisions"]["path"])
    config = world.root / "cand_screened.yaml"
    config.write_text(yaml.safe_dump({"version": "cand_screened", "data_prep_config": str(world.config_path),
                                      "sources": [{"name": "fixture", "decisions": str(decisions)}],
                                      "semantic_screening": "screen_v1"}))
    assert cli(world.config_path, "build-candidates", "--candidates", str(config))[0] == 0
    opened: list[str] = []
    real_open = io.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.path.realpath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    pairs = load_candidates(world.prepared, world.passages, world.config, "cand_screened", "train")
    assert pairs and pair["pair_id"] in {p.pair_id for p in pairs}
    assert not [f for f in opened if f"{os.sep}screening{os.sep}" in f]
