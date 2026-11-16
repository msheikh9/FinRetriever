"""A small validation world for the validation-set and preparation tests: a synthetic corpus prepared, training
pairs and candidates built, and a three-passage validation plan generated through a mocked API (offline)."""

from __future__ import annotations

import hashlib
import io
import json
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from finretriever.data.schemas import Passage
from finretriever.eval_set.annotation_store import load_corpus_passages
from finretriever.training.data_cli import main
from finretriever.training.data_prep import (
    DataPrepConfig,
    PreparedData,
    load_data_prep_config,
    open_prepared,
    verify_frozen_corpus,
)
from finretriever.training.pair_quality import load_evaluation_questions
from finretriever.training.validation_set import (
    REVIEW_ASPECTS,
    build_validation_set,
    load_validation_config,
    pair_id_for,
)
from tests.unit.eval_set_fixtures import build_expanded_inputs
from tests.unit.training_fixtures import (
    SECRET,
    MockTransport,
    WhitespaceTokenizer,
    accept_all,
    extra_ids,
    training_extra_passages,
    write_prep_config,
)

EXTRA = extra_ids()
PLAN = "val_a"
PASS = dict.fromkeys(REVIEW_ASPECTS, "pass")
VALIDATION_QUESTIONS = {  # passage text marker -> the one question the mocked model writes for it
    "$900 million of borrowings": {
        "question": "How much had Gamma Energy drawn on its revolving credit facility at December 31, 2024?",
        "type": "debt_liquidity", "answer": "$900 million",
        "evidence": "At December 31, 2024, Gamma Energy had $900 million of borrowings outstanding"},
    "revenue of $3,036 million for 2023": {  # a train passage states this figure as its prior year
        "question": "What revenue did Gamma Energy report for fiscal 2023?", "type": "revenue",
        "answer": "$3,036 million", "evidence": "Gamma Energy reported revenue of $3,036 million for 2023"},
    "$410 million of cash": {
        "question": "How much cash did Gamma Energy hold when fiscal 2022 ended?", "type": "debt_liquidity",
        "answer": "$410 million", "evidence": "Gamma Energy ended 2022 with $410 million of cash"},
}


def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No socket may connect: every API call goes to a mocked transport."""

    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


def validation_reply(payload: Mapping[str, Any]) -> Any:
    user = payload["messages"][-1]["content"]
    return {"questions": [q for marker, q in VALIDATION_QUESTIONS.items() if marker in user]}


def cli(config: Path, *argv: str, transport: Any = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main([argv[0], "--config", str(config), *argv[1:]], tokenizer_loader=lambda settings: WhitespaceTokenizer(),
                transport=transport, sleep=lambda seconds: None, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@dataclass
class World:
    root: Path
    config_path: Path
    config: DataPrepConfig
    prepared: PreparedData
    passages: dict[str, Passage]

    @property
    def directory(self) -> Path:
        return Path(self.config.output_dir)

    def records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (self.directory / "generation" / "validation.records.jsonl").read_text().splitlines()]

    def plan_ids(self) -> list[str]:
        return [p["passage_id"] for p in json.loads((self.directory / "plans" / f"validation.{PLAN}.json").read_text())["passages"]]

    def review(self, name: str, decide: Callable[[str], tuple[str, dict[str, str]]] = lambda pid: ("accept", PASS)) -> Path:
        """AI review decisions for the plan's questions; ``decide(passage_id)`` gives the decision and aspects."""
        state = {r["passage_id"]: r for r in self.records()}
        rows = []
        for position, pid in enumerate(self.plan_ids(), start=1):
            for candidate in state[pid]["accepted"]:
                decision, aspects = decide(pid)
                rows.append({"pair_id": pair_id_for(position), "passage_id": pid, "decision": decision,
                             "question_sha256": hashlib.sha256(candidate["question"].encode()).hexdigest(),
                             "issues": [], "aspects": aspects, "reviewer": "ai_review", "gold": False,
                             **({} if decision == "accept" else {"reason": "test"})})
        path = self.root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def validation_config(self, version: str, decisions: Path, candidates: tuple[str, ...] = ("cand_v1",)) -> Path:
        path = self.root / f"{version}.yaml"
        path.write_text(yaml.safe_dump({"version": version, "data_prep_config": str(self.config_path), "plan": PLAN,
                                        "decisions": str(decisions), "training_candidates": list(candidates)}))
        return path

    def build(self, config_path: Path) -> Any:
        config, digest = load_validation_config(config_path)
        return build_validation_set(
            config, config_path=str(config_path), config_sha256=digest, prepared=self.prepared, prep_config=self.config,
            prep_config_sha256=load_data_prep_config(self.config_path)[1], passages=self.passages,
            evaluation_questions=load_evaluation_questions(self.config.leakage.evaluation_question_files))


def build_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """Training pairs and candidates built, then a three-passage validation plan generated (mocked API)."""
    build_expanded_inputs(tmp_path, training_extra_passages())
    config_path = write_prep_config(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    for argv in (("prepare",), ("generate", "--split", "train", "--confirm-api-use"), ("build-pairs",)):
        assert cli(config_path, *argv, transport=MockTransport())[0] == 0
    config = load_data_prep_config(config_path)[0]
    (tmp_path / "cand_v1.yaml").write_text(yaml.safe_dump({
        "version": "cand_v1", "data_prep_config": str(config_path),
        "sources": [{"name": "fixture", "decisions": str(accept_all(Path(config.output_dir), tmp_path / "decisions.jsonl"))}]}))
    assert cli(config_path, "build-candidates", "--candidates", str(tmp_path / "cand_v1.yaml"))[0] == 0
    code, _, err = cli(config_path, "plan", "--split", "validation", "--name", PLAN, "--size", "3", "--method",
                       "balanced_years", "--max-company-share", "1.0")
    assert code == 0, err
    argv = ("generate", "--split", "validation", "--plan", PLAN, "--max-requests", "3", "--confirm-api-use")
    code, _, err = cli(config_path, *argv, transport=MockTransport(validation_reply))
    assert code == 0, err
    corpus = verify_frozen_corpus(config)
    passages = load_corpus_passages(corpus)
    return World(tmp_path, config_path, config, open_prepared(config.output_dir, corpus, passages), passages)


def log_of(world: World, version: str) -> dict[str, dict[str, Any]]:
    path = world.directory / "validation_sets" / version / "validation_log.jsonl"
    return {json.loads(line)["positive_passage_id"]: json.loads(line) for line in path.read_text().splitlines()}
