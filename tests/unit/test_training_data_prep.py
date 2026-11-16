"""Tests for the evaluation blocklist, the group split, visible text and the preparation manifest (offline)."""

from __future__ import annotations

import hashlib
import json
import re
import socket
from pathlib import Path
from typing import Any

import pytest

from finretriever.evaluation.provenance import CorpusMismatchError
from finretriever.training.blocklist import (
    Blocklist,
    BlocklistError,
    BlocklistSources,
    build_blocklist,
)
from finretriever.training.data_prep import (
    DataPrepError,
    load_data_prep_config,
    open_prepared,
    prepare,
    verify_frozen_corpus,
    write_prepared,
)
from finretriever.training.overlap import build_overlap_groups
from finretriever.training.splits import SplitError, SplitSettings, assign_splits, validate_split
from finretriever.training.visible_text import visible_span
from tests.unit.eval_set_fixtures import ExpandedInputs, build_expanded_inputs, passage_ids
from tests.unit.training_fixtures import (
    WhitespaceTokenizer,
    extra_ids,
    training_extra_passages,
    write_prep_config,
)

FIXTURE = passage_ids()
EXTRA = extra_ids()
CHAIN = ("gamma24_mdna", "gamma23_mdna", "gamma22_mdna")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def inputs(tmp_path: Path) -> ExpandedInputs:
    return build_expanded_inputs(tmp_path, training_extra_passages())


def sources(root: Path, **overrides: Any) -> BlocklistSources:
    values = {
        "annotation_workspaces": [str(root / "annotation")], "dev_pools": [str(root / "dev_ai_v3" / "pool")],
        "audit_workspaces": [str(root / "audit")], "gold_qrels": str(root / "gold_qrels.jsonl"),
        "gold_metadata": str(root / "gold_eval.metadata.json"), **overrides,
    }
    (root / "gold_qrels.jsonl").touch()
    return BlocklistSources(**values)


def tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_blocklist_blocks_every_evaluation_candidate_and_its_overlap_group(inputs: ExpandedInputs) -> None:
    passages = inputs.passages
    groups = build_overlap_groups(list(passages.values()))
    blocklist = build_blocklist(sources(inputs.root), corpus=inputs.corpus, passages=passages, groups=groups)
    assert set(FIXTURE.values()) <= set(blocklist.reasons)  # every pooled fixture passage
    assert "annotation_pool:" in " ".join(blocklist.reasons[FIXTURE["acme24_competition"]])
    copy = blocklist.reasons[EXTRA["acme22_competition"]]  # an exact copy of a pooled passage, never pooled itself
    assert copy == (f"overlap_group:{groups.group_of[FIXTURE['acme24_competition']]}",)
    assert not any(EXTRA[n] in blocklist for n in EXTRA if n.startswith("gamma"))
    assert blocklist.stats["direct"] == 12 and blocklist.stats["via_overlap_group"] == 1
    assert {s["kind"] for s in blocklist.sources} == {"annotation_pool", "dev_pool", "audit_sample", "gold_qrels"}
    assert all(set(row) == {"passage_id", "reasons"} for row in blocklist.rows(list(passages)))  # IDs and reasons only
    again = build_blocklist(sources(inputs.root), corpus=inputs.corpus, passages=passages, groups=groups)
    assert again == blocklist


def test_blocklist_fails_closed(inputs: ExpandedInputs) -> None:
    root, passages = inputs.root, inputs.passages
    groups = build_overlap_groups(list(passages.values()))

    def build(**overrides: Any) -> Blocklist:
        return build_blocklist(sources(root, **overrides), corpus=inputs.corpus, passages=passages, groups=groups)

    with pytest.raises(BlocklistError, match="does not exist"):
        build(dev_pools=[str(root / "missing_pool")])
    with pytest.raises(BlocklistError, match="no evaluation source"):
        build_blocklist(BlocklistSources(), corpus=inputs.corpus, passages=passages, groups=groups)
    meta = root / "dev_ai_v3" / "pool" / "pool.meta.json"
    original = meta.read_text()
    meta.write_text(original.replace(inputs.corpus.corpus_sha256, "f" * 64))
    with pytest.raises(CorpusMismatchError):  # a pool bound to another corpus
        build()
    meta.write_text(original)
    qrel = {"query_id": "g1", "passage_id": EXTRA["gamma24_debt"], "relevance": 2}
    (root / "gold_qrels.jsonl").write_text(json.dumps(qrel) + "\n")
    with pytest.raises(BlocklistError, match="metadata .* is missing"):
        build()
    metadata = {"corpus_sha256": "f" * 64, "corpus_version": inputs.corpus.corpus_version, "annotation_version": "gold_v1"}
    (root / "gold_eval.metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(CorpusMismatchError):
        build()
    (root / "gold_eval.metadata.json").write_text(json.dumps({**metadata, "corpus_sha256": inputs.corpus.corpus_sha256}))
    blocklist = build()
    assert blocklist.reasons[EXTRA["gamma24_debt"]] == ("gold_qrels",)
    (root / "gold_qrels.jsonl").write_text(json.dumps({**qrel, "passage_id": "passage_unknown"}) + "\n")
    with pytest.raises(BlocklistError, match="outside the corpus"):
        build()


def test_split_assigns_whole_groups_deterministically_and_rejects_leaks(inputs: ExpandedInputs) -> None:
    passages = inputs.passages
    groups = build_overlap_groups(list(passages.values()))
    blocklist = build_blocklist(sources(inputs.root), corpus=inputs.corpus, passages=passages, groups=groups)
    settings = SplitSettings(seed=7, validation_share=0.3)
    result = assign_splits(passages, groups, blocklist, settings)
    assert {result.split_of[EXTRA[n]] for n in CHAIN} in ({"train"}, {"validation"})  # one group, one side
    assert all(result.split_of[pid] == "blocked" for pid in blocklist.reasons)
    assert {s for pid, s in result.split_of.items() if pid not in blocklist} <= {"train", "validation"}
    stats = result.stats
    assert stats["passages"]["blocked"] == 13 and sum(stats["passages"].values()) == len(passages)
    assert stats["passages"]["validation"] > 0 and stats["by_ticker"]["validation"].keys() <= {"GAMMA"}
    assert assign_splits(passages, groups, blocklist, settings).split_of == result.split_of
    others = [assign_splits(passages, groups, blocklist, SplitSettings(seed=s, validation_share=0.3)).split_of for s in range(30)]
    assert any(o != result.split_of for o in others)  # the seed matters

    leaked = dict(result.split_of)
    member = EXTRA[CHAIN[0]]
    leaked[member] = "validation" if leaked[member] == "train" else "train"
    with pytest.raises(SplitError, match="spans"):
        validate_split(leaked, groups, blocklist)
    unblocked = {**result.split_of, FIXTURE["acme24_sales"]: "train"}
    with pytest.raises(SplitError, match="blocklist"):
        validate_split(unblocked, groups, blocklist)
    partial = Blocklist({EXTRA[CHAIN[0]]: ("test",)}, (), {})
    with pytest.raises(SplitError, match="only partly blocked"):
        assign_splits(passages, groups, partial, settings)


class CharTokenizer(WhitespaceTokenizer):
    """One token per character, so truncation can cut a word."""

    def __call__(self, text: str, *, max_length: int = 512, **_: Any) -> dict[str, Any]:
        spans = [(i, i + 1) for i in range(len(text))][: max_length - 2]
        return {"offset_mapping": [(0, 0), *spans, (0, 0)], "special_tokens_mask": [1] + [0] * len(spans) + [1]}


def test_visible_span_counts_special_and_prefix_tokens_and_never_cuts_a_word() -> None:
    text = " ".join(f"w{i}" for i in range(50))
    span = visible_span(text, WhitespaceTokenizer(), max_seq_length=12)
    assert (span.tokens, span.truncated, text[: span.chars]) == (10, True, " ".join(f"w{i}" for i in range(10)))
    prefixed = visible_span(text, WhitespaceTokenizer(), max_seq_length=12, prefix="passage: ")
    assert prefixed.tokens == 9 and text[: prefixed.chars].split()[-1] == "w8"
    short = visible_span("all of it", WhitespaceTokenizer(), max_seq_length=12)
    assert (short.chars, short.truncated) == (9, False)
    cut = visible_span("alpha beta gamma", CharTokenizer(), max_seq_length=10)  # 8 characters: "alpha be"
    assert "alpha beta gamma"[: cut.chars] == "alpha" and cut.truncated


def test_prepare_is_reproducible_versioned_and_bound_to_the_frozen_corpus(inputs: ExpandedInputs) -> None:
    root = inputs.root
    config_path = write_prep_config(root)
    config, config_sha = load_data_prep_config(config_path)
    corpus = verify_frozen_corpus(config)
    protected = {name: tree(root / name) for name in ("annotation", "audit", "dev_ai_v3", "dev_ai_v2", "corpus")}

    def run(cfg: Any = config) -> dict[str, bytes]:
        return prepare(cfg, config_path=str(config_path), config_sha256=config_sha, corpus=corpus,
                       passages=inputs.passages, tokenizer=WhitespaceTokenizer())

    files = run()
    assert run() == files  # same inputs, byte-identical outputs
    assert write_prepared(config.output_dir, files) is True and write_prepared(config.output_dir, files) is False
    prepared = open_prepared(config.output_dir, corpus, inputs.passages)
    manifest = prepared.manifest
    assert manifest["corpus"]["sha256"] == corpus.corpus_sha256 and manifest["config"]["sha256"] == config_sha
    assert manifest["overlap"]["settings"] == config.overlap.model_dump() and "connected components" in manifest["overlap"]["method"]
    assert manifest["split"]["settings"]["seed"] == 7 and manifest["split"]["settings"]["strategy"] == "overlap_group_stratified"
    assert manifest["blocklist"]["stats"]["blocked"] == 13
    assert manifest["visible_text"]["identity"]["tokenizer_sha256"] == hashlib.sha256(b"whitespace-tokenizer-v1").hexdigest()
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((Path(config.output_dir) / name).read_bytes()).hexdigest() == digest
    assert not re.search(r"\d{4}-\d{2}-\d{2}T", files["manifest.json"].decode())  # no timestamps
    rows = prepared.rows
    assert rows[EXTRA["gamma24_mdna"]]["truncated"] and rows[EXTRA["gamma24_mdna"]]["visible_tokens"] == 38
    assert {name: tree(root / name) for name in protected} == protected  # evaluation data and corpus only read

    with pytest.raises(DataPrepError, match="different preparation"):
        write_prepared(config.output_dir, run(config.model_copy(update={"split": SplitSettings(seed=8, validation_share=0.3)})))
    passages_file = Path(config.output_dir) / "passages.jsonl"
    passages_file.write_text(passages_file.read_text().replace('"split":"train"', '"split":"validation"', 1))
    with pytest.raises(DataPrepError, match="does not match"):
        open_prepared(config.output_dir, corpus, inputs.passages)


def test_config_keeps_outputs_gitignored_and_the_corpus_frozen(inputs: ExpandedInputs) -> None:
    root = inputs.root
    for bad in (root / "outputs", root / "data" / "eval" / "train"):
        with pytest.raises(DataPrepError, match="data/processed|data/eval"):
            load_data_prep_config(write_prep_config(root, output_dir=str(bad)))
    config, _ = load_data_prep_config(write_prep_config(root, corpus_sha256="e" * 64))
    with pytest.raises(CorpusMismatchError):
        verify_frozen_corpus(config)


def test_a_config_edit_outside_the_preparation_keeps_it_but_any_other_change_is_refused(inputs: ExpandedInputs) -> None:
    from finretriever.training.data_prep import prepared_status

    root = inputs.root
    config_path = write_prep_config(root)

    def files() -> dict[str, bytes]:
        config, sha = load_data_prep_config(config_path)
        return prepare(config, config_path=str(config_path), config_sha256=sha, corpus=verify_frozen_corpus(config),
                       passages=inputs.passages, tokenizer=WhitespaceTokenizer())

    config = load_data_prep_config(config_path)[0]
    assert prepared_status(config.output_dir, files()) == "absent" and write_prepared(config.output_dir, files())
    before = tree(Path(config.output_dir))
    write_prep_config(root, config_path, generation={"prompt_version": "sec-qgen-v2", "temperature": 0.5})
    assert prepared_status(config.output_dir, files()) == "config_edited"
    assert write_prepared(config.output_dir, files()) is False and tree(Path(config.output_dir)) == before
    write_prep_config(root, config_path, split={"seed": 8, "validation_share": 0.3})
    assert prepared_status(config.output_dir, files()) == "different"
    with pytest.raises(DataPrepError, match="different preparation"):
        write_prepared(config.output_dir, files())
