"""Final models in the benchmark harness: only the checkpoint its run's validation rule selected, verified by its
selection record and hashes, for every seed; smoke-test and fixed-step checkpoints never. Fake checkpoints, offline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from finretriever.data.schemas import Qrel, Query
from finretriever.evaluation.benchmark import (
    CHECKPOINT_CAUTION,
    SELECTED_CHECKPOINT_CAUTION,
    build_retriever,
    run_eval_set,
    write_eval_set_outputs,
)
from finretriever.evaluation.isolation import (
    ISOLATION_POST_TRAINING_ELIGIBLE,
    ISOLATION_POST_TRAINING_FULL,
    ISOLATION_STRICT,
    IsolationError,
    checkpoint_isolation,
)
from finretriever.evaluation.provenance import load_corpus_identity
from finretriever.retrieval import checkpoint as checkpoint_module
from finretriever.retrieval.checkpoint import inspect_checkpoint
from finretriever.training.checkpoint_selection import (
    SelectionState,
    checkpoint_files,
    files_digest,
    make_selection_record,
    write_selection_record,
)
from finretriever.utils.io import read_jsonl
from tests.unit.checkpoint_fixtures import HIDDEN, candidate, write_checkpoint, write_training_side
from tests.unit.dense_fixtures import FakeEncoder
from tests.unit.eval_set_fixtures import passage_ids, write_dev_eval_set
from tests.unit.test_checkpoint_evaluation import checkpoint_config, training_candidates

IDS = passage_ids()
SEEDS = (42, 1337, 2026)
CANDIDATE = "a candidate of the fixed validation rule; which checkpoint the rule selects is recorded in ../selection.json"


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    """The development set of the eval-set fixtures, a training side, a smoke-like fixed-step run, and a validated run
    of three seeds, each with its selection record."""
    eval_set = write_dev_eval_set(tmp_path)
    corpus = Path(eval_set["config"]["corpus_path"])
    judged = sorted({json.loads(line)["passage_id"] for line in (eval_set["labels"] / "dev_qrels.jsonl").read_text().splitlines()})
    inputs = write_training_side(tmp_path / "training", corpus, blocked=judged, candidates=training_candidates())
    smoke = [write_checkpoint(tmp_path / "models" / "fixture-seed42", step, inputs) for step in (5, 10)]
    world = {**eval_set, "corpus": corpus, "judged": judged, "inputs": inputs, "checkpoints": smoke}
    world["validation"] = write_validation_set(world, [("val_000001", IDS["acme24_sales"], "What were Acme's 2024 sales?")])
    world["runs"] = {seed: selected_run(world, seed) for seed in SEEDS}
    return world


def write_validation_set(world: Mapping[str, Any], questions: Sequence[tuple[str, str, str]], *,
                         name: str = "val_fixture") -> dict[str, Any]:
    """A validation set in the training side's preparation; returns what a checkpoint records about it."""
    directory = world["tmp"] / "training" / "prep" / "validation_sets" / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = [{"pair_id": pid, "positive_passage_id": positive, "query": query, "split": "validation",
             "equivalents": {"exact": [], "uncertain": []}} for pid, positive, query in questions]
    (directory / "validation_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = {"validation_pairs.jsonl": hashlib.sha256((directory / "validation_pairs.jsonl").read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps({"kind": "validation_pairs", "files": files}))
    return {"version": name, "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            "pairs": len(rows)}


def selected_run(world: Mapping[str, Any], seed: int, values: Mapping[int, float] | None = None, *,
                 name: str = "fixture_ft", validation: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Checkpoints at the rule's steps (0, 5, 10) of one seed's run, hashed, and its written selection record."""
    run = f"{name}-seed{seed}"
    run_dir = world["tmp"] / "models" / run
    inputs = {**world["inputs"], "validation": dict(validation or world["validation"])}
    state = SelectionState()
    for step, value in (values or {0: 0.30, 5: 0.50 + seed / 1e5, 10: 0.40}).items():
        directory = write_checkpoint(run_dir, step, inputs, weights=f"{run} {step}".encode(), run=run)
        metadata = json.loads((directory / "metadata.json").read_text())
        metadata["selection"] = CANDIDATE
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        entry = state.record(step, {"global": {"mrr@10": value}})
        files = checkpoint_files(directory)
        entry.update(checkpoint=directory.name, files=files, digest=files_digest(files))
    record = make_selection_record(
        run=run, seed=seed, seeds=SEEDS, config_sha256="c" * 64, schedule_sha256="d" * 64,
        run_record_sha256=hashlib.sha256((run_dir / "run.json").read_bytes()).hexdigest(),
        validation=inputs["validation"], state=state, planned_steps=10, steps_done=10,
    )
    write_selection_record(run_dir, record)
    return {"record": run_dir / "selection.json", "selected": run_dir / record["selected"]["checkpoint"],
            "run_dir": run_dir}


def final(world: Mapping[str, Any], seed: int, **changes: Any) -> dict[str, Any]:
    run = world["runs"][seed]
    return {"name": f"ft_seed{seed}", "type": "dense_checkpoint", "role": "final", "checkpoint_path": str(run["selected"]),
            "selection_record": str(run["record"]), "batch_size": 4, **changes}


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loaded: list[str] = []

    def load(ckpt: Any, **kwargs: Any) -> FakeEncoder:
        loaded.append(ckpt.model_id)
        return FakeEncoder(dim=HIDDEN)

    monkeypatch.setattr(checkpoint_module, "load_checkpoint_encoder", load)
    return loaded


def test_every_seeds_selected_checkpoint_is_scored_with_its_verified_selection(
    world: dict[str, Any], fake_loader: list[str]
) -> None:
    config = checkpoint_config(world, retrievers=[final(world, seed) for seed in SEEDS],
                               output_dir=str(world["tmp"] / "results" / "dev" / "test_final"))
    result = run_eval_set(config, "ckpt")
    assert "collection" not in result.settings and "isolation_mode" not in result.settings  # strict sets: unchanged
    runs = [r for r in result.runs if not r.reference]
    assert sorted({r.run_name for r in runs}) == sorted(f"ft_seed{s}" for s in SEEDS)
    for run in runs:
        chosen = run.retriever["validation_selection"]
        seed = int(run.run_name.removeprefix("ft_seed"))
        assert run.retriever["role"] == "final" and chosen["seed"] == seed and chosen["selected_step"] == 5
        assert chosen["digest"] == files_digest(checkpoint_files(world["runs"][seed]["selected"]))
        assert chosen["record"]["sha256"] == hashlib.sha256(world["runs"][seed]["record"].read_bytes()).hexdigest()
        assert run.retriever["evaluation_isolation"]["validation"]["passages_used_and_judged"] == 0
    write_eval_set_outputs(result)
    out = world["tmp"] / "results" / "dev" / "test_final"
    payload = json.loads((out / "global" / "ft_seed42" / "metrics.json").read_text())
    assert payload["checkpoint_caution"] == SELECTED_CHECKPOINT_CAUTION
    report = (out / "report.md").read_text()
    assert SELECTED_CHECKPOINT_CAUTION in report and CHECKPOINT_CAUTION not in report
    assert "final model: selected by its run's validation rule at step 5, seed 42 of [42, 1337, 2026]" in report
    assert "selecting validation set val_fixture: 0 of 1 passages" in report


def test_a_final_model_must_be_its_runs_validated_selection(world: dict[str, Any], fake_loader: list[str]) -> None:
    run = world["runs"][42]
    # The smoke-test checkpoints: no selection record, so never a final model (as diagnostics they still run).
    smoke = {"name": "smoke", "type": "dense_checkpoint", "checkpoint_path": str(world["checkpoints"][1])}
    with pytest.raises(ValueError, match="registered as a final model without a selection_record"):
        build_retriever({**smoke, "role": "final"})
    assert build_retriever(smoke).role == "diagnostic"
    # Another run's record, or another checkpoint of this run, is not this checkpoint's selection.
    with pytest.raises(ValueError, match="is not the checkpoint run fixture_ft-seed42 selected"):
        build_retriever({**smoke, "role": "final", "selection_record": str(run["record"])})
    with pytest.raises(ValueError, match="is not the checkpoint run"):
        build_retriever(final(world, 42, checkpoint_path=str(run["run_dir"] / "checkpoint-step-000010")))
    with pytest.raises(ValueError, match="role 'best'"):
        build_retriever(final(world, 42, role="best"))
    # A selected checkpoint changed after selection is refused.
    weights = run["selected"] / "model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(b"other weights")
    try:
        with pytest.raises(ValueError, match="differs from the checkpoint the selection hashed"):
            build_retriever(final(world, 42))
    finally:
        weights.write_bytes(original)
    # A run record changed after selection is refused.
    run_file = run["run_dir"] / "run.json"
    saved = run_file.read_bytes()
    run_file.write_bytes(saved + b" ")
    try:
        with pytest.raises(ValueError, match="changed since the selection was recorded"):
            build_retriever(final(world, 42))
    finally:
        run_file.write_bytes(saved)
    # A fixed-step checkpoint with a forged record that names it: refused by its own metadata.
    forged = selected_run(world, 42, name="forged")
    metadata = json.loads((forged["selected"] / "metadata.json").read_text())
    metadata["selection"] = "none: fixed-step checkpoint; no development or evaluation metric was computed"
    (forged["selected"] / "metadata.json").write_text(json.dumps(metadata, indent=2))
    record = json.loads(forged["record"].read_text())
    files = checkpoint_files(forged["selected"])
    for entry in record["evaluations"]:
        if entry["checkpoint"] == forged["selected"].name:
            entry.update(files=files, digest=files_digest(files))
    record["selected"]["digest"] = files_digest(files)
    forged["record"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="saved at a fixed step without validation"):
        build_retriever({"name": "forged", "type": "dense_checkpoint", "role": "final",
                         "checkpoint_path": str(forged["selected"]), "selection_record": str(forged["record"])})
    assert fake_loader == []


def test_no_seed_is_left_out_or_repeated(world: dict[str, Any], fake_loader: list[str]) -> None:
    for retrievers in ([final(world, 42), final(world, 1337)],
                       [final(world, 42), final(world, 1337), final(world, 2026), {**final(world, 42), "name": "again"}]):
        with pytest.raises(ValueError, match="every seed's selected checkpoint is reported, once"):
            run_eval_set(checkpoint_config(world, retrievers=retrievers), "ckpt")
    assert fake_loader == []


def test_the_selecting_validation_set_must_be_apart_from_the_evaluation_set(world: dict[str, Any]) -> None:
    labels = world["labels"]
    corpus = load_corpus_identity(world["corpus"])
    queries, qrels = read_jsonl(labels / "dev_queries.jsonl", model=Query), read_jsonl(labels / "dev_qrels.jsonl", model=Qrel)
    record = checkpoint_isolation(inspect_checkpoint(world["runs"][42]["selected"]), corpus=corpus, queries=queries,
                                  qrels=qrels)
    assert record["validation"]["questions"] == 1 and "validation_method" in record
    question = queries[0].text
    for name, rows, message in (
        ("judged", [("val_000001", world["judged"][0], "Which passage?")], "selected with an evaluation passage"),
        ("repeated", [("val_000001", IDS["acme24_sales"], question)], "repeats evaluation question"),
    ):
        validation = write_validation_set(world, rows, name=f"val_{name}")
        run = selected_run(world, 42, name=name, validation=validation)
        with pytest.raises(IsolationError, match=message):
            checkpoint_isolation(inspect_checkpoint(run["selected"]), corpus=corpus, queries=queries, qrels=qrels)
    # The selecting set changed after the checkpoint recorded it.
    pairs = world["tmp"] / "training" / "prep" / "validation_sets" / "val_fixture" / "validation_pairs.jsonl"
    pairs.write_text(pairs.read_text() + "\n")
    with pytest.raises(IsolationError, match="validation pairs .* does not match"):
        checkpoint_isolation(inspect_checkpoint(world["runs"][42]["selected"]), corpus=corpus, queries=queries,
                             qrels=qrels)


def _post_training_checkpoint(world: Mapping[str, Any], name: str, candidates: list[dict[str, Any]],
                              objective: Mapping[str, Any] | None = None) -> Any:
    """A checkpoint whose preparation was frozen before the evaluation set: an empty blocklist, as for gold v1."""
    inputs = write_training_side(world["tmp"] / name, world["corpus"], blocked=[], candidates=candidates)
    directory = write_checkpoint(world["tmp"] / "models" / f"{name}-seed42", 5, inputs)
    if objective is not None:
        metadata = json.loads((directory / "metadata.json").read_text())
        metadata["objective"] = dict(objective)
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return inspect_checkpoint(directory)


def test_isolation_modes_for_an_evaluation_set_built_after_training(world: dict[str, Any]) -> None:
    """D8: strict refuses any judged passage off the frozen blocklist; the post-training modes record it instead,
    refuse (eligible) or count (full corpus) judged training and validation passages, and always refuse a repeated
    question."""
    labels = world["labels"]
    corpus = load_corpus_identity(world["corpus"])
    queries, qrels = read_jsonl(labels / "dev_queries.jsonl", model=Query), read_jsonl(labels / "dev_qrels.jsonl", model=Qrel)

    def isolation(ckpt: Any, mode: str) -> dict[str, Any]:
        return checkpoint_isolation(ckpt, corpus=corpus, queries=queries, qrels=qrels, mode=mode)

    clean = _post_training_checkpoint(world, "clean", training_candidates())
    with pytest.raises(IsolationError, match="not on the training blocklist"):
        isolation(clean, ISOLATION_STRICT)
    record = isolation(clean, ISOLATION_POST_TRAINING_ELIGIBLE)
    assert record["isolated"] and record["mode"] == ISOLATION_POST_TRAINING_ELIGIBLE
    assert record["blocklist"]["judged_passages_blocked"] == 0  # recorded, not required
    full = isolation(clean, ISOLATION_POST_TRAINING_FULL)
    assert full["potentially_contaminated"] is False and full["contamination"]["judged_training_passages"] == 0

    trained = _post_training_checkpoint(
        world, "trained", training_candidates(train_000003=candidate("train_000003", world["judged"][0], "Which?")))
    for mode in (ISOLATION_STRICT, ISOLATION_POST_TRAINING_ELIGIBLE):
        with pytest.raises(IsolationError, match="uses judged passage"):
            isolation(trained, mode)
    full = isolation(trained, ISOLATION_POST_TRAINING_FULL)
    assert full["potentially_contaminated"] is True and full["contamination"]["judged_training_passages"] == 1

    repeated = _post_training_checkpoint(
        world, "repeated", training_candidates(train_000003=candidate("train_000003", IDS["acme24_table"],
                                                                      queries[0].text)))
    for mode in (ISOLATION_STRICT, ISOLATION_POST_TRAINING_ELIGIBLE, ISOLATION_POST_TRAINING_FULL):
        with pytest.raises(IsolationError, match="repeats evaluation question"):
            isolation(repeated, mode)

    mined = _post_training_checkpoint(world, "mined", training_candidates(), objective={"hard_negatives_per_query": 4})
    with pytest.raises(IsolationError, match="mined hard negatives"):
        isolation(mined, ISOLATION_POST_TRAINING_ELIGIBLE)

    # A selecting validation set that uses a judged passage: refused when eligible, counted on the full corpus.
    validation = write_validation_set(world, [("val_000001", world["judged"][0], "Which passage?")], name="val_judged")
    run = selected_run(world, 42, name="vjudged", validation=validation)
    with pytest.raises(IsolationError, match="selected with an evaluation passage"):
        isolation(inspect_checkpoint(run["selected"]), ISOLATION_POST_TRAINING_ELIGIBLE)
    full = isolation(inspect_checkpoint(run["selected"]), ISOLATION_POST_TRAINING_FULL)
    assert full["potentially_contaminated"] is True and full["contamination"]["judged_validation_passages"] == 1
    with pytest.raises(ValueError, match="unknown isolation mode"):
        isolation(clean, "lenient")
    strict = checkpoint_isolation(inspect_checkpoint(world["runs"][42]["selected"]), corpus=corpus, queries=queries,
                                  qrels=qrels)
    assert "mode" not in strict and "potentially_contaminated" not in strict  # the default record is unchanged
    assert strict == checkpoint_isolation(inspect_checkpoint(world["runs"][42]["selected"]), corpus=corpus,
                                          queries=queries, qrels=qrels, mode=ISOLATION_STRICT)


def test_an_exclusion_set_is_removed_before_ranking_for_every_retriever(
    world: dict[str, Any], fake_loader: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """D8: with an exclusion set, every retriever (the checkpoints and BM25) indexes only the eligible passages, the
    excluded passages' judgments are dropped, and the collection is recorded."""
    from finretriever.evaluation import benchmark, exclusion

    unjudged = next(pid for pid in IDS.values() if pid not in world["judged"])
    excluded = frozenset({world["judged"][0], unjudged})
    monkeypatch.setitem(exclusion.EXCLUSIONS, "fixture_exclusion", {"description": "fixture", "sources": {}})
    monkeypatch.setattr(benchmark, "exclusion_ids", lambda name: excluded)
    bm25 = {"name": "bm25", "type": "bm25", "tokenizer": "word", "backend": "python"}
    config = checkpoint_config(world, retrievers=[final(world, seed) for seed in SEEDS] + [bm25],
                               reference_runs_from=None, exclude_passages="fixture_exclusion",
                               isolation=ISOLATION_POST_TRAINING_ELIGIBLE,
                               output_dir=str(world["tmp"] / "results" / "dev" / "test_eligible"))
    result = run_eval_set(config, "ckpt")
    runs = [r for r in result.runs if not r.reference]
    assert sorted({r.run_name for r in runs}) == sorted([f"ft_seed{s}" for s in SEEDS] + ["bm25"])
    assert all(hit.passage_id not in excluded for run in runs for hit in run.hits)
    collection = result.settings["collection"]
    corpus_size = len(read_jsonl(world["corpus"]))
    assert (collection["corpus_passages"], collection["excluded_passages"], collection["eligible_passages"]) == (
        corpus_size, 2, corpus_size - 2)
    qrels = read_jsonl(world["labels"] / "dev_qrels.jsonl", model=Qrel)
    assert collection["judgments_excluded"] == sum(q.passage_id == world["judged"][0] for q in qrels) > 0
    assert collection["embedding_cache_key"] != load_corpus_identity(world["corpus"]).corpus_sha256
    assert result.settings["isolation_mode"] == ISOLATION_POST_TRAINING_ELIGIBLE
    for run in runs:
        if run.run_name != "bm25":
            assert run.retriever["evaluation_isolation"]["mode"] == ISOLATION_POST_TRAINING_ELIGIBLE
    write_eval_set_outputs(result)
    payload = json.loads((world["tmp"] / "results" / "dev" / "test_eligible" / "global" / "ft_seed42" / "metrics.json")
                         .read_text())
    assert payload["settings"]["collection"]["excluded_passages"] == 2


@pytest.mark.parametrize(("changes", "message"), [
    ({"exclude_passages": "fixture_exclusion"}, "goes with an exclusion set"),
    ({"isolation": ISOLATION_POST_TRAINING_ELIGIBLE}, "goes with an exclusion set"),
    ({"exclude_passages": "nonexistent", "isolation": ISOLATION_POST_TRAINING_ELIGIBLE}, "unknown exclusion set"),
    ({"exclude_passages": "fixture_exclusion", "isolation": ISOLATION_POST_TRAINING_ELIGIBLE},
     "stored runs were ranked on another corpus"),
    ({"isolation": "lenient"}, "unknown isolation mode"),
    ({"isolation": ISOLATION_POST_TRAINING_FULL}, "for final sets only"),
])
def test_exclusion_and_isolation_settings_are_checked(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any], message: str
) -> None:
    from finretriever.evaluation import exclusion

    monkeypatch.setitem(exclusion.EXCLUSIONS, "fixture_exclusion", {"description": "fixture", "sources": {}})
    with pytest.raises(ValueError, match=message):
        run_eval_set(checkpoint_config(world, **changes), "ckpt")
