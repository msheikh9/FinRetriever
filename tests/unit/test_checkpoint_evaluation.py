"""Local fine-tuned checkpoints in the benchmark harness: safe loading, embedding dimensions, ranking, provenance,
evaluation isolation, uncertainty and write-once outputs. Fake checkpoints and encoders only (offline)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from finretriever.data.schemas import Qrel, Query
from finretriever.evaluation.benchmark import (
    CHECKPOINT_CAUTION,
    build_retriever,
    evaluate_eval_set_run,
    load_eval_set,
    run_eval_set,
    write_eval_set_outputs,
)
from finretriever.evaluation.isolation import IsolationError, checkpoint_isolation
from finretriever.evaluation.provenance import load_corpus_identity
from finretriever.evaluation.uncertainty import mean_interval, paired_difference
from finretriever.retrieval import biencoder
from finretriever.retrieval import checkpoint as checkpoint_module
from finretriever.retrieval.biencoder import MODEL_FORMATS, BiEncoderRetriever, DenseModelError
from finretriever.retrieval.checkpoint import (
    REQUIRED_FILES,
    CheckpointError,
    CheckpointRetriever,
    inspect_checkpoint,
    load_checkpoint_encoder,
)
from finretriever.utils.io import read_jsonl
from tests.unit.checkpoint_fixtures import (
    BGE,
    BGE_REVISION,
    HIDDEN,
    MODULES,
    FakeModels,
    FakeSentenceTransformer,
    FakeTokenizer,
    FakeTorch,
    candidate,
    write_checkpoint,
    write_training_side,
)
from tests.unit.dense_fixtures import FakeEncoder
from tests.unit.eval_set_fixtures import fixture_passages, passage_ids, write_dev_eval_set

IDS = passage_ids()
QUESTION = "Which single supplier does Acme depend on?"


def tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def training_candidates(**changes: Any) -> list[dict[str, Any]]:
    rows = {
        "train_000001": candidate("train_000001", IDS["acme24_table"], "What does Acme's revenue table show for 2024?"),
        "train_000002": candidate("train_000002", IDS["beta24_business"], "Which banking businesses does Beta Bancorp run?",
                                  uncertain=[IDS["beta24_credit"]]),
    }
    return list({**rows, **changes}.values())


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    """The development evaluation set of the eval-set fixtures, a training side that blocks every judged passage, and
    two checkpoints of one run trained on it."""
    eval_set = write_dev_eval_set(tmp_path)
    corpus = Path(eval_set["config"]["corpus_path"])
    judged = sorted({json.loads(line)["passage_id"] for line in (eval_set["labels"] / "dev_qrels.jsonl").read_text().splitlines()})
    inputs = write_training_side(tmp_path / "training", corpus, blocked=judged, candidates=training_candidates())
    run_dir = tmp_path / "models" / "fixture-seed42"
    checkpoints = [write_checkpoint(run_dir, step, inputs, weights=f"weights {step}".encode()) for step in (5, 10)]
    return {**eval_set, "corpus": corpus, "judged": judged, "inputs": inputs, "checkpoints": checkpoints}


# --- inspecting and loading a checkpoint ---------------------------------------------------------


def test_a_checkpoint_is_inspected_and_hashed_before_anything_loads(world: dict[str, Any]) -> None:
    first, second = (inspect_checkpoint(path) for path in world["checkpoints"])
    assert first.model_id == "local:fixture-seed42/checkpoint-step-000005"
    assert first.revision == f"sha256:{first.digest}" and len(first.digest) == 64
    assert set(first.files) == set(REQUIRED_FILES)
    assert first.files["model.safetensors"] == hashlib.sha256(b"weights 5").hexdigest()
    assert (first.hidden_size, first.max_seq_length, first.max_position_embeddings, first.pooling) == (HIDDEN, 32, 64, "cls")
    assert first.input_format.query_prefix == MODEL_FORMATS[BGE].query_prefix and first.input_format.passage_prefix == ""
    assert second.digest != first.digest  # other weights, another identity
    assert inspect_checkpoint(world["checkpoints"][0]).digest == first.digest
    provenance = first.provenance()
    assert (provenance["step"], provenance["total_steps"], provenance["run"]) == (5, 15, "fixture-seed42")
    assert provenance["run_record"]["sha256"] == hashlib.sha256((world["checkpoints"][0].parent / "run.json").read_bytes()).hexdigest()
    assert provenance["trained_on"]["candidates"]["version"] == "cand_fixture"
    assert provenance["selection"].startswith("none") and provenance["trainer_state_present_not_loaded"] is False
    assert provenance["files"] == first.files and provenance["base_model"]["revision"] == BGE_REVISION


def _edit(path: Path, change: Callable[[Any], Any]) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


REFUSALS: list[tuple[str, Callable[[Path], Any], str]] = [
    ("missing file", lambda d: (d / "tokenizer.json").unlink(), "lacks"),
    ("pickled weights", lambda d: (d / "pytorch_model.bin").write_bytes(b"pickle"), "pickled weights"),
    ("not a checkpoint record", lambda d: _edit(d / "metadata.json", lambda m: m.update(kind="other")),
     "not a format-1 trainer checkpoint"),
    ("unknown base model", lambda d: _edit(d / "metadata.json", lambda m: m["base_model"].update(name="some/model")),
     "no built-in input format"),
    ("another input format", lambda d: _edit(d / "metadata.json", lambda m: m["encoding"].update(query_prefix="query: ")),
     "trained with query_prefix"),
    ("custom code", lambda d: _edit(d / "config.json", lambda c: c.update(auto_map={"AutoModel": "x.Model"})),
     "custom model code"),
    ("missing module", lambda d: (d / "modules.json").write_text(json.dumps(MODULES[:2])), "exactly Transformer"),
    ("foreign module", lambda d: (d / "modules.json").write_text(json.dumps([{**MODULES[0], "type": "evil.Module"},
                                                                             *MODULES[1:]])), "exactly Transformer"),
    ("other pooling", lambda d: _edit(d / "1_Pooling" / "config.json",
                                      lambda p: p.update(pooling_mode_cls_token=False, pooling_mode_mean_tokens=True)),
     "trained with cls pooling"),
    ("other length", lambda d: _edit(d / "sentence_bert_config.json", lambda s: s.update(max_seq_length=128)),
     "max_seq_length 128"),
    ("lower-casing", lambda d: _edit(d / "sentence_bert_config.json", lambda s: s.update(do_lower_case=True)), "lower-cases"),
    ("prompts", lambda d: _edit(d / "config_sentence_transformers.json", lambda s: s.update(prompts={"query": "q: "})),
     "defines prompts"),
    ("other hidden size", lambda d: _edit(d / "config.json", lambda c: c.update(hidden_size=32)), "records hidden size"),
    ("another run", lambda d: (d.parent / "run.json").write_text(json.dumps({"run": "x", "config_sha256": "c" * 64})),
     "another run"),
]


@pytest.mark.parametrize(("case", "change", "message"), REFUSALS, ids=[c[0] for c in REFUSALS])
def test_unsafe_or_inconsistent_checkpoints_are_refused(
    world: dict[str, Any], case: str, change: Callable[[Path], Any], message: str
) -> None:
    directory = world["checkpoints"][0]
    change(directory)
    with pytest.raises(CheckpointError, match=message):
        inspect_checkpoint(directory)


def test_links_and_non_directories_are_refused(world: dict[str, Any], tmp_path: Path) -> None:
    directory = world["checkpoints"][0]
    with pytest.raises(CheckpointError, match="not a checkpoint directory"):
        inspect_checkpoint(directory / "config.json")
    link = tmp_path / "linked-checkpoint"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(CheckpointError, match="not a checkpoint directory"):
        inspect_checkpoint(link)
    weights = directory / "model.safetensors"
    target = tmp_path / "elsewhere.safetensors"
    target.write_bytes(weights.read_bytes())
    weights.unlink()
    weights.symlink_to(target)
    with pytest.raises(CheckpointError, match="symbolic links"):
        inspect_checkpoint(directory)


def test_the_inspector_accepts_the_module_files_the_trainer_writes(world: dict[str, Any]) -> None:
    pytest.importorskip("torch")
    from finretriever.training.biencoder_trainer import write_sentence_transformers_files

    directory = world["checkpoints"][0]
    for name in ("modules.json", "sentence_bert_config.json", "config_sentence_transformers.json", "1_Pooling/config.json"):
        (directory / name).unlink()
    write_sentence_transformers_files(directory, dimension=HIDDEN, pooling="cls", max_seq_length=32)
    assert inspect_checkpoint(directory).pooling == "cls"


def _backend(monkeypatch: pytest.MonkeyPatch, models: FakeModels, **overrides: Any) -> list[FakeSentenceTransformer]:
    built: list[FakeSentenceTransformer] = []

    def construct(**kwargs: Any) -> FakeSentenceTransformer:
        built.append(FakeSentenceTransformer(**kwargs, **overrides))
        return built[-1]

    monkeypatch.setattr(biencoder, "_import_backend", lambda: (FakeTorch(), construct))
    monkeypatch.setattr(checkpoint_module, "_import_modules", lambda: models)
    return built


def test_the_loader_builds_the_modules_from_local_safetensors_and_checks_the_model(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = inspect_checkpoint(world["checkpoints"][0])
    with pytest.raises(DenseModelError, match="importing torch / sentence-transformers failed"):
        load_checkpoint_encoder(ckpt)  # tests/conftest.py refuses the real backend
    models = FakeModels()
    built = _backend(monkeypatch, models)
    encoder = load_checkpoint_encoder(ckpt, device="cpu")
    name, args, kwargs = models.calls[0]
    assert (name, args) == ("Transformer", (str(ckpt.path),))
    assert kwargs["model_args"] == {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True}
    assert kwargs["tokenizer_args"] == kwargs["config_args"] == {"local_files_only": True, "trust_remote_code": False}
    assert (kwargs["max_seq_length"], kwargs["do_lower_case"]) == (32, False)
    assert models.calls[1] == ("Pooling", (HIDDEN,), {"pooling_mode": "cls"}) and models.calls[2][0] == "Normalize"
    assert [m.name for m in built[0].modules] == ["Transformer", "Pooling", "Normalize"] and built[0].device == "cpu"
    assert (encoder.model_name, encoder.revision, encoder.embedding_dim) == (ckpt.model_id, ckpt.revision, HIDDEN)
    assert encoder.encode(["a passage"], 4).shape == (1, HIDDEN)

    for overrides, message in (
        ({"tokenizer": FakeTokenizer(text="another tokenizer")}, "not the .* the checkpoint was trained with"),
        ({"tokenizer": FakeTokenizer(size=101)}, "101 tokens, the model 100 embedding rows"),
        ({"dim": 8}, "embeds in 8 dimensions, not 16"),
        ({"max_seq_length": 128}, "exceeds 64 positions"),
    ):
        _backend(monkeypatch, FakeModels(), **overrides)
        with pytest.raises(CheckpointError, match=message):
            load_checkpoint_encoder(ckpt, device="cpu")
    with pytest.raises(CheckpointError, match="exceeds the 64 positions"):
        load_checkpoint_encoder(ckpt, device="cpu", max_seq_length=65)

    def swap_weights(**kwargs: Any) -> FakeSentenceTransformer:
        (ckpt.path / "model.safetensors").write_bytes(b"other weights")
        return FakeSentenceTransformer(**kwargs)

    monkeypatch.setattr(biencoder, "_import_backend", lambda: (FakeTorch(), swap_weights))
    with pytest.raises(CheckpointError, match="changed while it was being loaded"):
        load_checkpoint_encoder(ckpt, device="cpu")

    def broken(**kwargs: Any) -> Any:
        raise OSError("weights unreadable")

    monkeypatch.setattr(biencoder, "_import_backend", lambda: (FakeTorch(), broken))
    with pytest.raises(DenseModelError, match="could not load checkpoint .* weights unreadable"):
        load_checkpoint_encoder(ckpt, device="cpu")
    with pytest.raises(DenseModelError, match="CUDA is not available"):
        load_checkpoint_encoder(ckpt, device="cuda")


# --- the retriever: interface, dimensions, ranking, cache --------------------------------------


def test_a_checkpoint_retriever_ranks_like_any_dense_retriever(world: dict[str, Any]) -> None:
    passages = fixture_passages()
    ckpt = inspect_checkpoint(world["checkpoints"][0])
    encoder = FakeEncoder(dim=HIDDEN)
    retriever = CheckpointRetriever(ckpt, retriever_name="ckpt5", encoder=encoder)
    base = BiEncoderRetriever(BGE, model_revision=BGE_REVISION, encoder=FakeEncoder(dim=HIDDEN))
    for r in (retriever, base):
        r.index(passages)
    ranked = retriever.search(QUESTION, top_k=5, query_id="q")
    assert [(h.passage_id, h.score) for h in ranked] == [(h.passage_id, h.score) for h in base.search(QUESTION, top_k=5, query_id="q")]
    assert [h.rank for h in ranked] == [1, 2, 3, 4, 5] and all(h.retriever == "ckpt5" for h in ranked)
    assert encoder.encoded[-1] == MODEL_FORMATS[BGE].query_prefix + QUESTION  # the prefix it was trained with
    assert encoder.encoded[: len(passages)] == [p.text for p in passages]  # passages take none
    hits = retriever.batch_search({"q1": QUESTION}, top_k=3, tickers={"q1": ["BETA"]})
    assert len(hits) == 3 and {h.metadata["ticker"] for h in hits} == {"BETA"}
    embeddings = retriever.embeddings
    assert embeddings is not None and embeddings.shape == (len(passages), HIDDEN) and embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    described = retriever.describe()
    assert described["source"] == "local_checkpoint" and described["type"] == "dense"
    assert (described["model_name"], described["model_revision"], described["embedding_dim"]) == (ckpt.model_id, ckpt.revision, HIDDEN)
    assert described["checkpoint"]["digest"] == ckpt.digest and described["checkpoint"]["step"] == 5
    assert described["embeddings_sha256"] == hashlib.sha256(embeddings.tobytes()).hexdigest()

    with pytest.raises(ValueError, match="does not match the checkpoint's 16 dimensions"):
        CheckpointRetriever(ckpt, embedding_dim=768)
    with pytest.raises(ValueError, match="query_prefix 'query: ' does not match"):
        CheckpointRetriever(ckpt, query_prefix="query: ")
    with pytest.raises(ValueError, match="exceeds the checkpoint's 64 position embeddings"):
        CheckpointRetriever(ckpt, max_seq_length=128)
    with pytest.raises(DenseModelError, match="produces 8-dimensional embeddings"):
        CheckpointRetriever(ckpt, encoder=FakeEncoder(dim=8)).index(passages)


def test_each_checkpoint_gets_its_own_embedding_cache_entry(world: dict[str, Any], tmp_path: Path) -> None:
    passages = fixture_passages()
    cache = {"cache_dir": tmp_path / "cache", "corpus_sha256": "c" * 64}
    first = CheckpointRetriever(world["checkpoints"][0], encoder=FakeEncoder(dim=HIDDEN), **cache)
    second = CheckpointRetriever(world["checkpoints"][1], encoder=FakeEncoder(dim=HIDDEN), **cache)
    for r in (first, second):
        r.index(passages)
    keys = [r.describe()["cache"] for r in (first, second)]
    assert keys[0]["dir"] != keys[1]["dir"]
    assert keys[0]["key"]["model_revision"] == inspect_checkpoint(world["checkpoints"][0]).revision
    again = FakeEncoder(dim=HIDDEN)
    CheckpointRetriever(world["checkpoints"][0], encoder=again, **cache).index(passages)
    assert again.calls == []  # reused, not encoded again


# --- evaluation isolation ----------------------------------------------------------------------


def _isolation(world: dict[str, Any], directory: Path) -> dict[str, Any]:
    labels = world["labels"]
    return checkpoint_isolation(
        inspect_checkpoint(directory),
        corpus=load_corpus_identity(world["corpus"]),
        queries=read_jsonl(labels / "dev_queries.jsonl", model=Query),
        qrels=read_jsonl(labels / "dev_qrels.jsonl", model=Qrel),
    )


def test_isolation_is_shown_from_the_training_records(world: dict[str, Any]) -> None:
    record = _isolation(world, world["checkpoints"][0])
    assert record["isolated"] is True and record["judged_passages"] == len(world["judged"]) == 4
    assert record["blocklist"]["judged_passages_blocked"] == 4
    assert record["candidates"]["pairs_available"] == 2 and record["candidates"]["passages_used"] == 3
    assert record["candidates"]["passages_used_and_judged"] == 0 and record["candidates"]["max_question_similarity"] < 0.8


def _variant(world: dict[str, Any], name: str, *, blocked: list[str] | None = None,
             candidates: list[dict[str, Any]] | None = None) -> Path:
    inputs = write_training_side(world["tmp"] / name, world["corpus"], blocked=world["judged"] if blocked is None else blocked,
                                 candidates=candidates or training_candidates())
    return write_checkpoint(world["tmp"] / name / "run", 5, inputs)


def test_isolation_fails_closed(world: dict[str, Any]) -> None:
    judged = world["judged"]
    cases = [
        (_variant(world, "unblocked", blocked=judged[1:]), "judged passages are not on the training blocklist"),
        (_variant(world, "positive", candidates=training_candidates(
            train_000003=candidate("train_000003", judged[0], "Where does Acme buy its parts?"))), "uses judged passage"),
        (_variant(world, "equivalent", candidates=training_candidates(
            train_000003=candidate("train_000003", IDS["acme24_sales"], "Acme sales?", exact=[judged[1]]))),
         "uses judged passage"),
        (_variant(world, "question", candidates=training_candidates(
            train_000003=candidate("train_000003", IDS["acme24_sales"], QUESTION))), "repeats evaluation question"),
    ]
    for directory, message in cases:
        with pytest.raises(IsolationError, match=message):
            _isolation(world, directory)
    blocklist = world["tmp"] / "training" / "prep" / "blocklist.jsonl"
    blocklist.write_text(blocklist.read_text() + json.dumps({"passage_id": "x", "reasons": []}) + "\n")
    with pytest.raises(IsolationError, match="blocklist .* changed since training"):
        _isolation(world, world["checkpoints"][0])
    other = _variant(world, "corpus")
    _edit(other / "metadata.json", lambda m: m["inputs"]["corpus"].update(sha256="e" * 64))
    with pytest.raises(IsolationError, match="trained on corpus"):
        _isolation(world, other)


# --- the evaluation set ------------------------------------------------------------------------


def checkpoint_config(world: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """BM25 stored as the reference run, then a set that runs the two checkpoints beside it."""
    tmp = world["tmp"]
    config = {**world["config"], "retrieval_modes": ["global", "company"], "embedding_cache_dir": str(tmp / "cache"),
              "baselines": [{"name": "bm25", "type": "bm25", "tokenizer": "word", "backend": "python"}]}
    if not (tmp / "results" / "dev" / "test_v2").exists():
        write_eval_set_outputs(run_eval_set(config, "dev"))
    entry = {
        **config["evaluation_sets"]["dev"],
        "retrievers": [{"name": f"ckpt_step{step}", "type": "dense_checkpoint", "checkpoint_path": str(path), "batch_size": 4}
                       for step, path in zip((5, 10), world["checkpoints"], strict=True)],
        "reference_runs_from": str(tmp / "results" / "dev" / "test_v2"),
        "paired_reference": "bm25",
        "output_dir": str(tmp / "results" / "dev" / "test_checkpoints"),
        **changes,
    }
    return {**config, "evaluation_sets": {**config["evaluation_sets"], "ckpt": entry}}


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loaded: list[str] = []

    def load(ckpt: Any, **kwargs: Any) -> FakeEncoder:
        loaded.append(ckpt.model_id)
        return FakeEncoder(dim=HIDDEN)

    monkeypatch.setattr(checkpoint_module, "load_checkpoint_encoder", load)
    return loaded


def test_checkpoints_are_scored_beside_the_stored_baselines_and_written_once(
    world: dict[str, Any], fake_loader: list[str]
) -> None:
    config = checkpoint_config(world)
    stored = world["tmp"] / "results" / "dev" / "test_v2"
    stored_before = tree(stored)
    result = run_eval_set(config, "ckpt")
    assert [(r.run_id, r.reference) for r in result.runs] == [
        ("global/bm25", True), ("company/bm25", True), ("global/ckpt_step5", False), ("company/ckpt_step5", False),
        ("global/ckpt_step10", False), ("company/ckpt_step10", False),
    ]
    assert fake_loader == [inspect_checkpoint(p).model_id for p in world["checkpoints"]]  # one load per checkpoint
    entry = config["evaluation_sets"]["ckpt"]
    queries, qrels = read_jsonl(entry["queries_path"], model=Query), read_jsonl(entry["qrels_path"], model=Qrel)
    passages = {p.passage_id: p for p in fixture_passages()}
    for run in result.runs:
        assert (run.aggregate, run.per_query) == evaluate_eval_set_run(  # one metric function for every run
            queries, qrels, run.hits, passages, metrics=config["metrics"], judged_k=10,
            thresholds=config["coverage_thresholds"])
        assert set(run.uncertainty or {}) == {"ndcg@10", "mrr@10", "recall@100", "judged@10"}
        if run.reference:
            assert run.paired is None and run.source is not None
            continue
        assert run.retriever["evaluation_isolation"]["isolated"] is True
        assert run.retriever["checkpoint"]["digest"] == inspect_checkpoint(world["checkpoints"][0 if "5" in run.run_name else 1]).digest
        assert run.paired is not None and run.paired["reference"] == f"{run.mode}/bm25"
        assert run.paired["reference_source"]["run_sha256"] == next(
            r.source for r in result.runs if r.run_id == f"{run.mode}/bm25")["run_sha256"]
        assert run.paired["metrics"]["ndcg@10"]["n"] == 2

    paths = write_eval_set_outputs(result)
    out = world["tmp"] / "results" / "dev" / "test_checkpoints"
    assert sorted(paths) == sorted([f"{m}/ckpt_step{s}/{f}" for m in ("global", "company") for s in (5, 10)
                                    for f in ("metrics.json", "run.jsonl")] + ["report.md"])
    assert not (out / "global" / "bm25").exists() and tree(stored) == stored_before  # references are only read
    payload = json.loads((out / "global" / "ckpt_step5" / "metrics.json").read_text())
    assert payload["checkpoint_caution"] == CHECKPOINT_CAUTION and payload["paired_comparison"]["reference"] == "global/bm25"
    assert payload["retriever"]["checkpoint"]["files"]["model.safetensors"] == hashlib.sha256(b"weights 5").hexdigest()
    assert payload["retriever"]["evaluation_isolation"]["candidates"]["passages_used_and_judged"] == 0
    assert payload["settings"]["reference_runs_from"] == str(stored) and payload["settings"]["paired_reference"] == "bm25"
    assert set(payload["uncertainty"]) == {"ndcg@10", "mrr@10", "recall@100", "judged@10"}
    assert "relevant_outside_top10" in payload["per_query"][0]
    report = (out / "report.md").read_text()
    for text in (CHECKPOINT_CAUTION, "## Statistical uncertainty", "### Paired comparison with `bm25`",
                 "## Relevant retrieval failures", "bm25 (stored reference)", "local checkpoint", "evaluation isolation: 4 of 4"):
        assert text in report
    assert "best" not in report.lower()  # nothing is selected

    written = tree(out)
    write_eval_set_outputs(run_eval_set(config, "ckpt"))  # the same results: nothing changes
    assert tree(out) == written
    changed = checkpoint_config(world, paired_reference="ckpt_step5")
    with pytest.raises(ValueError, match="never overwritten, so nothing was written"):
        write_eval_set_outputs(run_eval_set(changed, "ckpt"))
    assert tree(out) == written


def test_a_pre_flight_check_inspects_every_checkpoint_and_loads_none(
    world: dict[str, Any], fake_loader: list[str]
) -> None:
    result = run_eval_set(checkpoint_config(world), "ckpt", load_models=False)
    assert fake_loader == [] and [(r.run_id, r.reference) for r in result.runs] == [("global/bm25", True), ("company/bm25", True)]
    assert result.checked is not None and set(result.checked) == {"ckpt_step5", "ckpt_step10"}
    assert all(c["evaluation_isolation"]["isolated"] and c["type"] == "dense_checkpoint" for c in result.checked.values())
    assert result.checked["ckpt_step10"]["checkpoint"]["step"] == 10 and all(r.uncertainty for r in result.runs)
    assert not (world["tmp"] / "results" / "dev" / "test_checkpoints").exists()


def test_a_checkpoint_that_may_have_seen_the_labels_is_never_loaded(
    world: dict[str, Any], fake_loader: list[str]
) -> None:
    leaky = _variant(world, "leaky", blocked=world["judged"][1:])
    config = checkpoint_config(world, retrievers=[{"name": "leaky", "type": "dense_checkpoint", "checkpoint_path": str(leaky)}])
    with pytest.raises(IsolationError, match="not isolated from the evaluation set"):
        run_eval_set(config, "ckpt")
    assert fake_loader == [] and not (world["tmp"] / "results" / "dev" / "test_checkpoints").exists()


def test_checkpoint_sets_are_configured_strictly(world: dict[str, Any], tmp_path: Path) -> None:
    config = checkpoint_config(world)
    sets = config["evaluation_sets"]

    def with_entry(**changes: Any) -> dict[str, Any]:
        return {**config, "evaluation_sets": {**sets, "ckpt": {**sets["ckpt"], **changes}}}

    retrievers = sets["ckpt"]["retrievers"]
    for changes, message in (
        ({"runs_from": sets["ckpt"]["reference_runs_from"], "reference_runs_from": None}, "runs no retrievers"),
        ({"retrievers": [retrievers[0], retrievers[0]]}, "names a retriever twice"),
        ({"retrievers": []}, "non-empty list"),
        ({"paired_reference": "e5"}, "paired_reference 'e5' is not one of its runs"),
        ({"reference_runs_from": sets["ckpt"]["output_dir"]}, "overwrite the stored runs"),
    ):
        with pytest.raises(ValueError, match=message):
            load_eval_set(with_entry(**changes), "ckpt")
    validation = tmp_path / "prep" / "validation_sets" / "v1"
    validation.mkdir(parents=True)
    (validation / "validation_pairs.jsonl").write_text("")
    with pytest.raises(ValueError, match="training or validation data"):
        load_eval_set(with_entry(queries_path=str(validation / "validation_pairs.jsonl")), "ckpt")
    with pytest.raises(ValueError, match="needs checkpoint_path"):
        build_retriever({"name": "x", "type": "dense_checkpoint"})
    with pytest.raises(ValueError, match="unknown settings"):
        build_retriever({**retrievers[0], "model_revision": "0" * 40})
    with pytest.raises(CheckpointError, match="not a checkpoint directory"):
        build_retriever({**retrievers[0], "checkpoint_path": str(tmp_path / "nowhere")})


def test_written_results_are_never_overwritten(world: dict[str, Any]) -> None:
    config = checkpoint_config(world)  # writes the BM25 dev runs
    out = world["tmp"] / "results" / "dev" / "test_v2"
    report = out / "report.md"
    report.write_text(report.read_text() + "edited\n")
    before = tree(out)
    with pytest.raises(ValueError, match="report.md already holds different results"):
        write_eval_set_outputs(run_eval_set(config, "dev"))
    assert tree(out) == before


# --- statistical uncertainty -------------------------------------------------------------------


def test_intervals_and_paired_tests_are_deterministic_and_exact_for_few_questions() -> None:
    values = [0.0, 0.5, 1.0, 0.25]
    interval = mean_interval(values)
    assert interval == mean_interval(values) and interval["mean"] == pytest.approx(0.4375)
    assert interval["low"] <= interval["mean"] <= interval["high"] and interval["n"] == 4
    a = {f"q{i}": 0.5 + 0.1 * i for i in range(5)}
    b = {f"q{i}": 0.4 + 0.1 * i for i in range(5)}
    better = paired_difference(a, b)
    assert better["mean_difference"] == pytest.approx(0.1) and (better["wins"], better["ties"], better["losses"]) == (5, 0, 0)
    assert better["test"] == "exact" and better["p_value"] == pytest.approx(2 / 32)  # all-positive pattern, both tails
    same = paired_difference(a, a)
    assert same["p_value"] == 1.0 and same["ties"] == 5 and same["interval_includes_zero"]
    mixed = paired_difference({"q1": 1.0, "q2": 0.0}, {"q1": 0.0, "q2": 1.0})
    assert mixed["mean_difference"] == 0.0 and mixed["p_value"] == 1.0
    many = paired_difference({f"q{i}": 1.0 for i in range(20)}, {f"q{i}": 0.0 for i in range(20)})
    assert many["test"] == "monte_carlo" and many["p_value"] < 0.001
    with pytest.raises(ValueError, match="same questions"):
        paired_difference({"q1": 1.0}, {"q2": 1.0})
    with pytest.raises(ValueError, match="no values"):
        mean_interval([])
