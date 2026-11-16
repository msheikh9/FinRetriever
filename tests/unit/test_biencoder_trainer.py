"""Tests for the contrastive bi-encoder trainer with tiny local models (offline, CPU, no downloads)."""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import io
import json
import logging
import os
import socket
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import yaml  # noqa: E402

from finretriever.training.biencoder_trainer import (  # noqa: E402
    HFEncoder,
    TrainerError,
    check_lengths,
    contrastive_loss,
    main,
    train,
)
from finretriever.training.candidates import (  # noqa: E402
    CANDIDATE_FILES,
    MANIFEST_FILE,
    candidates_dir,
)
from finretriever.training.data_cli import main as data_main  # noqa: E402
from finretriever.training.trainer_setup import (  # noqa: E402
    TrainerConfig,
    TrainerSetupError,
    TrainingInputs,
    check_batch,
    load_trainer_config,
    plan_steps,
    prepare_inputs,
)
from finretriever.training.training_data import TrainingDataError  # noqa: E402
from tests.unit.eval_set_fixtures import build_expanded_inputs  # noqa: E402
from tests.unit.trainer_fixtures import VISIBLE_TOKENS, TinyEncoder  # noqa: E402
from tests.unit.training_fixtures import (  # noqa: E402
    SECRET,
    MockTransport,
    WhitespaceTokenizer,
    accept_all,
    generic_candidates,
    training_extra_passages,
    write_prep_config,
)


def data_cli(config: Path, *argv: str, transport: Any = None) -> int:
    return data_main([argv[0], "--config", str(config), *argv[1:]], tokenizer_loader=lambda s: WhitespaceTokenizer(),
                     transport=transport, sleep=lambda seconds: None, out=io.StringIO(), err=io.StringIO())


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A prepared synthetic corpus, mocked generation, pairs, and two candidate datasets."""
    root = tmp_path_factory.mktemp("trainer")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network")))
        mp.setenv("GROQ_API_KEY", SECRET)
        build_expanded_inputs(root, training_extra_passages())
        prep = write_prep_config(root)
        transport = MockTransport(lambda payload: {"questions": generic_candidates(payload)})
        for argv in (("prepare",), ("generate", "--split", "train", "--confirm-api-use"), ("build-pairs",)):
            assert data_cli(prep, *argv, transport=transport) == 0
        prepared_dir = root / "data" / "processed" / "train" / "test_v1"
        for version, decision in (("cand_v1", "accept"), ("none_v1", "reject")):
            decisions = accept_all(prepared_dir, root / f"{version}.decisions.jsonl", decision)
            cfg = root / f"{version}.yaml"
            cfg.write_text(yaml.safe_dump({"version": version, "data_prep_config": str(prep),
                                           "sources": [{"name": "fixture", "decisions": str(decisions)}]}))
            assert data_cli(prep, "build-candidates", "--candidates", str(cfg)) == 0
    yield {"root": root, "prep": prep, "prepared_dir": prepared_dir}


def trainer_config(dataset: dict[str, Any], tmp_path: Path, **training: Any) -> tuple[TrainerConfig, Path]:
    values: dict[str, Any] = {
        "name": "tiny",
        "base_model": {"name": "fake/whitespace", "revision": "0" * 40, "query_prefix": "Find: ",
                       "passage_prefix": "", "pooling": "cls"},
        "data": {"data_prep_config": str(dataset["prep"]), "candidates": "cand_v1"},
        "training": {"seed": 42, "batch_size": 3, "learning_rate": 0.01, "epochs": 4, "max_steps": 4,
                     "temperature": 0.05, "query_max_length": 64, "passage_max_length": VISIBLE_TOKENS,
                     "device": "cpu", "warmup_ratio": 0.25, "weight_decay": 0.01},
        "checkpoints": {"output_dir": str(tmp_path / "models"), "save_every_steps": 2},
    }
    for key, value in training.items():
        section = next(s for s in ("training", "data", "checkpoints", "base_model") if key in values[s])
        values[section][key] = value
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"trainer_{len(list(tmp_path.glob('trainer_*.yaml')))}.yaml"
    path.write_text(yaml.safe_dump(values))
    return load_trainer_config(path)[0], path


def inputs_for(dataset: dict[str, Any], tmp_path: Path, **training: Any) -> TrainingInputs:
    config, path = trainer_config(dataset, tmp_path, **training)
    return prepare_inputs(config, hashlib.sha256(path.read_bytes()).hexdigest())


def weights(run_dir: Path, step: int) -> dict[str, Any]:
    return torch.load(run_dir / f"checkpoint-step-{step:06d}" / "weights.pt", weights_only=True)


# --- the objective ------------------------------------------------------------------------------


def test_the_loss_is_in_batch_infonce_over_normalised_embeddings() -> None:
    torch.manual_seed(0)
    q, p = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8)
    t = 0.05
    scores = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(p, dim=-1).T / t
    manual = -torch.log_softmax(scores, dim=1).diagonal().mean()
    loss = contrastive_loss(q, p, t)
    assert torch.allclose(loss, manual)
    assert torch.allclose(contrastive_loss(3 * q, p, t), loss)  # embeddings are normalised first
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    aligned = contrastive_loss(p, p, t)
    shuffled = contrastive_loss(p.roll(1, 0), p, t)
    assert aligned < shuffled
    with pytest.raises(TrainerError, match="at least two"):
        contrastive_loss(q[:1], p[:1], t)  # one pair has no negative


def test_the_loss_falls_on_a_fixed_batch() -> None:
    encoder = TinyEncoder()
    encoder.dropout.p = 0.0
    queries = [f"query about topic {w} and figure {i}" for i, w in enumerate("abcdef")]
    passages = [f"passage on topic {w} reporting figure {i} for the year" for i, w in enumerate("abcdef")]
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.05)
    losses = []
    for _ in range(30):
        loss = contrastive_loss(encoder.encode(queries, 64), encoder.encode(passages, 64), 0.05)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.5 * losses[0]


# --- training runs ------------------------------------------------------------------------------


def test_a_run_updates_parameters_and_records_how_it_trained(dataset: dict[str, Any], tmp_path: Path) -> None:
    inputs = inputs_for(dataset, tmp_path)
    encoder = TinyEncoder()
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    result = train(inputs, encoder)
    assert result.steps_done == result.total_steps == 4
    assert [p.name for p in result.checkpoints] == ["checkpoint-step-000002", "checkpoint-step-000004"]
    after = weights(result.run_dir, 4)
    assert any(not torch.equal(before[k], after[k]) for k in before)  # the optimizer moved the weights
    assert all(torch.isfinite(torch.tensor(h["loss"])) and h["grad_norm"] > 0 for h in result.history)
    rates = [h["learning_rate"] for h in result.history]
    assert rates[0] == pytest.approx(0.01) and rates[-1] < rates[0]  # warmup of 1 step, then linear decay
    meta = json.loads((result.checkpoints[-1] / "metadata.json").read_text())
    assert meta["objective"]["in_batch_negatives_per_query"] == 2 and meta["objective"]["temperature"] == 0.05
    assert "negatives come from each batch alone" in meta["objective"]["accumulation_note"]
    assert meta["selection"].startswith("none") and "no retrieval improvement is claimed" in meta["notice"]
    assert "not gold labels" in meta["inputs"]["notice"] and meta["environment"]["device"] == "cpu"
    assert meta["inputs"]["candidates"]["equivalence_method"] == "evidence-reuse-v2"
    assert meta["gpu_memory"] is None  # nothing is measured on CPU
    assert len(meta["loss_history"]) == 4 and (result.run_dir / "run.json").is_file()
    # Only the newest checkpoint keeps the resumable state; every checkpoint keeps its model.
    assert not (result.checkpoints[0] / "trainer_state.pt").exists()
    assert (result.checkpoints[1] / "trainer_state.pt").is_file() and (result.checkpoints[0] / "weights.pt").is_file()


def test_batches_never_hold_a_known_equivalent_or_blocked_passage(dataset: dict[str, Any], tmp_path: Path) -> None:
    inputs = inputs_for(dataset, tmp_path)
    schedule = plan_steps(inputs)
    by_id = inputs.by_id
    for step in schedule.steps:
        for batch in step.batches:
            members = [by_id[i] for i in batch]
            assert len(batch) >= 2  # a single pair has no negative and is never trained on
            for a in members:
                excluded = inputs.guard.exclusions(a)
                assert not any(b.positive_passage_id in excluded for b in members if b is not a)
    # A batch holding a blocked passage, or a pair's own equivalent, is refused before any forward pass.
    rows = inputs.prepared.rows
    a, b = inputs.pairs[:2]
    blocked = next(pid for pid, r in rows.items() if r["split"] == "blocked")
    swapped = [blocked, *sorted(inputs.guard.exclusions(a) - {a.positive_passage_id})]
    for passage_id in swapped:
        moved = b.model_copy(update={"positive_passage_id": passage_id})
        bad = dataclasses.replace(inputs, pairs=(a, moved), passages={**inputs.passages, passage_id: "text"})
        with pytest.raises(TrainerSetupError, match="may not be a negative"):
            check_batch(bad, [a.pair_id, b.pair_id])


def test_checkpoints_reload_and_a_resumed_run_matches_an_uninterrupted_one(dataset: dict[str, Any],
                                                                           tmp_path: Path) -> None:
    straight = train(inputs_for(dataset, tmp_path / "a"), TinyEncoder())
    interrupted = inputs_for(dataset, tmp_path / "b")
    first = train(interrupted, TinyEncoder(), stop_after=2)
    assert first.steps_done == 2 and [p.name for p in first.checkpoints] == ["checkpoint-step-000002"]
    with pytest.raises(TrainerError, match="--resume"):
        train(interrupted, TinyEncoder())  # never written over
    resumed = train(interrupted, TinyEncoder(), resume=True)
    assert resumed.steps_done == 4
    assert [h["loss"] for h in resumed.history] == [h["loss"] for h in straight.history]
    final_a, final_b = weights(straight.run_dir, 4), weights(resumed.run_dir, 4)
    assert all(torch.equal(final_a[k], final_b[k]) for k in final_a)
    reloaded = TinyEncoder()
    reloaded.load_weights(resumed.checkpoints[-1])
    reloaded.eval()
    trained = TinyEncoder()
    trained.load_state_dict(final_a)
    trained.eval()
    assert torch.equal(reloaded.encode(["net income"], 64), trained.encode(["net income"], 64))
    # The same directory with other settings is another run: refused.
    changed = inputs_for(dataset, tmp_path / "b", learning_rate=0.02)
    with pytest.raises(TrainerError, match="different run"):
        train(changed, TinyEncoder(), resume=True)


def test_seeds_make_runs_reproducible(dataset: dict[str, Any], tmp_path: Path) -> None:
    one = train(inputs_for(dataset, tmp_path / "one"), TinyEncoder())
    two = train(inputs_for(dataset, tmp_path / "two"), TinyEncoder())
    other = train(inputs_for(dataset, tmp_path / "other", seed=7), TinyEncoder())
    a, b, c = (weights(r.run_dir, 4) for r in (one, two, other))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


# --- failing closed -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"passage_max_length": 50}, "visible text was defined for another encoder"),
        ({"name": "BAAI/bge-base-en-v1.5", "query_prefix": "Represent this sentence for searching relevant "
          "passages: "}, "visible text was defined for another encoder"),
        ({"candidates": "missing_v1"}, "no candidate dataset"),
        ({"candidates": "none_v1"}, "0 training candidates"),
    ],
)
def test_incompatible_or_empty_inputs_are_refused(dataset: dict[str, Any], tmp_path: Path, change: dict[str, Any],
                                                  message: str) -> None:
    with pytest.raises((TrainerSetupError, TrainingDataError), match=message):
        inputs_for(dataset, tmp_path, **change)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m, d: m["preparation"].update(manifest_sha256="0" * 64), "another preparation"),
        (lambda m, d: m["checks"].update(equivalence_method="evidence-reuse-v1"), "build a new version"),
        (lambda m, d: (d / CANDIDATE_FILES["train"]).write_text("{}\n"), "does not match"),
        (lambda m, d: Path(m["sources"][0]["decisions"]["path"]).write_text(""), "changed since it was built"),
    ],
)
def test_tampered_candidates_are_refused(dataset: dict[str, Any], tmp_path: Path, edit: Any, message: str) -> None:
    directory = candidates_dir(inputs_for(dataset, tmp_path).prepared, "cand_v1")
    manifest_path = directory / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    decisions = Path(manifest["sources"][0]["decisions"]["path"])
    saved = {p: p.read_bytes() for p in (*directory.iterdir(), decisions)}
    try:
        edit(manifest, directory)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(TrainingDataError, match=message):
            inputs_for(dataset, tmp_path)
    finally:
        for path, data in saved.items():
            path.write_bytes(data)


def test_an_invalid_candidate_record_is_refused(dataset: dict[str, Any], tmp_path: Path) -> None:
    directory = candidates_dir(inputs_for(dataset, tmp_path).prepared, "cand_v1")
    path, manifest_path = directory / CANDIDATE_FILES["train"], directory / MANIFEST_FILE
    saved = {p: p.read_bytes() for p in (path, manifest_path)}
    try:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["equivalents"]["method"] = "evidence-reuse-v1"  # checked by the first locator
        data = "".join(json.dumps(r) + "\n" for r in records)
        path.write_text(data)
        manifest = json.loads(saved[manifest_path])
        manifest["files"][path.name] = hashlib.sha256(data.encode()).hexdigest()  # re-pinned, as if written so
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(TrainingDataError, match="re-check the pair"):
            inputs_for(dataset, tmp_path)
    finally:
        for p, data_bytes in saved.items():
            p.write_bytes(data_bytes)


def test_another_tokenizer_or_a_too_long_query_is_refused(dataset: dict[str, Any], tmp_path: Path) -> None:
    inputs = inputs_for(dataset, tmp_path)
    with pytest.raises(TrainerError, match="tokenizer is not the one"):
        train(inputs, TinyEncoder(tokenizer_sha256="0" * 64))
    with pytest.raises(TrainerError, match="exceed query_max_length"):
        train(inputs_for(dataset, tmp_path / "short", query_max_length=8), TinyEncoder())


def test_passage_truncation_is_reported_and_must_keep_the_visible_text(dataset: dict[str, Any],
                                                                      tmp_path: Path) -> None:
    """Long positives are truncated, as expected; the trainer says how many, and refuses a tokenizer whose
    truncation would cut into a pair's visible text (where the validator found its evidence)."""
    inputs = inputs_for(dataset, tmp_path / "ok")
    lengths = check_lengths(inputs, TinyEncoder())
    assert lengths["positive_passages"] == len(inputs.passages)
    assert lengths["longest_passage_tokens"] > VISIBLE_TOKENS and lengths["passages_truncated_to_visible_length"] > 0
    said: list[str] = []
    result = train(inputs, TinyEncoder(), say=said.append)
    assert [s for s in said if f"are truncated to {VISIBLE_TOKENS}, keeping every pair's visible text" in s]
    encoding = json.loads((result.checkpoints[-1] / "metadata.json").read_text())["encoding"]
    assert encoding["truncation_keeps_visible_text"] is True
    assert {k: encoding[k] for k in lengths} == lengths

    class FewerTokens(TinyEncoder):  # a tokenizer splitting the text into more pieces than the preparation's
        def kept_chars(self, texts: Sequence[str], max_length: int) -> list[int]:
            return super().kept_chars(texts, max_length - 5)

    cut = inputs_for(dataset, tmp_path / "cut")
    with pytest.raises(TrainerError, match="would cut the visible text, where the evidence lies"):
        train(cut, FewerTokens())
    assert not (tmp_path / "cut" / "models").exists()  # refused before anything is written


def test_measuring_long_passages_does_not_warn_and_no_batch_exceeds_its_limit(tmp_path: Path) -> None:
    """The smoke run's console showed "Token indices sequence length is longer than the specified maximum
    sequence length for this model (739 > 512)": the trainer measuring an untruncated passage, never a model
    input. Measuring stays silent; encoding truncates, and a wider batch never reaches the model."""
    try:
        try:
            from transformers import BertTokenizerFast as BertTokenizer
        except ImportError:
            from transformers import BertTokenizer  # type: ignore[assignment]
    except Exception as exc:
        pytest.skip(f"transformers BERT tokenizer unavailable: {exc}")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "net", "income", "was", "million"]))
    tokenizer = BertTokenizer(vocab_file=str(vocab), model_max_length=8)
    text = "net income was million " * 5  # 20 words, 22 tokens with [CLS] and [SEP]

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    messages: list[str] = []
    handler = Collect()
    logging.getLogger("transformers").addHandler(handler)
    try:
        encoder = HFEncoder(torch.nn.Linear(2, 2), tokenizer, "cls")
        assert encoder.token_counts([text, "net income"]) == [22, 4]
        assert text[: encoder.kept_chars([text], 8)[0]] == "net income was million net income"  # 6 + 2 special
        assert not messages
        tokenizer([text])  # the old, verbose measurement: the warning this test guards against
    finally:
        logging.getLogger("transformers").removeHandler(handler)
    assert len(messages) == 1 and "(22 > 8)" in messages[0]

    class Untruncating:
        def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, Any]:
            ids = torch.ones(len(texts), 9, dtype=torch.long)
            return {"input_ids": ids, "attention_mask": ids}

    class Unreachable(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **batch: Any) -> Any:
            raise AssertionError("a batch wider than its limit reached the model")

    with pytest.raises(TrainerError, match="9 tokens for a limit of 8"):
        HFEncoder(Unreachable(), Untruncating(), "cls").encode(["a", "b"], 8)


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("training", {"batch_size": 1}, "batch_size"),
        ("data", {"max_pairs": 1}, "max_pairs"),
        ("training", {"fp16": True, "device": "cpu"}, "fp16 needs CUDA"),
        ("checkpoints", {"output_dir": "data/eval/models"}, "under data/eval"),
        ("data", {"data_prep_config": "data/eval/x.yaml"}, "under data/eval"),
        ("base_model", {"name": "BAAI/bge-base-en-v1.5", "query_prefix": "query: "}, "does not match the format"),
        ("base_model", {"name": "BAAI/bge-base-en-v1.5", "query_prefix": "Represent this sentence for searching "
                        "relevant passages: ", "pooling": "mean"}, "pools by cls"),
        ("base_model", {"revision": "main"}, "revision"),
        ("training", {"diagnostics_eval_set": "dev_pilot_v3"}, "Extra inputs"),
    ],
)
def test_unsafe_settings_are_refused(dataset: dict[str, Any], tmp_path: Path, section: str, values: dict[str, Any],
                                     message: str) -> None:
    _, path = trainer_config(dataset, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw[section].update(values)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(TrainerSetupError, match=message):
        load_trainer_config(path)


def test_training_reads_no_evaluation_file(dataset: dict[str, Any], tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    root = dataset["root"]
    evaluation = [root / "annotation", root / "dev_ai_v3", root / "audit", root / "gold_qrels.jsonl",
                  root / "gold_queries.jsonl", root / "gold_eval.metadata.json"]
    opened: list[str] = []
    real_open = io.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.path.realpath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    config, path = trainer_config(dataset, tmp_path)
    out = io.StringIO()
    code = main(["--config", str(path), "--confirm-training"], encoder_loader=lambda model: TinyEncoder(), out=out)
    assert code == 0 and "dry run" not in out.getvalue()
    assert opened  # the spy saw the corpus, the preparation and the candidates
    forbidden = [os.path.realpath(p) for p in evaluation]
    assert not [f for f in opened if any(f == p or f.startswith(p + os.sep) for p in forbidden)]
    assert not [f for f in opened if f"{os.sep}data{os.sep}eval{os.sep}" in f]


def test_the_command_line_dry_run_loads_no_model_and_writes_nothing(dataset: dict[str, Any], tmp_path: Path) -> None:
    config, path = trainer_config(dataset, tmp_path)
    out = io.StringIO()

    def no_model(model: Any) -> Any:
        raise AssertionError("a dry run must not load a model")

    assert main(["--config", str(path)], encoder_loader=no_model, out=out) == 0
    text = out.getvalue()
    assert "ESTIMATED peak GPU memory (not measured)" in text and "dry run" in text
    assert not (tmp_path / "models").exists()
    bad = tmp_path / "bad.yaml"
    bad.write_text(path.read_text().replace("cand_v1", "none_v1"))
    assert main(["--config", str(bad)], out=io.StringIO(), err=io.StringIO()) == 1


def test_hugging_face_checkpoints_reload_and_load_in_sentence_transformers(tmp_path: Path) -> None:
    """A tiny BERT built locally (random weights, a 14-word vocabulary): no download."""
    try:
        from transformers import BertConfig, BertModel
        try:
            from transformers import BertTokenizerFast as BertTokenizer
        except ImportError:  # transformers 5 has one (fast) BERT tokenizer
            from transformers import BertTokenizer  # type: ignore[assignment]
    except Exception as exc:  # e.g. a broken scipy behind transformers' lazy imports
        pytest.skip(f"transformers BERT unavailable: {exc}")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "net", "income", "was", "million", "revenue", "find",
             ":", "the", "of"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(words) + "\n")
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64)
    encoder = HFEncoder(BertModel(config), BertTokenizer(vocab_file=str(vocab), do_lower_case=True), "cls")
    encoder.eval()
    texts = ["Find: net income", "revenue was million"]
    with torch.no_grad():
        embeddings = encoder.encode(texts, 32)
    assert embeddings.shape == (2, 16) and encoder.token_counts(["net income"]) == [4]
    described = encoder.describe()
    assert described["pooling"] == "cls" and len(described["tokenizer_sha256"]) == 64
    out = tmp_path / "checkpoint"
    out.mkdir()
    encoder.save(out, max_seq_length=32)
    for name in ("model.safetensors", "modules.json", "sentence_bert_config.json", "1_Pooling/config.json"):
        assert (out / name).is_file(), name
    pooling = json.loads((out / "1_Pooling" / "config.json").read_text())
    assert pooling["pooling_mode_cls_token"] and not pooling["pooling_mode_mean_tokens"]
    reloaded = HFEncoder.load(str(out), revision=None, pooling="cls", local_files_only=True)
    reloaded.eval()
    fresh = HFEncoder(BertModel(config), reloaded.tokenizer, "cls")
    fresh.load_weights(out)
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(reloaded.encode(texts, 32), embeddings, atol=1e-6)
        assert torch.allclose(fresh.encode(texts, 32), embeddings, atol=1e-6)
    assert reloaded.describe()["tokenizer_sha256"] == described["tokenizer_sha256"]
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # optional: evaluation loads checkpoints through it
        return
    loaded = SentenceTransformer(str(out), device="cpu")
    st = loaded.encode(texts, convert_to_tensor=True, normalize_embeddings=False)
    assert torch.allclose(st, torch.nn.functional.normalize(embeddings, dim=-1), atol=1e-5)
