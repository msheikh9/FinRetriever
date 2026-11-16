"""Checkpoint serialization of a Hugging Face encoder loaded with ``from_pretrained``, as the trainer loads BGE.

The T4 rehearsal's check 9 failed because transformers 5's ``save_pretrained`` reverses its legacy LayerNorm renaming
on save (``weight``/``bias`` written as ``gamma``/``beta``) for a ``from_pretrained`` model, and the trainer's strict
reload then found missing and unexpected keys. A model built from a config, as the earlier tests built it, is saved
under its own names, which is why they passed. These tests load through ``from_pretrained`` (offline, CPU, a tiny
BERT)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from finretriever.training.biencoder_trainer import (  # noqa: E402
    TrainerError,
    _save_checkpoint,
    contrastive_loss,
    restore_checkpoint,
    seed_everything,
)
from finretriever.training.gpu_rehearsal import Context, check_checkpoint  # noqa: E402
from finretriever.training.trainer_setup import lr_factor, plan_steps  # noqa: E402
from tests.unit.trainer_fixtures import validated_inputs, validated_world  # noqa: E402
from tests.unit.validation_fixtures import World  # noqa: E402

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "net", "income", "was", "million", "revenue", "find", ":",
         "the", "of", "cash", "debt", "and"]
QUERIES = ["find : net income", "find : revenue", "find : cash of the debt", "find : net income was million"]
PASSAGES = ["net income was million", "revenue was million of the", "cash and debt", "the net revenue"]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return validated_world(tmp_path_factory.mktemp("serialization"))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """A tiny BERT with its tokenizer on disk, so the encoder is loaded with ``from_pretrained`` like BGE."""
    try:
        from transformers import BertConfig, BertModel
        try:
            from transformers import BertTokenizerFast as BertTokenizer
        except ImportError:
            from transformers import BertTokenizer  # type: ignore[assignment]
    except Exception as exc:  # e.g. a broken scipy behind transformers' lazy imports
        pytest.skip(f"transformers BERT unavailable: {exc}")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(WORDS) + "\n")
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(WORDS), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64, hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1)
    directory = tmp_path / "base"
    BertModel(config).save_pretrained(directory)
    BertTokenizer(vocab_file=str(vocab), do_lower_case=True).save_pretrained(directory)
    return directory


def load(base: Path) -> Any:
    from finretriever.training.biencoder_trainer import HFEncoder

    return HFEncoder.load(str(base), revision=None, pooling="cls", local_files_only=True, attn_implementation="eager")


def snapshot(encoder: Any) -> dict[str, Any]:
    return {k: v.detach().clone() for k, v in encoder.model.state_dict().items()}


def test_a_checkpoint_keeps_every_parameter_under_its_own_name_and_value(base: Path, tmp_path: Path) -> None:
    from safetensors.torch import load_file

    encoder = load(base)
    with torch.no_grad():  # a "trained" LayerNorm: the parameters the legacy names covered
        encoder.model.embeddings.LayerNorm.weight.add_(0.25)
        encoder.model.encoder.layer[1].output.LayerNorm.bias.sub_(0.5)
    out = tmp_path / "checkpoint"
    out.mkdir()
    encoder.save(out, max_seq_length=32)
    saved = load_file(str(out / "model.safetensors"))
    ours = snapshot(encoder)
    assert set(saved) == set(ours) and not [k for k in saved if k.endswith(("gamma", "beta"))]
    assert all(torch.equal(saved[k], ours[k]) for k in ours)  # every value, exactly
    fresh = load(base)
    fresh.load_weights(out)  # the trainer's strict reload
    assert all(torch.equal(fresh.model.state_dict()[k], ours[k]) for k in ours)
    assert all(torch.equal(load(out).model.state_dict()[k], ours[k]) for k in ours)  # and from_pretrained


def test_the_legacy_names_are_refused_not_renamed(base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from safetensors.torch import load_file

    encoder = load(base)
    legacy = tmp_path / "legacy"
    encoder.model.save_pretrained(legacy, save_original_format=True)  # what save() wrote before the fix
    names = set(load_file(str(legacy / "model.safetensors")))
    assert {"embeddings.LayerNorm.gamma", "embeddings.LayerNorm.beta"} <= names  # transformers 5's reverse rename
    fresh = load(base)
    before = snapshot(fresh)
    with pytest.raises(TrainerError, match="legacy LayerNorm names gamma/beta.*cannot be resumed"):
        fresh.load_weights(legacy)
    assert all(torch.equal(fresh.model.state_dict()[k], v) for k, v in before.items())  # nothing loaded
    # A save that wrote other names than the model's is caught when it is written.
    original = encoder.model.save_pretrained
    monkeypatch.setattr(encoder.model, "save_pretrained",
                        lambda directory, **kwargs: original(directory, save_original_format=True))
    (tmp_path / "bad").mkdir()
    with pytest.raises(TrainerError, match="does not hold the model's parameters as named"):
        encoder.save(tmp_path / "bad", max_seq_length=32)


def _step(encoder: Any, optimizer: Any, scheduler: Any) -> float:
    encoder.train()
    loss = contrastive_loss(encoder.encode(QUERIES, 16), encoder.encode(PASSAGES, 16), 0.05)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def _objects(encoder: Any, total: int) -> tuple[Any, Any, Any]:
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, 1))
    return optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False)


def test_a_resumed_run_continues_exactly_as_the_uninterrupted_one(base: Path, world: World, tmp_path: Path) -> None:
    inputs = validated_inputs(world, tmp_path, max_steps=4)
    schedule = plan_steps(inputs)
    seed_everything(42)
    straight = load(base)
    objects = _objects(straight, 4)
    _step(straight, *objects[:2])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = _save_checkpoint(run_dir, 1, 4, straight, *objects, inputs, schedule, [{"step": 1, "epoch": 0}],
                            torch.device("cpu"), {}, straight.describe())
    after_one = snapshot(straight)
    adam_one = copy.deepcopy(objects[0].state_dict())
    schedule_one = copy.deepcopy(objects[1].state_dict())
    second = _step(straight, *objects[:2])  # the uninterrupted run's next step, from the saved random state
    resumed = load(base)
    fresh = _objects(resumed, 4)
    done, history, selection = restore_checkpoint(path, resumed, *fresh, schedule, validating=False)
    assert (done, history, selection) == (1, [{"step": 1, "epoch": 0}], None)
    assert all(torch.equal(resumed.model.state_dict()[k], v) for k, v in after_one.items())  # identical weights
    restored = fresh[0].state_dict()
    assert restored["param_groups"] == adam_one["param_groups"]
    assert all(torch.equal(restored["state"][i][key], adam_one["state"][i][key])  # AdamW's moments and step
               for i in adam_one["state"] for key in adam_one["state"][i])
    assert fresh[1].state_dict() == schedule_one and fresh[1].last_epoch == 1  # the scheduler's position
    assert fresh[2].state_dict() == torch.amp.GradScaler("cpu", enabled=False).state_dict()
    assert _step(resumed, *fresh[:2]) == second  # the same loss: same weights, optimizer state and dropout masks
    assert all(torch.equal(resumed.model.state_dict()[k], v) for k, v in snapshot(straight).items())


def test_the_rehearsal_checkpoint_check_passes_with_a_from_pretrained_model(base: Path, world: World, tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.training import gpu_rehearsal

    # The tiny vocabulary tokenizes the fixture passages differently from the fixture preparation, so the trainer's
    # length check (run on the real data by the preflight) would refuse them; it only feeds the metadata here.
    monkeypatch.setattr(gpu_rehearsal, "check_lengths", lambda inputs, encoder: {"longest_query_tokens": 0})
    # The benchmark's loader needs a known base model's input format (BGE's); the fixture's is "fake/whitespace", so
    # the checkpoint is read by sentence-transformers itself, the backend that loader uses.
    sentence_transformers = pytest.importorskip("sentence_transformers")

    class Loaded:
        def __init__(self, path: Path) -> None:
            self.model = sentence_transformers.SentenceTransformer(str(path), device="cpu")

        def encode(self, texts: list[str], batch_size: int) -> Any:
            return self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                                     normalize_embeddings=False)

    monkeypatch.setattr(gpu_rehearsal, "inspect_checkpoint", lambda path: path)
    monkeypatch.setattr(gpu_rehearsal, "load_checkpoint_encoder", lambda path, device: Loaded(path))
    inputs = validated_inputs(world, tmp_path, max_steps=4)
    encoder = load(base)
    ctx = Context(inputs.config, "x.yaml", inputs, {42: plan_steps(inputs)}, encoder, lambda model: load(base),
                  torch.device("cpu"), False, tmp_path / "out", lambda text: None)
    ctx.shared["validation"] = {"global": {"mrr@10": 0.5}, "company": {"mrr@10": 0.5}, "per_question": []}
    (tmp_path / "out").mkdir()
    check = check_checkpoint(ctx)
    assert check.status == "pass", check.detail
    for flag in ("hashes_match", "random_state_replays", "weights_identical", "optimizer_state_restored",
                 "scheduler_state_restored", "scaler_state_restored", "selection_state_restored", "overwrite_refused",
                 "no_partial_write"):
        assert check.data[flag] is True, flag
    st = check.data["sentence_transformers_max_abs_difference"]
    assert st is not None and st <= 1e-4  # the benchmark's loader reads the same weights
