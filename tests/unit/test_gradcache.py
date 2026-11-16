"""GradCache: exact gradients with dropout replayed, the GradScaler's scaling, and refusal of an inexact replay.

Exactness is tested in float64. GradCache computes the full batch's gradient from mini-batch passes, so in float32
its sums run in a different order from a plain full-batch pass and the two round differently, by a few units in
the last place. How many depends on the CPU's kernels and the PyTorch version (the same float32 comparison passes
on one machine and fails on another; docs/ci_gradcache_investigation.md). In float64 that rounding is ~1e-16 of
the largest gradient, so the tests below require agreement to 1e-12 of it: a wrong term, scale or replay would
differ by many orders of magnitude more. A separate float32 test checks that GradCache is as accurate as plain
float32 training against the float64 gradient.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from finretriever.training.biencoder_trainer import (  # noqa: E402
    RandomState,
    TrainerError,
    cached_contrastive_backward,
    chunked_contrastive_backward,
    contrastive_loss,
    gradcache_check,
    train,
)
from tests.unit.trainer_fixtures import TinyEncoder, validated_inputs, validated_world  # noqa: E402
from tests.unit.validation_fixtures import World  # noqa: E402

CPU = torch.device("cpu")
EPS32 = torch.finfo(torch.float32).eps
QUERIES = [f"Find: query about topic {w} and figure {i}" for i, w in enumerate("abcdefg")]
PASSAGES = [f"passage on topic {w} reporting figure {i} for the year {2020 + i}" for i, w in enumerate("abcdefg")]
SETTINGS = {"temperature": 0.05, "query_max_length": 64, "passage_max_length": 64, "device": CPU}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return validated_world(tmp_path_factory.mktemp("gradcache"))


def gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return out


def assert_same_to_float64_rounding(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    """Equal up to float64 rounding: within 1e-12 of the larger tensor's largest magnitude (rounding is ~1e-16)."""
    assert a.dtype == b.dtype == torch.float64, name
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-300)
    worst = float((a - b).abs().max()) / scale
    assert worst <= 1e-12, f"{name}: differs by {worst:.2e} of its largest magnitude"


def float64_model(**kwargs: Any) -> torch.nn.Module:
    model = TinyEncoder(**kwargs).to(torch.float64)
    return model


@pytest.mark.parametrize("mini_batch_size", [1, 2, 3, 7])
def test_gradients_equal_the_same_forward_passes_without_caching_with_dropout_on(mini_batch_size: int) -> None:
    model = float64_model()
    model.train()
    start = RandomState(CPU)
    loss_cached, replay = cached_contrastive_backward(model, QUERIES, PASSAGES, mini_batch_size=mini_batch_size,
                                                      **SETTINGS)
    cached = gradients(model)
    after_cached = torch.get_rng_state()
    start.restore()
    loss_reference = chunked_contrastive_backward(model, QUERIES, PASSAGES, mini_batch_size=mini_batch_size, **SETTINGS)
    reference = gradients(model)
    assert replay == 0.0  # the replayed mini-batches are the cached ones exactly
    assert loss_cached == pytest.approx(loss_reference, rel=1e-12)
    assert set(cached) == set(reference) and cached
    for name in cached:
        assert_same_to_float64_rounding(cached[name], reference[name], name)
    # The random state ends where the uncached passes leave it: the next step draws the same masks either way.
    assert torch.equal(after_cached, torch.get_rng_state())
    # Dropout is on: another random state gives other gradients, so the equality above is not trivial.
    torch.manual_seed(12345)
    cached_contrastive_backward(model, QUERIES, PASSAGES, mini_batch_size=mini_batch_size, **SETTINGS)
    other = gradients(model)
    assert any(not torch.allclose(other[n], cached[n]) for n in cached)


@pytest.mark.parametrize("mini_batch_size", [1, 2, 3, 7])
def test_without_dropout_it_equals_the_plain_full_batch_loss(mini_batch_size: int) -> None:
    model = float64_model()
    model.dropout.p = 0.0
    loss = contrastive_loss(model.encode(QUERIES, 64), model.encode(PASSAGES, 64), 0.05) / 2
    loss.backward()
    plain = gradients(model)
    value, _ = cached_contrastive_backward(model, QUERIES, PASSAGES, mini_batch_size=mini_batch_size, divisor=2,
                                           **SETTINGS)
    cached = gradients(model)
    assert value == pytest.approx(float(loss.detach()), rel=1e-12)
    assert set(cached) == set(plain) and cached
    for name in plain:
        assert_same_to_float64_rounding(cached[name], plain[name], name)


@pytest.mark.parametrize("mini_batch_size", [1, 2, 3])
def test_in_float32_gradcache_is_as_accurate_as_plain_training(mini_batch_size: int) -> None:
    """float32 GradCache and float32 plain training differ only by rounding: measured against the float64
    gradient of the same model, GradCache's error is of the same size as plain training's own."""
    reference_model = float64_model()
    reference_model.dropout.p = 0.0
    contrastive_loss(reference_model.encode(QUERIES, 64), reference_model.encode(PASSAGES, 64), 0.05).backward()
    exact = gradients(reference_model)
    model = TinyEncoder()
    model.dropout.p = 0.0
    contrastive_loss(model.encode(QUERIES, 64), model.encode(PASSAGES, 64), 0.05).backward()
    plain = gradients(model)
    cached_contrastive_backward(model, QUERIES, PASSAGES, mini_batch_size=mini_batch_size, **SETTINGS)
    cached = gradients(model)

    def error(g: torch.Tensor, name: str) -> float:
        return float((g.double() - exact[name]).norm() / exact[name].norm())

    for name in exact:
        plain_error, cached_error = error(plain[name], name), error(cached[name], name)
        assert plain_error < 64 * EPS32, (name, plain_error)  # float32 rounding of this small problem
        assert cached_error <= 4 * max(plain_error, EPS32), (name, cached_error, plain_error)


def test_gradients_are_scaled_and_unscaled_exactly_as_in_plain_training() -> None:
    """The GradScaler multiplies the loss before the backward pass; the optimizer step divides the gradients again.
    GradCache back-propagates the scaled loss to the cached embeddings, so the parameters get the same scaled
    gradients as the plain loss (CPU scaler here; the trainer's fp16 scaler on CUDA works the same way)."""
    results = []
    for cached in (False, True):
        model = float64_model()
        model.dropout.p = 0.0
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0**12)
        if cached:
            cached_contrastive_backward(model, QUERIES, PASSAGES, mini_batch_size=3, scale=scaler.scale, **SETTINGS)
        else:
            scaler.scale(contrastive_loss(model.encode(QUERIES, 64), model.encode(PASSAGES, 64), 0.05)).backward()
        scaled = {n: p.grad.clone() for n, p in model.named_parameters()}
        scaler.unscale_(optimizer)
        results.append(({n: p.grad.clone() for n, p in model.named_parameters()}, scaled))
    (plain, plain_scaled), (cached_grads, cached_scaled) = results
    for name in plain:
        assert_same_to_float64_rounding(cached_scaled[name], plain_scaled[name], name)
        assert_same_to_float64_rounding(cached_grads[name], plain[name], name)
        assert torch.equal(plain_scaled[name], plain[name] * 2.0**12), name  # really scaled (a power of two: exact)


class Drifting(TinyEncoder):
    """Each call adds noise the torch random state does not control: a replay can never match its cached pass."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def encode(self, texts: Sequence[str], max_length: int) -> Any:
        self.calls += 1
        return super().encode(texts, max_length) + 0.1 * self.calls


def test_a_replay_that_differs_from_its_cached_pass_is_refused_before_any_gradient() -> None:
    model = Drifting()
    with pytest.raises(TrainerError, match="replayed mini-batch differs from its cached pass"):
        cached_contrastive_backward(model, QUERIES, PASSAGES, mini_batch_size=2, **SETTINGS)
    assert all(p.grad is None for p in model.parameters())  # the first replay was refused: nothing accumulated


def test_the_check_used_by_the_preflight_reports_agreement() -> None:
    report = gradcache_check(TinyEncoder(), QUERIES, PASSAGES, mini_batch_size=2, temperature=0.05,
                             query_max_length=64, passage_max_length=64, device=CPU)
    assert report["relative_difference"] < 1e-6 and report["cosine"] > 0.999999
    assert report["replay_relative_difference"] == 0.0 and report["parameters"] > 0


def test_a_gradcache_run_trains_as_the_plain_run_and_records_it(world: World, tmp_path: Path) -> None:
    """Without dropout the cached and the plain runs compute the same gradients, so they train the same weights."""
    runs = []
    for name, mini in (("plain", None), ("cached", 1)):
        inputs = validated_inputs(world, tmp_path / name, validation=False, max_steps=4,
                                  **({} if mini is None else {"mini_batch_size": mini}))
        model = TinyEncoder()
        model.dropout.p = 0.0
        runs.append(train(inputs, model))
    plain, cached = runs
    assert [h["loss"] for h in cached.history] == pytest.approx([h["loss"] for h in plain.history], abs=1e-5)
    a = torch.load(plain.checkpoints[-1] / "weights.pt", weights_only=True)
    b = torch.load(cached.checkpoints[-1] / "weights.pt", weights_only=True)
    assert all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)
    assert all(h["gradcache_replay_difference"] == 0.0 for h in cached.history)
    assert all("gradcache_replay_difference" not in h for h in plain.history)
    objective = json.loads((cached.checkpoints[-1] / "metadata.json").read_text())["objective"]
    assert objective["gradient_cache"]["mini_batch_size"] == 1 and objective["in_batch_negatives_per_query"] == 2
    assert "gradient_cache" not in json.loads((plain.checkpoints[-1] / "metadata.json").read_text())["objective"]


def test_a_small_bert_with_dropout_and_eager_attention_replays_exactly(tmp_path: Path) -> None:
    """The real model class: hidden and attention dropout drawn from the generator GradCache replays."""
    try:
        from transformers import BertConfig, BertModel
        try:
            from transformers import BertTokenizerFast as BertTokenizer
        except ImportError:
            from transformers import BertTokenizer  # type: ignore[assignment]
    except Exception as exc:  # e.g. a broken scipy behind transformers' lazy imports
        pytest.skip(f"transformers BERT unavailable: {exc}")
    from finretriever.training.biencoder_trainer import HFEncoder

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "net", "income", "was", "million", "revenue", "find", ":",
             "the", "of", "cash", "debt"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(words) + "\n")
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64, hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1, attn_implementation="eager")
    encoder = HFEncoder(BertModel(config), BertTokenizer(vocab_file=str(vocab), do_lower_case=True), "cls")
    queries = ["find : net income", "find : revenue", "find : cash of the debt", "find : net income was million"]
    passages = ["net income was million", "revenue was million of the", "cash and debt", "the net revenue"]
    report = gradcache_check(encoder, queries, passages, mini_batch_size=2, temperature=0.05, query_max_length=16,
                             passage_max_length=16, device=CPU)
    assert report["replay_relative_difference"] < 1e-6
    assert report["relative_difference"] < 1e-5 and report["cosine"] > 0.99999
