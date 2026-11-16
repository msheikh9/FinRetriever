"""The GPU rehearsal tool: its checks on CPU, its no-training guard, and its stop on an fp16 overflow (offline).

The CUDA-only measurements (fp16 GradCache, the loss scale, peak memory) are skipped on CPU; their logic is shared
with the checks exercised here, and they run only on the GPU, after approval."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from finretriever.training import gpu_rehearsal  # noqa: E402
from finretriever.training.gpu_rehearsal import (  # noqa: E402
    Check,
    Context,
    RehearsalError,
    check_attention_dropout,
    check_dropout,
    check_gradcache_fp32,
    main,
    no_optimizer_steps,
    run_rehearsal,
    weights_digest,
)
from tests.unit.trainer_fixtures import (  # noqa: E402
    TinyAttentionEncoder,
    TinyEncoder,
    validated_config,
    validated_world,
)
from tests.unit.validation_fixtures import World  # noqa: E402

EXPECTED_ON_CPU = {
    "environment": "skipped", "dropout": "pass", "attention_dropout_replay": "pass", "gradcache_fp32": "pass",
    "gradcache_fp16": "skipped", "fp16_loss_scale": "skipped", "peak_memory": "skipped",
    "validation_full_corpus": "pass", "checkpoint_resume": "pass", "no_training": "pass",
}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return validated_world(tmp_path_factory.mktemp("rehearsal"))


def config_path(world: World, tmp_path: Path) -> Path:
    return validated_config(world, tmp_path, seeds=[42, 1337, 2026], mini_batch_size=2, batching="balanced",
                            max_steps=None, epochs=5)[1]


def test_no_optimizer_step_can_be_taken_while_the_guard_is_on() -> None:
    weight = torch.nn.Parameter(torch.ones(3))
    weight.grad = torch.ones(3)
    with no_optimizer_steps() as attempts:
        for optimizer in (torch.optim.AdamW([weight], lr=0.1), torch.optim.SGD([weight], lr=0.1)):
            with pytest.raises(RehearsalError, match="never takes an optimizer step"):
                optimizer.step()
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with pytest.raises(RehearsalError, match="GradScaler.step"):
            scaler.step(torch.optim.SGD([weight], lr=0.1))
    assert attempts == ["AdamW", "SGD", "GradScaler"] and torch.equal(weight.detach(), torch.ones(3))
    torch.optim.SGD([weight], lr=0.1).step()  # restored afterwards
    assert not torch.equal(weight.detach(), torch.ones(3))


def test_every_check_runs_on_cpu_and_nothing_is_trained(world: World, tmp_path: Path) -> None:
    path = config_path(world, tmp_path)
    out_dir = tmp_path / "rehearsal_outputs"
    text = io.StringIO()
    code = main(["--config", str(path), "--out", str(out_dir), "--allow-cpu"], out=text,
                encoder_loader=lambda model: TinyAttentionEncoder())
    assert code == 3 and "NOT A GPU REHEARSAL" in text.getvalue()  # CUDA checks skipped: not ready
    report = json.loads((out_dir / "rehearsal.json").read_text())
    assert {c["name"]: c["status"] for c in report["checks"]} == EXPECTED_ON_CPU
    assert report["trained"] is False and report["ready"] is False and report["stopped"] is None
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["no_training"]["data"]["optimizer_steps_attempted"] == []
    assert checks["no_training"]["data"]["weights_at_load"] == checks["no_training"]["data"]["weights_at_end"]
    assert checks["gradcache_fp32"]["data"]["relative_difference"] < 1e-5
    assert checks["attention_dropout_replay"]["data"]["restored"] is True
    resume = checks["checkpoint_resume"]["data"]
    assert all(resume[k] for k in ("hashes_match", "weights_identical", "optimizer_state_restored",
                                   "scheduler_state_restored", "scaler_state_restored", "selection_state_restored",
                                   "random_state_replays", "overwrite_refused", "no_partial_write"))
    validation = checks["validation_full_corpus"]["data"]
    assert validation["identical_on_repeat"] and validation["weights_unchanged"] and validation["random_state_unchanged"]
    step0 = json.loads((out_dir / "validation_step0.json").read_text())
    assert step0["questions"] == validation["questions"] and "mrr@10" not in text.getvalue()  # never printed
    assert (out_dir / "run" / "checkpoint-step-000000" / "trainer_state.pt").is_file()
    assert {s["batching"] for s in report["schedules"].values()} == {"balanced"}
    assert all(s["dropped"] == 0 for s in report["schedules"].values())
    assert report["environment"]["device_type"] == "cpu"
    err = io.StringIO()  # a rehearsal never writes into an earlier one's directory
    assert main(["--config", str(path), "--out", str(out_dir), "--allow-cpu"], out=io.StringIO(), err=err,
                encoder_loader=lambda model: TinyAttentionEncoder()) == 1 and "not empty" in err.getvalue()


def test_without_cuda_it_refuses_unless_allowed_and_the_dry_run_loads_nothing(world: World, tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    path = config_path(world, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RehearsalError, match="no CUDA device"):
        run_rehearsal(str(path), out_dir=tmp_path / "out", encoder_loader=lambda m: TinyAttentionEncoder())
    out = io.StringIO()
    assert main(["--config", str(path), "--dry-run"], out=out) == 0
    assert "dry run: no model loaded" in out.getvalue() and "balanced batches" in out.getvalue()
    assert not (tmp_path / "out").exists()
    _, unvalidated = validated_config(world, tmp_path / "plain", validation=False)
    with pytest.raises(RehearsalError, match="validated GradCache configs"):
        run_rehearsal(str(unvalidated), out_dir=tmp_path / "x", allow_cpu=True)


def test_an_fp16_overflow_stops_the_rehearsal_and_leaves_the_scaler_alone(world: World, tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    def overflowing(ctx: Context) -> Check:  # as check 6 reports an overflow at the initial scale
        ctx.stopped = "fp16 overflow at the GradScaler's initial scale 65536 on seed 42: first batch"
        return Check("fp16_loss_scale", "fail", ctx.stopped, {"initial_scale": 65536.0})

    monkeypatch.setattr(gpu_rehearsal, "check_loss_scale", overflowing)
    out_dir = tmp_path / "out"
    text = io.StringIO()
    code = main(["--config", str(config_path(world, tmp_path)), "--out", str(out_dir), "--allow-cpu"], out=text,
                encoder_loader=lambda model: TinyAttentionEncoder())
    report = json.loads((out_dir / "rehearsal.json").read_text())
    status = {c["name"]: c["status"] for c in report["checks"]}
    assert code == 1 and "REHEARSAL FAILED" in text.getvalue() and "stopped" in text.getvalue()
    assert status["fp16_loss_scale"] == "fail" and report["stopped"].startswith("fp16 overflow")
    assert [status[n] for n in ("peak_memory", "validation_full_corpus", "checkpoint_resume")] == ["skipped"] * 3
    assert status["no_training"] == "pass"  # the final weight check still runs
    assert not (out_dir / "run").exists()


def test_a_check_that_tries_to_train_fails_the_rehearsal(world: World, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def training(ctx: Context) -> Check:
        torch.optim.SGD(ctx.module.parameters(), lr=1.0).step()
        return Check("peak_memory", "pass", "unreachable")

    monkeypatch.setattr(gpu_rehearsal, "check_peak_memory", training)
    out_dir = tmp_path / "out"
    assert main(["--config", str(config_path(world, tmp_path)), "--out", str(out_dir), "--allow-cpu"],
                out=io.StringIO(), encoder_loader=lambda model: TinyAttentionEncoder()) == 1
    checks = {c["name"]: c for c in json.loads((out_dir / "rehearsal.json").read_text())["checks"]}
    assert checks["peak_memory"]["status"] == "fail" and "never takes an optimizer step" in checks["peak_memory"][
        "detail"]
    assert checks["no_training"]["status"] == "fail" and checks["no_training"]["data"]["optimizer_steps_attempted"]


def _context(encoder: Any, tmp_path: Path) -> Context:
    """A context over a few literal texts, for the checks that need only texts and the encoder."""
    texts = ["net income was million", "revenue was million of the", "cash and debt of the", "the net revenue was",
             "income of the cash", "debt was the revenue", "million of net cash", "the revenue and debt"] * 2
    pairs = [SimpleNamespace(pair_id=f"p{i}", positive_passage_id=f"x{i}") for i in range(len(texts))]
    inputs = SimpleNamespace(pairs=pairs, by_id={p.pair_id: p for p in pairs},
                             queries={p.pair_id: "find : " + t for p, t in zip(pairs, texts, strict=True)},
                             passages={p.positive_passage_id: t for p, t in zip(pairs, texts, strict=True)})
    training = SimpleNamespace(temperature=0.05, query_max_length=16, passage_max_length=16)
    return Context(SimpleNamespace(training=training), "x.yaml", inputs, {}, encoder, lambda m: encoder,  # type: ignore[arg-type]
                   torch.device("cpu"), False, tmp_path, lambda s: None)


def _bert(tmp_path: Path, attention: str) -> Any:
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
             "the", "of", "cash", "debt", "and"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(words) + "\n")
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64, hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1, attn_implementation=attention)
    return HFEncoder(BertModel(config), BertTokenizer(vocab_file=str(vocab), do_lower_case=True), "cls")


def test_the_dropout_checks_on_a_real_small_bert(tmp_path: Path) -> None:
    ctx = _context(_bert(tmp_path, "eager"), tmp_path)
    before = weights_digest(ctx.module)
    dropout = check_dropout(ctx)
    assert dropout.status == "pass" and dropout.data["attention_implementation"] == "eager"
    assert dropout.data["attention_dropout_modules"] == 2 and dropout.data["probabilities"] == [0.1]
    attention = check_attention_dropout(ctx)
    assert attention.status == "pass" and attention.data["restored"]
    assert attention.data["fp32"]["other_state_relative_difference"] > 1e-3  # attention dropout alone is active
    assert {m.p for m in gpu_rehearsal._dropouts(ctx.module).values()} == {0.1}
    equivalence = check_gradcache_fp32(ctx)
    assert equivalence.status == "pass" and equivalence.data["relative_difference"] < 1e-5
    assert weights_digest(ctx.module) == before and all(p.grad is None for p in ctx.module.parameters())
    # GradCache cannot replay attention it does not run itself: anything but eager attention fails the check.
    sdpa = check_dropout(_context(_bert(tmp_path, "sdpa"), tmp_path))
    assert sdpa.status == "fail" and "not eager" in sdpa.detail


def test_an_encoder_without_attention_dropout_fails_that_check(tmp_path: Path) -> None:
    check = check_attention_dropout(_context(TinyEncoder(), tmp_path))
    assert check.status == "fail" and "attention.self.dropout" in check.detail


# --- check 6's fp16-against-fp32 comparison (the T4 failure: cosine 0.4461 with dropout on) ---------------------


def _precision_context(tmp_path: Path) -> tuple[Context, list[str]]:
    ctx = _context(_bert(tmp_path, "eager"), tmp_path)
    return ctx, [p.pair_id for p in ctx.inputs.pairs[:8]]


def test_dropout_masks_decide_any_comparison_made_with_dropout_on(tmp_path: Path) -> None:
    ctx, ids = _precision_context(tmp_path)
    state = gpu_rehearsal.RandomState(ctx.device)
    gpu_rehearsal._cached_backward(ctx, ids, mini=4, fp16=False, scale=1.0)
    first = gpu_rehearsal._grads(ctx)
    state.restore()
    gpu_rehearsal._cached_backward(ctx, ids, mini=4, fp16=False, scale=1.0)
    replay = gpu_rehearsal._grads(ctx)
    gpu_rehearsal._cached_backward(ctx, ids, mini=4, fp16=False, scale=1.0)  # the generator moved on: other masks
    other = gpu_rehearsal._grads(ctx)
    assert torch.equal(first, replay)  # the same masks: the same gradients, exactly
    assert gpu_rehearsal._agreement(other, first)["cosine"] < 0.99  # other masks alone move them far


def test_the_loss_scale_is_removed_exactly_before_comparing(tmp_path: Path) -> None:
    ctx, ids = _precision_context(tmp_path)
    with gpu_rehearsal.dropout_disabled(ctx.module):
        gpu_rehearsal._cached_backward(ctx, ids, mini=4, fp16=False, scale=65536.0)
        scaled = gpu_rehearsal._grads(ctx, 65536.0)
        gpu_rehearsal._cached_backward(ctx, ids, mini=4, fp16=False, scale=1.0)
        plain = gpu_rehearsal._grads(ctx)
    assert torch.equal(scaled, plain)  # a power-of-two scale, multiplied and divided back: no change at all


def _dtype_dependent_dropout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropout whose mask, for the same random numbers, lands on other elements in an fp16 pass than in an fp32 one: on
    an fp16 tensor or inside fp16 autocast. That is what the T4 run suggests of CUDA's fused dropout. It consumes the
    random stream the same way in both cases. Autocast counts too because in this tiny random BERT the embedding
    dropout, whose input stays fp32 under autocast, dominates the gradient. A fake keyed on the tensor's dtype alone
    leaves it unchanged, and with it the gradients."""
    real = torch.nn.functional.dropout

    def dropout(input: Any, p: float = 0.5, training: bool = True, inplace: bool = False) -> Any:
        if not training or p == 0:
            return input
        keep = torch.rand(input.shape, device=input.device) >= p
        if input.dtype == torch.float16 or torch.is_autocast_enabled(input.device.type):
            keep = keep.flip(-1)
        return input * keep.to(input.dtype) / (1 - p)

    monkeypatch.setattr(torch.nn.functional, "dropout", dropout)
    assert torch.nn.functional.dropout is not real


def test_the_precision_comparison_is_valid_whatever_the_dropout_kernel_does(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, ids = _precision_context(tmp_path)
    before = weights_digest(ctx.module)
    probabilities = {n: m.p for n, m in gpu_rehearsal._dropouts(ctx.module).items()}
    cpu = gpu_rehearsal.precision_comparison(ctx, ids, mini=4, scale=65536.0)  # CPU fp16 autocast
    # On CPU one random state gives fp16 and fp32 the same masks: with dropout on or off, precision alone differs.
    assert cpu["dropout_masks_across_dtypes"]["identical"] is True
    assert cpu["dropout_off"]["cosine"] >= 0.99 and cpu["dropout_on_same_state"]["cosine"] >= 0.99
    assert cpu["dropout_off"]["fp16"]["finite"] and cpu["dropout_off"]["fp32"]["finite"]
    assert cpu["fp32_two_dropout_states"]["cosine"] < 0.99  # what a mask difference alone does
    _dtype_dependent_dropout(monkeypatch)
    simulated = gpu_rehearsal.precision_comparison(ctx, ids, mini=4, scale=65536.0)
    assert simulated["dropout_masks_across_dtypes"]["identical"] is False
    assert simulated["dropout_on_same_state"]["cosine"] < 0.99  # the T4 symptom: the old comparison fails
    assert simulated["dropout_off"] == cpu["dropout_off"]  # the valid comparison does not depend on dropout at all
    assert weights_digest(ctx.module) == before and all(p.grad is None for p in ctx.module.parameters())
    assert {n: m.p for n, m in gpu_rehearsal._dropouts(ctx.module).items()} == probabilities  # restored


def test_dropout_is_restored_even_when_the_comparison_fails(tmp_path: Path) -> None:
    ctx, _ = _precision_context(tmp_path)
    probabilities = {n: m.p for n, m in gpu_rehearsal._dropouts(ctx.module).items()}
    with pytest.raises(RuntimeError, match="boom"), gpu_rehearsal.dropout_disabled(ctx.module):
        assert all(m.p == 0 for m in gpu_rehearsal._dropouts(ctx.module).values())
        raise RuntimeError("boom")
    assert {n: m.p for n, m in gpu_rehearsal._dropouts(ctx.module).items()} == probabilities
