"""GPU rehearsal of a validated fine-tuning run: the CUDA path checked end to end, with no optimizer step.

Run on Kaggle from the bundle directory after ``smoke_preflight`` passes (``scripts/gpu_rehearsal.py``), with the
experiment's own config. It loads the pinned base model as a GradCache run does (eager attention, dropout on) and
runs ten checks with the trainer's own code. Each passes, warns, fails or is skipped:

1. ``environment``: the GPU, CUDA, cuDNN and library versions (``biencoder_trainer.run_environment``), and fp16
   autocast. Every seed of the experiment must run on this GPU type; the selection records and the decision rule
   check it.
2. ``dropout``: real BGE in training mode. Dropout is active (different random states give different embeddings) and
   replays exactly from a saved state, in fp32 and fp16. Attention is eager.
3. ``attention_dropout_replay``: as 2 with every other dropout set to p = 0 for the check only (restored and
   verified), so attention dropout alone is active, and replays.
4. ``gradcache_fp32``: GradCache's gradients against the same forward passes without caching. Run on the 8 pairs
   with the longest positives, mini-batch 4, on the GPU.
5. ``gradcache_fp16``: as 4 under fp16 autocast, with the loss scaled by the GradScaler's real initial scale and by
   1,024.
6. ``fp16_loss_scale``: a full cached backward at the GradScaler's initial scale on each seed's first batch and on the
   full 64-pair worst case; every gradient must be finite. Then fp16 gradients are compared with fp32 ones of the same
   batch **with dropout off in both** (``precision_comparison``), so both compute the same function and only precision
   differs. With dropout on, the two precisions need not draw the same masks: on CUDA one random state need not give
   fp16 and fp32 tensors the same dropout mask. Whether it does, the old dropout-on comparison, and the gradient change
   a mask difference alone causes are recorded, never deciding.
7. ``peak_memory``: one full cached step of the 64 pairs with the longest positives, with AdamW's two moment buffers
   allocated, measured by ``torch.cuda``.
8. ``validation_full_corpus``: the step-0 validation pass over the whole corpus, twice. It must be identical on repeat
   and change no weight, mode or random state. Its time and memory are recorded; its value is written, never printed.
9. ``checkpoint_resume``: ``checkpoint-step-000000`` is written by the trainer's ``_save_checkpoint`` with an
   un-stepped AdamW, the scheduler, the scaler, the random state and the step-0 evaluation. Then:
   - it is re-hashed;
   - it is reloaded into a fresh base model by the trainer's ``restore_checkpoint`` (weights, optimizer, scheduler,
     scaler, random and selection state);
   - it is loaded by the benchmark's sentence-transformers loader;
   - saving it again is refused.
10. ``no_training``: no optimizer step was attempted, and the final weights equal the weights at load.

**No optimizer step.** For the whole rehearsal, every ``torch.optim`` optimizer's ``step`` and ``GradScaler.step``
raise ``RehearsalError`` (``no_optimizer_steps``). Gradients are cleared after each check.

**fp16 overflow.** If any gradient is not finite at the GradScaler's real initial scale (checks 5 and 6), the
rehearsal stops and reports it. The remaining checks are skipped, except 10. Smaller scales are probed only to
inform your decision; no scaler setting is changed.

**No evaluation data.** It reads the bundle's config, preparation, candidates and validation set, as training does.

**Output.** Under ``--out`` (default ``rehearsal_outputs``, a runtime output of the bundle):
- ``rehearsal.json``, the report;
- ``validation_step0.json``, the step-0 validation result, kept for the N10 identity check across seeds;
- ``run/``, check 9's checkpoint.

**Exit codes:** 0 every check passed; 1 a check failed; 3 nothing failed but checks were skipped (no CUDA, or
``--allow-cpu``): not a GPU rehearsal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import torch
import torch.nn.functional as F

from finretriever.retrieval.checkpoint import inspect_checkpoint, load_checkpoint_encoder
from finretriever.training import validation_eval
from finretriever.training.biencoder_trainer import (
    REPLAY_TOLERANCE,
    STATE_FILE,
    Encoder,
    HFEncoder,
    RandomState,
    TrainerError,
    _parameter_groups,
    _save_checkpoint,
    _set_rng_state,
    cached_contrastive_backward,
    check_lengths,
    gradcache_check,
    load_base_encoder,
    restore_checkpoint,
    run_environment,
)
from finretriever.training.checkpoint_selection import (
    SelectionState,
    checkpoint_files,
    files_digest,
)
from finretriever.training.trainer_setup import (
    BaseModelConfig,
    TrainerConfig,
    TrainingInputs,
    load_trainer_config,
    lr_factor,
    plan_steps,
    prepare_inputs,
)

KIND = "finretriever_gpu_rehearsal"
FORMAT_VERSION = 1
PREFIX = "[gpu_rehearsal]"
DEFAULT_OUT = "rehearsal_outputs"
TEXTS = 8  # real passages for the dropout checks
EQUIVALENCE = (8, 4)  # pairs and mini-batch size of the GradCache equivalence checks
PRECISION = (16, 8)  # pairs and mini-batch size of the fp16-against-fp32 gradient comparison
DROPOUT_CHANGE = 1e-3  # another dropout mask changes the embeddings by more than this (relative)
GRADCACHE_TOLERANCE = {"fp32": 1e-5, "fp16": 1e-3}  # relative difference of the gradient vectors
GRADCACHE_COSINE = {"fp32": 0.999999, "fp16": 0.999}
PRECISION_COSINE = 0.99  # fp16 against fp32 gradients of the same batch, dropout off in both
MASK_SHAPE = (8, 512, 768)  # a hidden-state-sized tensor for the dropout-mask agreement diagnostic
MEMORY_LIMIT = 0.85  # peak reserved memory, as a share of the GPU's
SLICE = 256  # passages in the fp16-against-fp32 embedding comparison
PROBE_SCALES = tuple(2.0**k for k in range(15, 7, -1))  # after an overflow only, to inform the decision
NOTICE = (
    "A mechanics rehearsal: no optimizer step was taken and no model was trained. The validation pass is the "
    "base model's (step 0), kept for the identity check across seeds; it selects nothing and is not a result."
)


class RehearsalError(RuntimeError):
    """The rehearsal cannot run, or an optimizer step was attempted."""


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail | skipped
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    config: TrainerConfig
    config_path: str
    inputs: TrainingInputs
    schedules: dict[int, Any]
    encoder: Encoder
    loader: Callable[[BaseModelConfig], Encoder]
    device: torch.device
    fp16: bool
    out: Path
    say: Callable[[str], None]
    init_scale: float = 1.0
    stopped: str | None = None  # why the remaining checks are skipped (an fp16 overflow)
    shared: dict[str, Any] = field(default_factory=dict)

    @property
    def module(self) -> torch.nn.Module:
        return self.encoder  # type: ignore[return-value]

    @property
    def cuda(self) -> bool:
        return self.device.type == "cuda"


# --- helpers ------------------------------------------------------------------------------------


def _subclasses(cls: type) -> set[type]:
    found = set()
    for sub in cls.__subclasses__():
        found |= {sub, *_subclasses(sub)}
    return found


@contextmanager
def no_optimizer_steps() -> Iterator[list[str]]:
    """While active, every ``torch.optim`` optimizer's ``step`` and ``GradScaler.step`` raise; yields the attempts."""
    attempts: list[str] = []

    def forbidden(self: Any, *args: Any, **kwargs: Any) -> Any:
        attempts.append(type(self).__name__)
        raise RehearsalError(f"{type(self).__name__}.step was called: the rehearsal never takes an optimizer step")

    classes = {torch.optim.Optimizer, *_subclasses(torch.optim.Optimizer), torch.amp.GradScaler,
               *_subclasses(torch.amp.GradScaler)}
    saved = [(cls, cls.__dict__["step"]) for cls in classes if "step" in cls.__dict__]
    for cls, _ in saved:
        cls.step = forbidden  # type: ignore[method-assign]
    try:
        yield attempts
    finally:
        for cls, original in saved:
            cls.step = original  # type: ignore[method-assign]


def weights_digest(module: torch.nn.Module) -> str:
    """SHA-256 over every parameter and buffer (name, dtype, shape and bytes)."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().to("cpu").contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() if value.numel() else b"")
    return digest.hexdigest()


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def _forward(ctx: Context, texts: Sequence[str], fp16: bool, state: RandomState | None = None) -> torch.Tensor:
    if state is not None:
        state.restore()
    with torch.no_grad(), torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=fp16):
        return ctx.encoder.encode(list(texts), ctx.config.training.passage_max_length).float()


def _dropouts(module: torch.nn.Module) -> dict[str, torch.nn.Dropout]:
    return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Dropout)}


def _attention(dropouts: dict[str, torch.nn.Dropout]) -> list[str]:
    return sorted(name for name in dropouts if name.endswith("attention.self.dropout"))


def _replays(ctx: Context, texts: Sequence[str], precisions: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    """Problems and measurements: dropout must change the embeddings, and a restored state must replay them."""
    problems, data = [], {}
    ctx.module.train()
    for precision in precisions:
        fp16 = precision == "fp16"
        state = RandomState(ctx.device)
        first = _forward(ctx, texts, fp16, state)
        replay = _forward(ctx, texts, fp16, state)
        other = _forward(ctx, texts, fp16)  # the generator has moved on: other masks
        data[precision] = {"replay_relative_difference": _relative(replay, first),
                           "other_state_relative_difference": _relative(other, first)}
        if data[precision]["replay_relative_difference"] > REPLAY_TOLERANCE[precision]:
            problems.append(f"{precision}: a restored random state does not replay the forward pass")
        if data[precision]["other_state_relative_difference"] <= DROPOUT_CHANGE:
            problems.append(f"{precision}: another random state gives the same embeddings (dropout inactive)")
    return problems, data


def _longest(ctx: Context, count: int) -> list[str]:
    """Pair IDs with the longest positive passages (in the model's tokens)."""
    pairs = list(ctx.inputs.pairs)
    tokens = ctx.encoder.token_counts([ctx.inputs.passages[p.positive_passage_id] for p in pairs])
    ranked = sorted(zip(tokens, (p.pair_id for p in pairs), strict=True), key=lambda x: (-x[0], x[1]))
    return [pid for _, pid in ranked[:count]]


def _texts(ctx: Context, ids: Sequence[str]) -> tuple[list[str], list[str]]:
    by_id = ctx.inputs.by_id
    return ([ctx.inputs.queries[i] for i in ids],
            [ctx.inputs.passages[by_id[i].positive_passage_id] for i in ids])


def _settings(ctx: Context) -> dict[str, Any]:
    cfg = ctx.config.training
    return {"temperature": cfg.temperature, "query_max_length": cfg.query_max_length,
            "passage_max_length": cfg.passage_max_length, "device": ctx.device}


def _grads(ctx: Context, scale: float = 1.0) -> torch.Tensor:
    params = [p for p in ctx.module.parameters() if p.requires_grad]
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
                      for p in params]) / scale
    ctx.module.zero_grad(set_to_none=True)
    return flat


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a.split(1 << 22), b.split(1 << 22), strict=True))


def _agreement(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Cosine and relative difference of two flat gradient vectors (b the reference)."""
    return {"cosine": _dot(a, b) / max((_dot(a, a) * _dot(b, b)) ** 0.5, 1e-24),
            "relative_difference": _dot(a - b, a - b) ** 0.5 / max(_dot(b, b) ** 0.5, 1e-12)}


def _summary(flat: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(flat)
    return {"finite": bool(finite.all()), "zero_share": float((flat == 0).float().mean()),
            "max_abs": float(flat[finite].abs().max()) if finite.any() else None}


def dropout_mask_agreement(device: torch.device, p: float = 0.1) -> dict[str, Any]:
    """Whether one random state draws the same dropout mask for an fp16 and an fp32 tensor on ``device``. On CPU it
    does; on CUDA the fused dropout kernel's mapping of random numbers to elements may depend on the element size."""
    state = RandomState(device)
    full = F.dropout(torch.ones(MASK_SHAPE, device=device, dtype=torch.float32), p=p, training=True) != 0
    state.restore()
    half = F.dropout(torch.ones(MASK_SHAPE, device=device, dtype=torch.float16), p=p, training=True) != 0
    state.restore()
    return {"shape": list(MASK_SHAPE), "p": p, "identical": bool(torch.equal(full, half)),
            "agreement": float((full == half).float().mean())}


def precision_comparison(ctx: Context, ids: Sequence[str], *, mini: int, scale: float) -> dict[str, Any]:
    """fp16 against fp32 gradients of one batch, as a valid precision measurement, plus the diagnostics.

    Deciding: dropout off in both passes, so the same pairs, weights, loss and normalisation (the in-batch InfoNCE on
    float embeddings, mean over the batch) give the same function, and only precision differs. The fp16 loss is
    multiplied by ``scale``, as the GradScaler does, and its gradients are divided by the same ``scale`` before the
    comparison. Reported only: the dropout-mask agreement across dtypes, the same comparison with dropout on from one
    random state (valid only if the masks agree), and fp32 against fp32 under two random states (what a mask
    difference alone does)."""
    out: dict[str, Any] = {"pairs": len(ids), "mini_batch_size": mini, "scale": scale}
    with dropout_disabled(ctx.module):
        _cached_backward(ctx, ids, mini=mini, fp16=True, scale=scale)
        half = _grads(ctx, scale)
        _cached_backward(ctx, ids, mini=mini, fp16=False, scale=1.0)
        full = _grads(ctx)
    out["dropout_off"] = {**_agreement(half, full), "fp16": _summary(half), "fp32": _summary(full)}
    del half, full
    out["dropout_masks_across_dtypes"] = dropout_mask_agreement(ctx.device)
    first = RandomState(ctx.device)
    _cached_backward(ctx, ids, mini=mini, fp16=True, scale=scale)
    half = _grads(ctx, scale)
    first.restore()
    _cached_backward(ctx, ids, mini=mini, fp16=False, scale=1.0)
    full = _grads(ctx)  # the generator has moved on: the next pass draws other masks
    _cached_backward(ctx, ids, mini=mini, fp16=False, scale=1.0)
    other = _grads(ctx)
    out["dropout_on_same_state"] = {**_agreement(half, full), "note": "a precision measure only if the masks agree"}
    out["fp32_two_dropout_states"] = {**_agreement(other, full), "note": "the change a mask difference alone causes"}
    del half, full, other
    return out


@contextmanager
def dropout_disabled(module: torch.nn.Module) -> Iterator[None]:
    """Every dropout module at p = 0 (attention dropout too: BERT passes its module's p), restored and verified after."""
    dropouts = _dropouts(module)
    saved = {name: m.p for name, m in dropouts.items()}
    for m in dropouts.values():
        m.p = 0.0
    try:
        yield
    finally:
        for name, m in dropouts.items():
            m.p = saved[name]
    if any(m.p != saved[name] for name, m in dropouts.items()):
        raise RehearsalError("the dropout probabilities were not restored")


def _cached_backward(ctx: Context, ids: Sequence[str], *, mini: int, fp16: bool, scale: float) -> tuple[bool, float]:
    """One full cached backward (no step); whether every scaled gradient is finite, and the replay difference."""
    queries, passages = _texts(ctx, ids)
    ctx.module.train()
    _, replay = cached_contrastive_backward(ctx.encoder, queries, passages, mini_batch_size=mini, fp16=fp16,
                                            scale=lambda loss: loss * scale, **_settings(ctx))
    finite = all(bool(torch.isfinite(p.grad).all()) for p in ctx.module.parameters() if p.grad is not None)
    return finite, replay


# --- the checks ---------------------------------------------------------------------------------


def check_environment(ctx: Context) -> Check:
    env = run_environment(ctx.device)
    ctx.shared["environment"] = env
    if not ctx.cuda:
        return Check("environment", "skipped", "no CUDA device: not a GPU rehearsal (--allow-cpu)", env)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = torch.ones(8, 8, device=ctx.device) @ torch.ones(8, 8, device=ctx.device)
    env["autocast_dtype"] = str(y.dtype)
    if y.dtype != torch.float16 and ctx.config.training.fp16:
        return Check("environment", "fail", f"fp16 autocast gives {y.dtype} on {env['gpu_name']}", env)
    return Check("environment", "pass", f"{env['gpu_name']} (compute {env['gpu_capability']}), CUDA {env['cuda']}, "
                 f"torch {env['torch']}, transformers {env['transformers']}; every seed must run on this GPU type", env)


def check_dropout(ctx: Context) -> Check:
    dropouts = _dropouts(ctx.module)
    config = getattr(getattr(ctx.encoder, "model", None), "config", None)
    data: dict[str, Any] = {
        "dropout_modules": len(dropouts),
        "active": sum(m.p > 0 for m in dropouts.values()),
        "attention_dropout_modules": len(_attention(dropouts)),
        "probabilities": sorted({m.p for m in dropouts.values()}),
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", None),
        "attention_probs_dropout_prob": getattr(config, "attention_probs_dropout_prob", None),
    }
    problems = []
    if not data["active"]:
        problems.append("no active dropout module")
    if data["attention_implementation"] not in (None, "eager"):
        problems.append(f"attention is {data['attention_implementation']}, not eager: GradCache cannot replay it")
    texts = _texts(ctx, [p.pair_id for p in ctx.inputs.pairs[:TEXTS]])[1]
    found, measured = _replays(ctx, texts, ["fp32", "fp16"] if ctx.fp16 else ["fp32"])
    data.update(measured)
    if problems := problems + found:
        return Check("dropout", "fail", "; ".join(problems), data)
    return Check("dropout", "pass", f"{data['active']} active dropout modules ({data['attention_implementation'] or 'n/a'} "
                 f"attention); other random states change the embeddings and a restored state replays them "
                 f"({', '.join(measured)})", data)


def check_attention_dropout(ctx: Context) -> Check:
    dropouts = _dropouts(ctx.module)
    attention = _attention(dropouts)
    if not attention:
        return Check("attention_dropout_replay", "fail", "no attention-dropout module (*.attention.self.dropout): "
                     "attention dropout cannot be checked on its own")
    saved = {name: m.p for name, m in dropouts.items()}
    try:
        for name, m in dropouts.items():
            if name not in attention:
                m.p = 0.0
        texts = _texts(ctx, [p.pair_id for p in ctx.inputs.pairs[:TEXTS]])[1]
        problems, data = _replays(ctx, texts, ["fp32", "fp16"] if ctx.fp16 else ["fp32"])
    finally:
        for name, m in dropouts.items():
            m.p = saved[name]
    data["restored"] = all(m.p == saved[name] for name, m in dropouts.items())
    data["attention_modules"] = len(attention)
    if not data["restored"]:
        problems.append("the dropout probabilities were not restored")
    if problems:
        return Check("attention_dropout_replay", "fail", "; ".join(problems), data)
    return Check("attention_dropout_replay", "pass", f"attention dropout alone ({len(attention)} modules) changes the "
                 "embeddings and replays from a restored state; every probability restored", data)


def _equivalence(ctx: Context, fp16: bool, scale: float | None) -> dict[str, Any]:
    queries, passages = _texts(ctx, _longest(ctx, EQUIVALENCE[0]))
    report = gradcache_check(ctx.encoder, queries, passages, mini_batch_size=EQUIVALENCE[1], fp16=fp16,
                             loss_scale=scale, **_settings(ctx))
    ctx.module.zero_grad(set_to_none=True)
    return report


def check_gradcache_fp32(ctx: Context) -> Check:
    report = _equivalence(ctx, fp16=False, scale=None)
    ok = (report["finite"] and report["relative_difference"] <= GRADCACHE_TOLERANCE["fp32"]
          and report["cosine"] >= GRADCACHE_COSINE["fp32"]
          and report["replay_relative_difference"] <= REPLAY_TOLERANCE["fp32"])
    detail = (f"GradCache against the uncached passes on the {EQUIVALENCE[0]} longest pairs (mini-batch "
              f"{EQUIVALENCE[1]}, dropout on, {ctx.device.type}): relative {report['relative_difference']:.1e}, cosine "
              f"{report['cosine']:.7f}, replay {report['replay_relative_difference']:.1e}")
    return Check("gradcache_fp32", "pass" if ok else "fail", detail, report)


def check_gradcache_fp16(ctx: Context) -> Check:
    if not ctx.fp16:
        return Check("gradcache_fp16", "skipped", "fp16 needs CUDA and a config with fp16")
    reports = {}
    for scale in (ctx.init_scale, 1024.0):
        reports[f"{scale:g}"] = _equivalence(ctx, fp16=True, scale=scale)
    at_init = reports[f"{ctx.init_scale:g}"]
    if not at_init["finite"]:
        ctx.stopped = f"fp16 overflow in the GradCache check at the GradScaler's initial scale {ctx.init_scale:g}"
        return Check("gradcache_fp16", "fail", ctx.stopped + "; stopped, scaler unchanged", reports)
    bad = [s for s, r in reports.items() if not (r["finite"] and r["relative_difference"] <= GRADCACHE_TOLERANCE["fp16"]
                                                  and r["cosine"] >= GRADCACHE_COSINE["fp16"]
                                                  and r["replay_relative_difference"] <= REPLAY_TOLERANCE["fp16"])]
    detail = "; ".join(f"scale {s}: relative {r['relative_difference']:.1e}, cosine {r['cosine']:.5f}, replay "
                       f"{r['replay_relative_difference']:.1e}" for s, r in reports.items())
    return Check("gradcache_fp16", "fail" if bad else "pass", f"fp16 GradCache against the uncached passes: {detail}",
                 reports)


def check_loss_scale(ctx: Context) -> Check:
    if not ctx.fp16:
        return Check("fp16_loss_scale", "skipped", "fp16 needs CUDA and a config with fp16")
    mini = ctx.config.training.mini_batch_size or ctx.config.training.batch_size
    batches = {f"seed {seed}: first batch": list(s.steps[0].batches[0]) for seed, s in ctx.schedules.items()}
    batches[f"worst case: the {min(ctx.config.training.batch_size, len(ctx.inputs.pairs))} longest pairs"] = _longest(
        ctx, ctx.config.training.batch_size)
    data: dict[str, Any] = {"initial_scale": ctx.init_scale, "batches": {}}
    for name, ids in batches.items():
        finite, replay = _cached_backward(ctx, ids, mini=mini, fp16=True, scale=ctx.init_scale)
        ctx.module.zero_grad(set_to_none=True)
        data["batches"][name] = {"pairs": len(ids), "finite": finite, "replay_relative_difference": replay}
        if not finite:
            probe = {}
            for scale in (s for s in PROBE_SCALES if s < ctx.init_scale):  # informs the decision; changes nothing
                probe[f"{scale:g}"], _ = _cached_backward(ctx, ids, mini=mini, fp16=True, scale=scale)
                ctx.module.zero_grad(set_to_none=True)
                if probe[f"{scale:g}"]:
                    break
            data["probe"] = probe
            ctx.stopped = (f"fp16 overflow at the GradScaler's initial scale {ctx.init_scale:g} on {name}: a validated "
                           "run would stop at its first step")
            return Check("fp16_loss_scale", "fail", ctx.stopped + f"; finite at smaller scales: {probe}. Stopped; the "
                         "scaler was not changed: a fixed initial scale needs your decision", data)
    data["fp16_against_fp32"] = compared = precision_comparison(ctx, _longest(ctx, PRECISION[0]), mini=PRECISION[1],
                                                                scale=ctx.init_scale)
    off, masks = compared["dropout_off"], compared["dropout_masks_across_dtypes"]
    detail = (f"fp16 against fp32 gradients with dropout off: cosine {off['cosine']:.5f}, relative "
              f"{off['relative_difference']:.1e}; dropout masks across dtypes from one state: agreement "
              f"{masks['agreement']:.3f} (dropout-on cosine {compared['dropout_on_same_state']['cosine']:.4f}, two fp32 "
              f"dropout states {compared['fp32_two_dropout_states']['cosine']:.4f}; reported only)")
    if not (off["fp16"]["finite"] and off["fp32"]["finite"]) or off["cosine"] < PRECISION_COSINE:
        return Check("fp16_loss_scale", "fail", f"fp16 gradients differ from fp32 ones: {detail}", data)
    return Check("fp16_loss_scale", "pass", f"every gradient finite at the initial scale {ctx.init_scale:g} on "
                 f"{len(batches)} full batches (each seed's first, and the 64 longest pairs); {detail}", data)


def check_peak_memory(ctx: Context) -> Check:
    cfg = ctx.config.training
    ids = _longest(ctx, cfg.batch_size)
    params = [p for p in ctx.module.parameters() if p.requires_grad]
    moments = [torch.zeros_like(p, dtype=torch.float32) for p in params for _ in range(2)]  # AdamW's state, emulated
    if ctx.cuda:
        torch.cuda.synchronize(ctx.device)
        torch.cuda.reset_peak_memory_stats(ctx.device)
    started = time.perf_counter()
    finite, _ = _cached_backward(ctx, ids, mini=cfg.mini_batch_size or cfg.batch_size, fp16=ctx.fp16,
                                 scale=ctx.init_scale if ctx.fp16 else 1.0)
    if ctx.cuda:
        torch.cuda.synchronize(ctx.device)
    seconds = time.perf_counter() - started
    ctx.module.zero_grad(set_to_none=True)
    data: dict[str, Any] = {"pairs": len(ids), "mini_batch_size": cfg.mini_batch_size, "fp16": ctx.fp16,
                            "step_seconds": round(seconds, 3), "finite": finite,
                            "adamw_state_gib": round(sum(m.numel() for m in moments) * 4 / 1024**3, 3)}
    del moments
    if not ctx.cuda:
        return Check("peak_memory", "skipped", f"no CUDA device: one cached step of {len(ids)} pairs took "
                     f"{seconds:.1f} s on CPU; memory not measured", data)
    total = torch.cuda.get_device_properties(ctx.device).total_memory
    data.update(peak_allocated_gib=round(torch.cuda.max_memory_allocated(ctx.device) / 1024**3, 2),
                peak_reserved_gib=round(torch.cuda.max_memory_reserved(ctx.device) / 1024**3, 2),
                total_gib=round(total / 1024**3, 2))
    torch.cuda.empty_cache()
    share = data["peak_reserved_gib"] / data["total_gib"]
    status = "pass" if share <= MEMORY_LIMIT and finite else "fail"
    return Check("peak_memory", status, f"one cached step of the {len(ids)} longest pairs (mini-batch "
                 f"{cfg.mini_batch_size}) with AdamW's state: {data['peak_reserved_gib']} GiB reserved of "
                 f"{data['total_gib']} ({share:.0%}; limit {MEMORY_LIMIT:.0%}), {seconds:.2f} s (measured)", data)


def check_validation(ctx: Context) -> Check:
    cfg, inputs = ctx.config, ctx.inputs
    assert cfg.validation is not None and inputs.corpus is not None
    targets = validation_eval.build_targets(inputs.validation, inputs.guard, inputs.corpus,
                                            query_prefix=cfg.base_model.query_prefix,
                                            passage_prefix=cfg.base_model.passage_prefix)
    ctx.module.train()
    before = weights_digest(ctx.module)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(ctx.device) if ctx.cuda else None
    runs, seconds, peaks = [], [], []
    for _ in range(2):
        if ctx.cuda:
            torch.cuda.reset_peak_memory_stats(ctx.device)
        started = time.perf_counter()
        runs.append(validation_eval.evaluate(ctx.encoder, targets, query_max_length=cfg.training.query_max_length,
                                             passage_max_length=cfg.training.passage_max_length,
                                             batch_size=cfg.validation.batch_size, device=ctx.device, fp16=ctx.fp16))
        seconds.append(round(time.perf_counter() - started, 1))
        if ctx.cuda:
            peaks.append(round(torch.cuda.max_memory_allocated(ctx.device) / 1024**3, 2))
    first, second = runs

    def ranks(result: dict[str, Any]) -> list[Any]:
        return [[q["pair_id"], q["global"]["first_relevant_rank"], q["company"]["first_relevant_rank"]]
                for q in result["per_question"]]

    data: dict[str, Any] = {
        "questions": first["questions"], "passages_ranked": first["passages_ranked"], "seconds": seconds,
        "peak_allocated_gib": peaks or None, "fp16": ctx.fp16,
        "identical_on_repeat": ranks(first) == ranks(second) and first["per_question"] == second["per_question"],
        "weights_unchanged": weights_digest(ctx.module) == before,
        "training_mode_restored": ctx.module.training,
        "random_state_unchanged": torch.equal(cpu_state, torch.get_rng_state()) and (
            cuda_state is None or torch.equal(cuda_state, torch.cuda.get_rng_state(ctx.device))),
        "results_sha256": hashlib.sha256(json.dumps(ranks(first)).encode()).hexdigest(),
        "file": "validation_step0.json",
    }
    if ctx.fp16:  # how far fp16 embeddings are from fp32 ones on a slice of the corpus
        ctx.module.eval()
        try:
            texts = list(targets.passage_texts[:SLICE])
            with torch.no_grad():
                half = validation_eval._embed(ctx.encoder, texts, cfg.training.passage_max_length,
                                              cfg.validation.batch_size, ctx.device, True)
                full = validation_eval._embed(ctx.encoder, texts, cfg.training.passage_max_length,
                                              cfg.validation.batch_size, ctx.device, False)
            data["fp16_fp32_min_cosine"] = round(float((half * full).sum(dim=-1).min()), 6)
        finally:
            ctx.module.train()
    (ctx.out / "validation_step0.json").write_text(json.dumps(first, indent=1, sort_keys=True) + "\n")
    ctx.shared["validation"] = first
    problems = [k for k in ("identical_on_repeat", "weights_unchanged", "training_mode_restored",
                            "random_state_unchanged") if not data[k]]
    if problems:
        return Check("validation_full_corpus", "fail", f"the validation pass is not safe: {', '.join(problems)}", data)
    status = "warn" if data.get("fp16_fp32_min_cosine", 1.0) < 0.999 else "pass"
    return Check("validation_full_corpus", status, f"{data['questions']} questions over {data['passages_ranked']} "
                 f"passages in {seconds[0]} s and {seconds[1]} s, identical on repeat; no weight, mode or random state "
                 "changed (the value is written to validation_step0.json, not shown)", data)


def check_checkpoint(ctx: Context) -> Check:
    cfg = ctx.config.training
    evaluated = ctx.shared.get("validation")
    if evaluated is None:
        return Check("checkpoint_resume", "skipped", "no step-0 validation result to record (check 8 did not run)")
    inputs = ctx.inputs.for_seed(ctx.config.seed_list[0])
    schedule = ctx.schedules[ctx.config.seed_list[0]]
    total = len(schedule.steps)
    run_dir = ctx.out / "run"
    run_dir.mkdir(parents=True, exist_ok=False)

    def objects(encoder: Encoder) -> tuple[Any, Any, Any]:
        optimizer = torch.optim.AdamW(_parameter_groups(encoder, cfg.weight_decay), lr=cfg.learning_rate)  # type: ignore[arg-type]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, schedule.warmup_steps))
        return optimizer, scheduler, torch.amp.GradScaler(ctx.device.type, enabled=ctx.fp16)

    optimizer, scheduler, scaler = objects(ctx.encoder)
    lengths = check_lengths(inputs, ctx.encoder)
    selection = SelectionState()
    entry = selection.record(0, evaluated)
    started = time.perf_counter()
    path = _save_checkpoint(run_dir, 0, total, ctx.encoder, optimizer, scheduler, scaler, inputs, schedule, [],
                            ctx.device, lengths, ctx.encoder.describe(), evaluated=entry, selection=selection)
    data: dict[str, Any] = {"checkpoint": str(path), "save_seconds": round(time.perf_counter() - started, 2)}
    files = checkpoint_files(path)
    data["hashes_match"] = files == entry["files"] and files_digest(files) == entry["digest"]
    data["size_mib"] = round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**2, 1)
    data["trainer_state_mib"] = round((path / STATE_FILE).stat().st_size / 1024**2, 1)
    state = torch.load(path / STATE_FILE, map_location="cpu", weights_only=True)
    texts = _texts(ctx, [p.pair_id for p in ctx.inputs.pairs[:TEXTS]])[1]
    ctx.module.train()
    _set_rng_state(state["rng"])
    first = _forward(ctx, texts, False)
    _set_rng_state(state["rng"])
    again = _forward(ctx, texts, False)
    data["random_state_replays"] = bool(torch.equal(first, again))
    fresh = ctx.loader(ctx.config.base_model)
    fresh_module: torch.nn.Module = fresh  # type: ignore[assignment]
    fresh_module.to(ctx.device)
    try:
        fresh_objects = objects(fresh)
        done, history, restored = restore_checkpoint(path, fresh, *fresh_objects, schedule, validating=True)
        ours, theirs = ctx.module.state_dict(), fresh_module.state_dict()
        data["weights_identical"] = ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)
        data["optimizer_state_restored"] = json.dumps(optimizer.state_dict()["param_groups"], sort_keys=True,
                                                      default=str) == json.dumps(
            fresh_objects[0].state_dict()["param_groups"], sort_keys=True, default=str) and not fresh_objects[0].state
        data["scheduler_state_restored"] = scheduler.state_dict() == fresh_objects[1].state_dict()
        data["scaler_state_restored"] = scaler.state_dict() == fresh_objects[2].state_dict()
        data["selection_state_restored"] = restored is not None and restored.to_json() == selection.to_json()
        data["step_and_history"] = [done, history]
        if isinstance(ctx.encoder, HFEncoder):
            loaded = load_checkpoint_encoder(inspect_checkpoint(path), device=str(ctx.device))
            ctx.module.eval()
            with torch.no_grad():
                ours_vec = F.normalize(ctx.encoder.encode(texts, cfg.passage_max_length).float(), dim=-1).cpu()
            theirs_vec = F.normalize(torch.as_tensor(loaded.encode(texts, batch_size=TEXTS)), dim=-1)
            ctx.module.train()
            data["sentence_transformers_max_abs_difference"] = float((ours_vec - theirs_vec).abs().max())
            del loaded
        else:
            data["sentence_transformers_max_abs_difference"] = None  # not a Hugging Face checkpoint (tests)
    finally:
        del fresh, fresh_module
        if ctx.cuda:
            torch.cuda.empty_cache()
    try:
        _save_checkpoint(run_dir, 0, total, ctx.encoder, optimizer, scheduler, scaler, inputs, schedule, [], ctx.device,
                         lengths, ctx.encoder.describe(), evaluated=None, selection=selection)
        data["overwrite_refused"] = False
    except TrainerError:
        data["overwrite_refused"] = True
    data["no_partial_write"] = not any(p.name.startswith(".") for p in run_dir.iterdir())
    st = data["sentence_transformers_max_abs_difference"]
    flags = ("hashes_match", "random_state_replays", "weights_identical", "optimizer_state_restored",
             "scheduler_state_restored", "scaler_state_restored", "selection_state_restored", "overwrite_refused",
             "no_partial_write")
    problems = [k for k in flags if not data[k]] + ([] if done == 0 and history == [] else ["step_and_history"])
    if st is not None and st > 1e-4:
        problems.append("sentence_transformers_reload")
    if problems:
        return Check("checkpoint_resume", "fail", f"checkpoint save, reload or resume failed: {', '.join(problems)}",
                     data)
    return Check("checkpoint_resume", "pass", f"checkpoint-step-000000 ({data['size_mib']} MiB, trainer state "
                 f"{data['trainer_state_mib']} MiB) re-hashes; restore_checkpoint reloads identical weights and the "
                 "optimizer, scheduler, scaler, random and selection state into a fresh base model"
                 + ("" if st is None else f"; the benchmark's loader agrees ({st:.1e})") + "; a second save is refused",
                 data)


CHECKS = (  # checks 2-9 (1 runs before, 10 after), by name and function; looked up when run
    ("dropout", "check_dropout"),
    ("attention_dropout_replay", "check_attention_dropout"),
    ("gradcache_fp32", "check_gradcache_fp32"),
    ("gradcache_fp16", "check_gradcache_fp16"),
    ("fp16_loss_scale", "check_loss_scale"),
    ("peak_memory", "check_peak_memory"),
    ("validation_full_corpus", "check_validation"),
    ("checkpoint_resume", "check_checkpoint"),
)


def _guard(name: str, run: Callable[[Context], Check], ctx: Context) -> Check:
    if ctx.stopped is not None:
        return Check(name, "skipped", f"not run: {ctx.stopped}")
    started = time.perf_counter()
    try:
        check = run(ctx)
    except Exception as exc:  # a check that raises fails; the rehearsal goes on to report
        check = Check(name, "fail", f"{type(exc).__name__}: {exc}")
    finally:
        ctx.module.zero_grad(set_to_none=True)
    check.data["seconds"] = round(time.perf_counter() - started, 2)
    ctx.say(f"{check.status.upper():7s} {check.name}: {check.detail}")
    return check


def run_rehearsal(config_path: str, *, out_dir: str | Path = DEFAULT_OUT,
                  encoder_loader: Callable[[BaseModelConfig], Encoder] | None = None, allow_cpu: bool = False,
                  local_files_only: bool = False, say: Callable[[str], None] = print) -> dict[str, Any]:
    """Every check; the report (also written to ``<out_dir>/rehearsal.json``). See the module docstring."""
    config, config_sha256 = load_trainer_config(config_path)
    if config.validation is None or config.training.mini_batch_size is None:
        raise RehearsalError(f"{config_path}: the rehearsal is for validated GradCache configs (validation and "
                             "mini_batch_size set)")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise RehearsalError(f"{out} is not empty; every rehearsal writes to a new directory")
    cuda = torch.cuda.is_available()
    if not cuda and not allow_cpu:
        raise RehearsalError("no CUDA device: enable the GPU (a CPU run with --allow-cpu checks the code only)")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cuda else "cpu")
    inputs = prepare_inputs(config.for_seed(config.seed_list[0]), config_sha256)
    schedules = {seed: plan_steps(inputs.for_seed(seed)) for seed in config.seed_list}
    loader = encoder_loader or (lambda model: load_base_encoder(model, local_files_only=local_files_only,
                                                                gradient_cache=True))
    started = time.perf_counter()
    encoder = loader(config.base_model)
    module: torch.nn.Module = encoder  # type: ignore[assignment]
    module.to(device)
    load_seconds = round(time.perf_counter() - started, 1)
    ctx = Context(config, config_path, inputs, schedules, encoder, loader, device, config.training.fp16 and cuda, out,
                  say)
    if ctx.fp16:
        ctx.init_scale = float(torch.amp.GradScaler("cuda").get_scale())
    at_load = weights_digest(module)
    checks = [check_environment(ctx)]
    say(f"{checks[0].status.upper():7s} environment: {checks[0].detail}")
    with no_optimizer_steps() as attempts:
        for name, function in CHECKS:
            checks.append(_guard(name, globals()[function], ctx))
    final = weights_digest(module)
    no_training = Check("no_training", "pass" if not attempts and final == at_load else "fail",
                        f"optimizer steps attempted: {len(attempts)}; final weights "
                        f"{'equal' if final == at_load else 'DIFFER FROM'} the weights at load",
                        {"optimizer_steps_attempted": attempts, "weights_at_load": at_load, "weights_at_end": final})
    checks.append(no_training)
    say(f"{no_training.status.upper():7s} no_training: {no_training.detail}")
    failed = [c.name for c in checks if c.status == "fail"]
    skipped = [c.name for c in checks if c.status == "skipped"]
    report = {
        "kind": KIND, "format_version": FORMAT_VERSION, "notice": NOTICE,
        "config": {"path": config_path, "sha256": config_sha256, "name": config.name},
        "environment": ctx.shared.get("environment"),
        "base_model": config.base_model.model_dump(), "model_load_seconds": load_seconds,
        "schedules": {str(seed): {"steps": len(s.steps), "warmup_steps": s.warmup_steps, "batching": s.batching,
                                  "batches_per_epoch": list(s.batches_per_epoch), "dropped": s.dropped_batches,
                                  "in_batch_negatives_per_query": s.negatives, "sha256": s.sha256}
                      for seed, s in schedules.items()},
        "fp16_initial_scale": ctx.init_scale if ctx.fp16 else None,
        "stopped": ctx.stopped,
        "checks": [asdict(c) for c in checks],
        "failed": failed, "skipped": skipped,
        "ready": not failed and not skipped,
        "trained": False,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out / "rehearsal.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu_rehearsal.py", description="The GPU rehearsal of a validated run: ten "
                                     "checks, no optimizer step (see finretriever.training.gpu_rehearsal).")
    parser.add_argument("--config", required=True, help="The experiment's training config.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="A new directory for the report and the check outputs.")
    parser.add_argument("--allow-cpu", action="store_true", help="Run without CUDA (checks the code, not the GPU).")
    parser.add_argument("--local-files-only", action="store_true", help="Load the base model from the local cache.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify the config, inputs and schedules and list the checks; load no model.")
    return parser


def main(argv: list[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None,
         encoder_loader: Callable[[BaseModelConfig], Encoder] | None = None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    args = build_parser().parse_args(argv)

    def say(text: str) -> None:
        print(f"{PREFIX} {text}", file=out)

    try:
        if args.dry_run:
            config, sha = load_trainer_config(args.config)
            inputs = prepare_inputs(config.for_seed(config.seed_list[0]), sha)
            for seed in config.seed_list:
                s = plan_steps(inputs.for_seed(seed))
                say(f"seed {seed}: {len(s.steps)} steps, {s.batching} batches {list(s.batches_per_epoch)}, first batch "
                    f"{len(s.steps[0].batches[0])} pairs, {s.dropped_batches} dropped")
            say(f"checks: environment, {', '.join(name for name, _ in CHECKS)}, no_training; "
                f"{len(inputs.pairs)} pairs, {len(inputs.validation)} validation questions, "
                f"{len(inputs.corpus or {})} passages; dry run: no model loaded, nothing written")
            return 0
        report = run_rehearsal(args.config, out_dir=args.out, encoder_loader=encoder_loader, allow_cpu=args.allow_cpu,
                               local_files_only=args.local_files_only, say=say)
    except (RehearsalError, ValueError) as exc:
        print(f"{PREFIX} refused: {exc}", file=err)
        return 1
    if report["failed"]:
        say(f"REHEARSAL FAILED: {', '.join(report['failed'])}" + (f" (stopped: {report['stopped']})"
                                                                  if report["stopped"] else ""))
        code = 1
    elif report["skipped"]:
        say(f"NOT A GPU REHEARSAL: skipped {', '.join(report['skipped'])}")
        code = 3
    else:
        say("REHEARSAL PASSED: every check passed; no optimizer step was taken")
        code = 0
    say(f"report: {Path(args.out) / 'rehearsal.json'}; {NOTICE}")
    return code


__all__ = ["CHECKS", "DEFAULT_OUT", "KIND", "Check", "Context", "RehearsalError", "dropout_disabled",
           "dropout_mask_agreement", "main", "no_optimizer_steps", "precision_comparison", "run_rehearsal",
           "weights_digest"]
