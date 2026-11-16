"""Contrastive fine-tuning of a bi-encoder (BGE-base by default) on the training candidates.

**Objective.** In-batch InfoNCE. A batch holds B pairs (query q_i, positive passage p_i), planned
so that every other positive is an allowed negative for each query (``trainer_setup``). With
L2-normalised embeddings and temperature t::

    loss = mean_i  -log( exp(q_i . p_i / t) / sum_j exp(q_i . p_j / t) )

which is sentence-transformers' MultipleNegativesRankingLoss with scale 1/t (t = 0.02: scale 50,
BGE's fine-tuning setting). Each query has B - 1 negatives, all from its own batch. Gradient
accumulation averages the gradients of several batches per optimizer step; it does not add
negatives. Queries carry the model's query instruction and passages its passage prefix (none
for BGE), as evaluation encodes them; passages are truncated to the preparation's visible
length, and a query that does not fit ``query_max_length`` is an error. Before the first step
``check_lengths`` measures every query and positive passage and checks that the model's
truncation keeps each pair's visible text, where its evidence lies; no batch wider than its limit
reaches the model.

**GradCache** (``mini_batch_size``; Gao et al., 2021, "Scaling Deep Contrastive Learning Batch Size
under Memory Limited Setup"). The loss is the full batch's, computed in three passes
(``cached_contrastive_backward``): every mini-batch is encoded without gradients after its random
state is saved; the loss is back-propagated to those cached embeddings (scaled by the fp16
GradScaler exactly as the plain loss is); then each mini-batch is encoded again from its saved
random state, so dropout draws the same masks, and its embeddings' gradients are back-propagated
into the model. The parameter gradients equal those of the same forward passes without caching,
while only one mini-batch of activations is held at a time. Every replayed mini-batch is compared
with its cached embeddings, and a step whose replay differs is refused (``TrainerError``), never
trained on. The base model is loaded with eager attention for GradCache runs, so attention dropout
is drawn by the same random generator in both passes.

**False negatives.** Before every forward pass the batch is checked again
(``trainer_setup.check_batch``): no in-batch passage is a known exact or uncertain equivalent,
overlap sibling, reuse link, blocked or other-split passage of another query.

**Checkpoints.** ``<output_dir>/<name>-seed<seed>/checkpoint-step-NNNNNN/`` at every
``save_every_steps`` and at the last step: the model and tokenizer (Hugging Face format, plus the
sentence-transformers module files with the model's pooling and normalisation, so evaluation can
load it), ``metadata.json`` (what it was trained on, how, and the loss so far) and, in the newest
checkpoint only, ``trainer_state.pt`` (optimizer, scheduler, scaler and random states, and the
selection state) for resuming. ``run.json`` binds the directory to one configuration, inputs and
schedule: a different run is refused, never written over, and so is an existing checkpoint. A
resumed run continues the same schedule and gives the same result on CPU.

**Validation and selection** (``validation`` in the config). The validation set is evaluated
(``validation_eval``) at step 0, before any update, and at the steps of the fixed rule
(``checkpoint_selection``: every 5 steps, whole-corpus MRR@10, ties to the earlier step; selection
stops after 3 evaluations without improvement); a checkpoint is written at each evaluated step,
step 0 included (the base weights), and hashed. Early stopping decides which checkpoints are
selectable, not whether training goes on: after the rule's stop the run trains to its planned last
step and evaluates every remaining evaluation step, recorded as never selectable (the decision rule's
selection-aware test needs every planned evaluation). When the run reaches its last step,
``selection.json`` records every evaluation with its checkpoint's hashes, the selected checkpoint and
the hardware and software environment, once.

**Overflow.** A validated run stops at the first step whose gradients are not finite (an fp16
overflow), before the optimizer step: the GradScaler is never left to skip the step and lower its
scale on its own. Each step logs the loss scale it used.

**No evaluation data.** The trainer reads no evaluation file and computes no development or
evaluation metric. Without validation, checkpoints are kept at fixed steps and nothing chooses
among them; with it, only the synthetic validation set does. Nothing it writes claims a retrieval
improvement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import logging
import math
import platform
import random
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol, TextIO

import torch
import torch.nn.functional as F

from finretriever.training import validation_eval
from finretriever.training.checkpoint_selection import (
    RULE,
    SELECTION_FILE,
    SelectionState,
    checkpoint_files,
    evaluation_steps,
    files_digest,
    make_selection_record,
    write_selection_record,
)
from finretriever.training.trainer_setup import (
    BaseModelConfig,
    Schedule,
    TrainerSetupError,
    TrainingInputs,
    check_batch,
    checkpoint_name,
    config_record,
    estimate_gpu_memory_gib,
    load_trainer_config,
    lr_factor,
    plan_steps,
    prepare_inputs,
)

logger = logging.getLogger(__name__)
PREFIX = "[05_train_biencoder]"
CHECKPOINT_FORMAT = 1
RUN_FILE = "run.json"
LOG_FILE = "log.jsonl"
VALIDATION_LOG_FILE = "validation_log.jsonl"
STATE_FILE = "trainer_state.pt"
METADATA_FILE = "metadata.json"
CHECKPOINT_NOTICE = (
    "Fine-tuned on synthetic queries accepted by an AI quality review (not human-validated, not gold labels). "
    "Not evaluated by the trainer and not selected by any development metric; no retrieval improvement is claimed."
)
VALIDATED_CHECKPOINT_NOTICE = (
    "Fine-tuned on synthetic queries accepted by an AI quality review (not human-validated, not gold labels). "
    "Evaluated by the trainer on the synthetic validation set only to choose among this run's checkpoints; no "
    "development or evaluation metric was computed, and no retrieval improvement is claimed."
)
# How far a replayed mini-batch may differ from its cached pass (largest absolute difference relative to the largest
# absolute value): the same masks and inputs give the same result up to kernel rounding; another dropout mask
# changes it by far more.
REPLAY_TOLERANCE = {"fp32": 1e-5, "fp16": 1e-2}
MEASURED_T4 = "measured on a T4 in the smoke run: 5.0 GiB peak at 16 pairs per forward pass, 0.43 s per step"


class TrainerError(RuntimeError):
    """Training cannot start or continue safely; nothing is written over."""


class Encoder(Protocol):
    """What the trainer needs from a model (``HFEncoder``; tests use small fakes)."""

    def encode(self, texts: Sequence[str], max_length: int) -> torch.Tensor:
        """Pooled embeddings, one row per text (not normalised), with gradients."""
        ...

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        """Tokens per text, special tokens included, before truncation (a measurement: nothing is run)."""
        ...

    def kept_chars(self, texts: Sequence[str], max_length: int) -> list[int]:
        """Characters of each text that ``encode`` at ``max_length`` reads (where its last kept token ends)."""
        ...

    def save(self, directory: Path, *, max_seq_length: int) -> None: ...

    def load_weights(self, directory: Path) -> None: ...

    def enable_gradient_checkpointing(self) -> None: ...

    def describe(self) -> dict[str, Any]:
        """Model details, with ``tokenizer_sha256`` when the tokenizer can be serialised."""
        ...


def contrastive_loss(queries: torch.Tensor, passages: torch.Tensor, temperature: float) -> torch.Tensor:
    """In-batch InfoNCE: query i's positive is passage i, every other passage a negative."""
    if queries.shape != passages.shape or queries.shape[0] < 2:
        raise TrainerError("a batch needs at least two (query, passage) pairs of equal shape")
    q = F.normalize(queries.float(), dim=-1)
    p = F.normalize(passages.float(), dim=-1)
    scores = q @ p.T / temperature
    return F.cross_entropy(scores, torch.arange(q.shape[0], device=q.device))


# --- GradCache ----------------------------------------------------------------------------------


class RandomState:
    """The CPU random state and, on CUDA, the device's: restoring it replays a forward pass with the same dropout."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    def restore(self) -> None:
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda, self.device)


def _pieces(
    queries: Sequence[str], passages: Sequence[str], mini_batch_size: int, query_max_length: int,
    passage_max_length: int,
) -> list[tuple[list[str], int]]:
    """Queries, then passages, in mini-batches of at most ``mini_batch_size``, each with its token limit."""
    return [
        (list(texts[i : i + mini_batch_size]), limit)
        for texts, limit in ((queries, query_max_length), (passages, passage_max_length))
        for i in range(0, len(texts), mini_batch_size)
    ]


def cached_contrastive_backward(
    encoder: Encoder,
    queries: Sequence[str],
    passages: Sequence[str],
    *,
    temperature: float,
    mini_batch_size: int,
    query_max_length: int,
    passage_max_length: int,
    device: torch.device,
    fp16: bool = False,
    scale: Callable[[torch.Tensor], torch.Tensor] = lambda loss: loss,
    divisor: int = 1,
) -> tuple[float, float]:
    """GradCache: accumulate the gradients of the full batch's in-batch loss, one mini-batch of activations at a time.

    Returns the loss (divided by ``divisor``, unscaled) and the largest relative difference between a replayed
    mini-batch and its cached pass. ``scale`` is the GradScaler's (identity without fp16); the gradients it scales
    are unscaled by the optimizer step as in plain training. See the module docstring.
    """
    pieces = _pieces(queries, passages, mini_batch_size, query_max_length, passage_max_length)
    states, cached = [], []
    with torch.no_grad():
        for texts, limit in pieces:
            states.append(RandomState(device))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                cached.append(encoder.encode(texts, limit).float())
    after = RandomState(device)
    n_queries = -(-len(queries) // mini_batch_size)  # query mini-batches come first
    q = torch.cat(cached[:n_queries]).requires_grad_()
    p = torch.cat(cached[n_queries:]).requires_grad_()
    loss = contrastive_loss(q, p, temperature) / divisor
    scale(loss).backward()
    assert q.grad is not None and p.grad is not None
    grads = [*q.grad.split([len(t) for t, _ in pieces[:n_queries]]), *p.grad.split([len(t) for t, _ in pieces[n_queries:]])]
    tolerance = REPLAY_TOLERANCE["fp16" if fp16 else "fp32"]
    worst = 0.0
    for (texts, limit), state, before, grad in zip(pieces, states, cached, grads, strict=True):
        state.restore()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            embeddings = encoder.encode(texts, limit).float()
        difference = float((embeddings.detach() - before).abs().max() / before.abs().max().clamp_min(1e-12))
        worst = max(worst, difference)
        if difference > tolerance:
            after.restore()
            raise TrainerError(
                f"GradCache: a replayed mini-batch differs from its cached pass (relative {difference:.2e} > "
                f"{tolerance:.0e}); dropout is not replayed exactly on this model or device, so no step is taken"
            )
        (embeddings * grad).sum().backward()
    after.restore()
    return float(loss.detach()), worst


def chunked_contrastive_backward(
    encoder: Encoder,
    queries: Sequence[str],
    passages: Sequence[str],
    *,
    temperature: float,
    mini_batch_size: int,
    query_max_length: int,
    passage_max_length: int,
    device: torch.device,
    fp16: bool = False,
    scale: Callable[[torch.Tensor], torch.Tensor] = lambda loss: loss,
    divisor: int = 1,
) -> float:
    """The reference GradCache must match: the same mini-batch forward passes, with gradients, without caching."""
    outputs = []
    for texts, limit in _pieces(queries, passages, mini_batch_size, query_max_length, passage_max_length):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            outputs.append(encoder.encode(texts, limit).float())
    n_queries = -(-len(queries) // mini_batch_size)
    loss = contrastive_loss(torch.cat(outputs[:n_queries]), torch.cat(outputs[n_queries:]), temperature) / divisor
    scale(loss).backward()
    return float(loss.detach())


def gradcache_check(
    encoder: Encoder,
    queries: Sequence[str],
    passages: Sequence[str],
    *,
    temperature: float,
    mini_batch_size: int,
    query_max_length: int,
    passage_max_length: int,
    device: torch.device,
    fp16: bool = False,
    loss_scale: float | None = None,
) -> dict[str, float]:
    """GradCache against the reference from the same random state; nothing is updated and gradients are cleared.

    The loss is multiplied by ``loss_scale`` as the GradScaler would (default: a fixed 1024 with fp16, 1 without), so
    small gradients do not underflow; ``finite`` says whether every scaled gradient was finite (an fp16 overflow at
    that scale if not).
    """
    module: torch.nn.Module = encoder  # type: ignore[assignment]
    params = [p for p in module.parameters() if p.requires_grad]
    factor = loss_scale if loss_scale is not None else (1024.0 if fp16 else 1.0)
    settings = {"temperature": temperature, "mini_batch_size": mini_batch_size, "query_max_length": query_max_length,
                "passage_max_length": passage_max_length, "device": device, "fp16": fp16,
                "scale": lambda loss: loss * factor}

    def gradients() -> torch.Tensor:
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
                          for p in params]) / factor
        for p in params:
            p.grad = None
        return flat

    was_training = module.training
    module.train()
    for p in params:
        p.grad = None
    start = RandomState(device)
    loss_cached, replay = cached_contrastive_backward(encoder, queries, passages, **settings)  # type: ignore[arg-type]
    cached = gradients()
    start.restore()
    loss_reference = chunked_contrastive_backward(encoder, queries, passages, **settings)  # type: ignore[arg-type]
    reference = gradients()
    module.train(was_training)

    def dot(a: torch.Tensor, b: torch.Tensor) -> float:  # float64, in chunks: a float32 sum over 1e8 values drifts
        return sum(float((x.double() * y.double()).sum()) for x, y in zip(a.split(1 << 22), b.split(1 << 22),
                                                                          strict=True))

    difference = cached - reference
    norm_cached, norm_reference = dot(cached, cached) ** 0.5, dot(reference, reference) ** 0.5
    return {
        "loss_cached": loss_cached,
        "loss_reference": loss_reference,
        "relative_difference": dot(difference, difference) ** 0.5 / max(norm_reference, 1e-12),
        "cosine": dot(cached, reference) / max(norm_cached * norm_reference, 1e-24),
        "replay_relative_difference": replay,
        "parameters": float(sum(p.numel() for p in params)),
        "loss_scale": factor,
        "finite": bool(torch.isfinite(cached).all() and torch.isfinite(reference).all()),
    }


# --- the Hugging Face encoder -------------------------------------------------------------------


def _tokenizer_sha256(tokenizer: Any) -> str | None:
    """As ``visible_text.tokenizer_identity`` hashes it: the serialised fast tokenizer, untruncated."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    backend = copy.deepcopy(backend)
    backend.no_truncation()
    backend.no_padding()
    return hashlib.sha256(backend.to_str().encode("utf-8")).hexdigest()


def write_sentence_transformers_files(
    directory: Path, *, dimension: int, pooling: str, max_seq_length: int
) -> None:
    """Module files so ``SentenceTransformer(directory)`` pools and normalises as training did.

    Inputs are not lower-cased by these files, as in training (the tokenizer applies its own
    normalisation); prompts are empty, since evaluation applies the prefixes itself.
    """
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
    ]
    pooling_config = {
        "word_embedding_dimension": dimension,
        "pooling_mode_cls_token": pooling == "cls",
        "pooling_mode_mean_tokens": pooling == "mean",
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    }
    files = {
        "modules.json": modules,
        "sentence_bert_config.json": {"max_seq_length": max_seq_length, "do_lower_case": False},
        "config_sentence_transformers.json": {
            "prompts": {},
            "default_prompt_name": None,
            "similarity_fn_name": "cosine",
        },
        "1_Pooling/config.json": pooling_config,
    }
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    (directory / "2_Normalize").mkdir(exist_ok=True)


_SAFETENSORS_DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.float64: "F64",
                       torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
                       torch.bool: "BOOL"}


class HFEncoder(torch.nn.Module):
    """A Hugging Face transformer and its tokenizer, pooled by the first token (CLS) or the mean."""

    def __init__(self, model: Any, tokenizer: Any, pooling: str) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling

    @classmethod
    def load(
        cls,
        name_or_path: str,
        *,
        revision: str | None,
        pooling: str,
        local_files_only: bool = False,
        attn_implementation: str | None = None,
    ) -> HFEncoder:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise TrainerError(f"loading {name_or_path} needs transformers ({exc})") from None
        extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                name_or_path, revision=revision, local_files_only=local_files_only
            )
            model = AutoModel.from_pretrained(
                name_or_path, revision=revision, local_files_only=local_files_only, **extra
            )
        except (OSError, ValueError) as exc:
            raise TrainerError(f"could not load {name_or_path} at {revision}: {exc}") from None
        return cls(model, tokenizer, pooling)

    def encode(self, texts: Sequence[str], max_length: int) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        if batch["input_ids"].shape[1] > max_length:  # never run through the model
            raise TrainerError(
                f"the tokenizer returned {batch['input_ids'].shape[1]} tokens for a limit of {max_length}"
            )
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden = self.model(**batch).last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        # verbose=False: measuring a long text is not running it through the model, so no "indexing errors"
        # warning (as retrieval.biencoder counts them)
        encoded = self.tokenizer(list(texts), add_special_tokens=True, truncation=False, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def kept_chars(self, texts: Sequence[str], max_length: int) -> list[int]:
        encoded = self.tokenizer(
            list(texts), truncation=True, max_length=max_length, return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        return [
            max((end for (_, end), special in zip(offsets, mask, strict=True) if not special), default=0)
            for offsets, mask in zip(encoded["offset_mapping"], encoded["special_tokens_mask"], strict=True)
        ]

    def save(self, directory: Path, *, max_seq_length: int) -> None:
        """Write the model, tokenizer and sentence-transformers files. The weights keep the model's own parameter
        names, the names the base model's hub file uses: transformers 5's ``save_pretrained`` otherwise reverses its
        "legacy" renaming on every save (``LayerNorm.weight``/``bias`` written as ``gamma``/``beta``) for a model
        loaded with ``from_pretrained``, and the strict reload of a resumed run then fails. The written file is
        checked against the model, name by name, shape and dtype."""
        save = self.model.save_pretrained
        if "save_original_format" in inspect.signature(save).parameters:  # transformers >= 5
            save(directory, save_original_format=False)
        else:  # earlier versions write the model's own names
            save(directory)
        self.tokenizer.save_pretrained(directory)
        write_sentence_transformers_files(
            directory,
            dimension=int(self.model.config.hidden_size),
            pooling=self.pooling,
            max_seq_length=max_seq_length,
        )
        if problems := self._file_problems(directory / "model.safetensors"):
            raise TrainerError(f"{directory / 'model.safetensors'} does not hold the model's parameters as named: "
                               f"{problems}")

    def _file_problems(self, path: Path) -> list[str]:
        """How a weights file differs from the model's state dict: names, shapes and dtypes (values not read)."""
        from safetensors import safe_open

        expected = {k: (tuple(v.shape), _SAFETENSORS_DTYPES.get(v.dtype, str(v.dtype)))
                    for k, v in self.model.state_dict().items()}
        with safe_open(str(path), "pt") as fh:
            found = {}
            for k in fh.keys():
                tensor = fh.get_slice(k)
                found[k] = (tuple(tensor.get_shape()), tensor.get_dtype())
        problems = []
        if missing := sorted(set(expected) - set(found)):
            problems.append(f"missing {missing[:4]}{' …' if len(missing) > 4 else ''} ({len(missing)})")
        if unexpected := sorted(set(found) - set(expected)):
            problems.append(f"unexpected {unexpected[:4]}{' …' if len(unexpected) > 4 else ''} ({len(unexpected)})")
        shapes = [k for k in set(expected) & set(found) if expected[k][0] != found[k][0]]
        if shapes:
            problems.append(f"shapes differ for {sorted(shapes)[:4]}")
        dtypes = [k for k in set(expected) & set(found) if expected[k][1] != found[k][1]]
        if dtypes:
            problems.append(f"dtypes differ for {sorted(dtypes)[:4]}")
        return problems

    def load_weights(self, directory: Path) -> None:
        """Load a checkpoint's weights strictly: every parameter by its own name, nothing missing, renamed or extra.
        A file written in transformers' legacy names (``LayerNorm.gamma``/``beta``, by code before this check) is
        refused, not renamed."""
        from safetensors.torch import load_file

        path = directory / "model.safetensors"
        if not path.is_file():
            raise TrainerError(f"{path} is missing; the checkpoint cannot be resumed")
        if problems := self._file_problems(path):
            legacy = " (transformers' legacy LayerNorm names gamma/beta)" if any(
                "gamma" in p or "beta" in p for p in problems) else ""
            raise TrainerError(f"{path} does not match the model's parameters{legacy}: {problems}; it cannot be "
                               "resumed")
        self.model.load_state_dict(load_file(str(path)), strict=True)

    def enable_gradient_checkpointing(self) -> None:
        self.model.gradient_checkpointing_enable()

    def describe(self) -> dict[str, Any]:
        described = {
            "class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "hidden_size": int(self.model.config.hidden_size),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "pooling": self.pooling,
            "tokenizer_sha256": _tokenizer_sha256(self.tokenizer),
        }
        attention = getattr(self.model.config, "_attn_implementation", None)
        if attention == "eager":  # recorded when chosen for GradCache; other runs record what they always did
            described["attention"] = attention
        return described


def load_base_encoder(
    model: BaseModelConfig, *, local_files_only: bool = False, gradient_cache: bool = False
) -> HFEncoder:
    """The pinned base model (downloads from Hugging Face unless ``local_files_only``). With ``gradient_cache`` it uses
    eager attention, whose dropout draws from the generator GradCache replays."""
    return HFEncoder.load(
        model.name, revision=model.revision, pooling=model.pooling, local_files_only=local_files_only,
        attn_implementation="eager" if gradient_cache else None,
    )


# --- training -----------------------------------------------------------------------------------


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise TrainerError(f"device {device} requested but CUDA is not available")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "safetensors"):
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_environment(device: torch.device) -> dict[str, Any]:
    """The hardware and software a run used, as its selection record keeps it: library versions, CUDA and cuDNN, and
    the GPU's name, compute capability and memory (``None`` on CPU). ``checkpoint_selection.ENVIRONMENT_IDENTITY`` names the
    entries every seed of one experiment must share."""
    cuda = device.type == "cuda"
    props = torch.cuda.get_device_properties(device) if cuda else None
    return {
        **_versions(),
        "platform": platform.platform(),
        "device_type": device.type,
        "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu_name": props.name if props is not None else None,
        "gpu_capability": f"{props.major}.{props.minor}" if props is not None else None,
        "gpu_memory_gib": round(props.total_memory / 1024**3, 2) if props is not None else None,
    }


@dataclass
class TrainResult:
    run_dir: Path
    steps_done: int
    total_steps: int
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[Path] = field(default_factory=list)
    selection: Path | None = None  # selection.json, once a run with validation has ended


def _run_record(inputs: TrainingInputs, schedule: Schedule) -> dict[str, Any]:
    cfg = inputs.config
    record = {
        "kind": "finretriever_biencoder_run",
        "format_version": CHECKPOINT_FORMAT,
        "run": cfg.run_name,
        "config": config_record(cfg),
        "config_sha256": inputs.config_sha256,
        "inputs": inputs.identity,
        "schedule": {
            "steps": len(schedule.steps),
            "warmup_steps": schedule.warmup_steps,
            "batches_per_epoch": list(schedule.batches_per_epoch),
            "dropped_single_pair_batches": schedule.dropped_batches,
            "sha256": schedule.sha256,
        },
    }
    if cfg.training.batching is not None:  # recorded only when set, so earlier configs record what they did
        record["schedule"]["batching"] = schedule.batching
        record["schedule"]["in_batch_negatives_per_query"] = schedule.negatives
    if cfg.validation is not None:
        record["selection_rule"] = {**RULE, "evaluated_steps": evaluation_steps(len(schedule.steps))}
    return record


def restore_checkpoint(
    latest: Path,
    encoder: Encoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    schedule: Schedule,
    *,
    validating: bool,
) -> tuple[int, list[dict[str, Any]], SelectionState | None]:
    """Load a resumable checkpoint into the run's objects: the weights, then the optimizer, scheduler, scaler and random
    states; returns the steps done, the loss history and (with validation) the selection state. Refuses a checkpoint
    of another schedule or format."""
    encoder.load_weights(latest)
    state = torch.load(latest / STATE_FILE, map_location="cpu", weights_only=True)
    if state["schedule_sha256"] != schedule.sha256 or state["format_version"] != CHECKPOINT_FORMAT:
        raise TrainerError(f"{latest} belongs to another schedule; it cannot be resumed")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    _set_rng_state(state["rng"])
    selection = None
    if validating:
        if "selection" not in state:
            raise TrainerError(f"{latest} holds no selection state; it cannot resume a run with validation")
        selection = SelectionState.from_json(json.loads(state["selection"]))
    return int(state["step"]), list(json.loads(state["history"])), selection


def _latest_checkpoint(run_dir: Path) -> Path | None:
    done = sorted(p for p in run_dir.glob("checkpoint-step-*") if (p / STATE_FILE).is_file())
    return done[-1] if done else None


def check_lengths(inputs: TrainingInputs, encoder: Encoder) -> dict[str, int]:
    """Every query and positive passage as ``encoder`` reads them.

    A query longer than ``query_max_length`` is an error (queries are never truncated). Passages
    longer than ``passage_max_length`` are truncated, which is expected; but if the truncation ends
    before a pair's visible text does (the span its evidence was found in, measured by the
    preparation), the tokenizer does not truncate as the preparation did and training is refused.
    """
    cfg = inputs.config.training
    query_tokens = encoder.token_counts(list(inputs.queries.values()))
    too_long = [n for n in query_tokens if n > cfg.query_max_length]
    if too_long:
        raise TrainerError(
            f"{len(too_long)} queries exceed query_max_length {cfg.query_max_length} (longest {max(too_long)} "
            "tokens); queries are never truncated"
        )
    texts = list(inputs.passages.values())
    passage_tokens = encoder.token_counts(texts)
    kept = dict(zip(inputs.passages, encoder.kept_chars(texts, cfg.passage_max_length), strict=True))
    prefix = len(inputs.config.base_model.passage_prefix)
    cut = [p.pair_id for p in inputs.pairs if kept[p.positive_passage_id] - prefix < p.source.visible_chars]
    if cut:
        raise TrainerError(
            f"truncation at {cfg.passage_max_length} tokens would cut the visible text, where the evidence lies, "
            f"of {len(cut)} pairs ({', '.join(cut[:3])}); the tokenizer does not truncate as the preparation did"
        )
    return {
        "longest_query_tokens": max(query_tokens),
        "positive_passages": len(passage_tokens),
        "passages_truncated_to_visible_length": sum(n > cfg.passage_max_length for n in passage_tokens),
        "longest_passage_tokens": max(passage_tokens),
    }


def _parameter_groups(encoder: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or "norm" in name.lower() else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train(
    inputs: TrainingInputs,
    encoder: Encoder,
    *,
    resume: bool = False,
    stop_after: int | None = None,
    say: Callable[[str], None] = logger.info,
    evaluate: Callable[[Encoder, int], Mapping[str, Any]] | None = None,
) -> TrainResult:
    """Train ``encoder`` (the base model, freshly loaded) on ``inputs``; see the module docstring.

    ``stop_after`` stops cleanly after that step, saving a checkpoint (an interruption to resume). With validation,
    ``evaluate(encoder, step)`` gives the validation metrics (default: ``validation_eval.evaluate`` on the inputs'
    validation set); tests pass their own.
    """
    cfg = inputs.config.training
    device = resolve_device(cfg.device)
    if cfg.fp16 and device.type != "cuda":
        raise TrainerError("fp16 needs CUDA; set fp16: false to train on CPU")
    schedule = plan_steps(inputs)
    record = _run_record(inputs, schedule)
    run_dir = Path(inputs.config.checkpoints.output_dir) / inputs.config.run_name
    run_file = run_dir / RUN_FILE
    record_bytes = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    latest = None
    if run_file.exists():
        if run_file.read_bytes() != record_bytes:
            raise TrainerError(
                f"{run_dir} holds a different run (configuration, inputs or schedule); use a new name"
            )
        latest = _latest_checkpoint(run_dir)
        if not resume:
            raise TrainerError(f"{run_dir} already exists; pass --resume to continue it")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise TrainerError(f"{run_dir} exists without {RUN_FILE}; refusing to write into it")
    if (run_dir / SELECTION_FILE).exists():  # the run has ended: nothing is trained, loaded or written
        ended = json.loads((run_dir / SELECTION_FILE).read_text(encoding="utf-8"))
        say(f"{inputs.config.run_name} has ended: its selection is recorded in {run_dir / SELECTION_FILE}")
        return TrainResult(run_dir, int(ended["stopped"]["at_step"]), len(schedule.steps),
                           selection=run_dir / SELECTION_FILE)
    validating = inputs.config.validation is not None
    if validating and evaluate is None:
        if inputs.corpus is None or not inputs.validation:
            raise TrainerError("the config names a validation set, but the inputs hold none")
        targets = validation_eval.build_targets(
            inputs.validation, inputs.guard, inputs.corpus, query_prefix=inputs.config.base_model.query_prefix,
            passage_prefix=inputs.config.base_model.passage_prefix,
        )
        batch_size = inputs.config.validation.batch_size  # type: ignore[union-attr]

        def evaluate(model: Encoder, step: int) -> Mapping[str, Any]:
            return validation_eval.evaluate(
                model, targets, query_max_length=cfg.query_max_length, passage_max_length=cfg.passage_max_length,
                batch_size=batch_size, device=device, fp16=cfg.fp16,
            )

    expected_sha = inputs.identity["preparation"]["visible_text"].get("tokenizer_sha256")
    described = encoder.describe()
    found_sha = described.get("tokenizer_sha256")
    if found_sha is not None and expected_sha is not None and found_sha != expected_sha:
        raise TrainerError(
            "the model's tokenizer is not the one that defined the preparation's visible text "
            f"({found_sha[:12]} != {expected_sha[:12]})"
        )
    lengths = check_lengths(inputs, encoder)
    say(
        f"queries: longest {lengths['longest_query_tokens']} tokens (limit {cfg.query_max_length}, never "
        f"truncated); positive passages: {lengths['passages_truncated_to_visible_length']} of "
        f"{lengths['positive_passages']} longer than {cfg.passage_max_length} tokens (longest "
        f"{lengths['longest_passage_tokens']}) are truncated to {cfg.passage_max_length}, keeping every pair's "
        "visible text and evidence"
    )

    seed_everything(cfg.seed)
    module: torch.nn.Module = encoder  # type: ignore[assignment]
    module.to(device)
    if cfg.gradient_checkpointing:
        encoder.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        _parameter_groups(module, cfg.weight_decay), lr=cfg.learning_rate
    )
    total = len(schedule.steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total, schedule.warmup_steps)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.fp16)
    history: list[dict[str, Any]] = []
    selection = SelectionState() if validating else None
    done = 0
    if latest is not None:
        done, history, selection = restore_checkpoint(latest, encoder, optimizer, scheduler, scaler, schedule,
                                                      validating=validating)
        say(f"resuming {inputs.config.run_name} after step {done} of {total} ({latest.name})")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_file.write_bytes(record_bytes)
    (run_dir / LOG_FILE).write_text(
        "".join(json.dumps(h, sort_keys=True) + "\n" for h in history), encoding="utf-8"
    )
    if selection is not None:
        (run_dir / VALIDATION_LOG_FILE).write_text(
            "".join(json.dumps(e, sort_keys=True) + "\n" for e in [*selection.evaluations, *selection.after_stop]),
            encoding="utf-8",
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    by_id = inputs.by_id
    params = [p for p in module.parameters() if p.requires_grad]
    result = TrainResult(run_dir, done, total, history)

    def checkpoint(step: int, evaluated: dict[str, Any] | None) -> None:
        path = _save_checkpoint(
            run_dir, step, total, encoder, optimizer, scheduler, scaler, inputs, schedule, history, device,
            lengths, described, evaluated=evaluated, selection=selection,
        )
        result.checkpoints.append(path)
        if evaluated is not None:
            with (run_dir / VALIDATION_LOG_FILE).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(evaluated, sort_keys=True) + "\n")

    def validate(step: int) -> dict[str, Any]:
        assert selection is not None and evaluate is not None
        if selection.should_stop:  # after the rule's stop: recorded for the decision rule, never selectable
            entry = selection.record_after_stop(step, evaluate(encoder, step))
            say(f"validation at step {step} (after the rule's stop: recorded, never selectable): whole-corpus MRR@10 "
                f"{entry['value']:.4f}")
            return entry
        entry = selection.record(step, evaluate(encoder, step))
        kind = "a selection candidate" if entry["candidate"] else "the final step, reported only"
        best = ", best so far" if entry["improved"] else ""
        say(f"validation at step {step} ({kind}): whole-corpus MRR@10 {entry['value']:.4f}{best}")
        if selection.should_stop:
            say(f"the rule stops selecting at step {step} ({selection.since_best} evaluations without improvement): "
                f"step {selection.best_step} stays selected; training and evaluation continue to step {total}, "
                "recorded as never selectable, for the decision rule")
        return entry

    if selection is not None and latest is None:
        checkpoint(0, validate(0))  # the base model, before any update: step 0 of the selection rule
    evaluated_steps = set(evaluation_steps(total)) if validating else set()
    for step in schedule.steps[done:]:
        module.train()
        started = time.perf_counter()
        step_loss = 0.0
        replay = 0.0
        for batch in step.batches:
            check_batch(inputs, batch)  # again, right before the forward pass
            pairs = [by_id[i] for i in batch]
            queries = [inputs.queries[p.pair_id] for p in pairs]
            positives = [inputs.passages[p.positive_passage_id] for p in pairs]
            if cfg.mini_batch_size is not None:
                loss_value, difference = cached_contrastive_backward(
                    encoder, queries, positives, temperature=cfg.temperature, mini_batch_size=cfg.mini_batch_size,
                    query_max_length=cfg.query_max_length, passage_max_length=cfg.passage_max_length, device=device,
                    fp16=cfg.fp16, scale=scaler.scale, divisor=len(step.batches),
                )
                step_loss += loss_value
                replay = max(replay, difference)
                continue
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.fp16):
                q = encoder.encode(queries, cfg.query_max_length)
                p_emb = encoder.encode(positives, cfg.passage_max_length)
            loss = contrastive_loss(q, p_emb, cfg.temperature) / len(step.batches)
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm or float("inf"))
        lr = optimizer.param_groups[0]["lr"]
        if validating and not math.isfinite(float(grad_norm)):
            scale = f" at loss scale {scaler.get_scale():g}" if cfg.fp16 else ""
            raise TrainerError(
                f"step {step.number}: the gradients are not finite{scale} (an fp16 overflow or a diverged loss). The run "
                "stops here, before the optimizer step: the GradScaler is not left to skip the step and lower its "
                f"scale, and no scaler setting is changed. Checkpoints up to step {result.steps_done} are kept; a "
                "change of loss scale needs an explicit decision and a new run"
            )
        loss_scale = scaler.get_scale() if cfg.fp16 else None
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        entry = {
            "step": step.number,
            "epoch": step.epoch,
            "loss": round(step_loss, 6),
            "learning_rate": lr,
            "grad_norm": round(float(grad_norm), 6),
            "batches": [len(b) for b in step.batches],
            "seconds": round(time.perf_counter() - started, 3),
        }
        if cfg.mini_batch_size is not None:
            entry["gradcache_replay_difference"] = replay
        if validating and loss_scale is not None:
            entry["loss_scale"] = loss_scale
        history.append(entry)
        with (run_dir / LOG_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        say(f"step {step.number}/{total} epoch {step.epoch} loss {entry['loss']:.4f} lr {lr:.2e}")
        result.steps_done = step.number
        evaluated = validate(step.number) if step.number in evaluated_steps else None
        last = step.number == total or step.number == stop_after
        every = inputs.config.checkpoints.save_every_steps
        if evaluated is not None or last or (every is not None and step.number % every == 0):
            checkpoint(step.number, evaluated)
        if step.number == stop_after:
            break
    if selection is not None and result.steps_done == total:
        record_sha = hashlib.sha256(record_bytes).hexdigest()
        chosen = make_selection_record(
            run=inputs.config.run_name, seed=cfg.seed, seeds=inputs.config.seed_list,  # type: ignore[arg-type]
            config_sha256=inputs.config_sha256, schedule_sha256=schedule.sha256, run_record_sha256=record_sha,
            validation=inputs.identity["validation"], state=selection, planned_steps=total,
            steps_done=result.steps_done, environment=run_environment(device),
        )
        write_selection_record(run_dir, chosen)
        result.selection = run_dir / SELECTION_FILE
        picked = chosen["selected"]
        say(f"selected {picked['checkpoint']} (step {picked['step']}{', the base model' if picked['base_model'] else ''};"
            f" whole-corpus validation MRR@10 {picked['value']:.4f}); recorded with every checkpoint's hashes in "
            f"{result.selection}")
    return result


def _save_checkpoint(
    run_dir: Path,
    step: int,
    total: int,
    encoder: Encoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    inputs: TrainingInputs,
    schedule: Schedule,
    history: list[dict[str, Any]],
    device: torch.device,
    lengths: dict[str, int],
    described: dict[str, Any],
    *,
    evaluated: dict[str, Any] | None = None,
    selection: SelectionState | None = None,
) -> Path:
    """Write ``checkpoint-step-NNNNNN`` atomically; earlier checkpoints keep their model only.

    An evaluated checkpoint is hashed before its trainer state is added (``checkpoint_selection.checkpoint_files``),
    and ``evaluated`` (its selection entry) gains the checkpoint's name, file hashes and digest.
    """
    cfg = inputs.config.training
    final = run_dir / checkpoint_name(step)
    if final.exists():
        raise TrainerError(f"{final} exists; checkpoints are never written over")
    tmp = run_dir / f".{final.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)  # a crash mid-write left it; it was never a checkpoint
    tmp.mkdir()
    encoder.save(tmp, max_seq_length=cfg.passage_max_length)
    memory = None
    if device.type == "cuda":
        memory = {
            "max_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
            "device_name": torch.cuda.get_device_name(device),
            "measured": True,
        }
    objective = {
        "loss": "in-batch InfoNCE over L2-normalised embeddings (MultipleNegativesRankingLoss, scale 1/t)",
        "temperature": cfg.temperature,
        "batch_size": cfg.batch_size,
        "in_batch_negatives_per_query": cfg.batch_size - 1,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "accumulation_note": "gradients of several batches are averaged; negatives come from each batch alone",
        "false_negative_protection": "training_data.plan_batches and trainer_setup.check_batch",
        **({} if cfg.batching is None else {"batching": cfg.batching,
                                            "in_batch_negatives_per_query_measured": schedule.negatives}),
        "learning_rate": cfg.learning_rate,
        "warmup_steps": schedule.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "fp16": cfg.fp16,
        "seed": cfg.seed,
    }
    if cfg.mini_batch_size is not None:
        objective["gradient_cache"] = {
            "mini_batch_size": cfg.mini_batch_size,
            "method": "GradCache (Gao et al., 2021): the full batch's loss, back-propagated through cached embeddings; "
            "each mini-batch replayed with its saved random state and checked against its cached pass",
            "replay_tolerance": REPLAY_TOLERANCE["fp16" if cfg.fp16 else "fp32"],
        }
    metadata: dict[str, Any] = {
        "kind": "finretriever_biencoder_checkpoint",
        "format_version": CHECKPOINT_FORMAT,
        "notice": CHECKPOINT_NOTICE,
        "run": inputs.config.run_name,
        "step": step,
        "total_steps": total,
        "epoch": history[-1]["epoch"] if history else 0,
        "config_sha256": inputs.config_sha256,
        "schedule_sha256": schedule.sha256,
        "base_model": inputs.config.base_model.model_dump(),
        "model": described,
        "encoding": {
            "query_prefix": inputs.config.base_model.query_prefix,
            "passage_prefix": inputs.config.base_model.passage_prefix,
            "pooling": inputs.config.base_model.pooling,
            "normalize": True,
            "query_max_length": cfg.query_max_length,
            "passage_max_length": cfg.passage_max_length,
            **lengths,
            "truncation_keeps_visible_text": True,  # check_lengths refuses to train otherwise
        },
        "objective": objective,
        "inputs": inputs.identity,
        "loss_history": history,
        "selection": "none: fixed-step checkpoint; no development or evaluation metric was computed",
        "environment": {**_versions(), "device": str(device)},
        "gpu_memory": memory,
    }
    if selection is not None:
        metadata["notice"] = VALIDATED_CHECKPOINT_NOTICE
        if evaluated is None:
            metadata["selection"] = "not evaluated: an interruption checkpoint, never selected"
        elif evaluated.get("after_stop"):
            metadata["selection"] = (
                "after the rule's stop: evaluated on the validation set and recorded for the decision rule; never "
                f"selectable (which checkpoint the rule selected is recorded in ../{SELECTION_FILE})"
            )
        elif evaluated["candidate"]:
            metadata["selection"] = (
                f"a candidate of the fixed validation rule ({'the base model, ' if step == 0 else ''}evaluated at step "
                f"{step}); which checkpoint the rule selects is recorded in ../{SELECTION_FILE} when the run ends"
            )
        else:
            metadata["selection"] = "the final step: evaluated on the validation set and reported, never selected"
        metadata["validation"] = None if evaluated is None else {
            "set": inputs.identity["validation"]["version"],
            "step": step,
            "candidate": evaluated["candidate"],
            "selectable": evaluated["candidate"] and not evaluated.get("after_stop", False),
            "mrr@10_global": evaluated["value"],
            "metrics": {mode: evaluated["metrics"].get(mode) for mode in validation_eval.MODES},
        }
    (tmp / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if evaluated is not None:
        files = checkpoint_files(tmp)
        evaluated.update(checkpoint=final.name, files=files, digest=files_digest(files))
    state = {
        "format_version": CHECKPOINT_FORMAT,
        "step": step,
        "schedule_sha256": schedule.sha256,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _rng_state(),
        "history": json.dumps(history),
    }
    if selection is not None:
        state["selection"] = json.dumps(selection.to_json())
    torch.save(state, tmp / STATE_FILE)
    tmp.rename(final)
    for older in run_dir.glob("checkpoint-step-*"):
        if older != final and (older / STATE_FILE).is_file():
            (older / STATE_FILE).unlink()  # only the newest checkpoint resumes
    return final


# --- command line -------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="05_train_biencoder.py",
        description="Fine-tune the bi-encoder on the verified training candidates.",
    )
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Load the model and train; without it the inputs and schedule are verified and printed only.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue the run from its newest checkpoint.")
    parser.add_argument(
        "--seed", type=int, help="The seed to train, one of the config's seeds (one run, in its own directory, each)."
    )
    parser.add_argument(
        "--local-files-only", action="store_true", help="Load the base model from the local cache only."
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    encoder_loader: Callable[[BaseModelConfig], Encoder] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    args = build_parser().parse_args(argv)

    def say(text: str) -> None:
        print(f"{PREFIX} {text}", file=out)

    try:
        config, config_sha256 = load_trainer_config(args.config)
        seeds = [args.seed] if args.seed is not None else list(config.seed_list)
        inputs = prepare_inputs(config.for_seed(seeds[0]), config_sha256)
        cfg = config.training
        say(f"{config.name}: {len(inputs.pairs)} training candidates from {config.data.candidates} "
            f"({inputs.identity['candidates']['available']} available)")
        runs = [inputs.for_seed(seed) for seed in seeds]
        for run in runs:
            schedule = plan_steps(run)
            negatives = schedule.negatives
            say(f"run {run.config.run_name}: {len(schedule.steps)} optimizer steps (warmup {schedule.warmup_steps}); "
                f"{schedule.batching} batches, per epoch {list(schedule.batches_per_epoch)}; in-batch negatives per "
                f"query: at least {negatives['min']}, {negatives['mean']} on average (at most {cfg.batch_size - 1}); "
                f"{schedule.dropped_batches} single-pair batches dropped; schedule {schedule.sha256[:12]}")
        say("every batch verified: no known equivalent, overlap sibling, reuse link, blocked or other-split passage "
            "is an in-batch negative")
        if cfg.mini_batch_size is not None:
            say(f"GradCache: the loss of each batch of up to {cfg.batch_size} pairs, back-propagated in mini-batches of "
                f"{cfg.mini_batch_size} texts (memory follows the mini-batch, negatives the batch)")
        if config.validation is not None:
            validation = inputs.identity["validation"]
            say(f"validation: {validation['version']}, {validation['pairs']} synthetic questions accepted by an AI "
                f"review (not gold), isolated from the {len(inputs.pairs)} training pairs; checkpoint selection only, "
                f"by the fixed rule: whole-corpus MRR@10 at step 0 and every 5 steps (evaluations at steps "
                f"{evaluation_steps(len(plan_steps(runs[0]).steps))}), ties to the earlier step; selection stops after 3 "
                "evaluations without improvement, and training and evaluation continue to the last step, those "
                f"evaluations never selectable; {len(inputs.corpus or {})} passages ranked per evaluation")
        estimate = estimate_gpu_memory_gib(
            batch_size=cfg.mini_batch_size or cfg.batch_size, query_tokens=cfg.query_max_length,
            passage_tokens=cfg.passage_max_length, fp16=cfg.fp16, gradient_checkpointing=cfg.gradient_checkpointing,
        )
        per = f" per mini-batch of {cfg.mini_batch_size}" if cfg.mini_batch_size is not None else ""
        say(f"ESTIMATED peak GPU memory (not measured){per}: {estimate}; {MEASURED_T4}")
        if not args.confirm_training:
            say("dry run: no model loaded, nothing written (pass --confirm-training to train)")
            return 0
        if args.seed is None and cfg.seed is None:
            raise TrainerSetupError(f"train one seed per run: pass --seed (one of {list(config.seed_list)})")
        loader = encoder_loader or (
            lambda model: load_base_encoder(
                model, local_files_only=args.local_files_only, gradient_cache=cfg.mini_batch_size is not None
            )
        )
        result = train(runs[0], loader(config.base_model), resume=args.resume, say=say)
        say(f"done: {result.steps_done}/{result.total_steps} steps; checkpoints {[p.name for p in result.checkpoints]}")
        if result.selection is not None:
            say(f"under {result.run_dir}; checkpoints chosen only by the validation rule ({result.selection}); no "
                "development or evaluation metric computed, no improvement claimed")
        else:
            say(f"under {result.run_dir}; not evaluated, no checkpoint selected by a metric, no improvement claimed")
        return 0
    except TrainerError as exc:
        print(f"{PREFIX} error: {exc}", file=err)
        return 2
    except ValueError as exc:  # setup, data and contract errors (TrainerSetupError, TrainingDataError, ...)
        print(f"{PREFIX} refused: {exc}", file=err)
        return 1


__all__ = [
    "CHECKPOINT_NOTICE",
    "REPLAY_TOLERANCE",
    "VALIDATED_CHECKPOINT_NOTICE",
    "Encoder",
    "HFEncoder",
    "RandomState",
    "TrainResult",
    "TrainerError",
    "TrainerSetupError",
    "cached_contrastive_backward",
    "check_lengths",
    "chunked_contrastive_backward",
    "contrastive_loss",
    "gradcache_check",
    "load_base_encoder",
    "main",
    "resolve_device",
    "restore_checkpoint",
    "run_environment",
    "seed_everything",
    "train",
    "write_sentence_transformers_files",
]
