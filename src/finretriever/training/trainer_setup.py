"""Bi-encoder training setup: configuration, verified inputs, batch schedule and memory estimate.

Nothing here needs ``torch``; the training loop is ``finretriever.training.biencoder_trainer``.

**Configuration (``TrainerConfig``, ``load_trainer_config``).** A pinned base model (40-character
revision) with the input format evaluation uses (``retrieval.biencoder.resolve_format``: BGE's
query instruction, no passage prefix) and its own pooling; the training candidates to use; the
optimisation settings; where checkpoints go. Unknown keys are errors, so no evaluation set can
be configured for checkpoint selection, and no configured path may lie under ``data/eval``.

**Inputs (``prepare_inputs``).** The data-preparation config, the frozen corpus, the preparation
and the candidate dataset are each verified (``candidates.load_candidates``; no evaluation file
is read). It fails closed (``TrainerSetupError``) when the preparation's visible text was not
defined by the configured model, revision, sequence length and passage prefix, when fewer than
two candidates remain, or when a batch would hold a passage that is not an allowed negative.

**Schedule (``plan_steps``).** Each epoch's batches come from ``training_data.plan_batches``
seeded with ``seed + epoch``, so no known equivalent, overlap sibling, reuse link, blocked or
other-split passage is ever an in-batch negative, and each batch is checked again here. A batch
of one pair has no negative and is dropped (recorded). ``gradient_accumulation_steps`` batches
make one optimizer step; accumulation does not add in-batch negatives. ``batching`` chooses the
planner: unset (every historical config) is ``first_fit``, which packs full batches and leaves a
tail of small ones where conflicts cluster; ``balanced`` fills the fewest batches ``batch_size``
allows, of near-equal size, so every pair trains in every epoch (``batch_size`` is then a maximum).

**Seeds.** A config names one ``seed`` or a list of ``seeds``; a run trains one seed
(``TrainerConfig.for_seed``, ``--seed``) in its own directory ``<name>-seed<seed>``. The seed sets the
batch order and dropout; the data, schedule length and selection rule are the same for every seed.

**GradCache.** ``mini_batch_size`` (at most ``batch_size``) turns on gradient caching
(``biencoder_trainer.cached_contrastive_backward``): the loss is the full batch's, but only one
mini-batch of activations is held at a time, so ``batch_size`` sets the in-batch negatives and
``mini_batch_size`` the memory.

**Validation (``validation``).** A validation set built by ``validation_set`` (``validation_sets/``
of the same preparation) is loaded with ``load_validation_set(training=<the pairs trained on>)``,
which fails closed if any training pair shares a passage with it or (lexically) a question. It is
used only to choose among the run's checkpoints, by the fixed rule of ``checkpoint_selection``;
checkpoints are then written at the evaluated steps, so ``save_every_steps`` is not set. No
evaluation file is read.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finretriever.data.schemas import Passage
from finretriever.eval_set.annotation_store import load_corpus_passages
from finretriever.evaluation.provenance import sha256_file
from finretriever.retrieval.biencoder import resolve_format
from finretriever.training.candidates import load_candidates, read_candidates_manifest
from finretriever.training.checkpoint_selection import EVERY_STEPS, RULE
from finretriever.training.data_prep import (
    PreparedData,
    load_data_prep_config,
    open_prepared,
    verify_frozen_corpus,
)
from finretriever.training.pair_contract import TrainingPair
from finretriever.training.training_data import NegativeGuard, plan_batches
from finretriever.training.validation_set import NOTICE as VALIDATION_NOTICE
from finretriever.training.validation_set import load_validation_set, read_validation_manifest

MODEL_POOLING = {"BAAI/bge-base-en-v1.5": "cls", "intfloat/e5-base-v2": "mean"}  # as each model card
TRAINING_DATA_NOTICE = (
    "Synthetic queries written by an LLM and accepted by an AI quality review: not human-validated, not gold "
    "labels."
)
_NAME = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class TrainerSetupError(ValueError):
    """The configuration or inputs cannot be trusted for training; nothing is trained."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseModelConfig(_Strict):
    name: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    query_prefix: str
    passage_prefix: str
    pooling: Literal["cls", "mean"]

    @model_validator(mode="after")
    def _format(self) -> BaseModelConfig:
        resolve_format(self.name, self.query_prefix, self.passage_prefix)  # BGE: instruction on queries only
        expected = MODEL_POOLING.get(self.name)
        if expected is not None and self.pooling != expected:
            raise ValueError(f"{self.name} pools by {expected}, not {self.pooling}")
        return self


class DataConfig(_Strict):
    data_prep_config: str
    candidates: str = Field(pattern=_NAME)  # a version under <preparation>/candidates/
    max_pairs: int | None = Field(default=None, ge=2)  # a seeded subset (smoke tests); None: all
    subset_seed: int = 0


class OptimConfig(_Strict):
    seed: int | None = None  # the run's seed; set from ``seeds`` by TrainerConfig.for_seed
    seeds: tuple[int, ...] | None = Field(default=None, min_length=1)  # one run per seed, each in its own directory
    batch_size: int = Field(ge=2)  # pairs per forward pass: batch_size - 1 negatives per query
    mini_batch_size: int | None = Field(default=None, ge=1)  # GradCache: texts per forward pass (memory, not negatives)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    epochs: int = Field(ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    temperature: float = Field(gt=0, le=1)
    query_max_length: int = Field(ge=8)  # a longer query is an error, never truncated
    passage_max_length: int = Field(ge=8)  # the preparation's visible-text limit
    max_grad_norm: float | None = Field(default=1.0, gt=0)
    fp16: bool = False  # CUDA only
    gradient_checkpointing: bool = False
    batching: Literal["first_fit", "balanced"] | None = None  # the batch planner; unset: first_fit
    device: str = Field(default="auto", pattern=r"^(auto|cpu|cuda(:\d+)?)$")

    @model_validator(mode="after")
    def _fp16(self) -> OptimConfig:
        if self.fp16 and self.device == "cpu":
            raise ValueError("fp16 needs CUDA; set fp16: false to train on CPU")
        return self

    @model_validator(mode="after")
    def _seeds_and_mini_batch(self) -> OptimConfig:
        if self.seed is None and self.seeds is None:
            raise ValueError("set seed, or seeds (one run per seed)")
        if self.seeds is not None:
            if len(set(self.seeds)) != len(self.seeds):
                raise ValueError(f"seeds {list(self.seeds)} repeat a seed")
            if self.seed is not None and self.seed not in self.seeds:
                raise ValueError(f"seed {self.seed} is not one of seeds {list(self.seeds)}")
        if self.mini_batch_size is not None and self.mini_batch_size > self.batch_size:
            raise ValueError(f"mini_batch_size {self.mini_batch_size} exceeds batch_size {self.batch_size}")
        return self


class CheckpointConfig(_Strict):
    output_dir: str
    save_every_steps: int | None = Field(default=None, ge=1)  # the last step is always saved


class ValidationConfig(_Strict):
    """The validation set that chooses among a run's checkpoints (``checkpoint_selection.RULE``, fixed in code)."""

    version: str = Field(pattern=_NAME)  # a set under <preparation>/validation_sets/
    batch_size: int = Field(default=64, ge=1)  # texts per forward pass when encoding the corpus for validation


class TrainerConfig(_Strict):
    """Training settings; see the module docstring."""

    name: str = Field(pattern=_NAME)
    base_model: BaseModelConfig
    data: DataConfig
    training: OptimConfig
    checkpoints: CheckpointConfig
    validation: ValidationConfig | None = None

    @model_validator(mode="after")
    def _no_evaluation_paths(self) -> TrainerConfig:
        for path in (self.data.data_prep_config, self.checkpoints.output_dir):
            if under_evaluation_data(path):
                raise ValueError(f"{path} is under data/eval; training never reads or writes evaluation data")
        return self

    @model_validator(mode="after")
    def _checkpoints_follow_validation(self) -> TrainerConfig:
        if self.validation is not None and self.checkpoints.save_every_steps is not None:
            raise ValueError(
                f"with validation, checkpoints are written at the evaluated steps (step 0 and every {EVERY_STEPS} "
                "steps); leave save_every_steps unset"
            )
        return self

    @property
    def run_name(self) -> str:
        if self.training.seed is None:
            raise TrainerSetupError(f"{self.name} lists seeds {list(self.seed_list)}: choose one with --seed")
        return f"{self.name}-seed{self.training.seed}"

    @property
    def seed_list(self) -> tuple[int, ...]:
        if self.training.seeds is not None:
            return self.training.seeds
        assert self.training.seed is not None
        return (self.training.seed,)

    def for_seed(self, seed: int | None) -> TrainerConfig:
        """The config of one run: ``seed`` must be one of the config's seeds (``None`` keeps a single seed)."""
        if seed is None:
            if self.training.seed is None:
                raise TrainerSetupError(f"{self.name} lists seeds {list(self.seed_list)}: choose one with --seed")
            return self
        if seed not in self.seed_list:
            raise TrainerSetupError(f"seed {seed} is not one of the config's seeds {list(self.seed_list)}")
        return self.model_copy(update={"training": self.training.model_copy(update={"seed": seed})})


def config_record(config: TrainerConfig) -> dict[str, Any]:
    """The config as runs record it. Settings added after the first runs are left out while unset, so a config
    without them records exactly what it recorded before."""
    data = config.model_dump(mode="json")
    for key in ("seeds", "mini_batch_size", "batching"):
        if data["training"].get(key) is None:
            data["training"].pop(key, None)
    if data.get("validation") is None:
        data.pop("validation", None)
    return data


def under_evaluation_data(path: str | Path) -> bool:
    parts = Path(path).resolve().parts
    return any(parts[i : i + 2] == ("data", "eval") for i in range(len(parts) - 1))


def load_trainer_config(path: str | Path) -> tuple[TrainerConfig, str]:
    path = Path(path)
    try:
        return TrainerConfig.model_validate(yaml.safe_load(path.read_text("utf-8"))), sha256_file(path)
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        raise TrainerSetupError(f"{path} is not a valid training config: {exc}") from None


def checkpoint_name(step: int) -> str:
    return f"checkpoint-step-{step:06d}"


@dataclass(frozen=True)
class TrainingInputs:
    """Verified pairs, their texts as the model reads them, and what the run is bound to."""

    config: TrainerConfig
    config_sha256: str
    prepared: PreparedData
    pairs: tuple[TrainingPair, ...]
    guard: NegativeGuard
    queries: Mapping[str, str]  # pair ID -> query with the model's query prefix
    passages: Mapping[str, str]  # passage ID -> passage text with the model's passage prefix
    identity: Mapping[str, Any]
    validation: tuple[TrainingPair, ...] = ()  # the validation set (checkpoint selection only), if configured
    corpus: Mapping[str, Passage] | None = None  # every corpus passage, in corpus order (ranked by validation)

    @property
    def by_id(self) -> dict[str, TrainingPair]:
        return {p.pair_id: p for p in self.pairs}

    def for_seed(self, seed: int | None) -> TrainingInputs:
        """The same verified inputs for one seed's run (``TrainerConfig.for_seed``)."""
        return dataclasses.replace(self, config=self.config.for_seed(seed))


def check_compatible(config: TrainerConfig, prepared: PreparedData) -> None:
    """The preparation's visible text must be what this model reads: same tokenizer, revision,
    sequence length and passage prefix."""
    visible = prepared.manifest["visible_text"]["identity"]
    model = config.base_model
    expected = {
        "tokenizer": model.name,
        "revision": model.revision,
        "max_seq_length": config.training.passage_max_length,
        "passage_prefix": model.passage_prefix,
    }
    if differ := sorted(k for k, v in expected.items() if visible.get(k) != v):
        raise TrainerSetupError(
            f"the preparation's visible text was defined for another encoder ({', '.join(differ)} differ: "
            f"{ {k: visible.get(k) for k in differ} }); its leakage and false-negative checks do not cover this model"
        )


def select_subset(pairs: Sequence[TrainingPair], data: DataConfig) -> tuple[TrainingPair, ...]:
    """A seeded subset of ``max_pairs`` pairs, in candidate order (all pairs when unset)."""
    if data.max_pairs is None or data.max_pairs >= len(pairs):
        return tuple(pairs)
    chosen = set(random.Random(data.subset_seed).sample(sorted(p.pair_id for p in pairs), data.max_pairs))
    return tuple(p for p in pairs if p.pair_id in chosen)


def prepare_inputs(config: TrainerConfig, config_sha256: str) -> TrainingInputs:
    """Load and verify everything a run trains on; see the module docstring."""
    prep_config, prep_sha256 = load_data_prep_config(config.data.data_prep_config)
    corpus = verify_frozen_corpus(prep_config)
    passages = load_corpus_passages(corpus)
    prepared = open_prepared(prep_config.output_dir, corpus, passages)
    check_compatible(config, prepared)
    manifest, manifest_sha256 = read_candidates_manifest(prepared, config.data.candidates)
    pairs = select_subset(
        load_candidates(prepared, passages, prep_config, config.data.candidates, "train"), config.data
    )
    if len(pairs) < 2:
        raise TrainerSetupError(
            f"{len(pairs)} training candidates in {config.data.candidates!r}; in-batch negatives need at least 2"
        )
    model = config.base_model
    ids = [p.pair_id for p in pairs]
    identity = {
        "base_model": model.model_dump(),
        "corpus": prepared.manifest["corpus"],
        "preparation": {
            "data_prep_config": {"path": config.data.data_prep_config, "sha256": prep_sha256},
            "version": prepared.manifest["version"],
            "manifest_sha256": prepared.manifest_sha256,
            "evidence_index_digest": manifest["preparation"]["evidence_index_digest"],
            "visible_text": prepared.manifest["visible_text"]["identity"],
        },
        "candidates": {
            "version": config.data.candidates,
            "manifest_sha256": manifest_sha256,
            "equivalence_method": manifest["checks"]["equivalence_method"],
            "validator_version": manifest["checks"]["validator_version"],
            "available": manifest["stats"]["included_by_split"].get("train", 0),
        },
        "pairs": {
            "count": len(pairs),
            "pair_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "subset": {"max_pairs": config.data.max_pairs, "seed": config.data.subset_seed},
        },
        "notice": TRAINING_DATA_NOTICE,
    }
    validation: tuple[TrainingPair, ...] = ()
    if config.validation is not None:
        version = config.validation.version
        _, validation_sha256 = read_validation_manifest(prepared, version)
        # fails closed if a trained pair shares a passage with the set or (lexically) a question
        validation = tuple(load_validation_set(prepared, passages, prep_config, version, training=pairs))
        if not validation:
            raise TrainerSetupError(f"validation set {version!r} holds no questions; nothing could select a checkpoint")
        identity["validation"] = {
            "version": version,
            "manifest_sha256": validation_sha256,
            "pairs": len(validation),
            "pair_ids_sha256": hashlib.sha256("\n".join(p.pair_id for p in validation).encode()).hexdigest(),
            "isolation": f"checked against the {len(pairs)} pairs trained on (validation_set.isolation_problems): no "
            "shared passage or equivalent, no lexically overlapping question",
            "use": "checkpoint selection only",
            "rule": RULE,
            "notice": VALIDATION_NOTICE,
        }
    return TrainingInputs(
        config=config,
        config_sha256=config_sha256,
        prepared=prepared,
        pairs=pairs,
        guard=NegativeGuard(prepared),
        queries={p.pair_id: model.query_prefix + p.query for p in pairs},
        passages={
            p.positive_passage_id: model.passage_prefix + passages[p.positive_passage_id].text for p in pairs
        },
        identity=identity,
        validation=validation,
        corpus=passages if validation else None,
    )


def check_batch(inputs: TrainingInputs, batch: Sequence[str]) -> None:
    """Every other positive in ``batch`` must be an allowed negative for each pair."""
    by_id = inputs.by_id
    members = [by_id[i] for i in batch]
    positives = [p.positive_passage_id for p in members]
    if len(set(positives)) != len(positives):
        raise TrainerSetupError(f"a batch repeats a positive passage: {sorted(batch)}")
    for a in members:
        for b in members:
            if a is not b and not inputs.guard.allowed_negative(a, b.positive_passage_id):
                raise TrainerSetupError(
                    f"{b.positive_passage_id} (the positive of {b.pair_id}) may not be a negative for "
                    f"{a.pair_id}: a known equivalent, overlap sibling, reuse link, blocked or other-split passage"
                )


@dataclass(frozen=True)
class Step:
    number: int  # 1-based optimizer step
    epoch: int  # 0-based
    batches: tuple[tuple[str, ...], ...]  # pair IDs; each batch is one forward pass and its own negatives


@dataclass(frozen=True)
class Schedule:
    steps: tuple[Step, ...]
    warmup_steps: int
    dropped_batches: int  # single-pair batches (no negative), left out
    batches_per_epoch: tuple[int, ...]
    batching: str = "first_fit"

    @property
    def negatives(self) -> dict[str, float]:
        """In-batch negatives per query over the schedule: the fewest and the mean (each query weighted once per
        batch it is in)."""
        sizes = [len(b) for s in self.steps for b in s.batches]
        return {"min": min(sizes) - 1, "mean": round(sum(n * (n - 1) for n in sizes) / sum(sizes), 2)}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps([[s.number, s.epoch, s.batches] for s in self.steps]).encode()
        ).hexdigest()


def plan_steps(inputs: TrainingInputs) -> Schedule:
    """Every optimizer step of the run, deterministically; see the module docstring."""
    cfg = inputs.config.training
    method = cfg.batching or "first_fit"
    steps: list[Step] = []
    dropped, per_epoch = 0, []
    for epoch in range(cfg.epochs):
        plan = plan_batches(inputs.pairs, inputs.guard, batch_size=cfg.batch_size, seed=cfg.seed + epoch,
                            method=method)
        batches = [b for b in plan.batches if len(b) >= 2]
        dropped += len(plan.batches) - len(batches)
        per_epoch.append(len(batches))
        for batch in batches:
            check_batch(inputs, batch)
        for i in range(0, len(batches), cfg.gradient_accumulation_steps):
            steps.append(Step(len(steps) + 1, epoch, tuple(batches[i : i + cfg.gradient_accumulation_steps])))
            if cfg.max_steps is not None and len(steps) == cfg.max_steps:
                break
        if cfg.max_steps is not None and len(steps) == cfg.max_steps:
            break
    if not steps:
        raise TrainerSetupError("no batch of at least two pairs: nothing to train on")
    warmup = math.ceil(cfg.warmup_ratio * len(steps))
    return Schedule(tuple(steps), warmup, dropped, tuple(per_epoch), method)


def lr_factor(step_index: int, total: int, warmup: int) -> float:
    """Learning-rate multiplier at 0-based optimizer step ``step_index``: linear warmup to 1 over
    ``warmup`` steps, then linear decay; never 0 on a step that is taken."""
    if step_index < warmup:
        return (step_index + 1) / warmup
    return max(0.0, (total - step_index) / max(1, total - warmup))


def estimate_gpu_memory_gib(
    *,
    parameters: float = 109.5e6,
    layers: int = 12,
    hidden: int = 768,
    heads: int = 12,
    batch_size: int,
    query_tokens: int,
    passage_tokens: int,
    fp16: bool,
    gradient_checkpointing: bool,
) -> dict[str, float]:
    """An ESTIMATE, not a measurement, of peak GPU memory for one step of BERT-base training.

    Weights, gradients and AdamW states in fp32 take 16 bytes per parameter. Activations follow
    Korthikanti et al. (2022): about s*b*h*(34 + 5*a*s/h) bytes per layer with 16-bit activations
    (doubled in fp32), for the passage and query sequences of a batch; with gradient checkpointing,
    one layer's activations plus each layer's input. Memory-efficient attention kernels lower the
    attention term, so this is an upper estimate. CUDA context and allocator slack: 1 GiB.
    """
    width = 2 if fp16 else 4

    def per_layer(seq: int) -> float:
        return seq * batch_size * hidden * (34 + 5 * heads * seq / hidden) * width / 2

    full = sum(per_layer(s) for s in (passage_tokens, query_tokens))
    if gradient_checkpointing:
        activations = full + layers * batch_size * (passage_tokens + query_tokens) * hidden * width
    else:
        activations = layers * full
    gib = 1024**3
    static = 16 * parameters / gib
    estimate = {
        "weights_gradients_optimizer_gib": round(static, 2),
        "activations_gib": round(activations / gib, 2),
        "overhead_gib": 1.0,
    }
    estimate["total_gib"] = round(sum(estimate.values()), 2)
    return estimate


__all__ = [
    "MODEL_POOLING",
    "TRAINING_DATA_NOTICE",
    "BaseModelConfig",
    "CheckpointConfig",
    "DataConfig",
    "OptimConfig",
    "Schedule",
    "Step",
    "TrainerConfig",
    "TrainerSetupError",
    "TrainingInputs",
    "ValidationConfig",
    "check_batch",
    "check_compatible",
    "checkpoint_name",
    "config_record",
    "estimate_gpu_memory_gib",
    "load_trainer_config",
    "lr_factor",
    "plan_steps",
    "prepare_inputs",
    "select_subset",
    "under_evaluation_data",
]
