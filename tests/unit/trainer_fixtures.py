"""A tiny trainable encoder and a validated training world for the trainer tests (offline, CPU, no downloads)."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

torch = pytest.importorskip("torch")

from finretriever.training.trainer_setup import (  # noqa: E402
    TrainerConfig,
    TrainingInputs,
    load_trainer_config,
    prepare_inputs,
)
from tests.unit.validation_fixtures import World, build_world, cli  # noqa: E402

VISIBLE_TOKENS = 40  # write_prep_config's visible_text.max_seq_length
VALIDATION_SET = "val_v1"


class TinyEncoder(torch.nn.Module):
    """Hashed words, averaged and projected, with dropout (so the random state matters)."""

    def __init__(self, dim: int = 16, buckets: int = 257, tokenizer_sha256: str | None = None) -> None:
        super().__init__()
        torch.manual_seed(0)  # the same initial weights in every run
        self.buckets, self.tokenizer_sha256 = buckets, tokenizer_sha256
        self.embedding = torch.nn.EmbeddingBag(buckets, dim, mode="mean")
        self.proj = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(0.1)

    def _ids(self, text: str, max_length: int) -> list[int]:
        return [zlib.crc32(w.casefold().encode()) % self.buckets for w in text.split()[: max_length - 2]] or [0]

    def encode(self, texts: Sequence[str], max_length: int) -> Any:
        ids = [self._ids(t, max_length) for t in texts]
        device = self.proj.weight.device
        flat = torch.tensor([i for row in ids for i in row], device=device)
        offsets = torch.tensor([0, *itertools.accumulate(len(r) for r in ids)][:-1], device=device)
        return self.proj(self.dropout(self.embedding(flat, offsets)))

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) + 2 for t in texts]

    def kept_chars(self, texts: Sequence[str], max_length: int) -> list[int]:
        words = (re.finditer(r"\S+", t) for t in texts)  # as _ids truncates them
        return [max((m.end() for m in itertools.islice(w, max_length - 2)), default=0) for w in words]

    def save(self, directory: Path, *, max_seq_length: int) -> None:
        torch.save(self.state_dict(), directory / "weights.pt")
        (directory / "encoder.json").write_text(json.dumps({"max_seq_length": max_seq_length}))

    def load_weights(self, directory: Path) -> None:
        self.load_state_dict(torch.load(directory / "weights.pt", weights_only=True))

    def enable_gradient_checkpointing(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {"class": "TinyEncoder", "tokenizer_sha256": self.tokenizer_sha256}


class TinyAttentionEncoder(TinyEncoder):
    """``TinyEncoder`` with a self-attention over its words whose probabilities pass through a dropout module at
    ``attention.self.dropout`` (BERT's path), so attention dropout can be told apart from the other dropout."""

    def __init__(self, dim: int = 16, buckets: int = 257) -> None:
        super().__init__(dim, buckets)
        self.tokens = torch.nn.Embedding(buckets, dim)
        self.attention = torch.nn.Module()
        self.attention.self = torch.nn.Module()
        self.attention.self.dropout = torch.nn.Dropout(0.1)

    def encode(self, texts: Sequence[str], max_length: int) -> Any:
        device = self.proj.weight.device
        pooled = []
        for text in texts:
            words = self.tokens(torch.tensor(self._ids(text, max_length), device=device))
            weights = self.attention.self.dropout(torch.softmax(words @ words.T / 4.0, dim=-1))
            pooled.append((weights @ words).mean(dim=0))
        return self.proj(self.dropout(torch.stack(pooled)))


def validated_world(root: Path) -> World:
    """The validation world (training candidates ``cand_v1``) with its validation set ``val_v1`` built."""
    with pytest.MonkeyPatch.context() as mp:
        world = build_world(root, mp)
        config_path = world.validation_config(VALIDATION_SET, world.review("val_decisions"))
        code, _, err = cli(world.config_path, "build-validation", "--validation", str(config_path))
        assert code == 0, err
    return world


def validated_config(world: World, tmp_path: Path, *, validation: bool = True, **training: Any) -> tuple[
    TrainerConfig, Path
]:
    """A trainer config over ``world``: TinyEncoder-sized settings, with the validation set unless ``validation`` is
    false; ``training`` overrides settings of the training section (``seeds`` replaces ``seed``) and the data
    settings ``candidates``, ``max_pairs`` and ``subset_seed``."""
    settings: dict[str, Any] = {"seed": 42, "batch_size": 3, "learning_rate": 0.01, "epochs": 40, "max_steps": 12,
                                "temperature": 0.05, "query_max_length": 64, "passage_max_length": VISIBLE_TOKENS,
                                "device": "cpu", "warmup_ratio": 0.1, "weight_decay": 0.01}
    data: dict[str, Any] = {"data_prep_config": str(world.config_path), "candidates": "cand_v1"}
    for key in ("candidates", "max_pairs", "subset_seed"):  # data settings
        if key in training:
            data[key] = training.pop(key)
    if "seeds" in training:
        settings.pop("seed")
    settings.update(training)
    values: dict[str, Any] = {
        "name": "tiny",
        "base_model": {"name": "fake/whitespace", "revision": "0" * 40, "query_prefix": "Find: ",
                       "passage_prefix": "", "pooling": "cls"},
        "data": data,
        "training": settings,
        "checkpoints": {"output_dir": str(tmp_path / "models")},
    }
    if validation:
        values["validation"] = {"version": VALIDATION_SET, "batch_size": 4}
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"trainer_{len(list(tmp_path.glob('trainer_*.yaml')))}.yaml"
    path.write_text(yaml.safe_dump(values))
    return load_trainer_config(path)[0], path


def validated_inputs(world: World, tmp_path: Path, seed: int | None = None, **training: Any) -> TrainingInputs:
    config, path = validated_config(world, tmp_path, **training)
    return prepare_inputs(config.for_seed(seed), hashlib.sha256(path.read_bytes()).hexdigest())
