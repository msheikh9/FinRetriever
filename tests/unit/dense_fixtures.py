"""Fake embedding models for dense-retrieval tests. Nothing here downloads or loads a real model."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

REVISION = "0123456789abcdef0123456789abcdef01234567"
CORPUS_SHA256 = "c" * 64


class FakeEncoder:
    """A deterministic bag-of-words "model": every word adds to one hashed dimension.

    Its vectors are deliberately not unit length, so the retriever's own
    normalisation is exercised. Every ``encode`` call is recorded.
    """

    def __init__(self, dim: int = 16, max_seq_length: int = 24, device: str = "cpu") -> None:
        self.embedding_dim = dim
        self.max_seq_length = max_seq_length
        self.device = device
        self.calls: list[list[str]] = []

    def vector(self, text: str) -> np.ndarray:
        out = np.zeros(self.embedding_dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            out[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.embedding_dim] += 1.0
        out[0] += 0.01  # never the zero vector
        return out * 5.0

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([self.vector(t) for t in texts]).astype(np.float32)

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) + 2 for t in texts]

    def describe(self) -> dict[str, Any]:
        return {"pooling": "fake", "libraries": {"sentence-transformers": None}}

    @property
    def encoded(self) -> list[str]:
        return [text for call in self.calls for text in call]


class TableEncoder(FakeEncoder):
    """A fake whose vectors are given per text; any other text maps to ``default``."""

    def __init__(self, vectors: Mapping[str, Sequence[float]], default: Sequence[float]) -> None:
        super().__init__(dim=len(default))
        self.vectors = {text: np.asarray(v, dtype=np.float32) for text, v in vectors.items()}
        self.default = np.asarray(default, dtype=np.float32)

    def vector(self, text: str) -> np.ndarray:
        return self.vectors.get(text, self.default)
