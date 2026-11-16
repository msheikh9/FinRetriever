"""The part of a passage a bi-encoder actually reads.

The encoder reads at most ``max_seq_length`` tokens, special tokens and any passage prefix
included; about 91% of this corpus is longer, so its tail is never embedded. A training query
must therefore be answerable from the visible part alone: ``visible_span`` finds where the
tokenizer's truncation ends in the original text (from the fast tokenizer's offsets), backed off
to the last word boundary so no word is cut in half. Query generation sees only that prefix, and
evidence must lie inside it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VisibleTextError(RuntimeError):
    """The pinned tokenizer cannot be loaded or gives no offsets."""


class VisibleTextSettings(BaseModel):
    """The tokenizer and input format whose limit defines the visible text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokenizer: str
    revision: str
    max_seq_length: int = Field(default=512, ge=8)
    passage_prefix: str = ""

    @field_validator("revision")
    @classmethod
    def _revision(cls, v: str) -> str:
        if len(v) != 40 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("revision must be a full 40-character commit SHA")
        return v


class Tokenizer(Protocol):
    """A Hugging Face fast tokenizer, or anything answering the same call."""

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class VisibleSpan:
    chars: int  # the visible text is text[:chars]
    tokens: int  # content tokens kept (special and prefix tokens excluded)
    truncated: bool


def visible_span(
    text: str, tokenizer: Tokenizer, *, max_seq_length: int, prefix: str = ""
) -> VisibleSpan:
    """Where ``tokenizer`` truncation at ``max_seq_length`` ends in ``text`` (prefix and special tokens counted)."""
    encoded = tokenizer(
        prefix + text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets, special = encoded["offset_mapping"], encoded["special_tokens_mask"]
    content = [
        (s, e) for (s, e), sp in zip(offsets, special, strict=True) if not sp and e > len(prefix)
    ]
    end = max((e for _, e in content), default=len(prefix)) - len(prefix)
    end = max(0, min(end, len(text)))
    truncated = bool(text[end:].strip())
    if truncated and end and text[end - 1].isalnum() and text[end].isalnum():
        cut = text.rfind(" ", 0, end)  # do not keep half a word
        end = cut if cut > 0 else end
    return VisibleSpan(chars=end, tokens=len(content), truncated=truncated)


class TokenizerFile:
    """A ``tokenizers.Tokenizer`` (a model's ``tokenizer.json``) answering the call ``visible_span`` makes."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        **_: Any,
    ) -> dict[str, Any]:
        if truncation:
            self.tokenizer.enable_truncation(max_length=max_length)
        else:
            self.tokenizer.no_truncation()
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": encoding.ids,
            "offset_mapping": encoding.offsets,
            "special_tokens_mask": encoding.special_tokens_mask,
        }

    def to_str(self) -> str:
        self.tokenizer.no_truncation()
        return self.tokenizer.to_str()


def load_tokenizer(settings: VisibleTextSettings) -> TokenizerFile:
    """The pinned model's ``tokenizer.json`` from the local Hugging Face cache (never downloads).

    Uses the ``tokenizers`` library directly: the same fast tokenizer the model loads, without
    importing ``transformers``.
    """
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer as RustTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VisibleTextError(f"cannot load the tokenizer: {exc}") from None
    try:
        path = hf_hub_download(
            settings.tokenizer, "tokenizer.json", revision=settings.revision, local_files_only=True
        )
        return TokenizerFile(RustTokenizer.from_file(path))
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear, typed error
        raise VisibleTextError(
            f"cannot load tokenizer {settings.tokenizer} at {settings.revision} from the local cache ({exc}); "
            "download the model once (see docs/eval_methodology.md §11)"
        ) from None


def tokenizer_identity(tokenizer: Any, settings: VisibleTextSettings) -> dict[str, Any]:
    """What defines the visible text: the tokenizer, its serialized state's SHA-256 and the limit."""
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    serialized = backend.to_str() if hasattr(backend, "to_str") else repr(tokenizer)
    return {
        "tokenizer": settings.tokenizer,
        "revision": settings.revision,
        "class": type(tokenizer).__name__,
        "tokenizer_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "max_seq_length": settings.max_seq_length,
        "passage_prefix": settings.passage_prefix,
    }


__all__ = [
    "Tokenizer",
    "TokenizerFile",
    "VisibleSpan",
    "VisibleTextError",
    "VisibleTextSettings",
    "load_tokenizer",
    "tokenizer_identity",
    "visible_span",
]
