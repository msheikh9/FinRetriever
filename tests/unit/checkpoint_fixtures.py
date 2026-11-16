"""A fake trainer checkpoint and the training records it points to, for evaluation tests. Nothing here loads a model:
the weights file is a few bytes and the tokenizer a fixed string."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from finretriever.retrieval.biencoder import MODEL_FORMATS

BGE = "BAAI/bge-base-en-v1.5"
BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
FAKE_TOKENIZER = '{"model": "fake wordpiece", "version": 1}'
TOKENIZER_SHA256 = hashlib.sha256(FAKE_TOKENIZER.encode("utf-8")).hexdigest()
HIDDEN = 16
MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_training_side(
    root: Path, corpus_path: Path, *, blocked: Sequence[str], candidates: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """A preparation (config, manifest, blocklist) and a training-candidate dataset; returns the ``inputs`` a
    checkpoint's metadata records for them."""
    prep = root / "prep"
    _jsonl(prep / "blocklist.jsonl", [{"passage_id": pid, "reasons": ["dev_pool:fixture"]} for pid in blocked])
    manifest = {"version": "prep_fixture", "files": {"blocklist.jsonl": _sha(prep / "blocklist.jsonl")},
                "blocklist": {"sources": [{"kind": "dev_pool", "path": "fixture_pool"}]}}
    (prep / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cand_dir = prep / "candidates" / "cand_fixture"
    _jsonl(cand_dir / "train_candidates.jsonl", candidates)
    (cand_dir / "manifest.json").write_text(
        json.dumps({"kind": "training_candidates", "files": {"train_candidates.jsonl": _sha(cand_dir / "train_candidates.jsonl")}}),
        encoding="utf-8",
    )
    config = root / "data_prep.yaml"
    config.write_text(json.dumps({"version": "prep_fixture", "output_dir": str(prep),
                                  "leakage": {"lexical_overlap_threshold": 0.8}}), encoding="utf-8")
    corpus = json.loads(corpus_path.with_name("corpus.manifest.json").read_text())
    return {
        "corpus": {"path": str(corpus_path), "sha256": corpus["corpus"]["sha256"], "version": corpus["corpus_version"],
                   "passages": corpus["corpus"]["passages"]},
        "preparation": {"version": "prep_fixture", "manifest_sha256": _sha(prep / "manifest.json"),
                        "data_prep_config": {"path": str(config), "sha256": _sha(config)},
                        "visible_text": {"tokenizer_sha256": TOKENIZER_SHA256}},
        "candidates": {"version": "cand_fixture", "manifest_sha256": _sha(cand_dir / "manifest.json"),
                       "available": len(candidates)},
        "pairs": {"count": len(candidates)},
        "notice": "Synthetic queries accepted by an AI quality review: not human-validated, not gold labels.",
    }


def candidate(pair_id: str, positive: str, query: str, *, exact: Sequence[str] = (), uncertain: Sequence[str] = ()) -> dict[str, Any]:
    return {"pair_id": pair_id, "split": "train", "positive_passage_id": positive, "query": query,
            "equivalents": {"exact": list(exact), "uncertain": list(uncertain)}}


def write_checkpoint(
    run_dir: Path, step: int, inputs: dict[str, Any], *, weights: bytes = b"fake safetensors", run: str = "fixture-seed42"
) -> Path:
    """``<run_dir>/checkpoint-step-NNNNNN/`` with the files the trainer writes, and ``run.json`` beside it."""
    fmt = MODEL_FORMATS[BGE]
    directory = run_dir / f"checkpoint-step-{step:06d}"
    (directory / "1_Pooling").mkdir(parents=True)
    files: dict[str, Any] = {
        "config.json": {"architectures": ["BertModel"], "model_type": "bert", "hidden_size": HIDDEN,
                        "max_position_embeddings": 64, "vocab_size": 100},
        "tokenizer_config.json": {"model_max_length": 64, "tokenizer_class": "BertTokenizer"},
        "modules.json": MODULES,
        "sentence_bert_config.json": {"max_seq_length": 32, "do_lower_case": False},
        "config_sentence_transformers.json": {"prompts": {}, "default_prompt_name": None, "similarity_fn_name": "cosine"},
        "1_Pooling/config.json": {"word_embedding_dimension": HIDDEN, "pooling_mode_cls_token": True,
                                  "pooling_mode_mean_tokens": False, "pooling_mode_max_tokens": False,
                                  "pooling_mode_mean_sqrt_len_tokens": False, "pooling_mode_weightedmean_tokens": False,
                                  "pooling_mode_lasttoken": False, "include_prompt": True},
        "metadata.json": {
            "kind": "finretriever_biencoder_checkpoint", "format_version": 1, "run": run, "step": step,
            "total_steps": 15, "epoch": 1, "config_sha256": "c" * 64, "schedule_sha256": "d" * 64,
            "base_model": {"name": BGE, "revision": BGE_REVISION, "query_prefix": fmt.query_prefix,
                           "passage_prefix": fmt.passage_prefix, "pooling": "cls"},
            "encoding": {"query_prefix": fmt.query_prefix, "passage_prefix": fmt.passage_prefix, "pooling": "cls",
                         "normalize": True, "passage_max_length": 32, "query_max_length": 16},
            "model": {"class": "BertModel", "hidden_size": HIDDEN, "pooling": "cls", "tokenizer_sha256": TOKENIZER_SHA256},
            "inputs": inputs,
            "selection": "none: fixed-step checkpoint; no development or evaluation metric was computed",
            "notice": "Fine-tuned on synthetic queries; not evaluated by the trainer.",
            "environment": {"device": "cpu"},
        },
    }
    for name, content in files.items():
        (directory / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    (directory / "tokenizer.json").write_text(FAKE_TOKENIZER, encoding="utf-8")
    (directory / "model.safetensors").write_bytes(weights)
    (run_dir / "run.json").write_text(json.dumps({"run": run, "config_sha256": "c" * 64}), encoding="utf-8")
    return directory


class FakeBackend:
    """Stands in for a fast tokenizer's backend: ``to_str`` is what the trainer hashes."""

    def __init__(self, text: str) -> None:
        self.text = text

    def no_truncation(self) -> None:
        pass

    def no_padding(self) -> None:
        pass

    def to_str(self) -> str:
        return self.text


class FakeTokenizer:
    def __init__(self, text: str = FAKE_TOKENIZER, size: int = 100, model_max_length: int = 64) -> None:
        self.backend_tokenizer = FakeBackend(text)
        self.size = size
        self.model_max_length = model_max_length
        self.vocab_size = size
        self.do_lower_case = True
        self.truncation_side = "right"

    def __len__(self) -> int:
        return self.size

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class _Module:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs

    def get_pooling_mode_str(self) -> str:
        return "cls"


class FakeModels:
    """``sentence_transformers.models`` with recording stand-ins for the three modules."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def Transformer(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802 - the library's class name
        self.calls.append(("Transformer", args, kwargs))
        return type("Transformer", (_Module,), {})("Transformer", **kwargs)

    def Pooling(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
        self.calls.append(("Pooling", args, kwargs))
        return type("Pooling", (_Module,), {})("Pooling", **kwargs)

    def Normalize(self) -> Any:  # noqa: N802
        self.calls.append(("Normalize", (), {}))
        return type("Normalize", (_Module,), {})("Normalize")


class FakeSentenceTransformer:
    """Just enough of a ``SentenceTransformer`` built from modules."""

    def __init__(self, *, modules: list[Any], device: str, tokenizer: FakeTokenizer | None = None, dim: int = HIDDEN,
                 max_seq_length: int = 32) -> None:
        self.modules = modules
        self.device = device
        self.tokenizer = tokenizer or FakeTokenizer()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.prompts: dict[str, str] = {}
        self.similarity_fn_name = "cosine"

    def __iter__(self) -> Any:
        return iter(self.modules)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return np.ones((len(texts), self.dim), dtype=np.float32)


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})()
        self.backends = type("Backends", (), {"mps": type("Mps", (), {"is_available": staticmethod(lambda: False)})()})()

    def get_num_threads(self) -> int:
        return 2
