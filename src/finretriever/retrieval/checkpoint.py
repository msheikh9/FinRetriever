"""Local fine-tuned checkpoints as dense retrievers, opened without trusting the directory.

A checkpoint written by the bi-encoder trainer (``finretriever.training.biencoder_trainer``) holds
the Hugging Face model and tokenizer, the sentence-transformers module files (CLS or mean pooling,
then normalisation) and ``metadata.json``: the base model, the input format it was trained with,
what it was trained on and at which step. :func:`inspect_checkpoint` checks the directory before
any model code runs:

* ``metadata.json`` is a trainer checkpoint record (kind and format) for a base model with a
  built-in input format (``biencoder.MODEL_FORMATS``), trained with that format's prefixes;
* the weights are ``model.safetensors`` only: a pickled weight file beside them is refused, and
  ``trainer_state.pt`` (optimizer state for resuming training) is never opened;
* ``config.json`` names no custom code (``auto_map``), and ``modules.json`` lists exactly the
  trainer's three modules, whose settings agree with the model config and the metadata;
* every file that determines the embeddings is hashed, and the digest of those hashes is the
  checkpoint's revision: the embedding cache and the run provenance name the exact weights.

:func:`load_checkpoint_encoder` then builds the three modules itself (local files only, no remote
code, safetensors only) instead of letting ``modules.json`` choose classes, and checks the loaded
model: the tokenizer is the one the checkpoint was trained with (its SHA-256 as the trainer hashed
it), every token id it can produce has an embedding row, the sequence limit fits the position
embeddings, the embedding dimension is the model's, and the weights did not change while loading.

:class:`CheckpointRetriever` is a :class:`~finretriever.retrieval.biencoder.BiEncoderRetriever`
over such an encoder: the same indexing, search, company restriction, tie-breaking and embedding
cache, so a checkpoint is scored exactly like the zero-shot baselines. A checkpoint its run chose by
the validation rule carries that verified selection (``selection``, from
``training.checkpoint_selection.verify_selected_checkpoint``) and may be a ``final`` model. ``torch`` and
``sentence_transformers`` are imported only when a model is loaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finretriever.retrieval import biencoder
from finretriever.retrieval.biencoder import (
    MODEL_FORMATS,
    BiEncoderRetriever,
    DenseModelError,
    ModelFormat,
    SentenceTransformerEncoder,
    TextEncoder,
    resolve_device,
)

CHECKPOINT_KIND = "finretriever_biencoder_checkpoint"
CHECKPOINT_FORMAT = 1
METADATA_FILE = "metadata.json"
WEIGHTS_FILE = "model.safetensors"
RUN_FILE = "run.json"
TRAINER_STATE_FILE = "trainer_state.pt"
MODULES = (
    ("", "sentence_transformers.models.Transformer"),
    ("1_Pooling", "sentence_transformers.models.Pooling"),
    ("2_Normalize", "sentence_transformers.models.Normalize"),
)
REQUIRED_FILES = (
    "config.json",
    WEIGHTS_FILE,
    "tokenizer.json",
    "tokenizer_config.json",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
    METADATA_FILE,
)
OPTIONAL_FILES = ("special_tokens_map.json", "vocab.txt", "added_tokens.json")  # tokenizer files some versions write
PICKLED_WEIGHTS = ("pytorch_model.bin", "pytorch_model.bin.index.json", "tf_model.h5", "flax_model.msgpack")
POOLING_MODES = {"cls": "pooling_mode_cls_token", "mean": "pooling_mode_mean_tokens"}


class CheckpointError(ValueError):
    """A checkpoint is missing, incomplete, unsafe to load or inconsistent with its own records."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"{path} is not readable JSON ({exc})") from None


@dataclass(frozen=True)
class LocalCheckpoint:
    """A checkpoint directory that passed :func:`inspect_checkpoint`."""

    path: Path
    metadata: dict[str, Any]
    files: dict[str, str]  # every file that determines the embeddings -> SHA-256
    digest: str  # SHA-256 of the canonical JSON of ``files``
    hidden_size: int
    max_seq_length: int
    vocab_size: int
    max_position_embeddings: int
    pooling: str
    input_format: ModelFormat
    run_record_sha256: str | None = None  # the run's ``run.json`` beside the checkpoint, if present

    @property
    def model_id(self) -> str:
        """``local:<run>/<checkpoint directory>``: the readable half of the identity (the digest is the other)."""
        return f"local:{self.metadata['run']}/{self.path.name}"

    @property
    def revision(self) -> str:
        return f"sha256:{self.digest}"

    def provenance(self) -> dict[str, Any]:
        """What a run on this checkpoint must record: the files, the training run and what it was trained on."""
        m = self.metadata
        inputs = m.get("inputs") or {}
        preparation = inputs.get("preparation") or {}
        return {
            "path": str(self.path),
            "run": m["run"],
            "step": m["step"],
            "total_steps": m["total_steps"],
            "epoch": m.get("epoch"),
            "digest": self.digest,
            "files": dict(self.files),
            "run_record": None
            if self.run_record_sha256 is None
            else {"path": str(self.path.parent / RUN_FILE), "sha256": self.run_record_sha256},
            "base_model": m["base_model"],
            "encoding": m["encoding"],
            "trained_on": {
                "corpus": inputs.get("corpus"),
                "preparation": {k: preparation.get(k) for k in ("version", "manifest_sha256", "data_prep_config")},
                "candidates": inputs.get("candidates"),
                "pairs": inputs.get("pairs"),
            },
            "config_sha256": m.get("config_sha256"),
            "schedule_sha256": m.get("schedule_sha256"),
            "selection": m.get("selection"),
            "notice": m.get("notice"),
            "trainer_environment": m.get("environment"),
            "trainer_state_present_not_loaded": (self.path / TRAINER_STATE_FILE).exists(),
        }


def inspect_checkpoint(path: str | Path) -> LocalCheckpoint:
    """Check a trainer checkpoint directory and hash its files; nothing is imported or loaded."""
    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise CheckpointError(f"{directory} is not a checkpoint directory")
    if missing := [name for name in REQUIRED_FILES if not (directory / name).is_file()]:
        raise CheckpointError(f"{directory} lacks {missing}; it is not a complete trainer checkpoint")
    if pickled := [name for name in PICKLED_WEIGHTS if (directory / name).exists()]:
        raise CheckpointError(f"{directory} holds pickled weights {pickled}; only safetensors weights are loaded")
    present = [*REQUIRED_FILES, *(n for n in OPTIONAL_FILES if (directory / n).is_file())]
    if linked := [name for name in present if (directory / name).is_symlink()]:
        raise CheckpointError(f"{directory}: {linked} are symbolic links; a checkpoint's files must be its own")

    metadata = _read_json(directory / METADATA_FILE)
    where = f"{directory / METADATA_FILE}"
    if not isinstance(metadata, dict) or (metadata.get("kind"), metadata.get("format_version")) != (
        CHECKPOINT_KIND,
        CHECKPOINT_FORMAT,
    ):
        raise CheckpointError(f"{where} is not a format-{CHECKPOINT_FORMAT} trainer checkpoint record")
    if absent := [k for k in ("run", "step", "total_steps", "base_model", "encoding", "model") if k not in metadata]:
        raise CheckpointError(f"{where} lacks {absent}")
    base, encoding = metadata["base_model"], metadata["encoding"]
    known = MODEL_FORMATS.get(str(base.get("name")))
    if known is None:
        raise CheckpointError(f"{where}: base model {base.get('name')!r} has no built-in input format")
    for field in ("query_prefix", "passage_prefix"):
        expected = getattr(known, field)
        if encoding.get(field) != expected or base.get(field) != expected:
            raise CheckpointError(
                f"{where}: trained with {field} {encoding.get(field)!r}, but {base['name']} expects {expected!r}"
            )
    pooling = encoding.get("pooling")
    if pooling not in POOLING_MODES or base.get("pooling") != pooling or encoding.get("normalize") is not True:
        raise CheckpointError(f"{where}: pooling {pooling!r} with normalisation is not a trainer setting")

    config = _read_json(directory / "config.json")
    if "auto_map" in config:
        raise CheckpointError(f"{directory / 'config.json'} names custom model code (auto_map), which is never run")
    hidden, positions, vocab = (config.get(k) for k in ("hidden_size", "max_position_embeddings", "vocab_size"))
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (hidden, positions, vocab)):
        raise CheckpointError(f"{directory / 'config.json'} lacks hidden_size, max_position_embeddings or vocab_size")
    if metadata["model"].get("hidden_size") != hidden:
        raise CheckpointError(f"{where} records hidden size {metadata['model'].get('hidden_size')}; config.json {hidden}")
    modules = _read_json(directory / "modules.json")
    listed = [(m.get("path"), m.get("type")) for m in modules] if isinstance(modules, list) else None
    if listed != list(MODULES):
        raise CheckpointError(
            f"{directory / 'modules.json'} lists {listed}; a trainer checkpoint has exactly Transformer, Pooling and "
            "Normalize"
        )
    pool = _read_json(directory / "1_Pooling" / "config.json")
    modes = sorted(k for k, v in pool.items() if k.startswith("pooling_mode_") and v is True)
    if pool.get("word_embedding_dimension") != hidden or modes != [POOLING_MODES[pooling]]:
        raise CheckpointError(
            f"{directory / '1_Pooling' / 'config.json'} pools {modes} over {pool.get('word_embedding_dimension')} "
            f"dimensions; the checkpoint was trained with {pooling} pooling over {hidden}"
        )
    sbert = _read_json(directory / "sentence_bert_config.json")
    limit = sbert.get("max_seq_length")
    if not isinstance(limit, int) or not 0 < limit <= positions or limit != encoding.get("passage_max_length"):
        raise CheckpointError(
            f"{directory / 'sentence_bert_config.json'}: max_seq_length {limit!r} must be the training passage length "
            f"{encoding.get('passage_max_length')!r} and fit the {positions} position embeddings"
        )
    if sbert.get("do_lower_case") is not False:
        raise CheckpointError(f"{directory / 'sentence_bert_config.json'} lower-cases inputs; training did not")
    if _read_json(directory / "config_sentence_transformers.json").get("prompts"):
        raise CheckpointError(f"{directory} defines prompts; evaluation applies the prefixes itself")

    files = {name: _sha256_file(directory / name) for name in present}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    run_sha = None
    run_file = directory.parent / RUN_FILE
    if run_file.is_file():
        record = _read_json(run_file)
        if (record.get("run"), record.get("config_sha256")) != (metadata["run"], metadata.get("config_sha256")):
            raise CheckpointError(f"{run_file} records another run than {where}")
        run_sha = _sha256_file(run_file)
    return LocalCheckpoint(
        path=directory,
        metadata=metadata,
        files=files,
        digest=digest,
        hidden_size=hidden,
        max_seq_length=limit,
        vocab_size=vocab,
        max_position_embeddings=positions,
        pooling=pooling,
        input_format=known,
        run_record_sha256=run_sha,
    )


def checkpoint_format(
    checkpoint: LocalCheckpoint, query_prefix: str | None = None, passage_prefix: str | None = None
) -> ModelFormat:
    """The input format the checkpoint was trained with; configured prefixes must agree with it."""
    known = checkpoint.input_format
    for field, configured in (("query_prefix", query_prefix), ("passage_prefix", passage_prefix)):
        if configured is not None and configured != getattr(known, field):
            raise ValueError(
                f"{field} {configured!r} does not match the {getattr(known, field)!r} checkpoint "
                f"{checkpoint.model_id} was trained with"
            )
    return ModelFormat(
        known.query_prefix,
        known.passage_prefix,
        source=f"{known.source} The checkpoint was trained with the same prefixes (its metadata.json).",
    )


def tokenizer_sha256(tokenizer: Any) -> str | None:
    """The tokenizer's identity as the trainer records it (``biencoder_trainer._tokenizer_sha256``): the serialised
    fast tokenizer without truncation or padding; ``None`` for a tokenizer without a fast backend."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    backend = copy.deepcopy(backend)
    backend.no_truncation()
    backend.no_padding()
    return hashlib.sha256(backend.to_str().encode("utf-8")).hexdigest()


def loaded_model_problems(checkpoint: LocalCheckpoint, model: Any) -> list[str]:
    """Every way a loaded model is not the checkpoint as recorded (see the module docstring)."""
    problems = []
    names = [type(module).__name__ for module in model]
    if names != ["Transformer", "Pooling", "Normalize"]:
        problems.append(f"loaded modules {names}, not Transformer, Pooling, Normalize")
    tokenizer = model.tokenizer
    found, recorded = tokenizer_sha256(tokenizer), checkpoint.metadata["model"].get("tokenizer_sha256")
    if recorded is None or found != recorded:
        problems.append(f"the tokenizer hashes to {found}, not the {recorded} the checkpoint was trained with")
    preparation = ((checkpoint.metadata.get("inputs") or {}).get("preparation") or {}).get("visible_text") or {}
    if preparation.get("tokenizer_sha256") not in (None, recorded):
        problems.append("the checkpoint's tokenizer differs from the tokenizer its training data was measured with")
    if len(tokenizer) > checkpoint.vocab_size:
        problems.append(f"the tokenizer has {len(tokenizer)} tokens, the model {checkpoint.vocab_size} embedding rows")
    if int(getattr(tokenizer, "model_max_length", 0)) < model.max_seq_length:
        problems.append(f"the tokenizer allows {tokenizer.model_max_length} tokens, fewer than {model.max_seq_length}")
    if model.max_seq_length > checkpoint.max_position_embeddings:
        problems.append(f"max_seq_length {model.max_seq_length} exceeds {checkpoint.max_position_embeddings} positions")
    if model.get_sentence_embedding_dimension() != checkpoint.hidden_size:
        problems.append(
            f"the model embeds in {model.get_sentence_embedding_dimension()} dimensions, not {checkpoint.hidden_size}"
        )
    if _sha256_file(checkpoint.path / WEIGHTS_FILE) != checkpoint.files[WEIGHTS_FILE]:
        problems.append(f"{WEIGHTS_FILE} changed while it was being loaded")
    return problems


def _import_modules() -> Any:
    from sentence_transformers import models

    return models


def load_checkpoint_encoder(
    checkpoint: LocalCheckpoint, *, device: str = "auto", max_seq_length: int | None = None
) -> SentenceTransformerEncoder:
    """Build the checkpoint's three modules from its files and check the result; see the module docstring.

    A missing backend or a load failure is a :class:`DenseModelError`; a model that loads but is not the
    checkpoint as recorded is a :class:`CheckpointError`.
    """
    try:
        torch, sentence_transformer = biencoder._import_backend()
        models = _import_modules()
    except Exception as exc:  # ImportError, or a broken binary dependency failing at import
        raise DenseModelError(
            f"cannot load {checkpoint.path}: importing torch / sentence-transformers failed "
            f"({type(exc).__name__}: {exc}). Dense retrieval needs the full requirements.txt."
        ) from exc
    resolved = resolve_device(torch, device)
    limit = checkpoint.max_seq_length if max_seq_length is None else max_seq_length
    if limit > checkpoint.max_position_embeddings:
        raise CheckpointError(f"max_seq_length {limit} exceeds the {checkpoint.max_position_embeddings} positions")
    local = {"local_files_only": True, "trust_remote_code": False}
    try:
        transformer = models.Transformer(
            str(checkpoint.path),
            max_seq_length=limit,
            model_args={**local, "use_safetensors": True},
            tokenizer_args=dict(local),
            config_args=dict(local),
            do_lower_case=False,
        )
        pooling = models.Pooling(checkpoint.hidden_size, pooling_mode=checkpoint.pooling)
        model = sentence_transformer(modules=[transformer, pooling, models.Normalize()], device=resolved)
    except Exception as exc:
        raise DenseModelError(
            f"could not load checkpoint {checkpoint.path} ({type(exc).__name__}: {exc})"
        ) from exc
    if problems := loaded_model_problems(checkpoint, model):
        raise CheckpointError(f"checkpoint {checkpoint.path}: {'; '.join(problems)}")
    return SentenceTransformerEncoder(
        model,
        model_name=checkpoint.model_id,
        revision=checkpoint.revision,
        device=resolved,
        torch_threads=torch.get_num_threads(),
    )


class CheckpointRetriever(BiEncoderRetriever):
    """A dense retriever over a local trainer checkpoint, scored exactly like the zero-shot baselines.

    The embedding cache is keyed by the checkpoint's identity (``local:<run>/<checkpoint>`` and the
    digest of its files), so another checkpoint, or the same directory with other weights, never
    reuses these embeddings.
    """

    def __init__(
        self,
        checkpoint: str | Path | LocalCheckpoint,
        *,
        retriever_name: str | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        batch_size: int = 32,
        device: str = "auto",
        max_seq_length: int | None = None,
        embedding_dim: int | None = None,
        cache_dir: str | Path | None = None,
        corpus_sha256: str | None = None,
        encoder: TextEncoder | None = None,
        selection: Mapping[str, Any] | None = None,
        role: str = "diagnostic",
        passage_view: str = "text",
    ) -> None:
        self.checkpoint = checkpoint if isinstance(checkpoint, LocalCheckpoint) else inspect_checkpoint(checkpoint)
        self.selection = None if selection is None else dict(selection)  # the verified validation selection, if any
        self.role = role
        if embedding_dim is not None and embedding_dim != self.checkpoint.hidden_size:
            raise ValueError(
                f"embedding_dim {embedding_dim} does not match the checkpoint's {self.checkpoint.hidden_size} dimensions"
            )
        if max_seq_length is not None and max_seq_length > self.checkpoint.max_position_embeddings:
            raise ValueError(
                f"max_seq_length {max_seq_length} exceeds the checkpoint's "
                f"{self.checkpoint.max_position_embeddings} position embeddings"
            )
        self._configure(
            self.checkpoint.model_id,
            self.checkpoint.revision,
            checkpoint_format(self.checkpoint, query_prefix, passage_prefix),
            retriever_name=retriever_name or self.checkpoint.model_id,
            batch_size=batch_size,
            device=device,
            max_seq_length=max_seq_length,
            embedding_dim=self.checkpoint.hidden_size,
            local_files_only=True,
            cache_dir=cache_dir,
            corpus_sha256=corpus_sha256,
            encoder=encoder,
            passage_view=passage_view,
        )

    def _load_encoder(self) -> TextEncoder:
        return load_checkpoint_encoder(self.checkpoint, device=self.device, max_seq_length=self.max_seq_length)

    def describe(self) -> dict[str, Any]:
        described = {**super().describe(), "source": "local_checkpoint", "checkpoint": self.checkpoint.provenance()}
        if self.selection is not None or self.role != "diagnostic":
            described.update(role=self.role, validation_selection=self.selection)
        return described


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_KIND",
    "MODULES",
    "PICKLED_WEIGHTS",
    "REQUIRED_FILES",
    "CheckpointError",
    "CheckpointRetriever",
    "LocalCheckpoint",
    "checkpoint_format",
    "inspect_checkpoint",
    "load_checkpoint_encoder",
    "loaded_model_problems",
    "tokenizer_sha256",
]
