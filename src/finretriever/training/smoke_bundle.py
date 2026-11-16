"""A portable bundle for running the trainer elsewhere (a Kaggle T4), and its checks: the smoke test, or a
fine-tuning run whose checkpoints its validation set selects (``finretriever_training_bundle``).

**What goes in (``discover``).** Dependencies are found by doing what the run does: the trainer's
input loading and schedule (``trainer_setup.prepare_inputs``, ``plan_steps``) and the preflight's
provenance check (``candidates.verify_candidate_provenance``) run under a spy on ``open``, and
every repository file they read is bundled, together with the ``finretriever`` modules they import
(and those the trainer, preflight, GPU rehearsal and results collector import), the scripts that run them, the
configurations, the Kaggle requirements and the guide. Nothing else is bundled.

**What never goes in.** Evaluation data (anything under ``data/eval``, or named as an evaluation
source by the data-preparation config), raw SEC downloads, caches, models, results, ``.env`` files
and keys: such a path stops the build, and so does any file that looks like it holds a credential.
Every bundled path must be relative to the repository and stay inside it.

**The bundle (``build_bundle``).** ``<name>/`` with the files at their repository paths,
``BUNDLE_MANIFEST.json`` (every file's source path, role, size and SHA-256; the corpus, preparation,
candidate, model and library versions; the values the preflight must find again) and ``SHA256SUMS``
(``sha256sum -c`` format), packed as a deterministic ``<name>.tar.gz`` (sorted entries, fixed
times and owners), with its SHA-256 beside it. Sources are only read, and each copy is checked
against its source.

**Checking a copy (``verify_bundle``).** Every listed file has its recorded SHA-256, no unlisted
file is present (runtime outputs aside), and no evaluation data is.

**Training runs.** A config with a validation set reads that set (``validation_sets/`` of the
preparation, with the generation records, plan, stop record and decision files its manifest binds),
so discovery bundles it; it is training-side data, never evaluation data. A config listing seeds is
traced for each seed, and the bundle records each seed's run and schedule.
"""

from __future__ import annotations

import builtins
import gzip
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from finretriever.evaluation.provenance import sha256_file
from finretriever.training.data_prep import load_data_prep_config

FORMAT_VERSION = 1
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
SUMS_NAME = "SHA256SUMS"
DEFAULT_NAME = "finretriever_smoke_v1"
KINDS = {"smoke": "finretriever_smoke_bundle", "training": "finretriever_training_bundle"}
REQUIREMENTS = "requirements-kaggle.txt"
GUIDE = "docs/kaggle_smoke_test.md"
TRAINING_GUIDE = "docs/kaggle_training.md"
SCRIPTS = ("scripts/05_train_biencoder.py", "scripts/smoke_preflight.py", "scripts/smoke_results.py",
           "scripts/gpu_rehearsal.py")
# The modules the run imports beyond input loading (the trainer needs torch).
RUNTIME_MODULES = (
    "finretriever.training.biencoder_trainer",
    "finretriever.training.validation_eval",
    "finretriever.training.checkpoint_selection",
    "finretriever.training.smoke_preflight",
    "finretriever.training.smoke_results",
    "finretriever.training.gpu_rehearsal",
)
# Written by the run inside the bundle directory; never part of the bundle.
RUNTIME_OUTPUTS = ("models", "smoke_outputs", "rehearsal_outputs")
_IGNORED_NAMES = ("__pycache__", ".ipynb_checkpoints")
_FORBIDDEN_PREFIXES = (
    ("data", "eval"),
    ("data", "raw"),
    ("data", "interim"),
    ("data", "cache"),
    ("results",),
    ("models",),
    ("dist",),
    (".git",),
)
_FORBIDDEN_NAME_RE = re.compile(
    r"(^\.env|\.pem$|\.key$|^id_rsa|credential|secret|token\.json$|\.netrc$|kaggle\.json$)", re.I
)
_SECRET_RES = (
    re.compile(r"gsk_[A-Za-z0-9_]{20,}"),  # Groq
    re.compile(r"hf_[A-Za-z0-9]{30,}"),  # Hugging Face
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{20,}"),  # OpenAI / Anthropic style
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_FIXED_MTIME = 1767225600  # 2026-01-01T00:00:00Z: every archive entry gets the same time


class BundleError(ValueError):
    """A bundle cannot be built safely, or a copy does not match its manifest."""


def _parts(path: str) -> tuple[str, ...]:
    return Path(path).parts


def under(path: str, prefix: tuple[str, ...]) -> bool:
    parts = _parts(path)
    return parts[: len(prefix)] == prefix


def forbidden_reason(path: str) -> str | None:
    """Why a repository-relative path may never be bundled, or ``None``."""
    if Path(path).is_absolute() or ".." in _parts(path):
        return "not a path inside the repository"
    for prefix in _FORBIDDEN_PREFIXES:
        if under(path, prefix):
            return f"under {'/'.join(prefix)}/"
    if any(_FORBIDDEN_NAME_RE.search(part) for part in _parts(path)):
        return "looks like a credential or environment file"
    return None


def secret_found(data: bytes) -> str | None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    for pattern in _SECRET_RES:
        if pattern.search(text):
            return pattern.pattern
    return None


def evaluation_sources(data_prep_config: str) -> list[str]:
    """Every evaluation path the data-preparation config names (blocklist and leakage sources)."""
    config, _ = load_data_prep_config(data_prep_config)
    b = config.blocklist.model_dump()
    out: list[str] = []
    for value in b.values():
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            out.extend(str(v) for v in value)
    out.extend(config.leakage.evaluation_question_files)
    return sorted({p for p in out if p})


@contextmanager
def trace_opened(root: Path) -> Iterator[set[str]]:
    """Repository-relative paths of the files opened under ``root`` while the block runs."""
    root = root.resolve()
    opened: set[str] = set()
    real = io.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            path = Path(os.path.realpath(file))
            if path.is_relative_to(root) and path.is_file():
                opened.add(path.relative_to(root).as_posix())
        return real(file, *args, **kwargs)

    saved = builtins.open, io.open
    builtins.open = io.open = spy  # type: ignore[assignment]
    try:
        yield opened
    finally:
        builtins.open, io.open = saved  # type: ignore[assignment]


def code_root() -> Path:
    """The repository holding the ``finretriever`` package (``<root>/src/finretriever``)."""
    import finretriever

    return Path(finretriever.__file__).resolve().parents[2]


@dataclass(frozen=True)
class Discovered:
    data: tuple[str, ...]  # repository-relative files the run reads (configs and data)
    code: tuple[str, ...]  # ``src/finretriever/...`` modules the run imports
    expected: dict[str, Any]  # what the preflight must find again
    evaluation_sources: tuple[str, ...]


def discover(config_path: str, *, root: Path, extra_configs: Sequence[str] = ()) -> Discovered:
    """The files a smoke run reads and the modules it imports, by running its loading steps.

    ``root`` is the repository (the working directory of the run); paths in the configs are
    relative to it. ``extra_configs`` are alternative trainer configs (e.g. with gradient
    checkpointing) whose inputs must be the same files.
    """
    from finretriever.training.candidates import verify_candidate_provenance
    from finretriever.training.trainer_setup import (
        estimate_gpu_memory_gib,
        load_trainer_config,
        plan_steps,
        prepare_inputs,
    )

    for name in RUNTIME_MODULES:
        importlib.import_module(name)
    expected: dict[str, Any] = {"configs": {}}
    data: set[str] = set()
    for path in (config_path, *extra_configs):
        with trace_opened(root) as opened:
            config, config_sha256 = load_trainer_config(path)
            inputs = prepare_inputs(config.for_seed(config.seed_list[0]), config_sha256)
            schedule = plan_steps(inputs)
            provenance = verify_candidate_provenance(inputs.prepared, config.data.candidates)
        data |= opened
        cfg = config.training
        every = config.checkpoints.save_every_steps
        steps = len(schedule.steps)
        validated = config.validation is not None
        entry: dict[str, Any] = {
            "sha256": config_sha256,
            "run": inputs.config.run_name,
            "pairs": len(inputs.pairs),
            "pair_ids_sha256": inputs.identity["pairs"]["pair_ids_sha256"],
            "steps": steps,
            "warmup_steps": schedule.warmup_steps,
            "schedule_sha256": schedule.sha256,
            "checkpoint_steps": checkpoint_steps(steps, validated=validated, every=every),
            "output_dir": config.checkpoints.output_dir,
            "estimated_gpu_memory_gib": estimate_gpu_memory_gib(
                batch_size=cfg.mini_batch_size or cfg.batch_size,
                query_tokens=cfg.query_max_length,
                passage_tokens=cfg.passage_max_length,
                fp16=cfg.fp16,
                gradient_checkpointing=cfg.gradient_checkpointing,
            ),
        }
        if cfg.seeds is not None:
            entry["seeds"] = {}
            for seed in config.seed_list:
                run = plan_steps(inputs.for_seed(seed))
                entry["seeds"][str(seed)] = {
                    "run": config.for_seed(seed).run_name, "steps": len(run.steps), "warmup_steps": run.warmup_steps,
                    "schedule_sha256": run.sha256,
                    "checkpoint_steps": checkpoint_steps(len(run.steps), validated=validated, every=every),
                }
        if cfg.mini_batch_size is not None:
            entry["mini_batch_size"] = cfg.mini_batch_size
        if config.validation is not None:
            validation = inputs.identity["validation"]
            entry["validation"] = {k: validation[k] for k in ("version", "manifest_sha256", "pairs", "pair_ids_sha256")}
        expected["configs"][path] = entry
        identity = inputs.identity
        expected.update(
            {
                "base_model": identity["base_model"],
                "corpus": identity["corpus"],
                "preparation": {
                    k: identity["preparation"][k]
                    for k in ("version", "manifest_sha256", "evidence_index_digest", "visible_text")
                },
                "candidates": {**identity["candidates"], "provenance": provenance},
            }
        )
    prep_config = expected_prep_config(data)
    sources = evaluation_sources(prep_config)
    package = code_root()
    code = sorted(
        Path(module.__file__).resolve().relative_to(package).as_posix()
        for name, module in list(sys.modules.items())
        if name.split(".")[0] == "finretriever" and getattr(module, "__file__", None)
    )
    data = {p for p in data if not p.startswith("src/") and p != MANIFEST_NAME}
    return Discovered(tuple(sorted(data)), tuple(code), expected, tuple(sources))


def checkpoint_steps(total: int, *, validated: bool, every: int | None) -> list[int]:
    """The checkpoints a run of ``total`` steps writes: every evaluated step with validation (early stopping may end
    the run sooner), else every ``every`` steps and the last."""
    if validated:
        from finretriever.training.checkpoint_selection import evaluation_steps

        return evaluation_steps(total)
    return sorted({s for s in range(1, total + 1) if every and s % every == 0} | {total})


def expected_prep_config(files: set[str]) -> str:
    found = sorted(p for p in files if p.startswith("configs/train/data_prep_"))
    if len(found) != 1:
        raise BundleError(f"expected one data-preparation config among the files read, found {found}")
    return found[0]


def _role(path: str, corpus_files: set[str]) -> str:
    if path in corpus_files:
        return "corpus"
    if path.startswith("src/"):
        return "code"
    if path.startswith("scripts/"):
        return "script"
    if path.startswith("configs/"):
        return "config"
    if path.startswith("docs/") or path == REQUIREMENTS:
        return "instructions"
    if path.startswith("data/train_review/"):
        return "quality_decisions"
    if path.startswith("data/validation_review/"):
        return "validation_decisions"
    if "/validation_sets/" in path:
        return "validation_set"
    if "/plans/" in path:
        return "sampling_plan"
    if "/candidates/" in path:
        return "training_candidates"
    if "/generation/" in path:
        return "generation_records"
    if "/pairs/" in path:
        return "source_pairs"
    if path.startswith("data/processed/train/"):
        return "preparation"
    return "other"


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub",
                 "sentence-transformers", "numpy", "pydantic", "PyYAML"):
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _tar(staging: Path, name: str, archive: Path) -> None:
    """A deterministic ``.tar.gz``: sorted entries, fixed times, owners and modes."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        entries = sorted(staging.rglob("*"))
        for directory in sorted({p.parent for p in entries if p.is_file()} | {staging}):
            rel = directory.relative_to(staging).as_posix()
            info = tarfile.TarInfo(name if rel == "." else f"{name}/{rel}")
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, _FIXED_MTIME
            tar.addfile(info)
        for path in entries:
            if not path.is_file():
                continue
            info = tarfile.TarInfo(f"{name}/{path.relative_to(staging).as_posix()}")
            data = path.read_bytes()
            info.size, info.mode, info.mtime = len(data), 0o644, _FIXED_MTIME
            tar.addfile(info, io.BytesIO(data))
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(buffer.getvalue())


def build_bundle(
    config_path: str,
    *,
    root: Path,
    output_dir: Path,
    name: str = DEFAULT_NAME,
    extra_configs: Sequence[str] = (),
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build ``<output_dir>/<name>.tar.gz`` (and its ``.sha256``); see the module docstring.

    ``root`` is the repository the data and configs are read from; code and scripts come from the
    repository holding the ``finretriever`` package (the same one, except in tests).
    """
    root = root.resolve()
    found = discover(config_path, root=root, extra_configs=extra_configs)
    training = any("validation" in c for c in found.expected["configs"].values())
    guide = TRAINING_GUIDE if training else GUIDE
    corpus = Path(load_data_prep_config(root / expected_prep_config(set(found.data)))[0].corpus_path)
    corpus_files = {corpus.as_posix(), corpus.with_suffix(".manifest.json").as_posix()}
    package = code_root()
    files: dict[str, Path] = {}
    for rel in found.data:
        files[rel] = root / rel
    for rel in (*found.code, *SCRIPTS, REQUIREMENTS, guide):
        files[rel] = package / rel
    for rel in (config_path, *extra_configs):
        files[Path(rel).as_posix()] = root / rel
    problems = []
    for rel, source in sorted(files.items()):
        if reason := forbidden_reason(rel):
            problems.append(f"{rel}: {reason}")
        elif not source.is_file() or source.is_symlink():
            problems.append(f"{rel}: missing or not a regular file")
        elif any(rel == s or under(rel, _parts(s)) for s in found.evaluation_sources):
            problems.append(f"{rel}: an evaluation source of the data-preparation config")
        elif secret := secret_found(source.read_bytes()):
            problems.append(f"{rel}: holds something that looks like a credential ({secret})")
    if problems:
        raise BundleError(f"{len(problems)} files may not be bundled: {problems[:5]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{name}.tar.gz"
    entries = []
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        staging = Path(tmp) / name
        for rel, source in sorted(files.items()):
            digest = sha256_file(source)
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256_file(target) != digest or sha256_file(source) != digest:
                raise BundleError(f"{rel} changed while it was copied")
            entries.append(
                {"path": rel, "source": rel, "role": _role(rel, corpus_files), "bytes": target.stat().st_size,
                 "sha256": digest}
            )
        head = _git(package, "rev-parse", "HEAD")
        manifest = {
            "kind": KINDS["training" if training else "smoke"],
            "format_version": FORMAT_VERSION,
            "name": name,
            "purpose": (
                "Fine-tuning the BGE bi-encoder on one GPU, one seed per run; checkpoints are chosen only by the "
                "fixed rule on the synthetic validation set; no evaluation data, no retrieval claim."
                if training else
                "Mechanical smoke test of the BGE contrastive trainer on one GPU: no evaluation, no checkpoint "
                "selection, no retrieval claim."
            ),
            "notice": "Training data are synthetic queries accepted by an AI quality review: not human-validated, "
            "not gold labels. No evaluation data is included."
            + (" The validation questions are synthetic too, accepted by an AI review: not gold labels, used only "
               "to choose among a run's checkpoints." if training else ""),
            "source": {
                "repository": "FinRetriever",
                "git_head": head,
                "git_uncommitted_changes": bool(_git(package, "status", "--porcelain")),
                "note": "files are bundled from the working tree; each one's SHA-256 identifies it",
            },
            "run_from": f"the {name}/ directory (paths in the configs are relative to it)",
            "trainer_configs": sorted({Path(p).as_posix() for p in (config_path, *extra_configs)}),
            "default_config": Path(config_path).as_posix(),
            "expected": found.expected,
            "excluded": {
                "evaluation_data": "data/eval/ and every evaluation source the data-preparation config names "
                f"({len(found.evaluation_sources)} paths): not bundled, and the preflight checks they are absent",
                "never_bundled": ["raw SEC downloads (data/raw)", "caches (data/cache)", "models", "results",
                                  ".env files and credentials", "generation runs not named by a pairs manifest"],
            },
            "evaluation_sources": list(found.evaluation_sources),
            "runtime_outputs": list(RUNTIME_OUTPUTS),
            "built_with": _versions(),
            "files": entries,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        (staging / MANIFEST_NAME).write_bytes(manifest_bytes)
        sums = sorted(
            [(e["sha256"], e["path"]) for e in entries]
            + [(hashlib.sha256(manifest_bytes).hexdigest(), MANIFEST_NAME)],
            key=lambda x: x[1],
        )
        (staging / SUMS_NAME).write_text("".join(f"{d}  {p}\n" for d, p in sums), encoding="utf-8")
        report = verify_bundle(staging)  # the staged copy must already pass its own check
        new = Path(tmp) / archive.name
        _tar(staging, name, new)
        if archive.exists() and archive.read_bytes() != new.read_bytes():
            raise BundleError(f"{archive} exists with different content; use a new bundle name")
        os.replace(new, archive)
    digest = sha256_file(archive)
    (output_dir / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    say(f"bundle {archive} ({archive.stat().st_size / 1e6:.1f} MB, SHA-256 {digest})")
    say(f"{len(entries)} files: {report['by_role']}")
    return {"archive": str(archive), "sha256": digest, "manifest": manifest, "report": report}


def verify_bundle(root: Path) -> dict[str, Any]:
    """Check an extracted bundle directory against its manifest; raises ``BundleError``."""
    root = Path(root)
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise BundleError(f"no {MANIFEST_NAME} in {root}: not a bundle directory")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") not in KINDS.values() or manifest.get("format_version") != FORMAT_VERSION:
        raise BundleError(f"{path} is not a format-{FORMAT_VERSION} trainer bundle manifest")
    problems = []
    listed = {e["path"]: e for e in manifest["files"]}
    for rel, entry in sorted(listed.items()):
        if reason := forbidden_reason(rel):
            problems.append(f"{rel}: listed but {reason}")
        file = root / rel
        if not file.is_file():
            problems.append(f"{rel}: missing")
        elif sha256_file(file) != entry["sha256"]:
            problems.append(f"{rel}: SHA-256 differs from the manifest")
    sums_path = root / SUMS_NAME
    if not sums_path.is_file():
        problems.append(f"{SUMS_NAME}: missing")
    else:
        sums = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            digest, _, rel = line.partition("  ")
            sums[rel] = digest
        expected = {rel: e["sha256"] for rel, e in listed.items()}
        expected[MANIFEST_NAME] = sha256_file(path)
        if sums != expected:
            problems.append(f"{SUMS_NAME} does not match the manifest")
    present = set()
    for file in root.rglob("*"):
        rel = file.relative_to(root).as_posix()
        if any(part in _IGNORED_NAMES for part in file.relative_to(root).parts):
            continue
        if rel.split("/")[0] in RUNTIME_OUTPUTS:
            continue
        if file.is_file():
            present.add(rel)
    unexpected = sorted(present - set(listed) - {MANIFEST_NAME, SUMS_NAME})
    if unexpected:
        problems.append(f"files not in the manifest: {unexpected[:5]}")
    evaluation = [p for p in ("data/eval", *manifest.get("evaluation_sources", [])) if (root / p).exists()]
    if evaluation:
        problems.append(f"evaluation data present: {evaluation[:5]}")
    if problems:
        raise BundleError(f"{len(problems)} problems in {root}: {problems[:8]}")
    roles: dict[str, int] = {}
    for entry in manifest["files"]:
        roles[entry["role"]] = roles.get(entry["role"], 0) + 1
    return {"files": len(listed), "by_role": dict(sorted(roles.items())), "manifest_sha256": sha256_file(path)}


def extract_bundle(archive: Path, destination: Path) -> Path:
    """Extract ``archive`` into ``destination`` (regular files and directories only)."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not (member.isfile() or member.isdir()) or not target.is_relative_to(destination.resolve()):
                raise BundleError(f"{archive}: unsafe entry {member.name}")
        tar.extractall(destination, members=members, filter="data")
    names = {m.name.split("/")[0] for m in members}
    if len(names) != 1:
        raise BundleError(f"{archive} must hold one top-level directory, found {sorted(names)}")
    return destination / names.pop()


__all__ = [
    "DEFAULT_NAME",
    "FORMAT_VERSION",
    "KINDS",
    "MANIFEST_NAME",
    "SUMS_NAME",
    "BundleError",
    "Discovered",
    "build_bundle",
    "discover",
    "evaluation_sources",
    "extract_bundle",
    "forbidden_reason",
    "secret_found",
    "trace_opened",
    "verify_bundle",
]
