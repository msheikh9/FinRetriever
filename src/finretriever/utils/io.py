"""I/O helpers for FinRetriever.

The pipeline reads and writes two file formats:

* **JSONL** — one record per line, used for the corpus, training pairs, the
  gold eval set, retriever runs, and benchmark outputs.
* **YAML** — plain configuration files under ``configs/``.

This module centralises both so the rest of the codebase never touches
``json`` / ``orjson`` / ``yaml`` directly. ``orjson`` is preferred when
available because it is materially faster on the corpus-sized JSONL we
expect to handle; otherwise we fall back to the stdlib ``json`` module
without any change in behaviour.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

try:  # pragma: no cover - exercised by import side effects
    import orjson  # type: ignore[import-not-found]

    _HAS_ORJSON = True
except ImportError:  # pragma: no cover
    orjson = None  # type: ignore[assignment]
    _HAS_ORJSON = False


T = TypeVar("T", bound=BaseModel)


def _json_loads(s: str) -> Any:
    if _HAS_ORJSON:
        return orjson.loads(s)
    return json.loads(s)


def _json_dumps_bytes(obj: Any) -> bytes:
    if _HAS_ORJSON:
        return orjson.dumps(obj)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory of ``path`` if needed. Returns ``Path(path)``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def iter_jsonl(
    path: str | Path,
    model: type[T] | None = None,
) -> Iterator[T] | Iterator[dict[str, Any]]:
    """Yield records from a JSONL file one at a time.

    Blank lines (whitespace-only) are skipped. If ``model`` is provided, each
    record is validated through ``model.model_validate`` and the model
    instance is yielded; otherwise the raw decoded dict is yielded.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = _json_loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Failed to parse JSON at {p}:{line_no}: {exc}") from exc
            if model is None:
                yield obj
                continue
            try:
                yield model.model_validate(obj)
            except ValidationError as exc:
                raise ValueError(
                    f"Failed to validate row against {model.__name__} at {p}:{line_no}: {exc}"
                ) from exc


def read_jsonl(
    path: str | Path,
    model: type[T] | None = None,
) -> list[T] | list[dict[str, Any]]:
    """Eagerly read a JSONL file into a list (see ``iter_jsonl`` for semantics)."""
    return list(iter_jsonl(path, model=model))


def write_jsonl(
    path: str | Path,
    rows: Iterable[BaseModel | dict[str, Any]],
    atomic: bool = True,
) -> None:
    """Write Pydantic models or dicts to a JSONL file.

    Parent directories are created automatically. When ``atomic=True`` (the
    default) the file is written to a sibling temp file and ``os.replace``d
    into place, so a crash mid-write never leaves a half-written file at
    ``path``.
    """
    p = ensure_parent_dir(path)
    if atomic:
        fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                _write_rows(fh, rows)
            os.replace(tmp_path, p)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        with p.open("wb") as fh:
            _write_rows(fh, rows)


def _write_rows(fh: Any, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    for row in rows:
        if isinstance(row, BaseModel):
            payload: Any = row.model_dump(mode="json")
        elif isinstance(row, dict):
            payload = row
        else:
            raise TypeError(
                f"write_jsonl expected BaseModel or dict, got {type(row).__name__}"
            )
        fh.write(_json_dumps_bytes(payload))
        fh.write(b"\n")


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Empty files return an empty dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root at {p} must be a mapping, got {type(data).__name__}"
        )
    return data


def load_config(
    path: str | Path,
    schema: type[BaseModel] | None = None,
) -> BaseModel | dict[str, Any]:
    """Load a YAML config and optionally validate it against a Pydantic model."""
    data = read_yaml(path)
    if schema is None:
        return data
    return schema.model_validate(data)


__all__ = [
    "ensure_parent_dir",
    "iter_jsonl",
    "load_config",
    "read_jsonl",
    "read_yaml",
    "write_jsonl",
]
