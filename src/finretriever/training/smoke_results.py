"""Results of a trainer smoke run, collected after it ends; nothing is trained or written to a model.

``collect_results`` reads the run directory (``run.json``, ``log.jsonl``, every checkpoint's
``metadata.json``) and, optionally, the captured console log, and reports:

* the run: name, seed, base model and revision, configuration, candidate and schedule bindings;
* the steps planned and completed, and the checkpoints expected and found;
* the training loss of the first and last step (and its range), and any non-finite loss or
  gradient norm;
* the peak GPU memory the trainer measured (``torch.cuda.max_memory_allocated``: tensors only, not
  the CUDA context), and the training and wall-clock durations;
* checkpoint verification: each checkpoint's files are present and match its step; its model
  reloads (Hugging Face, and sentence-transformers when installed) and encodes fixed probe texts
  to finite vectors; the newest one's resumable state loads and matches the schedule; and the
  final weights differ from the base model's (the optimizer moved them);
* the console log: every line that is not the trainer's own report, a progress bar or blank is
  reported, by category (tokenizer, model load, CUDA or memory, numerical, error, warning, other).
  Transformers' model LOAD REPORT is parsed; only keys listed in ``BENIGN_LOAD_KEYS`` count as known
  benign, and every other line needs review. (A keyword search missed transformers' "Token indices
  sequence length is longer than the specified maximum sequence length" warning, which names no
  warning, error or CUDA.)

For a run with validation, the checkpoints expected are the evaluated ones its ``selection.json`` lists
(early stopping may end the run before the planned last step), the record is re-read (the rule
re-derives its selection) and every evaluated checkpoint is re-hashed against it, and the summary
reports the selected checkpoint. The validation set is synthetic and AI-reviewed (not gold): it
chooses among the run's checkpoints only.

It is a mechanics check. No retrieval metric is computed, and the summary says so.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

STATEMENT = (
    "Mechanics check only: no retrieval evaluation was run and no retrieval improvement is claimed. The training "
    "data are 48 synthetic queries accepted by an AI quality review (not human-validated, not gold labels), and no "
    "checkpoint was selected by any metric."
)
PROBES = (
    "Represent this sentence for searching relevant passages: What was the company's net income for fiscal 2020?",
    "Represent this sentence for searching relevant passages: How much cash was used in financing activities?",
    "Net income | Years ended September 26, 2020: 57,411 | Years ended September 28, 2019: 55,256",
    "The Company's investment policy and strategy are focused on preservation of capital and support of liquidity.",
)
VALIDATED_STATEMENT = (
    "Mechanics and selection check only: no retrieval evaluation was run and no retrieval improvement is claimed. "
    "The training data are synthetic queries accepted by an AI quality review (not human-validated, not gold labels); "
    "the checkpoint was selected only by the fixed rule on the synthetic, AI-reviewed validation set (not gold)."
)
CHECKPOINT_FILES = ("metadata.json", "config.json", "model.safetensors", "tokenizer.json", "modules.json",
                    "sentence_bert_config.json", "1_Pooling/config.json")
BENIGN_LOAD_KEYS = {
    ("embeddings.position_ids", "UNEXPECTED"): "BERT's position-id buffer: stored in the checkpoint, rebuilt by the "
    "model class; not a weight",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PROGRESS_RE = re.compile(r"\d+%\|.*\|\s*\d+/\d+")  # tqdm bars, e.g. "Loading weights: 100%|###| 199/199"
_TRAINER_RE = re.compile(r"^\[05_train_biencoder\] (?!error:|refused:)")  # the trainer's own report
_LOAD_REPORT_RE = re.compile(r"load report", re.IGNORECASE)
_LOAD_KEY_RE = re.compile(r"^(\S+)\s*\|\s*(UNEXPECTED|MISSING|MISMATCH)\b")
_REPORT_LINE_RE = re.compile(r"\||^-[-+]*$|^notes:|^- ", re.IGNORECASE)  # a load report's table, rule and notes
# A library line's category is the first that matches, else "other"; every library line is reported.
_CATEGORIES = (
    ("tokenizer", re.compile(r"token indices sequence length|maximum sequence length|indexing errors|truncat|"
                             r"overflowing tokens|tokeniz", re.IGNORECASE)),
    ("model_load", re.compile(r"load report|\b(?:unexpected|missing|mismatch(?:ed)?)\b|initiali[sz]ed|"
                              r"were not used|should probably train", re.IGNORECASE)),
    ("cuda_or_memory", re.compile(r"cuda|cublas|cudnn|nccl|out of memory|outofmemory|device-side assert",
                                  re.IGNORECASE)),
    ("numerical", re.compile(r"\bnan\b|\binf\b|overflow|non-finite|skipping step", re.IGNORECASE)),
    ("error", re.compile(r"traceback|error|exception|refused", re.IGNORECASE)),
    ("warning", re.compile(r"warn|deprecat", re.IGNORECASE)),
)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _console_lines(lines: list[str]) -> tuple[list[tuple[str, str, int | None]], list[list[tuple[str, str]]]]:
    """Library lines as (text, category, load report index or None), and each load report's (key, status)."""
    items: list[tuple[str, str, int | None]] = []
    reports: list[list[tuple[str, str]]] = []
    in_report = False
    for line in lines:
        text = _ANSI_RE.sub("", line).strip()
        if not text or _PROGRESS_RE.search(text) or _TRAINER_RE.match(text):
            continue
        if _LOAD_REPORT_RE.search(text):
            reports.append([])
            in_report = True
        elif in_report and not _REPORT_LINE_RE.search(text):
            in_report = False
        if in_report:
            if key := _LOAD_KEY_RE.match(text):
                reports[-1].append((key[1], key[2]))
            items.append((text, "model_load", len(reports) - 1))
        else:
            items.append((text, next((name for name, rx in _CATEGORIES if rx.search(text)), "other"), None))
    return items, reports


def console_warnings(path: Path | None, limit: int = 40) -> dict[str, Any]:
    """Every library line of the training console, by category; see the module docstring."""
    if path is None:
        return {"log": None, "lines": [], "note": "no console log given"}
    if not path.is_file():
        return {"log": str(path), "lines": [], "note": "console log not found"}
    items, reports = _console_lines(path.read_text(encoding="utf-8", errors="replace").splitlines())
    benign = {i for i, keys in enumerate(reports) if keys and all(k in BENIGN_LOAD_KEYS for k in keys)}
    seen: dict[str, dict[str, Any]] = {}
    for text, category, report in items:
        entry = seen.setdefault(text[:300], {"line": text[:300], "category": category, "count": 0,
                                             "known_benign": True})
        entry["count"] += 1
        entry["known_benign"] = entry["known_benign"] and report in benign
    review = [e for e in seen.values() if not e["known_benign"]]
    by_category: dict[str, int] = {}
    for e in review:
        by_category[e["category"]] = by_category.get(e["category"], 0) + 1
    return {
        "log": str(path),
        "distinct": len(seen),
        "needs_review": len(review),
        "needs_review_by_category": by_category,
        "known_benign": len(seen) - len(review),
        "lines": (review + [e for e in seen.values() if e["known_benign"]])[:limit],
        "omitted": max(0, len(seen) - limit),
        "load_reports": [[{"key": k, "status": s, "known_benign": BENIGN_LOAD_KEYS.get((k, s))} for k, s in keys]
                         for keys in reports],
    }


def _encode_probe(model_dir: Path, pooling: str) -> dict[str, Any]:
    import torch

    from finretriever.training.biencoder_trainer import HFEncoder

    encoder = HFEncoder.load(str(model_dir), revision=None, pooling=pooling, local_files_only=True)
    encoder.eval()
    with torch.no_grad():
        vectors = torch.nn.functional.normalize(encoder.encode(list(PROBES), 512).float(), dim=-1)
    out: dict[str, Any] = {"finite": bool(torch.isfinite(vectors).all()), "dimension": int(vectors.shape[1])}
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # optional
        out["sentence_transformers"] = f"not checked ({type(exc).__name__})"
        return out | {"vectors": vectors}
    loaded = SentenceTransformer(str(model_dir), device="cpu")
    st = torch.as_tensor(loaded.encode(list(PROBES), convert_to_numpy=True, normalize_embeddings=False))
    out["sentence_transformers_max_abs_diff"] = float((st - vectors).abs().max())
    out["sentence_transformers"] = "loads" if out["sentence_transformers_max_abs_diff"] < 1e-4 else "differs"
    return out | {"vectors": vectors}


def _weights_changed(final_dir: Path, base: dict[str, Any], *, offline: bool) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    base_file = hf_hub_download(base["name"], "model.safetensors", revision=base["revision"],
                                local_files_only=offline)
    before, after = load_file(base_file), load_file(str(final_dir / "model.safetensors"))
    shared = sorted(set(before) & set(after))
    changed = [k for k in shared if not before[k].float().equal(after[k].float())]
    delta = sum(float((after[k].float() - before[k].float()).abs().sum()) for k in shared)
    count = sum(after[k].numel() for k in shared)
    return {"tensors": len(shared), "tensors_changed": len(changed), "mean_abs_change": delta / max(1, count),
            "missing_from_checkpoint": sorted(set(before) - set(after))[:5]}


def collect_results(
    run_dir: Path,
    *,
    console_log: Path | None = None,
    verify_checkpoints: bool = True,
    compare_base: bool = True,
    offline: bool = True,
) -> dict[str, Any]:
    """Everything the smoke-test summary reports; see the module docstring."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    history = [json.loads(line) for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines() if line]
    config = run["config"]
    planned = run["schedule"]["steps"]
    every = config["checkpoints"].get("save_every_steps")
    expected_steps = sorted({s for s in range(1, planned + 1) if every and s % every == 0} | {planned})
    checkpoints = sorted(p for p in run_dir.glob("checkpoint-step-*") if p.is_dir())
    found_steps = [int(p.name.rsplit("-", 1)[1]) for p in checkpoints]
    selection = _selection(run_dir) if "validation" in config else None
    if selection is not None:
        expected_steps = selection.pop("evaluated_steps")
    losses = [h["loss"] for h in history]
    norms = [h["grad_norm"] for h in history]
    newest_meta = json.loads((checkpoints[-1] / "metadata.json").read_text(encoding="utf-8")) if checkpoints else {}
    started = (run_dir / "run.json").stat().st_mtime
    ended = (checkpoints[-1] / "metadata.json").stat().st_mtime if checkpoints else started
    result: dict[str, Any] = {
        "kind": "finretriever_smoke_results",
        "statement": STATEMENT if selection is None else VALIDATED_STATEMENT,
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": {
            "name": run["run"],
            "seed": config["training"]["seed"],
            "base_model": config["base_model"]["name"],
            "revision": config["base_model"]["revision"],
            "config_sha256": run["config_sha256"],
            "candidates": run["inputs"]["candidates"]["version"],
            "candidates_manifest_sha256": run["inputs"]["candidates"]["manifest_sha256"],
            "pairs": run["inputs"]["pairs"]["count"],
            "schedule_sha256": run["schedule"]["sha256"],
            "batch_size": config["training"]["batch_size"],
            "fp16": config["training"]["fp16"],
            "gradient_checkpointing": config["training"]["gradient_checkpointing"],
        },
        "steps": {"planned": planned, "completed": len(history),
                  "complete": len(history) == planned if selection is None else selection.get("ended", False)
                  and selection["stopped"]["at_step"] == len(history),
                  "checkpoints_expected": expected_steps, "checkpoints_found": found_steps},
        "loss": {
            "initial": losses[0] if losses else None,
            "final": losses[-1] if losses else None,
            "min": min(losses) if losses else None,
            "max": max(losses) if losses else None,
            "non_finite": sum(not _finite(x) for x in losses),
            "per_step": losses,
            "note": "training loss on each step's own batch (in-batch InfoNCE); not a retrieval metric",
        },
        "gradients": {"non_finite_norms": sum(not _finite(x) for x in norms),
                      "max_norm": max((x for x in norms if _finite(x)), default=None)},
        "gpu_memory": newest_meta.get("gpu_memory"),
        "gpu_memory_note": "measured by torch.cuda.max_memory_allocated over the run (tensors only, without the "
        "CUDA context or allocator cache); null when not trained on CUDA",
        "duration": {"training_steps_seconds": round(sum(h["seconds"] for h in history), 1),
                     "wall_seconds_run_start_to_last_checkpoint": round(ended - started, 1)},
        "environment": newest_meta.get("environment"),
        "warnings": console_warnings(console_log),
    }
    checks = []
    for path, step in zip(checkpoints, found_steps, strict=True):
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        item: dict[str, Any] = {
            "checkpoint": path.name,
            "missing_files": [f for f in CHECKPOINT_FILES if not (path / f).is_file()],
            "metadata_step_matches": meta.get("step") == step,
            "resumable_state": (path / "trainer_state.pt").is_file(),
        }
        if verify_checkpoints and not item["missing_files"]:
            try:
                probe = _encode_probe(path, config["base_model"]["pooling"])
                item["reload"] = {k: v for k, v in probe.items() if k != "vectors"}
            except Exception as exc:
                item["reload"] = {"error": f"{type(exc).__name__}: {exc}"}
        checks.append(item)
    if checkpoints and (checkpoints[-1] / "trainer_state.pt").is_file():
        import torch

        state = torch.load(checkpoints[-1] / "trainer_state.pt", map_location="cpu", weights_only=True)
        result["resume_state"] = {"step": state["step"], "schedule_matches": state["schedule_sha256"]
                                  == run["schedule"]["sha256"], "older_checkpoints_hold_state":
                                  any(c["resumable_state"] for c in checks[:-1])}
    result["checkpoints"] = checks
    if selection is not None:
        result["selection"] = selection
    if compare_base and checkpoints and (checkpoints[-1] / "model.safetensors").is_file():
        try:
            result["weights_vs_base"] = _weights_changed(checkpoints[-1], config["base_model"], offline=offline)
        except Exception as exc:
            result["weights_vs_base"] = {"error": f"{type(exc).__name__}: {exc}"}
    result["mechanics_ok"] = bool(
        result["steps"]["complete"]
        and (found_steps == expected_steps if selection is None else set(expected_steps) <= set(found_steps))
        and (selection is None or selection.get("hashes_verified") is True)
        and result["loss"]["non_finite"] == 0
        and result["gradients"]["non_finite_norms"] == 0
        and all(not c["missing_files"] and c["metadata_step_matches"] for c in checks)
        and all(c.get("reload", {}).get("finite", not verify_checkpoints) for c in checks)
        and result.get("resume_state", {}).get("schedule_matches", False)
    )
    return result


def _selection(run_dir: Path) -> dict[str, Any]:
    """The selection record of a validated run, re-read and re-hashed; or, before it ends, what it has evaluated."""
    from finretriever.training.checkpoint_selection import (
        SELECTION_FILE,
        SelectionError,
        checkpoint_files,
        read_selection_record,
        verify_selected_checkpoint,
    )

    path = run_dir / SELECTION_FILE
    log = run_dir / "validation_log.jsonl"
    if not path.is_file():
        evaluated = [json.loads(line)["step"] for line in log.read_text(encoding="utf-8").splitlines() if line] \
            if log.is_file() else []
        return {"ended": False, "evaluated_steps": evaluated, "note": f"no {SELECTION_FILE}: the run has not ended"}
    try:
        record = read_selection_record(path)
        verify_selected_checkpoint(path, run_dir / record["selected"]["checkpoint"])
        everything = [*record["evaluations"], *record["after_stop_evaluations"]]
        changed = [e["checkpoint"] for e in everything if checkpoint_files(run_dir / e["checkpoint"]) != e["files"]]
    except (SelectionError, OSError, KeyError) as exc:
        return {"ended": True, "evaluated_steps": [], "hashes_verified": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ended": True,
        "evaluated_steps": [e["step"] for e in everything],
        "evaluations": [{"step": e["step"], "candidate": e["candidate"], "mrr@10_global": e["value"],
                         "after_stop": e.get("after_stop", False)} for e in everything],
        "selected": record["selected"],
        "rule_stop": record["rule_stop"],
        "stopped": record["stopped"],
        "environment": record["environment"],
        "validation": record["validation"].get("version"),
        "hashes_verified": not changed,
        "changed_checkpoints": changed,
        "note": "selected only by the fixed rule on the synthetic validation set (AI-reviewed, not gold)",
    }


def _fmt(x: Any, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def summary_markdown(result: dict[str, Any]) -> str:
    run, steps, loss, memory = result["run"], result["steps"], result["loss"], result.get("gpu_memory") or {}
    reloads = []
    for c in result.get("checkpoints", []):
        r = c.get("reload", {})
        state = "missing " + ", ".join(c["missing_files"]) if c["missing_files"] else (
            "reloads, finite" if r.get("finite") else r.get("error", "not reloaded"))
        if "sentence_transformers" in r:
            state += f"; sentence-transformers {r['sentence_transformers']}"
        reloads.append(f"{c['checkpoint']}: {state}")
    base = result.get("weights_vs_base", {})
    changed = (f"{base['tensors_changed']} of {base['tensors']} tensors changed (mean |change| "
               f"{base['mean_abs_change']:.2e})" if "tensors" in base else base.get("error", "not compared"))
    warnings = result["warnings"]
    if "note" in warnings:
        warn_text = warnings["note"]
    elif warnings["needs_review"]:
        kinds = ", ".join(f"{k} {n}" for k, n in warnings["needs_review_by_category"].items())
        warn_text = f"{warnings['needs_review']} distinct lines need review ({kinds}); listed below"
    else:
        warn_text = f"none need review ({warnings['known_benign']} known-benign load-report lines)"
    load_keys = [k for report in warnings.get("load_reports", []) for k in report]
    load_text = "; ".join(f"{k['key']} {k['status']}" + (" (known benign)" if k["known_benign"] else " (review)")
                          for k in load_keys) or ("no load report in the log" if "note" not in warnings else "n/a")
    resume = result.get("resume_state", {})
    rows = [
        ("Run", run["name"]),
        ("Seed", run["seed"]),
        ("Base model", f"{run['base_model']} @ {run['revision']}"),
        ("Training data", f"{run['pairs']} candidates of {run['candidates']} (AI-reviewed; not gold)"),
        ("Batch size, fp16, gradient checkpointing", f"{run['batch_size']}, {run['fp16']}, "
         f"{run['gradient_checkpointing']}"),
        ("Steps completed", f"{steps['completed']} of {steps['planned']}"),
        ("Training loss, first step -> last step", f"{_fmt(loss['initial'])} -> {_fmt(loss['final'])} "
         f"(range {_fmt(loss['min'])}-{_fmt(loss['max'])})"),
        ("Non-finite losses / gradient norms", f"{loss['non_finite']} / {result['gradients']['non_finite_norms']}"),
        ("Peak GPU memory (measured)", f"{memory.get('max_allocated_gib', 'n/a')} GiB on "
         f"{memory.get('device_name', 'n/a')}"),
        ("Duration: training steps / wall clock", f"{result['duration']['training_steps_seconds']} s / "
         f"{result['duration']['wall_seconds_run_start_to_last_checkpoint']} s"),
        ("Checkpoints expected / found", f"{steps['checkpoints_expected']} / {steps['checkpoints_found']}"),
        ("Checkpoint reload", "; ".join(reloads) or "none"),
        ("Resumable state", f"step {resume.get('step')}, schedule matches: {resume.get('schedule_matches')}"),
        ("Final weights vs base", changed),
        ("Console: tokenizer, model and other library output", warn_text),
        ("Model load report keys", load_text),
        ("Mechanics OK", result["mechanics_ok"]),
    ]
    chosen = result.get("selection")
    if chosen is not None:
        picked = chosen.get("selected") or {}
        rows[-1:-1] = [
            ("Validation evaluations (whole-corpus MRR@10)", "; ".join(
                f"step {e['step']}{' (after the rule stopped, never selectable)' if e.get('after_stop') else '' if e['candidate'] else ' (final, reported only)'}: {e['mrr@10_global']:.4f}"
                for e in chosen.get("evaluations", [])) or chosen.get("note") or chosen.get("error")),
            ("Selected checkpoint", f"{picked.get('checkpoint')} (step {picked.get('step')}"
             f"{', the base model' if picked.get('base_model') else ''}); digest {str(picked.get('digest'))[:16]}"
             if picked else "none yet"),
            ("Rule stopped selecting", f"at step {chosen['rule_stop']['step']}; training continued to step "
             f"{chosen['stopped']['at_step']} of {chosen['stopped']['planned_steps']}" if chosen.get("rule_stop")
             else f"never; trained to step {chosen['stopped']['at_step']} of {chosen['stopped']['planned_steps']}"
             if "stopped" in chosen else "n/a"),
            ("Environment", ", ".join(f"{k} {v}" for k, v in (chosen.get("environment") or {}).items()
                                      if k in ("gpu_name", "gpu_capability", "torch", "transformers", "cuda"))
             or "n/a"),
            ("Checkpoint hashes match the selection record", chosen.get("hashes_verified", "n/a")),
        ]
    title = "Trainer smoke test" if chosen is None else "Trainer run"
    lines = [f"# {title}: {run['name']}", "", result["statement"], "", "| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {str(v).replace('|', '/')} |" for k, v in rows]
    review = [w for w in warnings.get("lines", []) if not w["known_benign"]]
    if review:
        lines += ["", "Console lines needing review:", ""]
        lines += [f"- {w['category']}: `{w['line']}` ({w['count']}x)" for w in review]
        if warnings.get("omitted"):
            lines.append(f"- … and {warnings['omitted']} more distinct lines (see the console log)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="smoke_results.py", description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="models/<output>/<name>-seed<seed>")
    parser.add_argument("--console-log", help="The captured stdout and stderr of the training command.")
    parser.add_argument("--out-dir", default="smoke_outputs", help="Where results.json and results.md go.")
    parser.add_argument("--no-reload", action="store_true", help="Skip reloading the checkpoints.")
    parser.add_argument("--no-base-compare", action="store_true", help="Skip comparing with the base weights.")
    parser.add_argument("--online", action="store_true", help="Allow downloading the base weights to compare.")
    args = parser.parse_args(argv)
    result = collect_results(Path(args.run_dir), console_log=Path(args.console_log) if args.console_log else None,
                             verify_checkpoints=not args.no_reload, compare_base=not args.no_base_compare,
                             offline=not args.online)
    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (target / "results.md").write_text(summary_markdown(result), encoding="utf-8")
    print(summary_markdown(result), file=out)
    print(f"wrote {target / 'results.json'} and {target / 'results.md'}", file=out)
    return 0 if result["mechanics_ok"] else 1


__all__ = [
    "BENIGN_LOAD_KEYS",
    "PROBES",
    "STATEMENT",
    "VALIDATED_STATEMENT",
    "collect_results",
    "console_warnings",
    "main",
    "summary_markdown",
]
