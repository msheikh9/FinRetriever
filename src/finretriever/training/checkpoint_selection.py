"""Checkpoint selection on the validation set: the fixed rule, its running state and the write-once record.

**The rule** (``RULE``; fixed in code, not configurable, so no run can choose its own):

* the validation set is evaluated at step 0 (the base model, before any update) and after every 5th optimizer
  step (5, 10, 15, ...): these are the selection candidates;
* when the last step of a run is not a multiple of 5 it is evaluated too and reported, but it is never a
  candidate;
* the selection metric is MRR@10 over the whole corpus (``global`` mode); the company-restricted metrics are
  reported only;
* the highest value wins, and a tie (values equal to 12 decimal places) keeps the earlier step;
* selection stops (early stopping) after 3 candidate evaluations in a row without improvement over the best so
  far: no later checkpoint is selectable;
* after the stop, training and evaluation continue to the planned last step: those evaluations are recorded
  separately, marked not selectable, and never change the selection. They give the decision rule
  (``decision_rule``) every planned evaluation, which its selection-aware test needs.

Step 0 can win: then the base model is the selected checkpoint (``checkpoint-step-000000``, the base weights
written by the trainer before its first update), and the record says so.

**Hashes.** Each evaluated checkpoint is hashed when it is written (``checkpoint_files``: every file of the
directory except ``trainer_state.pt``, which only the newest checkpoint keeps), and the record lists every
evaluation with its checkpoint's file hashes and digest. ``verify_selected_checkpoint`` re-hashes a checkpoint
directory and checks it against the record, and re-derives the selection from the recorded values, so an edited
record or checkpoint is refused.

**The record.** ``selection.json`` in the run directory, written once (atomically) when the run reaches its planned
last step. ``evaluations`` holds the rule's evaluations up to its stop, ``after_stop_evaluations`` the rest; together
they must be every planned evaluation step (``evaluation_steps``), and reading the record checks that, re-derives the
selection and checks that the after-stop evaluations could not have changed it. The record also carries the run's
hardware and software ``environment``. An identical record is not written again; a different one is refused.

The validation questions are synthetic, accepted by an AI review: not human-validated, not gold labels. They choose
among the checkpoints of one run and are never evaluation data.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SELECTION_KIND = "finretriever_checkpoint_selection"
SELECTION_FORMAT = 2  # 2: after-stop evaluations, completeness, environment
SELECTION_FILE = "selection.json"
METRIC = "mrr@10"
MODE = "global"
EVERY_STEPS = 5
PATIENCE = 3
TIE_DECIMALS = 12
UNHASHED = frozenset({"trainer_state.pt"})  # optimizer state, removed from all but the newest checkpoint
RULE = {
    "metric": METRIC,
    "mode": MODE,
    "candidates": f"step 0 (the base model) and every {EVERY_STEPS} optimizer steps",
    "final_step": f"evaluated and reported when it is not a multiple of {EVERY_STEPS}; never a candidate",
    "ties": f"the earlier step (values compared to {TIE_DECIMALS} decimal places)",
    "early_stopping": f"selection stops after {PATIENCE} candidate evaluations in a row without improvement; no later "
    "checkpoint is selectable",
    "after_stop": "training and evaluation continue to the planned last step; those evaluations are recorded as not "
    "selectable and never change the selection",
}
# The hardware and software every seed of one experiment must share (the trainer records them with
# ``biencoder_trainer.run_environment``; ``decision_rule`` compares them across the seeds).
ENVIRONMENT_IDENTITY = ("python", "torch", "transformers", "tokenizers", "cuda", "cudnn", "device_type", "gpu_name",
                        "gpu_capability")
SELECTION_NOTICE = (
    "Chosen by the fixed rule on synthetic validation questions accepted by an AI review (not human-validated, not "
    "gold labels), only among this run's checkpoints. No development or evaluation metric was computed or consulted."
)


class SelectionError(ValueError):
    """A selection record or a selected checkpoint cannot be trusted."""


def is_candidate_step(step: int) -> bool:
    return step >= 0 and step % EVERY_STEPS == 0


def evaluation_steps(total_steps: int) -> list[int]:
    """The steps a run of ``total_steps`` evaluates (early stopping may end it sooner)."""
    return sorted({0, *range(EVERY_STEPS, total_steps + 1, EVERY_STEPS), total_steps})


def _key(value: float) -> float:
    return round(float(value), TIE_DECIMALS)


@dataclass
class SelectionState:
    """The evaluations so far and the rule's bookkeeping; saved with the trainer state so a resumed run continues it."""

    evaluations: list[dict[str, Any]] = field(default_factory=list)
    best_step: int | None = None
    best_value: float | None = None
    since_best: int = 0
    after_stop: list[dict[str, Any]] = field(default_factory=list)  # evaluations after the stop: never selectable

    def record(self, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Add the evaluation at ``step`` (its ``metrics`` hold ``metrics[MODE][METRIC]``); returns its entry."""
        if self.evaluations and step <= self.evaluations[-1]["step"]:
            raise SelectionError(f"step {step} is not after the last evaluated step {self.evaluations[-1]['step']}")
        if self.should_stop:
            raise SelectionError(f"training stopped early at step {self.evaluations[-1]['step']}; no step follows it")
        value = float(metrics[MODE][METRIC])
        candidate = is_candidate_step(step)
        improved = None
        if candidate:
            improved = self.best_value is None or _key(value) > _key(self.best_value)
            if improved:
                self.best_step, self.best_value, self.since_best = step, value, 0
            else:
                self.since_best += 1
        entry = {"step": step, "candidate": candidate, "value": value, "improved": improved,
                 "since_best": self.since_best if candidate else None, "metrics": dict(metrics)}
        self.evaluations.append(entry)
        return entry

    def record_after_stop(self, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Add an evaluation made after the rule stopped: recorded, never a candidate, never selectable."""
        if not self.should_stop:
            raise SelectionError(f"the rule has not stopped: step {step} is one of its evaluations (use record)")
        last = (self.after_stop or self.evaluations)[-1]["step"]
        if step <= last:
            raise SelectionError(f"step {step} is not after the last evaluated step {last}")
        entry = {"step": step, "candidate": False, "selectable": False, "after_stop": True,
                 "value": float(metrics[MODE][METRIC]), "improved": None, "since_best": None, "metrics": dict(metrics)}
        self.after_stop.append(entry)
        return entry

    @property
    def should_stop(self) -> bool:
        return self.since_best >= PATIENCE

    def to_json(self) -> dict[str, Any]:
        return {"evaluations": self.evaluations, "best_step": self.best_step, "best_value": self.best_value,
                "since_best": self.since_best, "after_stop": self.after_stop}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SelectionState:
        return cls([dict(e) for e in data["evaluations"]], data["best_step"], data["best_value"], data["since_best"],
                   [dict(e) for e in data.get("after_stop", [])])


def select(evaluations: Sequence[Mapping[str, Any]]) -> tuple[int, bool]:
    """The step the rule selects from recorded evaluations (in step order), and whether it stopped early there."""
    state = SelectionState()
    for entry in evaluations:
        if state.should_stop:
            raise SelectionError(f"an evaluation at step {entry['step']} follows the early stop")
        state.record(int(entry["step"]), {MODE: {METRIC: entry["value"]}})
    if state.best_step is None:
        raise SelectionError("no candidate evaluation: nothing to select")
    return state.best_step, state.should_stop


def select_through(evaluations: Sequence[Mapping[str, Any]]) -> tuple[int, int | None]:
    """The rule applied to every planned evaluation of a run, in step order (``step`` and ``value`` each): the step it
    selects, and the step at which it stopped selecting (``None`` if it never stopped). Evaluations after the stop are
    not selectable and are ignored, as a run records them. The decision rule re-applies the rule this way."""
    state = SelectionState()
    for entry in evaluations:
        if state.should_stop:
            break
        state.record(int(entry["step"]), {MODE: {METRIC: entry["value"]}})
    if state.best_step is None:
        raise SelectionError("no candidate evaluation: nothing to select")
    return state.best_step, (state.evaluations[-1]["step"] if state.should_stop else None)


def checkpoint_files(directory: str | Path) -> dict[str, str]:
    """Every file of a checkpoint directory (relative POSIX path -> SHA-256), except ``trainer_state.pt``."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise SelectionError(f"{root} is not a checkpoint directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SelectionError(f"{path} is a symbolic link; a checkpoint's files must be its own")
        if path.is_file() and path.name not in UNHASHED:
            digest = hashlib.sha256()
            with path.open("rb") as fh:
                while block := fh.read(1 << 20):
                    digest.update(block)
            files[path.relative_to(root).as_posix()] = digest.hexdigest()
    if not files:
        raise SelectionError(f"{root} holds no files")
    return files


def files_digest(files: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(files), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_selection_record(
    *,
    run: str,
    seed: int,
    seeds: Sequence[int],
    config_sha256: str,
    schedule_sha256: str,
    run_record_sha256: str,
    validation: Mapping[str, Any],
    state: SelectionState,
    planned_steps: int,
    steps_done: int,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The record of a run that reached its planned last step; every evaluation, after-stop ones included, must carry
    its checkpoint's name, files and digest, and together they must be every planned evaluation step."""
    everything = [*state.evaluations, *state.after_stop]
    if missing := [e["step"] for e in everything if not all(k in e for k in ("checkpoint", "files", "digest"))]:
        raise SelectionError(f"evaluations at steps {missing} have no hashed checkpoint")
    if steps_done != planned_steps or [e["step"] for e in everything] != evaluation_steps(planned_steps):
        raise SelectionError(f"the run evaluated steps {[e['step'] for e in everything]} of {steps_done}, not every "
                             f"planned evaluation step {evaluation_steps(planned_steps)}")
    step, early = select(state.evaluations)
    by_step = {e["step"]: e for e in state.evaluations}
    chosen, stop, last = by_step[step], state.evaluations[-1], everything[-1]
    keys = ("step", "candidate", "value", "improved", "since_best", "checkpoint", "digest", "files", "metrics")
    return {
        "kind": SELECTION_KIND,
        "format_version": SELECTION_FORMAT,
        "notice": SELECTION_NOTICE,
        "run": run,
        "seed": seed,
        "seeds": list(seeds),
        "config_sha256": config_sha256,
        "schedule_sha256": schedule_sha256,
        "run_record": {"file": "run.json", "sha256": run_record_sha256},
        "rule": RULE,
        "validation": dict(validation),
        "evaluations": [{k: e[k] for k in keys} for e in state.evaluations],
        "after_stop_evaluations": [{**{k: e[k] for k in keys}, "selectable": False, "after_stop": True}
                                   for e in state.after_stop],
        "selected": {"step": step, "checkpoint": chosen["checkpoint"], "digest": chosen["digest"],
                     "value": chosen["value"], "base_model": step == 0},
        "rule_stop": None if not early else {"step": stop["step"], "checkpoint": stop["checkpoint"],
                                             "digest": stop["digest"], "value": stop["value"]},
        "last_evaluated": {"step": last["step"], "checkpoint": last["checkpoint"], "digest": last["digest"],
                           "value": last["value"], "candidate": last["candidate"]},
        "stopped": {"early": early, "at_step": steps_done, "planned_steps": planned_steps},
        "environment": None if environment is None else dict(environment),
    }


def write_selection_record(run_dir: str | Path, record: Mapping[str, Any]) -> bool:
    """Write ``selection.json`` once; ``False`` if an identical record is there, an error if a different one is."""
    path = Path(run_dir) / SELECTION_FILE
    data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() == data:
            return False
        raise SelectionError(f"{path} already records a different selection; it is never written over")
    tmp = path.with_name(f".{SELECTION_FILE}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def read_selection_record(path: str | Path) -> dict[str, Any]:
    """A selection record, checked: kind and format, and a selection the rule re-derives from its evaluations."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SelectionError(f"{path} is not a readable selection record ({exc})") from None
    if not isinstance(record, dict) or (record.get("kind"), record.get("format_version")) != (
        SELECTION_KIND,
        SELECTION_FORMAT,
    ):
        raise SelectionError(f"{path} is not a format-{SELECTION_FORMAT} selection record")
    if record.get("rule") != RULE:
        raise SelectionError(f"{path} was made under another selection rule")
    step, early = select(record["evaluations"])
    if (record["selected"]["step"], record["stopped"]["early"]) != (step, early):
        raise SelectionError(
            f"{path} names step {record['selected']['step']}, but the rule selects step {step} from its evaluations"
        )
    after = record["after_stop_evaluations"]
    if after and not early:
        raise SelectionError(f"{path}: evaluations are recorded after a stop the rule did not make")
    if any(e["candidate"] or e.get("selectable") is not False or e.get("after_stop") is not True for e in after):
        raise SelectionError(f"{path}: an evaluation after the stop is not marked as never selectable")
    everything = [*record["evaluations"], *after]
    planned, steps = record["stopped"]["planned_steps"], [e["step"] for e in everything]
    if record["stopped"]["at_step"] != planned or steps != evaluation_steps(planned):
        raise SelectionError(f"{path}: steps {steps} are not every planned evaluation step {evaluation_steps(planned)}")
    if select_through(everything) != (step, record["evaluations"][-1]["step"] if early else None):
        raise SelectionError(f"{path}: the rule over every evaluation does not stop and select where the record says")
    stop = record["rule_stop"]
    if (stop is not None) != early or (early and stop["step"] != record["evaluations"][-1]["step"]):
        raise SelectionError(f"{path}: the recorded rule stop does not match the evaluations")
    chosen = next(e for e in record["evaluations"] if e["step"] == step)
    for e in everything:
        if files_digest(e["files"]) != e["digest"]:
            raise SelectionError(f"{path}: the digest of step {e['step']} does not match its file hashes")
    if (record["selected"]["checkpoint"], record["selected"]["digest"]) != (chosen["checkpoint"], chosen["digest"]):
        raise SelectionError(f"{path}: the selected checkpoint is not the evaluated one at step {step}")
    return record


def verify_selected_checkpoint(record_path: str | Path, checkpoint_path: str | Path) -> dict[str, Any]:
    """Check that ``checkpoint_path`` is the checkpoint the run's selection record chose, unchanged; returns a
    summary for provenance. Raises ``SelectionError``."""
    record_path = Path(record_path)
    record = read_selection_record(record_path)
    run_dir = record_path.parent
    selected = record["selected"]
    expected = run_dir / selected["checkpoint"]
    if Path(checkpoint_path).resolve() != expected.resolve():
        raise SelectionError(
            f"{checkpoint_path} is not the checkpoint run {record['run']} selected ({expected}, step {selected['step']})"
        )
    run_file = run_dir / record["run_record"]["file"]
    if not run_file.is_file() or _sha256_file(run_file) != record["run_record"]["sha256"]:
        raise SelectionError(f"{run_file} is missing or changed since the selection was recorded")
    files = checkpoint_files(expected)
    chosen = next(e for e in record["evaluations"] if e["step"] == selected["step"])
    if files != chosen["files"] or files_digest(files) != selected["digest"]:
        changed = sorted(k for k in set(files) | set(chosen["files"]) if files.get(k) != chosen["files"].get(k))
        raise SelectionError(f"{expected} differs from the checkpoint the selection hashed ({changed[:5]})")
    metadata = json.loads((expected / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("run"), metadata.get("step")) != (record["run"], selected["step"]):
        raise SelectionError(f"{expected}/metadata.json names run {metadata.get('run')} step {metadata.get('step')}")
    return {
        "record": {"path": str(record_path), "sha256": _sha256_file(record_path)},
        "run": record["run"],
        "seed": record["seed"],
        "seeds": record["seeds"],
        "config_sha256": record["config_sha256"],
        "rule": record["rule"],
        "selected_step": selected["step"],
        "base_model": selected["base_model"],
        "digest": selected["digest"],
        "value": selected["value"],
        "validation": record["validation"],
        "stopped": record["stopped"],
        "notice": record["notice"],
    }


__all__ = [
    "ENVIRONMENT_IDENTITY",
    "EVERY_STEPS",
    "METRIC",
    "MODE",
    "PATIENCE",
    "RULE",
    "SELECTION_FILE",
    "SELECTION_KIND",
    "SELECTION_NOTICE",
    "SelectionError",
    "SelectionState",
    "checkpoint_files",
    "evaluation_steps",
    "files_digest",
    "is_candidate_step",
    "make_selection_record",
    "read_selection_record",
    "select",
    "select_through",
    "verify_selected_checkpoint",
    "write_selection_record",
]
