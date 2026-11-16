"""Gold set v1, the question stage: people write the 100 planned questions, they are checked for leakage, and the set
is frozen before any candidate is pooled (docs/gold_set_v1_protocol.md §2.1 and §7).

1. ``write_question_sheet`` writes a worksheet from the query plan (``configs/eval/gold_set_v1_query_plan.csv``). Each
   row is one information-need slot, with empty columns for the question, the writer and an AI-assistance attestation.
2. ``import_questions`` validates the filled sheet and adds each question to the annotation workspace, written
   ``independent`` by the named person and tagged with its slot. It refuses, before adding anything:
   - a writer who is a configured AI identity;
   - any AI assistance (``ai_assistance`` must be ``none``);
   - a slot that is not in the plan, or one whose companies or question type were changed;
   - other changes to the slot's fields without a ``replaced:`` note;
   - a slot already imported with different text.
3. ``leakage_record`` is written by ``leakage-check``. It records each question's closest training or validation
   question, the threshold and the sources' SHA-256.
4. ``export_questions`` writes the frozen question export once, and only if:
   - every slot has exactly one active, human-written question;
   - the leakage record is current and no question reaches the threshold.
5. ``check_poolable``: ``pool`` refuses a slot question that is not in the frozen export with the same text.
6. **Versions.** The exports are versioned and immutable.
   - Version 1 keeps its original names (``questions_frozen.jsonl``); later versions are ``questions_frozen_v<N>.jsonl``.
   - Each version archives the leakage record it was frozen with (``leakage_check_v<N>.json``), and is made read-only.
   - A new version is written only after every earlier one verifies against its manifest, and only if the questions
     differ from the latest version. It records what it supersedes and every revised slot.
   - ``pool`` uses the latest version.
7. **Revisions.** ``write_revision_sheet`` lists the slots to revise, with their current question and its writer.
   ``revise_questions`` validates the filled sheet, all or nothing, and refuses:
   - a writer who is not the original writer, or is an AI identity;
   - any AI assistance;
   - changed slot metadata;
   - a stale current question;
   - an unrevised, empty or duplicate text.

   For each slot it then withdraws the old question, with a note, and adds the revision as a new independent
   question for the same slot. The workspace history keeps both.

The question stage reads no pool and no judgment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finretriever.eval_set.annotation_store import AnnotationError, AnnotationStore, utc_now
from finretriever.eval_set.schemas import GoldQuestion, normalize_question

SHEET_FILE = "question_sheet.csv"
LEAKAGE_FILE = "leakage_check.json"
EXPORT_FILE = "questions_frozen.jsonl"  # version 1; later versions are questions_frozen_v<N>.jsonl
EXPORT_MANIFEST = "questions_frozen.manifest.json"
_VERSIONED_MANIFEST = re.compile(r"questions_frozen_v(\d+)\.manifest\.json")
PLAN_COLUMNS = ("slot", "tickers", "topic", "question_type", "comparison", "period", "fiscal_years")
SHEET_COLUMNS = (*PLAN_COLUMNS, "question", "writer", "ai_assistance", "notes")
REVISION_COLUMNS = (*PLAN_COLUMNS, "current_query_id", "current_question", "original_writer", "revision_reason",
                    "revised_question", "writer", "ai_assistance", "notes")
REPLACEABLE = ("topic", "comparison", "period", "fiscal_years")  # with a "replaced:" note; companies and type stay
NO_ASSISTANCE = "none"
_SLOT_TAG = re.compile(r"^\[slot (S\d{3})\]")


@dataclass(frozen=True)
class Slot:
    """One information need of the query plan."""

    slot: str
    tickers: tuple[str, ...]
    topic: str
    question_type: str
    comparison: str
    period: str
    fiscal_years: tuple[int, ...]

    def row(self) -> dict[str, str]:
        return {"slot": self.slot, "tickers": ",".join(self.tickers), "topic": self.topic,
                "question_type": self.question_type, "comparison": self.comparison, "period": self.period,
                "fiscal_years": ";".join(map(str, self.fiscal_years))}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _years(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(y) for y in re.split(r"[;,\s]+", text.strip()) if y}))


def load_plan(path: str | Path) -> dict[str, Slot]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    plan = {}
    for row in rows:
        plan[row["slot"]] = Slot(row["slot"], tuple(t for t in row["tickers"].split(",") if t), row["topic"],
                                 row["question_type"], row["comparison"], row["period"], _years(row["fiscal_years"]))
    if len(plan) != len(rows):
        raise AnnotationError(f"{path} names a slot twice")
    return plan


def write_question_sheet(plan: Mapping[str, Slot], path: str | Path) -> Path:
    """The writers' worksheet: one row per slot, with empty question, writer, ai_assistance and notes columns."""
    path = Path(path)
    if path.exists():
        raise AnnotationError(f"{path} exists; it may hold written questions and is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for slot in sorted(plan):
            writer.writerow({**plan[slot].row(), "question": "", "writer": "", "ai_assistance": "", "notes": ""})
    return path


def slot_of(question: GoldQuestion) -> str | None:
    """The slot a question was written for (its notes start with ``[slot Snnn]``), or ``None``."""
    match = _SLOT_TAG.match(question.notes or "")
    return match.group(1) if match else None


def import_questions(store: AnnotationStore, sheet: str | Path, plan: Mapping[str, Slot], *,
                     ai_identities: Iterable[str]) -> dict[str, list[str]]:
    """Validate every filled row of the sheet, then add the new ones (all or nothing); see the module docstring."""
    with Path(sheet).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if missing := [c for c in SHEET_COLUMNS if c not in (reader.fieldnames or [])]:
            raise AnnotationError(f"{sheet} lacks the columns {missing}")
        rows = list(reader)
    ai = {name.casefold() for name in ai_identities}
    existing = {slot_of(q): q for q in store.questions() if slot_of(q) and q.status != "withdrawn"}
    problems: list[str] = []
    to_add: list[tuple[Slot, dict[str, str]]] = []
    unchanged: list[str] = []
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        text = (row.get("question") or "").strip()
        if not text:
            continue
        slot_id = (row.get("slot") or "").strip()
        where = f"{sheet}:{line} ({slot_id or 'no slot'})"
        planned = plan.get(slot_id)
        if planned is None:
            problems.append(f"{where}: not a slot of the plan")
            continue
        if slot_id in seen:
            problems.append(f"{where}: the slot is filled twice in the sheet")
        seen.add(slot_id)
        writer = (row.get("writer") or "").strip()
        if not writer:
            problems.append(f"{where}: no writer")
        elif writer.casefold() in ai:
            problems.append(f"{where}: {writer!r} is an AI identity; gold questions are written by people")
        if (row.get("ai_assistance") or "").strip().casefold() != NO_ASSISTANCE:
            problems.append(f"{where}: ai_assistance must be {NO_ASSISTANCE!r}; AI-assisted questions are refused")
        notes = (row.get("notes") or "").strip()
        expected = planned.row()
        changed = [c for c in PLAN_COLUMNS if (row.get(c) or "").strip().replace(" ", "") != expected[c].replace(" ", "")]
        if fixed := [c for c in changed if c not in REPLACEABLE]:
            problems.append(f"{where}: {fixed} differ from the plan; a slot keeps its companies and question type")
        elif changed and "replaced:" not in notes:
            problems.append(f"{where}: {changed} differ from the plan without a 'replaced: <reason>' note")
        if slot_id in existing:
            if existing[slot_id].text == text:
                unchanged.append(existing[slot_id].query_id)
            else:
                problems.append(f"{where}: already imported as {existing[slot_id].query_id} with other text; withdraw "
                                "it first to rewrite the slot")
            continue
        to_add.append((planned, row))
    if problems:
        raise AnnotationError(f"{len(problems)} problem(s), nothing imported: " + "; ".join(problems))
    added = []
    for planned, row in to_add:
        question = store.add_question(
            row["question"].strip(),
            topic=row["topic"].strip(),
            question_type=planned.question_type,
            comparison=row["comparison"].strip(),
            tickers=planned.tickers,
            fiscal_years=_years(row["fiscal_years"]),
            authoring="independent",
            author=row["writer"].strip(),
            notes=f"[slot {planned.slot}] {(row.get('notes') or '').strip()}".strip(),
            ai_assisted=False,
        )
        added.append(question.query_id)
    return {"added": added, "unchanged": unchanged}


def questions_sha256(questions: Iterable[tuple[str, str]]) -> str:
    """The SHA-256 of a set of (query_id, text) pairs, sorted."""
    return hashlib.sha256(json.dumps(sorted(questions), ensure_ascii=False).encode()).hexdigest()


def leakage_record(checked: Sequence[tuple[str, str]], results: Mapping[str, tuple[float, str | None]], *,
                   threshold: float, sources: Mapping[str, str], against: int) -> dict[str, Any]:
    """What ``leakage-check`` records: each question's closest training or validation question."""
    return {
        "kind": "finretriever_gold_question_leakage_check",
        "checked_at": utc_now(),
        "questions_sha256": questions_sha256(checked),
        "questions": len(checked),
        "against_questions": against,
        "sources": dict(sources),
        "method": "word-set Jaccard (qrels_builder.find_gold_overlap); question texts only",
        "threshold": threshold,
        "results": {qid: {"similarity": sim, "closest": closest, "passes": sim < threshold}
                    for qid, (sim, closest) in sorted(results.items())},
        "all_pass": all(sim < threshold for sim, _ in results.values()),
    }


def write_leakage_record(directory: str | Path, record: Mapping[str, Any]) -> Path:
    path = Path(directory) / LEAKAGE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def export_paths(directory: str | Path, version: int) -> tuple[Path, Path, Path]:
    """The questions file, the manifest and the archived leakage record of one export version."""
    directory = Path(directory)
    if version == 1:
        return directory / EXPORT_FILE, directory / EXPORT_MANIFEST, directory / "leakage_check_v1.json"
    return (directory / f"questions_frozen_v{version}.jsonl", directory / f"questions_frozen_v{version}.manifest.json",
            directory / f"leakage_check_v{version}.json")


def export_versions(directory: str | Path) -> list[int]:
    """The export versions present, in order."""
    directory = Path(directory)
    versions = {1} if (directory / EXPORT_MANIFEST).is_file() else set()
    for path in directory.glob("questions_frozen_v*.manifest.json"):
        if match := _VERSIONED_MANIFEST.fullmatch(path.name):
            versions.add(int(match.group(1)))
    return sorted(versions)


def verify_export(directory: str | Path, version: int) -> dict[str, Any]:
    """One version's manifest, after checking its questions file and archived leakage record against it."""
    out, manifest_path, leakage = export_paths(directory, version)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not out.is_file() or _sha256(out) != manifest.get("sha256"):
        raise AnnotationError(f"question export version {version}: {out} is missing or differs from its manifest")
    if not leakage.is_file() or _sha256(leakage) != manifest.get("leakage_check_sha256"):
        raise AnnotationError(f"question export version {version}: its leakage record {leakage} is missing or changed")
    return manifest


def _read_export(directory: str | Path, version: int) -> list[dict[str, Any]]:
    out = export_paths(directory, version)[0]
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]


def export_questions(store: AnnotationStore, directory: str | Path, plan: Mapping[str, Slot], *,
                     created_by: str) -> dict[str, Any]:
    """Freeze the next question export version; see the module docstring for its preconditions."""
    directory = Path(directory)
    versions = export_versions(directory)
    for version in versions:  # every earlier version must still be exactly as frozen
        verify_export(directory, version)
    version = (versions[-1] + 1) if versions else 1
    out, manifest_path, leakage_archive = export_paths(directory, version)
    if out.exists() or manifest_path.exists():
        raise AnnotationError(f"{out} exists; a question export version is written once")
    active = [q for q in store.questions() if q.status != "withdrawn"]
    by_slot: dict[str, list[GoldQuestion]] = {}
    for q in active:
        if slot := slot_of(q):
            by_slot.setdefault(slot, []).append(q)
    problems = []
    if missing := sorted(set(plan) - set(by_slot)):
        problems.append(f"{len(missing)} slot(s) without a question: {missing[:10]}")
    if doubled := sorted(s for s, qs in by_slot.items() if len(qs) > 1):
        problems.append(f"slots with more than one active question: {doubled}")
    if unknown := sorted(set(by_slot) - set(plan)):
        problems.append(f"questions for slots not in the plan: {unknown}")
    slot_questions = [q for qs in by_slot.values() for q in qs]
    if assisted := sorted(q.query_id for q in slot_questions if q.ai_assisted or q.authoring != "independent"):
        problems.append(f"questions not written independently by a person: {assisted}")
    leakage_path = directory / LEAKAGE_FILE
    if not leakage_path.is_file():
        problems.append("no leakage check: run leakage-check first")
    else:
        record = json.loads(leakage_path.read_text(encoding="utf-8"))
        if record.get("questions_sha256") != questions_sha256((q.query_id, q.text) for q in active):
            problems.append("the leakage check is not current: questions changed since it ran; run it again")
        failing = sorted(qid for qid, r in (record.get("results") or {}).items()
                         if not r.get("passes") and qid in {q.query_id for q in slot_questions})
        if failing:
            problems.append(f"questions at or above the leakage threshold: {failing}; rewrite them")
    if problems:
        raise AnnotationError("the questions cannot be frozen: " + "; ".join(problems))
    rows = [{"query_id": q.query_id, "slot": slot_of(q), "text": q.text, "topic": q.topic,
             "question_type": q.question_type, "comparison": q.comparison, "tickers": list(q.tickers),
             "fiscal_years": list(q.fiscal_years), "author": q.author, "authoring": q.authoring}
            for q in sorted(slot_questions, key=lambda q: q.query_id)]
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    previous: dict[str, Any] | None = None
    revisions: list[dict[str, Any]] = []
    if versions:
        before = {r["slot"]: r for r in _read_export(directory, versions[-1])}
        now = {r["slot"]: r for r in rows}
        if {s: (r["query_id"], r["text"]) for s, r in before.items()} == \
                {s: (r["query_id"], r["text"]) for s, r in now.items()}:
            raise AnnotationError(f"nothing to freeze: the questions are identical to version {versions[-1]}")
        by_id = {q.query_id: q for q in store.questions()}
        for slot in sorted(now):
            old, new = before.get(slot), now[slot]
            if old is None or (old["query_id"], old["text"]) != (new["query_id"], new["text"]):
                revisions.append({
                    "slot": slot,
                    "previous_query_id": None if old is None else old["query_id"],
                    "previous_text_sha256": None if old is None else hashlib.sha256(old["text"].encode()).hexdigest(),
                    "query_id": new["query_id"],
                    "text_sha256": hashlib.sha256(new["text"].encode()).hexdigest(),
                    "writer": new["author"],
                    "previous_status": None if old is None else by_id[old["query_id"]].status,
                    "note": by_id[new["query_id"]].notes,
                })
        prev_out, prev_manifest, _ = export_paths(directory, versions[-1])
        previous = {"version": versions[-1], "file": prev_out.name, "sha256": _sha256(prev_out),
                    "manifest_sha256": _sha256(prev_manifest)}
    pooled = sum(1 for q in slot_questions if store.pool(q.query_id))
    directory.mkdir(parents=True, exist_ok=True)
    leakage_archive.write_bytes(leakage_path.read_bytes())
    out.write_text(content, encoding="utf-8")
    manifest = {
        "kind": "finretriever_gold_question_export",
        "version": version,
        "annotation_version": store.info.evaluation.annotation_version,
        "questions": len(rows),
        "file": out.name,
        "sha256": _sha256(out),
        "questions_sha256": questions_sha256((r["query_id"], r["text"]) for r in rows),
        "leakage_check": leakage_archive.name,
        "leakage_check_sha256": _sha256(leakage_archive),
        "supersedes": previous,
        "revisions": revisions,
        "slot_questions_with_a_pool": pooled,
        "created_by": created_by,
        "created_at": utc_now(),
        "notice": "Questions only: no pool, no judgment. Earlier versions are kept unchanged.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for path in (out, manifest_path, leakage_archive):
        path.chmod(0o444)
    return manifest


def write_revision_sheet(store: AnnotationStore, plan: Mapping[str, Slot], slots: Sequence[str], path: str | Path, *,
                         reason: str) -> Path:
    """The revision worksheet: one row per slot to revise, with its plan metadata, its current question and that
    question's writer, and empty columns for the revised question, the writer and the AI-assistance attestation."""
    path = Path(path)
    if path.exists():
        raise AnnotationError(f"{path} exists; a revision sheet is never overwritten")
    if not reason.strip():
        raise AnnotationError("a revision needs a reason")
    active = {slot_of(q): q for q in store.questions() if slot_of(q) and q.status != "withdrawn"}
    problems = [f"{s}: not a slot of the plan" for s in slots if s not in plan]
    problems += [f"{s}: no active question" for s in slots if s in plan and s not in active]
    if len(set(slots)) != len(slots):
        problems.append("a slot is listed twice")
    if problems:
        raise AnnotationError("; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVISION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for slot in sorted(slots):
            q = active[slot]
            writer.writerow({**plan[slot].row(), "current_query_id": q.query_id, "current_question": q.text,
                             "original_writer": q.author, "revision_reason": reason.strip(), "revised_question": "",
                             "writer": "", "ai_assistance": "", "notes": ""})
    return path


def revise_questions(store: AnnotationStore, sheet: str | Path, plan: Mapping[str, Slot], *,
                     ai_identities: Iterable[str], annotator: str) -> list[dict[str, str]]:
    """Validate the filled revision sheet, then replace each slot's question (all or nothing); see the module
    docstring. Returns, per slot, the withdrawn and the added question IDs."""
    with Path(sheet).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if missing := [c for c in REVISION_COLUMNS if c not in (reader.fieldnames or [])]:
            raise AnnotationError(f"{sheet} lacks the columns {missing}")
        rows = list(reader)
    ai = {name.casefold() for name in ai_identities}
    active = {slot_of(q): q for q in store.questions() if slot_of(q) and q.status != "withdrawn"}
    existing = {normalize_question(q.text) for q in store.questions()}
    problems: list[str] = []
    planned_changes: list[tuple[GoldQuestion, dict[str, str]]] = []
    seen_texts: set[str] = set()
    for line, row in enumerate(rows, start=2):
        slot = (row.get("slot") or "").strip()
        where = f"{sheet}:{line} ({slot or 'no slot'})"
        planned = plan.get(slot)
        if planned is None:
            problems.append(f"{where}: not a slot of the plan")
            continue
        if changed := [c for c in PLAN_COLUMNS if (row.get(c) or "").strip().replace(" ", "")
                       != planned.row()[c].replace(" ", "")]:
            problems.append(f"{where}: {changed} differ from the plan; a revision keeps the slot's metadata")
        current = active.get(slot)
        if current is None or current.query_id != (row.get("current_query_id") or "").strip() \
                or current.text != (row.get("current_question") or "").strip():
            problems.append(f"{where}: the sheet's current question is not the slot's active question (stale sheet)")
            continue
        revised = (row.get("revised_question") or "").strip()
        if not revised:
            problems.append(f"{where}: not revised (every listed slot is revised together)")
        elif normalize_question(revised) == normalize_question(current.text):
            problems.append(f"{where}: the revised question is the current one")
        elif normalize_question(revised) in existing or normalize_question(revised) in seen_texts:
            problems.append(f"{where}: the revised question repeats an existing question")
        seen_texts.add(normalize_question(revised))
        writer = (row.get("writer") or "").strip()
        if not writer:
            problems.append(f"{where}: no writer")
        elif writer.casefold() in ai:
            problems.append(f"{where}: {writer!r} is an AI identity; gold questions are written by people")
        elif writer != current.author:
            problems.append(f"{where}: the revision is by {writer!r}; the slot's original writer is {current.author!r}")
        if (row.get("ai_assistance") or "").strip().casefold() != NO_ASSISTANCE:
            problems.append(f"{where}: ai_assistance must be {NO_ASSISTANCE!r}; AI-assisted questions are refused")
        if not (row.get("revision_reason") or "").strip():
            problems.append(f"{where}: no revision reason")
        planned_changes.append((current, row))
    if not rows:
        problems.append(f"{sheet} lists no slot")
    if problems:
        raise AnnotationError(f"{len(problems)} problem(s), nothing revised: " + "; ".join(problems))
    done = []
    for current, row in planned_changes:
        slot, reason = slot_of(current), row["revision_reason"].strip()
        notes = (row.get("notes") or "").strip()
        store.set_status(current.query_id, "withdrawn", annotator=annotator,
                         note=f"revised by {row['writer'].strip()}: {reason}")
        added = store.add_question(
            row["revised_question"].strip(),
            topic=current.topic,
            question_type=current.question_type,
            comparison=current.comparison,
            tickers=current.tickers,
            fiscal_years=current.fiscal_years,
            authoring="independent",
            author=row["writer"].strip(),
            notes=f"[slot {slot}] revision of {current.query_id}: {reason}" + (f"; {notes}" if notes else ""),
            ai_assisted=False,
        )
        done.append({"slot": str(slot), "withdrawn": current.query_id, "added": added.query_id})
    return done


def frozen_questions(directory: str | Path) -> dict[str, str] | None:
    """The latest export version's ``{query_id: text}``, verified against its manifest; ``None`` if there is none."""
    versions = export_versions(directory)
    if not versions:
        return None
    verify_export(directory, versions[-1])
    return {row["query_id"]: row["text"] for row in _read_export(directory, versions[-1])}


def check_poolable(question: GoldQuestion, directory: str | Path) -> None:
    """``pool`` refuses a slot question that is not in the frozen question export with the same text."""
    if slot_of(question) is None:
        return
    frozen = frozen_questions(directory)
    if frozen is None:
        raise AnnotationError(f"{question.query_id} ({slot_of(question)}): the questions are not frozen yet; run "
                              "leakage-check, then export-questions, before pooling")
    if frozen.get(question.query_id) != question.text:
        raise AnnotationError(f"{question.query_id} is not in the frozen question export with this text")


__all__ = [
    "EXPORT_FILE",
    "EXPORT_MANIFEST",
    "LEAKAGE_FILE",
    "REVISION_COLUMNS",
    "SHEET_COLUMNS",
    "SHEET_FILE",
    "Slot",
    "check_poolable",
    "export_paths",
    "export_questions",
    "export_versions",
    "frozen_questions",
    "import_questions",
    "leakage_record",
    "load_plan",
    "questions_sha256",
    "revise_questions",
    "slot_of",
    "verify_export",
    "write_leakage_record",
    "write_question_sheet",
    "write_revision_sheet",
]
