"""Gold set v1: plain-text grading assignments for the two primary annotators (docs/gold_set_v1_remote_annotation.md §7).

Each annotator grades their share of the approved questions from two plain-text files and returns one of them:

1. ``split_questions`` divides the remaining approved questions between the two annotators. Within each question type
   the counts are as even as the totals allow, with the grades an annotator already has counted toward their share.
   Draws come from ``random.Random(SPLIT_SEED)``, and the first draw with the best passage balance is kept.
2. ``build_assignment`` writes one assignment: an assignment document (the task, the rubric v1.1 summary, how to
   respond and return, then every question with all its passages in the approved blind order) and a response file
   listing every anonymous passage ID, to be filled in as ``ID | grade | note``.
   - It holds only the question IDs and texts, and each passage's anonymous ID, filing line, section and text.
   - An anonymous ID is an HMAC of the assignment, question and passage under a random key that is discarded. The
     only link back to the passages is ``private/mapping.json``, which never leaves this machine.
   - ``audit_assignment`` checks both files in memory before anything is written: the texts against the workspace and
     corpus, the blind order, the IDs, and no passage ID, pooling detail, rewording or forbidden term
     (``remote_package.FORBIDDEN``) outside the question and passage texts.
3. ``parse_responses`` and ``import_responses`` take a returned response file, as an attachment or pasted into an
   email. The file is refused whole, with nothing imported, if any of these holds:
   - the header names another assignment or check code, the initials differ, or the YES is missing;
   - an ID is unknown (fabricated), duplicated, or under another question;
   - a question has grades but a line is missing;
   - a grade is not 2, 1, 0 or s, a mark is misplaced, or a required note is missing;
   - a question already recorded comes back with different grades.

   Every fully graded question is then recorded: its grades are appended to the workspace history under the
   annotator's initials, and it is marked complete. Questions with no grade or only some grades wait. A question is
   recorded once; a later file may repeat it unchanged. Each import keeps the returned file read-only, with its record.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import re
import secrets
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

from finretriever.eval_set.annotation_store import AnnotationError, AnnotationStore, utc_now
from finretriever.eval_set.remote_package import (
    CANDIDATE_KEYS,
    FORBIDDEN,
    MIN_NOTE,
    QUESTION_KEYS,
    RUBRIC_VERSION,
    _canonical,
    _deterministic_zip,
    _fill,
    _read_only,
    _sha256,
    blind_order,
    note_required,
    outstanding_questions,
)
from finretriever.eval_set.schemas import SKIPPED

ASSIGN_DIR = "assignments"
SPLIT_FILE = "split_v1.json"
ASSIGNMENT_PATTERN = re.compile(r"^a\d{3}$")
PAYLOAD_FORMAT = "goldset_text_assignment"
FORMAT_VERSION = 1
PAYLOAD_KEYS = {"format", "format_version", "assignment_id", "annotator", "rubric_version", "questions"}
DOCUMENT_TEMPLATE = Path(__file__).with_name("assignment_document.txt")
SPLIT_SEED = "gold-split-v1"
SPLIT_DRAWS = 200_000
CROSSCHECK_SEED = "gold-crosscheck-v1"
CROSSCHECK_PER_ANNOTATOR = 10
CROSSCHECK_RULE = (f"after primary grading, for each primary annotator X: random.Random('{CROSSCHECK_SEED}:X')"
                   f".sample(sorted(the approved questions X graded and completed), {CROSSCHECK_PER_ANNOTATOR}), "
                   "regraded blind by the other annotator; agreement (Cohen's kappa, linear-weighted kappa, "
                   "disagreements) is computed over every passage of those questions")
BOM = "﻿"
MAX_RETURN_BYTES = 16 * 1024 * 1024
INITIALS_LABEL = "INITIALS:"
DECLARATION_LABEL = "GRADED ALONE, WITH NO AI TOOLS (YES):"
END_LINE = "== END =="
ID_LINE = re.compile(r"^(c[0-9a-f]{12})\b(.*)$")
HEADING = re.compile(r"^==\s*(q_\d{4})\b.*$")
HEADER = re.compile(r"ASSIGNMENT\s+(a\d{3})\b.*\bCHECK\s+([0-9a-f]{12})\b")
TAG = re.compile(r"^(P3|P4|ASSISTED)\b[\s:;,.\-]*", re.I)
SKIP_WORDS = {"s", "skip"}


def assign_root(stage_dir: str | Path) -> Path:
    return Path(stage_dir) / ASSIGN_DIR


# --- the split -----------------------------------------------------------------------------------------------------


def split_questions(sizes: Mapping[str, int], types: Mapping[str, str], counts: Mapping[str, int], *,
                    credited: Mapping[str, Sequence[str]], credited_sizes: Mapping[str, int],
                    credited_types: Mapping[str, str], seed: str = SPLIT_SEED,
                    draws: int = SPLIT_DRAWS) -> dict[str, Any]:
    """Divide ``sizes`` (question -> passages) between two annotators, ``counts[a]`` questions each.

    Within each question type, an annotator's share (with ``credited``, the questions they already graded) is as even
    as the counts allow. Among ``draws`` seeded draws, the first with the smallest difference in total passages wins.
    """
    names = sorted(counts)
    if len(names) != 2:
        raise AnnotationError(f"the split is between two annotators, not {names}")
    questions = sorted(sizes)
    if sum(counts.values()) != len(questions):
        raise AnnotationError(f"the counts {dict(counts)} do not add up to the {len(questions)} questions to split")
    a, b = names
    by_type: dict[str, list[str]] = {}
    for q in questions:
        by_type.setdefault(types[q], []).append(q)
    held = {n: Counter(credited_types[q] for q in credited.get(n, ())) for n in names}
    # For each type, how many of its questions go to `a`: the floor or ceiling of an even share, summing to counts[a].
    options = []
    for t, qs in sorted(by_type.items()):
        ideal = (len(qs) + held[b][t] - held[a][t]) / 2
        lo, hi = max(0, int(ideal // 1)), min(len(qs), -int(-ideal // 1))
        options.append(sorted({lo, hi}))
    allocations = [dict(zip(sorted(by_type), combo, strict=True)) for combo in product(*options)
                   if sum(combo) == counts[a]]
    if not allocations:
        raise AnnotationError("no per-type allocation matches the requested counts")
    base = {n: sum(credited_sizes[q] for q in credited.get(n, ())) for n in names}
    rng = random.Random(seed)
    best: tuple[int, int, dict[str, list[str]]] | None = None
    for draw in range(1, draws + 1):
        allocation = allocations[rng.randrange(len(allocations))] if len(allocations) > 1 else allocations[0]
        share: dict[str, list[str]] = {a: [], b: []}
        for t in sorted(by_type):
            chosen = set(rng.sample(by_type[t], allocation[t]))
            share[a] += [q for q in by_type[t] if q in chosen]
            share[b] += [q for q in by_type[t] if q not in chosen]
        totals = {n: base[n] + sum(sizes[q] for q in share[n]) for n in names}
        diff = abs(totals[a] - totals[b])
        if best is None or diff < best[0]:
            best = (diff, draw, {n: sorted(share[n]) for n in names})
        if diff <= (sum(base.values()) + sum(sizes.values())) % 2:
            break
    assert best is not None
    diff, draw, share = best
    return {
        "seed": seed, "draws_allowed": draws, "draw_kept": draw, "stratum": "question_type",
        "per_type_allocations": allocations,
        "assignments": share,
        "credited": {n: sorted(credited.get(n, ())) for n in names},
        "totals": {n: {"credited_questions": len(credited.get(n, ())), "credited_passages": base[n],
                       "new_questions": len(share[n]), "new_passages": sum(sizes[q] for q in share[n]),
                       "total_questions": len(credited.get(n, ())) + len(share[n]),
                       "total_passages": base[n] + sum(sizes[q] for q in share[n]),
                       "types": dict(sorted((Counter(types[q] for q in share[n])
                                             + held[n]).items()))} for n in names},
        "passage_difference": diff,
    }


def load_split(stage_dir: str | Path) -> dict[str, Any]:
    path = assign_root(stage_dir) / SPLIT_FILE
    if not path.is_file():
        raise AnnotationError(f"no split yet ({path}); run assign-split")
    return json.loads(path.read_text(encoding="utf-8"))


def write_split(stage_dir: str | Path, record: Mapping[str, Any]) -> Path:
    path = assign_root(stage_dir) / SPLIT_FILE
    if path.exists():
        raise AnnotationError(f"{path} exists; the split is made once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _read_only(path)
    return path


# --- building an assignment ----------------------------------------------------------------------------------------


def assignment_dirs(stage_dir: str | Path) -> list[Path]:
    root = assign_root(stage_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if ASSIGNMENT_PATTERN.match(p.name) and (p / "manifest.json").is_file())


def load_assignment_manifest(directory: str | Path) -> dict[str, Any]:
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def import_records(directory: str | Path) -> list[dict[str, Any]]:
    imports = Path(directory) / "imports"
    if not imports.is_dir():
        return []
    return [json.loads((d / "import_record.json").read_text(encoding="utf-8"))
            for d in sorted(imports.iterdir()) if (d / "import_record.json").is_file()]


def recorded_questions(directory: str | Path) -> set[str]:
    return {q for record in import_records(directory) for q in record["recorded"]}


def assigned_questions(stage_dir: str | Path) -> dict[str, tuple[str, str, bool]]:
    """Every question in an assignment: ``{query_id: (assignment_id, annotator, recorded)}``."""
    out: dict[str, tuple[str, str, bool]] = {}
    for d in assignment_dirs(stage_dir):
        manifest = load_assignment_manifest(d)
        done = recorded_questions(d)
        for q in manifest["questions"]:
            out[q["query_id"]] = (manifest["assignment_id"], manifest["annotator"], q["query_id"] in done)
    return out


def outstanding_assigned(stage_dir: str | Path) -> dict[str, str]:
    """Assigned questions whose grades have not been recorded: ``{query_id: assignment_id}``."""
    return {q: a for q, (a, _, done) in assigned_questions(stage_dir).items() if not done}


def _anonymous_id(key: bytes, assignment_id: str, query_id: str, passage_id: str) -> str:
    digest = hmac.new(key, f"{assignment_id}\0{query_id}\0{passage_id}".encode(), hashlib.sha256).hexdigest()
    return "c" + digest[:12]


def _names(assignment_id: str, annotator: str) -> dict[str, str]:
    stem = f"gold_v1_{assignment_id}_{annotator}"
    return {"document": f"{stem}_ASSIGNMENT.txt", "responses": f"{stem}_RESPONSES.txt", "zip": f"{stem}.zip",
            "folder": stem}


def render_document(payload: Mapping[str, Any], check: str) -> str:
    """The assignment document: the instructions, then every question with all of its passages."""
    names = _names(payload["assignment_id"], payload["annotator"])
    questions = payload["questions"]
    total = sum(len(q["candidates"]) for q in questions)
    listing = "\n".join(f"  {i:>3}. {q['query_id']}  ({len(q['candidates'])} passages)"
                        for i, q in enumerate(questions, start=1))
    head = _fill(DOCUMENT_TEMPLATE.read_text(encoding="utf-8"), {
        "__ASSIGNMENT_ID__": payload["assignment_id"], "__ANNOTATOR__": payload["annotator"],
        "__QUESTIONS__": str(len(questions)), "__CANDIDATES__": f"{total:,}",
        "__RESPONSE_FILE__": names["responses"], "__CHECK__": check, "__QUESTION_LIST__": listing})
    rule = "=" * 100
    parts = [head]
    for i, q in enumerate(questions, start=1):
        n = len(q["candidates"])
        parts.append(f"\n{rule}\nQUESTION {i} of {len(questions)}  -  {q['query_id']}  -  {n} passages\n{q['text']}\n"
                     f"{rule}\n")
        for j, c in enumerate(q["candidates"], start=1):
            parts.append(f"\n--- {q['query_id']}  -  passage {j} of {n}  -  ID {c['id']} ---\n"
                         f"Filing:  {c['filing']}\nSection: {c['section']}\n\n{c['text']}\n")
    parts.append(f"\n{rule}\nEND OF ASSIGNMENT {payload['assignment_id']}\n")
    return BOM + "".join(parts)


def render_responses(payload: Mapping[str, Any], check: str) -> str:
    """The response file: a header, the two declaration lines, and one empty line per passage ID."""
    lines = [f"GOLD SET V1 - RESPONSES - ASSIGNMENT {payload['assignment_id']} - {payload['annotator']} - CHECK {check}",
             "Complete every passage line as:   ID | grade | note     (see the assignment document)", "",
             f"{INITIALS_LABEL} ", f"{DECLARATION_LABEL} ", ""]
    for q in payload["questions"]:
        lines.append(f"== {q['query_id']} - {len(q['candidates'])} passages ==")
        lines += [f"{c['id']} |  |" for c in q["candidates"]]
        lines.append("")
    lines.append(END_LINE)
    return BOM + "\n".join(lines) + "\n"


def audit_assignment(document: str, responses: str, *, payload: Mapping[str, Any], check: str,
                     mapping: Sequence[Mapping[str, Any]], store: AnnotationStore, order_seed: str,
                     filing: Callable[[Any], str], section: Callable[[Any], str],
                     forbidden_texts: Iterable[str] = ()) -> dict[str, Any]:
    """Check what the annotator would receive; raises ``AnnotationError`` listing every problem."""
    problems: list[str] = []
    if set(payload) != PAYLOAD_KEYS:
        problems.append(f"payload fields {sorted(payload)}")
    if _sha256(_canonical(payload))[:12] != check:
        problems.append("the check code differs from the payload")
    if document != render_document(payload, check) or responses != render_responses(payload, check):
        problems.append("the files differ from the payload")
    by_id = {m["id"]: m for m in mapping}
    passage_ids = {m["passage_id"] for m in mapping}
    for q in payload["questions"]:
        qid = q.get("query_id")
        if set(q) != QUESTION_KEYS:
            problems.append(f"{qid}: question fields {sorted(q)}")
        if q["text"] != store.question(qid).text:
            problems.append(f"{qid}: question text differs from the workspace")
        order = [by_id[c["id"]]["passage_id"] for c in q["candidates"] if c["id"] in by_id]
        if order != blind_order(store, qid, order_seed):
            problems.append(f"{qid}: passages are not all there in the approved blind order")
        for c in q["candidates"]:
            if set(c) != CANDIDATE_KEYS:
                problems.append(f"{qid}: candidate fields {sorted(c)}")
            if c["id"] not in by_id or by_id[c["id"]]["query_id"] != qid:
                problems.append(f"{qid}: candidate ID {c['id']!r} unmapped")
                continue
            passage = store.passages[by_id[c["id"]]["passage_id"]]
            if (c["text"], c["filing"], c["section"]) != (passage.text, filing(passage), section(passage)):
                problems.append(f"{qid}: a passage's text or context differs from the corpus")
    ids = [c["id"] for q in payload["questions"] for c in q["candidates"]]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        problems.append("candidate IDs are duplicated or differ from the mapping")
    parsed = parse_responses(responses.encode("utf-8"))
    if [(r["query_id"], r["id"]) for r in parsed["lines"]] != [(m["query_id"], m["id"]) for m in mapping] \
            or any(r["grade"] or r["note"] for r in parsed["lines"]):
        problems.append("the response file does not list every ID once, empty, in order")
    texts = sorted({t for t in forbidden_texts if t})
    for name, text in (("document", document), ("responses", responses)):
        if hits := [pid for pid in passage_ids if pid in text]:
            problems.append(f"{name}: {len(hits)} passage ID(s) appear")
        if hits := [t for t in texts if t in text]:
            problems.append(f"{name}: {len(hits)} pooling detail or rewording text(s) appear")
    redacted = json.loads(json.dumps(payload))
    for q in redacted["questions"]:
        q["text"] = ""
        for c in q["candidates"]:
            c["text"] = c["filing"] = c["section"] = ""
    terms: dict[str, list[str]] = {}
    for name, text in (("document", render_document(redacted, check)), ("responses", responses)):
        for label, pattern in FORBIDDEN.items():
            if re.search(pattern, text, re.I):
                terms.setdefault(name, []).append(label)
    if terms:
        problems.append(f"forbidden terms in what is sent: {terms}")
    if problems:
        raise AnnotationError(f"assignment audit failed ({len(problems)}): " + "; ".join(problems[:20]))
    return {"passed": True, "checked": {"candidates": len(ids), "passage_ids_absent": len(passage_ids),
                                        "forbidden_texts_absent": len(texts),
                                        "forbidden_terms_absent": sorted(FORBIDDEN)}}


def build_assignment(store: AnnotationStore, stage_dir: str | Path, query_ids: Iterable[str], *, annotator: str,
                     approved_pool_snapshot: Mapping[str, Any], question_export: Mapping[str, Any], created_by: str,
                     filing: Callable[[Any], str], section: Callable[[Any], str], order_seed: str,
                     forbidden_texts: Iterable[str] = (), split: Mapping[str, Any] | None = None,
                     key: bytes | None = None) -> dict[str, Any]:
    """Write the next assignment for ``query_ids``; returns its manifest (see the module docstring)."""
    qids = sorted(set(query_ids))
    if not qids:
        raise AnnotationError("no question to assign")
    taken = assigned_questions(stage_dir)
    if clash := [f"{q} ({taken[q][0]})" for q in qids if q in taken]:
        raise AnnotationError(f"already assigned: {', '.join(clash)}")
    remote = outstanding_questions(stage_dir)
    if clash := [f"{q} ({remote[q]})" for q in qids if q in remote]:
        raise AnnotationError(f"out in a remote package: {', '.join(clash)}; withdraw the package first")
    for qid in qids:
        question = store.question(qid)
        if question.status != "open":
            raise AnnotationError(f"{qid} is {question.status}; only open questions are assigned")
        if store.judgments(qid):
            raise AnnotationError(f"{qid} already has grades; existing grades are never sent")
        if not store.pool(qid):
            raise AnnotationError(f"{qid} has no candidates")
    assignment_id = f"a{len(assignment_dirs(stage_dir)) + 1:03d}"
    directory = assign_root(stage_dir) / assignment_id
    if directory.exists():
        raise AnnotationError(f"{directory} exists")
    key = key or secrets.token_bytes(32)
    questions, mapping = [], []
    for qid in qids:
        candidates = []
        for position, passage_id in enumerate(blind_order(store, qid, order_seed), start=1):
            passage = store.passages[passage_id]
            cid = _anonymous_id(key, assignment_id, qid, passage_id)
            candidates.append({"id": cid, "filing": filing(passage), "section": section(passage), "text": passage.text})
            mapping.append({"id": cid, "query_id": qid, "passage_id": passage_id, "position": position})
        questions.append({"query_id": qid, "text": store.question(qid).text, "candidates": candidates})
    if len({m["id"] for m in mapping}) != len(mapping):
        raise AnnotationError("anonymous ID collision; build again")
    payload = {"format": PAYLOAD_FORMAT, "format_version": FORMAT_VERSION, "assignment_id": assignment_id,
               "annotator": annotator, "rubric_version": RUBRIC_VERSION, "questions": questions}
    payload_sha = _sha256(_canonical(payload))
    check = payload_sha[:12]
    document = render_document(payload, check)
    responses = render_responses(payload, check)
    audit = audit_assignment(document, responses, payload=payload, check=check, mapping=mapping, store=store,
                             order_seed=order_seed, filing=filing, section=section, forbidden_texts=forbidden_texts)
    names = _names(assignment_id, annotator)
    files = {names["document"]: document.encode("utf-8"), names["responses"]: responses.encode("utf-8")}
    zipped = _deterministic_zip({f"{names['folder']}/{n}": b for n, b in files.items()})
    (directory / "send").mkdir(parents=True)
    (directory / "private").mkdir()
    written = {}
    for name, data in {**files, names["zip"]: zipped}.items():
        path = directory / "send" / name
        path.write_bytes(data)
        _read_only(path)
        written[f"send/{name}"] = _sha256(data)
    mapping_doc = {"kind": "finretriever_text_assignment_mapping", "assignment_id": assignment_id,
                   "notice": "PRIVATE: links anonymous candidate IDs to passages. Never send it.", "candidates": mapping}
    mapping_bytes = json.dumps(mapping_doc, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
    (directory / "private" / "mapping.json").write_bytes(mapping_bytes)
    _read_only(directory / "private" / "mapping.json")
    manifest = {
        "kind": "finretriever_text_assignment", "assignment_id": assignment_id, "role": "primary",
        "created_at": utc_now(), "created_by": created_by, "annotator": annotator,
        "approved_pool_snapshot": dict(approved_pool_snapshot), "question_export": dict(question_export),
        "split": dict(split) if split else None,
        "order": f"random.Random('{order_seed}:<query_id>').shuffle(pool order), as in blind annotate",
        "questions": [{"query_id": q["query_id"], "candidates": len(q["candidates"])} for q in questions],
        "candidates": len(mapping), "payload_sha256": payload_sha, "check": check, "files": written,
        "private_mapping_sha256": _sha256(mapping_bytes), "return_file": names["responses"], "audit": audit,
        "notice": "Send only the zip (or the two .txt files) under send/. private/ never leaves this machine.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _read_only(directory / "manifest.json")
    return manifest


# --- the returned response file ------------------------------------------------------------------------------------


def _decode(raw: bytes) -> str:
    if len(raw) > MAX_RETURN_BYTES:
        raise AnnotationError("the returned file is too large")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise AnnotationError("the returned file is neither UTF-8 nor UTF-16 text; save it as plain text") from None


def parse_responses(raw: bytes) -> dict[str, Any]:
    """Read a response file as written (no checking against an assignment): its header, and every ID line in order.

    Leading email quote marks (``>``) are dropped. A line that is not an ID line, a heading, a header line or blank
    continues the note above it (an email may wrap a long note); each is listed in ``joined``.
    """
    text = _decode(raw).replace("\r\n", "\n").replace("\r", "\n")
    header: dict[str, Any] = {"assignment_id": None, "check": None, "initials": None, "declaration": None}
    lines: list[dict[str, Any]] = []
    headings: list[str] = []
    problems: list[str] = []
    joined: list[str] = []
    ended = False
    current: str | None = None
    for number, line in enumerate(text.split("\n"), start=1):
        line = re.sub(r"^(\s*>)+", "", line).strip()
        if not line:
            continue
        if line.upper().startswith(END_LINE):
            ended = True
            break
        if current is None and header["assignment_id"] is None and (m := HEADER.search(line)):
            header["assignment_id"], header["check"] = m.group(1), m.group(2)
            continue
        if line.upper().startswith(INITIALS_LABEL):
            header["initials"] = line[len(INITIALS_LABEL):].strip()
            continue
        if line.upper().startswith(DECLARATION_LABEL):
            header["declaration"] = line[len(DECLARATION_LABEL):].strip()
            continue
        if m := HEADING.match(line):
            current = m.group(1)
            headings.append(current)
            continue
        if m := ID_LINE.match(line):
            if current is None:
                problems.append(f"line {number}: an ID before the first question heading")
                continue
            rest = m.group(2).strip()
            if rest and not rest.startswith("|"):
                problems.append(f"line {number}: {m.group(1)} is not written as 'ID | grade | note'")
                continue
            grade, _, note = rest[1:].partition("|")
            lines.append({"id": m.group(1), "query_id": current, "grade": grade.strip(), "note": note.strip(),
                          "line": number})
            continue
        if current is None:
            continue  # the instruction line(s) of the header
        if lines and lines[-1]["query_id"] == current:
            lines[-1]["note"] = f"{lines[-1]['note']} {line}".strip()
            joined.append(f"line {number} joined to the note of {lines[-1]['id']}: {line[:60]!r}")
            continue
        problems.append(f"line {number}: not an ID line: {line[:40]!r}")
    return {"header": header, "lines": lines, "headings": headings, "joined": joined, "ended": ended,
            "problems": problems}


def split_note(note: str) -> tuple[set[str], str]:
    """The leading marks (P3, P4, ASSISTED) of a note, and the rest of it."""
    tags: set[str] = set()
    rest = note.strip()
    while m := TAG.match(rest):
        tags.add(m.group(1).upper())
        rest = rest[m.end():].strip()
    return tags, rest


def tagged_note(tags: set[str], rest: str) -> str:
    """The note as recorded in the history, with the marks as tags: ``[P3]``, ``[P4]``, ``[assisted]``."""
    labels = [label for tag, label in (("P3", "[P3]"), ("P4", "[P4]"), ("ASSISTED", "[assisted]")) if tag in tags]
    return " ".join([*labels, rest]).strip()


def _grade(value: str) -> str | None:
    v = value.strip().lower()
    if v in SKIP_WORDS:
        return "s"
    return v if v in ("0", "1", "2") else None


def validate_responses(raw: bytes, *, manifest: Mapping[str, Any], mapping: Sequence[Mapping[str, Any]],
                       ai_identities: Iterable[str]) -> dict[str, Any]:
    """Check a returned response file against its assignment; raises ``AnnotationError`` listing every problem.

    Returns the fully graded questions (``complete``: query_id -> [(id, grade, tags, note)] in assignment order), the
    questions with only some grades (``partial``), those with none (``blank``), and observations."""
    parsed = parse_responses(raw)
    problems = list(parsed["problems"])
    observations: list[str] = []
    header = parsed["header"]
    if header["assignment_id"] != manifest["assignment_id"] or header["check"] != manifest["check"]:
        problems.append(f"the header names assignment {header['assignment_id']!r} check {header['check']!r}; expected "
                        f"{manifest['assignment_id']} check {manifest['check']}")
    initials = (header["initials"] or "").strip()
    if initials.casefold() != manifest["annotator"].casefold():
        problems.append(f"initials {initials!r}; the assignment was issued to {manifest['annotator']}")
    if initials.casefold() in {n.casefold() for n in ai_identities}:
        problems.append(f"{initials!r} is an AI identity")
    if (header["declaration"] or "").strip().upper() != "YES":
        problems.append("the YES declaration is missing")
    if not parsed["ended"]:
        observations.append(f"no '{END_LINE}' line: the file may be cut short")
    observations += parsed["joined"]
    order = [q["query_id"] for q in manifest["questions"]]
    if extra := sorted(set(parsed["headings"]) - set(order)):
        problems.append(f"headings for questions not in this assignment: {extra}")
    if dup := sorted(q for q, n in Counter(parsed["headings"]).items() if n > 1):
        problems.append(f"a question heading appears twice: {dup}")
    by_id = {m["id"]: m for m in mapping}
    seen = Counter(r["id"] for r in parsed["lines"])
    if dup := sorted(i for i, n in seen.items() if n > 1):
        problems.append(f"{len(dup)} ID(s) appear twice, e.g. {dup[:3]}")
    if fake := sorted(i for i in seen if i not in by_id):
        problems.append(f"{len(fake)} unknown (fabricated) ID(s), e.g. {fake[:3]}")
    rows: dict[str, dict[str, dict[str, Any]]] = {q: {} for q in order}
    for r in parsed["lines"]:
        m = by_id.get(r["id"])
        if m is None:
            continue
        if m["query_id"] != r["query_id"]:
            problems.append(f"{r['id']} is under {r['query_id']}, but belongs to {m['query_id']}")
            continue
        rows[m["query_id"]][r["id"]] = r
    complete: dict[str, list[tuple[str, str, set[str], str]]] = {}
    partial, blank = [], []
    for qid in order:
        expected = [m["id"] for m in sorted((m for m in mapping if m["query_id"] == qid), key=lambda m: m["position"])]
        got = rows[qid]
        graded = [i for i in expected if i in got and got[i]["grade"]]
        if not graded:
            blank.append(qid)
            if qid not in parsed["headings"]:
                observations.append(f"{qid} is not in the file")
            continue
        if missing := [i for i in expected if i not in got]:
            problems.append(f"{qid}: {len(missing)} ID line(s) missing, e.g. {missing[:3]}")
            continue
        out, ok = [], True
        for cid in expected:
            r = got[cid]
            if not r["grade"]:
                continue
            grade = _grade(r["grade"])
            tags, rest = split_note(r["note"])
            if grade is None:
                problems.append(f"{cid} ({qid}): grade {r['grade']!r} is not 2, 1, 0 or s")
                ok = False
                continue
            if "P4" in tags and grade != "0":
                problems.append(f"{cid} ({qid}): P4 goes only with a 0")
                ok = False
            if grade == "s" and tags & {"P3", "P4"}:
                problems.append(f"{cid} ({qid}): a skip with P3 or P4")
                ok = False
            if note_required(grade, "P3" in tags, "P4" in tags, "ASSISTED" in tags) and len(rest) < MIN_NOTE:
                problems.append(f"{cid} ({qid}): a required note is missing")
                ok = False
            out.append((cid, grade, tags, rest))
        if len(graded) < len(expected):
            partial.append(qid)
        elif ok:
            complete[qid] = out
    if problems:
        raise AnnotationError(f"{len(problems)} problem(s), nothing imported: " + "; ".join(problems[:25]))
    grades = Counter(g for q in complete.values() for _, g, _, _ in q)
    return {"complete": complete, "partial": partial, "blank": blank, "observations": observations,
            "initials": manifest["annotator"],
            "counts": {"questions_complete": len(complete), "candidates": sum(len(q) for q in complete.values()),
                       "grades": {g: grades.get(g, 0) for g in ("2", "1", "0", "s")},
                       "notes": sum(1 for q in complete.values() for _, _, _, n in q if n),
                       "p3": sum("P3" in t for q in complete.values() for _, _, t, _ in q),
                       "p4": sum("P4" in t for q in complete.values() for _, _, t, _ in q),
                       "assisted": sum("ASSISTED" in t for q in complete.values() for _, _, t, _ in q)}}


def load_assignment(stage_dir: str | Path, assignment_id: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    directory = assign_root(stage_dir) / assignment_id
    if not ASSIGNMENT_PATTERN.match(assignment_id) or not (directory / "manifest.json").is_file():
        raise AnnotationError(f"no assignment {assignment_id!r}")
    manifest = load_assignment_manifest(directory)
    mapping_bytes = (directory / "private" / "mapping.json").read_bytes()
    if _sha256(mapping_bytes) != manifest["private_mapping_sha256"]:
        raise AnnotationError(f"{assignment_id}: the private mapping differs from its manifest")
    for rel, sha in manifest["files"].items():
        if _sha256((directory / rel).read_bytes()) != sha:
            raise AnnotationError(f"{assignment_id}: {rel} differs from its manifest")
    return directory, manifest, json.loads(mapping_bytes)["candidates"]


def _history_state(store: AnnotationStore) -> dict[str, Any]:
    raw = (store.directory / "history.jsonl").read_bytes()
    lines = raw.decode("utf-8").splitlines()
    return {"bytes": len(raw), "sha256": _sha256(raw), "last_seq": json.loads(lines[-1])["seq"] if lines else 0}


def import_responses(store: AnnotationStore, stage_dir: str | Path, assignment_id: str, returned: str | Path, *,
                     registered: Iterable[str], ai_identities: Iterable[str], operator: str,
                     verify_pools: Callable[[], Mapping[str, Any]], dry_run: bool = False) -> dict[str, Any]:
    """Validate a returned response file and record its fully graded questions (see the module docstring)."""
    directory, manifest, mapping = load_assignment(stage_dir, assignment_id)
    who = manifest["annotator"]
    if who not in set(registered):
        raise AnnotationError(f"{who!r} is not a registered primary annotator")
    returned = Path(returned)
    raw = returned.read_bytes()
    checked = validate_responses(raw, manifest=manifest, mapping=mapping, ai_identities=ai_identities)
    approved = verify_pools()
    if approved["snapshot"]["sha256"] != manifest["approved_pool_snapshot"]["sha256"]:
        raise AnnotationError("the assignment was built from another pool snapshot than the approved one")
    before_recorded = recorded_questions(directory)
    by_id = {m["id"]: m for m in mapping}
    new, repeated = [], []
    for qid, rows in checked["complete"].items():
        if qid in before_recorded:
            for cid, grade, tags, rest in rows:
                j = store.judgment(qid, by_id[cid]["passage_id"])
                state = SKIPPED if grade == "s" else grade
                if j is None or j.state != state or j.note != tagged_note(tags, rest) or j.annotator != who:
                    raise AnnotationError(f"{qid} was already recorded, and this file changes {cid}; a recorded "
                                          "question is never changed by a new file (nothing imported)")
            repeated.append(qid)
            continue
        question = store.question(qid)
        if question.status != "open" or store.judgments(qid):
            raise AnnotationError(f"{qid} is {question.status} or already has grades; nothing is overwritten")
        pool = {e.passage_id for e in store.pool(qid)}
        mapped = [m["passage_id"] for m in mapping if m["query_id"] == qid]
        if len(mapped) != len(pool) or set(mapped) != pool:
            raise AnnotationError(f"{qid}: the assignment's passages differ from the question's pool")
        new.append(qid)
    order = [q["query_id"] for q in manifest["questions"]]
    new.sort(key=order.index)
    sha = _sha256(raw)
    summary = {"assignment_id": assignment_id, "annotator": who, "file": returned.name, "sha256": sha,
               "bytes": len(raw), "recorded": new, "already_recorded_unchanged": sorted(repeated),
               "partial": checked["partial"], "blank": checked["blank"], "observations": checked["observations"],
               "counts": checked["counts"],
               "new_candidates": sum(len(checked["complete"][q]) for q in new),
               "new_grades": dict(Counter(g for q in new for _, g, _, _ in checked["complete"][q]))}
    if dry_run:
        return {**summary, "dry_run": True}
    if not new:
        raise AnnotationError("no newly completed question in this file; nothing to import")
    imports = directory / "imports"
    number = len([d for d in imports.iterdir() if d.is_dir()]) + 1 if imports.is_dir() else 1
    target = imports / f"{number:03d}"
    target.mkdir(parents=True)
    kept = target / returned.name
    shutil.copyfile(returned, kept)
    _read_only(kept)
    if _sha256(kept.read_bytes()) != sha:
        raise AnnotationError("the kept copy of the returned file differs from it")
    before = _history_state(store)
    started = {"assignment_id": assignment_id, "import": number, "started_at": utc_now(), "operator": operator,
               "history_before": before, "returned_sha256": sha, "recording": new}
    (target / "import_started.json").write_text(json.dumps(started, indent=2) + "\n", encoding="utf-8")
    for qid in new:
        for cid, grade, tags, rest in checked["complete"][qid]:
            pid = by_id[cid]["passage_id"]
            if grade == "s":
                store.skip(qid, pid, annotator=who, note=tagged_note(tags, rest))
            else:
                store.judge(qid, pid, int(grade), annotator=who, note=tagged_note(tags, rest))
        store.set_status(qid, "complete", annotator=who,
                         note=f"assignment {assignment_id}, import {number:03d}; response file sha256 {sha[:12]}")
    after = _history_state(store)
    record = {"kind": "finretriever_text_assignment_import", **summary, "import": number, "imported_at": utc_now(),
              "operator": operator, "kept_as": f"imports/{number:03d}/{kept.name}",
              "approved_pool_snapshot": manifest["approved_pool_snapshot"],
              "history": {"before": before, "after": after, "events_added": after["last_seq"] - before["last_seq"],
                          "first_seq": before["last_seq"] + 1, "last_seq": after["last_seq"]}}
    (target / "import_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    _read_only(target / "import_record.json")
    _read_only(target / "import_started.json")
    return record


__all__ = [
    "CROSSCHECK_RULE",
    "assigned_questions",
    "assignment_dirs",
    "audit_assignment",
    "build_assignment",
    "import_responses",
    "load_assignment",
    "load_split",
    "outstanding_assigned",
    "parse_responses",
    "recorded_questions",
    "render_document",
    "render_responses",
    "split_note",
    "split_questions",
    "tagged_note",
    "validate_responses",
    "write_split",
]
