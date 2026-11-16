"""Gold set v1: blinded remote annotation packages (docs/gold_set_v1_remote_annotation.md).

The primary annotator grades on their own computer, without the repository. Three steps:

1. ``build_package`` writes one package for a set of approved questions. Only these fields go into it:
   - each question's ID and text;
   - each candidate's anonymous ID, its filing and section context, and its passage text;
   - the candidates in the approved blind order, ``random.Random(f"{order_seed}:{query_id}")``.

   They are embedded in a self-contained, offline HTML grading page (``remote_grading_app.html``; its
   Content-Security-Policy forbids every network request), shipped with plain-text instructions. An anonymous ID is
   an HMAC of the package, question and passage under a random key that is discarded. The only link back to the
   passages is the package's private mapping file, which is never sent.

   Before anything is written, ``audit_package`` checks the package in memory. It refuses the package if:
   - a field is not one of the allowed ones;
   - a text differs from its source;
   - the order is not the blind order;
   - a passage ID appears;
   - a pooling source detail or a rewording appears;
   - a forbidden term appears outside the passage and question texts.

2. The annotator grades in the page. It records their initials and a declaration, saves progress in the browser and
   in progress files, and enforces the required notes. It writes a completed grades file.
3. ``validate_returned`` and ``import_returned`` check that file strictly. It is refused whole if:
   - a key is unknown or duplicated;
   - a candidate ID is missing, duplicated or fabricated;
   - a candidate's question does not match the mapping;
   - a grade, mark or required note is missing or wrong;
   - the annotator is not the registered one, or the declaration is missing;
   - the file is incomplete.

   The file is then kept read-only, and each final grade is appended to the workspace history under the
   annotator's initials. P3, P4 and assistance marks become note tags. Nothing that already exists is overwritten:
   every question must still be open and ungraded.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import random
import re
import secrets
import shutil
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from finretriever.eval_set.annotation_store import AnnotationError, AnnotationStore, utc_now

REMOTE_DIR = "remote"
PAYLOAD_FORMAT = "goldset_remote_package"
RESULTS_FORMAT = "goldset_remote_grades"
FORMAT_VERSION = 1
APP_VERSION = "1"
RUBRIC_VERSION = "v1.1"
APP_TEMPLATE = Path(__file__).with_name("remote_grading_app.html")
INSTRUCTIONS_TEMPLATE = Path(__file__).with_name("remote_grading_instructions.txt")
ID_PATTERN = re.compile(r"^c[0-9a-f]{12}$")
PACKAGE_PATTERN = re.compile(r"^p\d{3}$")
GRADES = ("0", "1", "2", "s")
MIN_NOTE = 3
MAX_RETURN_BYTES = 64 * 1024 * 1024
WITHDRAWN_FILE = "withdrawn.json"

PAYLOAD_KEYS = {"format", "format_version", "package_id", "annotator", "rubric_version", "questions"}
QUESTION_KEYS = {"query_id", "text", "candidates"}
CANDIDATE_KEYS = {"id", "filing", "section", "text"}
RESULT_KEYS = {"format", "format_version", "package_id", "payload_sha256", "app_version", "annotator", "declaration",
               "started_at", "saved_at", "complete", "questions", "records"}
RECORD_KEYS = {"query_id", "grade", "p3", "p4", "assisted", "note", "at", "history"}
HISTORY_KEYS = {"grade", "p3", "p4", "assisted", "note", "at"}
DECLARATION_KEYS = {"accepted", "text", "accepted_at"}

# Terms that must never appear in what the annotator receives, outside the question and passage texts themselves.
FORBIDDEN = {
    "BM25": r"bm25", "stratified sample": r"stratif", "reformulation": r"reformulat", "rewording": r"reword",
    "rank": r"\brank(s|ed|ing)?\b", "score": r"\bscor(e|es|ed|ing)\b", "search depth": r"\bdepth\b",
    "search text": r"query=|tickers=", "method": r"\bmethods?\b", "source": r"\bsources?\b",
    "model": r"\bmodels?\b", "retrieval": r"retriev", "BGE": r"\bbge\b", "E5": r"\be5\b", "A0 or A1": r"\ba[01]\b",
    "Claude": r"claude", "exclusion": r"exclu(ded|sion|de)\b", "training": r"\btrain(ing|ed)?\b",
    "validation": r"validation", "pool": r"\bpool(s|ed|ing)?\b", "snapshot": r"snapshot", "eligible": r"eligib",
    "passage ID": r"passage_?id|passage_filing_", "random": r"\brandom\b", "seed": r"\bseed\b",
}


def remote_root(stage_dir: str | Path) -> Path:
    return Path(stage_dir) / REMOTE_DIR


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _read_only(path: Path) -> None:
    path.chmod(0o444)


def package_dirs(stage_dir: str | Path) -> list[Path]:
    root = remote_root(stage_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if PACKAGE_PATTERN.match(p.name) and (p / "manifest.json").is_file())


def load_manifest(package_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(package_dir) / "manifest.json").read_text(encoding="utf-8"))


def is_withdrawn(package_dir: str | Path) -> bool:
    return (Path(package_dir) / WITHDRAWN_FILE).is_file()


def withdraw_package(stage_dir: str | Path, package_id: str, *, reason: str, operator: str) -> dict[str, Any]:
    """Withdraw a package that was never imported: it keeps its files, its questions can be issued again, and it can
    never be imported. Writes ``withdrawn.json`` (read-only)."""
    package_dir = remote_root(stage_dir) / package_id
    if not PACKAGE_PATTERN.match(package_id) or not (package_dir / "manifest.json").is_file():
        raise AnnotationError(f"no package {package_id!r}")
    if is_withdrawn(package_dir):
        raise AnnotationError(f"{package_id} is already withdrawn")
    if (package_dir / "import_record.json").exists() or (package_dir / "returned").exists():
        raise AnnotationError(f"{package_id} was imported (or an import started); it cannot be withdrawn")
    if not reason.strip():
        raise AnnotationError("a withdrawal needs a reason")
    manifest_bytes = (package_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    record = {"kind": "finretriever_remote_package_withdrawal", "package_id": package_id, "withdrawn_at": utc_now(),
              "operator": operator, "reason": reason.strip(), "manifest_sha256": _sha256(manifest_bytes),
              "questions": [q["query_id"] for q in manifest["questions"]],
              "notice": "Withdrawn: never imported; its questions may be issued again. The files are kept unchanged."}
    path = package_dir / WITHDRAWN_FILE
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _read_only(path)
    return record


def packaged_questions(stage_dir: str | Path) -> dict[str, tuple[str, bool]]:
    """Every question in a package that is not withdrawn: ``{query_id: (package_id, imported)}``."""
    out: dict[str, tuple[str, bool]] = {}
    for d in package_dirs(stage_dir):
        if is_withdrawn(d):
            continue
        manifest = load_manifest(d)
        for q in manifest["questions"]:
            out[q["query_id"]] = (manifest["package_id"], (d / "import_record.json").is_file())
    return out


def outstanding_questions(stage_dir: str | Path) -> dict[str, str]:
    """Questions issued in a package whose grades have not come back: ``{query_id: package_id}``."""
    return {q: p for q, (p, imported) in packaged_questions(stage_dir).items() if not imported}


def blind_order(store: AnnotationStore, query_id: str, order_seed: str) -> list[str]:
    """The approved display order (the same as the local blind ``annotate``): passage IDs."""
    entries = list(store.pool(query_id))
    random.Random(f"{order_seed}:{query_id}").shuffle(entries)
    return [e.passage_id for e in entries]


def _anonymous_id(key: bytes, package_id: str, query_id: str, passage_id: str) -> str:
    digest = hmac.new(key, f"{package_id}\0{query_id}\0{passage_id}".encode(), hashlib.sha256).hexdigest()
    return "c" + digest[:12]


PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*[A-Z0-9]__")


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Fill the template's placeholders, which must be exactly ``values``' keys. Only the template is checked, never
    the filled-in data (passage text may hold runs of underscores); the values are inserted in order, so put the
    data last."""
    found = set(PLACEHOLDER.findall(template))
    if found != set(values):
        raise AnnotationError(f"template placeholders {sorted(found)} differ from the values {sorted(values)}")
    for key, value in values.items():
        template = template.replace(key, value)
    return template


def render_app(payload_bytes: bytes, payload_sha256: str, package_id: str) -> str:
    embedded = payload_bytes.decode("utf-8").replace("<", "\\u003c")  # no "</script>" can end the data block
    template = APP_TEMPLATE.read_text(encoding="utf-8")
    if template.count("__PAYLOAD_JSON__") != 1 or template.count("__PAYLOAD_SHA256__") != 1:
        raise AnnotationError("the grading page template must hold each data placeholder exactly once")
    return _fill(template, {"__PAYLOAD_SHA256__": payload_sha256, "__PACKAGE_ID__": package_id,
                            "__PAYLOAD_JSON__": embedded})


def _names(package_id: str, annotator: str) -> dict[str, str]:
    stem = f"gold_v1_{package_id}_{annotator}"
    return {"app": f"{stem}.html", "instructions": "INSTRUCTIONS.txt", "zip": f"{stem}.zip", "folder": stem,
            "return": f"{stem}_grades_COMPLETE.json"}


def render_instructions(package_id: str, annotator: str, questions: int, candidates: int) -> str:
    names = _names(package_id, annotator)
    return _fill(INSTRUCTIONS_TEMPLATE.read_text(encoding="utf-8"), {
        "__PACKAGE_ID__": package_id, "__ANNOTATOR__": annotator, "__QUESTIONS__": str(questions),
        "__CANDIDATES__": str(candidates), "__APP_FILE__": names["app"], "__RETURN_FILE__": names["return"]})


def _embedded_payload(html: str) -> Any:
    match = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.S)
    if match is None:
        raise AnnotationError("the grading page holds no data block")
    return json.loads(match.group(1))


def audit_package(html: str, instructions: str, *, payload: Mapping[str, Any], payload_sha256: str,
                  mapping: Sequence[Mapping[str, Any]], store: AnnotationStore, order_seed: str,
                  filing: Callable[[Any], str], section: Callable[[Any], str],
                  forbidden_texts: Iterable[str] = ()) -> dict[str, Any]:
    """Check what the annotator would receive; raises ``AnnotationError`` listing every problem."""
    problems: list[str] = []
    embedded = _embedded_payload(html)
    if _canonical(embedded) != _canonical(payload) or _sha256(_canonical(embedded)) != payload_sha256:
        problems.append("the page's data block differs from the payload")
    if set(payload) != PAYLOAD_KEYS:
        problems.append(f"payload fields {sorted(payload)}")
    by_id = {m["id"]: m for m in mapping}
    passage_ids = {m["passage_id"] for m in mapping}
    for q in payload["questions"]:
        if set(q) != QUESTION_KEYS:
            problems.append(f"{q.get('query_id')}: question fields {sorted(q)}")
        question = store.question(q["query_id"])
        if q["text"] != question.text:
            problems.append(f"{q['query_id']}: question text differs from the workspace")
        ids = [c["id"] for c in q["candidates"]]
        order = [by_id[i]["passage_id"] for i in ids if i in by_id]
        if order != blind_order(store, q["query_id"], order_seed):
            problems.append(f"{q['query_id']}: candidates are not in the approved blind order")
        for c in q["candidates"]:
            if set(c) != CANDIDATE_KEYS:
                problems.append(f"{q['query_id']}: candidate fields {sorted(c)}")
            if not ID_PATTERN.match(c["id"]) or c["id"] not in by_id or by_id[c["id"]]["query_id"] != q["query_id"]:
                problems.append(f"{q['query_id']}: candidate ID {c['id']!r} malformed or unmapped")
                continue
            passage = store.passages[by_id[c["id"]]["passage_id"]]
            if (c["text"], c["filing"], c["section"]) != (passage.text, filing(passage), section(passage)):
                problems.append(f"{q['query_id']}: a candidate's text or context differs from the corpus")
    ids_all = [c["id"] for q in payload["questions"] for c in q["candidates"]]
    if len(ids_all) != len(set(ids_all)) or set(ids_all) != set(by_id):
        problems.append("candidate IDs are duplicated or differ from the mapping")
    for name, text in (("page", html), ("instructions", instructions)):
        if hits := [pid for pid in passage_ids if pid in text]:
            problems.append(f"{name}: {len(hits)} passage ID(s) appear")
        if hits := [t for t in forbidden_texts if t and t in text]:
            problems.append(f"{name}: {len(hits)} pooling detail or rewording text(s) appear")
    # The shell: the page without the question and passage texts, and the instructions.
    redacted = json.loads(json.dumps(payload))
    for q in redacted["questions"]:
        q["text"] = ""
        for c in q["candidates"]:
            c["text"] = c["filing"] = c["section"] = ""
    shell = re.sub(r'(<script type="application/json" id="payload">).*?(</script>)',
                   lambda m: m.group(1) + json.dumps(redacted) + m.group(2), html, flags=re.S)
    terms = {}
    for name, text in (("page", shell), ("instructions", instructions)):
        for label, pattern in FORBIDDEN.items():
            if re.search(pattern, text, re.I):
                terms.setdefault(name, []).append(label)
    if terms:
        problems.append(f"forbidden terms in what is sent: {terms}")
    if problems:
        raise AnnotationError(f"package audit failed ({len(problems)}): " + "; ".join(problems[:20]))
    return {"passed": True, "fields": {"payload": sorted(PAYLOAD_KEYS), "question": sorted(QUESTION_KEYS),
                                       "candidate": sorted(CANDIDATE_KEYS)},
            "checked": {"candidates": len(ids_all), "passage_ids_absent": len(passage_ids),
                        "forbidden_texts_absent": len([t for t in forbidden_texts if t]),
                        "forbidden_terms_absent": sorted(FORBIDDEN)}}


def _deterministic_zip(files: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return buffer.getvalue()


def build_package(store: AnnotationStore, stage_dir: str | Path, query_ids: Iterable[str], *, annotator: str,
                  approved_pool_snapshot: Mapping[str, Any], question_export: Mapping[str, Any], created_by: str,
                  filing: Callable[[Any], str], section: Callable[[Any], str], order_seed: str,
                  forbidden_texts: Iterable[str] = (), key: bytes | None = None) -> dict[str, Any]:
    """Write the next package for ``query_ids``; returns its manifest (see the module docstring)."""
    qids = sorted(set(query_ids))
    if not qids:
        raise AnnotationError("no question to package")
    taken = packaged_questions(stage_dir)
    if clash := [f"{q} ({taken[q][0]})" for q in qids if q in taken]:
        raise AnnotationError(f"already in a package: {', '.join(clash)}")
    for qid in qids:
        question = store.question(qid)
        if question.status != "open":
            raise AnnotationError(f"{qid} is {question.status}; only open questions are packaged")
        if store.judgments(qid):
            raise AnnotationError(f"{qid} already has grades; existing grades are never exported")
        if not store.pool(qid):
            raise AnnotationError(f"{qid} has no candidates")
    number = len(package_dirs(stage_dir)) + 1
    package_id = f"p{number:03d}"
    package_dir = remote_root(stage_dir) / package_id
    if package_dir.exists():
        raise AnnotationError(f"{package_dir} exists")
    key = key or secrets.token_bytes(32)
    questions, mapping = [], []
    for qid in qids:
        candidates = []
        for position, passage_id in enumerate(blind_order(store, qid, order_seed), start=1):
            passage = store.passages[passage_id]
            cid = _anonymous_id(key, package_id, qid, passage_id)
            candidates.append({"id": cid, "filing": filing(passage), "section": section(passage), "text": passage.text})
            mapping.append({"id": cid, "query_id": qid, "passage_id": passage_id, "position": position})
        questions.append({"query_id": qid, "text": store.question(qid).text, "candidates": candidates})
    if len({m["id"] for m in mapping}) != len(mapping):
        raise AnnotationError("anonymous ID collision; build again")
    payload = {"format": PAYLOAD_FORMAT, "format_version": FORMAT_VERSION, "package_id": package_id,
               "annotator": annotator, "rubric_version": RUBRIC_VERSION, "questions": questions}
    payload_bytes = _canonical(payload)
    payload_sha = _sha256(payload_bytes)
    html = render_app(payload_bytes, payload_sha, package_id)
    instructions = render_instructions(package_id, annotator, len(qids), len(mapping))
    forbidden = sorted({t for t in forbidden_texts if t})
    audit = audit_package(html, instructions, payload=payload, payload_sha256=payload_sha, mapping=mapping, store=store,
                          order_seed=order_seed, filing=filing, section=section, forbidden_texts=forbidden)
    names = _names(package_id, annotator)
    files = {names["app"]: html.encode("utf-8"), names["instructions"]: instructions.encode("utf-8")}
    zipped = _deterministic_zip({f"{names['folder']}/{n}": b for n, b in files.items()})
    (package_dir / "send").mkdir(parents=True)
    (package_dir / "private").mkdir()
    written = {}
    for name, data in {**files, names["zip"]: zipped}.items():
        path = package_dir / "send" / name
        path.write_bytes(data)
        _read_only(path)
        written[f"send/{name}"] = _sha256(data)
    mapping_doc = {"kind": "finretriever_remote_package_mapping", "package_id": package_id,
                   "notice": "PRIVATE: links anonymous candidate IDs to passages. Never send it.",
                   "candidates": mapping}
    mapping_bytes = json.dumps(mapping_doc, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
    (package_dir / "private" / "mapping.json").write_bytes(mapping_bytes)
    _read_only(package_dir / "private" / "mapping.json")
    manifest = {
        "kind": "finretriever_remote_package", "package_id": package_id, "created_at": utc_now(),
        "created_by": created_by, "annotator": annotator,
        "approved_pool_snapshot": dict(approved_pool_snapshot), "question_export": dict(question_export),
        "order": f"random.Random('{order_seed}:<query_id>').shuffle(pool order), as in blind annotate",
        "questions": [{"query_id": q["query_id"], "candidates": len(q["candidates"])} for q in questions],
        "candidates": len(mapping), "payload_sha256": payload_sha, "files": written,
        "private_mapping_sha256": _sha256(mapping_bytes), "return_file": names["return"], "audit": audit,
        "notice": "Send only the zip (or the two files) under send/. private/ never leaves this machine.",
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    _read_only(package_dir / "manifest.json")
    return manifest


# --- the returned grades -------------------------------------------------------------------------------------------


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if dups := sorted(k for k, n in Counter(keys).items() if n > 1):
        raise AnnotationError(f"duplicate keys in the returned file: {dups[:5]}")
    return dict(pairs)


def _timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def note_required(grade: str, p3: bool, p4: bool, assisted: bool) -> bool:
    """Rubric v1.1 P6 and protocol §2.3: a note for every 1, every skip, every P3 comparative, every P4 0 and any
    assistance."""
    return grade in ("1", "s") or p3 or (grade == "0" and p4) or assisted


def _record_problems(cid: str, r: Mapping[str, Any]) -> list[str]:
    out = []
    if set(r) != RECORD_KEYS:
        return [f"{cid}: fields {sorted(r)}"]
    grade, p3, p4, assisted, note = r["grade"], r["p3"], r["p4"], r["assisted"], r["note"]
    if grade not in GRADES:
        out.append(f"{cid}: grade {grade!r}")
    if not all(isinstance(v, bool) for v in (p3, p4, assisted)) or not isinstance(note, str):
        return [*out, f"{cid}: malformed marks or note"]
    if p4 and grade != "0":
        out.append(f"{cid}: P4 on a grade other than 0")
    if grade == "s" and (p3 or p4):
        out.append(f"{cid}: a skip with a P3 or P4 mark")
    if grade in GRADES and note_required(grade, p3, p4, assisted) and len(note.strip()) < MIN_NOTE:
        out.append(f"{cid}: a required note is missing")
    if not _timestamp(r["at"]):
        out.append(f"{cid}: no valid time")
    history = r["history"]
    if not isinstance(history, list) or not history:
        out.append(f"{cid}: no grading history")
    else:
        if any(not isinstance(h, dict) or set(h) != HISTORY_KEYS or h["grade"] not in GRADES or not _timestamp(h["at"])
               for h in history):
            out.append(f"{cid}: malformed history")
        else:
            last = history[-1]
            if any(last[k] != r[k] for k in ("grade", "p3", "p4", "assisted", "note")) or last["at"] != r["at"]:
                out.append(f"{cid}: the final grade differs from the last history entry")
    return out


def validate_returned(raw: bytes, *, manifest: Mapping[str, Any], mapping: Sequence[Mapping[str, Any]], primary: str,
                      ai_identities: Iterable[str]) -> dict[str, Any]:
    """Check a returned grades file against its package; raises ``AnnotationError`` listing every problem."""
    if len(raw) > MAX_RETURN_BYTES:
        raise AnnotationError("the returned file is too large")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise AnnotationError("the returned file is not UTF-8") from None
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"the returned file is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise AnnotationError("the returned file is not a grades file")
    problems: list[str] = []
    observations: list[str] = []
    if set(data) != RESULT_KEYS:
        raise AnnotationError(f"unexpected fields: missing {sorted(RESULT_KEYS - set(data))}, "
                              f"extra {sorted(set(data) - RESULT_KEYS)}")
    if (data["format"], data["format_version"]) != (RESULTS_FORMAT, FORMAT_VERSION):
        problems.append(f"format {data['format']!r} version {data['format_version']!r}")
    if data["package_id"] != manifest["package_id"]:
        problems.append(f"package {data['package_id']!r}, expected {manifest['package_id']!r}")
    if data["payload_sha256"] != manifest["payload_sha256"]:
        problems.append("the file was made from different package data")
    who = data["annotator"]
    if not isinstance(who, str) or who != manifest["annotator"] or who != primary:
        problems.append(f"annotator {who!r}; the package was issued to {manifest['annotator']!r} and the registered "
                        f"primary annotator is {primary!r}")
    if isinstance(who, str) and who.casefold() in {n.casefold() for n in ai_identities}:
        problems.append(f"{who!r} is an AI identity")
    decl = data["declaration"]
    if not isinstance(decl, dict) or set(decl) != DECLARATION_KEYS or decl["accepted"] is not True \
            or not isinstance(decl["text"], str) or not decl["text"].strip() or not _timestamp(decl["accepted_at"]):
        problems.append("the annotator's declaration is missing or incomplete")
    for field in ("started_at", "saved_at"):
        if not _timestamp(data[field]):
            problems.append(f"{field} is not a valid time")
    if data["complete"] is not True:
        problems.append("the file is a progress file, not completed grades")
    package_qids = [q["query_id"] for q in manifest["questions"]]
    questions = data["questions"]
    if not isinstance(questions, dict) or sorted(questions) != sorted(package_qids):
        problems.append("the questions differ from the package's")
    else:
        for qid, q in questions.items():
            if not isinstance(q, dict) or set(q) != {"done_at"} or not _timestamp(q["done_at"]):
                problems.append(f"{qid} is not marked done")
    by_id = {m["id"]: m for m in mapping}
    records = data["records"]
    if not isinstance(records, dict):
        raise AnnotationError("; ".join([*problems, "no grade records"]))
    if missing := sorted(set(by_id) - set(records)):
        problems.append(f"{len(missing)} candidate(s) missing, e.g. {missing[:3]}")
    if extra := sorted(set(records) - set(by_id)):
        problems.append(f"{len(extra)} unknown (fabricated) candidate ID(s), e.g. {extra[:3]}")
    for cid, r in records.items():
        if cid not in by_id:
            continue
        if not isinstance(r, dict):
            problems.append(f"{cid}: malformed")
            continue
        if r.get("query_id") != by_id[cid]["query_id"]:
            problems.append(f"{cid}: listed under {r.get('query_id')!r}, but belongs to {by_id[cid]['query_id']}")
        problems += _record_problems(cid, r)
        if isinstance(r.get("history"), list) and len(r["history"]) > 1:
            times = [h.get("at") for h in r["history"] if isinstance(h, dict)]
            if times != sorted(times):
                observations.append(f"{cid}: history times are not in order (clock change?)")
    if problems:
        raise AnnotationError(f"{len(problems)} problem(s), nothing imported: " + "; ".join(problems[:25]))
    grades = Counter(r["grade"] for r in records.values())
    return {"data": data, "observations": observations,
            "counts": {"candidates": len(records), "grades": {g: grades.get(g, 0) for g in GRADES},
                       "notes": sum(1 for r in records.values() if r["note"].strip()),
                       "p3": sum(r["p3"] for r in records.values()), "p4": sum(r["p4"] for r in records.values()),
                       "assisted": sum(r["assisted"] for r in records.values()),
                       "changes_before_final": sum(len(r["history"]) - 1 for r in records.values())}}


def tagged_note(record: Mapping[str, Any]) -> str:
    """The note as written, prefixed with the marks: ``[P3]``, ``[P4]``, ``[assisted]``."""
    tags = [tag for tag, on in (("[P3]", record["p3"]), ("[P4]", record["p4"]), ("[assisted]", record["assisted"])) if on]
    return " ".join([*tags, record["note"].strip()]).strip()


def _history_state(store: AnnotationStore) -> dict[str, Any]:
    path = store.directory / "history.jsonl"
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    last = json.loads(lines[-1])["seq"] if lines else 0
    return {"bytes": len(raw), "sha256": _sha256(raw), "last_seq": last}


def load_package(stage_dir: str | Path, package_id: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    package_dir = remote_root(stage_dir) / package_id
    if not PACKAGE_PATTERN.match(package_id) or not (package_dir / "manifest.json").is_file():
        raise AnnotationError(f"no package {package_id!r}")
    if is_withdrawn(package_dir):
        raise AnnotationError(f"{package_id} was withdrawn; it is never imported")
    manifest = load_manifest(package_dir)
    mapping_bytes = (package_dir / "private" / "mapping.json").read_bytes()
    if _sha256(mapping_bytes) != manifest["private_mapping_sha256"]:
        raise AnnotationError(f"{package_id}: the private mapping differs from its manifest")
    for rel, sha in manifest["files"].items():
        if _sha256((package_dir / rel).read_bytes()) != sha:
            raise AnnotationError(f"{package_id}: {rel} differs from its manifest")
    return package_dir, manifest, json.loads(mapping_bytes)["candidates"]


def import_returned(store: AnnotationStore, stage_dir: str | Path, package_id: str, returned: str | Path, *,
                    primary: str, ai_identities: Iterable[str], operator: str,
                    verify_pools: Callable[[], Mapping[str, Any]], dry_run: bool = False) -> dict[str, Any]:
    """Validate a returned grades file and append its grades to the history (see the module docstring)."""
    package_dir, manifest, mapping = load_package(stage_dir, package_id)
    if (package_dir / "import_record.json").exists() or (package_dir / "returned").exists():
        raise AnnotationError(f"{package_id} was already imported (or an import started); nothing is imported twice")
    returned = Path(returned)
    raw = returned.read_bytes()
    checked = validate_returned(raw, manifest=manifest, mapping=mapping, primary=primary, ai_identities=ai_identities)
    approved = verify_pools()
    if approved["snapshot"]["sha256"] != manifest["approved_pool_snapshot"]["sha256"]:
        raise AnnotationError("the package was built from another pool snapshot than the approved one")
    for q in manifest["questions"]:
        qid = q["query_id"]
        question = store.question(qid)
        if question.status != "open":
            raise AnnotationError(f"{qid} is {question.status}; nothing is overwritten")
        if store.judgments(qid):
            raise AnnotationError(f"{qid} already has grades; nothing is overwritten")
        pool = {e.passage_id for e in store.pool(qid)}
        mapped = [m["passage_id"] for m in mapping if m["query_id"] == qid]
        if len(mapped) != len(pool) or set(mapped) != pool:
            raise AnnotationError(f"{qid}: the package's candidates differ from the question's pool")
    data = checked["data"]
    summary = {"package_id": package_id, "file": returned.name, "sha256": _sha256(raw), "bytes": len(raw),
               "annotator": data["annotator"], **checked["counts"], "observations": checked["observations"],
               "questions": [q["query_id"] for q in manifest["questions"]]}
    if dry_run:
        return {**summary, "dry_run": True}
    returned_dir = package_dir / "returned"
    returned_dir.mkdir()
    kept = returned_dir / returned.name
    shutil.copyfile(returned, kept)
    _read_only(kept)
    if _sha256(kept.read_bytes()) != summary["sha256"]:
        raise AnnotationError("the kept copy of the returned file differs from it")
    before = _history_state(store)
    started = {"package_id": package_id, "started_at": utc_now(), "operator": operator, "history_before": before,
               "returned_sha256": summary["sha256"]}
    (package_dir / "import_started.json").write_text(json.dumps(started, indent=2) + "\n", encoding="utf-8")
    who = data["annotator"]
    records = data["records"]
    for q in manifest["questions"]:
        qid = q["query_id"]
        for m in sorted((m for m in mapping if m["query_id"] == qid), key=lambda m: m["position"]):
            r = records[m["id"]]
            if r["grade"] == "s":
                store.skip(qid, m["passage_id"], annotator=who, note=tagged_note(r))
            else:
                store.judge(qid, m["passage_id"], int(r["grade"]), annotator=who, note=tagged_note(r))
        store.set_status(qid, "complete", annotator=who,
                         note=f"remote package {package_id}; returned file sha256 {summary['sha256'][:12]}; "
                              f"marked done {data['questions'][qid]['done_at']}")
    after = _history_state(store)
    record = {"kind": "finretriever_remote_import", **summary, "imported_at": utc_now(), "operator": operator,
              "kept_as": f"returned/{kept.name}", "approved_pool_snapshot": manifest["approved_pool_snapshot"],
              "history": {"before": before, "after": after, "events_added": after["last_seq"] - before["last_seq"],
                          "first_seq": before["last_seq"] + 1, "last_seq": after["last_seq"]},
              "remote_times": {"started_at": data["started_at"], "saved_at": data["saved_at"],
                               "declaration_accepted_at": data["declaration"]["accepted_at"]}}
    (package_dir / "import_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
    _read_only(package_dir / "import_record.json")
    _read_only(package_dir / "import_started.json")
    return record


__all__ = [
    "FORBIDDEN",
    "audit_package",
    "blind_order",
    "build_package",
    "import_returned",
    "is_withdrawn",
    "load_package",
    "note_required",
    "outstanding_questions",
    "package_dirs",
    "packaged_questions",
    "remote_root",
    "tagged_note",
    "validate_returned",
    "withdraw_package",
]
