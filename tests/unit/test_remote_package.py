"""Remote, blinded annotation packages: export, the offline grading page's logic, and the validated import.

Synthetic grades only, on the fixture corpus; no real question or grade is involved."""

from __future__ import annotations

import copy
import json
import random
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from finretriever.eval_set.remote_package import FORBIDDEN
from finretriever.utils.io import read_jsonl
from tests.unit.test_question_stage import (  # noqa: F401 (stage is a fixture)
    approve,
    pooled_v1,
    register,
    stage,
)

T = "2026-11-14T10:00:00.000Z"


def ready(stage: dict[str, Any]) -> None:  # noqa: F811
    pooled_v1(stage)
    approve(stage, 1)
    register(stage, "T1")


def package_dir(stage: dict[str, Any], package_id: str = "p001") -> Path:  # noqa: F811
    return stage["dir"] / "remote" / package_id


def page_payload(html: str) -> dict[str, Any]:
    return json.loads(re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.S).group(1))


def built(stage: dict[str, Any], *questions: str) -> tuple[Path, str, dict[str, Any]]:  # noqa: F811
    code, out, err = stage["run"]("remote-package", "--questions", ",".join(questions or ("q_0001",)),
                                  annotator="claude")
    assert code == 0, err
    d = package_dir(stage, re.search(r"package (p\d{3})", out).group(1))
    html = next((d / "send").glob("*.html")).read_text()
    return d, html, page_payload(html)


def synthetic(d: Path, payload: dict[str, Any], *, annotator: str = "T1") -> dict[str, Any]:
    """A completed grades file as the page writes it, with synthetic grades cycling 0, 2, 1 (note), s (note)."""
    manifest = json.loads((d / "manifest.json").read_text())
    records = {}
    for i, c in enumerate(cand for q in payload["questions"] for cand in [(q["query_id"], x) for x in q["candidates"]]):
        qid, cand = c
        g = ["0", "2", "1", "s"][i % 4]
        rec = {"grade": g, "p3": False, "p4": False, "assisted": False, "note": "synthetic note" if g in "1s" else ""}
        records[cand["id"]] = {"query_id": qid, **rec, "at": T, "history": [{**rec, "at": T}]}
    return {"format": "goldset_remote_grades", "format_version": 1, "package_id": manifest["package_id"],
            "payload_sha256": manifest["payload_sha256"], "app_version": "1", "annotator": annotator,
            "declaration": {"accepted": True, "text": "synthetic declaration", "accepted_at": T},
            "started_at": T, "saved_at": T, "complete": True,
            "questions": {q["query_id"]: {"done_at": T} for q in payload["questions"]}, "records": records}


def write(path: Path, data: dict[str, Any] | str) -> Path:
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def test_the_package_holds_only_the_allowed_fields_in_the_blind_order(stage: dict[str, Any]) -> None:  # noqa: F811
    ready(stage)
    pool_file = stage["tmp"] / "annotation" / "pool.jsonl"
    history_file = stage["tmp"] / "annotation" / "history.jsonl"
    before = (pool_file.read_bytes(), history_file.read_bytes(),
              {p.name: p.read_bytes() for p in (stage["dir"] / "pool_v1").iterdir()})
    d, html, payload = built(stage, "q_0001")
    assert sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()) == [
        "manifest.json", "private/mapping.json", "send/INSTRUCTIONS.txt", "send/gold_v1_p001_T1.html",
        "send/gold_v1_p001_T1.zip"]
    assert all(not p.stat().st_mode & 0o222 for p in d.rglob("*") if p.is_file())  # read-only
    # Only the allowed fields.
    assert set(payload) == {"format", "format_version", "package_id", "annotator", "rubric_version", "questions"}
    assert [set(q) for q in payload["questions"]] == [{"query_id", "text", "candidates"}]
    cands = payload["questions"][0]["candidates"]
    assert all(set(c) == {"id", "filing", "section", "text"} and re.fullmatch(r"c[0-9a-f]{12}", c["id"]) for c in cands)
    # The approved blind order, recomputed independently from the pool file.
    rows = [r for r in read_jsonl(pool_file) if r["query_id"] == "q_0001"]
    expected = [r["passage_id"] for r in sorted(rows, key=lambda r: r["position"])]
    random.Random("gold-blind-v1:q_0001").shuffle(expected)
    mapping = {m["id"]: m for m in json.loads((d / "private" / "mapping.json").read_text())["candidates"]}
    assert [mapping[c["id"]]["passage_id"] for c in cands] == expected
    # No leak: no passage ID, no pooling detail, no forbidden term outside the texts, no grade.
    sent = html + (d / "send" / "INSTRUCTIONS.txt").read_text()
    assert not any(r["passage_id"] in sent for r in read_jsonl(pool_file))
    assert not any(s["detail"] in sent for r in rows for s in r["sources"])
    shell = re.sub(r'(id="payload">).*?(</script>)', r"\1\2", html, flags=re.S)
    shell += (d / "send" / "INSTRUCTIONS.txt").read_text()
    for label, pattern in FORBIDDEN.items():
        assert not re.search(pattern, shell, re.I), label
    assert '"grade"' not in html.split('id="payload">')[1].split("</script>")[0]
    assert "connect-src" not in html and "default-src 'none'" in html  # the page can make no network request
    # The zip holds exactly the two files sent.
    with zipfile.ZipFile(d / "send" / "gold_v1_p001_T1.zip") as zf:
        assert sorted(zf.namelist()) == ["gold_v1_p001_T1/INSTRUCTIONS.txt", "gold_v1_p001_T1/gold_v1_p001_T1.html"]
        assert zf.read("gold_v1_p001_T1/gold_v1_p001_T1.html") == (d / "send" / "gold_v1_p001_T1.html").read_bytes()
    instructions = (d / "send" / "INSTRUCTIONS.txt").read_text()
    assert "PACKAGE p001" in instructions and "gold_v1_p001_T1_grades_COMPLETE.json" in instructions
    # Packaging writes nothing else.
    assert before == (pool_file.read_bytes(), history_file.read_bytes(),
                      {p.name: p.read_bytes() for p in (stage["dir"] / "pool_v1").iterdir()})


def test_packaging_refuses_unapproved_graded_or_already_packaged_questions(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    pooled_v1(stage)
    code, _, err = run("remote-package", "--questions", "q_0001", annotator="claude")
    assert code == 1 and "need approved_pool_snapshot and primary_annotator" in err
    approve(stage, 1)
    register(stage, "T1")
    code, _, err = run("remote-package", "--questions", "q_0009", annotator="claude")
    assert code == 1 and "not approved questions" in err
    assert run("annotate", "q_0002", keys=["2", "q"], annotator="T1")[0] == 0  # a local grade exists
    code, _, err = run("remote-package", "--questions", "q_0002", annotator="claude")
    assert code == 1 and "already has grades" in err
    built(stage, "q_0001")
    code, _, err = run("remote-package", "--questions", "q_0001", annotator="claude")
    assert code == 1 and "already in a package: q_0001 (p001)" in err
    code, out, err = run("remote-package", "--remaining", "--batch-size", "5", annotator="claude")
    assert code == 0 and "package p002: 1 question(s)" in out  # q_0003 only: q_0001 packaged, q_0002 graded
    code, out, _ = run("remote-status")
    assert "p001  out for grading" in out and "p002  out for grading" in out
    code, _, err = run("annotate", "q_0001", keys=["q"], annotator="T1")
    assert code == 1 and "out for remote grading in package p001" in err


def test_synthetic_grades_import_into_the_append_only_history(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    ready(stage)
    d, _, payload = built(stage, "q_0001", "q_0002")
    history_file = stage["tmp"] / "annotation" / "history.jsonl"
    pool_file = stage["tmp"] / "annotation" / "pool.jsonl"
    data = synthetic(d, payload)
    first = next(iter(data["records"]))
    data["records"][first].update(p3=True, grade="2", note="synthetic comparative")
    data["records"][first]["history"] = [{"grade": "0", "p3": False, "p4": False, "assisted": False, "note": "",
                                          "at": "2026-11-14T09:59:00.000Z"},
                                         {k: data["records"][first][k] for k in ("grade", "p3", "p4", "assisted", "note",
                                                                                   "at")}]
    returned = write(stage["tmp"] / "gold_v1_p001_T1_grades_COMPLETE.json", data)
    hb, pb = history_file.read_bytes(), pool_file.read_bytes()
    code, out, err = run("remote-import", "--package", "p001", "--file", str(returned), "--dry-run", annotator="claude")
    assert code == 0, err
    assert "dry run: p001 from T1" in out and history_file.read_bytes() == hb and not (d / "returned").exists()
    code, out, err = run("remote-import", "--package", "p001", "--file", str(returned), annotator="claude")
    assert code == 0, err
    assert history_file.read_bytes().startswith(hb) and pool_file.read_bytes() == pb  # appended; pools unchanged
    events = [json.loads(line) for line in history_file.read_bytes()[len(hb):].decode().splitlines()]
    mapping = {m["id"]: m for m in json.loads((d / "private" / "mapping.json").read_text())["candidates"]}
    grades = [e for e in events if e["kind"] in ("grade", "skip")]
    assert len(grades) == len(mapping) and {e["annotator"] for e in events} == {"T1"}
    by_passage = {(e["query_id"], e["passage_id"]): e for e in grades}
    head = by_passage[(mapping[first]["query_id"], mapping[first]["passage_id"])]
    assert head["grade"] == 2 and head["note"] == "[P3] synthetic comparative" and head["previous"] is None
    assert [e["status"] for e in events if e["kind"] == "question_status"] == ["complete", "complete"]
    kept = d / "returned" / returned.name
    assert kept.read_bytes() == returned.read_bytes() and not kept.stat().st_mode & 0o222
    record = json.loads((d / "import_record.json").read_text())
    assert record["history"]["events_added"] == len(events) and record["sha256"] in out
    assert {p.name: p.read_bytes() for p in (stage["dir"] / "pool_v1").iterdir()}  # still readable
    code, _, err = run("remote-import", "--package", "p001", "--file", str(returned), annotator="claude")
    assert code == 1 and "already imported" in err  # never twice
    assert run("remote-status")[1].startswith("p001  imported")


def _mutations() -> list[tuple[str, Any, str]]:
    def first(d: dict[str, Any]) -> str:
        return next(iter(d["records"]))

    def missing(d: dict[str, Any]) -> None:
        del d["records"][first(d)]

    def fabricated(d: dict[str, Any]) -> None:
        d["records"]["c000000000000"] = copy.deepcopy(d["records"][first(d)])

    def wrong_question(d: dict[str, Any]) -> None:
        d["records"][first(d)]["query_id"] = "q_0003"

    def set_rec(**changes: Any) -> Any:
        def f(d: dict[str, Any]) -> None:
            r = d["records"][first(d)]
            r.update(changes)
            r["history"][-1].update(changes)
        return f

    def top(**changes: Any) -> Any:
        return lambda d: d.update(changes)

    def history_mismatch(d: dict[str, Any]) -> None:
        d["records"][first(d)]["history"][-1]["grade"] = "1" if d["records"][first(d)]["grade"] != "1" else "2"

    return [
        ("missing", missing, "candidate(s) missing"),
        ("fabricated", fabricated, "unknown (fabricated) candidate ID"),
        ("wrong question", wrong_question, "but belongs to"),
        ("1 without note", set_rec(grade="1", note=""), "a required note is missing"),
        ("skip without note", set_rec(grade="s", note=" "), "a required note is missing"),
        ("P3 without note", set_rec(grade="2", p3=True, note=""), "a required note is missing"),
        ("P4 0 without note", set_rec(grade="0", p4=True, note=""), "a required note is missing"),
        ("assistance without note", set_rec(grade="0", assisted=True, note=""), "a required note is missing"),
        ("P4 on a 2", set_rec(grade="2", p4=True, note="x note"), "P4 on a grade other than 0"),
        ("bad grade", set_rec(grade="3"), "grade '3'"),
        ("other annotator", top(annotator="T2"), "the package was issued to 'T1'"),
        ("AI annotator", top(annotator="claude"), "is an AI identity"),
        ("no declaration", top(declaration={"accepted": False, "text": "", "accepted_at": T}), "declaration"),
        ("progress file", top(complete=False), "a progress file, not completed grades"),
        ("question not done", lambda d: d["questions"].update({"q_0001": {"done_at": None}}), "not marked done"),
        ("other package data", top(payload_sha256="0" * 64), "different package data"),
        ("extra field", top(comment="hi"), "unexpected fields"),
        ("history mismatch", history_mismatch, "differs from the last history entry"),
    ]


@pytest.mark.parametrize(("label", "mutate", "message"), _mutations(), ids=[m[0] for m in _mutations()])
def test_a_returned_file_with_any_problem_is_refused_whole(stage: dict[str, Any], label: str, mutate: Any,  # noqa: F811
                                                           message: str) -> None:
    ready(stage)
    d, _, payload = built(stage, "q_0001")
    data = synthetic(d, payload)
    mutate(data)
    history_file = stage["tmp"] / "annotation" / "history.jsonl"
    hb = history_file.read_bytes()
    returned = write(stage["tmp"] / "returned.json", data)
    code, _, err = stage["run"]("remote-import", "--package", "p001", "--file", str(returned), annotator="claude")
    assert code == 1 and message in err, (label, err)
    assert history_file.read_bytes() == hb and not (d / "returned").exists() and not (d / "import_record.json").exists()


def test_duplicate_keys_and_existing_grades_are_refused(stage: dict[str, Any]) -> None:  # noqa: F811
    from finretriever.eval_set.annotation_store import AnnotationStore

    ready(stage)
    d, _, payload = built(stage, "q_0001")
    data = synthetic(d, payload)
    cid = next(iter(data["records"]))
    raw = json.dumps(data).replace('"records": {', f'"records": {{"{cid}": {json.dumps(data["records"][cid])}, ', 1)
    returned = write(stage["tmp"] / "dup.json", raw)
    code, _, err = stage["run"]("remote-import", "--package", "p001", "--file", str(returned), annotator="claude")
    assert code == 1 and "duplicate keys" in err
    # A grade that appeared locally meanwhile is never overwritten.
    from finretriever.eval_set.annotation_cli import load_gold_set_config
    from finretriever.eval_set.annotation_store import verify_corpus
    config = load_gold_set_config(stage["run"].config)
    store = AnnotationStore.open(config.workspace_dir, verify_corpus(config.corpus_path))
    store.judge("q_0001", store.pool("q_0001")[0].passage_id, 1, annotator="T1", note="local")
    returned = write(stage["tmp"] / "ok.json", synthetic(d, payload))
    code, _, err = stage["run"]("remote-import", "--package", "p001", "--file", str(returned), annotator="claude")
    assert code == 1 and "already has grades; nothing is overwritten" in err


NODE_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const [htmlPath, outPath] = process.argv.slice(2);
const html = fs.readFileSync(htmlPath, "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
new vm.Script(scripts[0]);  // the whole page script compiles
const payload = JSON.parse(html.match(/<script type="application\/json" id="payload">([\s\S]*?)<\/script>/)[1]);
const sha = scripts[0].match(/PAYLOAD_SHA256 = "([0-9a-f]{64})"/)[1];
const core = scripts[0].split("/* CORE-START */")[1].split("/* CORE-END */")[0];
const CORE = new Function("APP_VERSION", core + "\nreturn CORE;")("1");
let t = Date.UTC(2026, 10, 14, 9, 0, 0);
const now = () => new Date(t += 1000).toISOString();
const R = (g, o) => Object.assign({ grade: g, p3: false, p4: false, assisted: false, note: "" }, o || {});
const checks = {
  plain_0: CORE.complete(R("0")), plain_2: CORE.complete(R("2")),
  one_needs_note: !CORE.complete(R("1")), one_with_note: CORE.complete(R("1", { note: "partial" })),
  skip_needs_note: !CORE.complete(R("s")), p3_needs_note: !CORE.complete(R("2", { p3: true })),
  p4_needs_note: !CORE.complete(R("0", { p4: true })), p4_only_on_0: !CORE.complete(R("2", { p4: true, note: "xyz" })),
  assisted_needs_note: !CORE.complete(R("0", { assisted: true })), ungraded: !CORE.complete(R(null)),
};
const state = CORE.newState(payload, now());
state.annotator = payload.annotator;
state.declaration = { accepted: true, text: "synthetic declaration", accepted_at: now() };
let i = 0;
for (const q of payload.questions) {
  let refused = false; try { CORE.markDone(state, payload, q.query_id, now()); } catch (e) { refused = true; }
  checks["done_refused_while_open_" + q.query_id] = refused;
  for (const c of q.candidates) {
    const k = i++ % 5;
    if (k === 0) CORE.update(state, c.id, { grade: "0" }, now());
    if (k === 1) { CORE.update(state, c.id, { grade: "2" }, now()); CORE.update(state, c.id, { p3: true, note: "synthetic comparative" }, now()); }
    if (k === 2) CORE.update(state, c.id, { grade: "1", note: "synthetic partial" }, now());
    if (k === 3) { CORE.update(state, c.id, { p4: true }, now()); CORE.update(state, c.id, { grade: "0", note: "synthetic other year" }, now()); }
    if (k === 4) { CORE.update(state, c.id, { grade: "2" }, now()); CORE.update(state, c.id, { grade: "s", note: "synthetic undecided" }, now()); }
  }
  CORE.markDone(state, payload, q.query_id, now());
}
const q0 = payload.questions[0], c0 = q0.candidates[0].id;
CORE.update(state, c0, { note: "synthetic revised" }, now());
checks.change_reopens = state.questions[q0.query_id].done_at === null;
checks.history_kept = state.records[c0].history.length === 2;
CORE.markDone(state, payload, q0.query_id, now());
const res = CORE.results(state, payload, sha, now());
const back = CORE.restore(JSON.parse(JSON.stringify(res)), payload, sha);
checks.resume_roundtrip = JSON.stringify(CORE.results(back, payload, sha, res.saved_at)) === JSON.stringify(res);
let foreign = false; try { CORE.restore(Object.assign({}, res, { package_id: "p999" }), payload, sha); } catch (e) { foreign = true; }
checks.other_package_refused = foreign;
fs.writeFileSync(outPath, JSON.stringify(res));
console.log(JSON.stringify(checks));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_grading_page_logic_produces_a_file_the_import_accepts(stage: dict[str, Any]) -> None:  # noqa: F811
    ready(stage)
    d, _, payload = built(stage, "q_0001", "q_0002")
    harness = stage["tmp"] / "harness.js"
    harness.write_text(NODE_HARNESS)
    out = stage["tmp"] / "gold_v1_p001_T1_grades_COMPLETE.json"
    html = next((d / "send").glob("*.html"))
    proc = subprocess.run(["node", str(harness), str(html), str(out)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    checks = json.loads(proc.stdout)
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    code, msg, err = stage["run"]("remote-import", "--package", "p001", "--file", str(out), annotator="claude")
    assert code == 0, err
    events = [json.loads(line) for line in (stage["tmp"] / "annotation" / "history.jsonl").read_text().splitlines()]
    notes = [e["note"] for e in events if e["kind"] in ("grade", "skip")]
    assert any(n.startswith("[P3] ") for n in notes) and any(n.startswith("[P4] ") for n in notes)
    assert sum(1 for e in events if e["kind"] == "skip") >= 1


def test_filling_checks_the_template_not_the_data() -> None:
    from finretriever.eval_set.annotation_store import AnnotationError
    from finretriever.eval_set.remote_package import _fill

    data = "Signature ____________________ and __________________ blanks"  # as in real filings
    assert _fill("<p>__ID__</p><x>__DATA__</x>", {"__ID__": "p002", "__DATA__": data}) == f"<p>p002</p><x>{data}</x>"
    with pytest.raises(AnnotationError, match="differ from the values"):
        _fill("<p>__ID__</p>", {"__ID__": "p002", "__DATA__": data})
    with pytest.raises(AnnotationError, match="differ from the values"):
        _fill("<p>__ID__ __OTHER__</p>", {"__ID__": "p002"})
