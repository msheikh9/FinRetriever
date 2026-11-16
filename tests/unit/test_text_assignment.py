"""Plain-text grading assignments for two primary annotators: the split, the documents, and the validated import.

Synthetic grades only, on the fixture corpus; no real question or grade is involved."""

from __future__ import annotations

import json
import random
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from finretriever.eval_set.annotation_store import AnnotationError
from finretriever.eval_set.remote_package import FORBIDDEN
from finretriever.eval_set.text_assignment import parse_responses, split_note, split_questions
from finretriever.utils.io import read_jsonl
from tests.unit.test_question_stage import (  # noqa: F401 (stage is a fixture)
    approve,
    pooled_v1,
    stage,
)
from tests.unit.test_remote_package import built, synthetic, write


def two(stage: dict[str, Any], names: tuple[str, str] = ("T1", "T2")) -> None:  # noqa: F811
    pooled_v1(stage)
    approve(stage, 1)
    with stage["run"].config.open("a") as fh:
        fh.write(f"primary_annotators: {json.dumps(list(names))}\n")


def split_and_build(stage: dict[str, Any], t1: int = 1, t2: int = 2) -> dict[str, Path]:  # noqa: F811
    run = stage["run"]
    code, out, err = run("assign-split", "--count", f"T1={t1}", "--count", f"T2={t2}", annotator="claude")
    assert code == 0, err
    dirs = {}
    for who in ("T1", "T2"):
        code, out, err = run("assign-build", "--for", who, annotator="claude")
        assert code == 0, err
        dirs[who] = stage["dir"] / "assignments" / re.search(r"assignment (a\d{3})", out).group(1)
    return dirs


def files(d: Path) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]]:
    doc = next((d / "send").glob("*_ASSIGNMENT.txt")).read_text(encoding="utf-8-sig")
    resp = next((d / "send").glob("*_RESPONSES.txt")).read_text(encoding="utf-8-sig")
    manifest = json.loads((d / "manifest.json").read_text())
    mapping = json.loads((d / "private" / "mapping.json").read_text())["candidates"]
    return doc, resp, manifest, mapping


def filled(template: str, grade: Callable[[int, str, str], str] | None = None, *, initials: str = "T1",
           yes: str = "YES", only: set[str] | None = None) -> str:
    """The response template filled in as an annotator would: grades cycling 0, 2, 1 (note), s (note)."""
    grade = grade or (lambda i, qid, cid: ["0 |", "2 |", "1 | synthetic partial note", "s | synthetic skip note"][i % 4])
    out, qid, i = [], None, 0
    for line in template.splitlines():
        if line.startswith("INITIALS:"):
            line = f"INITIALS: {initials}"
        elif line.startswith("GRADED ALONE"):
            line = f"GRADED ALONE, WITH NO AI TOOLS (YES): {yes}"
        elif line.startswith("== q_"):
            qid = line.split()[1]
        elif re.match(r"^c[0-9a-f]{12} \|  \|$", line) and (only is None or qid in only):
            cid = line.split()[0]
            line = f"{cid} | {grade(i, qid, cid)}"
            i += 1
        out.append(line)
    return "\n".join(out) + "\n"


def history_lines(stage: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: F811
    return list(read_jsonl(stage["tmp"] / "annotation" / "history.jsonl"))


# --- the split -----------------------------------------------------------------------------------------------------


def test_the_split_is_seeded_stratified_and_balanced_with_completed_work_counted() -> None:
    rng = random.Random(7)
    sizes = {f"q_{i:04d}": rng.randint(20, 120) for i in range(12, 111)}
    types = {q: ["exact_fact", "semantic", "table", "multi_passage"][i % 4] for i, q in enumerate(sorted(sizes))}
    types["q_0011"] = "exact_fact"
    kwargs = {"credited": {"AA": ["q_0011"]}, "credited_sizes": {"q_0011": 45}, "credited_types": types}
    a = split_questions(sizes, types, {"AA": 49, "BB": 50}, **kwargs)
    b = split_questions(sizes, types, {"AA": 49, "BB": 50}, **kwargs)
    assert a == b  # reproducible
    assert len(a["assignments"]["AA"]) == 49 and len(a["assignments"]["BB"]) == 50
    assert set(a["assignments"]["AA"]) | set(a["assignments"]["BB"]) == set(sizes)
    assert not set(a["assignments"]["AA"]) & set(a["assignments"]["BB"])
    t = a["totals"]
    assert t["AA"]["total_passages"] == 45 + sum(sizes[q] for q in a["assignments"]["AA"])
    assert a["passage_difference"] == abs(t["AA"]["total_passages"] - t["BB"]["total_passages"]) <= 1
    for kind in set(types.values()):  # each type split as evenly as the counts allow, q_0011 counted for AA
        assert abs(t["AA"]["types"].get(kind, 0) - t["BB"]["types"].get(kind, 0)) <= 1
    with pytest.raises(AnnotationError, match="do not add up"):
        split_questions(sizes, types, {"AA": 49, "BB": 49}, **kwargs)


# --- building -------------------------------------------------------------------------------------------------------


def test_each_assignment_holds_every_passage_of_its_questions_and_nothing_else(stage: dict[str, Any]) -> None:  # noqa: F811
    two(stage)
    pool_file = stage["tmp"] / "annotation" / "pool.jsonl"
    history_file = stage["tmp"] / "annotation" / "history.jsonl"
    before = (pool_file.read_bytes(), history_file.read_bytes())
    dirs = split_and_build(stage)
    split = json.loads((stage["dir"] / "assignments" / "split_v1.json").read_text())
    assert sorted(split["assignments"]["T1"] + split["assignments"]["T2"]) == ["q_0001", "q_0002", "q_0003"]
    assert "crosscheck_rule" in split and split["seed"] == "gold-split-v1"
    rows = list(read_jsonl(pool_file))
    for who, d in dirs.items():
        doc, resp, manifest, mapping = files(d)
        assert manifest["annotator"] == who and [q["query_id"] for q in manifest["questions"]] == \
            split["assignments"][who]
        assert sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()) == sorted([
            "manifest.json", "private/mapping.json", f"send/gold_v1_{d.name}_{who}_ASSIGNMENT.txt",
            f"send/gold_v1_{d.name}_{who}_RESPONSES.txt", f"send/gold_v1_{d.name}_{who}.zip"])
        assert all(not p.stat().st_mode & 0o222 for p in d.rglob("*") if p.is_file())
        for q in manifest["questions"]:
            qid = q["query_id"]
            expected = [r["passage_id"] for r in sorted((r for r in rows if r["query_id"] == qid),
                                                        key=lambda r: r["position"])]
            random.Random(f"gold-blind-v1:{qid}").shuffle(expected)
            got = [m["passage_id"] for m in mapping if m["query_id"] == qid]
            assert got == expected  # every pooled passage, once, in the approved blind order
            ids_in_doc = re.findall(rf"--- {qid}  -  passage \d+ of \d+  -  ID (c[0-9a-f]{{12}}) ---", doc)
            assert ids_in_doc == [m["id"] for m in mapping if m["query_id"] == qid]
        # The response file lists every ID once, empty, in the same order.
        assert re.findall(r"^(c[0-9a-f]{12}) \|  \|$", resp, re.M) == [m["id"] for m in mapping]
        assert f"CHECK {manifest['check']}" in resp and "INITIALS:" in resp
        # No leak: no passage ID, no pooling detail, no forbidden term outside the texts.
        sent = doc + resp
        assert not any(r["passage_id"] in sent for r in rows)
        assert not any(s["detail"] in sent for r in rows for s in r["sources"])
        shell = doc
        texts = re.findall(r"^Filing:  [^\n]*\nSection: [^\n]*\n\n(.*?)\n(?=\n--- |\n=+\n)", doc, re.M | re.S)
        assert len(texts) == len(mapping)
        for t in texts:
            shell = shell.replace(t, "")
        shell = re.sub(r"^(Filing|Section): .*$", "", shell, flags=re.M)
        shell = re.sub(r"^QUESTION \d+ of \d+  -  q_\d{4}  -  \d+ passages\n.*$", "", shell, flags=re.M)
        for label, pattern in FORBIDDEN.items():
            assert not re.search(pattern, shell + resp, re.I), label
        with zipfile.ZipFile(next((d / "send").glob("*.zip"))) as zf:
            assert sorted(zf.namelist()) == sorted(f"gold_v1_{d.name}_{who}/{n.name}"
                                                   for n in (d / "send").glob("*.txt"))
    assert before == (pool_file.read_bytes(), history_file.read_bytes())  # building writes nothing else


def test_building_refuses_questions_already_out_or_assigned(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    two(stage)
    code, _, err = run("remote-package", "--questions", "q_0001", annotator="claude")
    assert code == 1 and "pass --for" in err
    built_dir = run("remote-package", "--questions", "q_0001", "--for", "T1", annotator="claude")
    assert built_dir[0] == 0, built_dir[2]
    code, _, err = run("assign-split", "--count", "T1=1", "--count", "T2=2", annotator="claude")
    assert code == 1 and "withdraw or finish them first" in err
    code, out, err = run("remote-withdraw", "--package", "p001", "--reason", "plan changed", annotator="claude")
    assert code == 0 and "p001 withdrawn" in out
    assert "p001  withdrawn" in run("remote-status")[1]
    code, _, err = run("remote-withdraw", "--package", "p001", "--reason", "again", annotator="claude")
    assert code == 1 and "already withdrawn" in err
    dirs = split_and_build(stage)
    code, _, err = run("assign-build", "--for", "T1", annotator="claude")
    assert code == 1 and "already assigned" in err
    code, _, err = run("remote-package", "--questions", "q_0001", "--for", "T1", annotator="claude")
    assert code == 1 and "already in a plain-text assignment" in err
    qid = json.loads((dirs["T1"] / "manifest.json").read_text())["questions"][0]["query_id"]
    code, _, err = run("annotate", qid, keys=["q"], annotator="T1")
    assert code == 1 and "assign-import" in err
    code, _, err = run("annotate", qid, keys=["q"], annotator="T9")
    assert code == 1


def test_a_withdrawn_package_is_never_imported(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    two(stage)
    code, out, err = run("remote-package", "--questions", "q_0001", "--for", "T1", annotator="claude")
    assert code == 0, err
    d = stage["dir"] / "remote" / "p001"
    from tests.unit.test_remote_package import page_payload
    payload = page_payload(next((d / "send").glob("*.html")).read_text())
    returned = write(stage["tmp"] / "grades.json", synthetic(d, payload))
    assert run("remote-import", "--package", "p001", "--file", str(returned), "--dry-run", annotator="claude")[0] == 0
    assert run("remote-withdraw", "--package", "p001", "--reason", "plan changed", annotator="claude")[0] == 0
    code, _, err = run("remote-import", "--package", "p001", "--file", str(returned), annotator="claude")
    assert code == 1 and "withdrawn" in err


def test_a_package_that_was_imported_cannot_be_withdrawn(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    pooled_v1(stage)
    approve(stage, 1)
    with stage["run"].config.open("a") as fh:
        fh.write('primary_annotator: "T1"\n')
    d, html, payload = built(stage, "q_0001")
    returned = write(stage["tmp"] / "grades.json", synthetic(d, payload))
    assert run("remote-import", "--package", "p001", "--file", str(returned), annotator="claude")[0] == 0
    code, _, err = run("remote-withdraw", "--package", "p001", "--reason", "x", annotator="claude")
    assert code == 1 and "already withdrawn or imported" in err


# --- the returned response file -------------------------------------------------------------------------------------


def test_fully_graded_questions_are_recorded_once_and_partial_ones_wait(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    two(stage)
    dirs = split_and_build(stage)
    d = dirs["T2"]
    _, resp, manifest, mapping = files(d)
    first, second = (q["query_id"] for q in manifest["questions"])
    before = len(history_lines(stage))
    # Only the first question graded, and the second half done.
    some = filled(resp, initials="T2", only={first})
    half = set([m["id"] for m in mapping if m["query_id"] == second][:1])
    some = "\n".join(f"{ln.split()[0]} | 0 |" if ln.split()[:1] and ln.split()[0] in half else ln
                     for ln in some.splitlines()) + "\n"
    f1 = write(stage["tmp"] / "resp1.txt", some)
    code, out, err = run("assign-import", "--assignment", d.name, "--file", str(f1), "--dry-run", annotator="claude")
    assert code == 0, err
    assert "to record: 1 question(s)" in out and f"partly graded (waiting): {second}" in out
    assert len(history_lines(stage)) == before  # a dry run writes nothing
    code, out, err = run("assign-import", "--assignment", d.name, "--file", str(f1), annotator="claude")
    assert code == 0, err
    events = history_lines(stage)[before:]
    n_first = sum(1 for m in mapping if m["query_id"] == first)
    assert len(events) == n_first + 1 and all(e["annotator"] == "T2" for e in events)
    assert events[-1]["kind"] == "question_status" and events[-1]["query_id"] == first
    notes = [e["note"] for e in events if e["kind"] in ("grade", "skip")]
    assert "synthetic partial note" in notes and "synthetic skip note" in notes
    assert [e["kind"] for e in events[:4]] == ["grade", "grade", "grade", "skip"][:min(4, n_first)]
    # The same file again: nothing new.
    code, _, err = run("assign-import", "--assignment", d.name, "--file", str(f1), annotator="claude")
    assert code == 1 and "no newly completed question" in err
    # The whole file: the second question is recorded, the first repeats unchanged.
    f2 = write(stage["tmp"] / "resp2.txt", filled(resp, initials="T2"))
    code, out, err = run("assign-import", "--assignment", d.name, "--file", str(f2), annotator="claude")
    assert code == 0 and "already recorded, unchanged: 1" in out, err
    assert "2/2 questions recorded" in run("assign-status")[1]
    # A changed grade on a recorded question is refused whole.
    changed = filled(resp, initials="T2", grade=lambda i, q, c: "2 |" if i == 0 else
                     ["0 |", "2 |", "1 | synthetic partial note", "s | synthetic skip note"][i % 4])
    f3 = write(stage["tmp"] / "resp3.txt", changed)
    code, _, err = run("assign-import", "--assignment", d.name, "--file", str(f3), annotator="claude")
    assert code == 1 and "already recorded, and this file changes" in err
    assert sorted(p.name for p in (d / "imports").iterdir()) == ["001", "002"]
    assert all(not p.stat().st_mode & 0o222 for p in (d / "imports").rglob("*") if p.is_file())


def test_an_emailed_response_is_read_despite_quoting_wrapping_and_crlf(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    two(stage)
    d = split_and_build(stage)["T1"]
    _, resp, _, _ = files(d)
    body = filled(resp, grade=lambda i, q, c: "1 | the figure is there but the period is only\nin the heading above")
    quoted = "Hi, here are my grades.\n\n" + "\r\n".join("> " + ln for ln in body.splitlines()) + "\r\n\nThanks\n"
    f = write(stage["tmp"] / "pasted.txt", quoted)
    code, out, err = run("assign-import", "--assignment", d.name, "--file", str(f), "--dry-run", annotator="claude")
    assert code == 0, err
    assert "joined to the note of" in out and "to record: 1 question(s)" in out
    utf16 = stage["tmp"] / "utf16.txt"
    utf16.write_bytes(filled(resp).encode("utf-16"))
    code, out, err = run("assign-import", "--assignment", d.name, "--file", str(utf16), "--dry-run", annotator="claude")
    assert code == 0, err


def _bad() -> list[tuple[str, Callable[[str, list[dict[str, Any]]], str], str]]:
    def replace_line(i: int, new: Callable[[str], str]) -> Callable[[str, list[dict[str, Any]]], str]:
        def f(text: str, mapping: list[dict[str, Any]]) -> str:
            lines = text.splitlines()
            idx = [k for k, ln in enumerate(lines) if re.match(r"^c[0-9a-f]{12} \|", ln)][i]
            lines[idx] = new(lines[idx])
            return "\n".join(lines) + "\n"
        return f

    def cid(line: str) -> str:
        return line.split()[0]

    return [
        ("initials", lambda t, m: t.replace("INITIALS: T1", "INITIALS: T2"), "was issued to T1"),
        ("ai-initials", lambda t, m: t.replace("INITIALS: T1", "INITIALS: claude"), "was issued to T1"),
        ("no-yes", lambda t, m: t.replace("TOOLS (YES): YES", "TOOLS (YES): "), "YES declaration is missing"),
        ("check", lambda t, m: re.sub(r"CHECK [0-9a-f]{12}", "CHECK 000000000000", t), "expected"),
        ("fabricated", replace_line(0, lambda ln: "c0123456789ab | 0 |"), "unknown (fabricated)"),
        ("duplicate", lambda t, m: t.replace("\n== END ==", f"\n{m[0]['id']} | 0 |\n== END =="), "appear twice"),
        ("missing", lambda t, m: "\n".join(ln for ln in t.splitlines() if not ln.startswith(m[1]["id"])) + "\n",
         "ID line(s) missing"),
        ("grade-3", replace_line(0, lambda ln: f"{cid(ln)} | 3 |"), "is not 2, 1, 0 or s"),
        ("one-no-note", replace_line(0, lambda ln: f"{cid(ln)} | 1 |"), "required note is missing"),
        ("skip-no-note", replace_line(0, lambda ln: f"{cid(ln)} | s |"), "required note is missing"),
        ("p3-no-note", replace_line(0, lambda ln: f"{cid(ln)} | 2 | P3"), "required note is missing"),
        ("assisted-no-note", replace_line(0, lambda ln: f"{cid(ln)} | 2 | ASSISTED"), "required note is missing"),
        ("p4-on-2", replace_line(0, lambda ln: f"{cid(ln)} | 2 | P4 another year's copy"), "P4 goes only with a 0"),
        ("skip-p3", replace_line(0, lambda ln: f"{cid(ln)} | s | P3 a comparative"), "a skip with P3 or P4"),
        ("no-pipe", replace_line(0, lambda ln: f"{cid(ln)} 2"), "is not written as 'ID | grade | note'"),
    ]


@pytest.mark.parametrize(("label", "mutate", "message"), _bad(), ids=[b[0] for b in _bad()])
def test_a_response_with_any_problem_is_refused_whole(stage: dict[str, Any], label: str, mutate: Any,  # noqa: F811
                                                      message: str) -> None:
    run = stage["run"]
    two(stage)
    d = split_and_build(stage)["T1"]
    _, resp, _, mapping = files(d)
    if len(mapping) < 2:
        pytest.skip("needs two passages")
    before = (stage["tmp"] / "annotation" / "history.jsonl").read_bytes()
    f = write(stage["tmp"] / "bad.txt", mutate(filled(resp), mapping))
    code, _, err = run("assign-import", "--assignment", d.name, "--file", str(f), annotator="claude")
    assert code == 1 and message in err, err
    assert (stage["tmp"] / "annotation" / "history.jsonl").read_bytes() == before
    assert not (d / "imports").exists()


def test_an_id_under_another_question_is_refused(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    two(stage)
    d = split_and_build(stage)["T2"]
    _, resp, manifest, mapping = files(d)
    first, second = (q["query_id"] for q in manifest["questions"])
    moved = next(m["id"] for m in mapping if m["query_id"] == second)
    lines = [ln for ln in filled(resp, initials="T2").splitlines() if not ln.startswith(moved)]
    at = lines.index(f"== {second} - {sum(1 for m in mapping if m['query_id'] == second)} passages ==")
    lines.insert(at - 1, f"{moved} | 0 |")
    f = write(stage["tmp"] / "moved.txt", "\n".join(lines) + "\n")
    code, _, err = run("assign-import", "--assignment", d.name, "--file", str(f), annotator="claude")
    assert code == 1 and f"is under {first}, but belongs to {second}" in err


def test_marks_become_note_tags() -> None:
    assert split_note("P3: FY2022 column of the FY2023 report") == ({"P3"}, "FY2022 column of the FY2023 report")
    assert split_note("p4 - same paragraph, FY2021") == ({"P4"}, "same paragraph, FY2021")
    assert split_note("ASSISTED P3 asked a colleague") == ({"ASSISTED", "P3"}, "asked a colleague")
    assert split_note("Pension plan note") == (set(), "Pension plan note")
    parsed = parse_responses(b"x ASSIGNMENT a001 - T1 - CHECK 0123456789ab\n== q_0001 - 1 passages ==\n"
                             b"c0123456789ab | 2 | a | b\n== END ==\n")
    assert parsed["lines"][0]["note"] == "a | b" and parsed["header"]["check"] == "0123456789ab"


def test_the_config_registers_people_once_and_never_an_ai(stage: dict[str, Any]) -> None:  # noqa: F811
    run = stage["run"]
    pooled_v1(stage)
    original = run.config.read_text()
    run.config.write_text(original + 'primary_annotator: "T1"\nprimary_annotators: ["T1", "T2"]\n')
    code, _, err = run("assign-status")
    assert code != 0 and "not both" in err
    run.config.write_text(original + 'primary_annotators: ["T1", "claude"]\n')
    code, _, err = run("assign-status")
    assert code != 0 and "AI identity" in err
