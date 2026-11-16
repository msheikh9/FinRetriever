"""Gold set v1's question stage, through the command line: the worksheet, the validated import (human-written only),
the leakage check, the frozen question export, the pooling gate and enforced blind grading."""

from __future__ import annotations

import csv
import io
import json
import socket
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from finretriever.eval_set import annotation_cli
from finretriever.eval_set.annotation_cli import main
from finretriever.utils.io import read_jsonl
from tests.unit.eval_set_fixtures import write_config, write_corpus

PLAN = [
    {"slot": "S001", "tickers": "ACME", "topic": "risk_factors", "question_type": "semantic", "comparison": "none",
     "period": "FY2024", "fiscal_years": "2024"},
    {"slot": "S002", "tickers": "BETA", "topic": "debt_liquidity", "question_type": "exact_fact", "comparison": "none",
     "period": "FY2024", "fiscal_years": "2024"},
    {"slot": "S003", "tickers": "ACME", "topic": "financial_statements", "question_type": "table",
     "comparison": "cross_year", "period": "FY2023 vs FY2024", "fiscal_years": "2023;2024"},
]
QUESTIONS = {
    "S001": "How exposed is Acme to a single chip supplier?",
    "S002": "What borrowing capacity did Beta Bancorp have available at the end of 2024?",
    "S003": "How did Acme's reported revenue change between 2023 and 2024?",
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted during annotation")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.delenv("FINRETRIEVER_ANNOTATOR", raising=False)


class Run:
    def __init__(self, config: Path) -> None:
        self.config = config

    def __call__(self, *argv: str, keys: Iterable[str] = (), annotator: str | None = "T1") -> tuple[int, str, str]:
        pending = list(keys)

        def type_key(prompt: str) -> str:
            if not pending:
                raise EOFError
            return pending.pop(0)

        out, err = io.StringIO(), io.StringIO()
        args = [*argv, "--config", str(self.config)] + (["--annotator", annotator] if annotator else [])
        return main(args, input_fn=type_key, out=out, err=err), out.getvalue(), err.getvalue()


@pytest.fixture
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    plan = tmp_path / "plan.csv"
    with plan.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PLAN[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(PLAN)
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"pair_id": "train_1", "query": "What were Acme's 2024 net sales?"}) + "\n")
    # The leakage check's default sources are the real training and validation files, which are not under git: the
    # tests check against this fixture file instead, so they run the same in a fresh checkout (CI) as locally.
    monkeypatch.setattr(annotation_cli, "LEAKAGE_SOURCES", (str(training),))
    config = write_config(tmp_path / "gold.yaml", write_corpus(tmp_path / "corpus"), query_plan=str(plan),
                          question_stage_dir=str(tmp_path / "stage"), require_blind=True)
    run = Run(config)
    assert run("init")[0] == 0
    return {"run": run, "dir": tmp_path / "stage", "training": training, "tmp": tmp_path}


def fill(sheet: Path, rows: dict[str, dict[str, str]]) -> None:
    """Fill rows of the worksheet as a writer would."""
    with sheet.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, data = reader.fieldnames, list(reader)
    for row in data:
        row.update(rows.get(row["slot"], {}))
    with sheet.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def written(slot: str, /, **changes: str) -> dict[str, str]:
    return {"question": QUESTIONS[slot], "writer": "AB", "ai_assistance": "none", **changes}


def test_the_worksheet_has_one_empty_row_per_slot_and_is_never_overwritten(stage: dict[str, Any]) -> None:
    run = stage["run"]
    code, out, _ = run("question-sheet")
    assert code == 0 and "before any search" in out
    rows = list(csv.DictReader((stage["dir"] / "question_sheet.csv").open()))
    assert [r["slot"] for r in rows] == ["S001", "S002", "S003"]
    assert all(r["question"] == r["writer"] == r["ai_assistance"] == "" for r in rows)
    code, _, err = run("question-sheet")
    assert code == 1 and "never overwritten" in err


def test_import_adds_human_written_independent_questions_tagged_with_their_slot(stage: dict[str, Any]) -> None:
    run, sheet = stage["run"], stage["dir"] / "question_sheet.csv"
    run("question-sheet")
    fill(sheet, {"S001": written("S001"), "S003": written("S003", writer="CD")})
    code, out, err = run("import-questions")
    assert code == 0, err
    assert "imported 2 question(s)" in out and "q_0001, q_0002" in out
    questions = {q["query_id"]: q for q in read_jsonl(stage["tmp"] / "annotation" / "questions.jsonl")}
    assert questions["q_0001"]["notes"].startswith("[slot S001]") and questions["q_0002"]["notes"].startswith("[slot S003]")
    assert {q["authoring"] for q in questions.values()} == {"independent"}
    assert {q["ai_assisted"] for q in questions.values()} == {False}
    assert (questions["q_0001"]["author"], questions["q_0002"]["author"]) == ("AB", "CD")
    assert questions["q_0002"]["fiscal_years"] == [2023, 2024] and questions["q_0002"]["comparison"] == "cross_year"
    code, out, _ = run("import-questions")  # importing again changes nothing
    assert code == 0 and "imported 0 question(s)" in out and "2 already present unchanged" in out


@pytest.mark.parametrize(("changes", "message"), [
    ({"writer": "claude"}, "is an AI identity"),
    ({"writer": "Claude-Opus-5-5"}, "is an AI identity"),
    ({"ai_assistance": "yes"}, "AI-assisted questions are refused"),
    ({"ai_assistance": ""}, "AI-assisted questions are refused"),
    ({"writer": ""}, "no writer"),
    ({"tickers": "BETA"}, "keeps its companies and question type"),
    ({"question_type": "exact_fact"}, "keeps its companies and question type"),
    ({"topic": "mdna"}, "without a 'replaced: <reason>' note"),
    ({"slot": "S999"}, "not a slot of the plan"),
])
def test_import_refuses_and_adds_nothing(stage: dict[str, Any], changes: dict[str, str], message: str) -> None:
    run, sheet = stage["run"], stage["dir"] / "question_sheet.csv"
    run("question-sheet")
    fill(sheet, {"S001": written("S001", **changes), "S002": written("S002")})
    code, _, err = run("import-questions")
    assert code == 1 and message in err and "nothing imported" in err
    assert not (stage["tmp"] / "annotation" / "questions.jsonl").read_text().strip()  # all or nothing


def test_a_replaced_slot_needs_its_reason_and_a_slot_is_never_rewritten_silently(stage: dict[str, Any]) -> None:
    run, sheet = stage["run"], stage["dir"] / "question_sheet.csv"
    run("question-sheet")
    fill(sheet, {"S001": written("S001", topic="mdna", notes="replaced: no single-supplier risk is disclosed")})
    assert run("import-questions")[0] == 0
    fill(sheet, {"S001": written("S001", topic="mdna", notes="replaced: x", question="Another wording entirely?")})
    code, _, err = run("import-questions")
    assert code == 1 and "already imported as q_0001 with other text" in err


def test_pooling_waits_for_the_frozen_question_export_which_needs_a_passing_current_leakage_check(
        stage: dict[str, Any]) -> None:
    run, sheet, training = stage["run"], stage["dir"] / "question_sheet.csv", str(stage["training"])
    run("question-sheet")
    fill(sheet, {"S001": written("S001"), "S002": written("S002")})
    run("import-questions")
    code, _, err = run("pool", "q_0001")
    assert code == 1 and "not frozen yet" in err  # no pooling before the leakage check and the export
    code, _, err = run("export-questions")
    assert code == 1 and "no leakage check" in err and "1 slot(s) without a question: ['S003']" in err
    assert run("leakage-check", "--against", training)[0] == 0
    fill(sheet, {"S003": written("S003")})
    run("import-questions")
    code, _, err = run("export-questions")
    assert code == 1 and "not current" in err  # a question was added after the check
    code, out, _ = run("leakage-check", "--against", training)
    assert code == 0 and "recorded in" in out
    record = json.loads((stage["dir"] / "leakage_check.json").read_text())
    assert record["all_pass"] and record["questions"] == 3 and set(record["results"]) == {"q_0001", "q_0002", "q_0003"}
    assert QUESTIONS["S001"] not in json.dumps(record["results"])  # texts are hashed, not copied, into the results
    code, out, err = run("export-questions")
    assert code == 0, err
    assert "froze version 1: 3 questions" in out
    frozen = read_jsonl(stage["dir"] / "questions_frozen.jsonl")
    assert [(r["query_id"], r["slot"]) for r in frozen] == [("q_0001", "S001"), ("q_0002", "S002"), ("q_0003", "S003")]
    assert run("pool", "q_0001", "--reformulation", "supplier concentration")[0] == 0
    code, _, err = run("export-questions")
    assert code == 1 and "identical to version 1" in err  # a new version only when the questions changed


def test_a_question_too_close_to_a_training_question_cannot_be_frozen(stage: dict[str, Any]) -> None:
    run, sheet = stage["run"], stage["dir"] / "question_sheet.csv"
    run("question-sheet")
    fill(sheet, {s: written(s) for s in QUESTIONS} | {"S002": written("S002", question="What were Acme's 2024 net sales?")})
    run("import-questions")
    code, out, _ = run("leakage-check", "--against", str(stage["training"]))
    assert code == 1 and "REWRITE" in out
    code, _, err = run("export-questions")
    assert code == 1 and "at or above the leakage threshold: ['q_0002']" in err


def test_grading_is_blind_whenever_the_config_requires_it(stage: dict[str, Any]) -> None:
    run = stage["run"]
    run("add-question", "--text", "How dependent is Acme on a single semiconductor supplier?", "--topic", "risk_factors",
        "--type", "semantic", "--ticker", "ACME", "--authoring", "independent")
    assert run("pool", "q_0001", "--reformulation", "supplier disruption")[0] == 0  # no slot: not gated
    code, out, _ = run("annotate", "q_0001", keys=["q"])  # no --blind flag given
    assert code == 0 and "blind: pool sources, ranks and scores hidden" in out
    assert {line for line in out.splitlines() if line.startswith("source")} == {"source    hidden (blind grading)"}


def frozen_three(stage: dict[str, Any]) -> None:
    """Write, import, check and freeze version 1 of the three-slot fixture."""
    run, sheet = stage["run"], stage["dir"] / "question_sheet.csv"
    run("question-sheet")
    fill(sheet, {s: written(s) for s in QUESTIONS})
    assert run("import-questions")[0] == 0
    assert run("leakage-check", "--against", str(stage["training"]))[0] == 0
    assert run("export-questions")[0] == 0


def fill_revision(path: Path, revised: dict[str, dict[str, str]]) -> None:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, data = reader.fieldnames, list(reader)
    for row in data:
        row.update(revised.get(row["slot"], {}))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


REVISED = {"S002": "How much undrawn borrowing capacity did Beta Bancorp report at the end of 2024?",
           "S003": "What revenue did Acme report for 2024 compared with 2023?"}


def revision(slot: str, **changes: str) -> dict[str, str]:
    return {"revised_question": REVISED[slot], "writer": "AB", "ai_assistance": "none", **changes}


def test_a_revision_keeps_version_1_and_freezes_version_2_with_its_history(stage: dict[str, Any]) -> None:
    run, directory = stage["run"], stage["dir"]
    frozen_three(stage)
    v1 = {name: (directory / name).read_bytes() for name in ("questions_frozen.jsonl", "questions_frozen.manifest.json",
                                                              "leakage_check_v1.json")}
    code, out, _ = run("revision-sheet", "--slots", "S002,S003", "--reason", "one information need per question")
    assert code == 0 and "revision_sheet_v2.csv" in out
    sheet = directory / "revision_sheet_v2.csv"
    rows = list(csv.DictReader(sheet.open()))
    assert [(r["slot"], r["current_query_id"], r["original_writer"]) for r in rows] == [
        ("S002", "q_0002", "AB"), ("S003", "q_0003", "AB")]
    assert all(r["revised_question"] == "" and r["current_question"] == QUESTIONS[r["slot"]] for r in rows)
    fill_revision(sheet, {s: revision(s) for s in REVISED})
    code, out, err = run("revise-questions", "--sheet", str(sheet))
    assert code == 0, err
    assert "S002: q_0002 withdrawn, revision added as q_0004" in out and "S003: q_0003 withdrawn" in out
    questions = {q["query_id"]: q for q in read_jsonl(stage["tmp"] / "annotation" / "questions.jsonl")}
    assert questions["q_0002"]["status"] == "withdrawn" and questions["q_0001"]["status"] == "open"
    assert questions["q_0004"]["notes"].startswith("[slot S002] revision of q_0002: one information need")
    assert (questions["q_0004"]["topic"], questions["q_0004"]["tickers"], questions["q_0004"]["fiscal_years"]) == (
        "debt_liquidity", ["BETA"], [2024])  # the slot's metadata is kept
    history = (stage["tmp"] / "annotation" / "history.jsonl").read_text()
    assert "revised by AB: one information need per question" in history  # the withdrawal is recorded
    code, _, err = run("export-questions")
    assert code == 1 and "not current" in err  # the leakage check must run again on the revised set
    assert run("leakage-check", "--against", str(stage["training"]))[0] == 0
    code, out, err = run("export-questions")
    assert code == 0, err
    assert "froze version 2: 3 questions in questions_frozen_v2.jsonl" in out and "supersedes version 1" in out
    assert all((directory / name).read_bytes() == data for name, data in v1.items())  # version 1 is untouched
    manifest = json.loads((directory / "questions_frozen_v2.manifest.json").read_text())
    assert manifest["supersedes"]["version"] == 1 and [r["slot"] for r in manifest["revisions"]] == ["S002", "S003"]
    assert manifest["revisions"][0]["previous_query_id"] == "q_0002" and manifest["revisions"][0]["query_id"] == "q_0004"
    assert not (directory / "questions_frozen_v2.jsonl").stat().st_mode & 0o222  # read-only
    code, _, err = run("pool", "q_0002")
    assert code == 1 and "not in the frozen question export" in err  # pooling follows the latest version
    assert run("pool", "q_0004", "--reformulation", "borrowing capacity")[0] == 0
    (directory / "questions_frozen.jsonl").chmod(0o644)
    (directory / "questions_frozen.jsonl").write_text("tampered\n")
    code, _, err = run("export-questions")
    assert code == 1 and "version 1" in err and "differs from its manifest" in err  # earlier versions are verified


@pytest.mark.parametrize(("slot_changes", "message"), [
    ({"writer": "CD"}, "the slot's original writer is 'AB'"),
    ({"writer": "claude"}, "is an AI identity"),
    ({"ai_assistance": "assisted"}, "AI-assisted questions are refused"),
    ({"revised_question": ""}, "not revised"),
    ({"revised_question": QUESTIONS["S002"]}, "is the current one"),
    ({"revised_question": QUESTIONS["S001"]}, "repeats an existing question"),
    ({"topic": "mdna"}, "a revision keeps the slot's metadata"),
    ({"current_question": "Something else?"}, "stale sheet"),
])
def test_a_revision_refuses_and_changes_nothing(stage: dict[str, Any], slot_changes: dict[str, str],
                                                message: str) -> None:
    run, directory = stage["run"], stage["dir"]
    frozen_three(stage)
    run("revision-sheet", "--slots", "S002,S003", "--reason", "one information need per question")
    sheet = directory / "revision_sheet_v2.csv"
    fill_revision(sheet, {"S002": revision("S002", **slot_changes), "S003": revision("S003")})
    before = (stage["tmp"] / "annotation" / "questions.jsonl").read_bytes()
    code, _, err = run("revise-questions", "--sheet", str(sheet))
    assert code == 1 and message in err and "nothing revised" in err
    assert (stage["tmp"] / "annotation" / "questions.jsonl").read_bytes() == before


def test_a_provisional_slot_is_flagged_when_pooled(stage: dict[str, Any], tmp_path: Path) -> None:
    frozen_three(stage)
    config = stage["run"].config
    config.write_text(config.read_text() + 'provisional_slots: {"S001": "check that the filings answer it"}\n')
    code, out, _ = stage["run"]("pool", "q_0001", "--reformulation", "supplier")
    assert code == 0 and "NOTE q_0001 (S001) is provisional: check that the filings answer it" in out


def pooled_v1(stage: dict[str, Any]) -> None:
    """Freeze version 1 of the questions, pool each one (base pool) and freeze pool snapshot version 1."""
    frozen_three(stage)
    for qid in ("q_0001", "q_0002", "q_0003"):
        assert stage["run"]("pool", qid, annotator="claude")[0] == 0
    assert stage["run"]("pool-snapshot", annotator="claude")[0] == 0


REWORDINGS = {"q_0001": ["sole source semiconductor vendor dependence", "single supplier chip concentration risk"],
              "q_0002": ["undrawn committed credit facility availability"],
              "q_0003": ["total revenues year over year change 2024 2023"]}


def fill_rewordings(path: Path, changes: dict[str, dict[str, str]] | None = None) -> None:
    """Fill a copy of the issued sheet (which is read-only) and put the copy at ``path``."""
    issued = path.with_name("rewording_sheet_pool_v2.csv")
    with issued.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, data = reader.fieldnames, list(reader)
    for row in data:
        texts = REWORDINGS[row["query_id"]]
        row.update({f"rewording_{i}": t for i, t in enumerate(texts, start=1)}, writer="EF", ai_assistance="none")
        row.update((changes or {}).get(row["query_id"], {}))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def test_the_rewording_sheet_lists_every_approved_question_with_context_and_empty_fields(stage: dict[str, Any]) -> None:
    pooled_v1(stage)
    code, out, _ = stage["run"]("rewording-sheet")
    assert code == 0 and "3 questions of question export version 1" in out and "4 rewordings required" in out
    rows = list(csv.DictReader((stage["dir"] / "rewording_sheet_pool_v2.csv").open()))
    assert [(r["query_id"], r["slot"], r["rewordings_required"]) for r in rows] == [
        ("q_0001", "S001", "2"), ("q_0002", "S002", "1"), ("q_0003", "S003", "1")]  # semantic: 2; the others: 1
    assert rows[0]["question"] == QUESTIONS["S001"] and rows[2]["period"] == "FY2023 vs FY2024"
    assert all(r[f] == "" for r in rows for f in ("rewording_1", "rewording_2", "rewording_3", "writer", "ai_assistance"))
    assert not (stage["dir"] / "rewording_sheet_pool_v2.csv").stat().st_mode & 0o222  # issued read-only
    code, _, err = stage["run"]("rewording-sheet")
    assert code == 1 and "never overwritten" in err


def test_applying_rewordings_adds_to_the_pools_and_freezes_snapshot_2_keeping_snapshot_1(stage: dict[str, Any]) -> None:
    run, directory = stage["run"], stage["dir"]
    pooled_v1(stage)
    run("rewording-sheet")
    sheet = directory / "rewording_sheet_pool_v2_completed.csv"
    fill_rewordings(sheet)
    pool_file = stage["tmp"] / "annotation" / "pool.jsonl"
    before = pool_file.read_bytes()
    v1 = {p.name: p.read_bytes() for p in (directory / "pool_v1").iterdir()}
    code, out, err = run("apply-rewordings", "--sheet", str(sheet), "--dry-run", annotator="claude")
    assert code == 0, err
    assert "dry run: 3 questions, 4 reworded searches" in out
    assert pool_file.read_bytes() == before and not (directory / "pool_v2").exists()  # a dry run writes nothing
    code, out, err = run("apply-rewordings", "--sheet", str(sheet), annotator="claude")
    assert code == 0, err
    assert "froze pool snapshot version 2" in out and "parent version 1" in out
    assert {p.name: p.read_bytes() for p in (directory / "pool_v1").iterdir()} == v1  # snapshot 1 untouched
    rows = read_jsonl(pool_file)
    old = {(r["query_id"], r["passage_id"]) for r in map(json.loads, before.decode().splitlines())}
    reformulated = [r for r in rows if any("reformulation by EF" in s["detail"] for s in r["sources"])]
    assert reformulated  # provenance: the rewording and its writer are in the source record
    for r in rows:
        if (r["query_id"], r["passage_id"]) in old:
            assert r["sources"][0]["method"] in ("bm25", "random", "stratified")  # existing rows keep their first source
    assert len({(r["query_id"], r["passage_id"]) for r in rows}) == len(rows)  # deduplicated
    manifest = json.loads((directory / "pool_v2" / "pool_manifest.json").read_text())
    assert manifest["parent"]["version"] == 1 and manifest["inputs"]["question_export"]["version"] == 1
    assert manifest["inputs"]["rewordings"]["q_0001"] == {"writer": "EF", "count": 2}
    assert manifest["inputs"]["issued_sheet"]["file"] == "rewording_sheet_pool_v2.csv"
    assert manifest["inputs"]["rewording_sheet"]["file"] == "rewording_sheet_pool_v2_completed.csv"
    assert set(manifest["per_question_counts"]) == {"q_0001", "q_0002", "q_0003"}
    assert not (directory / "pool_v2" / "pool_snapshot.jsonl").stat().st_mode & 0o222  # read-only
    after = pool_file.read_bytes()
    code, _, err = run("apply-rewordings", "--sheet", str(sheet), annotator="claude")
    assert code == 1 and "already applied (pool snapshot version 2)" in err  # one round
    assert pool_file.read_bytes() == after and not (directory / "pool_v3").exists()


def test_reworded_searches_need_an_unchanged_frozen_base_pool(stage: dict[str, Any]) -> None:
    run, directory = stage["run"], stage["dir"]
    frozen_three(stage)
    for qid in ("q_0001", "q_0002", "q_0003"):
        run("pool", qid, annotator="claude")
    code, _, err = run("rewording-sheet")
    assert code == 1 and "freeze the base pool" in err
    assert run("pool-snapshot", annotator="claude")[0] == 0
    run("rewording-sheet")
    sheet = directory / "rewording_sheet_pool_v2_completed.csv"
    fill_rewordings(sheet)
    pool_file = stage["tmp"] / "annotation" / "pool.jsonl"
    lines = pool_file.read_text().splitlines()
    pool_file.write_text("\n".join(lines[:-1]) + "\n")  # the workspace drifts from snapshot 1
    code, _, err = run("apply-rewordings", "--sheet", str(sheet), "--dry-run", annotator="claude")
    assert code == 1 and "differ from pool snapshot version 1" in err


@pytest.mark.parametrize(("changes", "message"), [
    ({"q_0001": {"rewording_2": ""}}, "the protocol requires 2"),
    ({"q_0002": {"rewording_1": "", "rewording_2": "undrawn committed credit facility availability"}}, "in order"),
    ({"q_0003": {"rewording_1": QUESTIONS["S003"]}}, "repeats the question itself"),
    ({"q_0001": {"rewording_2": "sole source semiconductor vendor dependence"}}, "two rewordings are the same"),
    ({"q_0002": {"writer": "claude"}}, "is an AI identity"),
    ({"q_0002": {"ai_assistance": "some"}}, "ai_assistance must be exactly 'none'"),
    ({"q_0003": {"question": "Another question?"}}, "context columns changed"),
    ({"q_0003": {"rewording_1": "What were Acme's 2024 net sales?"}}, "leakage threshold"),
])
def test_a_rewording_sheet_with_any_problem_is_refused_whole(stage: dict[str, Any], changes: dict[str, dict[str, str]],
                                                             message: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.eval_set import annotation_cli

    monkeypatch.setattr(annotation_cli, "LEAKAGE_SOURCES", (str(stage["training"]),))
    run, directory = stage["run"], stage["dir"]
    pooled_v1(stage)
    run("rewording-sheet")
    sheet = directory / "rewording_sheet_pool_v2_completed.csv"
    fill_rewordings(sheet, changes)
    before = (stage["tmp"] / "annotation" / "pool.jsonl").read_bytes()
    code, _, err = run("apply-rewordings", "--sheet", str(sheet), annotator="claude")
    assert code == 1 and message in err and "nothing applied" in err
    assert (stage["tmp"] / "annotation" / "pool.jsonl").read_bytes() == before and not (directory / "pool_v2").exists()


def test_a_missing_question_and_another_company_are_reported(stage: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.eval_set import annotation_cli

    monkeypatch.setattr(annotation_cli, "LEAKAGE_SOURCES", (str(stage["training"]),))
    run, directory = stage["run"], stage["dir"]
    pooled_v1(stage)
    run("rewording-sheet")
    sheet = directory / "rewording_sheet_pool_v2_completed.csv"
    fill_rewordings(sheet, {"q_0002": {"rewording_1": "undrawn credit facility versus Apple liquidity"}})
    code, out, err = run("apply-rewordings", "--sheet", str(sheet), "--dry-run", annotator="claude")
    assert code == 0, err
    assert "observation: q_0002: a rewording names ['AAPL'], outside its slot" in out
    lines = sheet.read_text().splitlines()
    sheet.write_text("\n".join(lines[:-1]) + "\n")  # drop the last question
    code, _, err = run("apply-rewordings", "--sheet", str(sheet), "--dry-run", annotator="claude")
    assert code == 1 and "each of the 3 approved questions once" in err


def test_a_spreadsheet_saved_sheet_with_a_byte_order_mark_and_crlf_is_read(stage: dict[str, Any]) -> None:
    run, directory = stage["run"], stage["dir"]
    pooled_v1(stage)
    run("rewording-sheet")
    sheet = directory / "rewording_sheet_pool_v2_completed.csv"
    fill_rewordings(sheet)
    sheet.write_bytes(b"\xef\xbb\xbf" + sheet.read_bytes().replace(b"\n", b"\r\n"))
    code, out, err = run("apply-rewordings", "--sheet", str(sheet), "--dry-run", annotator="claude")
    assert code == 0, err
    assert "4 reworded searches" in out
    sheet.write_bytes(sheet.read_bytes().replace(b"Acme", b"Acm\xe9"))  # Latin-1, not UTF-8
    code, _, err = run("apply-rewordings", "--sheet", str(sheet), "--dry-run", annotator="claude")
    assert code == 1 and "not UTF-8" in err


def test_validate_only_runs_every_check_but_no_search(stage: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.eval_set import annotation_cli, rewording

    def no_search(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a BM25 search ran")

    run, directory = stage["run"], stage["dir"]
    pooled_v1(stage)  # the base pool uses BM25; from here on, no search may run
    monkeypatch.setattr(rewording.CandidateGenerator, "bm25", no_search)
    run("rewording-sheet")
    sheet = directory / "rewording_sheet_pool_v2_completed.csv"
    fill_rewordings(sheet)
    before = (stage["tmp"] / "annotation" / "pool.jsonl").read_bytes()
    code, out, err = run("apply-rewordings", "--sheet", str(sheet), "--validate-only", annotator="claude")
    assert code == 0, err
    assert "valid: 3 questions of question export version 1" in out and "4 rewordings by ['EF']" in out
    assert "extends pool snapshot version 1; no search run, nothing written" in out
    assert (stage["tmp"] / "annotation" / "pool.jsonl").read_bytes() == before and not (directory / "pool_v2").exists()
    monkeypatch.setattr(annotation_cli, "LEAKAGE_SOURCES", (str(stage["training"]),))
    fill_rewordings(sheet, {"q_0003": {"rewording_1": "What were Acme's 2024 net sales?"}})
    code, _, err = run("apply-rewordings", "--sheet", str(sheet), "--validate-only", annotator="claude")
    assert code == 1 and "leakage threshold" in err


def approve(stage: dict[str, Any], version: int, sha256: str | None = None) -> str:
    """Record a pool snapshot as approved in the gold-set config, as the reviewer's approval does."""
    manifest = json.loads((stage["dir"] / f"pool_v{version}" / "pool_manifest.json").read_text())
    sha = sha256 or manifest["snapshot"]["sha256"]
    with stage["run"].config.open("a") as fh:
        fh.write(f"approved_pool_snapshot: {json.dumps({'version': version, 'sha256': sha})}\n")
    return sha


def test_an_approved_pool_snapshot_gates_grading_and_closes_pooling(stage: dict[str, Any]) -> None:
    run = stage["run"]
    pooled_v1(stage)
    sha = approve(stage, 1)
    code, out, err = run("annotate", "q_0001", keys=["q"])
    assert code == 0, err
    assert f"grading pool snapshot version 1 ({sha}): verified, unchanged" in out
    assert "source    hidden (blind grading)" in out and "bm25" not in out
    for argv in (("pool", "q_0001"), ("pool-snapshot",), ("apply-rewordings", "--sheet", "x.csv")):
        code, _, err = run(*argv, annotator="claude")
        assert code == 1 and "pooling is closed" in err, argv
    code, out, _ = run("history", "q_0001")
    assert "details hidden (blind grading)" in out and "bm25" not in out  # pooling notes are hidden while blind
    assert len(list((stage["dir"]).glob("pool_v*"))) == 1


def test_grading_stops_if_the_pools_drift_from_the_approved_snapshot(stage: dict[str, Any]) -> None:
    run = stage["run"]
    pooled_v1(stage)
    approve(stage, 1)
    pool_file = stage["tmp"] / "annotation" / "pool.jsonl"
    lines = pool_file.read_text().splitlines()
    pool_file.write_text("\n".join(lines[:-1]) + "\n")  # one candidate removed after the approval
    for argv in (("annotate", "q_0001"), ("regrade-init",)):
        code, _, err = run(*argv, keys=["q"])
        assert code == 1 and "differ from the approved pool snapshot version 1" in err, argv


def test_grading_refuses_a_snapshot_other_than_the_approved_one(stage: dict[str, Any]) -> None:
    run = stage["run"]
    pooled_v1(stage)
    approve(stage, 1, sha256="0" * 64)
    code, _, err = run("annotate", "q_0001", keys=["q"])
    assert code == 1 and "is not the approved one" in err


def test_with_an_approved_pool_grading_serves_only_its_questions(stage: dict[str, Any]) -> None:
    run = stage["run"]
    # An older, non-slot (pilot) question, open with pending candidates, listed before the slot questions.
    code, _, err = run("add-question", "--text", "Which Acme risks did management list?", "--ticker", "ACME",
                       "--topic", "risk_factors", "--type", "semantic", "--year", "2024", "--authoring", "independent")
    assert code == 0, err
    assert run("pool", "q_0001", annotator="claude")[0] == 0
    frozen_three(stage)
    for qid in ("q_0002", "q_0003", "q_0004"):
        assert run("pool", qid, annotator="claude")[0] == 0
    assert run("pool-snapshot", annotator="claude")[0] == 0
    approve(stage, 1)
    code, out, err = run("annotate", keys=["q"])  # the default skips the pilot question
    assert code == 0, err
    assert "q_0002 ·" in out and "verified, unchanged" in out
    code, _, err = run("annotate", "q_0001", keys=["q"])
    assert code == 1 and "not one of the 3 questions of the approved pool snapshot" in err


def register(stage: dict[str, Any], initials: str) -> None:
    with stage["run"].config.open("a") as fh:
        fh.write(f"primary_annotator: {json.dumps(initials)}\n")


def test_only_the_registered_primary_annotator_grades_the_approved_questions(stage: dict[str, Any]) -> None:
    run = stage["run"]
    pooled_v1(stage)
    approve(stage, 1)
    register(stage, "T1")
    code, out, err = run("annotate", "q_0001", keys=["q"], annotator="T1")
    assert code == 0 and "verified, unchanged" in out, err
    code, _, err = run("annotate", "q_0001", keys=["q"], annotator="T2")
    assert code == 1 and "not a registered primary annotator" in err and "(T1)" in err
    code, _, err = run("annotate", "q_0001", keys=["q"], annotator="claude")
    assert code == 1 and "AI identity" in err


def test_status_shows_only_the_approved_questions_and_no_pool_sources(stage: dict[str, Any]) -> None:
    run = stage["run"]
    code, _, err = run("add-question", "--text", "Which Acme risks did management list?", "--ticker", "ACME",
                       "--topic", "risk_factors", "--type", "semantic", "--year", "2024", "--authoring", "independent")
    assert code == 0, err
    assert run("pool", "q_0001", annotator="claude")[0] == 0
    run("annotate", "q_0001", keys=["0 an earlier pilot grade", "q"])
    frozen_three(stage)
    for qid in ("q_0002", "q_0003", "q_0004"):
        run("pool", qid, annotator="claude")
    run("pool-snapshot", annotator="claude")
    code, out, _ = run("status", "--json")
    assert "q_0001" in out and '"sources"' not in out  # blind: no source counts even before the approval
    approve(stage, 1)
    code, out, _ = run("status")
    assert code == 0 and "the 3 questions of approved pool snapshot version 1 only" in out
    assert "q_0001 " not in out and "graded 0 (0: 0" in out  # the pilot question and its grade are out of view
    data = json.loads(run("status", "--json")[1])
    assert [q["query_id"] for q in data["queries"]] == ["q_0002", "q_0003", "q_0004"]
