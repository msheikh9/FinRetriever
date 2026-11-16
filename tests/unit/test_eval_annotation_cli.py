"""Tests for the gold-set annotation CLI and scripts/03_annotate_gold_set.py.

``main(argv, input_fn=..., out=...)`` runs in-process with scripted keyboard
input against the synthetic corpus. Network access is blocked throughout.
"""

from __future__ import annotations

import importlib.util
import io
import json
import socket
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from finretriever.eval_set.annotation_cli import main
from finretriever.utils.io import read_jsonl
from tests.unit.eval_set_fixtures import passage_ids, write_config, write_corpus

IDS = passage_ids()
_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "03_annotate_gold_set.py"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted during annotation")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.delenv("FINRETRIEVER_ANNOTATOR", raising=False)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(tmp_path / "gold.yaml", write_corpus(tmp_path / "corpus"))


class Run:
    """Runs CLI commands against one config, feeding scripted keyboard input."""

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
        code = main(args, input_fn=type_key, out=out, err=err)
        return code, out.getvalue(), err.getvalue()


@pytest.fixture
def run(config: Path) -> Run:
    return Run(config)


@pytest.fixture
def pooled(run: Run) -> Run:
    assert run("init")[0] == 0
    code, out, _ = run(
        "add-question", "--text", "How dependent is Acme on a single semiconductor supplier?",
        "--topic", "risk_factors", "--type", "semantic", "--ticker", "ACME", "--authoring", "independent",
    )
    assert (code, out.splitlines()[0]) == (0, "[03_annotate_gold_set] added q_0001: How dependent is Acme on a single semiconductor supplier?")
    code, out, _ = run(
        "pool", "q_0001", "--reformulation", "supplier disruption",
        "--passage-id", IDS["acme24_table"], "--manual-note", "found by reading Item 8",
    )
    assert code == 0 and "pool now 5" in out
    return run


def test_session_shows_candidate_context_and_saves_each_judgment(pooled: Run) -> None:
    code, out, _ = pooled("annotate", "q_0001", keys=["2", "1 covers 2023 only", "s unsure", "q"])
    assert code == 0
    assert "Q: How dependent is Acme on a single semiconductor supplier?" in out
    assert f"passage   {IDS['acme24_supplier']}" in out
    assert "filing    Acme Devices Inc. (ACME) · 10-K filed 2025-02-20 · accession 0000000001-25-000001" in out
    assert "section   Item 1A · Risk Factors" in out
    assert "source    bm25 #1 (score " in out and "query='supplier disruption'" in out
    assert "saved q_0001: graded 2 · skipped 1 · pending 2 of 5 (3 changes this session)" in out
    _, out, _ = pooled("annotate", "q_0001", keys=["g 5", "q"])
    assert "source    manual — found by reading Item 8" in out

    code, out, _ = pooled("history", "q_0001")
    assert code == 0
    assert "pending -> 1 — covers 2023 only" in out and "pending -> skipped — unsure" in out


def test_resume_starts_at_the_first_pending_candidate_and_edits_old_ones(pooled: Run) -> None:
    pooled("annotate", "q_0001", keys=["2", "0", "q"])
    code, out, _ = pooled("annotate", "q_0001", keys=["g 2", "1 on reflection", "q"])
    assert code == 0
    assert out.splitlines()[0] == "q_0001: graded 2 · skipped 0 · pending 3 of 5 — type ? for commands"
    assert "candidate 3/5" in out.split("judge> ")[0]  # resumed at the first pending one
    _, history, _ = pooled("history", "q_0001", IDS["acme23_supplier"])
    assert "pending -> 0" in history and "0 -> 1 — on reflection" in history


def test_unjudged_candidates_never_become_qrels(pooled: Run, tmp_path: Path) -> None:
    pooled("annotate", "q_0001", keys=["2", "0", "s", "s", "s", "done"])
    code, out, _ = pooled("status")
    assert code == 0 and "q_0001   complete     5      2   1   0   1    3    0  skipped_candidates" in out
    code, out, _ = pooled("export", "--out-dir", str(tmp_path / "eval"))
    assert code == 0 and "exported 1 queries and 2 qrels (0: 1, 1: 0, 2: 1)" in out
    qrels = read_jsonl(tmp_path / "eval" / "gold_qrels.jsonl")
    assert sorted(q["relevance"] for q in qrels) == [0, 2]
    report = json.loads((tmp_path / "eval" / "gold_eval.report.json").read_text())
    assert report["exported"]["unjudged_pool_entries"] == {"q_0001": 3}


def test_complete_questions_must_be_reopened_to_edit(pooled: Run) -> None:
    code, _, err = pooled("complete", "q_0001")
    assert code == 1 and "5 candidates without a grade or skip" in err
    pooled("annotate", "q_0001", keys=["2", "2", "0", "0", "s", "done"])
    code, _, err = pooled("annotate", "q_0001", keys=["0"])
    assert code == 1 and "run `reopen q_0001`" in err
    assert pooled("reopen", "q_0001")[0] == 0
    code, out, _ = pooled("annotate", "q_0001", keys=["g 1", "1", "q"])
    assert code == 0 and "(1 changes this session)" in out


def test_session_handles_navigation_and_bad_input(pooled: Run) -> None:
    code, out, _ = pooled("annotate", "q_0001", keys=["?", "g 99", "x", "5", "f", "h", "n", "p", "c"])
    assert code == 0
    assert "s [note]    skip: leave it unjudged" in out
    assert "! usage: g N with N from 1 to 5" in out
    assert "! unknown command 'x'" in out and "! unknown command '5'" in out
    assert "no judgments recorded for this passage" in out
    assert "nothing to clear: this passage is already pending" in out
    assert out.rstrip().endswith("(0 changes this session)")  # input ran out: everything saved


def test_hide_scores_hides_ranks_and_scores_but_keeps_them_stored(pooled: Run, config: Path) -> None:
    code, out, _ = pooled("annotate", "q_0001", "--hide-scores", keys=["g 5", "q"])
    assert code == 0
    assert "retrieval scores and ranks hidden" in out.splitlines()[0]
    sources = [line for line in out.splitlines() if line.startswith("source")]
    assert sources and all("#" not in line and "score" not in line for line in sources)
    assert "source    bm25 — query='supplier disruption'; tickers=ACME; depth=5" in sources
    assert "source    manual — found by reading Item 8" in sources
    pool = read_jsonl(config.parent / "annotation" / "pool.jsonl")
    assert pool[0]["sources"][0]["rank"] == 1 and pool[0]["sources"][0]["score"] > 0
    _, shown, _ = pooled("annotate", "q_0001", keys=["q"])
    assert "source    bm25 #1 (score " in shown  # default display is unchanged


def test_blind_grading_hides_every_source_and_shuffles_the_order_but_stores_the_same(
        pooled: Run, config: Path) -> None:
    pool_file = config.parent / "annotation" / "pool.jsonl"
    stored_before = pool_file.read_bytes()
    stored_order = [row["passage_id"] for row in read_jsonl(pool_file)]

    def shown_order(out: str) -> list[str]:
        return [line.split()[1] for line in out.splitlines() if line.startswith("passage   ")]

    code, out, _ = pooled("annotate", "q_0001", "--blind", keys=["n", "n", "n", "n", "q"])
    assert code == 0
    assert "blind: pool sources, ranks and scores hidden; order shuffled" in out.splitlines()[0]
    sources = [line for line in out.splitlines() if line.startswith("source")]
    assert sources and set(sources) == {"source    hidden (blind grading)"}
    for hidden in ("bm25", "manual", "random", "stratified", "#", "score", "query=", "Item 8"):
        assert all(hidden not in line for line in sources), hidden
    blind_order = shown_order(out)
    assert sorted(blind_order) == sorted(stored_order) and blind_order != stored_order  # a shuffle of the same pool
    _, again, _ = pooled("annotate", "q_0001", "--blind", keys=["n", "n", "n", "n", "q"])
    assert shown_order(again) == blind_order  # the same order in every session

    # Grading in blind mode judges the passage shown, and changes nothing stored but the history.
    code, out, _ = pooled("annotate", "q_0001", "--blind", keys=["2", "q"])
    assert code == 0 and "graded 1" in out
    first_shown = blind_order[0]
    _, history, _ = pooled("history", "q_0001")
    assert first_shown in history and "pending -> 2" in history
    assert pool_file.read_bytes() == stored_before
    _, default, _ = pooled("annotate", "q_0001", keys=["q"])
    assert "source    bm25 #" in default  # the default display is unchanged


def test_leakage_check_flags_a_question_that_repeats_a_training_question(pooled: Run, tmp_path: Path) -> None:
    training = tmp_path / "train.jsonl"
    training.write_text(
        json.dumps({"pair_id": "p1", "query": "How dependent is Acme on a single semiconductor supplier?"}) + "\n"
        + json.dumps({"pair_id": "p2", "query": "What was Acme's revenue in 2023?"}) + "\n"
    )
    code, out, _ = pooled("leakage-check", "--against", str(training))
    assert code == 1
    assert "q_0001  1.000  REWRITE closest train.jsonl:p1" in out and "1 question(s) at or above 0.8" in out
    training.write_text(json.dumps({"pair_id": "p2", "query": "What was Acme's revenue in 2023?"}) + "\n")
    code, out, _ = pooled("leakage-check", "--against", str(training))
    assert code == 0 and "q_0001" in out and " ok" in out


def test_ai_assisted_questions_record_their_reviewer(run: Run, config: Path) -> None:
    run("init")
    base = ["add-question", "--topic", "risk_factors", "--type", "semantic", "--ticker", "ACME"]
    code, out, _ = run(
        *base, "--text", "Which supplier risks does Acme describe?", "--authoring", "passage_inspired",
        "--ai-assisted", "--reviewed-by", "RV", "--notes", "approved without changes", annotator="claude",
    )
    assert code == 0 and "provenance: AI-assisted, author claude, reviewed by RV" in out
    question = read_jsonl(config.parent / "annotation" / "questions.jsonl")[0]
    assert (question["author"], question["ai_assisted"], question["reviewed_by"], question["authoring"]) == (
        "claude", True, "RV", "passage_inspired",
    )
    code, _, err = run(*base, "--text", "Is Acme profitable?", "--authoring", "independent",
                       "--ai-assisted", "--reviewed-by", "RV", annotator="claude")
    assert code == 1 and "cannot be recorded as independently authored" in err
    code, _, err = run(*base, "--text", "Is Acme growing?", "--authoring", "passage_inspired",
                       "--ai-assisted", annotator="claude")
    assert code == 1 and "need reviewed_by" in err


def test_corpus_that_does_not_match_the_frozen_hash_is_refused(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "corpus")
    config = write_config(tmp_path / "gold.yaml", corpus, corpus_sha256="0" * 64)
    code, _, err = Run(config)("init")
    assert code == 2 and "frozen to 000000000000" in err
    config = write_config(tmp_path / "gold.yaml", corpus, corpus_version="corpus_v1@1.2")
    code, _, err = Run(config)("status")
    assert code == 2 and "frozen to corpus_v1@1.2" in err


def test_edited_corpus_is_refused_even_by_an_existing_workspace(pooled: Run, config: Path) -> None:
    corpus = config.parent / "corpus" / "corpus.jsonl"
    corpus.write_text(corpus.read_text().replace("Taiwan", "Korea"), encoding="utf-8")
    code, _, err = pooled("status")
    assert code == 2 and "hashes to" in err


def test_changes_need_an_annotator(run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    code, _, err = run("init", annotator=None)
    assert code == 1 and "pass --annotator" in err
    monkeypatch.setenv("FINRETRIEVER_ANNOTATOR", "T9")
    assert run("init", annotator=None)[0] == 0


def test_search_and_show_are_read_only_lookups(run: Run) -> None:
    code, out, _ = run("search", "credit facility", "--ticker", "BETA", "--top-k", "1")
    assert code == 0 and out.splitlines()[0] == f"  1. {IDS['beta24_liquidity']}"
    assert "LIQUIDITY RISK MANAGEMENT (appended report section; no Item number)" in out
    code, out, _ = run("show", IDS["acme24_table"])
    assert code == 0 and "Net sales | 2024: $1,200 | 2023: $1,000" in out
    assert run("show", "passage_missing")[0] == 1


def test_script_runs_the_cli() -> None:
    spec = importlib.util.spec_from_file_location("annotate_gold_set_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_blind_regrade_measures_agreement_between_two_people(pooled: Run, config: Path, tmp_path: Path) -> None:
    """D7 re-grading: a separate workspace with no grade, note, source or rank; the second person grades blind with
    `audit`; the report compares the two people's grades once, and refuses unfair comparisons."""
    first = ["2", "1", "0", "0", "s unsure"]
    code, out, _ = pooled("annotate", "q_0001", keys=[*first, "done"])
    assert code == 0 and "q_0001 marked complete" in out
    regrade_dir = tmp_path / "regrade"
    code, out, _ = pooled("regrade-init", "--regrade-dir", str(regrade_dir), "--questions", "1", "--seed", "7")
    assert code == 0 and "blind re-grade of 5 passages across 1 questions" in out
    # Nothing of the first grading is copied: no grade, no note, no pool source or rank.
    history = [json.loads(line) for line in (regrade_dir / "history.jsonl").read_text().splitlines()]
    assert history and all(event["grade"] is None for event in history)
    for name in ("pool.jsonl", "history.jsonl", "audit.json"):
        text = (regrade_dir / name).read_text()
        assert "unsure" not in text and "bm25" not in text and "supplier disruption" not in text
    code, _, err = pooled("regrade-report", "--regrade-dir", str(regrade_dir))
    assert code == 1 and "incomplete" in err

    code, out, _ = pooled("audit", "--audit-dir", str(regrade_dir), keys=["2", "2", "0", "1", "0", "q"], annotator="T2")
    assert code == 0 and "bm25" not in out and "#1" not in out  # blind: no source or rank shown
    code, out, err = pooled("regrade-report", "--regrade-dir", str(regrade_dir))
    assert code == 0, err
    report = json.loads((regrade_dir / "regrade_report.json").read_text())
    assert report["items"] == 5 and report["compared"] + report["skipped"]["first"] == 5
    assert report["skipped"]["first"] == 1 and report["cohen_kappa"] is not None
    agreed = sum(cell["agree"] for cell in report["per_question"].values())
    assert len(report["disagreements"]) == report["compared"] - agreed > 0  # every disagreement is listed
    assert all(d["first"] != d["second"] and d["query_id"] == "q_0001" for d in report["disagreements"])
    assert "disagreement(s)" in out and "Cohen's kappa" in out and "never deciding" in out
    code, _, err = pooled("regrade-report", "--regrade-dir", str(regrade_dir))
    assert code == 1 and "reported once" in err


def test_regrade_refuses_one_person_on_both_sides_and_changed_first_grades(pooled: Run, tmp_path: Path) -> None:
    pooled("annotate", "q_0001", keys=["2", "1", "0", "0", "0", "done"])
    same = tmp_path / "same"
    assert pooled("regrade-init", "--regrade-dir", str(same), "--questions", "1")[0] == 0
    pooled("audit", "--audit-dir", str(same), keys=["2", "1", "0", "0", "0", "q"], annotator="T1")
    code, _, err = pooled("regrade-report", "--regrade-dir", str(same))
    assert code == 1 and "graded both sides" in err

    changed = tmp_path / "changed"
    assert pooled("regrade-init", "--regrade-dir", str(changed), "--questions", "1")[0] == 0
    pooled("reopen", "q_0001")
    pooled("annotate", "q_0001", keys=["1", "q"])  # the first annotator edits a grade after the sample was drawn
    pooled("audit", "--audit-dir", str(changed), keys=["2", "1", "0", "0", "0", "q"], annotator="T2")
    code, _, err = pooled("regrade-report", "--regrade-dir", str(changed))
    assert code == 1 and "changed after the re-grade sample was drawn" in err
    code, _, err = pooled("regrade-init", "--regrade-dir", str(changed), "--questions", "1")
    assert code == 1 and "already holds" in err


def test_weighted_kappa_known_answers() -> None:
    from finretriever.eval_set.regrade import weighted_kappa

    assert weighted_kappa([(0, 0), (1, 1), (2, 2)]) == pytest.approx(1.0)
    assert weighted_kappa([]) is None
    # One full disagreement (0 vs 2) costs more than one adjacent disagreement (1 vs 2).
    far = weighted_kappa([(0, 0), (1, 1), (2, 2), (0, 2)])
    near = weighted_kappa([(0, 0), (1, 1), (2, 2), (1, 2)])
    assert far is not None and near is not None and far < near
