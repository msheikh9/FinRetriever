"""Unit tests for finretriever.eval_set.annotation_store (synthetic corpus, offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finretriever.eval_set.annotation_store import (
    HISTORY_FILE,
    POOL_FILE,
    QUESTIONS_FILE,
    AnnotationError,
    AnnotationStore,
    load_corpus_passages,
    verify_corpus,
)
from finretriever.eval_set.schemas import Candidate, CandidateSource
from finretriever.evaluation.provenance import CorpusIntegrityError, CorpusMismatchError
from tests.unit.eval_set_fixtures import (
    fixed_clock,
    fixture_passages,
    new_store,
    passage_ids,
    reopen,
    write_corpus,
)

IDS = passage_ids()
BM25 = CandidateSource(method="bm25", rank=1, score=2.5, detail="query='supplier'")


def add_question(store: AnnotationStore, text: str = "How dependent is Acme on one supplier?", **kw):
    values = {"topic": "risk_factors", "question_type": "semantic", "authoring": "independent", "author": "T1"}
    return store.add_question(text, **{**values, **kw})


def pooled(store: AnnotationStore, *names: str, query_id: str = "q_0001") -> list[str]:
    store.add_candidates(query_id, [Candidate(passage_id=IDS[n], source=BM25) for n in names], added_by="T1")
    return [IDS[n] for n in names]


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    store = new_store(tmp_path)
    add_question(store, tickers=["ACME"])
    pooled(store, "acme24_supplier", "acme23_supplier", "acme24_competition", "acme24_sales")
    return store


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- corpus binding ------------------------------------------------------------------


def test_corpus_must_match_its_manifest(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "corpus")
    corpus.write_text(corpus.read_text().replace("Taiwan", "Taiwan "), encoding="utf-8")
    with pytest.raises(CorpusIntegrityError, match="hashes to"):
        verify_corpus(corpus)


def test_corpus_must_match_the_frozen_identity(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "corpus")
    identity = verify_corpus(corpus)
    assert verify_corpus(corpus, expected_sha256=identity.corpus_sha256, expected_version=identity.corpus_version)
    with pytest.raises(CorpusMismatchError, match="frozen to 000000000000"):
        verify_corpus(corpus, expected_sha256="0" * 64)
    with pytest.raises(CorpusMismatchError, match="frozen to corpus_v1@1.2"):
        verify_corpus(corpus, expected_version="corpus_v1@1.2")


def test_workspace_refuses_a_different_corpus(tmp_path: Path, store: AnnotationStore) -> None:
    rebuilt = fixture_passages()
    rebuilt[0] = rebuilt[0].model_copy(update={"text": "Different text behind the same passage ID."})
    other = verify_corpus(write_corpus(tmp_path / "rebuilt", rebuilt))
    with pytest.raises(CorpusMismatchError, match="point at different text"):
        AnnotationStore.open(store.directory, other)
    relabelled = verify_corpus(write_corpus(tmp_path / "relabelled", version="fixture_corpus@9.9"))
    with pytest.raises(CorpusMismatchError, match="corpus_version"):
        AnnotationStore.open(store.directory, relabelled)


def test_corpus_with_duplicate_passage_ids_is_rejected(tmp_path: Path) -> None:
    passages = fixture_passages()
    corpus = verify_corpus(write_corpus(tmp_path / "corpus", [*passages, passages[0]]))
    with pytest.raises(CorpusIntegrityError, match="duplicate passage_id"):
        load_corpus_passages(corpus)


def test_create_never_overwrites_a_workspace(store: AnnotationStore) -> None:
    with pytest.raises(AnnotationError, match="already holds annotation files"):
        AnnotationStore.create(store.directory, store.corpus, annotation_version="gold_test", created_by="T1")
    info = json.loads((store.directory / "workspace.json").read_text())
    assert info["evaluation"] == {
        "corpus_sha256": store.corpus.corpus_sha256,
        "corpus_version": "fixture_corpus@1.2",
        "annotation_version": "gold_test",
    }


# --- questions -------------------------------------------------------------------------


def test_query_ids_are_sequential_and_questions_unique(store: AnnotationStore) -> None:
    second = add_question(store, "What were Beta's credit losses?", topic="financial_statements", tickers=["beta"])
    assert [q.query_id for q in store.questions()] == ["q_0001", "q_0002"]
    assert second.tickers == ["BETA"]
    with pytest.raises(AnnotationError, match="already exists as q_0002"):
        add_question(store, "  what were BETA's credit losses ")
    with pytest.raises(AnnotationError, match="tickers not in the corpus"):
        add_question(store, "Is Gamma profitable?", tickers=["GAMMA"])
    with pytest.raises(AnnotationError, match="topic"):
        add_question(store, "Something else?", topic="gossip")
    assert [q.query_id for q in reopen(store).questions()] == ["q_0001", "q_0002"]


def test_ai_assisted_questions_are_never_independent(store: AnnotationStore) -> None:
    with pytest.raises(AnnotationError, match="cannot be recorded as independently authored"):
        add_question(store, "Is Acme exposed to tariffs?", ai_assisted=True, reviewed_by="RV")
    with pytest.raises(AnnotationError, match="need reviewed_by"):
        add_question(store, "Is Acme exposed to tariffs?", ai_assisted=True, authoring="passage_inspired")
    question = add_question(
        store, "Is Acme exposed to tariffs?", ai_assisted=True, authoring="passage_inspired", reviewed_by="RV",
        author="claude",
    )
    assert reopen(store).question(question.query_id).reviewed_by == "RV"


def test_duplicate_query_ids_on_disk_are_rejected(store: AnnotationStore) -> None:
    path = store.directory / QUESTIONS_FILE
    path.write_text(path.read_text() * 2, encoding="utf-8")
    with pytest.raises(AnnotationError, match="duplicate query_id q_0001"):
        reopen(store)


# --- judgments --------------------------------------------------------------------------


@pytest.mark.parametrize("grade", [3, -1, True, "2", 1.0, None])
def test_only_grades_0_1_2_are_accepted(store: AnnotationStore, grade: object) -> None:
    with pytest.raises(AnnotationError, match="relevance grade must be one of"):
        store.judge("q_0001", IDS["acme24_supplier"], grade, annotator="T1")  # type: ignore[arg-type]
    assert [e.kind for e in reopen(store).history()] == ["question_added", "candidates_added"]


def test_judgments_need_a_pooled_corpus_passage(store: AnnotationStore) -> None:
    with pytest.raises(AnnotationError, match="not in the corpus"):
        store.judge("q_0001", "passage_missing", 2, annotator="T1")
    with pytest.raises(AnnotationError, match="not in the candidate pool of q_0001"):
        store.judge("q_0001", IDS["beta24_credit"], 2, annotator="T1")
    with pytest.raises(AnnotationError, match="unknown query_id q_0009"):
        store.judge("q_0009", IDS["acme24_supplier"], 2, annotator="T1")
    with pytest.raises(AnnotationError, match="not in the corpus"):
        store.add_candidates("q_0001", [Candidate(passage_id="passage_missing", source=BM25)], added_by="T1")


def test_skipped_and_pending_passages_are_unjudged_not_irrelevant(store: AnnotationStore) -> None:
    supplier, old_supplier, competition, sales = (e.passage_id for e in store.pool("q_0001"))
    store.judge("q_0001", supplier, 2, annotator="T1")
    store.judge("q_0001", competition, 0, annotator="T1")
    skipped = store.skip("q_0001", old_supplier, annotator="T1", note="unsure if 2023 counts")
    assert skipped.grade is None and skipped.state == "skipped"
    assert store.judgment("q_0001", competition).grade == 0
    assert store.judgment("q_0001", sales) is None
    assert [e.passage_id for e in store.pending("q_0001")] == [sales]
    store.clear("q_0001", old_supplier, annotator="T1")
    assert store.judgment("q_0001", old_supplier) is None


def test_progress_resumes_and_edits_keep_their_history(store: AnnotationStore) -> None:
    supplier = IDS["acme24_supplier"]
    store.judge("q_0001", supplier, 2, annotator="T1")
    store.skip("q_0001", IDS["acme23_supplier"], annotator="T1")

    resumed = reopen(store)
    assert resumed.judgment("q_0001", supplier).grade == 2
    assert [e.passage_id for e in resumed.pending("q_0001")] == [IDS["acme24_competition"], IDS["acme24_sales"]]
    resumed.judge("q_0001", supplier, 1, annotator="T2", note="only partly answers")
    resumed.judge("q_0001", supplier, 1, annotator="T2")  # unchanged: no new revision

    again = reopen(resumed)
    edited = again.judgment("q_0001", supplier)
    assert (edited.grade, edited.annotator, edited.note) == (1, "T2", "only partly answers")
    revisions = [(e.kind, e.previous, e.grade, e.annotator) for e in again.history("q_0001", supplier)]
    assert revisions == [("grade", None, 2, "T1"), ("grade", "2", 1, "T2")]


def test_complete_questions_are_frozen_until_reopened(store: AnnotationStore) -> None:
    with pytest.raises(AnnotationError, match="4 candidates without a grade or skip"):
        store.set_status("q_0001", "complete", annotator="T1")
    for entry in store.pool("q_0001")[:3]:
        store.judge("q_0001", entry.passage_id, 0, annotator="T1")
    store.skip("q_0001", IDS["acme24_sales"], annotator="T1")
    store.set_status("q_0001", "complete", annotator="T1")
    with pytest.raises(AnnotationError, match="q_0001 is complete; reopen it"):
        store.judge("q_0001", IDS["acme24_supplier"], 2, annotator="T1")
    with pytest.raises(AnnotationError, match="reopen it"):
        pooled(store, "acme24_table")
    store.set_status("q_0001", "open", annotator="T1")
    store.judge("q_0001", IDS["acme24_supplier"], 2, annotator="T1")
    store.set_status("q_0001", "withdrawn", annotator="T1")
    assert reopen(store).question("q_0001").status == "withdrawn"


# --- pool -------------------------------------------------------------------------------


def test_pool_merges_sources_and_keeps_positions(store: AnnotationStore) -> None:
    manual = CandidateSource(method="manual", detail="manual lookup")
    new = store.add_candidates(
        "q_0001",
        [Candidate(passage_id=IDS["acme24_supplier"], source=manual), Candidate(passage_id=IDS["acme24_table"], source=manual)],
        added_by="T1",
    )
    assert [e.passage_id for e in new] == [IDS["acme24_table"]]
    pool = reopen(store).pool("q_0001")
    assert [e.position for e in pool] == [1, 2, 3, 4, 5]
    assert pool[0].sources == [BM25, manual]
    events = len(store.history())
    assert pooled(store, "acme24_supplier") == [IDS["acme24_supplier"]]  # same source again: no change
    assert len(store.history()) == events


# --- history and integrity ---------------------------------------------------------------


def test_history_records_every_change_in_order(store: AnnotationStore) -> None:
    store.judge("q_0001", IDS["acme24_supplier"], 2, annotator="T1")
    store.skip("q_0001", IDS["acme24_supplier"], annotator="T1")
    store.clear("q_0001", IDS["acme24_supplier"], annotator="T1")
    events = lines(store.directory / HISTORY_FILE)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert [e["kind"] for e in events] == ["question_added", "candidates_added", "grade", "skip", "clear"]
    assert [e["previous"] for e in events[2:]] == [None, "2", "skipped"]
    assert events[0]["note"] == "How dependent is Acme on one supplier?"


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda events: events + [events[-1]], "expected event 4, got 3"),
        (lambda events: [*events[:-1], {**events[-1], "previous": "0"}], "conflicting or out-of-order"),
        (lambda events: [*events[:-1], {**events[-1], "passage_id": IDS["beta24_credit"]}], "never pooled"),
        (lambda events: [*events[:-1], {**events[-1], "grade": 5}], "relevance grade must be one of"),
    ],
)
def test_tampered_history_is_rejected(store: AnnotationStore, tamper, message: str) -> None:
    store.judge("q_0001", IDS["acme24_supplier"], 2, annotator="T1")
    path = store.directory / HISTORY_FILE
    path.write_text("".join(json.dumps(e) + "\n" for e in tamper(lines(path))), encoding="utf-8")
    with pytest.raises(AnnotationError, match=message):
        reopen(store)


def test_status_must_agree_with_history(store: AnnotationStore) -> None:
    path = store.directory / QUESTIONS_FILE
    path.write_text(path.read_text().replace('"status":"open"', '"status":"complete"'), encoding="utf-8")
    with pytest.raises(AnnotationError, match="history.jsonl leaves it 'open'"):
        reopen(store)


def test_pool_entries_must_reference_the_corpus(store: AnnotationStore) -> None:
    path = store.directory / POOL_FILE
    path.write_text(path.read_text().replace(IDS["acme24_sales"], "passage_gone"), encoding="utf-8")
    with pytest.raises(AnnotationError, match="passage_gone is not in the corpus"):
        reopen(store)


def test_two_sessions_cannot_interleave_judgments(store: AnnotationStore) -> None:
    other = AnnotationStore.open(store.directory, store.corpus, clock=fixed_clock())
    store.judge("q_0001", IDS["acme24_supplier"], 2, annotator="T1")
    with pytest.raises(AnnotationError, match="changed since this workspace was opened"):
        other.judge("q_0001", IDS["acme24_supplier"], 0, annotator="T2")
    assert reopen(store).judgment("q_0001", IDS["acme24_supplier"]).grade == 2
