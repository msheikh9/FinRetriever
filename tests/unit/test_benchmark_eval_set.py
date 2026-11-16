"""Tests for named evaluation sets in the benchmark harness (synthetic corpus, offline)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from finretriever.data.schemas import Qrel, Query, SearchHit
from finretriever.eval_set.qrels_builder import judged_at_k
from finretriever.evaluation import benchmark
from finretriever.evaluation.benchmark import (
    COMPARISON_CAUTION,
    DEV_BANNER,
    METRIC_DEFINITIONS,
    POOL_DISCLOSURE,
    RETRIEVAL_MODES,
    UNJUDGED_ASSUMPTION,
    build_retriever,
    company_restrictions,
    evaluate_eval_set_run,
    load_eval_set,
    load_retrieval_modes,
    pool_composition,
    pool_disclosure,
    run_eval_set,
    validate_eval_inputs,
    write_eval_set_outputs,
)
from finretriever.evaluation.provenance import CorpusMismatchError
from finretriever.evaluation.stored_runs import StoredRunError
from finretriever.retrieval import biencoder
from finretriever.utils.io import read_jsonl
from tests.unit.dense_fixtures import REVISION, FakeEncoder
from tests.unit.eval_set_fixtures import fixture_passages, passage_ids, write_dev_eval_set

IDS = passage_ids()
PASSAGES = {p.passage_id: p for p in fixture_passages()}


def hit(query_id: str, name: str, rank: int) -> SearchHit:
    return SearchHit(query_id=query_id, passage_id=IDS[name], rank=rank, score=10.0 - rank, retriever="t")


def dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def test_per_query_metrics_keep_unjudged_apart_from_confirmed_negatives() -> None:
    queries = [
        Query(query_id="q1", text="Acme supplier risk?", source="dev_ai", metadata={"tickers": "ACME"}),
        Query(query_id="q2", text="Beta rates?", source="dev_ai", metadata={"tickers": "BETA"}),
    ]
    qrels = [
        Qrel(query_id="q1", passage_id=IDS["acme24_supplier"], relevance=2),
        Qrel(query_id="q1", passage_id=IDS["acme24_competition"], relevance=0),
        Qrel(query_id="q1", passage_id=IDS["acme24_sales"], relevance=1),
        Qrel(query_id="q2", passage_id=IDS["beta24_rates"], relevance=0),
    ]
    hits = [
        hit("q1", "acme24_competition", 1),  # confirmed negative
        hit("q1", "acme24_liquidity", 2),  # unjudged, same company: unknown relevance
        hit("q1", "acme24_supplier", 3),  # grade 2
        hit("q1", "beta24_credit", 4),  # unjudged, other company
        hit("q1", "acme24_sales", 5),  # grade 1
        hit("q2", "beta24_rates", 1),
    ]
    aggregate, rows = evaluate_eval_set_run(
        queries, qrels, hits, PASSAGES, metrics=["ndcg@10", "mrr@10", "recall@100"], judged_k=10,
        thresholds={"min_positives": 1, "min_judged_at_10": 0.7},
    )
    q1, q2 = rows
    ideal = dcg([2, 1])
    assert q1["metrics"]["ndcg@10"] == pytest.approx(dcg([0, 0, 2, 0, 1]) / ideal)
    assert q1["metrics"]["mrr@10"] == pytest.approx(1 / 3)
    assert q1["metrics"]["recall@100"] == pytest.approx(1.0)
    assert q1["metrics"]["ndcg@10_judged_only"] == pytest.approx(dcg([0, 2, 1]) / ideal)
    assert q1["metrics"]["mrr@10_judged_only"] == pytest.approx(1 / 2)
    assert q1["judged@10"] == pytest.approx(3 / 5)
    assert q1["judged@10"] == judged_at_k(qrels, hits, k=10)["q1"]  # one definition of judged@k
    assert q1["top10"] == {
        "returned": 5, "relevant": 2, "grade_2": 1, "grade_1": 1, "confirmed_negatives": 1,
        "unjudged": 2, "unjudged_in_scope": 1, "unjudged_out_of_scope": 1,
    }
    assert q1["judgments"] == {"judged": 3, "positives": 2, "grade_2": 1, "grade_1": 1, "confirmed_negatives": 1}
    assert (q1["first_relevant_rank"], q1["coverage_flags"]) == (3, ["low_judged_at_10"])
    assert q2["metrics"]["ndcg@10"] == 0.0 and q2["coverage_flags"] == ["no_positive_judgments"]
    assert q2["judged@10"] == 1.0 and q2["first_relevant_rank"] is None
    assert aggregate["ndcg@10"] == pytest.approx((dcg([0, 0, 2, 0, 1]) / ideal) / 2)
    assert aggregate["top10_unjudged_total"] == 2 and aggregate["top10_confirmed_negatives_total"] == 2
    assert aggregate["queries_with_insufficient_coverage"] == ["q1", "q2"]


def test_validate_eval_inputs_rejects_conflicts_unknown_passages_and_unjudged_queries() -> None:
    passages = list(PASSAGES.values())
    queries = [Query(query_id="q1", text="x", source="dev_ai"), Query(query_id="q2", text="y", source="dev_ai")]
    ok = [Qrel(query_id="q1", passage_id=IDS["acme24_sales"], relevance=1), Qrel(query_id="q2", passage_id=IDS["beta24_rates"], relevance=0)]
    validate_eval_inputs(passages, queries, ok)
    with pytest.raises(ValueError, match="conflicting labels"):
        validate_eval_inputs(passages, queries, [*ok, Qrel(query_id="q1", passage_id=IDS["acme24_sales"], relevance=2)])
    with pytest.raises(ValueError, match="not in the corpus"):
        validate_eval_inputs(passages, queries, [*ok, Qrel(query_id="q1", passage_id="passage_x", relevance=1)])
    with pytest.raises(ValueError, match="without any judgment"):
        validate_eval_inputs(passages, queries, ok[:1])
    with pytest.raises(ValueError, match="not in the query set"):
        validate_eval_inputs(passages, queries[:1], ok)


# --- end to end ---------------------------------------------------------------------------


@pytest.fixture
def eval_set(tmp_path: Path) -> dict[str, Any]:
    return write_dev_eval_set(tmp_path)


def tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_run_eval_set_verifies_provenance_and_writes_only_dev_results(eval_set: dict[str, Any]) -> None:
    tmp = eval_set["tmp"]
    before = tree(tmp)
    result = run_eval_set(eval_set["config"], "dev", config_path=eval_set["config_path"])
    assert [r.run_id for r in result.runs] == ["global/bm25", "global/bm25_whitespace"]
    # The dense model cannot load here (tests never load real models): skipped, with the reason.
    assert [s["name"] for s in result.skipped] == ["bge_zero_shot"]
    assert "importing torch / sentence-transformers failed" in result.skipped[0]["reason"]
    paths = write_eval_set_outputs(result)
    out = tmp / "results" / "dev" / "test_v2"
    after = tree(tmp)
    written = {name for name in after if name not in before or after[name] != before[name]}
    expected = {
        "global/bm25/metrics.json", "global/bm25/run.jsonl", "global/bm25_whitespace/metrics.json",
        "global/bm25_whitespace/run.jsonl", "report.md",
    }
    assert written == {str((out / rel).relative_to(tmp)) for rel in expected}
    assert set(paths) == expected

    payload = json.loads((out / "global" / "bm25" / "metrics.json").read_text())
    assert payload["status"] == "provisional_development" and payload["notice"] == DEV_BANNER
    assert (payload["run"], payload["mode"], payload["mode_definition"]) == ("bm25", "global", RETRIEVAL_MODES["global"])
    assert payload["comparison_caution"] == COMPARISON_CAUTION
    assert payload["candidate_pools"] == {"disclosure": POOL_DISCLOSURE, "composition": None}
    assert payload["skipped_retrievers"] == result.skipped
    assert payload["corpus"]["sha256"] == result.corpus.corpus_sha256 and payload["corpus"]["passages"] == 12
    assert (payload["label_version"], payload["annotation_version"]) == ("test_v2", "dev_test_v2")
    assert payload["retriever"]["tokenizer"] == "word" and payload["retriever"]["backend"] == "python"
    assert payload["settings"]["benchmark_config_sha256"] == hashlib.sha256(eval_set["config_path"].read_bytes()).hexdigest()
    assert set(METRIC_DEFINITIONS) <= set(payload["metric_definitions"])
    assert [row["query_id"] for row in payload["per_query"]] == ["q_0001", "q_0002"]
    assert {"judged@10", "top10", "judgments", "coverage_flags"} <= set(payload["per_query"][0])
    run_rows = [json.loads(line) for line in (out / "global" / "bm25" / "run.jsonl").read_text().splitlines()]
    assert len(run_rows) == 10 and {r["judged"] for r in run_rows} == {True, False}
    assert all((r["grade"] is None) != r["judged"] for r in run_rows)  # unjudged rows carry no grade
    run_sha = hashlib.sha256((out / "global" / "bm25" / "run.jsonl").read_bytes()).hexdigest()
    assert payload["run_file"]["sha256"] == run_sha
    report = (out / "report.md").read_text()
    assert DEV_BANNER in report and "insufficient relevance coverage" in report and "(diagnostic)" in report
    assert COMPARISON_CAUTION in report and POOL_DISCLOSURE in report and "Not run" in report

    write_eval_set_outputs(run_eval_set(eval_set["config"], "dev", config_path=eval_set["config_path"]))
    assert tree(tmp) == after  # rerunning gives identical outputs


def test_corpus_mismatch_is_rejected_before_scoring(eval_set: dict[str, Any]) -> None:
    metadata_path = eval_set["labels"] / "dev_eval.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_path.write_text(json.dumps({**metadata, "corpus_sha256": "f" * 64}))
    with pytest.raises(CorpusMismatchError):
        run_eval_set(eval_set["config"], "dev")
    assert not (eval_set["tmp"] / "results").exists()


def test_qrels_must_match_the_export_report(eval_set: dict[str, Any]) -> None:
    qrels = eval_set["labels"] / "dev_qrels.jsonl"
    qrels.write_text(qrels.read_text().replace('"relevance":0', '"relevance":1', 1))
    with pytest.raises(ValueError, match="does not match the SHA-256"):
        run_eval_set(eval_set["config"], "dev")


def test_load_eval_set_guards_status_and_output_location(eval_set: dict[str, Any]) -> None:
    config = eval_set["config"]
    entry = config["evaluation_sets"]["dev"]
    with pytest.raises(ValueError, match="unknown evaluation set"):
        load_eval_set(config, "missing")
    outside = {**config, "evaluation_sets": {"dev": {**entry, "output_dir": str(eval_set["tmp"] / "results" / "elsewhere")}}}
    with pytest.raises(ValueError, match="under results/dev/"):
        load_eval_set(outside, "dev")
    bad_status = {**config, "evaluation_sets": {"dev": {**entry, "status": "gold"}}}
    with pytest.raises(ValueError, match="status"):
        load_eval_set(bad_status, "dev")
    incomplete = {**config, "evaluation_sets": {"dev": {k: v for k, v in entry.items() if k != "qrels_path"}}}
    with pytest.raises(ValueError, match="missing"):
        load_eval_set(incomplete, "dev")


# --- retrieval modes and dense retrievers ---------------------------------------------------


def dense_config(eval_set: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """The fixture config with BM25 and a (fake) dense retriever in both retrieval modes."""
    return {
        **eval_set["config"],
        "retrieval_modes": ["global", "company"],
        "embedding_cache_dir": str(eval_set["tmp"] / "cache"),
        "baselines": [
            {"name": "bm25", "type": "bm25", "tokenizer": "word", "backend": "python"},
            {"name": "bge_zero_shot", "type": "dense", "model_name": "BAAI/bge-base-en-v1.5",
             "model_revision": REVISION, "embedding_dim": 16, "batch_size": 4},
        ],
        **changes,
    }


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> list[FakeEncoder]:
    loaded: list[FakeEncoder] = []

    def load(*args: Any, **kwargs: Any) -> FakeEncoder:
        loaded.append(FakeEncoder())
        return loaded[-1]

    monkeypatch.setattr(biencoder, "load_sentence_transformer", load)
    return loaded


def test_every_retriever_runs_in_every_mode_with_identical_metric_definitions(
    eval_set: dict[str, Any], fake_models: list[FakeEncoder]
) -> None:
    config = dense_config(eval_set)
    result = run_eval_set(config, "dev")
    assert [r.run_id for r in result.runs] == [
        "global/bm25", "company/bm25", "global/bge_zero_shot", "company/bge_zero_shot",
    ]
    assert not result.skipped
    assert len(fake_models) == 1 and len(fake_models[0].calls[0]) == 4  # loaded once, batches of 4
    passages = fixture_passages()
    assert fake_models[0].encoded[: len(passages)] == [p.text for p in passages]  # encoded once for both modes

    entry = config["evaluation_sets"]["dev"]
    queries = read_jsonl(entry["queries_path"], model=Query)
    qrels = read_jsonl(entry["qrels_path"], model=Qrel)
    tickers = {q.query_id: str(q.metadata["tickers"]) for q in queries}
    for run in result.runs:
        # Every run is scored by the same function with the same settings.
        assert (run.aggregate, run.per_query) == evaluate_eval_set_run(
            queries, qrels, run.hits, PASSAGES, metrics=config["metrics"], judged_k=10,
            thresholds=config["coverage_thresholds"],
        )
        if run.mode == "company":
            assert all(h.metadata["ticker"] == tickers[h.query_id] for h in run.hits)
            assert all(r["top10"]["unjudged_out_of_scope"] == 0 for r in run.per_query)
    global_tickers = {h.metadata["ticker"] for r in result.runs if r.mode == "global" for h in r.hits}
    assert global_tickers == {"ACME", "BETA"}

    paths = write_eval_set_outputs(result)
    payloads = {name[: -len("/metrics.json")]: json.loads(path.read_text()) for name, path in paths.items() if name.endswith("metrics.json")}
    assert sorted(payloads) == sorted(r.run_id for r in result.runs)
    first = next(iter(payloads.values()))
    for run_id, payload in payloads.items():
        assert payload["mode"] == run_id.split("/")[0] and payload["mode_definition"] == RETRIEVAL_MODES[payload["mode"]]
        assert payload["metric_definitions"] == METRIC_DEFINITIONS and payload["unjudged_assumption"] == UNJUDGED_ASSUMPTION
        assert payload["settings"] == first["settings"] and payload["corpus"] == first["corpus"]
    assert first["settings"]["retrieval_modes"] == ["global", "company"]
    dense_run = payloads["global/bge_zero_shot"]["retriever"]
    assert (dense_run["model_name"], dense_run["model_revision"], dense_run["embedding_dim"]) == (
        "BAAI/bge-base-en-v1.5", REVISION, 16,
    )
    assert dense_run["query_prefix"].startswith("Represent this sentence") and dense_run["passage_prefix"] == ""
    assert dense_run["cache"]["key"]["corpus_sha256"] == result.corpus.corpus_sha256
    assert payloads["company/bge_zero_shot"]["retriever"] == dense_run

    report = (eval_set["tmp"] / "results" / "dev" / "test_v2" / "report.md").read_text()
    assert "## Comparison" in report and "judged@10 per question" in report
    assert "| bge_zero_shot | company |" in report and "| q_0002 |" in report
    assert RETRIEVAL_MODES["company"] in report and "Not run" not in report


def test_rerunning_reuses_the_embedding_cache_and_reproduces_the_outputs(
    eval_set: dict[str, Any], fake_models: list[FakeEncoder]
) -> None:
    config = dense_config(eval_set)
    write_eval_set_outputs(run_eval_set(config, "dev"))
    out = eval_set["tmp"] / "results" / "dev" / "test_v2"
    first = tree(out)
    write_eval_set_outputs(run_eval_set(config, "dev"))
    assert len(fake_models) == 2
    assert all(len(call) == 1 for call in fake_models[1].calls)  # only queries were encoded
    assert tree(out) == first


def test_corpus_provenance_is_checked_before_any_model_is_loaded(
    eval_set: dict[str, Any], fake_models: list[FakeEncoder]
) -> None:
    metadata_path = eval_set["labels"] / "dev_eval.metadata.json"
    metadata_path.write_text(json.dumps({**json.loads(metadata_path.read_text()), "corpus_sha256": "f" * 64}))
    with pytest.raises(CorpusMismatchError):
        run_eval_set(dense_config(eval_set), "dev")
    assert fake_models == [] and not (eval_set["tmp"] / "cache").exists()


def test_dense_runs_write_only_dev_results_and_the_cache(
    eval_set: dict[str, Any], fake_models: list[FakeEncoder]
) -> None:
    tmp = eval_set["tmp"]
    before = tree(tmp)
    write_eval_set_outputs(run_eval_set(dense_config(eval_set), "dev"))
    after = tree(tmp)
    written = {name for name in after if before.get(name) != after[name]}
    assert written and all(name.startswith(("results/dev/test_v2/", "cache/")) for name in written)
    assert all(after[name] == digest for name, digest in before.items())  # inputs untouched


def test_pool_composition_shows_which_source_proposed_each_judged_passage(
    eval_set: dict[str, Any], tmp_path: Path
) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [
        ("q_0001", "acme24_supplier", ["bm25"]), ("q_0001", "acme23_supplier", ["random"]),
        ("q_0002", "beta24_liquidity", ["bm25", "stratified"]), ("q_0002", "beta24_rates", ["manual"]),
    ]
    pool.write_text("".join(
        json.dumps({"query_id": q, "passage_id": IDS[name], "sources": [{"method": m} for m in methods]}) + "\n"
        for q, name, methods in rows
    ))
    entry = {**eval_set["config"]["evaluation_sets"]["dev"], "candidate_pool_path": str(pool)}
    config = {**eval_set["config"], "evaluation_sets": {"dev": entry}}
    result = run_eval_set(config, "dev")
    assert result.pool == {
        "judged": 4, "judged_proposed_by_bm25": 2, "positives": 2, "positives_proposed_by_bm25": 2,
        "judged_by_source": {"bm25": 1, "bm25+stratified": 1, "manual": 1, "random": 1},
        "positives_by_source": {"bm25": 1, "bm25+stratified": 1},
        "sources": ["bm25", "manual", "random", "stratified"],
    }
    assert result.inputs["candidate_pool"]["sha256"] == hashlib.sha256(pool.read_bytes()).hexdigest()
    qrels = read_jsonl(entry["qrels_path"], model=Qrel)
    assert pool_composition(pool, qrels[:1])["judged_by_source"] == {"bm25": 1}
    write_eval_set_outputs(result)
    report = (eval_set["tmp"] / "results" / "dev" / "test_v2" / "report.md").read_text()
    assert "2 of 4 (50.0%) judged passages" in report


def test_config_errors_are_rejected_before_anything_runs() -> None:
    with pytest.raises(ValueError, match="unknown type"):
        build_retriever({"name": "x", "type": "colbert"})
    with pytest.raises(ValueError, match="unknown settings"):
        build_retriever({"name": "b", "type": "bm25", "tokenizr": "word"})
    with pytest.raises(ValueError, match="needs model_name"):
        build_retriever({"name": "d", "type": "dense", "model_revision": REVISION})
    with pytest.raises(ValueError, match="commit SHA"):
        build_retriever({"name": "d", "type": "dense", "model_name": "intfloat/e5-base-v2"})
    with pytest.raises(ValueError, match="must not be used for another"):
        build_retriever({"name": "d", "type": "dense", "model_name": "intfloat/e5-base-v2",
                         "model_revision": REVISION, "query_prefix": "Represent this sentence: "})
    assert load_retrieval_modes({}) == ["global"]
    assert load_retrieval_modes({"retrieval_modes": ["company", "global"]}) == ["company", "global"]
    with pytest.raises(ValueError, match="unknown retrieval modes"):
        load_retrieval_modes({"retrieval_modes": ["global", "sector"]})
    with pytest.raises(ValueError, match="twice"):
        load_retrieval_modes({"retrieval_modes": ["global", "global"]})
    with pytest.raises(ValueError, match="needs tickers for every query"):
        company_restrictions([Query(query_id="q1", text="x", source="dev_ai", metadata={"tickers": "ACME"}),
                              Query(query_id="q2", text="y", source="dev_ai")])


# --- re-scoring stored runs with newer labels --------------------------------------------------


def newer_labels(eval_set: dict[str, Any]) -> dict[str, Any]:
    """The fixture config plus ``dev_v3``: the dev labels and one more positive, scoring the stored dev runs."""
    tmp, labels = eval_set["tmp"], eval_set["labels"]
    v3 = tmp / "labels_v3"
    v3.mkdir()
    (v3 / "dev_queries.jsonl").write_bytes((labels / "dev_queries.jsonl").read_bytes())
    extra = Qrel(query_id="q_0001", passage_id=IDS["acme24_sales"], relevance=1)  # a newly judged positive
    (v3 / "dev_qrels.jsonl").write_text((labels / "dev_qrels.jsonl").read_text() + extra.model_dump_json() + "\n")
    metadata = json.loads((labels / "dev_eval.metadata.json").read_text())
    (v3 / "dev_eval.metadata.json").write_text(json.dumps({**metadata, "annotation_version": "dev_test_v3"}))
    report = json.loads((labels / "dev_eval.report.json").read_text())
    report.update(label_version="test_v3", files={
        name: hashlib.sha256((v3 / name).read_bytes()).hexdigest() for name in ("dev_queries.jsonl", "dev_qrels.jsonl")
    })
    (v3 / "dev_eval.report.json").write_text(json.dumps(report))
    config = {**eval_set["config"], "retrieval_modes": ["global", "company"], "baselines": eval_set["config"]["baselines"][:1]}
    config["evaluation_sets"] = {
        **config["evaluation_sets"],
        "dev_v3": {
            "status": "provisional_development",
            "queries_path": str(v3 / "dev_queries.jsonl"),
            "qrels_path": str(v3 / "dev_qrels.jsonl"),
            "metadata_path": str(v3 / "dev_eval.metadata.json"),
            "report_path": str(v3 / "dev_eval.report.json"),
            "runs_from": str(tmp / "results" / "dev" / "test_v2"),
            "compare_with": "dev",
            "output_dir": str(tmp / "results" / "dev" / "test_v3"),
        },
    }
    write_eval_set_outputs(run_eval_set(config, "dev"))  # the stored runs and their dev scores
    return config


def test_stored_runs_are_rescored_with_newer_labels_without_running_any_retriever(
    eval_set: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = newer_labels(eval_set)
    stored_dir = eval_set["tmp"] / "results" / "dev" / "test_v2"
    stored_before = tree(stored_dir)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("re-scoring must not build a retriever")

    monkeypatch.setattr(benchmark, "build_retriever", refuse)
    result = run_eval_set(config, "dev_v3")
    assert [r.run_id for r in result.runs] == ["global/bm25", "company/bm25"]
    assert result.settings["runs_from"] == str(stored_dir)
    entry = config["evaluation_sets"]["dev_v3"]
    queries, qrels = read_jsonl(entry["queries_path"], model=Query), read_jsonl(entry["qrels_path"], model=Qrel)
    for run in result.runs:
        stored = json.loads((stored_dir / run.mode / run.run_name / "metrics.json").read_text())
        assert run.source["run_sha256"] == stored["run_file"]["sha256"] and run.retriever == stored["retriever"]
        assert (run.aggregate, run.per_query) == evaluate_eval_set_run(  # the same metric definitions
            queries, qrels, run.hits, PASSAGES, metrics=config["metrics"], judged_k=10, thresholds=config["coverage_thresholds"],
        )
        assert run.aggregate["positives_total"] == stored["aggregate"]["positives_total"] + 1
    assert result.comparison["label_version"] == "test_v2"
    assert all(r["same_ranking"] for r in result.comparison["runs"].values())

    write_eval_set_outputs(result)
    out = eval_set["tmp"] / "results" / "dev" / "test_v3"
    for run in result.runs:
        keys = ("query_id", "rank", "passage_id", "score", "ticker")
        rows = [json.loads(line) for line in (out / run.mode / run.run_name / "run.jsonl").read_text().splitlines()]
        before = [json.loads(line) for line in (stored_dir / run.mode / run.run_name / "run.jsonl").read_text().splitlines()]
        assert [tuple(r[k] for k in keys) for r in rows] == [tuple(r[k] for k in keys) for r in before]  # not re-ranked
        payload = json.loads((out / run.mode / run.run_name / "metrics.json").read_text())
        assert payload["source_run"] == run.source and payload["label_version"] == "test_v3"
        assert payload["compared_with"]["label_version"] == "test_v2" and payload["compared_with"]["same_ranking"]
    report = (out / "report.md").read_text()
    assert "## Change from dev (labels test_v2 → test_v3)" in report and "No retriever was run for this set" in report
    assert "nDCG@10 per question, earlier → this set" in report and DEV_BANNER in report
    assert tree(stored_dir) == stored_before  # the stored runs are only read
    first = tree(out)
    write_eval_set_outputs(run_eval_set(config, "dev_v3"))
    assert tree(out) == first


def test_rescoring_refuses_modified_or_mismatched_stored_runs(eval_set: dict[str, Any]) -> None:
    config = newer_labels(eval_set)
    stored = eval_set["tmp"] / "results" / "dev" / "test_v2" / "company" / "bm25" / "run.jsonl"
    original = stored.read_bytes()
    stored.write_bytes(original.replace(b'"rank":1', b'"rank":2', 1))
    with pytest.raises(StoredRunError, match="was modified"):
        run_eval_set(config, "dev_v3")
    stored.write_bytes(original)
    with pytest.raises(ValueError, match="ranked to depth 5"):
        run_eval_set({**config, "top_k": 100}, "dev_v3")
    with pytest.raises(StoredRunError, match="missing"):
        run_eval_set({**config, "retrieval_modes": ["global", "company"], "baselines": [{"name": "e5", "type": "bm25"}]}, "dev_v3")
    sets = config["evaluation_sets"]
    with pytest.raises(ValueError, match="compares with unknown set"):
        load_eval_set({**config, "evaluation_sets": {**sets, "dev_v3": {**sets["dev_v3"], "compare_with": "nope"}}}, "dev_v3")
    same_dir = {**sets["dev_v3"], "output_dir": sets["dev_v3"]["runs_from"]}
    with pytest.raises(ValueError, match="overwrite the stored runs"):
        load_eval_set({**config, "evaluation_sets": {**sets, "dev_v3": same_dir}}, "dev_v3")
    assert not (eval_set["tmp"] / "results" / "dev" / "test_v3").exists()


def test_pool_composition_reads_expanded_pools_and_discloses_them(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"query_id": "q_0001", "passage_id": IDS["acme24_supplier"], "base_sources": [{"method": "bm25"}],
         "run_sources": [{"run": "bge", "mode": "global", "rank": 1, "score": 0.9}]},
        {"query_id": "q_0001", "passage_id": IDS["acme24_sales"], "base_sources": [],
         "run_sources": [{"run": "bge", "mode": "company", "rank": 4, "score": 0.8}]},
        {"query_id": "q_0002", "passage_id": IDS["beta24_rates"], "base_sources": [],
         "run_sources": [{"run": "bm25", "mode": "global", "rank": 7, "score": 3.0}]},
    ]
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    qrels = [
        Qrel(query_id="q_0001", passage_id=IDS["acme24_supplier"], relevance=2),
        Qrel(query_id="q_0001", passage_id=IDS["acme24_sales"], relevance=1),
        Qrel(query_id="q_0002", passage_id=IDS["beta24_rates"], relevance=0),
    ]
    composition = pool_composition(pool, qrels)
    assert composition["judged_by_source"] == {"bge": 1, "bge+bm25": 1, "bm25": 1}
    assert (composition["judged_proposed_by_bm25"], composition["positives_proposed_by_bm25"]) == (2, 1)
    assert composition["run_sources"] == ["company/bge", "global/bge", "global/bm25"] and composition["run_depth"] == 7
    disclosure = pool_disclosure(composition)
    assert "company/bge, global/bge, global/bm25 ranked in their top 7" in disclosure and "Recall@100" in disclosure
    assert pool_disclosure(None) == POOL_DISCLOSURE
