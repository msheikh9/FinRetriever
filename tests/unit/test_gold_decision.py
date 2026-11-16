"""The frozen gold-set decision (``finretriever.evaluation.gold_decision``), on synthetic runs and judgments only.

No gold label exists, and none is read. Every question, judgment and ranking here is made up in the test. The
benchmark's two evaluation sets are written as the benchmark would write them:
- the primary set ranks only the eligible passages;
- every ``metrics.json`` holds per-question metrics computed the benchmark's way.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from finretriever.data.schemas import Qrel, Query, SearchHit
from finretriever.eval_set.qrels_builder import judged_at_k
from finretriever.evaluation import gold_decision as gd
from finretriever.evaluation.exclusion import ids_sha256
from finretriever.evaluation.metrics import compute_per_query_metrics

ENCODING = {"query_prefix": "Represent this sentence for searching relevant passages: ", "passage_prefix": "",
            "max_seq_length": 512}
SOURCES = {"train_candidates_v3": "a" * 64, "validation_pairs_v4": "b" * 64}
CORPUS = 1000


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def checkpoint(run: str, step: int, weights: str, mode: str, **extra: Any) -> dict[str, Any]:
    contamination = {"judged_training_passages": 3, "judged_validation_passages": 0} if mode.endswith("corpus") else None
    return {**ENCODING,
            "checkpoint": {"run": run, "step": step, "digest": extra.pop("digest", "d0"),
                           "files": {"model.safetensors": weights},
                           "trained_on": {"candidates": {"manifest_sha256": SOURCES["train_candidates_v3"]}}},
            "evaluation_isolation": {"mode": mode, "isolated": True, "contamination": contamination,
                                     "validation": {"manifest": {"sha256": SOURCES["validation_pairs_v4"]}}},
            **extra}


def a0_retriever(mode: str) -> dict[str, Any]:
    return checkpoint("bge_base_ft_v3_a1-seed42", 0, gd.A0_WEIGHTS_SHA256, mode)


def a1_retriever(seed: int, mode: str) -> dict[str, Any]:
    pin = gd.OFFICIAL_A1[seed]
    return checkpoint(pin["run"], pin["step"], pin["weights_sha256"], mode, digest=pin["digest"], role="final",
                      validation_selection={"record": {"sha256": pin["selection_sha256"]}, "seed": seed,
                                            "seeds": list(gd.SEEDS)})


def write_run(directory: Path, ranks: dict[str, list[str]], retriever: dict[str, Any], qrels: list[Qrel], *,
              eval_set: str, collection: dict[str, Any], isolation: str, **overrides: Any) -> None:
    """``<set>/global/<run>/``: run.jsonl and the metrics.json the benchmark would write, per-question rows included."""
    directory.mkdir(parents=True)
    hits = [SearchHit(query_id=q, passage_id=p, rank=i, score=1.0 / i, retriever="r")
            for q, passages in sorted(ranks.items()) for i, p in enumerate(passages, start=1)]
    rows = [{"query_id": h.query_id, "rank": h.rank, "passage_id": h.passage_id, "score": h.score} for h in hits]
    run_bytes = _jsonl(rows)
    (directory / "run.jsonl").write_bytes(run_bytes)
    scores = compute_per_query_metrics(qrels, hits)
    judged = judged_at_k(qrels, hits, k=10)
    per_query = [{"query_id": q, "metrics": s, "judged@10": judged.get(q, 0.0)} for q, s in sorted(scores.items())]
    metrics = {"status": "final", "mode": "global", "eval_set": eval_set, "annotation_version": "gold_v1",
               "corpus": {"sha256": gd.CORPUS_SHA256, "version": gd.CORPUS_VERSION}, "retriever": retriever,
               "settings": {"isolation_mode": isolation, "collection": collection}, "per_query": per_query,
               "run_file": {"path": "run.jsonl", "sha256": hashlib.sha256(run_bytes).hexdigest()}, **overrides}
    (directory / "metrics.json").write_text(json.dumps(metrics))


def ranking(question: str, relevant_rank: int, *, length: int = 20, front: list[str] | None = None) -> list[str]:
    """Unjudged filler passages, with the question's grade-2 passage at ``relevant_rank`` (0: not retrieved)."""
    passages = list(front or []) + [f"{question}-u{i}" for i in range(length)]
    if relevant_rank:
        passages.insert(relevant_rank - 1, f"{question}-rel")
    return passages[:length]


class Setup:
    """Synthetic questions and judgments, and the benchmark's two sets for A0, the three A1 seeds and BM25."""

    def __init__(self, tmp_path: Path, a0_ranks: list[int], seed_ranks: dict[int, list[int]], *,
                 ai_assisted: int = 0, front: dict[str, list[str]] | None = None,
                 excluded: set[str] | None = None, extra_qrels: list[Qrel] | None = None) -> None:
        self.tmp = tmp_path
        n = len(a0_ranks)
        self.questions = [f"q_{i:04d}" for i in range(1, n + 1)]
        self.queries = [Query(query_id=q, text=f"question {q}", source="gold",
                              metadata={"ai_assisted": i < ai_assisted}) for i, q in enumerate(self.questions)]
        self.qrels = [Qrel(query_id=q, passage_id=f"{q}-rel", relevance=2) for q in self.questions]
        self.qrels += [Qrel(query_id=q, passage_id=f"{q}-u0", relevance=0) for q in self.questions]
        self.qrels += list(extra_qrels or [])
        self.metadata = {"corpus_sha256": gd.CORPUS_SHA256, "corpus_version": gd.CORPUS_VERSION,
                         "annotation_version": "gold_v1"}
        self.excluded = frozenset(excluded or {"x-never-ranked"})
        self.collection = {"exclusion": "test_exclusion", "exclusion_sha256": ids_sha256(self.excluded),
                           "corpus_passages": CORPUS, "excluded_passages": len(self.excluded),
                           "eligible_passages": CORPUS - len(self.excluded), "sources": SOURCES}
        front = front or {}
        self.full_ranks = {"A0": {q: ranking(q, r) for q, r in zip(self.questions, a0_ranks, strict=True)},
                           "bm25": {q: ranking(q, r) for q, r in zip(self.questions, a0_ranks, strict=True)}}
        for seed, ranks in seed_ranks.items():
            self.full_ranks[seed] = {q: ranking(q, r, front=front.get(q))
                                     for q, r in zip(self.questions, ranks, strict=True)}
        self.write()

    def write(self) -> None:
        for directory in (self.tmp / "primary", self.tmp / "secondary"):
            shutil.rmtree(directory, ignore_errors=True)
        eligible_qrels = [q for q in self.qrels if q.passage_id not in self.excluded]
        for label, set_dir, set_name, isolation, collection, qrels in (
                ("primary", self.tmp / "primary", gd.PRIMARY_SET, gd.ISOLATION_PRIMARY, self.collection, eligible_qrels),
                ("secondary", self.tmp / "secondary", gd.SECONDARY_SET, gd.ISOLATION_SECONDARY,
                 {"exclusion": None, "corpus_passages": CORPUS, "excluded_passages": 0}, self.qrels)):
            def ranks(name: Any, primary: bool = label == "primary") -> dict[str, list[str]]:
                full = self.full_ranks[name]
                # The benchmark excludes before ranking: the excluded passages are simply not in the eligible index.
                return {q: [p for p in ps if p not in self.excluded] for q, ps in full.items()} if primary else full
            base = set_dir / "global"
            common = {"eval_set": set_name, "collection": collection, "isolation": isolation}
            write_run(base / gd.A0_RUN, ranks("A0"), a0_retriever(isolation), qrels, **common)
            for seed, name in gd.A1_RUNS.items():
                write_run(base / name, ranks(seed), a1_retriever(seed, isolation), qrels, **common)
            write_run(base / "bm25", ranks("bm25"), {"name": "bm25"}, qrels, **common)

    def analyse(self) -> dict[str, Any]:
        return gd.analyse(queries=self.queries, qrels=self.qrels, export_metadata=self.metadata,
                          primary_dir=self.tmp / "primary", secondary_dir=self.tmp / "secondary",
                          excluded=self.excluded, expected_collection=self.collection)

    def edit(self, set_name: str, run: str, change: Any) -> None:
        """Edit one metrics.json in place (keeps its run.jsonl and the recorded SHA-256 of it)."""
        path = self.tmp / set_name / "global" / run / "metrics.json"
        payload = json.loads(path.read_text())
        change(payload)
        path.write_text(json.dumps(payload))


def random_ranks(n: int, low: int, high: int, seed: int) -> list[int]:
    return [int(r) for r in np.random.default_rng(seed).integers(low, high + 1, size=n)]


def improving(tmp_path: Path, n: int = 60, **kwargs: Any) -> Setup:
    return Setup(tmp_path, random_ranks(n, 4, 12, 1), {s: random_ranks(n, 1, 3, s) for s in gd.SEEDS}, **kwargs)


def test_the_frozen_constants_are_the_approved_ones() -> None:
    assert (gd.PRIMARY_METRIC, gd.MODE, gd.FLIPS, gd.ALPHA, gd.MIN_QUESTIONS) == ("ndcg@10", "global", 10_000, 0.05, 50)
    assert (gd.RNG_SEED, gd.BOOTSTRAP, gd.BOOTSTRAP_SEED, gd.JUDGED_GAP) == (20261112, 10_000, 20261113, 0.10)
    assert gd.SEEDS == (42, 1337, 2026) and gd.A1_RUNS == {42: "a1_seed42", 1337: "a1_seed1337", 2026: "a1_seed2026"}
    assert {s: (p["step"], p["digest"][:8], p["selection_sha256"][:8]) for s, p in gd.OFFICIAL_A1.items()} == {
        42: (55, "24c2aaae", "19cca1cb"), 1337: (40, "8237ec0b", "165064f5"), 2026: (40, "cba9110b", "2afbdae2")}
    assert gd.A0_WEIGHTS_SHA256.startswith("0d4db737") and gd.CORPUS_SHA256.startswith("ff9dadf3")
    frozen = gd.FROZEN_COLLECTION
    assert (frozen["corpus_passages"], frozen["excluded_passages"], frozen["eligible_passages"]) == (10137, 1745, 8392)
    assert frozen["exclusion_sha256"].startswith("9c408e00") and frozen["exclusion"] == "a1_training_and_validation"
    assert (gd.PRIMARY_SET, gd.SECONDARY_SET) == ("gold_v1", "gold_v1_full_corpus")


def test_the_metric_implementation_is_the_frozen_one(monkeypatch: pytest.MonkeyPatch) -> None:
    gd.check_metric_implementation()
    assert gd.metrics_module_sha256() == gd.METRICS_MODULE_SHA256
    qrels = [Qrel(query_id="q", passage_id="a", relevance=1), Qrel(query_id="q", passage_id="b", relevance=2)]
    hits = [SearchHit(query_id="q", passage_id=p, rank=i, score=1.0, retriever="r")
            for i, p in enumerate(["a", "b", "c"], start=1)]
    scores = gd.per_question(qrels, hits, ["q"])["q"]
    assert scores["ndcg@10"] == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    monkeypatch.setattr(gd, "METRICS_MODULE_SHA256", "0" * 64)
    with pytest.raises(gd.GoldDecisionError, match="not the frozen implementation"):
        gd.check_metric_implementation()


def test_a_gain_on_every_seed_is_an_improvement_on_the_eligible_corpus(tmp_path: Path) -> None:
    record = improving(tmp_path).analyse()
    decision = record["result"]["decision"]
    assert decision["a1_improves"] is True and decision["every_seed_improves"] is True
    assert decision["p_value"] <= gd.ALPHA and decision["statistic"] > 0
    assert record["conclusion"].startswith("A1 improves ndcg@10 over A0 on gold_v1 (eligible corpus: 999 of 1,000")
    assert record["caveat"] is None and record["unjudged_bias"] == gd.UNJUDGED_BIAS
    assert record["implementation_sha256"] == gd.implementation_sha256()
    assert record["metric_module_sha256"] == gd.METRICS_MODULE_SHA256
    assert record["corpus"]["corpus_passages"] == CORPUS and record["corpus"]["excluded_passages"] == 1
    assert set(record["result"]["other_metrics"]) == {"mrr@10", "recall@100", "ndcg@10 judged only"}
    assert "bm25" in record["result"]["references"] and "bm25" in record["result"]["judged@10"]
    sensitivity = record["reported"]["full_corpus_sensitivity"]
    assert sensitivity["label"] == gd.CONTAMINATION_LABEL and "POTENTIALLY CONTAMINATED" in sensitivity["label"]
    assert record["identity"]["secondary"]["contamination"]["secondary A1 seed 42"]["judged_training_passages"] == 3
    assert set(record["reported"]["primary_counts"]["benchmark_metrics_identical"].values()) == {60}


def test_no_difference_is_no_improvement_and_carries_the_caveat(tmp_path: Path) -> None:
    ranks = random_ranks(60, 1, 12, 2)
    record = Setup(tmp_path, ranks, {s: ranks for s in gd.SEEDS}).analyse()
    assert record["result"]["decision"]["a1_improves"] is False and record["result"]["decision"]["p_value"] == 1.0
    assert record["conclusion"].startswith("no improvement shown") and record["caveat"] == gd.NEGATIVE_CAVEAT
    assert "not evidence of no improvement" in record["caveat"]


def test_a_seed_that_does_not_improve_blocks_the_claim(tmp_path: Path) -> None:
    n = 60
    a0 = random_ranks(n, 2, 4, 3)
    record = Setup(tmp_path, a0, {42: [1] * n, 1337: [1] * n, 2026: [r + 1 for r in a0]}).analyse()
    decision = record["result"]["decision"]
    assert decision["p_value"] <= gd.ALPHA and decision["statistic"] > 0  # the mean gain alone would pass
    assert decision["every_seed_improves"] is False and decision["a1_improves"] is False
    assert "2026" in decision["reason"] and record["result"]["per_seed"]["2026"]["improves"] is False


@pytest.mark.parametrize(("set_name", "run", "change", "message"), [
    ("primary", "a1_seed1337", lambda m: m["retriever"]["validation_selection"]["record"].update(sha256="0" * 64),
     "not the official seed 1337"),
    ("primary", "a0_step0", lambda m: m["retriever"]["checkpoint"]["files"].update({"model.safetensors": "1" * 64}),
     "not a step-0 checkpoint"),
    ("primary", "a0_step0", lambda m: m.update(status="provisional_development"), "not a final result"),
    ("primary", "a0_step0", lambda m: m.update(annotation_version="gold_v0"), "annotation version"),
    ("primary", "a1_seed42", lambda m: m["retriever"].update(max_seq_length=256), "encode differently"),
    ("primary", "bm25", lambda m: m["settings"]["collection"].update(excluded_passages=0), "not the frozen eligible"),
    ("primary", "a0_step0", lambda m: m["settings"].update(isolation_mode="strict"), "isolation mode"),
    ("primary", "a1_seed2026", lambda m: m["retriever"]["checkpoint"]["trained_on"]["candidates"].update(
        manifest_sha256="c" * 64), "not the exclusion's sources"),
    ("secondary", "a0_step0", lambda m: m["settings"]["collection"].update(exclusion="x"), "records an exclusion"),
    ("secondary", "a1_seed42", lambda m: m.update(eval_set="gold_v1"), "evaluation set"),
])
def test_each_identity_precondition_refuses(tmp_path: Path, set_name: str, run: str, change: Any, message: str) -> None:
    setup = improving(tmp_path)
    setup.edit(set_name, run, change)
    with pytest.raises(gd.GoldDecisionError, match=message):
        setup.analyse()


def test_a_missing_seed_or_a_changed_run_file_is_refused(tmp_path: Path) -> None:
    setup = improving(tmp_path)
    shutil.rmtree(tmp_path / "primary" / "global" / "a1_seed2026")
    with pytest.raises(gd.GoldDecisionError, match="needs metrics.json and run.jsonl"):
        setup.analyse()
    setup.write()
    with open(tmp_path / "secondary" / "global" / "a0_step0" / "run.jsonl", "a") as fh:
        fh.write(json.dumps({"query_id": "q_0001", "rank": 99, "passage_id": "x", "score": 0.0}) + "\n")
    with pytest.raises(gd.GoldDecisionError, match="does not match the SHA-256"):
        setup.analyse()


def test_an_excluded_passage_in_a_primary_ranking_is_refused(tmp_path: Path) -> None:
    setup = improving(tmp_path, excluded={"q_0001-u5"})
    path = tmp_path / "primary" / "global" / "a1_seed42"
    rows = [json.loads(line) for line in (path / "run.jsonl").read_text().splitlines()]
    rows.append({"query_id": "q_0001", "rank": 99, "passage_id": "q_0001-u5", "score": 0.0})  # filtered after ranking
    run_bytes = _jsonl(rows)
    (path / "run.jsonl").write_bytes(run_bytes)
    setup.edit("primary", "a1_seed42", lambda m: m["run_file"].update(sha256=hashlib.sha256(run_bytes).hexdigest()))
    with pytest.raises(gd.GoldDecisionError, match="not applied before ranking"):
        setup.analyse()


def test_the_benchmark_and_the_frozen_metrics_must_agree(tmp_path: Path) -> None:
    setup = improving(tmp_path)
    setup.edit("primary", "a1_seed1337", lambda m: m["per_query"][0]["metrics"].update({"ndcg@10": 0.123}))
    with pytest.raises(gd.GoldDecisionError, match="the benchmark and the frozen metrics disagree"):
        setup.analyse()


def test_a_wrong_exclusion_set_is_refused(tmp_path: Path) -> None:
    setup = improving(tmp_path)
    with pytest.raises(gd.GoldDecisionError, match="not the frozen one"):
        gd.analyse(queries=setup.queries, qrels=setup.qrels, export_metadata=setup.metadata,
                   primary_dir=tmp_path / "primary", secondary_dir=tmp_path / "secondary",
                   excluded=frozenset({"another"}), expected_collection=setup.collection)


def test_fewer_than_the_minimum_questions_is_refused(tmp_path: Path) -> None:
    with pytest.raises(gd.GoldDecisionError, match=f"at least {gd.MIN_QUESTIONS}"):
        improving(tmp_path, n=gd.MIN_QUESTIONS - 1).analyse()


def test_ai_assisted_questions_are_left_out_of_the_primary_analysis_and_reported(tmp_path: Path) -> None:
    record = improving(tmp_path, n=70, ai_assisted=10).analyse()
    assert record["result"]["questions"] == 60
    assert record["reported"]["primary_with_ai_assisted"]["questions"] == 70
    assert record["reported"]["primary_counts"]["ai_assisted_excluded"] == 10


def test_judgments_of_excluded_passages_count_only_in_the_full_corpus_sensitivity(tmp_path: Path) -> None:
    """q_0001's only positive is a training passage: it leaves the primary analysis. q_0002 has a second positive
    that was a training passage: it counts on the full corpus only."""
    n = 61
    excluded = {"q_0001-rel", "q_0002-trained"}
    extra = [Qrel(query_id="q_0002", passage_id="q_0002-trained", relevance=2)]
    front = {"q_0002": ["q_0002-trained"]}
    record = Setup(tmp_path, random_ranks(n, 4, 12, 1), {s: random_ranks(n, 1, 3, s) for s in gd.SEEDS},
                   excluded=excluded, extra_qrels=extra, front=front).analyse()
    assert record["result"]["questions"] == 60
    assert record["corpus"]["judgments_excluded"] == 2
    assert record["corpus"]["questions_without_a_positive_after_exclusion"] == 1
    assert record["reported"]["full_corpus_sensitivity"]["primary"]["questions"] == 61


def deepen_judged_pool(setup: Setup) -> None:
    """Judge fillers u1-u8 as 0 too, so a run's top 10 without new passages is fully judged."""
    setup.qrels += [Qrel(query_id=q, passage_id=f"{q}-u{i}", relevance=0) for q in setup.questions for i in range(1, 9)]
    setup.write()


def test_unjudged_passages_count_as_non_relevant_and_are_disclosed(tmp_path: Path) -> None:
    """D6: A1's passages outside the judged pool count as 0; judged@10 shows it, and the judged-only score does not."""
    n = 60
    front = {f"q_{i:04d}": [f"q_{i:04d}-new{j}" for j in range(3)] for i in range(1, n + 1)}  # never pooled
    setup = Setup(tmp_path, [1] * n, {s: [4] * n for s in gd.SEEDS}, front=front)
    deepen_judged_pool(setup)
    record = setup.analyse()
    result = record["result"]
    assert result["decision"]["a1_improves"] is False and record["caveat"] == gd.NEGATIVE_CAVEAT
    assert result["judged@10"]["A1"] < result["judged@10"]["A0"]
    assert result["other_metrics"]["ndcg@10 judged only"]["difference"] == pytest.approx(0.0)
    assert "biased downwards" in record["unjudged_bias"]
    assert set(result["a1_judged_lower_by_more_than_gap_than"]) >= {"A0", "bm25"}


def test_the_sign_flip_test_holds_its_level_and_is_reproducible() -> None:
    rng = np.random.default_rng(7)
    rejections = sum(gd.sign_flip_p(rng.normal(size=60), flips=999, seed=s) <= 0.05 for s in range(300))
    assert 0.02 <= rejections / 300 <= 0.09
    d = rng.normal(0.1, 1.0, size=80)
    assert gd.sign_flip_p(d) == gd.sign_flip_p(d)
    assert gd.sign_flip_p(np.full(25, 0.2)) == pytest.approx(1 / (gd.FLIPS + 1), abs=1e-4)


def test_the_record_is_written_once(tmp_path: Path) -> None:
    out = tmp_path / "results" / "gold.json"
    assert gd.write_decision(out, {"a": 1}) is True
    assert gd.write_decision(out, {"a": 1}) is False
    with pytest.raises(gd.GoldDecisionError, match="never written over"):
        gd.write_decision(out, {"a": 2})
    with pytest.raises(gd.GoldDecisionError, match="data/eval"):
        gd.write_decision(tmp_path / "data" / "eval" / "x.json", {"a": 1})


def test_the_cli_writes_the_record_or_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup = improving(tmp_path)
    export = tmp_path / "export"
    export.mkdir()
    (export / "q.jsonl").write_bytes(_jsonl([q.model_dump() for q in setup.queries]))
    (export / "r.jsonl").write_bytes(_jsonl([q.model_dump() for q in setup.qrels]))
    (export / "m.json").write_text(json.dumps(setup.metadata))
    monkeypatch.setattr(gd, "exclusion_ids", lambda name: setup.excluded)
    monkeypatch.setattr(gd, "FROZEN_COLLECTION", setup.collection)
    argv = ["--queries", str(export / "q.jsonl"), "--qrels", str(export / "r.jsonl"), "--metadata",
            str(export / "m.json"), "--primary", str(tmp_path / "primary"), "--secondary", str(tmp_path / "secondary")]
    out, err = io.StringIO(), io.StringIO()
    assert gd.main([*argv, "--out", str(tmp_path / "decision.json")], out=out, err=err) == 0
    assert "A1 improves ndcg@10" in out.getvalue() and (tmp_path / "decision.json").is_file()
    shutil.rmtree(tmp_path / "secondary")
    assert gd.main([*argv, "--out", str(tmp_path / "other.json")], out=out, err=err) == 1
    assert "refused" in err.getvalue()
