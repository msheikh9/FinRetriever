"""The gold-set decision: A1 against A0 on human relevance judgments, frozen before any gold label existed.

**Approval and freeze.** Approved on 2026-11-12: D5 and D6 (the analysis), D8 (the eligible corpus) and D9 (A0).
The protocol is docs/gold_set_v1_protocol.md.
- Every number that decides is a constant of this module.
- Its SHA-256 is recorded in the protocol before annotation starts, and the decision record repeats it.
- It refuses to run unless the metric implementation is the frozen one (``METRICS_MODULE_SHA256``).

**What is compared.**
- **A0** is the step-0 checkpoint of the A1 runs: the base weights ``A0_WEIGHTS_SHA256``, scored through the same
  checkpoint pipeline as A1.
- **A1** is the three official selected checkpoints (``OFFICIAL_A1``): seed 42 step 55, seed 1337 step 40 and seed
  2026 step 40. Each is bound by its selection record, digest and weights.

The A1 checkpoints were fixed by the validation rule before the gold set existed. The gold set selects nothing.

**Inputs.**
- **The gold export:** questions, judgments and the corpus binding.
- **The benchmark's two evaluation sets,** in mode ``global``:
  - ``gold_v1`` (``PRIMARY_SET``) ranks the eligible corpus. The frozen exclusion set (``EXCLUSION``: the 1,745 passages
    the A1 runs used as training or validation positives or equivalents) is removed before ranking for every system.
  - ``gold_v1_full_corpus`` (``SECONDARY_SET``) ranks the whole corpus. It is a sensitivity analysis, labelled
    potentially contaminated.
- **Fixed run names:** ``A0_RUN``, ``A1_RUNS`` and ``REFERENCE_RUNS``. The references are reported, never deciding.

**Preconditions.** Any failure stops the analysis and no decision is made.
1. **The frozen implementations:** the metric module, and the exclusion set (its size and SHA-256).
2. **Every run:** a final result in ``global`` mode, on the pinned corpus, for its evaluation set and the export's
   annotation version, with its ``run.jsonl`` matching the SHA-256 its ``metrics.json`` records.
3. **The collection.**
   - The primary set records exactly the frozen exclusion (10,137 corpus passages, 1,745 excluded, 8,392 eligible) and
     isolation mode ``post_training_eligible``. No primary ranking contains an excluded passage.
   - The secondary set excludes nothing, with mode ``post_training_full_corpus``.
4. **The models.** A1 is exactly the three official checkpoints, and A0 is the base weights. All four use the same
   encoding settings and recorded the training and validation manifests the exclusion was derived from.
5. **Every run answers every question analysed,** and its per-question metrics in ``metrics.json`` equal the
   recomputation here: one implementation for every system.
6. **At least ``MIN_QUESTIONS`` questions** enter the primary analysis.

**The primary analysis.**
- **Questions:** the exported gold questions with a passage graded 1 or 2 in the eligible corpus. AI-assisted
  questions are excluded and reported separately.
- **Per question:** nDCG@10 for A0 and each A1 seed. The gain is 2^grade − 1 and the discount log2(rank + 1), by the
  frozen metric module. A1's score is the mean over the three seeds.
- **The statistic:** T = mean over questions of (A1 − A0).
- **The test:** a one-sided paired sign-flip test. ``FLIPS`` sign vectors over the questions (sorted by ID) come from
  ``numpy.random.Generator(PCG64(RNG_SEED))``. p = (1 + #{T*_b ≥ T − TOLERANCE}) / (FLIPS + 1).
- **The decision:** "A1 improves nDCG@10 over A0" if and only if p ≤ ``ALPHA`` and every seed's own mean difference
  from A0 is above 0. Otherwise "no improvement shown", never read as evidence of no improvement.

**Unjudged passages (D6).** A passage outside the human-judged pool counts as non-relevant for every system. The pool
was built without any model, never from A1's output, so A1's measured performance can be biased downwards.
- A positive result can support an improvement under this protocol.
- A negative result must not be read as evidence of no improvement.
- judged@10 is reported for every system, with a judged-only sensitivity score.
- A1 is flagged when its judged@10 falls more than ``JUDGED_GAP`` below A0's or a reference's.

**Reported, never deciding:**
- MRR@10 and Recall@100;
- each seed's own test;
- wins, ties and losses;
- question-bootstrap 95% intervals (``BOOTSTRAP``, ``BOOTSTRAP_SEED``);
- the analysis with the AI-assisted questions included;
- the full-corpus sensitivity analysis, labelled potentially contaminated;
- the references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from finretriever.data.schemas import Qrel, Query, SearchHit
from finretriever.eval_set.qrels_builder import judged_at_k
from finretriever.evaluation import metrics as metrics_module
from finretriever.evaluation.exclusion import (
    A1_TRAINING_AND_VALIDATION,
    EXCLUSIONS,
    exclusion_ids,
    ids_sha256,
)

KIND = "finretriever_gold_a1_vs_a0_decision"
FORMAT_VERSION = 1
PRIMARY_METRIC = "ndcg@10"
METRICS = ("ndcg@10", "mrr@10", "recall@100")
MODE = "global"
STATUS_FINAL = "final"  # benchmark.STATUS_FINAL; development results are refused
FLIPS = 10_000
RNG_SEED = 20261112
ALPHA = 0.05
TOLERANCE = 1e-12
METRIC_TOLERANCE = 1e-12  # the benchmark's per-question metrics against the recomputation here
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261113
MIN_QUESTIONS = 50
JUDGED_GAP = 0.10
SEEDS = (42, 1337, 2026)

# The frozen metric implementation (finretriever/evaluation/metrics.py): gain 2^grade - 1, discount log2(rank + 1).
METRICS_MODULE_SHA256 = "57dddd8606873558b21707caeb2714412f2aa02ed70c111c485802de03f1a7bb"

PRIMARY_SET = "gold_v1"
SECONDARY_SET = "gold_v1_full_corpus"
ISOLATION_PRIMARY = "post_training_eligible"
ISOLATION_SECONDARY = "post_training_full_corpus"
A0_RUN = "a0_step0"
A1_RUNS = {42: "a1_seed42", 1337: "a1_seed1337", 2026: "a1_seed2026"}
REFERENCE_RUNS = ("bm25", "bge_zero_shot", "e5_zero_shot")

CORPUS_SHA256 = "ff9dadf37bb9820e2c87df70a411fd189bb02776609e5ec06173dd03f2344004"
CORPUS_VERSION = "corpus_v1@1.2"
EXCLUSION = A1_TRAINING_AND_VALIDATION
FROZEN_COLLECTION: dict[str, Any] = {
    "exclusion": EXCLUSION,
    "exclusion_sha256": "9c408e005267eb66c7a7ede7c4785e2e91ccdce9757b1fafe08136f4383995c3",
    "corpus_passages": 10137,
    "excluded_passages": 1745,
    "eligible_passages": 8392,
    "sources": {"train_candidates_v3": "0586c7ca6982fbc1d9a15a952d483fbecb9f4589779d86bbb99bfba62b065284",
                "validation_pairs_v4": "40238cb06d73287821115dbe1b5c33a2dbbd294c7dd5b3c9fdc569c475e24b0e"},
}
A0_WEIGHTS_SHA256 = "0d4db737f56aaea90796b5a8d219de0eee958295a575c611f6b417ad340151da"
OFFICIAL_A1: dict[int, dict[str, Any]] = {
    42: {"run": "bge_base_ft_v3_a1-seed42", "step": 55,
         "selection_sha256": "19cca1cbaebe15ed8ea78950a2e9e1415424b89da42b7b9ea222df8599fd3556",
         "digest": "24c2aaae977ce319110a7dd979cb365fccfdac6cc89d63b1ba566e31a678beb9",
         "weights_sha256": "3dfbfa924b4d004a322ba7858899e92701b8acc0470bddd0c404b40429aa27ff"},
    1337: {"run": "bge_base_ft_v3_a1-seed1337", "step": 40,
           "selection_sha256": "165064f5ecbb97dfafdc31ac24ff7137ce81bb5d7a38d44f0f246ec87ea5bd25",
           "digest": "8237ec0b82a12830c5e41653f3597fb82516475203b8e61b262462a40597b675",
           "weights_sha256": "6d9952c3574ea3a6873ef7397e542bce96c074cf5da8b01f68427040dac84284"},
    2026: {"run": "bge_base_ft_v3_a1-seed2026", "step": 40,
           "selection_sha256": "2afbdae2f797fcffead66fa5e8cb49de104d76a83a507f7f9b9149204cc37892",
           "digest": "cba9110b7f744b0c47c3a8d428b61f3fd44224ef8aea779a5ce8b54e83c54d6c",
           "weights_sha256": "8f9dbd383b01074f5e56c03b9420155d29a758396a65d1b515e8ac95251a927b"},
}
ENCODING_KEYS = ("query_prefix", "passage_prefix", "max_seq_length")
PREFIX = "[08_decide_gold_a1_vs_a0]"
UNJUDGED_BIAS = (
    "Passages outside the human-judged pool count as non-relevant for every system. The pool was built without any "
    "model and never from A1's output, so A1's measured performance can be biased downwards (judged@10 shows by how "
    "much it rests on unjudged passages). A positive result can support an improvement under this protocol; a "
    "negative result must not be read as evidence of no improvement."
)
NEGATIVE_CAVEAT = (
    "No improvement shown is not evidence of no improvement: A1 is scored with its unjudged passages counted as "
    "non-relevant (downward bias), and the test has limited power (docs/gold_set_v1_protocol.md §3)."
)
CONTAMINATION_LABEL = (
    "Sensitivity analysis on the full corpus: POTENTIALLY CONTAMINATED. The A1 checkpoints were trained or selected "
    "with some of the judged passages, so A1's scores here may be inflated. It never decides."
)
NOTICE = (
    "Human relevance judgments (tier 3) of the gold set, scored once. The A1 checkpoints were fixed by the validation "
    "rule before the gold set existed, and the gold set selected nothing. The primary analysis ranks only the "
    "eligible corpus: the passages the A1 runs used in training or validation were removed before ranking for every "
    "system. " + UNJUDGED_BIAS
)
METHOD = {
    "primary_metric": PRIMARY_METRIC,
    "metric_convention": "finretriever.evaluation.metrics (frozen, SHA-256 in metric_module_sha256): DCG gain "
                         "2^grade - 1 with a log2(rank + 1) discount, IDCG from the judged grades; MRR and Recall "
                         "count grades 1 and 2 as relevant; an unjudged passage counts as 0",
    "mode": MODE,
    "primary": f"{PRIMARY_SET}: the eligible corpus (exclusion {EXCLUSION} removed before ranking)",
    "secondary": f"{SECONDARY_SET}: the full corpus, potentially contaminated, never deciding",
    "a0": f"the step-0 checkpoint (weights {A0_WEIGHTS_SHA256}), through the same checkpoint pipeline as A1",
    "a1": "per question, the mean over the three official selected checkpoints",
    "questions": "exported gold questions with a passage graded 1 or 2 in the eligible corpus; AI-assisted questions "
                 "excluded from the primary analysis",
    "statistic": "mean over questions of (A1 - A0)",
    "test": "one-sided paired sign flip",
    "flips": FLIPS,
    "rng": f"numpy PCG64({RNG_SEED}); columns in sorted query-ID order",
    "p_value": "(1 + #{T* >= T - 1e-12}) / (flips + 1)",
    "alpha": ALPHA,
    "consistency": "every seed's own mean difference from A0 is above 0",
    "decision": "A1 improves on A0 if and only if p <= alpha and every seed improves",
    "minimum_questions": MIN_QUESTIONS,
    "judged_gap": JUDGED_GAP,
    "bootstrap": {"resamples": BOOTSTRAP, "rng": f"numpy PCG64({BOOTSTRAP_SEED})", "interval": "percentile 95%"},
}


class GoldDecisionError(ValueError):
    """The inputs cannot support a decision; nothing is concluded."""


@dataclass(frozen=True)
class RunFiles:
    """One benchmark run: its ``metrics.json`` and ranked hits."""

    label: str
    path: Path
    metrics: dict[str, Any]
    hits: list[SearchHit]
    metrics_sha256: str
    run_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    """The SHA-256 of this module's source, recorded in every decision (the protocol pins the frozen value)."""
    return _sha256(Path(__file__))


def metrics_module_sha256() -> str:
    return _sha256(Path(metrics_module.__file__))


def check_metric_implementation() -> None:
    """Refuse unless the metric module is the frozen one."""
    if (found := metrics_module_sha256()) != METRICS_MODULE_SHA256:
        raise GoldDecisionError(f"the metric module {metrics_module.__file__} ({found[:12]}) is not the frozen "
                                f"implementation ({METRICS_MODULE_SHA256[:12]})")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _under_evaluation_data(path: str | Path) -> bool:
    parts = Path(os.path.abspath(path)).parts
    return any(parts[i:i + 2] == ("data", "eval") for i in range(len(parts) - 1))


def load_run(label: str, directory: str | Path) -> RunFiles:
    """A run directory: ``metrics.json`` and the ``run.jsonl`` whose SHA-256 it records."""
    directory = Path(directory)
    metrics_path, run_path = directory / "metrics.json", directory / "run.jsonl"
    if not metrics_path.is_file() or not run_path.is_file():
        raise GoldDecisionError(f"{label}: {directory} needs metrics.json and run.jsonl")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_sha = _sha256(run_path)
    if (metrics.get("run_file") or {}).get("sha256") != run_sha:
        raise GoldDecisionError(f"{label}: {run_path} does not match the SHA-256 its metrics.json records")
    hits = [SearchHit(query_id=r["query_id"], passage_id=r["passage_id"], rank=int(r["rank"]), score=float(r["score"]),
                      retriever=label) for r in _jsonl(run_path)]
    return RunFiles(label, directory, metrics, hits, _sha256(metrics_path), run_sha)


def load_set(set_dir: str | Path, label: str) -> tuple[RunFiles, dict[int, RunFiles], list[RunFiles]]:
    """A0, the three A1 seeds and whichever references are present, from ``<set_dir>/global/<run>/``."""
    base = Path(set_dir) / MODE
    a0 = load_run(f"{label} A0", base / A0_RUN)
    a1 = {seed: load_run(f"{label} A1 seed {seed}", base / name) for seed, name in A1_RUNS.items()}
    references = [load_run(f"{label} {name}", base / name) for name in REFERENCE_RUNS if (base / name).is_dir()]
    return a0, a1, references


def check_set(a0: RunFiles, a1: Mapping[int, RunFiles], references: Sequence[RunFiles], *, expected_set: str,
              isolation_mode: str, collection: Mapping[str, Any] | None, export_metadata: Mapping[str, Any],
              expected_collection: Mapping[str, Any]) -> dict[str, Any]:
    """Preconditions 2-4 for one evaluation set; returns its identity (collection, isolation, contamination)."""
    problems: list[str] = []
    runs = [a0, *a1.values(), *references]
    for run in runs:
        m = run.metrics
        settings = m.get("settings") or {}
        if m.get("status") != STATUS_FINAL:
            problems.append(f"{run.label}: status {m.get('status')!r}, not a final result")
        if m.get("mode") != MODE:
            problems.append(f"{run.label}: mode {m.get('mode')!r}, not {MODE!r}")
        if m.get("eval_set") != expected_set:
            problems.append(f"{run.label}: evaluation set {m.get('eval_set')!r}, not {expected_set!r}")
        if m.get("annotation_version") != export_metadata.get("annotation_version"):
            problems.append(f"{run.label}: annotation version {m.get('annotation_version')!r}, not the export's")
        corpus = m.get("corpus") or {}
        if (corpus.get("sha256"), corpus.get("version")) != (CORPUS_SHA256, CORPUS_VERSION):
            problems.append(f"{run.label}: corpus {corpus.get('version')} {corpus.get('sha256')}, not the pinned one")
        if settings.get("isolation_mode") != isolation_mode:
            problems.append(f"{run.label}: isolation mode {settings.get('isolation_mode')!r}, not {isolation_mode!r}")
        recorded = settings.get("collection") or {}
        if collection is None:
            if recorded.get("exclusion") is not None or recorded.get("excluded_passages"):
                problems.append(f"{run.label}: the full-corpus set records an exclusion")
        else:
            keys = ("exclusion", "exclusion_sha256", "corpus_passages", "excluded_passages", "eligible_passages")
            if wrong := [k for k in keys if recorded.get(k) != collection.get(k)]:
                problems.append(f"{run.label}: the collection is not the frozen eligible corpus ({', '.join(wrong)})")
    if (export_metadata.get("corpus_sha256"), export_metadata.get("corpus_version")) != (CORPUS_SHA256, CORPUS_VERSION):
        problems.append("the gold export is not bound to the pinned corpus")
    for seed, run in a1.items():
        pin = OFFICIAL_A1[seed]
        retriever = run.metrics.get("retriever") or {}
        checkpoint = retriever.get("checkpoint") or {}
        selection = retriever.get("validation_selection") or {}
        found = {"role": retriever.get("role"), "run": checkpoint.get("run"), "step": checkpoint.get("step"),
                 "digest": checkpoint.get("digest"),
                 "weights_sha256": (checkpoint.get("files") or {}).get("model.safetensors"),
                 "selection_sha256": (selection.get("record") or {}).get("sha256"), "seed": selection.get("seed"),
                 "seeds": selection.get("seeds")}
        expected = {"role": "final", "run": pin["run"], "step": pin["step"], "digest": pin["digest"],
                    "weights_sha256": pin["weights_sha256"], "selection_sha256": pin["selection_sha256"],
                    "seed": seed, "seeds": list(SEEDS)}
        if wrong := sorted(k for k in expected if found[k] != expected[k]):
            problems.append(f"{run.label}: not the official seed {seed} checkpoint ({', '.join(wrong)} differ)")
    a0_checkpoint = (a0.metrics.get("retriever") or {}).get("checkpoint") or {}
    if a0_checkpoint.get("step") != 0 or (a0_checkpoint.get("files") or {}).get("model.safetensors") != A0_WEIGHTS_SHA256:
        problems.append(f"{a0.label}: not a step-0 checkpoint with the base weights {A0_WEIGHTS_SHA256[:12]}")
    if a0_checkpoint.get("run") not in {pin["run"] for pin in OFFICIAL_A1.values()}:
        problems.append(f"{a0.label}: its checkpoint is not from an official A1 run")
    sources = expected_collection.get("sources") or {}
    contamination: dict[str, Any] = {}
    for run in (a0, *a1.values()):
        retriever = run.metrics.get("retriever") or {}
        trained = ((retriever.get("checkpoint") or {}).get("trained_on") or {}).get("candidates") or {}
        isolation = retriever.get("evaluation_isolation") or {}
        validation = ((isolation.get("validation") or {}).get("manifest") or {}).get("sha256")
        if (trained.get("manifest_sha256"), validation) != (sources.get("train_candidates_v3"),
                                                            sources.get("validation_pairs_v4")):
            problems.append(f"{run.label}: its recorded training or validation data are not the exclusion's sources")
        if isolation.get("mode") != isolation_mode or isolation.get("isolated") is not True:
            problems.append(f"{run.label}: evaluation isolation {isolation.get('mode')!r} "
                            f"(isolated {isolation.get('isolated')}), not {isolation_mode!r}")
        contamination[run.label] = isolation.get("contamination")
    encodings = {run.label: {k: (run.metrics.get("retriever") or {}).get(k) for k in ENCODING_KEYS}
                 for run in (a0, *a1.values())}
    if len({json.dumps(e, sort_keys=True) for e in encodings.values()}) != 1:
        problems.append(f"A0 and A1 encode differently: {encodings}")
    if problems:
        raise GoldDecisionError(f"{expected_set}: {len(problems)} precondition(s) failed, no decision: "
                                + "; ".join(problems))
    return {"eval_set": expected_set, "isolation_mode": isolation_mode,
            "collection": (a0.metrics.get("settings") or {}).get("collection"),
            "encoding": encodings[a0.label], "contamination": contamination,
            "references": [run.label for run in references]}


def _condensed(hits: Sequence[SearchHit], judged_pairs: set[tuple[str, str]]) -> list[SearchHit]:
    by_query: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in sorted(hits, key=lambda h: (h.query_id, h.rank)):
        if (hit.query_id, hit.passage_id) in judged_pairs:
            by_query[hit.query_id].append(hit)
    return [h.model_copy(update={"rank": i}) for query_hits in by_query.values() for i, h in enumerate(query_hits, 1)]


def per_question(qrels: Sequence[Qrel], hits: Sequence[SearchHit], questions: Sequence[str]
                 ) -> dict[str, dict[str, float]]:
    """Every reported number per question, by the frozen metric module (one implementation for every system)."""
    scores = metrics_module.compute_per_query_metrics(list(qrels), list(hits), list(METRICS))
    judged = judged_at_k(list(qrels), list(hits), k=10)
    condensed = metrics_module.compute_per_query_metrics(
        list(qrels), _condensed(hits, {(q.query_id, q.passage_id) for q in qrels}), [PRIMARY_METRIC])
    return {q: {**scores[q], "judged@10": judged.get(q, 0.0),
                f"{PRIMARY_METRIC} judged only": condensed[q][PRIMARY_METRIC]} for q in questions}


def cross_check(run: RunFiles, scores: Mapping[str, Mapping[str, float]]) -> int:
    """The benchmark's per-question metrics must equal the recomputation; returns how many questions were compared."""
    rows = {row["query_id"]: row for row in run.metrics.get("per_query") or []}
    compared = 0
    for qid, values in scores.items():
        row = rows.get(qid)
        if row is None:
            raise GoldDecisionError(f"{run.label}: metrics.json has no per-question record for {qid}")
        recorded = {**(row.get("metrics") or {}), "judged@10": row.get("judged@10")}
        for name in (*METRICS, "judged@10"):
            if recorded.get(name) is None or abs(float(recorded[name]) - values[name]) > METRIC_TOLERANCE:
                raise GoldDecisionError(f"{run.label}: {qid} {name} is {recorded.get(name)} in metrics.json but "
                                        f"{values[name]} recomputed: the benchmark and the frozen metrics disagree")
        compared += 1
    return compared


def sign_flip_p(differences: np.ndarray, *, flips: int = FLIPS, seed: int = RNG_SEED) -> float:
    """One-sided paired sign-flip p-value of mean(differences) > 0 (columns in the given order)."""
    signs = (np.random.Generator(np.random.PCG64(seed)).integers(0, 2, size=(flips, len(differences))) * 2 - 1)
    null = signs.astype(float) @ differences / len(differences)
    return (1 + int(np.sum(null >= float(differences.mean()) - TOLERANCE))) / (flips + 1)


def _interval(values: np.ndarray, resamples: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> list[float]:
    rng = np.random.Generator(np.random.PCG64(seed))
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 6), round(float(np.quantile(means, 0.975)), 6)]


def _arrays(scores: Mapping[str, Mapping[str, float]], questions: Sequence[str]) -> dict[str, np.ndarray]:
    names = next(iter(scores.values())).keys() if scores else ()
    return {name: np.array([scores[q][name] for q in questions]) for name in names}


def compare(a0: Mapping[str, np.ndarray], a1: Mapping[int, Mapping[str, np.ndarray]],
            references: Mapping[str, Mapping[str, np.ndarray]], questions: Sequence[str]) -> dict[str, Any]:
    """The test, the consistency condition, the decision and everything reported, on one set of questions."""
    seeds = sorted(a1)
    mean_a1 = {name: np.mean([a1[s][name] for s in seeds], axis=0) for name in a0}
    d = mean_a1[PRIMARY_METRIC] - a0[PRIMARY_METRIC]
    p = sign_flip_p(d)
    per_seed = {}
    for s in seeds:
        ds = a1[s][PRIMARY_METRIC] - a0[PRIMARY_METRIC]
        per_seed[str(s)] = {"mean": round(float(a1[s][PRIMARY_METRIC].mean()), 6),
                            "difference": round(float(ds.mean()), 6), "improves": bool(ds.mean() > 0),
                            "p_value": round(sign_flip_p(ds), 6),
                            "wins_ties_losses": [int((ds > 0).sum()), int((ds == 0).sum()), int((ds < 0).sum())],
                            "judged@10": round(float(a1[s]["judged@10"].mean()), 4)}
    consistent = all(v["improves"] for v in per_seed.values())
    improved = p <= ALPHA and consistent
    judged = {"A0": round(float(a0["judged@10"].mean()), 4), "A1": round(float(mean_a1["judged@10"].mean()), 4),
              **{name: round(float(r["judged@10"].mean()), 4) for name, r in references.items()}}
    lower_bound = [name for name, value in judged.items() if name != "A1" and judged["A1"] < value - JUDGED_GAP]
    return {
        "questions": len(questions),
        "decision": {
            "a1_improves": improved,
            "p_value": round(p, 6),
            "statistic": round(float(d.mean()), 6),
            "every_seed_improves": consistent,
            "reason": (f"p = {p:.4f} <= {ALPHA} and every seed improves" if improved else
                       f"p = {p:.4f} > {ALPHA}" if p > ALPHA else
                       f"p = {p:.4f} <= {ALPHA}, but seed(s) "
                       f"{[s for s, v in per_seed.items() if not v['improves']]} do not improve"),
        },
        "means": {"A0": round(float(a0[PRIMARY_METRIC].mean()), 6), "A1": round(float(mean_a1[PRIMARY_METRIC].mean()), 6),
                  **{name: round(float(r[PRIMARY_METRIC].mean()), 6) for name, r in references.items()}},
        "interval_a1_minus_a0": _interval(d),
        "wins_ties_losses": [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())],
        "per_seed": per_seed,
        "judged@10": judged,
        "a1_judged_lower_by_more_than_gap_than": lower_bound,
        "other_metrics": {name: {"A0": round(float(a0[name].mean()), 6), "A1": round(float(mean_a1[name].mean()), 6),
                                 "difference": round(float((mean_a1[name] - a0[name]).mean()), 6),
                                 "interval": _interval(mean_a1[name] - a0[name])}
                          for name in a0 if name not in (PRIMARY_METRIC, "judged@10")},
        "references": {name: {"difference_a1_minus_reference": round(float((mean_a1[PRIMARY_METRIC]
                                                                           - r[PRIMARY_METRIC]).mean()), 6),
                               "interval": _interval(mean_a1[PRIMARY_METRIC] - r[PRIMARY_METRIC])}
                       for name, r in references.items()},
    }


def _analyse_set(qrels: Sequence[Qrel], ai_assisted: set[str], a0: RunFiles, a1: Mapping[int, RunFiles],
                 references: Sequence[RunFiles], *, minimum: int) -> dict[str, Any]:
    positives = sorted({q.query_id for q in qrels if q.relevance >= 1})
    runs = [a0, *a1.values(), *references]
    for run in runs:
        if missing := sorted(set(positives) - {h.query_id for h in run.hits}):
            raise GoldDecisionError(f"{run.label} returns nothing for {len(missing)} question(s), e.g. {missing[0]}")
    scores = {run.label: per_question(qrels, run.hits, positives) for run in runs}
    compared = {run.label: cross_check(run, scores[run.label]) for run in runs}
    result: dict[str, Any] = {"questions_with_a_positive": len(positives),
                              "ai_assisted_excluded": len([q for q in positives if q in ai_assisted]),
                              "benchmark_metrics_identical": compared}
    for variant, questions in (("primary", [q for q in positives if q not in ai_assisted]),
                               ("with_ai_assisted", positives)):
        if len(questions) < (minimum if variant == "primary" else 2):
            if variant == "primary" and minimum > 2:
                raise GoldDecisionError(f"{len(questions)} questions in the primary analysis; the frozen rule needs "
                                        f"at least {MIN_QUESTIONS}")
            result[variant] = None
            continue
        result[variant] = compare(_arrays(scores[a0.label], questions),
                                  {seed: _arrays(scores[run.label], questions) for seed, run in a1.items()},
                                  {run.label.split(" ", 1)[1]: _arrays(scores[run.label], questions)
                                   for run in references}, questions)
    return result


def analyse(*, queries: Sequence[Query], qrels: Sequence[Qrel], export_metadata: Mapping[str, Any],
            primary_dir: str | Path, secondary_dir: str | Path, excluded: frozenset[str],
            expected_collection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The decision record (see the module docstring); raises ``GoldDecisionError`` if a precondition fails.
    ``expected_collection`` replaces ``FROZEN_COLLECTION`` (tests only)."""
    check_metric_implementation()
    expected_collection = expected_collection or FROZEN_COLLECTION
    if (len(excluded), ids_sha256(excluded)) != (expected_collection["excluded_passages"],
                                                 expected_collection["exclusion_sha256"]):
        raise GoldDecisionError(f"the exclusion set ({len(excluded)}, {ids_sha256(excluded)[:12]}) is not the frozen one")
    by_id = {q.query_id: q for q in queries}
    if unknown := sorted({q.query_id for q in qrels} - set(by_id)):
        raise GoldDecisionError(f"judgments for questions not in the export: {unknown[:5]}")
    if bad := [q for q in qrels if q.relevance not in (0, 1, 2)]:
        raise GoldDecisionError(f"grades outside 0, 1, 2: {bad[0]}")
    ai_assisted = {q.query_id for q in queries if q.metadata.get("ai_assisted") in (True, "true", "yes")}
    p_a0, p_a1, p_refs = load_set(primary_dir, "primary")
    s_a0, s_a1, s_refs = load_set(secondary_dir, "secondary")
    primary_identity = check_set(p_a0, p_a1, p_refs, expected_set=PRIMARY_SET, isolation_mode=ISOLATION_PRIMARY,
                                 collection=expected_collection, export_metadata=export_metadata,
                                 expected_collection=expected_collection)
    secondary_identity = check_set(s_a0, s_a1, s_refs, expected_set=SECONDARY_SET,
                                   isolation_mode=ISOLATION_SECONDARY, collection=None,
                                   export_metadata=export_metadata, expected_collection=expected_collection)
    for run in (p_a0, *p_a1.values(), *p_refs):
        if leaked := sorted({h.passage_id for h in run.hits} & excluded):
            raise GoldDecisionError(f"{run.label} ranks {len(leaked)} excluded passage(s), e.g. {leaked[0]}: the "
                                    "exclusion was not applied before ranking")
    eligible_qrels = [q for q in qrels if q.passage_id not in excluded]
    primary = _analyse_set(eligible_qrels, ai_assisted, p_a0, p_a1, p_refs, minimum=MIN_QUESTIONS)
    secondary = _analyse_set(list(qrels), ai_assisted, s_a0, s_a1, s_refs, minimum=2)
    positives_full = {q.query_id for q in qrels if q.relevance >= 1}
    positives_eligible = {q.query_id for q in eligible_qrels if q.relevance >= 1}
    result = primary["primary"]
    improved = result["decision"]["a1_improves"]
    sizes = (f"eligible corpus: {expected_collection['eligible_passages']:,} of "
             f"{expected_collection['corpus_passages']:,} passages, the {expected_collection['excluded_passages']:,} "
             "training and validation passages excluded")
    conclusion = (f"A1 improves {PRIMARY_METRIC} over A0 on {PRIMARY_SET} ({sizes})" if improved
                  else f"no improvement shown on {PRIMARY_SET} ({sizes})")
    return {
        "kind": KIND,
        "format_version": FORMAT_VERSION,
        "implementation_sha256": implementation_sha256(),
        "metric_module_sha256": metrics_module_sha256(),
        "notice": NOTICE,
        "method": METHOD,
        "corpus": {
            "corpus_passages": expected_collection["corpus_passages"],
            "eligible_passages": expected_collection["eligible_passages"],
            "excluded_passages": expected_collection["excluded_passages"],
            "exclusion": expected_collection["exclusion"],
            "exclusion_sha256": expected_collection["exclusion_sha256"],
            "judgments_excluded": len(qrels) - len(eligible_qrels),
            "questions_without_a_positive_after_exclusion": len(positives_full - positives_eligible),
        },
        "identity": {"primary": primary_identity, "secondary": secondary_identity},
        "inputs": [{"label": run.label, "path": str(run.path), "metrics_sha256": run.metrics_sha256,
                    "run_sha256": run.run_sha256}
                   for run in (p_a0, *p_a1.values(), *p_refs, s_a0, *s_a1.values(), *s_refs)],
        "conclusion": conclusion,
        "caveat": None if improved else NEGATIVE_CAVEAT,
        "unjudged_bias": UNJUDGED_BIAS,
        "result": result,
        "reported": {
            "primary_with_ai_assisted": primary["with_ai_assisted"],
            "primary_counts": {k: primary[k] for k in ("questions_with_a_positive", "ai_assisted_excluded",
                                                       "benchmark_metrics_identical")},
            "full_corpus_sensitivity": {"label": CONTAMINATION_LABEL, **secondary},
        },
    }


def write_decision(path: str | Path, decision: Mapping[str, Any]) -> bool:
    """Write the decision record once; ``False`` if an identical one is there, an error if a different one is."""
    path = Path(path)
    if _under_evaluation_data(path):
        raise GoldDecisionError(f"{path} is under data/eval; the decision is never written there")
    data = (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() == data:
            return False
        raise GoldDecisionError(f"{path} already records a different decision; it is never written over")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="08_decide_gold_a1_vs_a0.py",
                                     description="The frozen A1-versus-A0 decision on the gold set (writes once).")
    parser.add_argument("--queries", default="data/eval/gold_queries.jsonl", help="The gold export's questions.")
    parser.add_argument("--qrels", default="data/eval/gold_qrels.jsonl", help="The gold export's judgments.")
    parser.add_argument("--metadata", default="data/eval/gold_eval.metadata.json", help="The export's corpus binding.")
    parser.add_argument("--primary", default=f"results/{PRIMARY_SET}", help="The eligible-corpus set's output dir.")
    parser.add_argument("--secondary", default=f"results/{SECONDARY_SET}", help="The full-corpus set's output dir.")
    parser.add_argument("--out", default="results/decisions/gold_v1_a1_vs_a0.json",
                        help="The decision record to write once (JSON).")
    return parser


def main(argv: list[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        queries = [Query(**row) for row in _jsonl(Path(args.queries))]
        qrels = [Qrel(**{k: row[k] for k in ("query_id", "passage_id", "relevance")}) for row in _jsonl(Path(args.qrels))]
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        decision = analyse(queries=queries, qrels=qrels, export_metadata=metadata, primary_dir=args.primary,
                           secondary_dir=args.secondary, excluded=exclusion_ids(EXCLUSION))
        written = write_decision(args.out, decision)
    except (GoldDecisionError, OSError, ValueError) as exc:
        print(f"{PREFIX} refused: {exc}", file=err)
        return 1
    result = decision["result"]["decision"]
    print(f"{PREFIX} {'wrote' if written else 'unchanged (identical record present)'} {args.out}", file=out)
    print(f"{PREFIX} {decision['conclusion']}: {result['reason']}; A1 - A0 {result['statistic']:+.4f}", file=out)
    if decision["caveat"]:
        print(f"{PREFIX} {decision['caveat']}", file=out)
    print(f"{PREFIX} {UNJUDGED_BIAS}", file=out)
    return 0


__all__ = [
    "A0_RUN",
    "A1_RUNS",
    "ALPHA",
    "EXCLUSIONS",
    "FLIPS",
    "FROZEN_COLLECTION",
    "KIND",
    "METHOD",
    "METRICS_MODULE_SHA256",
    "MIN_QUESTIONS",
    "OFFICIAL_A1",
    "PRIMARY_METRIC",
    "PRIMARY_SET",
    "SECONDARY_SET",
    "GoldDecisionError",
    "RunFiles",
    "analyse",
    "check_metric_implementation",
    "compare",
    "implementation_sha256",
    "load_run",
    "main",
    "per_question",
    "sign_flip_p",
    "write_decision",
]
