"""End-to-end retrieval benchmark harness.

Glues the existing pieces together so a single function call (or one CLI
invocation of ``scripts/04_run_baselines.py``) takes the v1 inputs:

    corpus.jsonl + gold_queries.jsonl + gold_qrels.jsonl

and produces:

    results/baselines.json
    results/final_ablation_table.md

The legacy path above runs BM25 only. Every retriever plugs into the harness
via :class:`BaseRetriever`.

Named evaluation sets (``evaluation_sets`` in the benchmark config, run with
``--eval-set``) add what real, incompletely judged labels need: the corpus is
verified against the labels' metadata before anything is scored, every query
gets its own record (metrics, judged@10, and its top 10 split into judged
positives, confirmed negatives and unjudged passages), and the output records
the corpus SHA-256, the label version, the metric definitions and the
configuration. Development sets are written under ``results/dev/`` and
labelled as provisional development metrics.

A named set runs every configured retriever (BM25 and dense bi-encoders) in
every configured retrieval mode (:data:`RETRIEVAL_MODES`): ``global`` ranks
the whole corpus, ``company`` ranks only the passages of the companies a
question asks about, for every retriever alike. All runs are scored by the
same function with the same metric definitions.

A set with ``runs_from`` runs no retriever: it scores the stored rankings of an
earlier set (``<runs_from>/<mode>/<run>/run.jsonl``, SHA-256 checked) against
its own labels, so newer labels can be compared on exactly the same rankings.
With ``compare_with`` the report sets each run's scores beside those the other
set recorded for it.

A set may name its own ``retrievers`` instead of the configured baselines, such
as local fine-tuned checkpoints (type ``dense_checkpoint``,
:mod:`finretriever.retrieval.checkpoint`). Before any of them is loaded, each
checkpoint must show that it never trained on the set's judged passages or
questions (:mod:`finretriever.evaluation.isolation`). With
``reference_runs_from`` the stored baseline rankings are scored beside them with
the same labels (SHA-256 checked, never re-ranked, not written again), and with
``paired_reference`` each run is compared with that run question by question.
Every run reports a bootstrap interval for each metric and a paired comparison
when asked (:mod:`finretriever.evaluation.uncertainty`); no run is selected.

Outputs are written once: a file that already holds different results is never
replaced, and nothing is written in that case.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finretriever.data.schemas import Passage, Qrel, Query, RunResult, SearchHit
from finretriever.eval_set.qrels_builder import judged_at_k
from finretriever.evaluation.exclusion import EXCLUSIONS, collection_record, exclusion_ids
from finretriever.evaluation.isolation import (
    ISOLATION_MODES,
    ISOLATION_POST_TRAINING_ELIGIBLE,
    ISOLATION_POST_TRAINING_FULL,
    ISOLATION_STRICT,
    checkpoint_isolation,
    isolation_summary,
)
from finretriever.evaluation.metrics import compute_metrics, compute_per_query_metrics
from finretriever.evaluation.provenance import (
    CorpusIdentity,
    load_evaluation_metadata,
    sha256_file,
    verify_evaluation_corpus,
)
from finretriever.evaluation.stored_runs import METRICS_FILE, load_stored_run
from finretriever.evaluation.uncertainty import describe_method, mean_interval, paired_difference
from finretriever.retrieval.base import BaseRetriever
from finretriever.retrieval.biencoder import BiEncoderRetriever, DenseModelError
from finretriever.retrieval.bm25 import BM25Retriever
from finretriever.retrieval.checkpoint import CheckpointRetriever
from finretriever.training.checkpoint_selection import SelectionError, verify_selected_checkpoint
from finretriever.utils.io import ensure_parent_dir, read_jsonl

_DEFAULT_METRICS: tuple[str, ...] = ("ndcg@10", "mrr@10", "recall@100")
_DEFAULT_TOP_K = 100


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_benchmark_inputs(
    corpus_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
) -> tuple[list[Passage], list[Query], list[Qrel]]:
    """Load the three JSONL inputs and validate them against the schemas.

    Raises :class:`ValueError` if any of the three files is empty so a silent
    no-op benchmark (which would emit zeros) can never make it into
    ``results/``.
    """
    passages = read_jsonl(corpus_path, model=Passage)
    queries = read_jsonl(queries_path, model=Query)
    qrels = read_jsonl(qrels_path, model=Qrel)

    if not passages:
        raise ValueError(f"Empty corpus at {corpus_path!s}: nothing to index.")
    if not queries:
        raise ValueError(f"Empty query set at {queries_path!s}: nothing to retrieve.")
    if not qrels:
        raise ValueError(f"Empty qrels at {qrels_path!s}: cannot score anything.")

    return passages, queries, qrels


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def _resolve_run_name(retriever: BaseRetriever, override: str | None) -> str:
    if override:
        return override
    attr = getattr(retriever, "retriever_name", None)
    if isinstance(attr, str) and attr:
        return attr
    return type(retriever).__name__


def run_retriever_benchmark(
    retriever: BaseRetriever,
    passages: list[Passage],
    queries: list[Query],
    qrels: list[Qrel],
    top_k: int = _DEFAULT_TOP_K,
    metrics: list[str] | None = None,
    run_name: str | None = None,
) -> RunResult:
    """Index ``passages``, run every query, score against ``qrels``."""
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}.")

    resolved_run_name = _resolve_run_name(retriever, run_name)

    # Pass a shallow copy of the passage list to the retriever so we never
    # hand it the caller's own list reference (which it could choose to sort
    # or otherwise mutate). Passage objects themselves are immutable enough
    # for our purposes — the schemas use ``extra="forbid"`` and we never
    # rebind their fields.
    retriever.index(list(passages))

    query_mapping = {q.query_id: q.text for q in queries}
    hits = retriever.batch_search(query_mapping, top_k=top_k)

    aggregate = compute_metrics(qrels, hits, metrics=metrics)

    metadata: dict[str, str | int | float | bool | None] = {
        "corpus_size": len(passages),
        "query_count": len(queries),
        "qrel_count": len(qrels),
        "top_k": top_k,
        "retriever": resolved_run_name,
    }

    return RunResult(
        run_name=resolved_run_name,
        metrics=aggregate,
        hits=hits,
        metadata=metadata,
    )


def run_bm25_baseline(
    passages: list[Passage],
    queries: list[Query],
    qrels: list[Qrel],
    top_k: int = _DEFAULT_TOP_K,
    metrics: list[str] | None = None,
) -> RunResult:
    """Convenience wrapper that benchmarks a fresh :class:`BM25Retriever`."""
    retriever = BM25Retriever(retriever_name="bm25")
    return run_retriever_benchmark(
        retriever,
        passages,
        queries,
        qrels,
        top_k=top_k,
        metrics=metrics,
        run_name="bm25",
    )


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------


def results_to_jsonable(result: RunResult) -> dict[str, Any]:
    """Convert a :class:`RunResult` into a JSON-compatible dict."""
    return result.model_dump(mode="json")


def write_results(
    result: RunResult,
    output_json_path: str | Path,
    output_table_path: str | Path | None = None,
) -> None:
    """Write the result as pretty-printed JSON, plus an optional markdown table."""
    json_path = ensure_parent_dir(output_json_path)
    payload = results_to_jsonable(result)

    fd, tmp_name = tempfile.mkstemp(
        prefix=json_path.name + ".",
        suffix=".tmp",
        dir=str(json_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, json_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    if output_table_path is not None:
        table_path = ensure_parent_dir(output_table_path)
        table_path.write_text(
            format_results_markdown([result]),
            encoding="utf-8",
        )


def format_results_markdown(results: list[RunResult]) -> str:
    """Return a README-ready markdown table for one or more :class:`RunResult`."""
    columns = ("run", "ndcg@10", "mrr@10", "recall@100")
    lines: list[str] = [
        "# Baseline Results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for result in results:
        row = [result.run_name]
        for metric_name in columns[1:]:
            value = result.metrics.get(metric_name, 0.0)
            row.append(f"{value:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_metrics_from_config(config: dict[str, Any]) -> list[str]:
    """Return the metrics list from a config dict, or the default trio."""
    raw = config.get("metrics")
    if raw is None:
        return list(_DEFAULT_METRICS)
    return [str(m) for m in raw]


def load_top_k_from_config(config: dict[str, Any]) -> int:
    """Return ``top_k`` from a config dict, validated > 0."""
    raw = config.get("top_k", _DEFAULT_TOP_K)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise ValueError(f"top_k from config must be a positive integer, got {raw!r}.")
    return raw


# ---------------------------------------------------------------------------
# Named evaluation sets: provenance-checked, per-query, judged@k-aware
# ---------------------------------------------------------------------------

STATUS_DEV = "provisional_development"
STATUS_FINAL = "final"
DEV_BANNER = (
    "PROVISIONAL DEVELOPMENT METRICS. Labels are AI-generated development labels (see the "
    "label provenance), checked only by an AI-assisted audit; these numbers are not benchmark "
    "results."
)
METRIC_DEFINITIONS: dict[str, str] = {
    "ndcg@10": (
        "Graded nDCG over the top 10: gain 2^grade - 1, discount log2(rank + 1), ideal ranking "
        "from the query's judged grades. Unjudged passages earn gain 0."
    ),
    "mrr@10": (
        "1 / rank of the first passage judged relevant (grade >= 1) in the top 10, else 0. "
        "Unjudged passages count as not relevant."
    ),
    "recall@100": (
        "Judged-relevant passages (grade >= 1) in the top 100 / all judged-relevant passages of "
        "the query. The denominator counts known positives only: recall against the pooled "
        "judgments, not the corpus."
    ),
    "judged@10": "Share of the top 10 that carries a judgment of any grade (0, 1 or 2).",
    "ndcg@10_judged_only": (
        "nDCG@10 after removing unjudged passages from the ranking (condensed list). Sensitivity "
        "check: the score if every unjudged passage were set aside instead of counted as "
        "non-relevant."
    ),
    "mrr@10_judged_only": "MRR@10 on the same condensed ranking.",
}
UNJUDGED_ASSUMPTION = (
    "Qrels hold judged passages only (grades 0, 1, 2). An unjudged passage is not a confirmed "
    "negative: the standard metrics count it as non-relevant (an assumption), so each query "
    "reports how much of its top 10 is judged, how much is a confirmed negative, and how much is "
    "unjudged, split into passages from the companies the question asks about (in scope) and "
    "from other companies (out of scope)."
)
RETRIEVAL_MODES: dict[str, str] = {
    "global": "Each question ranks every passage in the corpus; no company restriction.",
    "company": (
        "Each question ranks only the passages whose ticker is one of the question's tickers "
        "(query metadata 'tickers'). Other passages are removed before ranking, never filtered "
        "out of a global top-k, and every retriever gets the same restriction."
    ),
}
POOL_DISCLOSURE = (
    "Only passages in a question's candidate pool were judged. The pools were built mainly with "
    "BM25, plus seeded random and per-filing samples and a few passages found by hand; no dense "
    "retriever proposed candidates. A retriever that ranks passages BM25 did not surface gets more "
    "unjudged passages, which count as non-relevant, so the metrics favour BM25-like rankings."
)
EXPANDED_POOL_DISCLOSURE = (
    "Only passages in a question's candidate pool were judged. The original pools were built mainly "
    "with BM25, plus seeded random and per-filing samples and a few passages found by hand. This "
    "set's pools add every passage the stored runs {runs} ranked in their top {depth}, labelled by "
    "AI proposals, so the top {depth} of those runs is judged. Below rank {depth}, passages BM25 did "
    "not pool are still unjudged and count as non-relevant, so Recall@100 still favours BM25-like "
    "rankings."
)
STORED_RUNS_NOTE = (
    "No retriever was run for this set: it scores the stored rankings in {runs_from} (each run "
    "file's SHA-256 checked against its record). Nothing was re-encoded or re-ranked; only the "
    "labels differ."
)
COMPARISON_CAUTION = (
    "Do not conclude that one retriever is better from these numbers. Labels are AI-generated "
    "development labels on 10 questions, the judged pools come mostly from BM25, and unjudged "
    "passages count as non-relevant. Read every score with its judged@10 and unjudged counts; a "
    "comparison needs pools that include the dense retrievers' results and human-judged labels."
)
CHECKPOINT_CAUTION = (
    "Every checkpoint is reported; none was selected, by this set or by any validation set. Checkpoints of a short "
    "run (the 15-step smoke test on 48 synthetic pairs) check the training mechanics: their scores are not evidence "
    "of a retrieval improvement, and none is claimed. Passages a checkpoint ranks in its top 10 that no stored run "
    "ranked there are unjudged and count as non-relevant, so read every score with its judged@10, its interval and the "
    "paired comparison."
)
SELECTED_CHECKPOINT_CAUTION = (
    "Each validation-selected checkpoint was chosen by its run's fixed rule on synthetic validation questions accepted "
    "by an AI review (not gold), never by this set, and every seed of its run is reported. Passages it ranks in its top "
    "10 that no stored run ranked there are unjudged and count as non-relevant (pools are never deepened with a "
    "fine-tuned model's output), so its scores are lower bounds: read each with its judged@10, its judged-only scores, "
    "its interval and the paired comparison. Development labels support no claim of a retrieval improvement."
)
RELEVANT_FAILURES = (
    "A relevant retrieval failure is a passage judged relevant (grade >= 1) that a run ranks below its top {k}, or "
    "does not return at all in its top {top_k}; a question with no judged-relevant passage in the top {k} has failed."
)
_DEFAULT_JUDGED_K = 10
_DEFAULT_THRESHOLDS = {"min_positives": 1, "min_judged_at_10": 0.7}
_DEFAULT_MODES = ("global",)
# Evaluation labels never come from training or validation data: these directories of a preparation are refused.
_TRAINING_DATA_DIRS = frozenset({"validation_sets", "candidates", "pairs", "carryover", "generation"})
_ROLES = ("diagnostic", "final")  # a final model must be the checkpoint its run's validation rule selected
_RETRIEVER_KEYS = {
    "bm25": {"name", "type", "tokenizer", "backend", "diagnostic"},
    "dense": {
        "name",
        "type",
        "model_name",
        "model_revision",
        "query_prefix",
        "passage_prefix",
        "batch_size",
        "device",
        "max_seq_length",
        "embedding_dim",
        "local_files_only",
        "diagnostic",
    },
    "dense_checkpoint": {
        "name",
        "type",
        "checkpoint_path",
        "selection_record",
        "role",
        "query_prefix",
        "passage_prefix",
        "batch_size",
        "device",
        "max_seq_length",
        "embedding_dim",
        "diagnostic",
    },
}


@dataclass(frozen=True)
class EvalSet:
    """A named evaluation set from the benchmark config."""

    name: str
    status: str
    queries_path: Path
    qrels_path: Path
    metadata_path: Path
    report_path: Path | None
    output_dir: Path
    candidate_pool_path: Path | None = None
    runs_from: Path | None = None  # score these stored rankings instead of running retrievers
    compare_with: str | None = None  # another set whose stored scores the report shows beside these
    retrievers: tuple[Mapping[str, Any], ...] | None = None  # run these instead of the configured baselines
    reference_runs_from: Path | None = None  # stored baseline rankings scored beside the runs, never re-ranked
    paired_reference: str | None = None  # every other run is compared with this run, question by question
    exclusion: str | None = None  # a frozen exclusion set (exclusion.EXCLUSIONS): removed before ranking
    isolation: str = ISOLATION_STRICT  # the checkpoint isolation mode (isolation.ISOLATION_MODES)


def _under_dev_results(path: Path) -> bool:
    parts = path.resolve().parts
    return any(a == "results" and b == "dev" for a, b in zip(parts, parts[1:], strict=False))


def load_eval_set(config: Mapping[str, Any], name: str) -> EvalSet:
    """Read ``evaluation_sets[name]``; development sets must write under ``results/dev/``."""
    sets = config.get("evaluation_sets") or {}
    if name not in sets:
        raise ValueError(
            f"unknown evaluation set {name!r}; the config defines {sorted(sets) or 'none'}"
        )
    entry = sets[name]
    required = ("status", "queries_path", "qrels_path", "metadata_path", "output_dir")
    if missing := [key for key in required if not entry.get(key)]:
        raise ValueError(f"evaluation set {name!r} is missing {missing}")
    status = entry["status"]
    if status not in (STATUS_DEV, STATUS_FINAL):
        raise ValueError(
            f"evaluation set {name!r} has status {status!r}; use {STATUS_DEV!r} or {STATUS_FINAL!r}"
        )
    output_dir = Path(entry["output_dir"])
    if status == STATUS_DEV and not _under_dev_results(output_dir):
        raise ValueError(
            f"development results must be written under results/dev/, not {output_dir}"
        )
    report = entry.get("report_path")
    pool = entry.get("candidate_pool_path")
    runs_from = entry.get("runs_from")
    compare_with = entry.get("compare_with")
    if compare_with is not None and (compare_with == name or compare_with not in sets):
        raise ValueError(f"evaluation set {name!r} compares with unknown set {compare_with!r}")
    if runs_from and Path(runs_from).resolve() == output_dir.resolve():
        raise ValueError(f"evaluation set {name!r} would overwrite the stored runs it scores")
    for key in ("queries_path", "qrels_path"):
        if _TRAINING_DATA_DIRS & set(Path(entry[key]).resolve().parts):
            raise ValueError(
                f"evaluation set {name!r}: {entry[key]} lies in a preparation's training or validation data; "
                "evaluation labels never come from there"
            )
    retrievers = entry.get("retrievers")
    if retrievers is not None:
        if runs_from:
            raise ValueError(f"evaluation set {name!r} scores stored runs (runs_from), so it runs no retrievers")
        if not isinstance(retrievers, list) or not retrievers or not all(isinstance(r, Mapping) for r in retrievers):
            raise ValueError(f"evaluation set {name!r}: retrievers must be a non-empty list of retriever entries")
        names = [str(r.get("name")) for r in retrievers]
        if len(set(names)) != len(names):
            raise ValueError(f"evaluation set {name!r} names a retriever twice: {names}")
    reference = entry.get("reference_runs_from")
    if reference:
        if runs_from:
            raise ValueError(f"evaluation set {name!r}: use runs_from or reference_runs_from, not both")
        if Path(reference).resolve() == output_dir.resolve():
            raise ValueError(f"evaluation set {name!r} would overwrite the stored runs it refers to")
    exclusion = entry.get("exclude_passages")
    isolation = entry.get("isolation", ISOLATION_STRICT)
    if exclusion is not None and exclusion not in EXCLUSIONS:
        raise ValueError(f"evaluation set {name!r}: unknown exclusion set {exclusion!r}; use one of {sorted(EXCLUSIONS)}")
    if isolation not in ISOLATION_MODES:
        raise ValueError(f"evaluation set {name!r}: unknown isolation mode {isolation!r}; use one of {list(ISOLATION_MODES)}")
    if (isolation == ISOLATION_POST_TRAINING_ELIGIBLE) != (exclusion is not None):
        raise ValueError(f"evaluation set {name!r}: isolation {ISOLATION_POST_TRAINING_ELIGIBLE!r} goes with an "
                         "exclusion set (exclude_passages), and an exclusion set with that isolation mode")
    if exclusion is not None and (runs_from or reference):
        raise ValueError(f"evaluation set {name!r}: stored runs were ranked on another corpus; an exclusion set needs "
                         "its retrievers run on the eligible corpus")
    if isolation == ISOLATION_POST_TRAINING_FULL and status != STATUS_FINAL:
        raise ValueError(f"evaluation set {name!r}: {ISOLATION_POST_TRAINING_FULL!r} is for final sets only")
    paired = entry.get("paired_reference")
    if paired is not None:
        known = {str(r.get("name")) for r in (retrievers or config.get("baselines") or [])}
        if reference:
            known |= {str(r.get("name")) for r in config.get("baselines") or []}
        if paired not in known:
            raise ValueError(f"evaluation set {name!r}: paired_reference {paired!r} is not one of its runs {sorted(known)}")
    return EvalSet(
        name=name,
        status=status,
        queries_path=Path(entry["queries_path"]),
        qrels_path=Path(entry["qrels_path"]),
        metadata_path=Path(entry["metadata_path"]),
        report_path=Path(report) if report else None,
        output_dir=output_dir,
        candidate_pool_path=Path(pool) if pool else None,
        runs_from=Path(runs_from) if runs_from else None,
        compare_with=compare_with,
        retrievers=tuple(retrievers) if retrievers is not None else None,
        reference_runs_from=Path(reference) if reference else None,
        paired_reference=paired,
        exclusion=exclusion,
        isolation=isolation,
    )


def build_retriever(
    entry: Mapping[str, Any],
    *,
    corpus_sha256: str | None = None,
    cache_dir: str | Path | None = None,
) -> BM25Retriever | BiEncoderRetriever:
    """The retriever a ``baselines`` or ``retrievers`` entry describes; unknown types and keys are errors.

    A dense retriever caches its passage embeddings under ``cache_dir``, keyed by
    ``corpus_sha256``. Its model is loaded when it indexes, not here; a local checkpoint
    is inspected (files checked and hashed) here, before anything is loaded.
    """
    kind = entry.get("type")
    if kind not in _RETRIEVER_KEYS:
        raise ValueError(
            f"retriever {entry.get('name')!r} has unknown type {kind!r}; use one of "
            f"{sorted(_RETRIEVER_KEYS)}"
        )
    if unknown := sorted(set(entry) - _RETRIEVER_KEYS[kind]):
        raise ValueError(f"retriever {entry.get('name')!r} has unknown settings {unknown}")
    if kind == "bm25":
        return BM25Retriever(
            retriever_name=entry["name"],
            tokenizer=entry.get("tokenizer", "whitespace"),
            backend=entry.get("backend", "auto"),
        )
    if kind == "dense_checkpoint":
        name = entry.get("name")
        if not entry.get("checkpoint_path"):
            raise ValueError(f"checkpoint retriever {name!r} needs checkpoint_path")
        role = entry.get("role", "diagnostic")
        if role not in _ROLES:
            raise ValueError(f"checkpoint retriever {name!r} has role {role!r}; use one of {list(_ROLES)}")
        selection = None
        if entry.get("selection_record"):
            try:
                selection = verify_selected_checkpoint(entry["selection_record"], entry["checkpoint_path"])
            except SelectionError as exc:
                raise ValueError(f"checkpoint retriever {name!r}: {exc}") from None
        if role == "final" and selection is None:
            raise ValueError(
                f"checkpoint retriever {name!r} is registered as a final model without a selection_record: a final "
                "model is the checkpoint its run's validation rule selected (smoke-test and fixed-step checkpoints "
                "never are)"
            )
        retriever = CheckpointRetriever(
            entry["checkpoint_path"],
            retriever_name=entry["name"],
            query_prefix=entry.get("query_prefix"),
            passage_prefix=entry.get("passage_prefix"),
            batch_size=entry.get("batch_size", 32),
            device=entry.get("device", "auto"),
            max_seq_length=entry.get("max_seq_length"),
            embedding_dim=entry.get("embedding_dim"),
            cache_dir=cache_dir,
            corpus_sha256=corpus_sha256 if cache_dir is not None else None,
            selection=selection,
            role=role,
        )
        if role == "final" and str(retriever.checkpoint.metadata.get("selection", "")).startswith("none"):
            raise ValueError(
                f"checkpoint retriever {name!r}: {entry['checkpoint_path']} was saved at a fixed step without "
                "validation; it cannot be a final model"
            )
        return retriever
    if not entry.get("model_name"):
        raise ValueError(f"dense retriever {entry.get('name')!r} needs model_name")
    return BiEncoderRetriever(
        entry["model_name"],
        retriever_name=entry["name"],
        model_revision=entry.get("model_revision"),
        query_prefix=entry.get("query_prefix"),
        passage_prefix=entry.get("passage_prefix"),
        batch_size=entry.get("batch_size", 32),
        device=entry.get("device", "auto"),
        max_seq_length=entry.get("max_seq_length"),
        embedding_dim=entry.get("embedding_dim"),
        local_files_only=bool(entry.get("local_files_only", False)),
        cache_dir=cache_dir,
        corpus_sha256=corpus_sha256 if cache_dir is not None else None,
    )


def load_retrieval_modes(config: Mapping[str, Any]) -> list[str]:
    """``retrieval_modes`` from the config (default ``[global]``), each a key of :data:`RETRIEVAL_MODES`."""
    modes = list(config.get("retrieval_modes") or _DEFAULT_MODES)
    if unknown := [m for m in modes if m not in RETRIEVAL_MODES]:
        raise ValueError(f"unknown retrieval modes {unknown}; use {sorted(RETRIEVAL_MODES)}")
    if len(set(modes)) != len(modes):
        raise ValueError(f"retrieval_modes lists a mode twice: {modes}")
    return modes


def query_tickers(query: Query) -> list[str]:
    """The companies a question asks about (comma-separated ``tickers`` metadata)."""
    return sorted({t for t in str(query.metadata.get("tickers") or "").split(",") if t})


def company_restrictions(queries: Sequence[Query]) -> dict[str, list[str]]:
    """Each query's allowed tickers for ``company`` mode; every query must name at least one."""
    restrictions = {q.query_id: query_tickers(q) for q in queries}
    if missing := sorted(qid for qid, tickers in restrictions.items() if not tickers):
        raise ValueError(f"company mode needs tickers for every query; none for {missing}")
    return restrictions


def pool_composition(pool_path: str | Path, qrels: Sequence[Qrel]) -> dict[str, Any]:
    """Which candidate source proposed each judged passage, from the pool file.

    Reads an annotation pool (``sources``) or an expanded development pool
    (``base_sources`` plus ``run_sources``, where a stored run counts as a source under
    its run name, so the BM25 run merges with BM25 pooling).
    """
    sources: dict[tuple[str, str], set[str]] = {}
    runs: set[str] = set()
    depth = 0
    for line in Path(pool_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            methods = {
                str(s["method"]) for s in row.get("sources", []) + row.get("base_sources", [])
            }
            for source in row.get("run_sources", []):
                methods.add(str(source["run"]))
                runs.add(f"{source['mode']}/{source['run']}")
                depth = max(depth, int(source["rank"]))
            sources.setdefault((row["query_id"], row["passage_id"]), set()).update(methods)
    judged: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    for qrel in qrels:
        methods = sources.get((qrel.query_id, qrel.passage_id))
        key = "+".join(sorted(methods)) if methods else "not_in_pool"
        judged[key] += 1
        if qrel.relevance > 0:
            positives[key] += 1

    def from_bm25(counts: Counter[str]) -> int:
        return sum(n for key, n in counts.items() if "bm25" in key.split("+"))

    composition: dict[str, Any] = {
        "judged": sum(judged.values()),
        "judged_proposed_by_bm25": from_bm25(judged),
        "positives": sum(positives.values()),
        "positives_proposed_by_bm25": from_bm25(positives),
        "judged_by_source": dict(sorted(judged.items())),
        "positives_by_source": dict(sorted(positives.items())),
        "sources": sorted({m for methods in sources.values() for m in methods}),
    }
    if runs:
        composition["run_sources"] = sorted(runs)
        composition["run_depth"] = depth
    return composition


def pool_disclosure(pool: Mapping[str, Any] | None) -> str:
    """The candidate-pool disclosure for a set: the expanded-pool text when stored runs pooled too."""
    if pool and pool.get("run_sources"):
        return EXPANDED_POOL_DISCLOSURE.format(
            runs=", ".join(pool["run_sources"]), depth=pool["run_depth"]
        )
    return POOL_DISCLOSURE


def validate_eval_inputs(
    passages: Sequence[Passage], queries: Sequence[Query], qrels: Sequence[Qrel]
) -> None:
    """Raise unless queries and qrels are consistent with each other and with the corpus."""
    query_ids = [q.query_id for q in queries]
    if duplicates := sorted({qid for qid in query_ids if query_ids.count(qid) > 1}):
        raise ValueError(f"duplicate query IDs: {duplicates}")
    pairs: dict[tuple[str, str], int] = {}
    for qrel in qrels:
        key = (qrel.query_id, qrel.passage_id)
        if key in pairs:
            raise ValueError(f"conflicting labels: more than one qrel for {key}")
        pairs[key] = qrel.relevance
    known = set(query_ids)
    if unknown_q := sorted({q for q, _ in pairs} - known):
        raise ValueError(f"qrels for queries not in the query set: {unknown_q}")
    corpus_ids = {p.passage_id for p in passages}
    if unknown_p := sorted({p for _, p in pairs} - corpus_ids):
        raise ValueError(f"qrels for passages not in the corpus: {unknown_p[:5]}")
    if unjudged := sorted(known - {q for q, _ in pairs}):
        raise ValueError(f"queries without any judgment cannot be scored: {unjudged}")


def _condensed(hits: Sequence[SearchHit], judged: Mapping[str, set[str]]) -> list[SearchHit]:
    """Each query's hits with unjudged passages removed and ranks renumbered."""
    out: list[SearchHit] = []
    rank: dict[str, int] = {}
    for hit in sorted(hits, key=lambda h: (h.query_id, h.rank)):
        if hit.passage_id in judged.get(hit.query_id, ()):
            rank[hit.query_id] = rank.get(hit.query_id, 0) + 1
            out.append(hit.model_copy(update={"rank": rank[hit.query_id]}))
    return out


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_eval_set_run(
    queries: Sequence[Query],
    qrels: Sequence[Qrel],
    hits: Sequence[SearchHit],
    passages: Mapping[str, Passage],
    *,
    metrics: Sequence[str],
    judged_k: int = _DEFAULT_JUDGED_K,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate and per-query results; unjudged passages stay apart from confirmed negatives."""
    limits = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    grades: dict[str, dict[str, int]] = {}
    for qrel in qrels:
        grades.setdefault(qrel.query_id, {})[qrel.passage_id] = qrel.relevance
    standard = compute_per_query_metrics(list(qrels), list(hits), list(metrics))
    judged_only = compute_per_query_metrics(
        list(qrels), _condensed(hits, {q: set(g) for q, g in grades.items()}), ["ndcg@10", "mrr@10"]
    )
    judged_share = judged_at_k(list(qrels), list(hits), k=judged_k)
    by_query: dict[str, list[SearchHit]] = {}
    seen: set[tuple[str, str]] = set()
    for hit in sorted(hits, key=lambda h: (h.query_id, h.rank)):
        if (hit.query_id, hit.passage_id) not in seen:  # ranked lists count each passage once
            seen.add((hit.query_id, hit.passage_id))
            by_query.setdefault(hit.query_id, []).append(hit)

    rows: list[dict[str, Any]] = []
    for query in sorted(queries, key=lambda q: q.query_id):
        qid = query.query_id
        judged = grades.get(qid, {})
        tickers = set(query_tickers(query))
        ranked = by_query.get(qid, [])
        top = ranked[:judged_k]
        top_grades = [judged.get(h.passage_id) for h in top]
        unjudged = [h for h, g in zip(top, top_grades, strict=True) if g is None]
        in_scope = [h for h in unjudged if not tickers or passages[h.passage_id].ticker in tickers]
        first = next(
            (i for i, h in enumerate(ranked, start=1) if judged.get(h.passage_id, 0) > 0), None
        )
        positives = sum(1 for g in judged.values() if g > 0)
        rank_of = {h.passage_id: i for i, h in enumerate(ranked, start=1)}
        missed = sorted(
            (
                {"passage_id": pid, "grade": grade, "rank": rank_of.get(pid)}
                for pid, grade in judged.items()
                if grade > 0 and rank_of.get(pid, judged_k + 1) > judged_k
            ),
            key=lambda m: (m["rank"] is None, m["rank"] or 0, -m["grade"], m["passage_id"]),
        )
        scores = {
            **standard.get(qid, {}),
            **{f"{k}_judged_only": v for k, v in judged_only.get(qid, {}).items()},
        }
        flags = []
        if positives < limits["min_positives"]:
            flags.append("no_positive_judgments")
        if judged_share.get(qid, 0.0) < limits["min_judged_at_10"]:
            flags.append("low_judged_at_10")
        rows.append(
            {
                "query_id": qid,
                "text": query.text,
                "tickers": sorted(tickers),
                "metrics": scores,
                f"judged@{judged_k}": judged_share.get(qid, 0.0),
                "judgments": {
                    "judged": len(judged),
                    "positives": positives,
                    "grade_2": sum(1 for g in judged.values() if g == 2),
                    "grade_1": sum(1 for g in judged.values() if g == 1),
                    "confirmed_negatives": sum(1 for g in judged.values() if g == 0),
                },
                f"top{judged_k}": {
                    "returned": len(top),
                    "relevant": sum(1 for g in top_grades if g is not None and g > 0),
                    "grade_2": sum(1 for g in top_grades if g == 2),
                    "grade_1": sum(1 for g in top_grades if g == 1),
                    "confirmed_negatives": sum(1 for g in top_grades if g == 0),
                    "unjudged": len(unjudged),
                    "unjudged_in_scope": len(in_scope),
                    "unjudged_out_of_scope": len(unjudged) - len(in_scope),
                },
                "first_relevant_rank": first,
                f"relevant_outside_top{judged_k}": missed,
                "coverage_flags": flags,
            }
        )
    metric_names = list(rows[0]["metrics"]) if rows else []
    aggregate = {name: _mean([r["metrics"][name] for r in rows]) for name in metric_names}
    aggregate[f"judged@{judged_k}"] = _mean([r[f"judged@{judged_k}"] for r in rows])
    aggregate["queries"] = len(rows)
    for key in (
        "unjudged",
        "unjudged_in_scope",
        "unjudged_out_of_scope",
        "confirmed_negatives",
        "relevant",
    ):
        aggregate[f"top{judged_k}_{key}_total"] = sum(r[f"top{judged_k}"][key] for r in rows)
    aggregate["positives_total"] = sum(r["judgments"]["positives"] for r in rows)
    aggregate["queries_with_insufficient_coverage"] = [
        r["query_id"] for r in rows if r["coverage_flags"]
    ]
    outside = f"relevant_outside_top{judged_k}"
    aggregate[f"{outside}_total"] = sum(len(r[outside]) for r in rows)
    aggregate["relevant_not_retrieved_total"] = sum(1 for r in rows for m in r[outside] if m["rank"] is None)
    aggregate[f"queries_without_relevant_in_top{judged_k}"] = [
        r["query_id"] for r in rows if r["judgments"]["positives"] and not r[f"top{judged_k}"]["relevant"]
    ]
    return aggregate, rows


@dataclass
class EvalSetRun:
    """One retriever scored on one evaluation set in one retrieval mode."""

    run_name: str
    retriever: dict[str, Any]
    hits: list[SearchHit]
    aggregate: dict[str, Any]
    per_query: list[dict[str, Any]]
    diagnostic: bool = False
    mode: str = "global"
    source: dict[str, Any] | None = None  # the stored run it re-scores, if any
    reference: bool = False  # a stored baseline scored beside the set's runs (reported, not written again)
    uncertainty: dict[str, Any] | None = None  # a bootstrap interval per metric
    paired: dict[str, Any] | None = None  # this run against the set's paired_reference, per metric

    @property
    def run_id(self) -> str:
        return f"{self.mode}/{self.run_name}"


@dataclass
class EvalSetResult:
    """Everything a named evaluation set run produced, plus the provenance to reproduce it."""

    eval_set: EvalSet
    corpus: CorpusIdentity
    annotation_version: str
    label_report: dict[str, Any] | None
    inputs: dict[str, dict[str, str]]
    settings: dict[str, Any]
    runs: list[EvalSetRun] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    pool: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    checked: dict[str, Any] | None = None  # with load_models=False: each retriever's checks, nothing run


def run_eval_set(
    config: Mapping[str, Any],
    name: str,
    *,
    config_path: str | Path | None = None,
    load_models: bool = True,
) -> EvalSetResult:
    """Verify provenance, then score every configured retriever in every retrieval mode.

    With ``load_models=False`` (a pre-flight check) everything up to loading a model is done: provenance,
    inputs, checkpoint inspection, evaluation isolation and the stored reference runs, with their intervals.
    No retriever is run; ``checked`` records each one's checks, and ``runs`` holds only the references.

    The corpus is checked against the set's metadata (SHA-256 and version) before any
    file is scored or any model is loaded, and when the set has an export report its
    queries and qrels must match the SHA-256 recorded there. Each retriever indexes the
    corpus once and then searches in each mode; all runs are scored by
    :func:`evaluate_eval_set_run` with the same metrics. A dense model that cannot be
    loaded is listed in ``skipped`` with the reason, and the other retrievers still run.
    """
    eval_set = load_eval_set(config, name)
    corpus_path = config.get("corpus_path")
    if not corpus_path:
        raise ValueError("the benchmark config needs corpus_path for named evaluation sets")
    corpus = verify_evaluation_corpus(eval_set.metadata_path, corpus_path)
    metadata = load_evaluation_metadata(eval_set.metadata_path)
    inputs = {
        "queries": {
            "path": str(eval_set.queries_path),
            "sha256": sha256_file(eval_set.queries_path),
        },
        "qrels": {"path": str(eval_set.qrels_path), "sha256": sha256_file(eval_set.qrels_path)},
        "metadata": {
            "path": str(eval_set.metadata_path),
            "sha256": sha256_file(eval_set.metadata_path),
        },
    }
    label_report = None
    if eval_set.report_path is not None:
        label_report = json.loads(eval_set.report_path.read_text(encoding="utf-8"))
        inputs["report"] = {
            "path": str(eval_set.report_path),
            "sha256": sha256_file(eval_set.report_path),
        }
        recorded = label_report.get("files", {})
        for key, path in (("queries", eval_set.queries_path), ("qrels", eval_set.qrels_path)):
            if recorded.get(path.name) != inputs[key]["sha256"]:
                raise ValueError(
                    f"{path} does not match the SHA-256 recorded in {eval_set.report_path}"
                )
        if eval_set.status == STATUS_DEV and label_report.get("status") != STATUS_DEV:
            raise ValueError(f"{eval_set.report_path} is not a provisional development export")
    passages, queries, qrels = load_benchmark_inputs(
        corpus_path, eval_set.queries_path, eval_set.qrels_path
    )
    collection: dict[str, Any] = {"corpus_passages": len(passages), "exclusion": None,
                                  "eligible_passages": len(passages), "excluded_passages": 0}
    indexed = list(passages)
    collection_sha256 = corpus.corpus_sha256  # the embedding-cache key of the passages each retriever indexes
    if eval_set.exclusion is not None:
        # Removed before ranking: every retriever indexes only the eligible passages, and judgments of excluded
        # passages are dropped (they can never be retrieved). Questions left without any judgment are set aside.
        excluded = exclusion_ids(eval_set.exclusion)
        collection = collection_record(eval_set.exclusion, excluded, [p.passage_id for p in passages])
        indexed = [p for p in passages if p.passage_id not in excluded]
        kept = [q for q in qrels if q.passage_id not in excluded]
        collection["judgments_excluded"] = len(qrels) - len(kept)
        left = {q.query_id for q in kept}
        collection["questions_set_aside_without_judgments"] = sorted(q.query_id for q in queries if q.query_id not in left)
        queries, qrels = [q for q in queries if q.query_id in left], kept
        collection_sha256 = hashlib.sha256(
            f"{corpus.corpus_sha256}:{collection['exclusion_sha256']}".encode()).hexdigest()
        collection["embedding_cache_key"] = collection_sha256
    validate_eval_inputs(passages, queries, qrels)
    pool = None
    if eval_set.candidate_pool_path is not None:
        inputs["candidate_pool"] = {
            "path": str(eval_set.candidate_pool_path),
            "sha256": sha256_file(eval_set.candidate_pool_path),
        }
        pool = pool_composition(eval_set.candidate_pool_path, qrels)

    metrics = load_metrics_from_config(dict(config))
    top_k = load_top_k_from_config(dict(config))
    judged_k = int(config.get("judged_k", _DEFAULT_JUDGED_K))
    thresholds = {**_DEFAULT_THRESHOLDS, **(config.get("coverage_thresholds") or {})}
    modes = load_retrieval_modes(config)
    restrictions = {
        "global": None,
        "company": company_restrictions(queries) if "company" in modes else None,
    }
    cache_dir = config.get("embedding_cache_dir")
    baselines = list(config.get("baselines") or [])
    entries = list(eval_set.retrievers) if eval_set.retrievers is not None else baselines
    retrievers = (
        []
        if eval_set.runs_from is not None
        else [
            (entry, build_retriever(entry, corpus_sha256=collection_sha256, cache_dir=cache_dir))
            for entry in entries
        ]
    )
    check_final_seeds([r for _, r in retrievers if isinstance(r, CheckpointRetriever)])
    isolation = {  # before any model is loaded: a checkpoint that may have seen these labels is never run
        retriever.retriever_name: checkpoint_isolation(retriever.checkpoint, corpus=corpus, queries=queries, qrels=qrels,
                                                       mode=eval_set.isolation)
        for _, retriever in retrievers
        if isinstance(retriever, CheckpointRetriever)
    }
    settings = {
        "benchmark_config": None if config_path is None else str(config_path),
        "benchmark_config_sha256": None if config_path is None else sha256_file(config_path),
        "metrics": metrics,
        "top_k": top_k,
        "judged_k": judged_k,
        "coverage_thresholds": thresholds,
        "retrieval_modes": modes,
        "embedding_cache_dir": None if cache_dir is None else str(cache_dir),
    }
    if eval_set.exclusion is not None or eval_set.isolation != ISOLATION_STRICT:
        # Recorded only for the post-training modes, so the outputs of every other set stay byte-identical.
        settings.update(collection=collection, isolation_mode=eval_set.isolation)
    if eval_set.runs_from is not None:
        settings["runs_from"] = str(eval_set.runs_from)
    if eval_set.reference_runs_from is not None:
        settings["reference_runs_from"] = str(eval_set.reference_runs_from)
    if eval_set.paired_reference is not None:
        settings["paired_reference"] = eval_set.paired_reference
    settings["uncertainty"] = describe_method()
    result = EvalSetResult(
        eval_set,
        corpus,
        metadata.annotation_version,
        label_report,
        inputs,
        settings,
        pool=pool,
    )
    by_id = {p.passage_id: p for p in passages}
    if eval_set.runs_from is not None:
        _score_stored_runs(result, entries, queries, qrels, by_id, eval_set.runs_from)
    if eval_set.reference_runs_from is not None:
        _score_stored_runs(result, baselines, queries, qrels, by_id, eval_set.reference_runs_from, reference=True)
    if not load_models:
        result.checked = {
            retriever.retriever_name: {
                "type": entry.get("type"),
                "checkpoint": retriever.checkpoint.provenance() if isinstance(retriever, CheckpointRetriever) else None,
                "evaluation_isolation": isolation.get(retriever.retriever_name),
            }
            for entry, retriever in retrievers
        }
        _add_uncertainty(result.runs, metrics=metrics, judged_k=judged_k, paired_reference=eval_set.paired_reference)
        return result
    query_mapping = {q.query_id: q.text for q in queries}
    while retrievers:  # popped, so each model is released before the next one loads
        entry, retriever = retrievers.pop(0)
        runs: list[EvalSetRun] = []
        try:
            retriever.index(indexed)
            for mode in modes:
                hits = retriever.batch_search(
                    query_mapping, top_k=top_k, tickers=restrictions[mode]
                )
                aggregate, per_query = evaluate_eval_set_run(
                    queries,
                    qrels,
                    hits,
                    by_id,
                    metrics=metrics,
                    judged_k=judged_k,
                    thresholds=thresholds,
                )
                runs.append(
                    EvalSetRun(
                        run_name=retriever.retriever_name,
                        retriever={},
                        hits=hits,
                        aggregate=aggregate,
                        per_query=per_query,
                        diagnostic=bool(entry.get("diagnostic", False)),
                        mode=mode,
                    )
                )
        except DenseModelError as exc:
            result.skipped.append({"name": str(entry.get("name")), "reason": str(exc)})
            continue
        description = retriever.describe()
        if retriever.retriever_name in isolation:
            description["evaluation_isolation"] = isolation[retriever.retriever_name]
        for run in runs:
            run.retriever = description
        result.runs.extend(runs)
    if not any(not run.reference for run in result.runs):
        reasons = "; ".join(f"{s['name']}: {s['reason']}" for s in result.skipped)
        raise ValueError(f"no configured retriever could be run ({reasons or 'none configured'})")
    _add_uncertainty(result.runs, metrics=metrics, judged_k=judged_k, paired_reference=eval_set.paired_reference)
    if eval_set.compare_with is not None:
        result.comparison = _load_comparison(config, eval_set.compare_with, result.runs)
    return result


def check_final_seeds(checkpoints: Sequence[CheckpointRetriever]) -> None:
    """Final models are reported for every seed of their run, once each: none is left out or picked."""
    runs: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for retriever in checkpoints:
        if retriever.role == "final" and retriever.selection is not None:
            selection = retriever.selection
            experiment = str(selection["run"]).rsplit("-seed", 1)[0]
            runs.setdefault((experiment, str(selection["config_sha256"])), []).append(selection)
    for (experiment, _), selections in runs.items():
        listed = sorted(int(s["seed"]) for s in selections)
        trained = sorted(int(seed) for seed in selections[0]["seeds"])
        if listed != trained:
            raise ValueError(
                f"final models of {experiment}: seeds {listed} are listed, but its config trains seeds {trained}; every "
                "seed's selected checkpoint is reported, once"
            )


def checkpoint_cautions(runs: Sequence[EvalSetRun]) -> list[str]:
    """The cautions a report on these runs carries: fixed-step checkpoints, validation-selected ones, or both."""
    described = [run.retriever for run in runs if "checkpoint" in run.retriever]
    cautions = []
    if any(not d.get("validation_selection") for d in described):
        cautions.append(CHECKPOINT_CAUTION)
    if any(d.get("validation_selection") for d in described):
        cautions.append(SELECTED_CHECKPOINT_CAUTION)
    return cautions


def _add_uncertainty(
    runs: Sequence[EvalSetRun], *, metrics: Sequence[str], judged_k: int, paired_reference: str | None
) -> None:
    """A bootstrap interval for every metric of every run; with ``paired_reference``, each other run against that run
    in the same mode, question by question. Nothing is selected or ranked by these numbers."""
    names = [*metrics, f"judged@{judged_k}"]

    def values(run: EvalSetRun, name: str) -> dict[str, float]:
        return {r["query_id"]: float(r[name] if name.startswith("judged@") else r["metrics"][name]) for r in run.per_query}

    for run in runs:
        run.uncertainty = {name: mean_interval(list(values(run, name).values())) for name in names}
    if paired_reference is None:
        return
    for run in runs:
        base = next((r for r in runs if r.run_name == paired_reference and r.mode == run.mode), None)
        if base is None or base is run:
            continue
        run.paired = {
            "reference": base.run_id,
            "reference_source": base.source,
            "metrics": {name: paired_difference(values(run, name), values(base, name)) for name in names},
        }


def _score_stored_runs(
    result: EvalSetResult,
    entries: Sequence[Mapping[str, Any]],
    queries: Sequence[Query],
    qrels: Sequence[Qrel],
    passages: Mapping[str, Passage],
    runs_from: Path,
    *,
    reference: bool = False,
) -> None:
    """Score every configured (retriever, mode) ranking stored under ``runs_from`` against this set's labels."""
    settings = result.settings
    query_ids = {q.query_id for q in queries}
    for entry in entries:
        for mode in settings["retrieval_modes"]:
            stored = load_stored_run(
                runs_from, mode, str(entry["name"]), corpus=result.corpus, passages=passages
            )
            if stored.top_k != settings["top_k"]:
                raise ValueError(
                    f"stored run {stored.run_id} was ranked to depth {stored.top_k}, but this set scores "
                    f"top_k {settings['top_k']}; Recall@{settings['top_k']} would not be comparable"
                )
            if (ranked := set(stored.ranked())) != query_ids:
                raise ValueError(
                    f"stored run {stored.run_id} ranks questions {sorted(ranked ^ query_ids)} that differ "
                    "from this set's queries"
                )
            aggregate, per_query = evaluate_eval_set_run(
                queries,
                qrels,
                list(stored.hits),
                passages,
                metrics=settings["metrics"],
                judged_k=settings["judged_k"],
                thresholds=settings["coverage_thresholds"],
            )
            result.runs.append(
                EvalSetRun(
                    run_name=stored.run,
                    retriever=dict(stored.metrics.get("retriever") or {}),
                    hits=list(stored.hits),
                    aggregate=aggregate,
                    per_query=per_query,
                    diagnostic=bool(entry.get("diagnostic", False)),
                    mode=mode,
                    source=stored.provenance(),
                    reference=reference,
                )
            )


def _load_comparison(
    config: Mapping[str, Any], name: str, runs: Sequence[EvalSetRun]
) -> dict[str, Any]:
    """The scores another set stored for the same runs, and whether its rankings are the same files."""
    other = load_eval_set(config, name)
    compared: dict[str, Any] = {}
    label_version = None
    for run in runs:
        path = other.output_dir / run.mode / run.run_name / METRICS_FILE
        if not path.is_file():
            raise ValueError(f"cannot compare with {name!r}: {path} is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        label_version = payload.get("label_version")
        same = (payload.get("run_file") or {}).get("sha256") == (run.source or {}).get("run_sha256")
        compared[run.run_id] = {
            "metrics_file": str(path),
            "metrics_sha256": sha256_file(path),
            "same_ranking": same,
            "aggregate": payload["aggregate"],
            "per_query": {row["query_id"]: row for row in payload["per_query"]},
        }
    return {"eval_set": name, "label_version": label_version, "runs": compared}


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_eval_set_outputs(result: EvalSetResult) -> dict[str, Path]:
    """Write ``<mode>/<run>/metrics.json`` and ``<mode>/<run>/run.jsonl`` per run, plus ``report.md``.

    Written once: if any of these files already holds something else, nothing is written. Reference runs (stored
    baselines scored beside the set's runs) appear in the report and in each paired comparison, not as files.
    """
    eval_set = result.eval_set
    if eval_set.status == STATUS_DEV and not _under_dev_results(eval_set.output_dir):
        raise ValueError(
            f"development results must be written under results/dev/, not {eval_set.output_dir}"
        )
    grades: dict[tuple[str, str], int] = {}
    for line in eval_set.qrels_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grades[row["query_id"], row["passage_id"]] = row["relevance"]
    paths: dict[str, Path] = {}
    outputs: dict[Path, bytes] = {}
    comparison = result.comparison or {}
    checkpoints = any("checkpoint" in run.retriever for run in result.runs)
    for run in result.runs:
        if run.reference:
            continue
        run_dir = eval_set.output_dir / run.mode / run.run_name
        hit_rows = [
            {
                "query_id": h.query_id,
                "rank": h.rank,
                "passage_id": h.passage_id,
                "score": h.score,
                "ticker": h.metadata.get("ticker"),
                "grade": grades.get((h.query_id, h.passage_id)),
                "judged": (h.query_id, h.passage_id) in grades,
            }
            for h in sorted(run.hits, key=lambda h: (h.query_id, h.rank))
        ]
        run_bytes = _jsonl_bytes(hit_rows)
        payload = {
            "status": eval_set.status,
            "notice": DEV_BANNER if eval_set.status == STATUS_DEV else "",
            "eval_set": eval_set.name,
            "run": run.run_name,
            "mode": run.mode,
            "mode_definition": RETRIEVAL_MODES[run.mode],
            "annotation_version": result.annotation_version,
            "label_version": (result.label_report or {}).get("label_version"),
            "corpus": {
                "path": str(result.corpus.corpus_path),
                "sha256": result.corpus.corpus_sha256,
                "version": result.corpus.corpus_version,
                "passages": result.corpus.passages,
            },
            "inputs": result.inputs,
            "labels": (
                None
                if result.label_report is None
                else {
                    "description": result.label_report.get("description"),
                    "summary": result.label_report.get("labels"),
                    "audit_independence": (result.label_report.get("audit") or {}).get(
                        "independence"
                    ),
                    "exceptions": result.label_report.get("exceptions"),
                }
            ),
            "retriever": run.retriever,
            "diagnostic": run.diagnostic,
            "settings": result.settings,
            "metric_definitions": METRIC_DEFINITIONS,
            "unjudged_assumption": UNJUDGED_ASSUMPTION,
            "candidate_pools": {
                "disclosure": pool_disclosure(result.pool),
                "composition": result.pool,
            },
            "comparison_caution": COMPARISON_CAUTION,
            "aggregate": run.aggregate,
            "uncertainty": run.uncertainty,
            "per_query": run.per_query,
            "run_file": {"path": "run.jsonl", "sha256": hashlib.sha256(run_bytes).hexdigest()},
            "skipped_retrievers": result.skipped,
        }
        if run.source is not None:
            payload["source_run"] = run.source
        if run.paired is not None:
            payload["paired_comparison"] = run.paired
        if checkpoints:
            cautions = checkpoint_cautions(result.runs)
            payload["checkpoint_caution"] = cautions[0] if len(cautions) == 1 else cautions
        if comparison:
            earlier = comparison["runs"][run.run_id]
            payload["compared_with"] = {
                "eval_set": comparison["eval_set"],
                "label_version": comparison["label_version"],
                "metrics_file": earlier["metrics_file"],
                "metrics_sha256": earlier["metrics_sha256"],
                "same_ranking": earlier["same_ranking"],
                "aggregate": earlier["aggregate"],
            }
        outputs[run_dir / "run.jsonl"] = run_bytes
        outputs[run_dir / "metrics.json"] = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        paths[f"{run.run_id}/metrics.json"] = run_dir / "metrics.json"
        paths[f"{run.run_id}/run.jsonl"] = run_dir / "run.jsonl"
    report = eval_set.output_dir / "report.md"
    outputs[report] = format_eval_set_markdown(result).encode("utf-8")
    paths["report.md"] = report
    if taken := [path for path, data in outputs.items() if path.exists() and path.read_bytes() != data]:
        raise ValueError(
            f"{taken[0]} already holds different results ({len(taken)} files would change); results are never "
            "overwritten, so nothing was written; use a new output_dir"
        )
    for path, data in outputs.items():
        if not path.exists():
            _atomic_write_bytes(path, data)
    return paths


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _pct(part: int, whole: int) -> str:
    return f"{part} of {whole} ({100 * part / whole:.1f}%)" if whole else f"{part} of {whole}"


def _describe_retriever(retriever: Mapping[str, Any]) -> str:
    """One line on the settings that determine a retriever's ranking."""
    if retriever.get("type") != "dense":
        return (
            f"BM25: tokenizer `{retriever.get('tokenizer')}`, backend `{retriever.get('backend')}`, "
            f"k1 {retriever.get('k1')}, b {retriever.get('b')}."
        )
    model = retriever.get("model") or {}
    truncation = retriever.get("passage_truncation") or {}
    cache = retriever.get("cache") or {}
    passage_prefix = retriever.get("passage_prefix")
    checkpoint = retriever.get("checkpoint")
    if checkpoint:
        trained = checkpoint.get("trained_on") or {}
        pairs, candidates = trained.get("pairs") or {}, trained.get("candidates") or {}
        identity = (
            f"local checkpoint `{checkpoint['path']}` (run `{checkpoint['run']}`, step {checkpoint['step']} of "
            f"{checkpoint['total_steps']}, files digest `{checkpoint['digest'][:16]}`), fine-tuned from "
            f"`{checkpoint['base_model'].get('name')}` at `{checkpoint['base_model'].get('revision')}` on "
            f"{pairs.get('count')} synthetic pairs of `{candidates.get('version')}`; selection: "
            f"{checkpoint.get('selection')}"
        )
        if chosen := retriever.get("validation_selection"):
            identity += (
                f"; {retriever.get('role')} model: selected by its run's validation rule at step "
                f"{chosen['selected_step']}{' (the base model)' if chosen['base_model'] else ''}, seed {chosen['seed']} "
                f"of {chosen['seeds']}, whole-corpus validation MRR@10 {chosen['value']:.4f} on "
                f"`{chosen['validation'].get('version')}` (record SHA-256 `{chosen['record']['sha256'][:16]}`)"
            )
    else:
        identity = f"`{retriever.get('model_name')}` at revision `{retriever.get('model_revision')}`"
    parts = [
        identity,
        f"query prefix `{retriever.get('query_prefix')!r}`, passage prefix "
        + (f"`{passage_prefix!r}`" if passage_prefix else "none"),
        f"{retriever.get('embedding_dim')} dimensions, {model.get('pooling')} pooling, "
        "L2-normalised, cosine (dot product)",
        f"max_seq_length {retriever.get('max_seq_length')}, batch size "
        f"{retriever.get('batch_size')}, device `{retriever.get('device')}`",
    ]
    if truncation:
        parts.append(
            f"{_pct(truncation['truncated'], truncation['texts'])} passages truncated, "
            f"{100 * truncation['tokens_beyond_limit_share']:.1f}% of passage tokens beyond the limit"
        )
    parts.append(f"embeddings SHA-256 `{retriever.get('embeddings_sha256')}`")
    if cache:
        parts.append(f"cache key `{cache.get('key_sha256', '')[:16]}`")
    if isolation := retriever.get("evaluation_isolation"):
        parts.append(f"evaluation isolation: {isolation_summary(isolation)}")
    return "; ".join(parts) + "."


def _interval(stats: Mapping[str, Any]) -> str:
    return f"{_fmt(stats['mean'])} [{_fmt(stats['low'])}, {_fmt(stats['high'])}]"


def _format_uncertainty(result: EvalSetResult, k: int) -> list[str]:
    """Each metric's mean with its bootstrap interval, then each run against the paired reference."""
    names = [*result.settings["metrics"], f"judged@{k}"]
    lines = [
        "## Statistical uncertainty",
        "",
        f"Mean over {result.runs[0].aggregate['queries']} questions with a 95% interval: "
        f"{result.settings.get('uncertainty', describe_method())}. With this few questions the intervals are wide; "
        "a difference whose interval includes zero is not evidence of a change.",
        "",
        _row(["Retriever", "Mode", *names]),
        _row(["---"] * (len(names) + 2)),
    ]
    for run in result.runs:
        if run.uncertainty:
            lines.append(_row([_run_label(run), run.mode, *(_interval(run.uncertainty[n]) for n in names)]))
    paired = [run for run in result.runs if run.paired]
    if paired:
        reference = result.settings.get("paired_reference")
        lines += [
            "",
            f"### Paired comparison with `{reference}`",
            "",
            f"Each run minus `{reference}` in the same mode, question by question: the mean difference with a 95% paired "
            "bootstrap interval, wins / ties / losses, and the p-value of a two-sided sign-flip test of no difference.",
            "",
            _row(["Retriever", "Mode", *names]),
            _row(["---"] * (len(names) + 2)),
        ]
        for run in paired:
            cells = []
            for n in names:
                d = run.paired["metrics"][n]
                cells.append(
                    f"{d['mean_difference']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]; "
                    f"{d['wins']}/{d['ties']}/{d['losses']}; p {d['p_value']:.3f}"
                )
            lines.append(_row([_run_label(run), run.mode, *cells]))
    lines.append("")
    return lines


def _format_failures(result: EvalSetResult, k: int) -> list[str]:
    """Relevant passages each run ranks below its top k or does not return, and questions with none in its top k."""
    top_k = result.settings["top_k"]
    outside = f"relevant_outside_top{k}"
    lines = [
        "## Relevant retrieval failures",
        "",
        RELEVANT_FAILURES.format(k=k, top_k=top_k),
        "",
        _row(["Retriever", "Mode", f"Questions without a relevant passage in the top {k}",
              f"Relevant passages outside the top {k}", f"Of them, not in the top {top_k}"]),
        _row(["---"] * 5),
    ]
    for run in result.runs:
        agg = run.aggregate
        lines.append(_row([
            _run_label(run), run.mode, ", ".join(agg.get(f"queries_without_relevant_in_top{k}", [])) or "—",
            agg.get(f"{outside}_total", "—"), agg.get("relevant_not_retrieved_total", "—"),
        ]))
    lines.append("")
    return lines


def _run_label(run: EvalSetRun) -> str:
    return run.run_name + (" (diagnostic)" if run.diagnostic else "") + (" (stored reference)" if run.reference else "")


def _arrow(before: float, after: float) -> str:
    return f"{_fmt(before)} → {_fmt(after)}"


def _format_comparison(result: EvalSetResult, k: int) -> list[str]:
    """The same rankings scored with the earlier set's labels and with this set's, side by side."""
    comparison = result.comparison or {}
    name, version = comparison.get("eval_set"), comparison.get("label_version")
    runs = comparison.get("runs", {})
    lines = [
        f"## Change from {name} (labels {version} → {result.label_report.get('label_version') if result.label_report else '?'})",
        "",
        "Each run below is the same ranking scored twice: with the earlier labels (as recorded by "
        f"{name}) and with this set's. Only the labels differ"
        + (
            "."
            if all(r["same_ranking"] for r in runs.values())
            else "; runs marked * were ranked again and are not the same files."
        ),
        "",
        _row(
            [
                "Retriever",
                "Mode",
                "nDCG@10",
                "MRR@10",
                "Recall@100",
                f"judged@{k}",
                f"Unjudged top {k}",
                f"Relevant top {k}",
                "Judged positives",
            ]
        ),
        _row(["---"] * 9),
    ]
    for run in result.runs:
        earlier = runs[run.run_id]
        before, after = earlier["aggregate"], run.aggregate
        lines.append(
            _row(
                [
                    run.run_name + ("" if earlier["same_ranking"] else " *"),
                    run.mode,
                    *(
                        _arrow(before[m], after[m])
                        for m in ("ndcg@10", "mrr@10", "recall@100", f"judged@{k}")
                    ),
                    f"{before[f'top{k}_unjudged_total']} → {after[f'top{k}_unjudged_total']}",
                    f"{before[f'top{k}_relevant_total']} → {after[f'top{k}_relevant_total']}",
                    f"{before['positives_total']} → {after['positives_total']}",
                ]
            )
        )
    lines += ["", "### nDCG@10 per question, earlier → this set", ""]
    lines.append(_row(["Question"] + [f"{r.run_name} {r.mode}" for r in result.runs]))
    lines.append(_row(["---"] * (len(result.runs) + 1)))
    for index, row in enumerate(result.runs[0].per_query if result.runs else []):
        cells = []
        for run in result.runs:
            before = runs[run.run_id]["per_query"][row["query_id"]]["metrics"]["ndcg@10"]
            cells.append(_arrow(before, run.per_query[index]["metrics"]["ndcg@10"]))
        lines.append(_row([row["query_id"], *cells]))
    lines.append("")
    return lines


def format_eval_set_markdown(result: EvalSetResult) -> str:
    """Human-readable report of a named evaluation set run, banner first."""
    eval_set = result.eval_set
    corpus = result.corpus
    settings = result.settings
    k = settings["judged_k"]
    label_report = result.label_report or {}
    labels = label_report.get("labels") or {}
    audit = label_report.get("audit") or {}
    modes = settings.get("retrieval_modes") or list(_DEFAULT_MODES)
    names = list(dict.fromkeys(r.run_name for r in result.runs))
    lines = [f"# {eval_set.name}: {', '.join(names)}", ""]
    if eval_set.status == STATUS_DEV:
        lines += [f"> **{DEV_BANNER}**", ""]
    lines += [
        f"- Corpus: `{corpus.corpus_version}`, {corpus.passages:,} passages, "
        f"SHA-256 `{corpus.corpus_sha256}`",
        f"- Labels: `{result.annotation_version}`: {labels.get('qrels', '?')} judged pairs, "
        f"by source {labels.get('by_source', {})}, open flags "
        f"{labels.get('open_flags_by_rule', {})}, audit {audit.get('independence', 'n/a')}",
        f"- Settings: top_k {settings['top_k']}, judged@{k}, coverage thresholds "
        f"{settings['coverage_thresholds']}, retrieval modes {', '.join(modes)}",
        "",
    ]
    if settings.get("runs_from"):
        lines += [f"Rankings: {STORED_RUNS_NOTE.format(runs_from=settings['runs_from'])}", ""]
    pool = result.pool
    lines += [
        f"Unjudged passages: {UNJUDGED_ASSUMPTION}",
        "",
        f"Candidate pools: {pool_disclosure(pool)}",
    ]
    if pool:
        lines.append(
            f"In this set, {_pct(pool['judged_proposed_by_bm25'], pool['judged'])} judged passages "
            f"and {_pct(pool['positives_proposed_by_bm25'], pool['positives'])} judged-relevant "
            f"passages were proposed by BM25 (pooling or the BM25 runs). Judged passages by source: "
            f"{pool['judged_by_source']}; relevant passages by source: "
            f"{pool['positives_by_source']}."
        )
    lines += ["", "Retrieval modes:", ""]
    lines += [f"- `{mode}`: {RETRIEVAL_MODES[mode]}" for mode in modes]
    lines += ["", f"> **{COMPARISON_CAUTION}**", ""]
    for caution in checkpoint_cautions(result.runs):
        lines += [f"> **{caution}**", ""]
    if settings.get("reference_runs_from"):
        lines += [
            f"Stored references: the runs marked (stored reference) are the rankings in "
            f"{settings['reference_runs_from']} (each run file's SHA-256 checked), scored with this set's labels; "
            "they were not run, re-encoded or re-ranked, and are not written again.",
            "",
        ]
    if result.skipped:
        lines += ["Not run:", ""]
        lines += [f"- `{s['name']}`: {s['reason']}" for s in result.skipped]
        lines.append("")

    lines += [
        "## Comparison",
        "",
        _row(
            [
                "Retriever",
                "Mode",
                "nDCG@10",
                "MRR@10",
                "Recall@100",
                f"judged@{k}",
                "nDCG@10 judged-only",
                "MRR@10 judged-only",
                f"Unjudged top {k} (in / out of scope)",
                f"Confirmed negatives top {k}",
                "Insufficient coverage",
            ]
        ),
        _row(["---"] * 11),
    ]
    scores = ("ndcg@10", "mrr@10", "recall@100", f"judged@{k}")
    sensitivity = ("ndcg@10_judged_only", "mrr@10_judged_only")
    for run in result.runs:
        agg = run.aggregate
        name = _run_label(run)
        lines.append(
            _row(
                [name, run.mode]
                + [_fmt(agg.get(metric, 0.0)) for metric in scores + sensitivity]
                + [
                    f"{agg[f'top{k}_unjudged_total']} ({agg[f'top{k}_unjudged_in_scope_total']} / "
                    f"{agg[f'top{k}_unjudged_out_of_scope_total']})",
                    agg[f"top{k}_confirmed_negatives_total"],
                    ", ".join(agg["queries_with_insufficient_coverage"]) or "—",
                ]
            )
        )
    lines += [
        "",
        f"Unjudged and confirmed-negative counts are summed over the top {k} of every question.",
        "",
    ]
    if any(run.uncertainty for run in result.runs):
        lines += _format_uncertainty(result, k)
    lines += _format_failures(result, k)
    if result.comparison:
        lines += _format_comparison(result, k)
    lines += [
        f"### judged@{k} per question",
        "",
        _row(["Question"] + [f"{r.run_name} {r.mode}" for r in result.runs]),
        _row(["---"] * (len(result.runs) + 1)),
    ]
    questions = result.runs[0].per_query if result.runs else []
    for index, row in enumerate(questions):
        lines.append(
            _row([row["query_id"]] + [_fmt(r.per_query[index][f"judged@{k}"]) for r in result.runs])
        )
    lines += ["", "## Retrievers", ""]
    described: set[str] = set()
    for run in result.runs:
        if run.run_name not in described:
            described.add(run.run_name)
            lines.append(f"- `{run.run_name}`: {_describe_retriever(run.retriever)}")
    lines.append("")

    for run in result.runs:
        agg = run.aggregate
        title = f"## {run.run_name}, {run.mode}" + (" (diagnostic)" if run.diagnostic else "")
        top = {key: agg[f"top{k}_{key}_total"] for key in ("relevant", "confirmed_negatives")}
        unjudged = {
            key: agg[f"top{k}_{key}_total"]
            for key in ("unjudged", "unjudged_in_scope", "unjudged_out_of_scope")
        }
        lines += [
            title,
            "",
            f"Top {k} over {agg['queries']} questions: {top['relevant']} judged relevant, "
            f"{top['confirmed_negatives']} confirmed negatives, {unjudged['unjudged']} unjudged "
            f"({unjudged['unjudged_in_scope']} from the company asked about, "
            f"{unjudged['unjudged_out_of_scope']} from other companies).",
            "",
            _row(
                [
                    "Question",
                    "Positives",
                    "nDCG@10",
                    "MRR@10",
                    "Recall@100",
                    f"judged@{k}",
                    f"Unjudged top {k} (in / out of scope)",
                    f"Confirmed negatives top {k}",
                    "First relevant rank",
                    f"Relevant outside top {k} (not returned)",
                    "Coverage flags",
                ]
            ),
            _row(["---"] * 11),
        ]
        for row in run.per_query:
            counts = row[f"top{k}"]
            m = row["metrics"]
            missed = row.get(f"relevant_outside_top{k}", [])
            lines.append(
                _row(
                    [
                        row["query_id"],
                        row["judgments"]["positives"],
                        _fmt(m.get("ndcg@10", 0.0)),
                        _fmt(m.get("mrr@10", 0.0)),
                        _fmt(m.get("recall@100", 0.0)),
                        _fmt(row[f"judged@{k}"]),
                        f"{counts['unjudged']} ({counts['unjudged_in_scope']} / "
                        f"{counts['unjudged_out_of_scope']})",
                        counts["confirmed_negatives"],
                        row["first_relevant_rank"] or "—",
                        f"{len(missed)} ({sum(1 for x in missed if x['rank'] is None)})",
                        ", ".join(row["coverage_flags"]) or "—",
                    ]
                )
            )
        insufficient = agg["queries_with_insufficient_coverage"]
        verdict = (
            ", ".join(insufficient) if insufficient else "none under the configured thresholds"
        )
        lines += ["", f"Questions with insufficient relevance coverage: {verdict}.", ""]
    if questions:
        lines += (
            ["## Questions", ""] + [f"- `{r['query_id']}` {r['text']}" for r in questions] + [""]
        )
    return "\n".join(lines)


__all__ = [
    "CHECKPOINT_CAUTION",
    "SELECTED_CHECKPOINT_CAUTION",
    "COMPARISON_CAUTION",
    "DEV_BANNER",
    "EXPANDED_POOL_DISCLOSURE",
    "METRIC_DEFINITIONS",
    "POOL_DISCLOSURE",
    "RELEVANT_FAILURES",
    "RETRIEVAL_MODES",
    "STATUS_DEV",
    "STATUS_FINAL",
    "STORED_RUNS_NOTE",
    "UNJUDGED_ASSUMPTION",
    "EvalSet",
    "EvalSetResult",
    "EvalSetRun",
    "build_retriever",
    "check_final_seeds",
    "checkpoint_cautions",
    "company_restrictions",
    "evaluate_eval_set_run",
    "format_eval_set_markdown",
    "format_results_markdown",
    "load_benchmark_inputs",
    "load_eval_set",
    "load_metrics_from_config",
    "load_retrieval_modes",
    "load_top_k_from_config",
    "pool_composition",
    "pool_disclosure",
    "query_tickers",
    "results_to_jsonable",
    "run_bm25_baseline",
    "run_eval_set",
    "run_retriever_benchmark",
    "validate_eval_inputs",
    "write_eval_set_outputs",
    "write_results",
]
