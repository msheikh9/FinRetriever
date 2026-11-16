"""Stored retrieval runs: ``<output_dir>/<mode>/<run>/run.jsonl`` read back with its provenance checked.

A named evaluation set writes one directory per (retrieval mode, retriever) with
``metrics.json`` and ``run.jsonl`` (see ``finretriever.evaluation.benchmark``).
Later steps, such as deepening the judged pools or scoring the same rankings
against newer labels, must use those rankings exactly as they were produced:
retrievers are not run again, embeddings are not recomputed and nothing is
re-ranked. ``load_stored_run`` therefore refuses a run unless

* ``metrics.json`` names the same run and mode as its directory,
* it was produced on the corpus being used (SHA-256 and version),
* ``run.jsonl`` still has the SHA-256 recorded in ``metrics.json``,
* every passage is in the corpus with the ticker the run recorded, and
* every query's ranks are 1, 2, 3, ... with no passage listed twice.

The grade and judged columns of ``run.jsonl`` describe the labels the run was
first scored with; they are ignored here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from finretriever.data.schemas import Passage, SearchHit
from finretriever.evaluation.provenance import CorpusIdentity

RUN_FILE = "run.jsonl"
METRICS_FILE = "metrics.json"


class StoredRunError(ValueError):
    """A stored run is missing, was modified, or does not belong to the corpus."""


@dataclass(frozen=True)
class StoredRun:
    """One stored ranking (a retriever in a retrieval mode) and the record it was written with."""

    run: str
    mode: str
    directory: Path
    run_sha256: str
    metrics_sha256: str
    metrics: dict[str, Any]
    hits: tuple[SearchHit, ...]

    @property
    def run_id(self) -> str:
        return f"{self.mode}/{self.run}"

    @property
    def top_k(self) -> int:
        return int(self.metrics["settings"]["top_k"])

    def provenance(self) -> dict[str, Any]:
        """Where the ranking came from: files, hashes and the labels it was first scored with."""
        return {
            "run": self.run,
            "mode": self.mode,
            "run_file": str(self.directory / RUN_FILE),
            "run_sha256": self.run_sha256,
            "metrics_file": str(self.directory / METRICS_FILE),
            "metrics_sha256": self.metrics_sha256,
            "eval_set": self.metrics.get("eval_set"),
            "label_version": self.metrics.get("label_version"),
            "top_k": self.top_k,
        }

    def ranked(self, k: int | None = None) -> dict[str, list[SearchHit]]:
        """Each query's hits in rank order, cut at ``k`` when given."""
        out: dict[str, list[SearchHit]] = {}
        for hit in self.hits:
            out.setdefault(hit.query_id, []).append(hit)
        return {qid: hits if k is None else hits[:k] for qid, hits in out.items()}


def load_stored_run(
    runs_dir: str | Path,
    mode: str,
    run: str,
    *,
    corpus: CorpusIdentity,
    passages: Mapping[str, Passage],
) -> StoredRun:
    """The run in ``runs_dir/<mode>/<run>/``, after the checks in the module docstring."""
    directory = Path(runs_dir) / mode / run
    metrics_path, run_path = directory / METRICS_FILE, directory / RUN_FILE
    for path in (metrics_path, run_path):
        if not path.is_file():
            raise StoredRunError(f"stored run {mode}/{run}: {path} is missing")
    metrics_bytes = metrics_path.read_bytes()
    try:
        metrics = json.loads(metrics_bytes)
        recorded_sha256 = metrics["run_file"]["sha256"]
        recorded_corpus = metrics["corpus"]
        top_k = int(metrics["settings"]["top_k"])
    except (ValueError, KeyError, TypeError) as exc:
        raise StoredRunError(f"{metrics_path} is not a run record ({exc!r})") from None
    if (metrics.get("run"), metrics.get("mode")) != (run, mode):
        raise StoredRunError(
            f"{metrics_path} records run {metrics.get('run')!r} in mode {metrics.get('mode')!r}, "
            f"not {run!r} in {mode!r}"
        )
    if (recorded_corpus.get("sha256"), recorded_corpus.get("version")) != (
        corpus.corpus_sha256,
        corpus.corpus_version,
    ):
        raise StoredRunError(
            f"stored run {mode}/{run} was produced on corpus {recorded_corpus.get('version')} "
            f"(sha256 {str(recorded_corpus.get('sha256'))[:12]}...), not {corpus.corpus_version} "
            f"(sha256 {corpus.corpus_sha256[:12]}...)"
        )
    run_bytes = run_path.read_bytes()
    run_sha256 = hashlib.sha256(run_bytes).hexdigest()
    if run_sha256 != recorded_sha256:
        raise StoredRunError(
            f"{run_path} does not match the SHA-256 recorded in {metrics_path}; it was modified"
        )

    hits: list[SearchHit] = []
    seen: dict[str, set[str]] = {}
    for line_no, line in enumerate(run_bytes.decode("utf-8").splitlines(), start=1):
        where = f"{run_path}:{line_no}"
        try:
            row = json.loads(line)
            hit = SearchHit(
                query_id=row["query_id"],
                passage_id=row["passage_id"],
                rank=row["rank"],
                score=row["score"],
                retriever=run,
                metadata={"ticker": row["ticker"]},
            )
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise StoredRunError(f"{where}: not a ranked hit ({exc})") from None
        passage = passages.get(hit.passage_id)
        if passage is None:
            raise StoredRunError(f"{where}: passage {hit.passage_id} is not in the corpus")
        if passage.ticker != hit.metadata["ticker"]:
            raise StoredRunError(
                f"{where}: {hit.passage_id} is recorded as {hit.metadata['ticker']} but the corpus "
                f"says {passage.ticker}"
            )
        ranked = seen.setdefault(hit.query_id, set())
        if hit.rank != len(ranked) + 1:
            raise StoredRunError(
                f"{where}: {hit.query_id} rank {hit.rank} follows rank {len(ranked)}; ranks must "
                "run 1, 2, 3, ..."
            )
        if hit.passage_id in ranked:
            raise StoredRunError(f"{where}: {hit.passage_id} is ranked twice for {hit.query_id}")
        if hit.rank > top_k:
            raise StoredRunError(f"{where}: rank {hit.rank} is beyond the recorded top_k {top_k}")
        ranked.add(hit.passage_id)
        hits.append(hit)
    return StoredRun(
        run=run,
        mode=mode,
        directory=directory,
        run_sha256=run_sha256,
        metrics_sha256=hashlib.sha256(metrics_bytes).hexdigest(),
        metrics=metrics,
        hits=tuple(hits),
    )


__all__ = ["METRICS_FILE", "RUN_FILE", "StoredRun", "StoredRunError", "load_stored_run"]
