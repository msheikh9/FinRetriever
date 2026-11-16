"""Export a development label set as queries, qrels and provenance for the benchmark.

``export-dev`` writes, next to the label set by default:

* ``dev_queries.jsonl`` — one ``Query`` per question, ``source`` ``"dev_ai"``;
* ``dev_qrels.jsonl`` — one ``Qrel`` per labelled passage (grades 0, 1, 2);
  pool entries without a label and every unpooled passage are unjudged;
* ``dev_qrels.provenance.jsonl`` — for each qrel, in the same order: where the
  label came from (AI proposal or human judgment), whether it is AI-generated,
  the assistance behind it, whether a rule changed it, its open flags, and for
  a set derived from another label set the version that introduced it;
* ``dev_eval.metadata.json`` — ``EvaluationMetadata`` (corpus SHA-256 and
  version, label version), read by ``verify_evaluation_corpus``;
* ``dev_eval.report.json`` — status, label provenance, audit independence,
  per-question coverage, exceptions and the SHA-256 of the files above.

Nothing here is gold: the export refuses the gold export paths, the human
workspace and the immutable inputs it was derived from. The output has no
timestamps, so exporting the same label set twice gives identical files.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finretriever.data.schemas import Passage, Qrel, Query
from finretriever.eval_set.dev_labels import DEV_NOTICE, DEV_STATUS, DevLabelSet
from finretriever.eval_set.qrels_builder import (
    METADATA_FILENAME,
    QRELS_FILENAME,
    QUERIES_FILENAME,
    REPORT_FILENAME,
)
from finretriever.eval_set.schemas import RELEVANCE_GRADES
from finretriever.evaluation.provenance import EvaluationMetadata

DEV_QUERY_SOURCE = "dev_ai"
DEV_QUERIES_FILENAME = "dev_queries.jsonl"
DEV_QRELS_FILENAME = "dev_qrels.jsonl"
DEV_PROVENANCE_FILENAME = "dev_qrels.provenance.jsonl"
DEV_METADATA_FILENAME = "dev_eval.metadata.json"
DEV_REPORT_FILENAME = "dev_eval.report.json"
GOLD_FILENAMES = frozenset({QUERIES_FILENAME, QRELS_FILENAME, METADATA_FILENAME, REPORT_FILENAME})

UNJUDGED_POLICY = (
    "Only labelled passages are qrels. Pool entries without a label and every passage outside the pool are "
    "unjudged: they are absent from the qrels, which is not grade 0. Retrieval metrics count an unjudged "
    "passage as non-relevant; that is an assumption, not a judgment, so report judged@k with every metric."
)


class DevExportError(ValueError):
    """The labels cannot be exported, or the export would touch a protected path."""


@dataclass(frozen=True)
class DevExport:
    metadata: EvaluationMetadata
    queries: tuple[Query, ...]
    qrels: tuple[Qrel, ...]
    provenance: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _line(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def build_dev_export(label_set: DevLabelSet, passages: Mapping[str, Passage]) -> DevExport:
    """Queries, qrels, per-qrel provenance and the report of ``label_set``; validates everything first."""
    info = label_set.info
    questions = sorted(label_set.questions, key=lambda q: q.query_id)
    labels = sorted(label_set.labels, key=lambda x: (x.query_id, x.passage_id))
    pairs = Counter((x.query_id, x.passage_id) for x in labels)
    if conflicts := sorted(pair for pair, n in pairs.items() if n > 1):
        raise DevExportError(f"conflicting labels: more than one label for {conflicts[:5]}")
    known = {q.query_id for q in questions}
    if unknown_q := sorted({x.query_id for x in labels} - known):
        raise DevExportError(f"labels for unknown questions: {unknown_q}")
    if unknown_p := sorted({x.passage_id for x in labels} - passages.keys()):
        raise DevExportError(f"labels for passages not in the corpus: {unknown_p[:5]}")
    if bad := sorted({x.grade for x in labels} - RELEVANCE_GRADES.keys()):
        raise DevExportError(f"grades outside {sorted(RELEVANCE_GRADES)}: {bad}")
    if unlabelled := sorted(known - {x.query_id for x in labels}):
        raise DevExportError(f"questions without any label: {unlabelled}")

    queries = tuple(
        Query(
            query_id=q.query_id,
            text=q.text,
            source=DEV_QUERY_SOURCE,
            metadata={
                "status": DEV_STATUS,
                "label_version": info.label_version,
                "annotation_version": info.evaluation.annotation_version,
                "topic": q.topic,
                "question_type": q.question_type,
                "comparison": q.comparison,
                "tickers": ",".join(q.tickers),
                "fiscal_years": ",".join(map(str, q.fiscal_years)),
                "authoring": q.authoring,
                "ai_assisted_question": q.ai_assisted,
                "reviewed_by": q.reviewed_by,
            },
        )
        for q in questions
    )
    qrels = tuple(Qrel(query_id=x.query_id, passage_id=x.passage_id, relevance=x.grade) for x in labels)
    provenance = tuple(
        {
            "query_id": x.query_id,
            "passage_id": x.passage_id,
            "relevance": x.grade,
            "label_source": x.label_source,
            "ai_generated": x.ai_generated,
            "assistance": x.assistance,
            "labelled_by": x.labelled_by,
            "base_grade": x.base_grade,
            "changed": x.changed,
            "open_flags": list(x.flags),
            "audit_item": None if x.audit is None else x.audit.item,
            **({"introduced_in": x.introduced_in} if x.introduced_in else {}),
        }
        for x in labels
    )
    introduced = Counter(x.introduced_in for x in labels if x.introduced_in)

    per_question = []
    for q in questions:
        rows = [x for x in labels if x.query_id == q.query_id]
        counts = info.counts["questions"][q.query_id]
        positives = sum(1 for x in rows if x.grade > 0)
        exceptions = []
        if positives == 0:
            exceptions.append("no_positive")
        if not any(x.grade == 2 for x in rows):
            exceptions.append("no_grade_2")
        per_question.append(
            {
                "query_id": q.query_id,
                "question_type": q.question_type,
                "tickers": list(q.tickers),
                "fiscal_years": list(q.fiscal_years),
                "pool": counts["pool"],
                "labelled": len(rows),
                "unlabelled_pool_entries": counts["unlabelled"],
                "grades": {str(g): sum(1 for x in rows if x.grade == g) for g in RELEVANCE_GRADES},
                "positives": positives,
                "ai_generated": sum(1 for x in rows if x.ai_generated),
                "human_judgment": sum(1 for x in rows if not x.ai_generated),
                "changed": sum(1 for x in rows if x.changed),
                "open_flags": sum(len(x.flags) for x in rows),
                "exceptions": exceptions,
                **({"introduced_in": dict(sorted(Counter(x.introduced_in for x in rows).items()))} if introduced else {}),
            }
        )
    grades = Counter(x.grade for x in labels)
    report = {
        "status": DEV_STATUS,
        "notice": DEV_NOTICE,
        "label_version": info.label_version,
        "description": info.description,
        "evaluation": info.evaluation.model_dump(),
        "label_set_files": dict(info.files),
        "derived_from": dict(info.derived_from),
        "rubric": info.rubric.model_dump(),
        "rules_applied": list(info.rules_applied),
        "audit": {
            "independence": info.audit_independence,
            "note": info.audit_independence_note,
            "dispositions": [d.model_dump() for d in info.audit_dispositions],
        },
        "labels": {
            "qrels": len(qrels),
            "grades": {str(g): grades.get(g, 0) for g in RELEVANCE_GRADES},
            "by_source": dict(sorted(Counter(x.label_source for x in labels).items())),
            "ai_generated": sum(1 for x in labels if x.ai_generated),
            "assistance": dict(sorted(Counter(x.assistance for x in labels).items())),
            "changed_by_rule": dict(info.counts["changes"]),
            "open_flags_by_rule": dict(info.counts["flags"]),
            **({"by_introduced_in": dict(sorted(introduced.items()))} if introduced else {}),
        },
        "queries": per_question,
        "exceptions": {
            "no_positive": [row["query_id"] for row in per_question if "no_positive" in row["exceptions"]],
            "no_grade_2": [row["query_id"] for row in per_question if "no_grade_2" in row["exceptions"]],
        },
        "unjudged_policy": UNJUDGED_POLICY,
    }
    return DevExport(info.evaluation, queries, qrels, provenance, report)


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve()


def write_dev_export(
    export: DevExport,
    out_dir: str | Path,
    *,
    gold_export_dir: str | Path,
    protected_dirs: Iterable[str | Path] = (),
) -> dict[str, Path]:
    """Write the export into ``out_dir``.

    Refuses the gold export directory itself (where the gold files live) and any of
    ``protected_dirs`` or a directory inside one (the human workspace, the base proposals,
    the audit). Subdirectories of the gold export directory, such as ``data/eval/dev_ai/``,
    are allowed.
    """
    out = Path(out_dir)
    target = _resolved(out)
    if target == _resolved(gold_export_dir):
        raise DevExportError(f"refusing to write development labels into the gold export directory {gold_export_dir}")
    for item in protected_dirs:
        guard = _resolved(item)
        if target == guard or guard in target.parents:
            raise DevExportError(f"refusing to write development labels into protected path {item}")
    paths = {
        "queries": out / DEV_QUERIES_FILENAME,
        "qrels": out / DEV_QRELS_FILENAME,
        "provenance": out / DEV_PROVENANCE_FILENAME,
        "metadata": out / DEV_METADATA_FILENAME,
        "report": out / DEV_REPORT_FILENAME,
    }
    data = {
        "queries": "".join(_line(q.model_dump(mode="json")) for q in export.queries).encode("utf-8"),
        "qrels": "".join(_line(r.model_dump(mode="json")) for r in export.qrels).encode("utf-8"),
        "provenance": "".join(_line(row) for row in export.provenance).encode("utf-8"),
        "metadata": (json.dumps(export.metadata.model_dump(), indent=2) + "\n").encode("utf-8"),
    }
    report = {
        **export.report,
        "files": {paths[key].name: hashlib.sha256(blob).hexdigest() for key, blob in data.items()},
    }
    data["report"] = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    out.mkdir(parents=True, exist_ok=True)
    for key, blob in data.items():
        paths[key].write_bytes(blob)
    return paths


__all__ = [
    "DEV_METADATA_FILENAME",
    "DEV_PROVENANCE_FILENAME",
    "DEV_QRELS_FILENAME",
    "DEV_QUERIES_FILENAME",
    "DEV_QUERY_SOURCE",
    "DEV_REPORT_FILENAME",
    "GOLD_FILENAMES",
    "UNJUDGED_POLICY",
    "DevExport",
    "DevExportError",
    "build_dev_export",
    "write_dev_export",
]
