"""Gold set v1: human-written reworded searches for candidate pooling (docs/gold_set_v1_protocol.md §2.2).

The protocol pools with BM25 on each question and on reformulations in filing vocabulary:
- at least two for semantic, multi-passage and cross-company questions (``rewordings_required``);
- one for every other question.

Reformulations are authored text, so a person writes them; no AI generates, suggests or edits their wording.

1. ``write_rewording_sheet`` writes one row per approved question, from the latest frozen question export. Each row
   holds the question and its context (slot, companies, topic, type, comparison, period) and the required number of
   rewordings. The rewording fields, the writer and the AI-assistance attestation are left empty.
2. ``validate_rewording_sheet`` checks the filled sheet and refuses it whole if any of these fail:
   - every approved question is present exactly once, with its context columns unchanged;
   - each question has at least its required number of rewordings, filled in order;
   - each rewording differs from its question and from the question's other rewordings;
   - the writer is a person, not a configured AI identity, and ``ai_assistance`` is exactly ``none``;
   - no rewording reaches the leakage threshold against the training and validation questions (word-set Jaccard,
     as for the questions).

   Observations, such as a rewording naming a company outside its slot, are reported, never blocking.
3. ``apply_rewordings`` runs BM25 (the pool's depth, the question's companies) on every rewording and adds the
   results to the existing pools.
   - **Deduplicated:** a passage already in the pool keeps its row and position and gains this source.
   - **Recorded:** each source's search detail records the rewording and its writer.
   - **Kept:** nothing is deleted.
4. ``write_pool_snapshot`` freezes the pools of the approved questions as a new, read-only snapshot version. It first
   verifies that every earlier snapshot is unchanged, then records its parent, its inputs' hashes and the
   per-question counts by source.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from finretriever.data.schemas import Query
from finretriever.eval_set.annotation_store import AnnotationError, AnnotationStore, utc_now
from finretriever.eval_set.candidate_generator import CandidateGenerator, Scope
from finretriever.eval_set.qrels_builder import find_gold_overlap
from finretriever.eval_set.schemas import normalize_question

COMPANY_NAMES = {"AAPL": "Apple", "AMZN": "Amazon", "JPM": "JPMorgan Chase", "PFE": "Pfizer", "TSLA": "Tesla",
                 "WMT": "Walmart", "XOM": "ExxonMobil"}
CONTEXT_COLUMNS = ("query_id", "slot", "tickers", "companies", "topic", "question_type", "comparison", "period",
                   "fiscal_years", "question", "rewordings_required")
REWORDING_FIELDS = ("rewording_1", "rewording_2", "rewording_3")
REWORDING_COLUMNS = (*CONTEXT_COLUMNS, *REWORDING_FIELDS, "writer", "ai_assistance", "notes")
NO_ASSISTANCE = "none"


def rewordings_required(question_type: str, comparison: str) -> int:
    """The protocol's minimum: two for semantic, multi-passage and cross-company questions, one otherwise."""
    return 2 if question_type in ("semantic", "multi_passage") or comparison == "cross_company" else 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_rows(export_rows: Sequence[Mapping[str, Any]], plan_periods: Mapping[str, str]) -> list[dict[str, str]]:
    """The sheet's context columns for each approved question (from the frozen export and the plan's periods)."""
    rows = []
    for r in sorted(export_rows, key=lambda r: r["query_id"]):
        rows.append({
            "query_id": r["query_id"], "slot": r["slot"], "tickers": ",".join(r["tickers"]),
            "companies": "; ".join(COMPANY_NAMES.get(t, t) for t in r["tickers"]), "topic": r["topic"],
            "question_type": r["question_type"], "comparison": r["comparison"], "period": plan_periods[r["slot"]],
            "fiscal_years": ";".join(map(str, r["fiscal_years"])), "question": r["text"],
            "rewordings_required": str(rewordings_required(r["question_type"], r["comparison"])),
        })
    return rows


def write_rewording_sheet(rows: Sequence[Mapping[str, str]], path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise AnnotationError(f"{path} exists; a rewording sheet is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REWORDING_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{f: "" for f in REWORDING_FIELDS}, "writer": "", "ai_assistance": "", "notes": ""})
    path.chmod(0o444)  # issued read-only: the writer fills a copy, and the issued sheet's hash stays verifiable
    return path


def validate_rewording_sheet(sheet: str | Path, expected: Sequence[Mapping[str, str]], *, ai_identities: Iterable[str],
                             training_texts: Sequence[str], threshold: float = 0.8) -> dict[str, Any]:
    """Validate the filled sheet against the issued context rows; returns ``{query_id: {...}}`` per question plus
    observations. Raises ``AnnotationError`` listing every problem; then nothing may be applied."""
    raw = Path(sheet).read_bytes()
    try:  # a spreadsheet's "CSV UTF-8" may add a byte-order mark and CRLF line ends; the content checks still apply
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AnnotationError(f"{sheet}: not UTF-8 ({exc}); save it as CSV UTF-8") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if list(reader.fieldnames or []) != list(REWORDING_COLUMNS):
        raise AnnotationError(f"{sheet}: the header is not {list(REWORDING_COLUMNS)}")
    rows = list(reader)
    issued = {r["query_id"]: r for r in expected}
    ai = {name.casefold() for name in ai_identities}
    problems: list[str] = []
    observations: list[str] = []
    ids = [r["query_id"] for r in rows]
    if sorted(ids) != sorted(issued) or len(ids) != len(set(ids)):
        problems.append(f"the sheet lists {len(ids)} rows; it must list each of the {len(issued)} approved questions "
                        f"once (missing {sorted(set(issued) - set(ids))[:5]}, extra {sorted(set(ids) - set(issued))[:5]})")
    accepted: dict[str, dict[str, Any]] = {}
    for line, row in enumerate(rows, start=2):
        qid = row["query_id"]
        where = f"{sheet}:{line} ({qid})"
        context = issued.get(qid)
        if context is None:
            continue
        if changed := [c for c in CONTEXT_COLUMNS if row.get(c) != context[c]]:
            problems.append(f"{where}: context columns changed: {changed}")
        texts = [(row.get(f) or "").strip() for f in REWORDING_FIELDS]
        filled = [t for t in texts if t]
        required = int(context["rewordings_required"])
        if texts[:len(filled)] != filled:
            problems.append(f"{where}: rewordings must be filled in order, from rewording_1")
        if len(filled) < required:
            problems.append(f"{where}: {len(filled)} rewording(s); the protocol requires {required}")
        keys = [normalize_question(t) for t in filled]
        if normalize_question(context["question"]) in keys:
            problems.append(f"{where}: a rewording repeats the question itself")
        if len(set(keys)) != len(keys):
            problems.append(f"{where}: two rewordings are the same")
        writer = (row.get("writer") or "").strip()
        if not writer:
            problems.append(f"{where}: no writer")
        elif writer.casefold() in ai:
            problems.append(f"{where}: {writer!r} is an AI identity; rewordings are written by people")
        if (row.get("ai_assistance") or "").strip() != NO_ASSISTANCE:
            problems.append(f"{where}: ai_assistance must be exactly {NO_ASSISTANCE!r}")
        slot_tickers = set(context["tickers"].split(","))
        for text in filled:
            named = {t for t, name in COMPANY_NAMES.items()
                     if t not in slot_tickers and (name.casefold() in text.casefold() or t in text.split())}
            if named:
                observations.append(f"{qid}: a rewording names {sorted(named)}, outside its slot "
                                    f"(the search stays restricted to {sorted(slot_tickers)})")
        accepted[qid] = {"rewordings": filled, "writer": writer, "notes": (row.get("notes") or "").strip()}
    # Leakage: every rewording against every training and validation question.
    queries = [Query(query_id=f"{qid}#{i}", text=t, source="gold")
               for qid, v in accepted.items() for i, t in enumerate(v["rewordings"], start=1)]
    best: dict[str, float] = {}
    for overlap in find_gold_overlap(queries, list(training_texts), threshold=0.0):
        best[overlap.query_id] = max(best.get(overlap.query_id, 0.0), overlap.similarity)
    if leaking := sorted(q for q, s in best.items() if s >= threshold):
        problems.append(f"rewordings at or above the leakage threshold {threshold}: {leaking}")
    if problems:
        raise AnnotationError(f"{len(problems)} problem(s), nothing applied: " + "; ".join(problems))
    return {"questions": accepted, "observations": observations,
            "leakage": {"threshold": threshold, "against": len(training_texts), "max": max(best.values(), default=0.0),
                        "per_rewording": dict(sorted(best.items()))}}


def apply_rewordings(store: AnnotationStore, accepted: Mapping[str, Mapping[str, Any]], *, depth: int, added_by: str,
                     sheet_label: str, dry_run: bool = False) -> dict[str, dict[str, int]]:
    """BM25 on every rewording, added to the existing pools (see the module docstring). With ``dry_run`` nothing is
    written; the counts are what would be added."""
    generator = CandidateGenerator(store.passages)
    counts: dict[str, dict[str, int]] = {}
    for qid, entry in sorted(accepted.items()):
        question = store.question(qid)
        scope = Scope.of(question)
        pooled = {e.passage_id for e in store.pool(qid)}
        proposed = []
        for text in entry["rewordings"]:
            for c in generator.bm25(text, depth=depth, scope=scope):
                detail = f"{c.source.detail}; reformulation by {entry['writer']} ({sheet_label})"
                proposed.append(c.model_copy(update={"source": c.source.model_copy(update={"detail": detail})}))
        new_ids = {c.passage_id for c in proposed} - pooled
        counts[qid] = {"searches": len(entry["rewordings"]), "proposed": len(proposed), "new": len(new_ids),
                       "already_pooled": len({c.passage_id for c in proposed} & pooled)}
        if not dry_run:
            store.add_candidates(qid, proposed, added_by=added_by)
    return counts


def _snapshot_paths(directory: Path, version: int) -> tuple[Path, Path]:
    folder = directory / f"pool_v{version}"
    return folder / "pool_snapshot.jsonl", folder / "pool_manifest.json"


def pool_snapshot_versions(directory: str | Path) -> list[int]:
    return sorted(int(p.name[len("pool_v"):]) for p in Path(directory).glob("pool_v*")
                  if p.name[len("pool_v"):].isdigit() and (p / "pool_manifest.json").is_file())


def verify_pool_snapshot(directory: str | Path, version: int) -> dict[str, Any]:
    snapshot, manifest_path = _snapshot_paths(Path(directory), version)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not snapshot.is_file() or _sha256(snapshot) != (manifest.get("snapshot") or {}).get("sha256"):
        raise AnnotationError(f"pool snapshot version {version}: {snapshot} is missing or differs from its manifest")
    return manifest


def _workspace_pool_lines(store: AnnotationStore, wanted: set[str]) -> list[str]:
    lines = (store.directory / "pool.jsonl").read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip() and json.loads(line)["query_id"] in wanted]


def check_rewording_base(store: AnnotationStore, directory: str | Path, query_ids: Iterable[str]) -> int:
    """Refuse reworded searches unless they extend an unchanged, frozen base pool exactly once. Returns the parent
    snapshot version.

    - A pool snapshot exists, and every snapshot still matches its manifest.
    - No snapshot records a rewording sheet and no pooled source is a reformulation: the protocol has one round.
    - The workspace pools of these questions are byte-identical to the latest snapshot, so the next snapshot is that
      parent plus the reworded searches' candidates and nothing else.
    """
    directory = Path(directory)
    versions = pool_snapshot_versions(directory)
    if not versions:
        raise AnnotationError("no pool snapshot: freeze the base pool before adding reworded searches")
    manifests = {v: verify_pool_snapshot(directory, v) for v in versions}
    if applied := [v for v, m in manifests.items() if "rewording_sheet" in (m.get("inputs") or {})]:
        raise AnnotationError(f"reworded searches were already applied (pool snapshot version {applied[0]}); "
                              "the protocol has one round")
    wanted = set(query_ids)
    lines = _workspace_pool_lines(store, wanted)
    if any("reformulation by" in s["detail"] for line in lines for s in json.loads(line)["sources"]):
        raise AnnotationError("the workspace pools already hold reformulation sources")
    latest = versions[-1]
    snapshot = _snapshot_paths(directory, latest)[0].read_text(encoding="utf-8").splitlines()
    if lines != snapshot:
        raise AnnotationError(f"the workspace pools of these questions differ from pool snapshot version {latest}; "
                              "reworded searches extend an unchanged frozen pool")
    return latest


def verify_approved_pool(store: AnnotationStore, directory: str | Path, query_ids: Iterable[str], *, version: int,
                         sha256: str) -> dict[str, Any]:
    """The grading gate: the reviewer-approved pool snapshot is intact and is what the annotators grade.

    - Every pool snapshot still matches its manifest.
    - The approved version exists, has the approved SHA-256, and is the latest version.
    - The workspace pools of ``query_ids`` are byte-identical to it: nothing was added, removed or changed since.

    Returns the approved snapshot's manifest; raises ``AnnotationError`` otherwise.
    """
    directory = Path(directory)
    versions = pool_snapshot_versions(directory)
    manifests = {v: verify_pool_snapshot(directory, v) for v in versions}
    if version not in manifests:
        raise AnnotationError(f"the approved pool snapshot version {version} does not exist")
    if manifests[version]["snapshot"]["sha256"] != sha256:
        raise AnnotationError(f"pool snapshot version {version} is not the approved one (SHA-256 "
                              f"{manifests[version]['snapshot']['sha256']}, approved {sha256})")
    if versions[-1] != version:
        raise AnnotationError(f"pool snapshot version {versions[-1]} is later than the approved version {version}")
    snapshot = _snapshot_paths(directory, version)[0].read_text(encoding="utf-8").splitlines()
    if _workspace_pool_lines(store, set(query_ids)) != snapshot:
        raise AnnotationError(f"the workspace pools differ from the approved pool snapshot version {version}; "
                              "grading stops until they match")
    return manifests[version]


def write_pool_snapshot(store: AnnotationStore, directory: str | Path, query_ids: Iterable[str], *,
                        created_by: str, inputs: Mapping[str, Any], notice: str) -> dict[str, Any]:
    """Freeze the pools of ``query_ids`` as the next read-only snapshot version (see the module docstring)."""
    directory = Path(directory)
    versions = pool_snapshot_versions(directory)
    parents = {v: verify_pool_snapshot(directory, v) for v in versions}
    version = (versions[-1] + 1) if versions else 1
    snapshot, manifest_path = _snapshot_paths(directory, version)
    if snapshot.parent.exists():
        raise AnnotationError(f"{snapshot.parent} exists; a pool snapshot is written once")
    ids = sorted(set(query_ids))
    wanted = set(ids)
    lines = _workspace_pool_lines(store, wanted)
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    per_question = {}
    for qid in ids:
        entries = store.pool(qid)
        per_question[qid] = {"candidates": len(entries),
                             "sources": dict(Counter(s.method for e in entries for s in e.sources)),
                             "reformulation_sources": sum(1 for e in entries for s in e.sources
                                                          if "reformulation by" in s.detail),
                             "found_by_more_than_one_source": sum(1 for e in entries if len(e.sources) > 1)}
    parent = None if not versions else {"version": versions[-1], "snapshot_sha256": parents[versions[-1]]["snapshot"]["sha256"],
                                        "manifest_sha256": _sha256(_snapshot_paths(directory, versions[-1])[1])}
    manifest = {
        "kind": "finretriever_gold_pool_snapshot",
        "version": version,
        "created_at": utc_now(),
        "created_by": created_by,
        "parent": parent,
        "snapshot": {"file": snapshot.name, "sha256": _sha256(snapshot), "rows": len(lines)},
        "inputs": dict(inputs),
        "per_question_counts": per_question,
        "notice": notice,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for path in (snapshot, manifest_path):
        path.chmod(0o444)
    return manifest


__all__ = [
    "COMPANY_NAMES",
    "CONTEXT_COLUMNS",
    "REWORDING_COLUMNS",
    "apply_rewordings",
    "check_rewording_base",
    "context_rows",
    "pool_snapshot_versions",
    "rewordings_required",
    "validate_rewording_sheet",
    "verify_approved_pool",
    "verify_pool_snapshot",
    "write_pool_snapshot",
    "write_rewording_sheet",
]
