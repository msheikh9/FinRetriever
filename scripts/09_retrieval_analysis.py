#!/usr/bin/env python
"""Retrieval Performance Analysis v1 (docs/retrieval_analysis_v1.md).

Seven systems (BM25, zero-shot BGE, zero-shot E5, A0 and the three A1 seeds), in two retrieval modes (global:
the whole corpus; company: only the question's companies), on:

1. the 93 synthetic validation questions (validation_pairs_v4), scored exactly as the trainer's checkpoint
   selection scored them (training.validation_eval: relevant = positive + exact equivalents; uncertain equivalents
   removed from the ranking). These questions chose each A1 checkpoint: never independent evidence;
2. the 10 AI-labelled development questions (dev_pilot_v3, provisional labels): every system's stored ranking
   (the zero-shot references, and the A0/A1 rankings of ``scripts/04_run_baselines.py --config
   configs/eval/retrieval_analysis_v1.yaml --eval-set dev_pilot_v3_a1``, which must be run first) scored with the
   harness's metrics and checked against the harness's records;
3. the 100 frozen human-written questions (gold set v1 export v2): a metadata alignment diagnostic only
   (evaluation.alignment). No relevance label is read and no relevance metric is computed for them.

    python scripts/09_retrieval_analysis.py [--config ...] [--output-dir ...] [--systems a,b]

Reads only frozen inputs (hashes checked), loads checkpoints through the benchmark's inspection with their selection
records verified, and reuses cached passage embeddings. Writes only under --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finretriever.data.schemas import Qrel, SearchHit  # noqa: E402
from finretriever.evaluation.alignment import (  # noqa: E402
    HitInfo,
    Scope,
    answer_figures,
    figure_in_text,
    fiscal_year,
    question_alignment,
)
from finretriever.evaluation.benchmark import build_retriever  # noqa: E402
from finretriever.evaluation.exclusion import (  # noqa: E402
    A1_TRAINING_AND_VALIDATION,
    exclusion_ids,
)
from finretriever.evaluation.metrics import _recall_at_k, compute_per_query_metrics  # noqa: E402
from finretriever.evaluation.uncertainty import mean_interval, paired_difference  # noqa: E402
from finretriever.training.trainer_setup import load_trainer_config, prepare_inputs  # noqa: E402
from finretriever.training.validation_eval import question_metrics  # noqa: E402
from finretriever.utils.io import read_yaml  # noqa: E402

SYSTEMS = ["bm25", "bge_zero_shot", "e5_zero_shot", "a0_step0", "a1_seed42", "a1_seed1337", "a1_seed2026"]
A1 = ["a1_seed42", "a1_seed1337", "a1_seed2026"]
MODES = ["global", "company"]
TRAIN_CONFIG = "configs/train/biencoder_bge_base_ft_v3_a1.yaml"
CORPUS_SHA256 = "ff9dadf37bb9820e2c87df70a411fd189bb02776609e5ec06173dd03f2344004"
EXPORT = ("data/eval/gold_v1/questions_frozen_v2.jsonl", "fbcf2f5a124a17895ae465fb79f671372d5a2033a869ab4ab214e8676e270202")
VALIDATION_MANIFEST = ("data/processed/train/bge_base_ft_v3/validation_sets/validation_pairs_v4/manifest.json",
                       "40238cb06d73287821115dbe1b5c33a2dbbd294c7dd5b3c9fdc569c475e24b0e")
SELECTED = {"a1_seed42": (42, 55), "a1_seed1337": (1337, 40), "a1_seed2026": (2026, 40)}
DEV_DIR = "results/dev/pilot_v3_a1"
DEV_REFERENCE_RECORDS = "results/dev/pilot_v3"  # the harness's record of the zero-shot rankings scored with pilot_v3
DEPTH = 100
NEAR_DUPLICATE = 0.8  # token-set Jaccard at or above which two passages count as the same text


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def by_query(hits: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    for h in hits:
        out[h.query_id].append(h)
    for rows in out.values():
        rows.sort(key=lambda h: h.rank)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/eval/retrieval_analysis_v1.yaml")
    p.add_argument("--output-dir", default="results/analysis/retrieval_analysis_v1")
    p.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated subset (for checks)")
    return p.parse_args()


def development(config: dict[str, Any]) -> dict[str, Any]:
    """Score every system's stored development ranking with dev_pilot_v3's labels, using the harness's metrics.

    The zero-shot rankings are the stored references (``reference_runs_from``), the A0/A1 rankings the harness run
    of ``dev_pilot_v3_a1``. Each aggregate is checked against the harness's own record (``results/dev/pilot_v3`` for
    the references, ``DEV_DIR`` for A0/A1) and the check is reported. Unjudged passages count as non-relevant."""
    spec = config["evaluation_sets"]["dev_pilot_v3_a1"]
    qrels = [Qrel(**json.loads(line)) for line in open(spec["qrels_path"], encoding="utf-8")]
    judged: dict[str, set[str]] = defaultdict(set)
    for q in qrels:
        judged[q.query_id].add(q.passage_id)
    reference_dir = Path(spec["reference_runs_from"])
    out: dict[str, Any] = {"labels": {"path": spec["qrels_path"], "sha256": sha256_file(spec["qrels_path"])}}
    per_q: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for name in SYSTEMS:
        for mode in MODES:
            reference = name in ("bm25", "bge_zero_shot", "e5_zero_shot")
            run_path = (reference_dir if reference else Path(DEV_DIR)) / mode / name / "run.jsonl"
            record_path = (Path(DEV_REFERENCE_RECORDS) if reference else Path(DEV_DIR)) / mode / name / "metrics.json"
            if not run_path.exists():
                continue
            hits = [SearchHit(query_id=r["query_id"], passage_id=r["passage_id"], rank=r["rank"], score=r["score"],
                              retriever=name) for r in map(json.loads, open(run_path, encoding="utf-8"))]
            scores = compute_per_query_metrics(qrels, hits, ["ndcg@10", "mrr@10", "recall@10", "recall@100"])
            top = by_query(sorted(hits, key=lambda h: (h.rank, -h.score)))
            for qid in scores:
                ids = [h.passage_id for h in top.get(qid, [])][:10]
                scores[qid]["judged@10"] = sum(i in judged[qid] for i in ids) / 10
            per_q[(name, mode)] = scores
            row: dict[str, Any] = {"run_file": str(run_path), "run_sha256": sha256_file(run_path),
                                   "queries": len(scores)}
            for m in ("ndcg@10", "mrr@10", "recall@10", "recall@100", "judged@10"):
                row[m] = mean_interval([s[m] for s in scores.values()])
            recorded = json.loads(record_path.read_text(encoding="utf-8"))["aggregate"]
            row["harness_record"] = {"file": str(record_path), "sha256": sha256_file(record_path),
                                     **{m: {"recorded": recorded[m], "abs_diff": abs(recorded[m] - row[m]["mean"])}
                                        for m in ("ndcg@10", "mrr@10", "recall@10", "recall@100", "judged@10")
                                        if m in recorded}}
            out.setdefault(name, {})[mode] = row
    paired: dict[str, Any] = {}
    for a, b in [(s, "a0_step0") for s in A1 + ["e5_zero_shot", "bm25"]] + [(s, "e5_zero_shot") for s in A1]:
        for mode in MODES:
            if (a, mode) in per_q and (b, mode) in per_q:
                for m in ("mrr@10", "ndcg@10", "recall@10"):
                    paired[f"{a} - {b} | {mode} | {m}"] = paired_difference(
                        {q: s[m] for q, s in per_q[(a, mode)].items()},
                        {q: s[m] for q, s in per_q[(b, mode)].items()})
    out["paired"] = paired
    return out


def main() -> int:
    args = parse_args()
    started = time.time()
    config = read_yaml(args.config)
    systems = [s for s in args.systems.split(",") if s]
    if unknown := sorted(set(systems) - set(SYSTEMS)):
        raise SystemExit(f"unknown systems {unknown}")
    entries = {e["name"]: e for e in config["baselines"] + config["evaluation_sets"]["dev_pilot_v3_a1"]["retrievers"]}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- frozen inputs -------------------------------------------------------------------------------------
    if sha256_file(config["corpus_path"]) != CORPUS_SHA256:
        raise SystemExit("corpus SHA-256 differs from corpus_v1@1.2")
    for path, digest in (EXPORT, VALIDATION_MANIFEST):
        if sha256_file(path) != digest:
            raise SystemExit(f"{path} differs from its frozen SHA-256")
    train_config, train_sha = load_trainer_config(TRAIN_CONFIG)
    inputs = prepare_inputs(train_config, train_sha)  # verifies the corpus, the 678 pairs and validation isolation
    corpus = inputs.corpus
    passages = list(corpus.values())
    pairs = list(inputs.validation)
    relevant = {p.pair_id: frozenset(inputs.guard.positives(p)) for p in pairs}
    removed = {p.pair_id: frozenset(inputs.guard.exclusions(p)) - relevant[p.pair_id] for p in pairs}
    depth = DEPTH + max(len(r) for r in removed.values())
    questions = [json.loads(line) for line in open(EXPORT[0], encoding="utf-8")]
    trained = exclusion_ids(A1_TRAINING_AND_VALIDATION)
    info = {
        pid: HitInfo(p.ticker, fiscal_year(p.metadata, p.filing_date), bool(p.metadata.get("contains_table")))
        for pid, p in corpus.items()
    }
    # a year-blind ranker's expected exact-year share in company mode: the asked companies' passages from the asked years
    by_company_year = Counter((h.ticker, h.fiscal_year) for h in info.values())
    by_company = Counter(h.ticker for h in info.values())

    def chance_exact_year(tickers: list[str], years: list[int]) -> float:
        hit = sum(by_company_year[(t, int(y))] for t in tickers for y in years)
        return hit / sum(by_company[t] for t in tickers)

    token_sets: dict[str, set[str]] = {}

    def toks(pid: str) -> set[str]:
        if pid not in token_sets:
            token_sets[pid] = tokens(corpus[pid].text)
        return token_sets[pid]

    figures = {p.pair_id: answer_figures(p.answer) for p in pairs}

    validation_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    top10: dict[tuple[str, str, str, str], list[str]] = {}
    systems_described: dict[str, Any] = {}

    for name in systems:
        entry = entries[name]
        t0 = time.time()
        retriever = build_retriever(entry, corpus_sha256=CORPUS_SHA256, cache_dir=config["embedding_cache_dir"])
        retriever.index(passages)
        described = retriever.describe()
        systems_described[name] = {
            "entry": entry,
            "cache": described.get("cache"),
            "checkpoint": (described.get("checkpoint") or {}).get("digest") if isinstance(described.get("checkpoint"), dict) else None,
            "describe_keys": sorted(described),
            "index_seconds": round(time.time() - t0, 1),
        }
        if hasattr(retriever, "checkpoint"):
            prov = retriever.checkpoint.provenance()
            systems_described[name]["checkpoint"] = {k: prov.get(k) for k in ("path", "digest", "step", "selection")
                                                     if k in prov}
        # 1. validation, as the trainer scored it
        for mode in MODES:
            q = {p.pair_id: p.query for p in pairs}
            restrict = {p.pair_id: [p.ticker] for p in pairs} if mode == "company" else None
            hits = by_query(retriever.batch_search(q, top_k=depth, tickers=restrict))
            for p in pairs:
                ranked = [h.passage_id for h in hits[p.pair_id] if h.passage_id not in removed[p.pair_id]][:DEPTH]
                rel = relevant[p.pair_id]
                m = question_metrics(ranked, rel)
                m["recall@10"] = _recall_at_k(dict.fromkeys(rel, 1), ranked, 10)
                first = next((r for r, pid in enumerate(ranked, start=1) if pid in rel), None)
                top1 = ranked[0]
                t1 = info[top1]
                qfy = int(p.fiscal_year)
                near = max((jaccard(toks(top1), toks(r)) for r in rel), default=0.0)
                validation_rows.append({
                    "system": name, "mode": mode, "pair_id": p.pair_id, "ticker": p.ticker, "fiscal_year": qfy,
                    "topic": p.question_type, "relevant": len(rel), "removed": len(removed[p.pair_id]),
                    "positive_has_table": info[p.positive_passage_id].table,
                    **{k: m[k] for k in ("mrr@10", "ndcg@10", "recall@10", "recall@100")},
                    "first_relevant_rank": first,
                    "top1": {"passage_id": top1, "relevant": top1 in rel, "ticker": t1.ticker,
                             "fiscal_year": t1.fiscal_year, "table": t1.table,
                             "section": corpus[top1].section_title,
                             "same_company": t1.ticker == p.ticker, "same_year": t1.fiscal_year == qfy,
                             "max_jaccard_to_relevant": round(near, 3)},
                    "figures": figures[p.pair_id],
                    "figure_in_top1": any(figure_in_text(f, corpus[top1].text) for f in figures[p.pair_id])
                    if figures[p.pair_id] else None,
                    "figure_in_top10": any(figure_in_text(f, corpus[pid].text) for pid in ranked[:10]
                                           for f in figures[p.pair_id]) if figures[p.pair_id] else None,
                    "figure_in_relevant": any(figure_in_text(f, corpus[r].text) for r in rel
                                              for f in figures[p.pair_id]) if figures[p.pair_id] else None,
                })
                top10[("validation", name, mode, p.pair_id)] = ranked[:10]
        # 3. the 100 frozen human-written questions: metadata alignment only
        for mode in MODES:
            q = {x["query_id"]: x["text"] for x in questions}
            restrict = {x["query_id"]: x["tickers"] for x in questions} if mode == "company" else None
            hits = by_query(retriever.batch_search(q, top_k=10, tickers=restrict))
            for x in questions:
                ids = [h.passage_id for h in hits[x["query_id"]]]
                scope = Scope.of(x["tickers"], x["fiscal_years"])
                gold_rows.append({
                    "system": name, "mode": mode, "query_id": x["query_id"], "question_type": x["question_type"],
                    "comparison": x["comparison"], "tickers": x["tickers"], "fiscal_years": x["fiscal_years"],
                    **question_alignment([info[i] for i in ids], scope, k=10),
                    "trained_share": sum(i in trained for i in ids) / len(ids),
                    "chance_exact_year": chance_exact_year(x["tickers"], x["fiscal_years"]),
                })
                top10[("gold", name, mode, x["query_id"])] = ids
        del retriever
        print(f"[09] {name}: done in {time.time() - t0:.0f}s", flush=True)

    # ---- aggregates -----------------------------------------------------------------------------------------
    metrics = ["mrr@10", "recall@10", "ndcg@10", "recall@100"]
    validation: dict[str, Any] = {}
    for name in systems:
        for mode in MODES:
            rows = [r for r in validation_rows if r["system"] == name and r["mode"] == mode]
            validation.setdefault(name, {})[mode] = {
                **{m: mean_interval([r[m] for r in rows]) for m in metrics},
                "relevant_in_top10": sum(r["first_relevant_rank"] is not None and r["first_relevant_rank"] <= 10
                                         for r in rows),
                "relevant_beyond_100": sum(r["first_relevant_rank"] is None for r in rows),
                "top1_relevant": sum(r["top1"]["relevant"] for r in rows),
                "top1_wrong_company": sum(not r["top1"]["same_company"] for r in rows),
                "top1_right_company_wrong_year": sum(r["top1"]["same_company"] and not r["top1"]["same_year"]
                                                     and not r["top1"]["relevant"] for r in rows),
                "top1_same_text_other_filing": sum(not r["top1"]["relevant"]
                                                   and r["top1"]["max_jaccard_to_relevant"] >= NEAR_DUPLICATE
                                                   for r in rows),
                "top1_wrong_company_by_topic": dict(Counter(r["topic"] for r in rows
                                                            if not r["top1"]["same_company"]).most_common()),
                "top1_wrong_company_relevant_in_top10": sum(
                    not r["top1"]["same_company"] and r["first_relevant_rank"] is not None
                    and r["first_relevant_rank"] <= 10 for r in rows),
                "top1_wrong_year_offsets": dict(sorted(Counter(
                    abs(r["top1"]["fiscal_year"] - r["fiscal_year"]) for r in rows
                    if r["top1"]["same_company"] and not r["top1"]["same_year"] and not r["top1"]["relevant"]).items())),
                "with_figures": sum(r["figures"] != [] for r in rows),
                "figure_in_top1": sum(bool(r["figure_in_top1"]) for r in rows),
                "figure_in_top10": sum(bool(r["figure_in_top10"]) for r in rows),
                "figure_in_relevant": sum(bool(r["figure_in_relevant"]) for r in rows),
                "questions": len(rows),
            }

    def per_q(name: str, mode: str, metric: str) -> dict[str, float]:
        return {r["pair_id"]: r[metric] for r in validation_rows if r["system"] == name and r["mode"] == mode}

    paired: dict[str, Any] = {}
    for a, b in ([(s, "a0_step0") for s in A1 + ["bm25", "bge_zero_shot", "e5_zero_shot"]]
                 + [(s, "e5_zero_shot") for s in A1] + [(s, "bm25") for s in A1]):
        if a in systems and b in systems:
            for mode in MODES:
                for metric in ("mrr@10", "ndcg@10", "recall@10"):
                    paired[f"{a} - {b} | {mode} | {metric}"] = paired_difference(per_q(a, mode, metric),
                                                                                  per_q(b, mode, metric))
    # breakdowns (global mode, MRR@10): by topic and by whether the positive passage is a table
    breakdown: dict[str, Any] = {"by_topic": {}, "by_positive_table": {}}
    for name in systems:
        rows = [r for r in validation_rows if r["system"] == name and r["mode"] == "global"]
        for key, field in (("by_topic", "topic"), ("by_positive_table", "positive_has_table")):
            groups: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                groups[str(r[field])].append(r["mrr@10"])
            breakdown[key][name] = {g: {"n": len(v), "mrr@10": sum(v) / len(v)} for g, v in sorted(groups.items())}

    # stage A reproduction: the recorded per-question first-relevant ranks and aggregates (Kaggle T4)
    verification: dict[str, Any] = {}
    for name in [s for s in ["a0_step0"] + A1 if s in systems]:
        seed, step = SELECTED.get(name, (42, 0))
        log = {json.loads(line)["step"]: json.loads(line) for line in
               open(f"models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed{seed}/validation_log.jsonl", encoding="utf-8")}
        recorded = log[step]["metrics"]
        rec_pq = {q["pair_id"]: q for q in recorded["per_question"]}
        out: dict[str, Any] = {"seed_log": seed, "step": step}
        for mode in MODES:
            mine = {r["pair_id"]: r for r in validation_rows if r["system"] == name and r["mode"] == mode}
            same = sum(mine[k]["first_relevant_rank"] == rec_pq[k][mode]["first_relevant_rank"] for k in mine)
            out[mode] = {
                "first_relevant_rank_identical": f"{same}/{len(mine)}",
                "differing": sorted((k, mine[k]["first_relevant_rank"], rec_pq[k][mode]["first_relevant_rank"])
                                    for k in mine if mine[k]["first_relevant_rank"] != rec_pq[k][mode]["first_relevant_rank"]),
                **{m: {"recorded": recorded[mode][m], "reproduced": validation[name][mode][m]["mean"],
                       "abs_diff": abs(recorded[mode][m] - validation[name][mode][m]["mean"])}
                   for m in ("mrr@10", "ndcg@10", "recall@100")},
            }
        verification[name] = out

    # the real-question alignment diagnostic
    align_fields = ["company_share", "cross_company_share", "exact_year_share", "later_comparative_share",
                    "cross_year_share", "table_share", "trained_share", "company_hit", "in_scope_hit",
                    "top1_company", "top1_in_scope", "cells_covered_exact", "all_cells_covered_exact",
                    "cells_covered_with_comparatives", "companies_covered", "chance_exact_year"]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"questions": len(rows)}
        for f in align_fields:
            vals = [float(r[f]) for r in rows if r[f] is not None]
            out[f] = sum(vals) / len(vals) if vals else None
        ranks = [r["first_in_scope_rank"] for r in rows if r["first_in_scope_rank"] is not None]
        out["mean_first_in_scope_rank_when_found"] = sum(ranks) / len(ranks) if ranks else None
        return out

    alignment: dict[str, Any] = {"overall": {}, "by_question_type": {}, "by_comparison": {}}
    for name in systems:
        for mode in MODES:
            rows = [r for r in gold_rows if r["system"] == name and r["mode"] == mode]
            alignment["overall"].setdefault(name, {})[mode] = summarize(rows)
            for key, field in (("by_question_type", "question_type"), ("by_comparison", "comparison")):
                for g in sorted({r[field] for r in rows}):
                    alignment[key].setdefault(name, {}).setdefault(mode, {})[g] = summarize(
                        [r for r in rows if r[field] == g])

    # agreement between systems (top-10 sets, global mode)
    agreement: dict[str, Any] = {}
    for kind, ids in (("validation", [p.pair_id for p in pairs]), ("gold", [x["query_id"] for x in questions])):
        for a, b in combinations([s for s in ["a0_step0", "bge_zero_shot"] + A1 if s in systems], 2):
            js = [jaccard(set(top10[(kind, a, "global", q)]), set(top10[(kind, b, "global", q)])) for q in ids]
            t1 = [top10[(kind, a, "global", q)][0] == top10[(kind, b, "global", q)][0] for q in ids]
            agreement.setdefault(kind, {})[f"{a} vs {b}"] = {"mean_top10_jaccard": sum(js) / len(js),
                                                              "same_top1": sum(t1), "questions": len(ids)}

    # complementarity on validation: does BM25 find in its top 10 what a dense system misses, and the reverse?
    def in_top10(r: dict[str, Any]) -> bool:
        return r["first_relevant_rank"] is not None and r["first_relevant_rank"] <= 10

    complementarity: dict[str, Any] = {}
    for s in [x for x in A1 + ["e5_zero_shot", "a0_step0"] if x in systems and "bm25" in systems]:
        for mode in MODES:
            dense = {r["pair_id"]: in_top10(r) for r in validation_rows if r["system"] == s and r["mode"] == mode}
            lexical = {r["pair_id"]: in_top10(r) for r in validation_rows if r["system"] == "bm25" and r["mode"] == mode}
            complementarity[f"{s} vs bm25 | {mode}"] = {
                "both": sum(dense[q] and lexical[q] for q in dense),
                "dense_only": sum(dense[q] and not lexical[q] for q in dense),
                "bm25_only": sum(lexical[q] and not dense[q] for q in dense),
                "neither": sum(not dense[q] and not lexical[q] for q in dense),
                "questions": len(dense),
            }

    # A1 consensus failures on validation (global): every seed misses the top 10
    consensus = []
    if all(s in systems for s in A1):
        for p in pairs:
            ranks = {s: next(r["first_relevant_rank"] for r in validation_rows if r["system"] == s
                             and r["mode"] == "global" and r["pair_id"] == p.pair_id) for s in ["a0_step0"] + A1}
            if all(ranks[s] is None or ranks[s] > 10 for s in A1):
                consensus.append({"pair_id": p.pair_id, "query": p.query, "ticker": p.ticker,
                                  "fiscal_year": p.fiscal_year, "ranks": ranks})

    # ---- the development set (harness outputs) ------------------------------------------------------------
    dev = development(config)

    results = {
        "kind": "finretriever_retrieval_analysis",
        "version": 1,
        "notice": ("Not a benchmark. Validation: 93 synthetic, AI-reviewed questions that chose each A1 checkpoint. "
                   "Development: 10 questions with AI-written provisional labels. Real questions: a metadata "
                   "alignment diagnostic on the 100 frozen human-written questions; no relevance label read, no "
                   "relevance metric computed."),
        "inputs": {
            "corpus": {"path": config["corpus_path"], "sha256": CORPUS_SHA256},
            "question_export_v2": {"path": EXPORT[0], "sha256": EXPORT[1]},
            "validation_manifest": {"path": VALIDATION_MANIFEST[0], "sha256": VALIDATION_MANIFEST[1]},
            "trainer_config": {"path": TRAIN_CONFIG, "sha256": train_sha},
            "analysis_config": {"path": args.config, "sha256": sha256_file(args.config)},
            "a1_training_and_validation_passages": len(trained),
        },
        "systems": systems_described,
        "validation": validation,
        "validation_paired": paired,
        "validation_breakdown": breakdown,
        "validation_reproduces_stage_a": verification,
        "validation_a1_consensus_misses": consensus,
        "validation_complementarity": complementarity,
        "development": dev,
        "alignment": alignment,
        "agreement": agreement,
        "settings": {"depth": DEPTH, "near_duplicate_jaccard": NEAR_DUPLICATE, "modes": MODES,
                     "systems": systems, "seconds": round(time.time() - started)},
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    with open(out_dir / "validation_per_question.jsonl", "w", encoding="utf-8") as f:
        for r in validation_rows:
            f.write(json.dumps(r, default=str) + "\n")
    with open(out_dir / "alignment_per_question.jsonl", "w", encoding="utf-8") as f:
        for r in gold_rows:
            f.write(json.dumps(r, default=str) + "\n")
    with open(out_dir / "top10.jsonl", "w", encoding="utf-8") as f:
        for (kind, name, mode, qid), ids in top10.items():
            f.write(json.dumps({"set": kind, "system": name, "mode": mode, "query_id": qid, "top10": ids}) + "\n")
    (out_dir / "tables.md").write_text(render_tables(results), encoding="utf-8")
    print(f"[09] wrote {out_dir} in {time.time() - started:.0f}s; counts {dict(Counter(r['system'] for r in validation_rows))}")
    return 0


LABEL = {"bm25": "BM25", "bge_zero_shot": "BGE zero-shot", "e5_zero_shot": "E5 zero-shot", "a0_step0": "A0 (step 0)",
         "a1_seed42": "A1 seed 42", "a1_seed1337": "A1 seed 1337", "a1_seed2026": "A1 seed 2026"}


def _f(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def render_tables(r: dict[str, Any]) -> str:
    """Every table of docs/retrieval_analysis_v1.md, generated from results.json (nothing typed by hand)."""
    systems = r["settings"]["systems"]
    out = ["# Retrieval Analysis v1: generated tables\n", f"_{r['notice']}_\n"]
    for mode in MODES:
        rows = []
        for s in systems:
            v = r["validation"][s][mode]
            rows.append([LABEL[s]] + [f"{_f(v[m]['mean'])} [{_f(v[m]['low'])}, {_f(v[m]['high'])}]"
                                      for m in ("mrr@10", "recall@10", "ndcg@10")]
                        + [_f(v["recall@100"]["mean"]), f"{v['relevant_in_top10']}/{v['questions']}",
                           f"{v['top1_relevant']}/{v['questions']}"])
        out += [f"\n## T1-{mode}. Validation (93 synthetic questions), {mode} mode\n",
                _table(["System", "MRR@10 [95% CI]", "Recall@10 [95% CI]", "nDCG@10 [95% CI]", "Recall@100",
                        "Relevant in top 10", "Top 1 relevant"], rows)]
    rows = []
    for key, d in r["validation_paired"].items():
        a_b, mode, metric = key.split(" | ")
        a, b = a_b.split(" - ")
        if metric != "mrr@10":
            continue
        rows.append([f"{LABEL[a]} − {LABEL[b]}", mode, f"{d['mean_difference']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]",
                     f"{d['wins']}/{d['ties']}/{d['losses']}", f"{d['p_value']:.4f}"])
    out += ["\n## T2. Validation, paired MRR@10 differences (question bootstrap; sign-flip test)\n",
            _table(["Comparison", "Mode", "Mean difference [95% CI]", "Wins/ties/losses", "p"], rows)]
    rows = []
    for s, v in r["validation_reproduces_stage_a"].items():
        for mode in MODES:
            x = v[mode]
            rows.append([LABEL[s], mode, f"{x['mrr@10']['recorded']:.6f}", f"{x['mrr@10']['reproduced']:.6f}",
                         f"{x['mrr@10']['abs_diff']:.2e}", f"{x['ndcg@10']['abs_diff']:.2e}",
                         x["first_relevant_rank_identical"]])
    out += ["\n## T3. Reproduction of the recorded Stage A validation scores (Kaggle T4 → this Mac, CPU)\n",
            _table(["System", "Mode", "MRR@10 recorded", "MRR@10 reproduced", "abs Δ MRR@10", "abs Δ nDCG@10",
                    "First relevant rank identical"], rows)]
    for mode in MODES:
        rows = []
        for s in SYSTEMS:
            d = r["development"].get(s, {}).get(mode)
            if d:
                diffs = [x["abs_diff"] for k, x in d["harness_record"].items() if isinstance(x, dict)]
                rows.append([LABEL[s]] + [f"{_f(d[m]['mean'])} [{_f(d[m]['low'])}, {_f(d[m]['high'])}]"
                                          for m in ("mrr@10", "recall@10", "ndcg@10")]
                            + [_f(d["recall@100"]["mean"]), _f(d["judged@10"]["mean"]), f"{max(diffs):.1e}"])
        out += [f"\n## T4-{mode}. Development set (10 questions, AI-written provisional labels), {mode} mode\n",
                _table(["System", "MRR@10 [95% CI]", "Recall@10 [95% CI]", "nDCG@10 [95% CI]", "Recall@100",
                        "judged@10", "Max abs Δ vs harness record"], rows)]
    rows = []
    for key, d in r["development"]["paired"].items():
        a_b, mode, metric = key.split(" | ")
        a, b = a_b.split(" - ")
        if metric == "mrr@10" or (metric == "recall@10" and b == "a0_step0"):
            rows.append([f"{LABEL[a]} − {LABEL[b]}", mode, metric,
                         f"{d['mean_difference']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]",
                         f"{d['wins']}/{d['ties']}/{d['losses']}", f"{d['p_value']:.4f}"])
    out += ["\n## T4-paired. Development set, paired differences (10 questions; question bootstrap; sign-flip test)\n",
            _table(["Comparison", "Mode", "Metric", "Mean difference [95% CI]", "Wins/ties/losses", "p"], rows)]
    for mode in MODES:
        rows = []
        for s in systems:
            a = r["alignment"]["overall"][s][mode]
            rows.append([LABEL[s], _pct(a["company_share"]), _pct(a["exact_year_share"]),
                         _pct(a["exact_year_share"] / a["company_share"]) if a["company_share"] else "n/a",
                         _pct(a["chance_exact_year"]),
                         _pct(a["later_comparative_share"]), _pct(a["cross_year_share"]),
                         _pct(a["cross_company_share"]), _pct(a["top1_in_scope"]), _pct(a["in_scope_hit"]),
                         _pct(a["all_cells_covered_exact"]), _pct(a["table_share"]), _pct(a["trained_share"])])
        out += [f"\n## T5-{mode}. Real-question alignment diagnostic (100 human-written questions, top 10), {mode} mode\n",
                _table(["System", "Correct company", "Exact year", "Exact year among correct company",
                        "Year-blind chance", "Later filing (comparative window)",
                        "Cross-year", "Cross-company", "Top 1 in exact scope", "Any in exact scope",
                        "All scope cells covered", "Table passages", "A1 training/validation passages"], rows)]
    for key, title in (("by_question_type", "question type"), ("by_comparison", "comparison")):
        rows = []
        groups = sorted(next(iter(r["alignment"][key].values()))["global"])
        for g in groups:
            for s in systems:
                a = r["alignment"][key][s]["global"][g]
                rows.append([g, str(a["questions"]), LABEL[s], _pct(a["company_share"]), _pct(a["exact_year_share"]),
                             _pct(a["cross_year_share"]), _pct(a["top1_in_scope"]), _pct(a["all_cells_covered_exact"]),
                             _pct(a["companies_covered"]), _pct(a["table_share"])])
        out += [f"\n## T6-{key}. Alignment by {title} (global mode, top 10)\n",
                _table([title.capitalize(), "Questions", "System", "Correct company", "Exact year", "Cross-year",
                        "Top 1 in exact scope", "All scope cells covered", "Companies covered", "Table passages"],
                       rows)]
    rows = []
    for s in systems:
        v = r["validation"][s]["global"]
        offsets = ", ".join(f"{k}y: {n}" for k, n in v["top1_wrong_year_offsets"].items())
        rows.append([LABEL[s], f"{v['top1_wrong_company']} ({v['top1_wrong_company_relevant_in_top10']})",
                     str(v["top1_right_company_wrong_year"]), offsets or "—",
                     str(v["top1_same_text_other_filing"]), str(v["questions"] - v["relevant_in_top10"]),
                     str(v["relevant_beyond_100"]), f"{v['figure_in_top1']}/{v['with_figures']}",
                     f"{v['figure_in_top10']}/{v['with_figures']}"])
    out += ["\n## T7. Validation error counts (global mode, 93 questions)\n",
            _table(["System", "Top 1 wrong company (of them, relevant still in top 10)",
                    "Top 1 right company, wrong year (not relevant)", "Years off (wrong-year top 1)",
                    "Top 1 = same text from another filing (not relevant)", "Relevant not in top 10",
                    "Relevant not in top 100", "Answer figure in top 1", "Answer figure in top 10"], rows)]
    rows = []
    for k, d in r["validation_complementarity"].items():
        pair, mode = k.split(" | ")
        s = pair.split(" vs ")[0]
        rows.append([LABEL[s], mode, str(d["both"]), str(d["dense_only"]), str(d["bm25_only"]), str(d["neither"]),
                     f"{d['questions'] - d['neither']}/{d['questions']}"])
    out += ["\n## T10. Validation complementarity: relevant passage in the top 10 of a dense system and of BM25\n",
            _table(["Dense system", "Mode", "Both", "Dense only", "BM25 only", "Neither", "Either (union)"], rows)]
    rows = []
    for g in ("True", "False"):
        for s in systems:
            d = r["validation_breakdown"]["by_positive_table"][s].get(g)
            if d:
                rows.append(["table" if g == "True" else "narrative", str(d["n"]), LABEL[s], _f(d["mrr@10"])])
    out += ["\n## T8. Validation MRR@10 by evidence type (global mode)\n",
            _table(["Positive passage", "Questions", "System", "MRR@10"], rows)]
    rows = []
    for kind, pairs_ in r["agreement"].items():
        for k, d in pairs_.items():
            a, b = k.split(" vs ")
            rows.append([kind, f"{LABEL[a]} vs {LABEL[b]}", _f(d["mean_top10_jaccard"]),
                         f"{d['same_top1']}/{d['questions']}"])
    out += ["\n## T9. Agreement between dense systems (global mode, top 10)\n",
            _table(["Questions", "Pair", "Mean top-10 Jaccard", "Same top 1"], rows)]
    return "".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
