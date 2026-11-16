#!/usr/bin/env python
"""FinRetriever v2: metadata-aware hybrid retrieval experiments (docs/retrieval_v2_metadata_hybrid.md).

Runs every experiment of ``configs/experiments/v2_metadata_hybrid.yaml`` for every A1 seed on:

1. the 93 synthetic validation questions (validation_pairs_v4), scored as checkpoint selection scored them
   (relevant = positive + exact equivalents; uncertain equivalents never ranked);
2. the 10 development questions (dev_pilot_v3, AI-written provisional labels), scored with the harness's metric
   code; unjudged passages count as non-relevant, and judged@10 and judged-only scores are reported;
3. the 100 frozen human-written questions: metadata alignment of the top 10 and the parser's accuracy against
   the authors' declared scope. No relevance label exists for them and none is read.

    python scripts/11_v2_experiments.py [--config ...] [--output-dir ...] [--seeds a1_seed42,...]
                                        [--experiments E1_a1,...] [--prepare-only]

Reads only frozen inputs (hashes checked), verifies every A1 checkpoint against its selection record, reuses the
embedding cache (new representations get their own cache entries) and writes only under --output-dir. The
outputs contain no timestamps or durations, so two runs can be compared byte for byte.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finretriever.data.schemas import Passage, Qrel, SearchHit  # noqa: E402
from finretriever.evaluation.alignment import HitInfo, Scope, question_alignment  # noqa: E402
from finretriever.evaluation.benchmark import _condensed  # noqa: E402
from finretriever.evaluation.metrics import _recall_at_k, compute_per_query_metrics  # noqa: E402
from finretriever.evaluation.uncertainty import mean_interval, paired_difference  # noqa: E402
from finretriever.retrieval.enrichment import passage_fiscal_year  # noqa: E402
from finretriever.retrieval.hybrid import (  # noqa: E402
    HybridConfig,
    HybridRetriever,
)
from finretriever.retrieval.metadata_scope import ScopePolicy, allowed_years  # noqa: E402
from finretriever.retrieval.pipeline import (  # noqa: E402
    SEEDLESS,
    build_bm25,
    build_seed_components,
    component_seeds,
    experiment_config,
    experiment_seeds,
)
from finretriever.retrieval.query_understanding import QueryParser, QueryScope  # noqa: E402
from finretriever.training.trainer_setup import load_trainer_config, prepare_inputs  # noqa: E402
from finretriever.training.validation_eval import question_metrics  # noqa: E402
from finretriever.utils.io import read_yaml  # noqa: E402

METRICS = ("mrr@10", "recall@10", "ndcg@10", "recall@100")
DEV_METRICS = ("ndcg@10", "mrr@10", "recall@10", "recall@100")
FROZEN_V1 = "results/analysis/retrieval_analysis_v1/validation_per_question.jsonl"
SOURCES = (  # hashed into the manifest: the code that defines every v2 ranking
    "scripts/11_v2_experiments.py",
    "src/finretriever/retrieval/query_understanding.py",
    "src/finretriever/retrieval/metadata_scope.py",
    "src/finretriever/retrieval/fusion.py",
    "src/finretriever/retrieval/hybrid.py",
    "src/finretriever/retrieval/enrichment.py",
    "src/finretriever/retrieval/windows.py",
    "src/finretriever/retrieval/pipeline.py",
    "src/finretriever/retrieval/bm25.py",
    "src/finretriever/retrieval/biencoder.py",
    "src/finretriever/retrieval/checkpoint.py",
)
LIBRARIES = ("numpy", "torch", "transformers", "sentence-transformers", "tokenizers", "pydantic", "PyYAML")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def log(message: str) -> None:
    print(f"[11] {message}", flush=True)


# ---- evaluation ----------------------------------------------------------------------------------------------


def oracle_scope(text: str, tickers: list[str], years: list[int], version: str) -> QueryScope:
    return QueryScope(text=text, tickers=tuple(sorted(tickers)), years=tuple(sorted(int(y) for y in years)),
                      measures=(), comparison="none", comparative=False, asks_for_figure=False,
                      parser_version=version)


def run_experiment(retriever: HybridRetriever, data: dict[str, Any], top_k: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"validation": [], "development": [], "real": []}
    info = data["info"]
    version = retriever.parser.version
    for p in data["pairs"]:
        hits, trace = retriever.search(p.query, top_k=top_k, query_id=p.pair_id, exclude=data["removed"][p.pair_id],
                                       oracle_scope=oracle_scope(p.query, [p.ticker], [p.fiscal_year], version))
        ranked = [h.passage_id for h in hits]
        rel = data["relevant"][p.pair_id]
        m = question_metrics(ranked, rel)
        m["recall@10"] = _recall_at_k(dict.fromkeys(rel, 1), ranked, 10)
        first = next((r for r, pid in enumerate(ranked, start=1) if pid in rel), None)
        out["validation"].append({"question_id": p.pair_id, **{k: m[k] for k in METRICS},
                                  "first_relevant_rank": first, "top10": ranked[:10], "trace": trace})
    dev_hits: list[SearchHit] = []
    for q in data["dev"]:
        meta = q["metadata"]
        hits, trace = retriever.search(q["text"], top_k=top_k, query_id=q["query_id"], oracle_scope=oracle_scope(
            q["text"], meta["tickers"].split(","), [int(y) for y in str(meta["fiscal_years"]).split(",") if y],
            version))
        dev_hits.extend(hits)
        out["development"].append({"question_id": q["query_id"], "ranked": [h.passage_id for h in hits],
                                   "trace": trace})
    scores = compute_per_query_metrics(data["qrels"], dev_hits, list(DEV_METRICS))
    judged_only = compute_per_query_metrics(data["qrels"], _condensed(dev_hits, data["judged"]),
                                            ["ndcg@10", "mrr@10"])
    for row in out["development"]:
        qid = row["question_id"]
        top10 = row.pop("ranked")[:10]
        row.update({**scores[qid], "judged@10": sum(pid in data["judged"][qid] for pid in top10) / 10,
                    "ndcg@10_judged_only": judged_only[qid]["ndcg@10"],
                    "mrr@10_judged_only": judged_only[qid]["mrr@10"], "top10": top10})
    for x in data["real"]:
        hits, trace = retriever.search(x["text"], top_k=10, query_id=x["query_id"],
                                       oracle_scope=oracle_scope(x["text"], x["tickers"], x["fiscal_years"], version))
        ids = [h.passage_id for h in hits]
        align = question_alignment([info[i] for i in ids], Scope.of(x["tickers"], x["fiscal_years"]), k=10)
        out["real"].append({"question_id": x["query_id"], "question_type": x["question_type"],
                            "comparison": x["comparison"], **align, "top10": ids, "trace": trace})
    return out


def parser_accuracy(parser: QueryParser, data: dict[str, Any]) -> dict[str, Any]:
    """The parser against each set's declared scope (validation: the positive's company and filing year)."""
    policy = ScopePolicy(year="filter")
    out: dict[str, Any] = {}
    sets = {
        "validation": [(p.query, [p.ticker], [int(p.fiscal_year)], None) for p in data["pairs"]],
        "development": [(q["text"], q["metadata"]["tickers"].split(","),
                         [int(y) for y in str(q["metadata"]["fiscal_years"]).split(",")],
                         q["metadata"]["comparison"]) for q in data["dev"]],
        "real": [(x["text"], x["tickers"], [int(y) for y in x["fiscal_years"]], x["comparison"])
                 for x in data["real"]],
    }
    roles_on = parser.version != "qu-v1"
    for name, rows in sets.items():
        c: Counter[str] = Counter()
        role_counts: Counter[str] = Counter()
        mistakes = []
        for text, tickers, years, comparison in rows:
            s = parser.parse(text)
            allowed = allowed_years(s, policy)
            if roles_on:
                role_counts.update(str(m.role) for m in s.year_mentions)
                non_reporting = {y for ys in s.non_reporting_years.values() for y in ys}
                c["questions_with_non_reporting_years"] += bool(non_reporting)
                c["declared_year_given_a_non_reporting_role"] += bool((set(years) & non_reporting) - set(s.years))
            c["questions"] += 1
            c["company_exact"] += set(s.tickers) == set(tickers)
            c["company_missed"] += not set(tickers) <= set(s.tickers)
            c["company_extra"] += bool(set(s.tickers) - set(tickers))
            c["company_none"] += not s.tickers
            c["year_none"] += not s.years
            c["year_exact"] += set(s.years) == set(years)
            c["declared_years_within_filter"] += bool(s.years) and set(years) <= allowed
            c["declared_years_excluded_by_filter"] += bool(s.years) and not set(years) <= allowed
            if comparison is not None:
                c["comparison_agrees"] += s.comparison == comparison
            if set(s.tickers) != set(tickers) or (s.years and not set(years) <= allowed):
                parsed: dict[str, Any] = {"tickers": list(s.tickers), "years": list(s.years)}
                if roles_on:
                    parsed["non_reporting_years"] = {r: list(ys) for r, ys in s.non_reporting_years.items()}
                mistakes.append({"text": text, "declared": {"tickers": tickers, "years": years}, "parsed": parsed})
        out[name] = {**dict(c), "mismatches": mistakes}
        if roles_on:
            out[name]["year_mention_roles"] = dict(sorted(role_counts.items()))
    return out


POSITION_BANDS = ((0, 128, "0-127"), (128, 256, "128-255"), (256, 10**9, "256+"))


def evidence_starts(pairs: list[Any], corpus: dict[str, Passage], tokenizer: Any) -> dict[str, int]:
    """Where each validation answer's evidence starts in its positive passage, in the encoder's tokens."""
    out = {}
    for pair in pairs:
        text = corpus[pair.positive_passage_id].text
        at = text.find(pair.evidence.strip()[:60])
        if at >= 0:
            out[pair.pair_id] = len(tokenizer(text[:at], add_special_tokens=False)["input_ids"])
    return out


WINDOW_POLICIES = {  # the year-window candidates compared on training + validation questions (never the real ones)
    "exact year + 2 later filings": {"before_explicit": 0, "before_implicit": 0},
    "chosen: also 1 earlier filing for years read from dates or plain numbers": {"before_explicit": 0,
                                                                                 "before_implicit": 1},
    "1 earlier filing for every year": {"before_explicit": 1, "before_implicit": 1},
}


def year_window_choice(parser: QueryParser, pairs: list[Any], passages: list[Passage]) -> dict[str, Any]:
    """How often each candidate year window keeps the positive passage's filing year, and how much it allows.

    Measured on the training and validation questions only: the evidence the window policy was chosen on."""
    years_by_ticker: dict[str, set[int]] = defaultdict(set)
    for p in passages:
        years_by_ticker[p.ticker].add(passage_fiscal_year(p))
    out: dict[str, Any] = {"questions": len(pairs), "policies": {}}
    for name, windows in WINDOW_POLICIES.items():
        policy = ScopePolicy(year="filter", **windows)
        n = kept = 0
        share = 0.0
        for pair in pairs:
            scope = parser.parse(pair.query)
            if not scope.years:
                continue
            allowed = allowed_years(scope, policy) & years_by_ticker[pair.ticker]
            n += 1
            kept += int(pair.fiscal_year) in allowed
            share += len(allowed) / len(years_by_ticker[pair.ticker])
        out["policies"][name] = {**windows, "after": policy.after, "questions_with_a_year": n,
                                 "filing_year_kept": kept, "kept_share": kept / n, "mean_years_allowed_share": share / n}
    return out


def summarize(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, Any]:
    return {m: mean_interval([r[m] for r in rows]) for m in metrics}


def scope_decisions(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        plan = r["trace"]["plan"]
        c["questions"] += 1
        c["company_filtered"] += plan["company_filtered"]
        c["year_filtered"] += plan["year_filtered"]
        c["year_fallback"] += any("fell back" in d for d in plan["decisions"])
        c["cells"] += bool(plan["cells"])
    return dict(c)


def alignment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["company_share", "cross_company_share", "exact_year_share", "later_comparative_share",
              "cross_year_share", "top1_in_scope", "in_scope_hit", "all_cells_covered_exact", "companies_covered"]
    out = {}
    for f in fields:
        vals = [float(r[f]) for r in rows if r[f] is not None]
        out[f] = sum(vals) / len(vals) if vals else None
    return out


REFERENCE_METRICS = {"validation": ("mrr@10", "recall@10", "ndcg@10", "recall@100"),
                     "development": ("mrr@10", "recall@10", "ndcg@10"), "real": ("exact_year_share",)}


def load_reference(cfg: dict[str, Any], configs: dict[str, HybridConfig]) -> dict[str, Any]:
    """A frozen earlier run to compare with, experiment by experiment: checked by its manifest's SHA-256, and every
    compared experiment must have the same settings there (so that only what the config changes, e.g. the parser,
    differs)."""
    ref = cfg["reference_results"]
    root = Path(ref["path"])
    if sha256_file(root / "manifest.json") != ref["manifest_sha256"]:
        raise SystemExit(f"{root / 'manifest.json'} differs from its configured SHA-256")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for eid, config in configs.items():
        theirs = manifest["experiments"].get(eid)
        if theirs is None or theirs["settings"] != config.describe():
            raise SystemExit(f"{eid}: the reference run has no experiment with these exact settings")
    return {"path": ref["path"], "manifest_sha256": ref["manifest_sha256"], "root": root,
            "parser_version": manifest["parser"]["parser_version"]}


def compare_with_reference(reference: dict[str, Any], eid: str, seed: str,
                           sets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Paired differences (this run minus the reference run of the same experiment and seed), and how many
    questions' top 10 changed."""
    paired: dict[str, Any] = {}
    changed: dict[str, int] = {}
    for set_name, metrics in REFERENCE_METRICS.items():
        theirs = {r["question_id"]: r for r in jsonl(reference["root"] / "runs" / eid / seed / f"{set_name}.jsonl")}
        mine = {r["question_id"]: r for r in sets[set_name]}
        if set(theirs) != set(mine):
            raise SystemExit(f"{eid} [{seed}] {set_name}: the reference run scored other questions")
        changed[set_name] = sum(mine[q]["top10"] != theirs[q]["top10"] for q in mine)
        for m in metrics:
            both = [q for q in mine if mine[q][m] is not None and theirs[q][m] is not None]
            paired[f"{set_name} | {m}"] = paired_difference({q: float(mine[q][m]) for q in both},
                                                            {q: float(theirs[q][m]) for q in both})
    return {"paired_vs_reference": paired, "top10_changed_vs_reference": changed}


# ---- main ----------------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/experiments/v2_metadata_hybrid.yaml")
    ap.add_argument("--output-dir", default="results/experiments/v2_metadata_hybrid")
    ap.add_argument("--seeds", default="", help="comma-separated subset of the configured seeds")
    ap.add_argument("--experiments", default="", help="comma-separated subset of experiment ids")
    ap.add_argument("--prepare-only", action="store_true", help="build and cache every component, run nothing")
    ap.add_argument("--tables-only", action="store_true", help="re-render tables.md from the written outputs")
    args = ap.parse_args()
    cfg = read_yaml(args.config)
    out_dir = Path(args.output_dir)
    if args.tables_only:
        loaded = [json.loads((out_dir / f).read_text(encoding="utf-8"))
                  for f in ("manifest.json", "summary.json", "parser_accuracy.json", "year_window_choice.json")]
        (out_dir / "tables.md").write_text(render_tables(*loaded, [e["id"] for e in cfg["experiments"]]),
                                           encoding="utf-8")
        log(f"rendered {out_dir / 'tables.md'}")
        return 0
    experiments = [e for e in cfg["experiments"] if not args.experiments or e["id"] in args.experiments.split(",")]
    if len({e["id"] for e in cfg["experiments"]}) != len(cfg["experiments"]):
        raise SystemExit("experiment ids must be unique")
    seeds = [s for s in cfg["seeds"] if not args.seeds or s in args.seeds.split(",")]
    configs = {e["id"]: experiment_config(e, cfg["defaults"]) for e in experiments}

    # frozen inputs
    if sha256_file(cfg["corpus_path"]) != cfg["corpus_sha256"]:
        raise SystemExit("corpus SHA-256 differs from the configured corpus")
    for key in ("validation_manifest", "real_questions"):
        path, digest = cfg["inputs"][key]
        if sha256_file(path) != digest:
            raise SystemExit(f"{path} differs from its frozen SHA-256")
    train_config, train_sha = load_trainer_config(cfg["inputs"]["trainer_config"])
    inputs = prepare_inputs(train_config, train_sha)
    passages = list(inputs.corpus.values())
    position = {p.passage_id: i for i, p in enumerate(passages)}
    pairs = list(inputs.validation)
    relevant = {p.pair_id: frozenset(inputs.guard.positives(p)) for p in pairs}
    removed = {p.pair_id: frozenset(inputs.guard.exclusions(p)) - relevant[p.pair_id] for p in pairs}
    qrels = [Qrel(**r) for r in jsonl(cfg["inputs"]["dev_qrels"])]
    judged: dict[str, set[str]] = defaultdict(set)
    for q in qrels:
        judged[q.query_id].add(q.passage_id)
    data = {
        "pairs": pairs, "relevant": relevant, "removed": removed, "qrels": qrels, "judged": judged,
        "dev": jsonl(cfg["inputs"]["dev_queries"]), "real": jsonl(cfg["inputs"]["real_questions"][0]),
        "info": {p.passage_id: HitInfo(p.ticker, passage_fiscal_year(p), bool(p.metadata.get("contains_table")))
                 for p in passages},
    }
    aliases = (read_yaml(cfg["parser"]["aliases_from"]).get("generation") or {}).get("company_aliases") or {}
    parser = QueryParser.from_passages(passages, aliases, version=cfg["parser"].get("version", "qu-v1"))
    reference = load_reference(cfg, configs) if cfg.get("reference_results") else None

    needed = sorted({c for e in experiments for c in e["components"]})
    bm25_scorers: dict[str, Any] = {}
    described: dict[str, Any] = {}
    for name in needed:
        if cfg["components"][name]["type"] == "bm25":
            t0 = time.time()
            bm25_scorers[name], described[name] = build_bm25(name, cfg["components"][name], passages)
            log(f"{name} indexed in {time.time() - t0:.0f}s")

    runs: dict[tuple[str, str], dict[str, list[dict]]] = {}
    window_cache: dict[str, Any] = {}
    seed_runs = [(seed, [e for e in experiments if seed in experiment_seeds(e, cfg)]) for seed in seeds]
    seedless = [e for e in experiments if experiment_seeds(e, cfg) == [SEEDLESS]]
    for seed, seed_experiments in [(SEEDLESS, seedless)] + seed_runs:
        if not seed_experiments:
            continue
        names = sorted({c for e in seed_experiments for c in e["components"]
                        if cfg["components"][c]["type"] != "bm25" and seed in component_seeds(c, cfg)})
        scorers = dict(bm25_scorers)
        if seed != SEEDLESS:
            seed_scorers, seed_described = build_seed_components(seed, names, cfg, passages, position, window_cache,
                                                                 log)
            scorers.update(seed_scorers)
            described.update({f"{seed}:{k}": v for k, v in seed_described.items()})
        if args.prepare_only:
            continue
        for e in seed_experiments:
            t0 = time.time()
            retriever = HybridRetriever(configs[e["id"]], scorers, passages, parser)
            runs[(e["id"], seed)] = run_experiment(retriever, data, cfg["top_k"])
            log(f"{e['id']} [{seed}] in {time.time() - t0:.0f}s")
        del scorers
        gc.collect()
    if args.prepare_only:
        log("components prepared; nothing run")
        return 0

    # ---- outputs -------------------------------------------------------------------------------------------
    for (eid, seed), sets in runs.items():
        for set_name, rows in sets.items():
            dump_jsonl(out_dir / "runs" / eid / seed / f"{set_name}.jsonl", rows)

    def per_question(eid: str, seed: str, set_name: str, metric: str) -> dict[str, float]:
        return {r["question_id"]: r[metric] for r in runs[(eid, seed)][set_name]}

    from transformers import (
        AutoTokenizer,  # the A1 checkpoints' own tokenizer (identical in every seed)
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["seeds"][seeds[0]]["checkpoint_path"], local_files_only=True)
    starts = evidence_starts(pairs, inputs.corpus, tokenizer)
    summary: dict[str, Any] = {}
    for (eid, seed), sets in sorted(runs.items()):
        entry = {
            "validation": summarize(sets["validation"], METRICS),
            "development": summarize(sets["development"], (*DEV_METRICS, "judged@10", "ndcg@10_judged_only",
                                                           "mrr@10_judged_only")),
            "real_alignment": alignment_summary(sets["real"]),
            "real_cells_by_comparison": {
                comp: {"questions": len(v), "all_cells_covered": sum(bool(r["all_cells_covered_exact"]) for r in v)}
                for comp in sorted({r["comparison"] for r in sets["real"]})
                for v in [[r for r in sets["real"] if r["comparison"] == comp]]
            },
            "scope_decisions": {s: scope_decisions(sets[s]) for s in sets},
            "validation_mrr_by_evidence_start": {
                label: {"n": len(v), "mrr@10": sum(v) / len(v) if v else None}
                for lo, hi, label in POSITION_BANDS
                for v in [[r["mrr@10"] for r in sets["validation"] if lo <= starts.get(r["question_id"], -1) < hi]]
            },
        }
        baseline_seeds = [seed] if seed != SEEDLESS else [s for s in seeds]
        entry["paired_vs_E1"] = {}
        for b_seed in baseline_seeds:
            if ("E1_a1", b_seed) not in runs or eid == "E1_a1":
                continue
            for set_name, metrics in (("validation", ("mrr@10", "recall@10", "ndcg@10")),
                                      ("development", ("mrr@10", "recall@10", "ndcg@10"))):
                for m in metrics:
                    entry["paired_vs_E1"][f"{b_seed} | {set_name} | {m}"] = paired_difference(
                        per_question(eid, seed, set_name, m), per_question("E1_a1", b_seed, set_name, m))
        if reference is not None:
            entry.update(compare_with_reference(reference, eid, seed, sets))
        summary.setdefault(eid, {})[seed] = entry

    # baseline preservation: E1 / R_bm25 / oracle-company reproduce the frozen v1 analysis rankings
    v1 = defaultdict(dict)
    for r in jsonl(FROZEN_V1):
        v1[(r["system"], r["mode"])][r["pair_id"]] = r["first_relevant_rank"]
    checks = {}
    for eid, system, mode in (("E1_a1", "{seed}", "global"), ("R_a1_oracle_company", "{seed}", "company"),
                              ("R_bm25", "bm25", "global")):
        for seed in ([SEEDLESS] if eid == "R_bm25" else seeds):
            if (eid, seed) not in runs:
                continue
            ref = v1[(system.format(seed=seed), mode)]
            mine = {r["question_id"]: r["first_relevant_rank"] for r in runs[(eid, seed)]["validation"]}
            checks[f"{eid} [{seed}] vs v1 {system.format(seed=seed)}/{mode}"] = (
                f"{sum(mine[q] == ref[q] for q in mine)}/{len(mine)} first-relevant ranks identical")

    versions = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            versions[lib] = importlib_metadata.version(lib)
        except importlib_metadata.PackageNotFoundError:
            versions[lib] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "kind": "finretriever_v2_experiments",
        "config": {"path": args.config, "sha256": sha256_file(args.config), "name": cfg["name"],
                   "version": cfg["version"]},
        "experiments": {e["id"]: {"role": e["role"], "seeds": experiment_seeds(e, cfg),
                                  "settings": configs[e["id"]].describe()} for e in experiments},
        "inputs": {
            "corpus": {"path": cfg["corpus_path"], "sha256": cfg["corpus_sha256"]},
            "trainer_config": {"path": cfg["inputs"]["trainer_config"], "sha256": train_sha},
            "validation_manifest": {"path": cfg["inputs"]["validation_manifest"][0],
                                    "sha256": cfg["inputs"]["validation_manifest"][1]},
            "dev_queries": {"path": cfg["inputs"]["dev_queries"], "sha256": sha256_file(cfg["inputs"]["dev_queries"])},
            "dev_qrels": {"path": cfg["inputs"]["dev_qrels"], "sha256": sha256_file(cfg["inputs"]["dev_qrels"])},
            "real_questions": {"path": cfg["inputs"]["real_questions"][0], "sha256": cfg["inputs"]["real_questions"][1]},
            "parser_aliases": {"path": cfg["parser"]["aliases_from"], "sha256": sha256_file(cfg["parser"]["aliases_from"])},
        },
        "parser": parser.describe(),
        **({"reference_results": {k: reference[k] for k in ("path", "manifest_sha256", "parser_version")}}
           if reference is not None else {}),
        "source_sha256": {path: sha256_file(path) for path in SOURCES},
        "components": described,
        "software": versions,
        "git_commit": commit,
        "baseline_preservation": checks,
        "notice": ("Not a benchmark. Validation questions are synthetic and selected the A1 checkpoints; development "
                   "labels are AI-written; the 100 real questions have no relevance labels and are scored for "
                   "metadata alignment and parser accuracy only."),
    }
    accuracy = parser_accuracy(parser, data)
    window_choice = year_window_choice(parser, list(inputs.pairs) + pairs, passages)
    dump(out_dir / "manifest.json", manifest)
    dump(out_dir / "summary.json", summary)
    dump(out_dir / "parser_accuracy.json", accuracy)
    dump(out_dir / "year_window_choice.json", window_choice)
    (out_dir / "tables.md").write_text(render_tables(manifest, summary, accuracy, window_choice,
                                                     [e["id"] for e in cfg["experiments"]]), encoding="utf-8")
    log(f"wrote {out_dir}: {len(runs)} experiment runs; baseline checks {checks}")
    return 0


# ---- tables --------------------------------------------------------------------------------------------------


def _f(x: float | None, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def _across(values: list[float]) -> str:
    """Mean over seeds, with the seed range when there are several."""
    mean = sum(values) / len(values)
    return _f(mean) if len(values) == 1 else f"{_f(mean)} ({_f(min(values))}–{_f(max(values))})"


def _count_across(values: list[int]) -> str:
    """A count over seeds: the value when every seed agrees, else the mean (one decimal) and the seed range."""
    if min(values) == max(values):
        return str(values[0])
    return f"{sum(values) / len(values):.1f} ({min(values)}–{max(values)})"


def _delta(entries: dict[str, Any], set_name: str, metric: str) -> str:
    diffs = [(d["mean_difference"], d["p_value"]) for seed_entry in entries.values()
             for key, d in seed_entry["paired_vs_E1"].items() if key.endswith(f"| {set_name} | {metric}")]
    if not diffs:
        return "—"
    mean = sum(d for d, _ in diffs) / len(diffs)
    return f"{mean:+.3f} (p ≤ {max(p for _, p in diffs):.4f})" if len(diffs) > 1 else f"{mean:+.3f} (p {diffs[0][1]:.4f})"


def render_tables(manifest: dict[str, Any], summary: dict[str, Any], accuracy: dict[str, Any],
                  window_choice: dict[str, Any], config_order: list[str]) -> str:
    """Every table of docs/retrieval_v2_metadata_hybrid.md, generated from the run outputs."""
    # the config's order (the manifest is written with sorted keys, so its own order is alphabetical)
    order = [e for e in config_order if e in manifest["experiments"]]
    roles = {e: manifest["experiments"][e]["role"] for e in order}
    out = ["# FinRetriever v2 experiments: generated tables\n", f"_{manifest['notice']}_\n"]

    def label(e: str) -> list[str]:
        seeds = [s for s in summary.get(e, {})]
        n = "seedless" if seeds == [SEEDLESS] else str(len(seeds))
        return [e, roles[e], n]

    rows = []
    for e in order:
        if e not in summary:
            continue
        v = [x["validation"] for x in summary[e].values()]
        rows.append(label(e) + [_across([s[m]["mean"] for s in v]) for m in METRICS]
                    + [_delta(summary[e], "validation", "mrr@10"), _delta(summary[e], "validation", "recall@10")])
    out += ["\n## V1. Validation (93 synthetic questions; they selected A1). Mean over seeds (seed range)\n",
            _table(["Experiment", "Role", "Seeds", "MRR@10", "Recall@10", "nDCG@10", "Recall@100",
                    "Δ MRR@10 vs E1 (largest p)", "Δ Recall@10 vs E1 (largest p)"], rows)]
    rows = []
    for e in order:
        if e not in summary:
            continue
        d = [x["development"] for x in summary[e].values()]
        rows.append(label(e) + [_across([s[m]["mean"] for s in d]) for m in
                                  ("mrr@10", "recall@10", "ndcg@10", "recall@100", "judged@10", "mrr@10_judged_only")]
                    + [_delta(summary[e], "development", "mrr@10")])
    out += ["\n## D1. Development (10 questions, AI-written provisional labels; unjudged = non-relevant)\n",
            _table(["Experiment", "Role", "Seeds", "MRR@10", "Recall@10", "nDCG@10", "Recall@100", "judged@10",
                    "MRR@10 judged-only", "Δ MRR@10 vs E1 (largest p)"], rows)]
    rows = []
    fields = ["company_share", "cross_company_share", "exact_year_share", "cross_year_share", "top1_in_scope",
              "in_scope_hit", "all_cells_covered_exact"]
    for e in order:
        if e not in summary:
            continue
        a = [x["real_alignment"] for x in summary[e].values()]
        rows.append(label(e) + [_pct(sum(s[f] for s in a) / len(a)) for f in fields])
    out += ["\n## A1. Real-question metadata alignment (100 human-written questions, top 10; mean over seeds). "
            "Not relevance; scoped systems align by construction when the parser is right\n",
            _table(["Experiment", "Role", "Seeds", "Correct company", "Cross-company", "Exact year", "Cross-year",
                    "Top 1 in exact scope", "Any in exact scope", "All scope cells covered"], rows)]
    rows = []
    for e in order:
        if e not in summary:
            continue
        cells = [x["real_cells_by_comparison"] for x in summary[e].values()]
        rows.append(label(e) + [
            f"{_count_across([c[comp]['all_cells_covered'] for c in cells])} of {cells[0][comp]['questions']}"
            for comp in ("cross_company", "cross_year", "none")])
    out += ["\n## A2. Real questions whose top 10 covers every company-year cell they ask about, by comparison type "
            "(mean over seeds, seed range)\n",
            _table(["Experiment", "Role", "Seeds", "Cross-company", "Cross-year", "Single scope"], rows)]
    rows = []
    for e in order:
        if e not in summary or manifest["experiments"][e]["settings"]["policy"] == ScopePolicy().describe():
            continue
        first = next(iter(summary[e].values()))["scope_decisions"]
        cells = []
        for s in ("validation", "development", "real"):
            d = first[s]
            cells.append(f"{d.get('company_filtered', 0)} / {d.get('year_filtered', 0)} / {d.get('year_fallback', 0)}"
                         f" / {d.get('cells', 0)} of {d['questions']}")
        rows.append([e] + cells)
    out += ["\n## S1. Scope decisions per question set: company filtered / year filtered / year fallback / cells\n",
            _table(["Experiment", "Validation", "Development", "Real questions"], rows)]
    rows = []
    for s in [k for k in ("validation", "development", "real") if k in accuracy]:
        d = accuracy[s]
        n = d["questions"]
        with_year = n - d.get("year_none", 0)
        rows.append([s, str(n), f"{d.get('company_exact', 0)}/{n}", f"{d.get('company_missed', 0)}",
                     f"{d.get('company_extra', 0)}", f"{d.get('year_exact', 0)}/{n}",
                     f"{d.get('declared_years_within_filter', 0)}/{with_year}", f"{d.get('year_none', 0)}",
                     f"{d['comparison_agrees']}/{n}" if "comparison_agrees" in d else "—"])
    out += ["\n## P1. Query understanding against each set's declared scope (parser "
            f"{manifest['parser']['parser_version']}; validation: the positive passage's company and filing year)\n",
            _table(["Set", "Questions", "Companies exact", "Declared company missed", "Extra company",
                    "Years exact", "Declared years inside the year filter", "No year parsed", "Comparison type agrees"],
                   rows)]
    if any("year_mention_roles" in accuracy[s] for s in accuracy):
        rows = []
        for s in [k for k in ("validation", "development", "real") if k in accuracy]:
            d = accuracy[s]
            counts = d.get("year_mention_roles", {})
            rows.append([s, str(d["questions"]), str(d.get("questions_with_non_reporting_years", 0)),
                         str(d.get("declared_year_given_a_non_reporting_role", 0))]
                        + [str(counts.get(r, 0)) for r in ("fiscal", "target", "historical", "maturity", "incidental")])
        out += ["\n## P2. Year roles (parser qu-v2): questions with a non-reporting year, declared years read as "
                "non-reporting, and year mentions by role\n",
                _table(["Set", "Questions", "With a non-reporting year", "Declared year given a non-reporting role",
                        "fiscal", "target", "historical", "maturity", "incidental"], rows)]
    rows = [[name, f"{d['filing_year_kept']}/{d['questions_with_a_year']} ({_pct(d['kept_share'])})",
             _pct(d["mean_years_allowed_share"])]
            for name in [k for k in WINDOW_POLICIES if k in window_choice["policies"]]
            for d in [window_choice["policies"][name]]]
    out += [f"\n## W1. Fiscal-year window candidates on the {window_choice['questions']} training and validation "
            "questions (the positive passage's filing year kept; never measured on the real questions)\n",
            _table(["Window", "Filing year kept", "Mean share of the company's fiscal years allowed"], rows)]
    rows = []
    for e in order:
        if e not in summary or roles[e] not in ("baseline", "primary"):
            continue
        for seed in [s for s in manifest["experiments"][e]["seeds"] if s in summary[e]]:
            x = summary[e][seed]
            rows.append([e, seed, _f(x["validation"]["mrr@10"]["mean"]), _f(x["validation"]["recall@10"]["mean"]),
                         _f(x["validation"]["ndcg@10"]["mean"]), _f(x["development"]["mrr@10"]["mean"]),
                         _f(x["development"]["recall@10"]["mean"])])
    out += ["\n## Q1. Baseline and primary experiments per seed\n",
            _table(["Experiment", "Seed", "Validation MRR@10", "Validation Recall@10", "Validation nDCG@10",
                    "Development MRR@10", "Development Recall@10"], rows)]
    rows = []
    for e in ("E1_a1", "R_bm25", "E4_hybrid_rrf", "E6_a1_enriched", "E8_a1_tail_windows", "X_a1_sliding256",
              "E7_combined"):
        if e not in summary:
            continue
        bands = [x["validation_mrr_by_evidence_start"] for x in summary[e].values()]
        rows.append(label(e) + [f"{_across([b[k]['mrr@10'] for b in bands])} (n {bands[0][k]['n']})"
                                for _, _, k in POSITION_BANDS])
    out += ["\n## V2. Validation MRR@10 by where the evidence starts in its positive passage (encoder tokens; "
            "every validation evidence span ends inside the 510-token window)\n",
            _table(["Experiment", "Role", "Seeds"] + [f"Evidence starts at token {k}" for _, _, k in POSITION_BANDS],
                   rows)]
    def check_order(key: str) -> tuple[int, int]:
        eid, seed = key.split(" [")[0], key.split("[")[1].split("]")[0]
        seeds_ = manifest["experiments"].get(eid, {}).get("seeds", [])
        return (order.index(eid) if eid in order else len(order), seeds_.index(seed) if seed in seeds_ else 0)

    if "reference_results" in manifest:
        ref = manifest["reference_results"]
        rows = []
        for e in order:
            if e not in summary or "paired_vs_reference" not in next(iter(summary[e].values())):
                continue
            entries = summary[e].values()

            def ref_delta(key: str, entries: Any = entries) -> str:
                diffs = [(x["paired_vs_reference"][key]["mean_difference"], x["paired_vs_reference"][key]["p_value"])
                         for x in entries]
                mean = sum(d for d, _ in diffs) / len(diffs)
                return f"{mean:+.3f} (p ≤ {max(p for _, p in diffs):.4f})"

            changed = [x["top10_changed_vs_reference"] for x in entries]
            rows.append(label(e) + [ref_delta("validation | mrr@10"), ref_delta("validation | recall@100"),
                                    ref_delta("development | mrr@10"), ref_delta("real | exact_year_share")]
                        + [_count_across([c[s] for c in changed]) for s in ("validation", "development", "real")])
        out += [f"\n## R1. This run minus the reference run `{ref['path']}` (parser {ref['parser_version']}), same "
                "experiment and seed: mean over seeds (largest p over seeds), and questions whose top 10 changed\n",
                _table(["Experiment", "Role", "Seeds", "Δ validation MRR@10", "Δ validation Recall@100",
                        "Δ development MRR@10", "Δ real exact-year share", "Validation top 10 changed",
                        "Development top 10 changed", "Real top 10 changed"], rows)]
    rows = [[k, manifest["baseline_preservation"][k]] for k in sorted(manifest["baseline_preservation"], key=check_order)]
    out += ["\n## B1. Baseline preservation: v2 runs against the frozen v1 analysis (validation)\n",
            _table(["Check", "Result"], rows)]
    return "".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
