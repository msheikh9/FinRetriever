#!/usr/bin/env python
"""Search demo: find the 10-K passages that answer a financial research question, with traceable evidence.

    python scripts/12_search_demo.py --question "How much long-term debt did Apple report at the end of FY2022?"
    python scripts/12_search_demo.py                      # interactive: one question per line, empty line to quit
    python scripts/12_search_demo.py --json --question "..."   # one JSON object per question

For each question it prints:

* what query understanding read from it (parser ``qu-v2``): the companies, the fiscal (reporting) years, and any
  year it recognised as something else (a target, maturity, historical or incidental year), with the measures and
  the comparison type;
* what the metadata scope decided (company filter, fiscal-year boost, fallbacks);
* the top passages: company, form and fiscal year of the filing, filing date, section, whether the fiscal year is
  the requested one, the passage's rank in BM25 and in the dense model before fusion, the passage and accession
  numbers, and the filing's address in SEC EDGAR, then the start of the passage text.

It retrieves evidence; it never writes an answer. When the top passages do not come from the requested company or
fiscal year, it says so instead of presenting them as the answer.

Default retriever: ``E5_hybrid_rrf_meta`` of configs/experiments/v2_1_year_roles.yaml -- BM25 and the fine-tuned
A1 model fused by Reciprocal Rank Fusion, a company filter, and a fiscal-year *boost* (no hard year filter). Any
experiment of a v2 config and any A1 seed can be chosen.

Needs the corpus (rebuilt from SEC EDGAR with configs/data/corpus_v1_pinned.yaml) and, for the default retriever,
the A1 checkpoints; dense embeddings are computed on first use and cached. Without checkpoints,
``--config configs/demo/public_bm25.yaml`` runs BM25 with the same metadata-aware scope. Setup: docs/demo.md.
Read-only apart from the embedding cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finretriever.data.schemas import Passage  # noqa: E402
from finretriever.retrieval.enrichment import passage_fiscal_year, section_label  # noqa: E402
from finretriever.retrieval.hybrid import HybridRetriever  # noqa: E402
from finretriever.retrieval.pipeline import (  # noqa: E402
    build_bm25,
    build_seed_components,
    experiment_config,
)
from finretriever.retrieval.query_understanding import QueryParser  # noqa: E402
from finretriever.utils.io import read_yaml  # noqa: E402

COMPONENT_LABELS = {"bm25": "BM25", "bm25_enriched": "BM25 (enriched)", "a1": "A1 dense",
                    "a1_enriched": "A1 dense (enriched)", "a1_tail": "A1 dense (tail windows)",
                    "a1_sliding256": "A1 dense (256-token windows)"}
WIDTH = 100


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def edgar_url(passage: Passage) -> str | None:
    """The filing's folder in the SEC EDGAR archive, from its CIK and accession number (no request is made)."""
    cik, accession = passage.metadata.get("cik"), passage.metadata.get("accession_number")
    if not cik or not accession:
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{int(str(cik))}/{str(accession).replace('-', '')}/"


def describe_retriever(experiment: str, config: Any, seed: str, parser: QueryParser) -> str:
    parts = [" + ".join(COMPONENT_LABELS.get(c, c) for c in config.components)]
    if config.fusion.method == "rrf":
        parts.append(f"Reciprocal Rank Fusion (k={config.fusion.k})")
    policy = config.policy
    if policy.company == "filter":
        parts.append("company filter")
    if policy.year != "none":
        parts.append(f"fiscal-year {policy.year}")
    if policy.cells == "interleave":
        parts.append("company-year cells")
    uses_a1 = any(not c.startswith("bm25") for c in config.components)
    return f"{experiment}: {', '.join(parts)}{f' · A1 {seed}' if uses_a1 else ''} · parser {parser.version}"


def wrap(text: str, indent: str, rest: str | None = None) -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent=indent,
                         subsequent_indent=rest if rest is not None else indent)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/experiments/v2_1_year_roles.yaml")
    ap.add_argument("--experiment", default=None,
                    help="default: E5_hybrid_rrf_meta if the config has it, else the config's first experiment")
    ap.add_argument("--seed", default="a1_seed42")
    ap.add_argument("--question", action="append", default=[], help="repeatable; omit for interactive mode")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chars", type=int, default=240, help="characters of passage text to show")
    ap.add_argument("--json", action="store_true", help="print one JSON object per question instead of text")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    ids = [e["id"] for e in cfg["experiments"]]
    args.experiment = args.experiment or ("E5_hybrid_rrf_meta" if "E5_hybrid_rrf_meta" in ids else ids[0])
    entry = next((e for e in cfg["experiments"] if e["id"] == args.experiment), None)
    if entry is None:
        raise SystemExit(f"unknown experiment {args.experiment!r}; choose one of "
                         f"{[e['id'] for e in cfg['experiments']]}")
    seeded_components = [c for c in entry["components"] if cfg["components"][c]["type"] != "bm25"]
    if seeded_components and args.seed not in cfg["seeds"]:
        raise SystemExit(f"unknown seed {args.seed!r}; choose one of {list(cfg['seeds'])}")
    if not Path(cfg["corpus_path"]).exists():
        raise SystemExit(
            f"The corpus is not at {cfg['corpus_path']}. It is not in git; rebuild it from SEC EDGAR (see docs/demo.md, "
            "'Setup'):\n  SEC_USER_AGENT=\"<name> <email>\" make download CORPUS_CONFIG=configs/data/corpus_v1_pinned.yaml"
            "\n  make corpus CORPUS_CONFIG=configs/data/corpus_v1_pinned.yaml")
    if sha256_file(cfg["corpus_path"]) != cfg["corpus_sha256"]:
        raise SystemExit(f"{cfg['corpus_path']} differs from the configured corpus (SHA-256 {cfg['corpus_sha256'][:12]}…); "
                         "rebuild it with configs/data/corpus_v1_pinned.yaml from an empty data/raw (docs/demo.md)")
    if seeded_components:
        checkpoint = Path(cfg["seeds"][args.seed]["checkpoint_path"])
        if not checkpoint.exists():
            raise SystemExit(
                f"The fine-tuned A1 checkpoint is not at {checkpoint}. Checkpoints are not in git (docs/demo.md, 'Setup'). "
                "Without them, run the public demo, which needs only the corpus:\n"
                "  python scripts/12_search_demo.py --config configs/demo/public_bm25.yaml --question \"...\"")
    passages = [Passage(**json.loads(line)) for line in open(cfg["corpus_path"], encoding="utf-8")]
    position = {p.passage_id: i for i, p in enumerate(passages)}
    aliases = (read_yaml(cfg["parser"]["aliases_from"]).get("generation") or {}).get("company_aliases") or {}
    parser = QueryParser.from_passages(passages, aliases, version=cfg["parser"].get("version", "qu-v1"))
    config = experiment_config(entry, cfg["defaults"])
    if config.policy.source == "oracle":
        raise SystemExit(f"{args.experiment} is a reference run that needs annotated scopes, not a retriever")

    quiet = (lambda _msg: None) if args.json else (lambda msg: print(f"  {msg}", file=sys.stderr))
    scorers: dict[str, Any] = {}
    seeded = []
    for name in config.components:
        spec = cfg["components"][name]
        if spec["type"] == "bm25":
            scorers[name], _ = build_bm25(name, spec, passages)
        else:
            seeded.append(name)
    if seeded:
        built, _ = build_seed_components(args.seed, seeded, cfg, passages, position, {}, quiet)
        scorers.update(built)
    retriever = HybridRetriever(config, scorers, passages, parser)
    tickers = sorted({p.ticker for p in passages})
    years = sorted({passage_fiscal_year(p) for p in passages})
    filings = len({p.filing_id for p in passages})
    if not args.json:
        print(f"FinRetriever · {filings} 10-K filings, {len(passages):,} passages · {', '.join(tickers)} · "
              f"FY{years[0]}–FY{years[-1]}")
        print(wrap(describe_retriever(args.experiment, config, args.seed, parser), ""))
        print("Retrieves evidence passages from the filings; it does not write answers.\n")

    def search(question: str) -> None:
        hits, trace = retriever.search(question, top_k=args.top_k)
        why = retriever.explain(question, [h.passage_id for h in hits])
        scope = trace["scope"]
        if args.json:
            print(json.dumps({"question": question, "retriever": args.experiment, "seed": args.seed, "trace": trace,
                              "hits": [{**h.model_dump(), "explain": why[h.passage_id],
                                        "edgar_url": edgar_url(passages[position[h.passage_id]])} for h in hits]},
                             ensure_ascii=False))
            return
        print("─" * WIDTH)
        print(wrap(question, "Question  ", " " * 10))
        named = ", ".join(f"{t} ({surface})" for t, surface in scope["company_mentions"]) or "none named"
        print(f"Parsed    companies: {named} · fiscal years: {', '.join(map(str, scope['years'])) or 'none named'}")
        others = [f"{y} ({m['role']})" for m in scope["year_mentions"] if m.get("role") not in (None, "fiscal")
                  for y in m["years"]]
        if others:
            print(f"          other years, not used as reporting periods: {', '.join(others)}")
        print(f"          measures: {', '.join(scope['measures']) or '—'} · comparison: {scope['comparison']}")
        for decision in trace["plan"]["decisions"]:
            print(f"Scope     {decision}")
        if not hits:
            print("\nNo passage in the corpus matches this question's scope: no evidence to show.\n")
            return
        for h in hits:
            p = passages[position[h.passage_id]]
            info = why[h.passage_id]
            company = p.metadata.get("company_name") or p.ticker
            section = section_label(p)
            header = (f"{company} ({p.ticker}) · {p.form_type} for FY{info['fiscal_year']} · filed {p.filing_date} · "
                      f"{section}{' · table' if p.metadata.get('contains_table') else ''}")
            print("\n" + wrap(header, f"{h.rank:>2}  ", "    "))
            ranks = " · ".join(f"{COMPONENT_LABELS.get(n, n)} rank {r if r is not None else '—'}"
                               for n, r in info["component_ranks"].items())
            tier = f"fiscal year: {info['year_tier']} · " if info["year_tier"] else ""
            print(f"    {tier}{ranks} (of {info['eligible_passages']:,} in scope) · fused score {h.score:.4f}")
            print(f"    passage {p.passage_id} · accession {p.metadata.get('accession_number', '—')}")
            if url := edgar_url(p):
                print(f"    {url}")
            text = " ".join(p.text.split())
            print(wrap("“" + text[: args.chars] + ("…" if len(text) > args.chars else "") + "”", "    "))
        notes = []
        if not scope["tickers"]:
            notes.append(f"No corpus company was named, so all of {', '.join(tickers)} were searched.")
        if scope["years"] and not any(why[h.passage_id]["year_tier"] in ("requested year", "later filing")
                                      for h in hits):
            notes.append(f"None of these passages comes from a filing for FY{'/'.join(map(str, scope['years']))} "
                         f"or a later filing that repeats it (the corpus covers FY{years[0]}–FY{years[-1]}); "
                         "the evidence for that year may be missing.")
        for note in notes:
            print("\n" + wrap(note, "Note      ", " " * 10))
        print()

    if args.question:
        for q in args.question:
            search(q)
        return 0
    print(f"Ask a question about {', '.join(tickers)} 10-K filings (empty line to quit).")
    for line in sys.stdin:
        if not line.strip():
            break
        search(line.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
