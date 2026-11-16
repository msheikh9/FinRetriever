#!/usr/bin/env python
"""Retrieval Performance Analysis v1, supplementary checks (docs/retrieval_analysis_v1.md, section 6).

Reads the outputs of ``scripts/09_retrieval_analysis.py`` (run it first), the frozen corpus and the frozen
validation pairs. No retrieval is run and no relevance label of the real questions is read.

1. Company names in passage text: how many passages name their own company (case-insensitive patterns below);
   and, in each system's real-question top 10, whether wrong-company passages name their company less often than
   correct-company ones (a label-free probe of why a ranker drifts to another company).
2. Evidence position (validation): where each validation answer's evidence starts inside its positive passage,
   in BGE WordPiece tokens (the tokenizer the A1 checkpoints saved), and MRR@10 by position band.
3. Real questions whose top 10 is mostly (> 50%) from other companies, per system and mode.

    python scripts/10_retrieval_analysis_supplement.py [--analysis-dir results/analysis/retrieval_analysis_v1]

Writes ``supplement.json`` and ``supplement_tables.md`` into the analysis directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

CORPUS = ("data/processed/corpus_v1.jsonl", "ff9dadf37bb9820e2c87df70a411fd189bb02776609e5ec06173dd03f2344004")
VALIDATION = "data/processed/train/bge_base_ft_v3/validation_sets/validation_pairs_v4/validation_pairs.jsonl"
TRAINING = "data/processed/train/bge_base_ft_v3/candidates/train_candidates_v3/train_candidates.jsonl"  # A1's 678 pairs
TOKENIZER = "models/bge_base_ft_v3_a1/bge_base_ft_v3_a1-seed42/checkpoint-step-000055"
NAMES = {  # a passage "names its company" when its text matches its ticker's pattern
    "AAPL": r"\bapple\b",
    "AMZN": r"\bamazon",
    "JPM": r"\bjpmorgan|\bjp morgan",
    "PFE": r"\bpfizer",
    "TSLA": r"\btesla",
    "WMT": r"\bwalmart|\bwal-mart",
    "XOM": r"\bexxon",
}
BANDS = ((0, 128, "0-127"), (128, 256, "128-255"), (256, 10**9, "256+"))
WINDOW = 510  # 512-token passages minus [CLS] and [SEP]
SYSTEMS = ["bm25", "bge_zero_shot", "e5_zero_shot", "a0_step0", "a1_seed42", "a1_seed1337", "a1_seed2026"]
LABEL = {"bm25": "BM25", "bge_zero_shot": "BGE zero-shot", "e5_zero_shot": "E5 zero-shot", "a0_step0": "A0 (step 0)",
         "a1_seed42": "A1 seed 42", "a1_seed1337": "A1 seed 1337", "a1_seed2026": "A1 seed 2026"}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--analysis-dir", default="results/analysis/retrieval_analysis_v1")
    args = p.parse_args()
    out_dir = Path(args.analysis_dir)
    if sha256_file(CORPUS[0]) != CORPUS[1]:
        raise SystemExit("corpus SHA-256 differs from corpus_v1@1.2")
    corpus = {c["passage_id"]: c for c in jsonl(CORPUS[0])}
    names = {pid: re.search(NAMES[c["ticker"]], c["text"].lower()) is not None for pid, c in corpus.items()}

    # 1. company names in passage text
    by_ticker = defaultdict(lambda: [0, 0])
    tables = [0, 0]
    for pid, c in corpus.items():
        by_ticker[c["ticker"]][0] += names[pid]
        by_ticker[c["ticker"]][1] += 1
        if c["metadata"].get("contains_table"):
            tables[0] += names[pid]
            tables[1] += 1
    top10 = [r for r in jsonl(out_dir / "top10.jsonl") if r["set"] == "gold"]
    questions = {r["query_id"]: r for r in jsonl(out_dir / "alignment_per_question.jsonl")}
    named_in_top10: dict[str, Any] = {}
    for r in top10:
        asked = set(questions[r["query_id"]]["tickers"])
        d = named_in_top10.setdefault(r["system"], {}).setdefault(
            r["mode"], {"correct_company": [0, 0], "wrong_company": [0, 0]})
        for pid in r["top10"]:
            key = "correct_company" if corpus[pid]["ticker"] in asked else "wrong_company"
            d[key][0] += names[pid]
            d[key][1] += 1
    def named(ticker: str, text: str) -> bool:
        return re.search(NAMES[ticker], text.lower()) is not None

    train = jsonl(TRAINING)
    training_pairs = {
        "path": TRAINING,
        "sha256": sha256_file(TRAINING),
        "pairs": len(train),
        "question_names_company": sum(named(p["ticker"], p["query"]) for p in train),
        "positive_names_company": sum(named(p["ticker"], corpus[p["positive_passage_id"]]["text"]) for p in train),
        "question_names_it_positive_does_not": sum(
            named(p["ticker"], p["query"]) and not named(p["ticker"], corpus[p["positive_passage_id"]]["text"])
            for p in train),
    }
    names_section = {
        "patterns": NAMES,
        "a1_training_pairs": training_pairs,
        "passages": len(corpus),
        "naming_their_company": sum(names.values()),
        "share": sum(names.values()) / len(corpus),
        "table_passages": tables[1],
        "table_share": tables[0] / tables[1],
        "by_ticker": {t: {"passages": n, "share": k / n} for t, (k, n) in sorted(by_ticker.items())},
        "real_question_top10": {
            s: {m: {k: {"passages": n, "naming_share": (named / n if n else None)} for k, (named, n) in d.items()}
                for m, d in modes.items()}
            for s, modes in named_in_top10.items()
        },
    }

    # 2. evidence position inside the positive passage (validation)
    from transformers import AutoTokenizer  # local files only: the checkpoint saved its tokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    pairs = jsonl(VALIDATION)
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for pair in pairs:
        text = corpus[pair["positive_passage_id"]]["text"]
        evidence = pair["evidence"].strip()
        i = text.find(evidence[:60])
        if i < 0:
            continue
        starts[pair["pair_id"]] = len(tok(text[:i], add_special_tokens=False)["input_ids"])
        ends[pair["pair_id"]] = len(tok(text[: i + len(evidence)], add_special_tokens=False)["input_ids"])
    rows = [r for r in jsonl(out_dir / "validation_per_question.jsonl") if r["mode"] == "global"]
    position: dict[str, Any] = {
        "tokenizer": TOKENIZER,
        "questions": len(pairs),
        "evidence_located": len(starts),
        "start_quartiles": statistics.quantiles(starts.values(), n=4),
        "start_median": statistics.median(starts.values()),
        "evidence_ending_beyond_window": sum(e > WINDOW for e in ends.values()),
        "window_tokens": WINDOW,
        "mrr@10_by_band": {},
    }
    for s in SYSTEMS:
        per = {r["pair_id"]: r["mrr@10"] for r in rows if r["system"] == s}
        if not per:
            continue
        position["mrr@10_by_band"][s] = {}
        for lo, hi, label in BANDS:
            v = [per[q] for q, t in starts.items() if lo <= t < hi]
            position["mrr@10_by_band"][s][label] = {"n": len(v), "mrr@10": sum(v) / len(v) if v else None}

    # 3. real questions whose top 10 is mostly from other companies
    alignment = jsonl(out_dir / "alignment_per_question.jsonl")
    mostly_other = {s: {m: sum(r["cross_company_share"] > 0.5 for r in alignment if r["system"] == s and r["mode"] == m)
                        for m in ("global", "company")} for s in SYSTEMS}
    by_ticker_cross: dict[str, Any] = {}
    for s in SYSTEMS:
        acc: dict[str, list[float]] = defaultdict(list)
        for r in alignment:
            if r["system"] == s and r["mode"] == "global" and len(r["tickers"]) == 1:
                acc[r["tickers"][0]].append(r["cross_company_share"])
        by_ticker_cross[s] = {t: {"questions": len(v), "cross_company_share": sum(v) / len(v)} for t, v in sorted(acc.items())}

    result = {
        "kind": "finretriever_retrieval_analysis_supplement",
        "version": 1,
        "inputs": {
            "corpus": {"path": CORPUS[0], "sha256": CORPUS[1]},
            "validation_pairs": {"path": VALIDATION, "sha256": sha256_file(VALIDATION)},
            **{f: {"path": str(out_dir / f), "sha256": sha256_file(out_dir / f)}
               for f in ("results.json", "validation_per_question.jsonl", "alignment_per_question.jsonl", "top10.jsonl")},
        },
        "company_names": names_section,
        "evidence_position": position,
        "real_questions_mostly_other_companies": mostly_other,
        "real_questions_cross_company_share_by_ticker": by_ticker_cross,
    }
    (out_dir / "supplement.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (out_dir / "supplement_tables.md").write_text(render(result), encoding="utf-8")
    print(f"[10] wrote {out_dir}/supplement.json and supplement_tables.md")
    return 0


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def render(r: dict[str, Any]) -> str:
    n = r["company_names"]
    out = ["# Retrieval Analysis v1: supplementary tables\n"]
    rows = [[t, str(d["passages"]), _pct(d["share"])] for t, d in n["by_ticker"].items()]
    rows.append(["All", str(n["passages"]), _pct(n["share"])])
    rows.append(["All table passages", str(n["table_passages"]), _pct(n["table_share"])])
    out += ["\n## S1. Passages whose text names their own company\n", _table(["Company", "Passages", "Naming it"], rows)]
    t = n["a1_training_pairs"]
    rows = [["Questions naming the company", f"{t['question_names_company']}/{t['pairs']}"],
            ["Positive passages naming the company", f"{t['positive_names_company']}/{t['pairs']}"],
            ["Question names it, positive passage does not", f"{t['question_names_it_positive_does_not']}/{t['pairs']}"]]
    out += ["\n## S6. A1's training pairs (train_candidates_v3): company names in question and positive passage\n",
            _table(["Training pairs", "Count"], rows)]
    rows = []
    for s in SYSTEMS:
        for m in ("global", "company"):
            d = n["real_question_top10"].get(s, {}).get(m)
            if d:
                rows.append([LABEL[s], m, _pct(d["correct_company"]["naming_share"]), str(d["correct_company"]["passages"]),
                             _pct(d["wrong_company"]["naming_share"]), str(d["wrong_company"]["passages"])])
    out += ["\n## S2. Real-question top 10: share of passages naming their company\n",
            _table(["System", "Mode", "Correct-company passages naming it", "Correct-company passages",
                    "Wrong-company passages naming it", "Wrong-company passages"], rows)]
    p = r["evidence_position"]
    rows = []
    for s, bands in p["mrr@10_by_band"].items():
        rows.append([LABEL[s]] + [f"{d['mrr@10']:.3f} ({d['n']})" for d in bands.values()])
    out += [f"\n## S3. Validation MRR@10 (global) by where the evidence starts in its passage (BGE tokens; "
            f"{p['evidence_located']}/{p['questions']} located; median start {p['start_median']:.0f}; "
            f"{p['evidence_ending_beyond_window']} ending beyond {p['window_tokens']})\n",
            _table(["System"] + [f"Starts at {b[2]} (n)" for b in BANDS], rows)]
    rows = [[LABEL[s], str(d["global"]), str(d["company"])] for s, d in r["real_questions_mostly_other_companies"].items()]
    out += ["\n## S4. Real questions whose top 10 is mostly (> 50%) from other companies (of 100)\n",
            _table(["System", "Global", "Company"], rows)]
    tickers = sorted(next(iter(r["real_questions_cross_company_share_by_ticker"].values())))
    rows = []
    for s, d in r["real_questions_cross_company_share_by_ticker"].items():
        rows.append([LABEL[s]] + [_pct(d[t]["cross_company_share"]) for t in tickers])
    qn = next(iter(r["real_questions_cross_company_share_by_ticker"].values()))
    out += ["\n## S5. Real single-company questions: cross-company share of the top 10 by company asked (global)\n",
            _table(["System"] + [f"{t} ({qn[t]['questions']} q)" for t in tickers], rows)]
    return "".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
