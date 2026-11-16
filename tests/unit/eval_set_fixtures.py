"""Synthetic corpus and helpers shared by the gold-set annotation tests.

Two invented companies (ACME, a device maker; BETA, a bank) with passages
shaped like the real corpus: Item sections, an appended-report section, and
table passages. Nothing here is real filing text.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from finretriever.data.schemas import (
    Passage,
    Qrel,
    Query,
    make_filing_id,
    make_passage_id,
    make_section_id,
)
from finretriever.eval_set.ai_proposals import ProposalSet
from finretriever.eval_set.annotation_store import (
    AnnotationStore,
    load_corpus_passages,
    verify_corpus,
)
from finretriever.eval_set.audit import REPORT_FILE, AuditWorkspace, audit_agreement, create_audit
from finretriever.eval_set.dev_labels import DevLabelSet, DevLabelSpec, derive_dev_labels
from finretriever.eval_set.dev_pool import (
    DevPoolSpec,
    ExpandedLabelSpec,
    create_pool_proposals,
    derive_expanded_labels,
    expand_pool,
    read_annotation_pool,
)
from finretriever.eval_set.schemas import Candidate, CandidateSource
from finretriever.evaluation.provenance import CorpusIdentity
from finretriever.evaluation.stored_runs import load_stored_run

FIXTURE_VERSION = "fixture_corpus@1.2"

_FILINGS = {
    "acme_2024": ("ACME", "Acme Devices Inc.", "0000000001-25-000001", "2025-02-20", "2024-12-31"),
    "acme_2023": ("ACME", "Acme Devices Inc.", "0000000001-24-000001", "2024-02-21", "2023-12-31"),
    "beta_2024": ("BETA", "Beta Bancorp", "0000000002-25-000003", "2025-02-27", "2024-12-31"),
}

_PASSAGES = [
    ("acme24_supplier", "acme_2024", 0, "Risk Factors", "1A",
     "We depend on a single supplier in Taiwan for advanced semiconductor components. Any disruption at "
     "this supplier could delay shipments and reduce net sales."),
    ("acme24_competition", "acme_2024", 0, "Risk Factors", "1A",
     "Our products compete on price and performance. Competitors may introduce lower-priced devices that "
     "reduce our gross margin."),
    ("acme24_sales", "acme_2024", 1, "Management's Discussion and Analysis", "7",
     "Net sales increased 20% to $1,200 million in 2024, driven by higher unit volumes of the Widget Pro "
     "line and price increases."),
    ("acme24_table", "acme_2024", 2, "Financial Statements and Supplementary Data", "8",
     "[TABLE]\nNet sales | 2024: $1,200 | 2023: $1,000\nGross margin | 2024: $480 | 2023: $390\n[/TABLE]"),
    ("acme24_liquidity", "acme_2024", 1, "Management's Discussion and Analysis", "7",
     "We had $350 million of cash and a $500 million revolving credit facility, of which none was drawn "
     "at December 31, 2024."),
    ("acme23_supplier", "acme_2023", 0, "Risk Factors", "1A",
     "We rely on one supplier for semiconductor components; a disruption could reduce net sales."),
    ("acme23_sales", "acme_2023", 1, "Management's Discussion and Analysis", "7",
     "Net sales increased 5% to $1,000 million in 2023."),
    ("acme23_table", "acme_2023", 2, "Financial Statements and Supplementary Data", "8",
     "[TABLE]\nNet sales | 2023: $1,000 | 2022: $952\n[/TABLE]"),
    ("beta24_liquidity", "beta_2024", 3, "LIQUIDITY RISK MANAGEMENT", None,
     "Beta maintains a liquidity buffer of high-quality liquid assets and a $2 billion committed credit "
     "facility to meet funding obligations."),
    ("beta24_credit", "beta_2024", 4, "CREDIT RISK MANAGEMENT", None,
     "Credit losses rose as consumer loan delinquencies increased; the allowance for credit losses was "
     "$310 million."),
    ("beta24_rates", "beta_2024", 0, "Risk Factors", "1A",
     "Rising interest rates could reduce the value of our securities portfolio and increase deposit costs."),
    ("beta24_business", "beta_2024", 1, "Business", "1",
     "Beta Bancorp operates consumer and commercial banking businesses in the United States."),
]


def fixture_passages() -> list[Passage]:
    passages: list[Passage] = []
    chunks: dict[tuple[str, int], int] = {}
    for _, key, section_index, title, item, text in _PASSAGES:
        ticker, company, accession, filed, report_date = _FILINGS[key]
        filing_id = make_filing_id(ticker, "10-K", accession)
        section_id = make_section_id(filing_id, title, section_index)
        chunk = chunks.get((key, section_index), 0)
        chunks[key, section_index] = chunk + 1
        metadata: dict[str, Any] = {
            "chunk_index": chunk,
            "section_title": title,
            "accession_number": accession,
            "company_name": company,
            "report_date": report_date,
            "section_index": section_index,
            "section_kind": "item" if item else "appended_report",
            "contains_table": "[TABLE]" in text,
        }
        if item:
            metadata["item_number"] = item
        passages.append(
            Passage(
                passage_id=make_passage_id(filing_id, section_id, chunk),
                filing_id=filing_id,
                section_id=section_id,
                ticker=ticker,
                form_type="10-K",
                filing_date=filed,
                section_title=title,
                text=text,
                token_count=len(text.split()),
                metadata=metadata,
            )
        )
    return passages


def write_corpus(directory: Path, passages: list[Passage] | None = None, *, version: str = FIXTURE_VERSION) -> Path:
    """Write a corpus JSONL plus a manifest that records its SHA-256, like the corpus builder."""
    passages = fixture_passages() if passages is None else passages
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / "corpus.jsonl"
    corpus.write_text("".join(p.model_dump_json() + "\n" for p in passages), encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "corpus_version": version,
        "corpus": {
            "name": "fixture_corpus",
            "path": str(corpus),
            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "bytes": corpus.stat().st_size,
            "passages": len(passages),
        },
    }
    corpus.with_name("corpus.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return corpus


def write_config(path: Path, corpus: Path, **overrides: Any) -> Path:
    """A gold-set config pinned to ``corpus``'s current SHA-256 and the fixture version."""
    manifest = json.loads(corpus.with_name("corpus.manifest.json").read_text())
    values: dict[str, Any] = {
        "name": "gold_set_test",
        "annotation_version": "gold_test",
        "corpus_path": str(corpus),
        "corpus_version": manifest["corpus_version"],
        "corpus_sha256": manifest["corpus"]["sha256"],
        "workspace_dir": str(path.parent / "annotation"),
        "export_dir": str(path.parent / "export"),
        "target_queries": 100,
        "bm25_depth": 5,
        "stratified_per_filing": 0,
        "random_candidates": 0,
        "seed": 7,
        "small_pool_below": 3,
        "ai_proposals_dir": str(path.parent / "dev_ai"),
        "audit_dir": str(path.parent / "audit"),
        "audit_sample_size": 6,
        "audit_seed": 11,
        "ai_annotators": ["claude", "claude-opus-5-5"],
    }
    values.update(overrides)
    path.write_text("".join(f"{key}: {json.dumps(value)}\n" for key, value in values.items()), encoding="utf-8")
    return path


def fixed_clock() -> Callable[[], str]:
    """Deterministic timestamps: 2026-01-01T00:00:01+00:00, ...:02, ..."""
    counter = itertools.count(1)

    def now() -> str:
        n = next(counter)
        return f"2026-01-01T00:{n // 60:02d}:{n % 60:02d}+00:00"

    return now


def new_store(tmp_path: Path, **kwargs: Any) -> AnnotationStore:
    corpus = verify_corpus(write_corpus(tmp_path / "corpus"))
    return AnnotationStore.create(
        tmp_path / "annotation", corpus, annotation_version="gold_test", created_by="T1", clock=fixed_clock(), **kwargs
    )


def reopen(store: AnnotationStore) -> AnnotationStore:
    return AnnotationStore.open(store.directory, store.corpus, clock=fixed_clock())


def passage_ids() -> dict[str, str]:
    """Fixture passage IDs by short name (``acme24_supplier``, ``beta24_credit``, ...)."""
    return {entry[0]: p.passage_id for entry, p in zip(_PASSAGES, fixture_passages(), strict=True)}


# --- development-label inputs: workspace, AI proposals and a finished audit ------------------

DEV_AI_GRADES = {
    "q_0001": {"acme24_supplier": 2, "acme23_supplier": 0, "acme24_sales": 1, "acme24_liquidity": 0, "acme24_table": 0},
    "q_0002": {"beta24_liquidity": 2, "beta24_credit": 1, "beta24_rates": 0, "beta24_business": 0},
}
# Audit grades: three disagreements, each handled differently by the spec below.
DEV_AUDIT_GRADES = {
    ("q_0001", "acme24_supplier"): 2,
    ("q_0001", "acme23_supplier"): 1,  # other filing year: kept under P4
    ("q_0001", "acme24_sales"): 2,  # open question, AI 1: flagged under P1
    ("q_0001", "acme24_liquidity"): 0,
    ("q_0001", "acme24_table"): 0,
    ("q_0002", "beta24_liquidity"): 1,  # no "December 31, 2024" in the passage: changed under P2
    ("q_0002", "beta24_credit"): 1,
    ("q_0002", "beta24_rates"): 0,
    ("q_0002", "beta24_business"): 0,
}


@dataclass
class DevInputs:
    config: Path
    store: AnnotationStore
    proposals: ProposalSet
    audit: AuditWorkspace
    audit_items: dict[tuple[str, str], int]  # (query_id, short name) -> audit item number

    def spec(self, **overrides: Any) -> dict[str, Any]:
        root = self.config.parent
        values: dict[str, Any] = {
            "label_version": "test_v2",
            "annotation_version": "dev_test_v2",
            "description": "fixture proposals under rubric v1.1",
            "base_proposals_dir": str(root / "dev_ai"),
            "audit_dir": str(root / "audit"),
            "audit_independence": "ai_assisted",
            "audit_independence_note": "the fixture auditor consulted another model",
            "output_dir": str(root / "dev_ai_v2"),
            "rubric": {
                "version": "v1.1",
                "adopted": "2026-10-28",
                "decisions": {"P1": "accepted", "P2": "accepted", "P3": "accepted", "P4": "A", "P5": "A", "P6": "accepted"},
            },
            "p2_basis": [
                {"query_id": "q_0002", "requires_any": ["December 31, 2024"], "why": "the question asks about year-end 2024"}
            ],
            "p1_open_questions": ["q_0001"],
            "flags": [],
            "kept_audit_disagreements": [
                {"item": self.audit_items["q_0001", "acme23_supplier"], "rule": "P4", "reason": "other filing year"}
            ],
        }
        values.update(overrides)
        return values

    def write_spec(self, **overrides: Any) -> Path:
        path = self.config.parent / "dev_labels.yaml"
        path.write_text(yaml.safe_dump(self.spec(**overrides), sort_keys=False), encoding="utf-8")
        return path


def build_dev_inputs(tmp_path: Path, extra_passages: Sequence[Passage] = ()) -> DevInputs:
    """A workspace with one human grade, AI proposals, and a finished audit with its report.

    ``extra_passages`` are appended to the corpus (after the fixture passages); nothing pools them.
    """
    ids = passage_ids()
    config = write_config(tmp_path / "gold.yaml", write_corpus(tmp_path / "corpus", [*fixture_passages(), *extra_passages]))
    corpus = verify_corpus(tmp_path / "corpus" / "corpus.jsonl")
    store = AnnotationStore.create(
        tmp_path / "annotation", corpus, annotation_version="gold_test", created_by="T1", clock=fixed_clock()
    )
    common = {"authoring": "independent", "author": "T1"}
    store.add_question(
        "How dependent is Acme on one supplier?", topic="risk_factors", question_type="semantic",
        tickers=["ACME"], fiscal_years=[2024], **common,
    )
    store.add_question(
        "How large was Beta's committed credit facility at December 31, 2024?", topic="debt_liquidity",
        question_type="exact_fact", tickers=["BETA"], fiscal_years=[2024], **common,
    )
    source = CandidateSource(method="bm25", rank=1, score=1.0, detail="query='x'")
    pools = {"q_0001": [*DEV_AI_GRADES["q_0001"], "acme24_competition"], "q_0002": [*DEV_AI_GRADES["q_0002"], "acme24_table"]}
    for query_id, names in pools.items():
        store.add_candidates(query_id, [Candidate(passage_id=ids[n], source=source) for n in names], added_by="T1")
    store.judge("q_0001", ids["acme24_competition"], 0, annotator="RV")  # human grade; acme24_table stays unlabelled
    proposals = ProposalSet.create(
        tmp_path / "dev_ai", store, model="claude-opus-5-5", method="test-read",
        method_description="read each passage in full", created_by="claude-opus-5-5", clock=fixed_clock(),
    )
    proposals.add(
        [
            {
                "query_id": q, "passage_id": ids[name], "grade": grade, "confidence": "high",
                "explanation": f"EXPLANATION-{name}", "model": "claude-opus-5-5", "method": "test-read",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            for q, grades in DEV_AI_GRADES.items()
            for name, grade in grades.items()
        ]
    )
    audit = create_audit(
        tmp_path / "audit", store, proposals, size=9, seed=11, created_by="claude-opus-5-5",
        annotation_version="gold_test-audit", clock=fixed_clock(),
    )
    names = {pid: name for name, pid in ids.items()}
    items = {}
    for item in audit.items():
        key = (item.query_id, names[item.passage_id])
        audit.store.judge(item.query_id, item.passage_id, DEV_AUDIT_GRADES[key], annotator="RV")
        items[key] = item.item
    audit = AuditWorkspace.open(tmp_path / "audit", corpus)
    report = audit_agreement(audit, proposals)
    (audit.directory / REPORT_FILE).write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return DevInputs(config, store, proposals, audit, items)


def write_dev_eval_set(tmp_path: Path) -> dict[str, Any]:
    """A provisional development evaluation set (queries, qrels, metadata, report) plus a benchmark config."""

    ids = passage_ids()
    corpus = write_corpus(tmp_path / "corpus")
    manifest = json.loads(corpus.with_name("corpus.manifest.json").read_text())
    labels = tmp_path / "labels"
    labels.mkdir()
    status = {"status": "provisional_development"}
    queries = [
        Query(query_id="q_0001", text="Which single supplier does Acme depend on?", source="dev_ai",
              metadata={"tickers": "ACME", **status}),
        Query(query_id="q_0002", text="How large is Beta's committed credit facility?", source="dev_ai",
              metadata={"tickers": "BETA", **status}),
    ]
    qrels = [
        Qrel(query_id="q_0001", passage_id=ids["acme24_supplier"], relevance=2),
        Qrel(query_id="q_0001", passage_id=ids["acme23_supplier"], relevance=0),
        Qrel(query_id="q_0002", passage_id=ids["beta24_liquidity"], relevance=2),
        Qrel(query_id="q_0002", passage_id=ids["beta24_rates"], relevance=0),
    ]
    for name, rows in (("dev_queries.jsonl", queries), ("dev_qrels.jsonl", qrels)):
        (labels / name).write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    metadata = {"corpus_sha256": manifest["corpus"]["sha256"], "corpus_version": manifest["corpus_version"],
                "annotation_version": "dev_test_v2"}
    (labels / "dev_eval.metadata.json").write_text(json.dumps(metadata))
    report = {
        **status, "label_version": "test_v2",
        "labels": {"qrels": 4, "by_source": {"ai_proposal": 4}, "open_flags_by_rule": {}},
        "audit": {"independence": "ai_assisted"}, "exceptions": {"no_positive": [], "no_grade_2": []},
        "files": {name: hashlib.sha256((labels / name).read_bytes()).hexdigest()
                  for name in ("dev_queries.jsonl", "dev_qrels.jsonl")},
    }
    (labels / "dev_eval.report.json").write_text(json.dumps(report))
    config = {
        "corpus_path": str(corpus),
        "metrics": ["ndcg@10", "mrr@10", "recall@100"],
        "top_k": 5,
        "judged_k": 10,
        "coverage_thresholds": {"min_positives": 1, "min_judged_at_10": 0.7},
        "baselines": [
            {"name": "bm25", "type": "bm25", "tokenizer": "word", "backend": "python"},
            {"name": "bm25_whitespace", "type": "bm25", "tokenizer": "whitespace", "backend": "python",
             "diagnostic": True},
            # Never loads in tests: the import is refused in-process (tests/conftest.py) and script
            # tests run offline with an empty Hugging Face cache, so it is reported as skipped.
            {"name": "bge_zero_shot", "type": "dense", "model_name": "BAAI/bge-base-en-v1.5",
             "model_revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", "local_files_only": True},
        ],
        "evaluation_sets": {
            "dev": {
                **status,
                "queries_path": str(labels / "dev_queries.jsonl"),
                "qrels_path": str(labels / "dev_qrels.jsonl"),
                "metadata_path": str(labels / "dev_eval.metadata.json"),
                "report_path": str(labels / "dev_eval.report.json"),
                "output_dir": str(tmp_path / "results" / "dev" / "test_v2"),
            }
        },
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(json.dumps(config))  # JSON is valid YAML
    return {"config": config, "config_path": config_path, "labels": labels, "tmp": tmp_path}


def write_stored_run(
    runs_dir: Path,
    mode: str,
    run: str,
    ranking: dict[str, list[str]],
    *,
    corpus_path: Path,
    top_k: int = 10,
    label_version: str = "earlier_labels",
    per_query: list[dict[str, Any]] | None = None,
) -> Path:
    """A stored run as ``write_eval_set_outputs`` leaves it: ``run.jsonl`` and a ``metrics.json`` pinning it.

    ``ranking`` maps each question to fixture passage short names in rank order.
    """
    ids = passage_ids()
    tickers = {p.passage_id: p.ticker for p in fixture_passages()}
    manifest = json.loads(corpus_path.with_name("corpus.manifest.json").read_text())
    rows = [
        {"query_id": query_id, "rank": rank, "passage_id": ids[name], "score": 1.0 - rank / 100,
         "ticker": tickers[ids[name]], "grade": None, "judged": False}
        for query_id, names in ranking.items()
        for rank, name in enumerate(names, start=1)
    ]
    data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    out = runs_dir / mode / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.jsonl").write_bytes(data)
    metrics = {
        "eval_set": "earlier", "run": run, "mode": mode, "label_version": label_version,
        "corpus": {"sha256": manifest["corpus"]["sha256"], "version": manifest["corpus_version"]},
        "settings": {"top_k": top_k, "judged_k": 10},
        "retriever": {"name": run, "type": "dense", "model_name": "fake/model", "model_revision": "0" * 40},
        "per_query": per_query or [],
        "run_file": {"path": "run.jsonl", "sha256": hashlib.sha256(data).hexdigest()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return out


# --- an expanded pool with proposals and derived labels (pilot_v3 shape) ------------------------

EXPANDED_DENSE = {
    "q_0001": ["acme24_supplier", "acme23_sales", "acme23_table", "beta24_credit"],
    "q_0002": ["beta24_liquidity", "acme24_table", "acme23_table", "beta24_business"],
}
EXPANDED_BM25 = {"q_0001": ["acme24_supplier", "acme24_sales"], "q_0002": ["beta24_liquidity", "beta24_rates"]}
# New proposals: (question, passage, grade, confidence); every unjudged pool entry gets one.
EXPANDED_PROPOSALS = [
    ("q_0001", "acme23_sales", 1, "low"),
    ("q_0001", "acme23_table", 0, "medium"),
    ("q_0001", "beta24_credit", 0, "high"),
    ("q_0002", "acme24_table", 0, "high"),
    ("q_0002", "acme23_table", 0, "high"),
]


@dataclass
class ExpandedInputs:
    dev: DevInputs
    root: Path
    corpus: CorpusIdentity
    passages: dict[str, Passage]
    labels_dir: Path
    pool_dir: Path
    proposals_dir: Path


def build_expanded_inputs(tmp_path: Path, extra_passages: Sequence[Passage] = ()) -> ExpandedInputs:
    """``build_dev_inputs`` plus base labels, two stored runs, an expanded pool, its proposals and labels."""
    dev = build_dev_inputs(tmp_path, extra_passages)
    corpus = verify_corpus(tmp_path / "corpus" / "corpus.jsonl")
    passages = load_corpus_passages(corpus)
    ids = passage_ids()
    base = derive_dev_labels(
        DevLabelSpec.model_validate(dev.spec()), spec_path="dev_labels.yaml", spec_sha256="0" * 64,
        store=dev.store, proposals=dev.proposals, audit=dev.audit,
    )
    base.write(tmp_path / "dev_ai_v2")
    base = DevLabelSet.open(tmp_path / "dev_ai_v2", corpus, passages)
    runs_dir = tmp_path / "runs"
    write_stored_run(runs_dir, "global", "dense", EXPANDED_DENSE, corpus_path=corpus.corpus_path, top_k=100)
    write_stored_run(runs_dir, "company", "bm25", EXPANDED_BM25, corpus_path=corpus.corpus_path, top_k=100)
    pool_dir = tmp_path / "dev_ai_v3" / "pool"
    pool_spec = DevPoolSpec(
        pool_version="test_v3", annotation_version="dev_test_v3", description="fixture expansion",
        base_labels_dir=str(tmp_path / "dev_ai_v2"), base_pool_path=str(tmp_path / "annotation" / "pool.jsonl"),
        runs_dir=str(runs_dir), runs=[{"run": "dense", "mode": "global"}, {"run": "bm25", "mode": "company"}],
        depth=10, output_dir=str(pool_dir),
    )
    runs = [load_stored_run(runs_dir, r.mode, r.run, corpus=corpus, passages=passages) for r in pool_spec.runs]
    pool = expand_pool(
        pool_spec, spec_path="dev_pool.yaml", spec_sha256="1" * 64, base=base,
        base_pool=read_annotation_pool(tmp_path / "annotation" / "pool.jsonl"), runs=runs, passages=passages,
    )
    pool.write()
    proposals_dir = tmp_path / "dev_ai_v3" / "proposals"
    proposals = create_pool_proposals(
        proposals_dir, pool, model="claude-opus-5-5", method="test-read", method_description="read in full",
        created_by="claude-opus-5-5", clock=fixed_clock(),
    )
    proposals.add(
        [
            {"query_id": q, "passage_id": ids[name], "grade": grade, "confidence": confidence,
             "explanation": f"EXPLANATION-{name}", "model": "claude-opus-5-5", "method": "test-read",
             "created_at": "2026-01-01T00:00:00+00:00"}
            for q, name, grade, confidence in EXPANDED_PROPOSALS
        ]
    )
    base_spec = dev.spec()
    label_spec = ExpandedLabelSpec(
        label_version="test_v3", annotation_version="dev_test_v3", description="fixture expanded labels",
        base_labels_dir=str(tmp_path / "dev_ai_v2"), pool_dir=str(pool_dir), proposals_dir=str(proposals_dir),
        output_dir=str(tmp_path / "dev_ai_v3"), rubric=base_spec["rubric"], p2_basis=base_spec["p2_basis"],
        p1_open_questions=base_spec["p1_open_questions"],
    )
    labels = derive_expanded_labels(
        label_spec, spec_path="dev_labels_v3.yaml", spec_sha256="2" * 64, base=base, pool=pool,
        proposals=proposals, passages=passages,
    )
    labels.write(tmp_path / "dev_ai_v3")
    return ExpandedInputs(dev, tmp_path, corpus, passages, tmp_path / "dev_ai_v3", pool_dir, proposals_dir)
