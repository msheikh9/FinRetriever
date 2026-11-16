"""Synthetic passages, a fake tokenizer and a mocked Groq API for the training-data tests.

Nothing here is real filing text, and nothing touches the network: the mocked transport answers
from the prompt it receives.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from finretriever.data.schemas import Passage, make_filing_id, make_passage_id, make_section_id
from finretriever.training.query_generation import TransportResponse
from tests.unit.eval_set_fixtures import _PASSAGES, passage_ids

SECRET = "gsk_TEST_SECRET_never_print_0123456789"


def make_passage(
    name: str, ticker: str, company: str, report_date: str, text: str, *, section: str = "Management's Discussion and Analysis",
    section_index: int = 0, chunk: int = 0, accession: str | None = None, item: str | None = "7",
) -> Passage:
    digest = int(hashlib.sha256(f"{ticker}:{report_date}".encode()).hexdigest()[:8], 16)
    accession = accession or f"0000009999-{report_date[2:4]}-{digest % 900000 + 100000:06d}"
    filing_id = make_filing_id(ticker, "10-K", accession)
    section_id = make_section_id(filing_id, section, section_index)
    metadata: dict[str, Any] = {
        "chunk_index": chunk, "section_title": section, "accession_number": accession, "company_name": company,
        "report_date": report_date, "section_index": section_index, "section_kind": "item" if item else "appended_report",
        "contains_table": "[TABLE]" in text, "fixture_name": name,
    }
    if item:
        metadata["item_number"] = item
    return Passage(
        passage_id=make_passage_id(filing_id, section_id, chunk), filing_id=filing_id, section_id=section_id,
        ticker=ticker, form_type="10-K", filing_date=f"{int(report_date[:4]) + 1}-02-20", section_title=section,
        text=text, token_count=len(text.split()), metadata=metadata,
    )


def block(i: int, n: int = 12) -> str:
    """``n`` words found in no other block (for exact overlap arithmetic)."""
    return " ".join(f"blk{i}word{j}" for j in range(n))


def chain_texts() -> tuple[str, str, str]:
    """A ~ B (73% of shingles shared) and B ~ C (73%), but A and C share only 45%: one group by transitivity."""
    return (" ".join(block(i) for i in (1, 2, 3, 4)), " ".join(block(i) for i in (2, 3, 4, 5)),
            " ".join(block(i) for i in (3, 4, 5, 6)))


def fixture_text(name: str) -> str:
    return next(row[5] for row in _PASSAGES if row[0] == name)


GAMMA = ("GAMMA", "Gamma Energy Corp.")
# Accessions are fixed, so passage IDs are stable across runs.
_GAMMA_FILINGS = {"2024": "0000007777-25-000001", "2023": "0000007777-24-000001", "2022": "0000007777-23-000001"}


def training_extra_passages() -> list[Passage]:
    """Passages outside every evaluation pool, plus an exact copy of a pooled one and a three-passage chain."""
    a, b, c = chain_texts()

    def gamma(name: str, year: str, text: str, section_index: int, **kw: Any) -> Passage:
        return make_passage(name, *GAMMA, f"{year}-12-31", text, section_index=section_index,
                            accession=_GAMMA_FILINGS[year], **kw)

    return [
        make_passage("acme22_competition", "ACME", "Acme Devices Inc.", "2022-12-31", fixture_text("acme24_competition"),
                     section="Risk Factors", item="1A", accession="0000000001-23-000001"),
        gamma("gamma24_revenue", "2024",
              "Gamma Energy reported revenue of $3,400 million for 2024, an increase of 12% from $3,036 million in 2023, "
              "driven by higher natural gas prices and new pipeline capacity in the Permian basin.", 1),
        gamma("gamma24_debt", "2024",
              "At December 31, 2024, Gamma Energy had $900 million of borrowings outstanding under its $2,000 million "
              "revolving credit facility, leaving $1,100 million of available liquidity; the facility matures in 2029.", 2),
        gamma("gamma24_risk", "2024",
              "Gamma Energy depends on third-party pipelines to move natural gas to market. Outages, capacity limits or "
              "regulatory changes affecting these pipelines could reduce our sales volumes and revenue.", 3,
              section="Risk Factors", item="1A"),
        gamma("gamma24_table", "2024",
              "[TABLE] Segment revenue | Upstream 2024: $2,100 | 2023: $1,900 Midstream | 2024: $1,300 | 2023: $1,136 "
              "[/TABLE] Upstream revenue rose on higher realized prices while midstream revenue grew with volumes.", 4),
        gamma("gamma24_mdna", "2024", a, 5),
        gamma("gamma23_revenue", "2023",
              "Gamma Energy reported revenue of $3,036 million for 2023, a decrease of 4% from $3,163 million in 2022, "
              "as lower oil prices offset higher production volumes across its gas fields.", 1),
        gamma("gamma23_capex", "2023",
              "Capital expenditures were $640 million in 2023, mainly for well completions and compression; Gamma Energy "
              "expects 2024 spending of about $700 million as it adds processing capacity.", 2),
        gamma("gamma23_mdna", "2023", b, 5),
        gamma("gamma22_mdna", "2022", c, 5),
        gamma("gamma22_liquidity", "2022",
              "Gamma Energy ended 2022 with $410 million of cash and no borrowings under its revolving credit facility, "
              "and it renewed the facility for five years at the same commitment.", 2),
    ]


def extra_ids() -> dict[str, str]:
    return {p.metadata["fixture_name"]: p.passage_id for p in training_extra_passages()}


def all_ids() -> dict[str, str]:
    return {**passage_ids(), **extra_ids()}


class WhitespaceTokenizer:
    """Fake fast tokenizer: one token per whitespace-separated word, [CLS]/[SEP] with (0, 0) offsets."""

    def __call__(self, text: str, *, add_special_tokens: bool = True, truncation: bool = True, max_length: int = 512,
                 **_: Any) -> dict[str, Any]:
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if truncation:
            spans = spans[: max_length - 2 if add_special_tokens else max_length]
        offsets = ([(0, 0)] if add_special_tokens else []) + spans + ([(0, 0)] if add_special_tokens else [])
        special = ([1] if add_special_tokens else []) + [0] * len(spans) + ([1] if add_special_tokens else [])
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets, "special_tokens_mask": special}

    def to_str(self) -> str:
        return "whitespace-tokenizer-v1"


def write_prep_config(root: Path, path: Path | None = None, **overrides: Any) -> Path:
    """A data-preparation config over the expanded fixture inputs in ``root``."""
    manifest = json.loads((root / "corpus" / "corpus.manifest.json").read_text())
    (root / "gold_qrels.jsonl").touch()
    (root / "gold_queries.jsonl").touch()
    values: dict[str, Any] = {
        "version": "test_v1",
        "corpus_path": str(root / "corpus" / "corpus.jsonl"),
        "corpus_version": manifest["corpus_version"],
        "corpus_sha256": manifest["corpus"]["sha256"],
        "output_dir": str(root / "data" / "processed" / "train" / "test_v1"),
        "min_source_words": 5,
        "overlap": {"shingle_words": 5, "max_shingle_df": 50, "min_shared_shingles": 10,
                    "containment_threshold": 0.5, "near_duplicate_threshold": 0.8},
        "blocklist": {
            "annotation_workspaces": [str(root / "annotation")],
            "dev_pools": [str(root / "dev_ai_v3" / "pool")],
            "audit_workspaces": [str(root / "audit")],
            "gold_qrels": str(root / "gold_qrels.jsonl"),
            "gold_metadata": str(root / "gold_eval.metadata.json"),
        },
        "split": {"seed": 7, "validation_share": 0.3},
        "visible_text": {"tokenizer": "fake/whitespace", "revision": "0" * 40, "max_seq_length": 40},
        "generation": {"model": "test-model", "requests_per_minute": 600, "max_retries": 2, "backoff_seconds": 1.0,
                       "queries_per_passage": {"train": 2, "validation": 1}},
        "leakage": {"evaluation_question_files": [str(root / "annotation" / "questions.jsonl"), str(root / "gold_queries.jsonl")]},
        "reuse": {"min_sentence_words": 8},  # the fixture's sentences are short; everything else as in data_prep_v2
    }
    for key, value in overrides.items():
        if value is None:  # e.g. reuse=None: a format-1 preparation without evidence-reuse checks
            values.pop(key, None)
            continue
        if isinstance(value, dict) and isinstance(values.get(key), dict):
            values[key] = {**values[key], **value}
        else:
            values[key] = value
    path = path or root / "data_prep.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def accept_all(prepared_dir: Path, path: Path, decision: str = "accept") -> Path:
    """Quality decisions (``decision`` for every built train pair) in the review-file format."""
    rows = [json.loads(line) for line in (prepared_dir / "pairs" / "train_pairs.jsonl").read_text().splitlines()]
    path.write_text("".join(json.dumps({
        "pair_id": r["pair_id"], "passage_id": r["positive_passage_id"],
        "question_sha256": hashlib.sha256(r["query"].encode()).hexdigest(), "decision": decision, "issues": [],
        "reviewer": "test_review", "gold": False, **({} if decision == "accept" else {"reason": "test"})}) + "\n"
        for r in rows))
    return path


# --- mocked Groq API ---------------------------------------------------------------------------


def _prompt_parts(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    user = payload["messages"][-1]["content"]
    company = re.search(r"Company: (.+?) \(", user).group(1)
    year = re.search(r"fiscal year ended (\d{4})", user).group(1)
    text = user.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
    return company.split()[0], year, text


def generic_candidates(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """A valid question built from the prompt, then one that copies the passage."""
    company, year, text = _prompt_parts(payload)
    words = text.split()
    evidence = " ".join(words[:8])
    number = re.search(r"\$?\d[\d,]*", evidence)
    content = [w.strip(".,;$%[]|") for w in words[:8] if len(w.strip(".,;$%[]|")) > 3 and not w[0].isdigit()]
    topic = " ".join(content[:2]) or "operations"
    return [
        {"question": f"What does {company} disclose about {topic} for fiscal {year}?",
         "type": "financial_table" if "[TABLE]" in text else "mdna",
         "answer": number.group(0) if number else content[0] if content else words[0], "evidence": evidence},
        {"question": f"{company} {' '.join(words[:12])}?", "type": "other", "answer": words[0], "evidence": evidence},
    ]


class MockTransport:
    """Records every request and answers it with ``respond(payload)`` (a status, headers, body triple or content)."""

    def __init__(self, respond: Callable[[Mapping[str, Any]], Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.respond = respond or (lambda payload: {"questions": generic_candidates(payload)})

    def __call__(self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float) -> TransportResponse:
        self.calls.append({"url": url, "headers": dict(headers), "payload": payload, "timeout": timeout})
        answer = self.respond(payload)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, TransportResponse):
            return answer
        content = answer if isinstance(answer, str) else json.dumps(answer)
        return TransportResponse(200, {}, json.dumps({"choices": [{"message": {"content": content}}]}))


def ok(content: Any) -> TransportResponse:
    text = content if isinstance(content, str) else json.dumps(content)
    return TransportResponse(200, {}, json.dumps({"choices": [{"message": {"content": text}}]}))
