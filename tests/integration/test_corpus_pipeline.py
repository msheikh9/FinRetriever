"""Integration tests for the corpus pipeline and scripts/01_build_corpus.py.

The CLI's ``main(argv)`` runs in-process against synthetic raw filings laid
out exactly as the downloader writes them, and — when it is available
locally — against Apple's real FY2025 10-K (``FINRETRIEVER_AAPL_10K`` or the
smoke-download path). Network access is blocked throughout; CI never has the
Apple filing, so those tests skip there.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import socket
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from finretriever.data.schemas import Passage
from finretriever.utils.io import read_jsonl
from tests.unit.test_corpus_builder import KEPT_ITEMS, filing_html, write_raw_filing

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "01_build_corpus.py"
_AAPL_DEFAULT = "/tmp/finretriever_smoke_raw/AAPL/10-K/0000320193-25-000079/filing.html"
AAPL_FILING = Path(os.environ.get("FINRETRIEVER_AAPL_10K", _AAPL_DEFAULT))
needs_aapl = pytest.mark.skipif(
    not (AAPL_FILING.is_file() and (AAPL_FILING.parent / "metadata.json").is_file()),
    reason=f"Apple 10-K fixture not found at {AAPL_FILING} (set FINRETRIEVER_AAPL_10K)",
)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted during a corpus build")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_corpus_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path: Path, tickers: list[str], **extra: Any) -> Path:
    lines = [
        "name: corpus_test",
        f"tickers: [{', '.join(tickers)}]",
        "form_type: 10-K",
        f"filings_per_ticker: {extra.pop('filings_per_ticker', 7)}",
        f"output_dir: {tmp_path / 'raw'}",
        f"chunk_size_tokens: {extra.pop('chunk_size_tokens', 512)}",
        f"chunk_overlap_tokens: {extra.pop('chunk_overlap_tokens', 80)}",
        "paths:",
        f"  corpus_jsonl: {tmp_path / 'processed' / 'corpus_test.jsonl'}",
    ]
    path = tmp_path / "corpus.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def two_ticker_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    write_raw_filing(raw, ticker="ACME", accession="0000000001-25-000001", filing_date="2025-10-31")
    write_raw_filing(raw, ticker="ACME", accession="0000000001-24-000001", filing_date="2024-11-01")
    write_raw_filing(raw, ticker="BETA", accession="0000000002-25-000003", filing_date="2025-02-20",
                     html=filing_html(company="BETA CORP"))
    return raw


def outputs(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "processed" / "corpus_test.jsonl"
    return corpus, corpus.with_name("corpus_test.manifest.json")


# --- synthetic SEC-like filings ---------------------------------------------------


def test_cli_builds_complete_corpus(tmp_path: Path, two_ticker_raw: Path, script: ModuleType, capsys) -> None:
    config = write_config(tmp_path, ["ACME", "BETA"])
    assert script.main(["--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "filings: discovered 3, included 3, rejected 0, excluded 0" in out
    assert "sections: parsed 27, kept 18, filtered 9" in out

    corpus, manifest_path = outputs(tmp_path)
    passages = read_jsonl(corpus, model=Passage)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["corpus"]["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert manifest["counts"]["passages"] == len(passages)
    assert manifest["source_accessions"] == [
        "0000000001-24-000001", "0000000001-25-000001", "0000000002-25-000003"
    ]
    assert [p.ticker for p in passages] == sorted(p.ticker for p in passages)
    for accession in manifest["source_accessions"]:
        items = [p.metadata["item_number"] for p in passages if p.metadata["accession_number"] == accession]
        assert list(dict.fromkeys(items)) == KEPT_ITEMS


def test_cli_rebuild_is_byte_identical(tmp_path: Path, two_ticker_raw: Path, script: ModuleType) -> None:
    config = write_config(tmp_path, ["ACME", "BETA"])
    corpus, manifest_path = outputs(tmp_path)
    assert script.main(["--config", str(config)]) == 0
    first, first_manifest = corpus.read_bytes(), json.loads(manifest_path.read_text())
    assert script.main(["--config", str(config)]) == 0
    second_manifest = json.loads(manifest_path.read_text())
    assert corpus.read_bytes() == first
    first_manifest.pop("built_at")
    second_manifest.pop("built_at")
    assert first_manifest == second_manifest


def test_cli_partial_build_is_not_reported_as_success(
    tmp_path: Path, two_ticker_raw: Path, script: ModuleType, capsys
) -> None:
    beta = next((two_ticker_raw / "BETA").rglob("filing.html"))
    beta.write_bytes(b"<html>tampered</html>")
    config = write_config(tmp_path, ["ACME", "BETA"])
    assert script.main(["--config", str(config)]) == 1
    captured = capsys.readouterr()
    assert "rejected BETA/10-K/0000000002-25-000003: sha256_mismatch" in captured.err
    assert "no filings for configured tickers: BETA" in captured.err
    corpus, manifest_path = outputs(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["counts"]["filings_rejected"] == 1
    assert {p["ticker"] for p in read_jsonl(corpus)} == {"ACME"}


def test_cli_strict_mode_writes_nothing_on_invalid_filing(
    tmp_path: Path, two_ticker_raw: Path, script: ModuleType, capsys
) -> None:
    next((two_ticker_raw / "BETA").rglob("metadata.json")).unlink()
    config = write_config(tmp_path, ["ACME", "BETA"])
    assert script.main(["--config", str(config), "--strict"]) == 1
    assert "strict mode" in capsys.readouterr().err
    corpus, manifest_path = outputs(tmp_path)
    assert not corpus.exists() and not manifest_path.exists()


def test_cli_overrides_raw_dir_output_and_manifest(tmp_path: Path, two_ticker_raw: Path, script: ModuleType) -> None:
    moved = tmp_path / "elsewhere"
    shutil.move(str(two_ticker_raw), moved)
    config = write_config(tmp_path, ["ACME", "BETA"])
    out, manifest = tmp_path / "x" / "c.jsonl", tmp_path / "y" / "m.json"
    argv = ["--config", str(config), "--raw-dir", str(moved), "--output", str(out), "--manifest", str(manifest)]
    assert script.main(argv) == 0
    assert out.is_file() and manifest.is_file()
    assert json.loads(manifest.read_text())["settings"]["raw_dir"] == moved.as_posix()


def test_cli_with_nothing_to_build_exits_2(tmp_path: Path, script: ModuleType, capsys) -> None:
    (tmp_path / "raw").mkdir()
    assert script.main(["--config", str(write_config(tmp_path, ["ACME"]))]) == 2
    assert "no passages could be built" in capsys.readouterr().err
    assert not outputs(tmp_path)[0].exists()


def test_cli_config_errors_exit_2(tmp_path: Path, script: ModuleType, capsys) -> None:
    assert script.main(["--config", str(tmp_path / "missing.yaml")]) == 2
    assert "config file not found" in capsys.readouterr().err
    bad = tmp_path / "bad.yaml"
    bad.write_text("tickers: [ACME]\nchunk_sise_tokens: 10\npaths: {corpus_jsonl: c.jsonl}\n")
    assert script.main(["--config", str(bad)]) == 2
    assert "did you mean 'chunk_size_tokens'" in capsys.readouterr().err


# --- real Apple 10-K ----------------------------------------------------------------


@pytest.fixture(scope="module")
def aapl_build(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if not (AAPL_FILING.is_file() and (AAPL_FILING.parent / "metadata.json").is_file()):
        pytest.skip(f"Apple 10-K fixture not found at {AAPL_FILING}")
    from finretriever.data.corpus_builder import CorpusBuildConfig, build_corpus, write_corpus

    tmp_path = tmp_path_factory.mktemp("aapl")
    target = tmp_path / "raw" / "AAPL" / "10-K" / AAPL_FILING.parent.name
    shutil.copytree(AAPL_FILING.parent, target)
    config = CorpusBuildConfig(
        name="corpus_aapl", raw_dir=tmp_path / "raw", output_path=tmp_path / "corpus_aapl.jsonl",
        tickers=["AAPL"], filings_per_ticker=1, strict=True,
    )
    build = build_corpus(config)
    manifest = write_corpus(build)
    rebuild = build_corpus(config.model_copy(update={"output_path": tmp_path / "again.jsonl"}))
    write_corpus(rebuild)
    return {"build": build, "manifest": manifest, "config": config, "tmp": tmp_path}


@needs_aapl
def test_aapl_sections_filtered_as_boilerplate(aapl_build: dict[str, Any]) -> None:
    build = aapl_build["build"]
    assert build.sections_parsed == 22
    assert {(s.item_number, s.reason) for s in build.filtered_sections} == {
        ("1B", "boilerplate:none"), ("9", "boilerplate:none"), ("16", "boilerplate:none"),
        ("4", "boilerplate:not_applicable"), ("9C", "boilerplate:not_applicable"),
        *{(item, "boilerplate:proxy_reference") for item in ("10", "11", "12", "13", "14")},
    }
    kept = list(dict.fromkeys(p.metadata["item_number"] for p in build.passages))
    assert kept == ["1", "1A", "1C", "2", "3", "5", "7", "7A", "8", "9A", "9B", "15"]


@needs_aapl
def test_aapl_passages_respect_budget_order_and_ids(aapl_build: dict[str, Any]) -> None:
    passages = aapl_build["build"].passages
    assert all(p.token_count <= 512 for p in passages)
    assert len({p.passage_id for p in passages}) == len(passages)
    order = [(p.metadata["section_index"], p.metadata["chunk_index"]) for p in passages]
    assert order == sorted(order)
    assert {p.ticker for p in passages} == {"AAPL"}
    assert {p.metadata["accession_number"] for p in passages} == {"0000320193-25-000079"}


@needs_aapl
def test_aapl_narrative_and_table_passages(aapl_build: dict[str, Any]) -> None:
    passages = aapl_build["build"].passages
    risk = [p for p in passages if p.metadata["item_number"] == "1A"]
    assert any("adversely affect" in p.text for p in risk)
    assert not any(p.metadata["contains_table"] for p in risk)
    item8 = " ".join(p.text for p in passages if p.metadata["item_number"] == "8")
    assert "Net sales > Products | Years ended September 27, 2025: $307,003" in item8
    assert "Cost of sales > Products | Years ended September 27, 2025: 194,116" in item8
    assert any(p.metadata["contains_table"] for p in passages if p.metadata["item_number"] == "8")


@needs_aapl
def test_aapl_has_no_partial_table_rows(aapl_build: dict[str, Any]) -> None:
    diagnostics = aapl_build["manifest"]["diagnostics"]
    assert diagnostics["table_rows"] == 400
    assert diagnostics["passages_starting_mid_table_row"] == 0
    assert diagnostics["passages_ending_mid_table_row"] == 0
    assert diagnostics["unlabelled_partial_row_fragments"] == 0
    assert diagnostics["table_rows_not_represented"] == 0
    assert diagnostics["table_rows_requiring_continuation"] == 0
    assert diagnostics["max_passage_tokens"] <= 512
    assert len(aapl_build["build"].passages) == 84


@pytest.fixture(scope="module")
def aapl_sections(aapl_build: dict[str, Any]) -> dict[str, Any]:
    from finretriever.data.filing_parser import parse_filing

    filing_id = aapl_build["build"].filings[0].filing_id
    return {s.section_id: s for s in parse_filing(AAPL_FILING.read_bytes(), filing_id)}


@needs_aapl
def test_aapl_narrative_sections_keep_fixed_window_chunks(
    aapl_build: dict[str, Any], aapl_sections: dict[str, Any]
) -> None:
    from finretriever.data.chunker import chunk_text
    from finretriever.data.filing_parser import TABLE_START

    narrative = 0
    for section_id, section in aapl_sections.items():
        passages = [p.text for p in aapl_build["build"].passages if p.section_id == section_id]
        if passages and TABLE_START not in section.text:
            assert passages == chunk_text(section.text, 512, 80), section.item_number
            narrative += 1
    assert narrative == 7  # Items 1, 1A, 1C, 2, 3, 9A, 9B


@needs_aapl
def test_aapl_passages_cover_every_section_token(
    aapl_build: dict[str, Any], aapl_sections: dict[str, Any]
) -> None:
    for section_id, section in aapl_sections.items():
        passages = [p.text.split() for p in aapl_build["build"].passages if p.section_id == section_id]
        if not passages:
            continue
        tokens, covered, previous = section.text.split(), 0, -1
        for part in passages:  # in-order contiguous slices with no gaps
            start = next(i for i in range(previous + 1, len(tokens)) if tokens[i:i + len(part)] == part)
            assert start <= covered, (section.item_number, "gap")
            covered, previous = max(covered, start + len(part)), start
        assert covered == len(tokens), section.item_number


@needs_aapl
def test_aapl_financial_values_preserved(aapl_build: dict[str, Any], aapl_sections: dict[str, Any]) -> None:
    import re

    numbers = re.compile(r"^\(?\$?\d[\d,]*(\.\d+)?\)?%?$")
    kept = {p.section_id for p in aapl_build["build"].passages}
    present = {t for p in aapl_build["build"].passages for t in p.text.split()}
    values = [t for sid in kept for t in aapl_sections[sid].text.split() if numbers.match(t)]
    assert len(values) == 2650
    assert [v for v in values if v not in present] == []


@needs_aapl
def test_aapl_rebuild_is_identical(aapl_build: dict[str, Any]) -> None:
    tmp = aapl_build["tmp"]
    assert (tmp / "corpus_aapl.jsonl").read_bytes() == (tmp / "again.jsonl").read_bytes()
