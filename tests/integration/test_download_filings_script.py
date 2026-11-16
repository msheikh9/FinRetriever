"""Integration tests for scripts/01_download_filings.py.

The script's ``main(argv)`` is called in-process so the HTTP boundary can be
replaced: ``requests.Session`` is swapped for a fake that serves canned SEC
responses, and DNS/socket connects are blocked as a backstop. Everything
else — YAML loading, CLI overrides, the real ``EdgarClient``, and the
on-disk layout — runs exactly as it does against sec.gov.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from finretriever.data import edgar_downloader
from finretriever.data.chunker import chunk_text

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "01_download_filings.py"

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 19617, "ticker": "JPM", "title": "JPMORGAN CHASE & CO"},
}

FILINGS = {
    "0000320193": [
        ("0000320193-23-000106", "2023-11-03", "10-K", "aapl-20230930.htm"),
        ("0000320193-23-000077", "2023-08-04", "10-Q", "aapl-20230701.htm"),
        ("0000320193-22-000108", "2022-10-28", "10-K", "aapl-20220924.htm"),
        ("0000320193-21-000105", "2021-10-29", "10-K", "aapl-20210925.htm"),
    ],
    "0000019617": [
        ("0000019617-24-000225", "2024-02-16", "10-K", "jpm-20231231.htm"),
        ("0000019617-24-000100", "2024-01-12", "8-K", "jpm-8k.htm"),
        ("0000019617-23-000231", "2023-02-21", "10-K", "jpm-20221231.htm"),
    ],
}


class _Response:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content
        self.headers: dict[str, str] = {}


class FakeSECSession:
    """Serves ticker map, submissions JSON and filing HTML from FILINGS."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.user_agents: set[str] = set()
        self.routes: dict[str, bytes] = {
            edgar_downloader.COMPANY_TICKERS_URL: json.dumps(TICKERS).encode()
        }
        for cik, rows in FILINGS.items():
            recent = {
                "accessionNumber": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "reportDate": ["" for _ in rows],
                "form": [r[2] for r in rows],
                "primaryDocument": [r[3] for r in rows],
            }
            self.routes[edgar_downloader.submissions_url(cik)] = json.dumps(
                {"cik": cik, "filings": {"recent": recent, "files": []}}
            ).encode()
            for accession, _, _, document in rows:
                url = edgar_downloader.build_filing_url(cik, accession, document)
                self.routes[url] = f"<html>{accession}</html>".encode()

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> _Response:
        self.calls.append(url)
        self.user_agents.add(headers["User-Agent"])
        if url not in self.routes:
            return _Response(404, b"")
        return _Response(200, self.routes[url])

    def archive_calls(self) -> list[str]:
        return [url for url in self.calls if "/Archives/" in url]


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("real network access attempted in an integration test")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def fake_sec(monkeypatch: pytest.MonkeyPatch) -> FakeSECSession:
    session = FakeSECSession()
    monkeypatch.setattr(edgar_downloader.requests, "Session", lambda: session)
    monkeypatch.setenv("SEC_USER_AGENT", "FinRetriever CI ci@example.com")
    return session


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_filings_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    config = {
        "name": "corpus_test",
        "tickers": ["aapl", "JPM"],
        "form_type": "10-K",
        "filings_per_ticker": 2,
        "output_dir": str(tmp_path / "raw"),
        "request_delay_seconds": 0,
        "timeout_seconds": 5,
        "max_retries": 0,
        # Later-stage settings must be ignored by the downloader.
        "chunk_size_tokens": 512,
        "chunk_overlap_tokens": 80,
        "paths": {"processed_dir": str(tmp_path / "processed")},
    }
    config.update(overrides)
    path = tmp_path / "corpus.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


EXPECTED_DIRS = {
    ("AAPL", "0000320193-23-000106"),
    ("AAPL", "0000320193-22-000108"),
    ("JPM", "0000019617-24-000225"),
    ("JPM", "0000019617-23-000231"),
}


def downloaded_dirs(raw: Path) -> set[tuple[str, str]]:
    return {
        (p.parent.parent.parent.name, p.parent.name)
        for p in raw.glob("*/10-K/*/filing.html")
    }


def test_download_writes_raw_layout_and_rerun_is_idempotent(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = write_config(tmp_path)
    raw = tmp_path / "raw"

    assert script.main(["--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "filings selected: 4" in out
    assert "downloaded: 4" in out
    assert "skipped: 0" in out
    assert "failed: 0" in out
    assert f"output dir: {raw}" in out

    assert downloaded_dirs(raw) == EXPECTED_DIRS
    filing = raw / "AAPL" / "10-K" / "0000320193-23-000106"
    assert (filing / "filing.html").read_bytes() == b"<html>0000320193-23-000106</html>"
    metadata = json.loads((filing / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["ticker"] == "AAPL"
    assert metadata["cik"] == "0000320193"
    assert metadata["company_name"] == "Apple Inc."
    assert metadata["form_type"] == "10-K"
    assert metadata["filing_date"] == "2023-11-03"
    assert metadata["report_date"] is None
    assert metadata["filing_url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    )
    assert metadata["local_html_path"] == (filing / "filing.html").as_posix()
    assert len(fake_sec.archive_calls()) == 4
    assert fake_sec.user_agents == {"FinRetriever CI ci@example.com"}

    before = {p: p.stat().st_mtime_ns for p in raw.rglob("*") if p.is_file()}
    fake_sec.calls.clear()
    assert script.main(["--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "downloaded: 0" in out
    assert "skipped: 4" in out
    assert fake_sec.archive_calls() == []
    assert {p: p.stat().st_mtime_ns for p in raw.rglob("*") if p.is_file()} == before


def test_dry_run_lists_filings_without_writing_html(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = write_config(tmp_path)

    assert script.main(["--config", str(config_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("would download") == 4
    assert (
        "would download AAPL 0000320193-23-000106 filed=2023-11-03 cik=0000320193 "
        "doc=aapl-20230930.htm url=https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019323000106/aapl-20230930.htm -> "
    ) in out
    assert "filings selected: 4" in out
    assert "downloaded: 0" in out
    assert fake_sec.archive_calls() == []
    assert not (tmp_path / "raw").exists()


def test_cli_overrides_output_dir_and_filings_per_ticker(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = write_config(tmp_path)
    override_dir = tmp_path / "elsewhere"

    argv = ["--config", str(config_path), "--output-dir", str(override_dir),
            "--filings-per-ticker", "1"]
    assert script.main(argv) == 0
    assert "downloaded: 2" in capsys.readouterr().out
    assert downloaded_dirs(override_dir) == {
        ("AAPL", "0000320193-23-000106"),
        ("JPM", "0000019617-24-000225"),
    }
    assert not (tmp_path / "raw").exists()


def test_unknown_ticker_is_reported_and_exit_code_is_nonzero(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = write_config(tmp_path, tickers=["AAPL", "ZZZZ"])

    assert script.main(["--config", str(config_path)]) == 1
    captured = capsys.readouterr()
    assert "downloaded: 2" in captured.out
    assert "failed: 1" in captured.out
    assert "failure: ZZZZ:" in captured.err


def test_missing_user_agent_exits_before_any_request(
    tmp_path: Path,
    script: ModuleType,
    fake_sec: FakeSECSession,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.delenv("SEC_USER_AGENT")
    config_path = write_config(tmp_path)

    assert script.main(["--config", str(config_path)]) == 2
    assert "SEC_USER_AGENT is not set" in capsys.readouterr().err
    assert fake_sec.calls == []


def test_invalid_config_exits_with_error(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = write_config(tmp_path)

    assert script.main(["--config", str(config_path), "--filings-per-ticker", "0"]) == 2
    assert "invalid config" in capsys.readouterr().err
    assert fake_sec.calls == []


def run_expecting_config_error(
    script: ModuleType, capsys: pytest.CaptureFixture, *argv: str
) -> str:
    assert script.main(list(argv)) == 2
    captured = capsys.readouterr()
    err = captured.err.strip()
    assert err.startswith("[01_download_filings] error: ")
    assert len(err.splitlines()) == 1, err
    assert "Traceback" not in err
    assert captured.out == ""
    return err


def test_missing_config_file_is_reported(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    missing = tmp_path / "nope.yaml"
    err = run_expecting_config_error(script, capsys, "--config", str(missing))
    assert f"config file not found: {missing} (pass --config PATH)" in err
    assert fake_sec.calls == []


def test_config_path_that_is_a_directory_is_reported(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    err = run_expecting_config_error(script, capsys, "--config", str(tmp_path))
    assert f"config path is not a file: {tmp_path}" in err


def test_invalid_yaml_is_reported_with_location(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("form_type: 10-K\ntickers: [AAPL, JPM\n", encoding="utf-8")
    err = run_expecting_config_error(script, capsys, "--config", str(config_path))
    assert f"invalid YAML in {config_path} at line 3" in err
    assert fake_sec.calls == []


def test_non_mapping_yaml_root_is_reported(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- AAPL\n- JPM\n", encoding="utf-8")
    err = run_expecting_config_error(script, capsys, "--config", str(config_path))
    assert f"cannot load config {config_path}" in err
    assert "must be a mapping, got list" in err


@pytest.mark.parametrize(
    "content,expected",
    [
        ("", "tickers: Field required"),
        ("tickers: AAPL\n", "tickers: Input should be a valid list"),
        ("tickers: [AAPL, '  ']\n", "tickers: tickers must not contain blank entries"),
        ("tickers: [AAPL]\nfilings_per_ticker: seven\n", "filings_per_ticker: Input should be"),
        ("tickers: [AAPL]\nrequest_delay_seconds: -1\n", "request_delay_seconds: Input should be"),
        ("tickers: []\n", "tickers: List should have at least 1 item"),
    ],
)
def test_malformed_settings_are_reported_concisely(
    tmp_path: Path,
    script: ModuleType,
    fake_sec: FakeSECSession,
    capsys: pytest.CaptureFixture,
    content: str,
    expected: str,
) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(content, encoding="utf-8")
    err = run_expecting_config_error(script, capsys, "--config", str(config_path))
    assert f"invalid config settings in {config_path}: " in err
    assert expected in err
    assert "errors.pydantic.dev" not in err
    assert fake_sec.calls == []


def test_misspelled_key_is_rejected_instead_of_defaulting(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = tmp_path / "typo.yaml"
    config_path.write_text("tickers: [AAPL]\nfiling_per_ticker: 1\n", encoding="utf-8")
    # A CLI override of the real key must not mask the typo either.
    err = run_expecting_config_error(
        script, capsys, "--config", str(config_path), "--filings-per-ticker", "1"
    )
    assert err == (
        f"[01_download_filings] error: invalid config settings in {config_path}: "
        "unknown config key 'filing_per_ticker' (did you mean 'filings_per_ticker'?)"
    )
    assert fake_sec.calls == []
    assert not (tmp_path / "raw").exists()


def test_all_invalid_settings_are_listed_together(
    tmp_path: Path, script: ModuleType, fake_sec: FakeSECSession, capsys: pytest.CaptureFixture
) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("tickers: []\nmax_retries: -1\n", encoding="utf-8")
    err = run_expecting_config_error(script, capsys, "--config", str(config_path))
    assert "tickers: " in err
    assert "; max_retries: " in err


# --- Makefile -----------------------------------------------------------------

_needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def make_dry_run(*args: str) -> str:
    """Return the commands ``make -n`` would run, without running them."""
    result = subprocess.run(
        ["make", "-n", *args, "PYTHON=python"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@_needs_make
def test_make_download_runs_downloader_with_corpus_config() -> None:
    assert make_dry_run("download").strip() == (
        "python scripts/01_download_filings.py --config configs/data/corpus_v1.yaml"
    )


@_needs_make
def test_make_download_forwards_extra_args() -> None:
    assert "--config configs/data/corpus_v1.yaml --dry-run" in make_dry_run(
        "download", "DOWNLOAD_ARGS=--dry-run"
    )


@_needs_make
def test_make_corpus_targets_offline_build_stage() -> None:
    assert make_dry_run("corpus", "CORPUS_ARGS=--strict").strip() == (
        "python scripts/01_build_corpus.py --config configs/data/corpus_v1.yaml --strict"
    )
    help_lines = make_dry_run("help")
    download_help = next(line for line in help_lines.splitlines() if "  download  " in line)
    corpus_help = next(line for line in help_lines.splitlines() if "  corpus  " in line)
    assert "SEC_USER_AGENT" in download_help
    assert "offline" in corpus_help


# --- repo config --------------------------------------------------------------


def test_repo_corpus_config_chunk_settings_are_accepted_by_chunker() -> None:
    raw = yaml.safe_load((_REPO_ROOT / "configs" / "data" / "corpus_v1.yaml").read_text())
    chunks = chunk_text(
        " ".join(f"w{i}" for i in range(1000)),
        chunk_size_tokens=raw["chunk_size_tokens"],
        chunk_overlap_tokens=raw["chunk_overlap_tokens"],
    )
    assert len(chunks) > 1


def test_repo_corpus_config_is_valid_for_downloader() -> None:
    config = edgar_downloader.EdgarDownloadConfig.model_validate(
        yaml.safe_load((_REPO_ROOT / "configs" / "data" / "corpus_v1.yaml").read_text())
    )
    assert config.tickers == ["AAPL", "JPM", "TSLA", "PFE", "AMZN", "XOM", "WMT"]
    assert config.form_type == "10-K"
    assert config.filings_per_ticker == 7
    assert config.output_dir == Path("data/raw")
    # Exxon's 10-K history is under its pre-2026 registrant (see the YAML comment).
    assert config.cik_overrides == {"XOM": "0000034088"}
