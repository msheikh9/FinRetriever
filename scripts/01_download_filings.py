#!/usr/bin/env python
"""Step 1: download raw 10-K filings from SEC EDGAR.

Reads the corpus YAML (``configs/data/corpus_v1.yaml`` by default), maps each
ticker to its CIK with the SEC ticker file, selects the most recent filings of
the configured form type from the submissions API, and stores each one under
``<output_dir>/<TICKER>/<FORM>/<accession>/{filing.html,metadata.json}``.
Filings already on disk are skipped, so re-running is cheap.

Requires ``SEC_USER_AGENT`` (name + contact email) in the environment.
``--dry-run`` fetches only SEC metadata (ticker map + submissions JSON) and
lists what would be downloaded; no filing document is fetched or written.
Parsing and chunking happen in later pipeline steps.

Exit codes: 0 success, 1 some tickers/filings failed, 2 configuration error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import yaml  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from finretriever.data.edgar_downloader import (  # noqa: E402
    EdgarDownloadConfig,
    EdgarError,
    download_corpus,
    filing_dir,
)
from finretriever.utils.io import read_yaml  # noqa: E402

_PREFIX = "[01_download_filings]"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="configs/data/corpus_v1.yaml",
        help="Path to the corpus YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from the config (raw filing root).",
    )
    parser.add_argument(
        "--filings-per-ticker",
        type=int,
        default=None,
        help="Override filings_per_ticker from the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch SEC metadata and list the selected filings without downloading them.",
    )
    return parser.parse_args(argv)


class ConfigError(Exception):
    """The config file is missing, unreadable, or has invalid settings."""


def load_config(
    config_path: str,
    output_dir: str | None = None,
    filings_per_ticker: int | None = None,
) -> EdgarDownloadConfig:
    """Read the corpus YAML, apply CLI overrides, and validate it.

    Every failure is raised as a one-line ``ConfigError`` naming the file
    and, for bad settings, each offending field.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path} (pass --config PATH)")
    if not path.is_file():
        raise ConfigError(f"config path is not a file: {path}")
    try:
        raw = read_yaml(path)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(exc, "problem", None) or type(exc).__name__
        raise ConfigError(f"invalid YAML in {path}{where}: {problem}") from None
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot load config {path}: {exc}") from None

    if output_dir is not None:
        raw["output_dir"] = output_dir
    if filings_per_ticker is not None:
        raw["filings_per_ticker"] = filings_per_ticker
    try:
        return EdgarDownloadConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            (f"{'.'.join(str(part) for part in error['loc'])}: " if error["loc"] else "")
            + error["msg"].removeprefix("Value error, ")
            for error in exc.errors()
        )
        raise ConfigError(f"invalid config settings in {path}: {problems}") from None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config, args.output_dir, args.filings_per_ticker)
    except ConfigError as exc:
        print(f"{_PREFIX} error: {exc}", file=sys.stderr)
        return 2

    print(
        f"{_PREFIX} config={args.config} form_type={config.form_type} "
        f"filings_per_ticker={config.filings_per_ticker} dry_run={args.dry_run}"
    )

    try:
        summary = download_corpus(
            config,
            dry_run=args.dry_run,
            progress=lambda message: print(f"{_PREFIX} {message}"),
        )
    except EdgarError as exc:
        print(f"{_PREFIX} error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for filing in summary.selected:
            print(
                f"{_PREFIX} would download {filing.ticker} {filing.accession_number} "
                f"filed={filing.filing_date} cik={filing.cik} doc={filing.primary_document} "
                f"url={filing.filing_url} -> {filing_dir(config.output_dir, filing)}"
            )

    print(
        f"{_PREFIX} tickers requested: {len(summary.requested_tickers)} "
        f"({', '.join(summary.requested_tickers)})"
    )
    print(f"{_PREFIX} filings selected: {summary.filings_selected}")
    print(f"{_PREFIX} downloaded: {summary.downloaded}")
    print(f"{_PREFIX} skipped: {summary.skipped}")
    print(f"{_PREFIX} failed: {len(summary.failures)}")
    print(f"{_PREFIX} output dir: {summary.output_dir}")
    for failure in summary.failures:
        where = failure.ticker + (f" {failure.accession_number}" if failure.accession_number else "")
        print(f"{_PREFIX} failure: {where}: {failure.error}", file=sys.stderr)
    return 1 if summary.failures else 0


if __name__ == "__main__":
    sys.exit(main())
