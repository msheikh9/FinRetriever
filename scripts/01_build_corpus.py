#!/usr/bin/env python
"""Step 1b: build the passage corpus from downloaded SEC filings.

Reads the corpus YAML (``configs/data/corpus_v1.yaml`` by default), validates
every raw filing that ``scripts/01_download_filings.py`` stored under the raw
directory (metadata schema + SHA-256), parses it into Item sections, drops
boilerplate sections, chunks the rest and writes the corpus JSONL plus a
``.manifest.json`` next to it. Works entirely offline.

Exit codes: 0 complete corpus written; 1 corpus written but some filings were
rejected or configured tickers have no filings, or strict mode refused to
build; 2 configuration error or nothing buildable.
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

from finretriever.data.corpus_builder import (  # noqa: E402
    CorpusBuild,
    CorpusBuildError,
    CorpusConfigError,
    StrictModeError,
    build_corpus,
    load_corpus_build_config,
    write_corpus,
)

_PREFIX = "[01_build_corpus]"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="configs/data/corpus_v1.yaml",
        help="Path to the corpus YAML config.",
    )
    parser.add_argument("--raw-dir", default=None, help="Override the raw filing directory (output_dir).")
    parser.add_argument("--output", default=None, help="Override the corpus JSONL path (paths.corpus_jsonl).")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest path (default: <output stem>.manifest.json next to the corpus).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Refuse to build if any filing is rejected or a configured ticker has no filing.",
    )
    return parser.parse_args(argv)


def _report_problems(build: CorpusBuild) -> None:
    for problem in build.problems:
        print(f"{_PREFIX} problem: {problem}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_corpus_build_config(
            args.config,
            raw_dir=args.raw_dir,
            output=args.output,
            manifest=args.manifest,
            strict=args.strict,
        )
    except CorpusConfigError as exc:
        print(f"{_PREFIX} error: {exc}", file=sys.stderr)
        return 2
    print(
        f"{_PREFIX} config={args.config} raw_dir={config.raw_dir} output={config.output_path} "
        f"chunk={config.chunk_size_tokens}/{config.chunk_overlap_tokens} strict={config.strict}"
    )

    try:
        build = build_corpus(config)
    except StrictModeError as exc:
        _report_problems(exc.build)
        print(f"{_PREFIX} error: {exc} (nothing written)", file=sys.stderr)
        return 1
    except CorpusBuildError as exc:
        print(f"{_PREFIX} error: {exc} (nothing written)", file=sys.stderr)
        return 2

    manifest = write_corpus(build)
    counts = manifest["counts"]
    reasons = ", ".join(f"{k}={v}" for k, v in manifest["filtered_sections_by_reason"].items()) or "none"
    passage_reasons = ", ".join(f"{k}={v}" for k, v in manifest["filtered_passages_by_reason"].items())
    diagnostics = manifest["diagnostics"]
    print(
        f"{_PREFIX} filings: discovered {counts['filing_directories_discovered']}, "
        f"included {counts['filings_included']}, rejected {counts['filings_rejected']}, "
        f"excluded {counts['filings_excluded']}"
    )
    print(
        f"{_PREFIX} sections: parsed {counts['sections_parsed']}, kept {counts['sections_kept']}, "
        f"filtered {counts['sections_filtered']} ({reasons})"
    )
    print(
        f"{_PREFIX} passages: {counts['passages']} written, {counts['passages_filtered']} filtered"
        + (f" ({passage_reasons})" if passage_reasons else "")
        + f", max {diagnostics.get('max_passage_tokens', 0)} tokens"
    )
    print(
        f"{_PREFIX} table rows: {diagnostics['table_rows']}, continued: "
        f"{diagnostics['table_rows_requiring_continuation']}, not represented: "
        f"{diagnostics['table_rows_not_represented']}; passages starting/ending mid-row: "
        f"{diagnostics['passages_starting_mid_table_row']}/{diagnostics['passages_ending_mid_table_row']}, "
        f"unlabelled fragments: {diagnostics['unlabelled_partial_row_fragments']}"
    )
    coverage = manifest["coverage"]
    ranges = ", ".join(f"{t} {v['min']:.1%}-{v['max']:.1%}" for t, v in coverage["by_ticker"].items())
    print(f"{_PREFIX} source-text coverage by ticker: {ranges}")
    for warning in coverage["warnings"]:
        print(
            f"{_PREFIX} warning: low coverage {warning['coverage']:.1%} for {warning['ticker']} "
            f"{warning['filing_date']} ({warning['unassigned_words']:,} unassigned words)",
            file=sys.stderr,
        )
    corpus = manifest["corpus"]
    print(f"{_PREFIX} corpus: {corpus['path']} ({corpus['bytes']:,} bytes, sha256 {corpus['sha256'][:16]})")
    print(f"{_PREFIX} manifest: {config.resolved_manifest_path}")
    if build.problems:
        _report_problems(build)
        print(f"{_PREFIX} incomplete: {len(build.problems)} problem(s); see manifest", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
