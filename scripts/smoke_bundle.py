#!/usr/bin/env python
"""Build (or verify) the portable bundle for a trainer run on a Kaggle T4: the smoke test, or a validated run.

    python scripts/smoke_bundle.py build                   # dist/kaggle/finretriever_smoke_v1.tar.gz
    python scripts/smoke_bundle.py build --config configs/train/biencoder_bge_base_ft_v3_rehearsal.yaml \
        --name finretriever_train_v3_rehearsal             # a validated fine-tuning run (docs/kaggle_training.md)
    python scripts/smoke_bundle.py verify --dir <extracted bundle directory>

See ``finretriever.training.smoke_bundle``, docs/kaggle_smoke_test.md and docs/kaggle_training.md. Runs locally from
the repository root; it reads the corpus, preparation, candidates and validation set only to trace what the run
needs, and never writes to them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.training.smoke_bundle import (  # noqa: E402
    DEFAULT_NAME,
    BundleError,
    build_bundle,
    verify_bundle,
)

SMOKE_CONFIG = "configs/train/biencoder_bge_base_smoke.yaml"
SMOKE_FALLBACK = "configs/train/biencoder_bge_base_smoke_gc.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--config", default=SMOKE_CONFIG)
    build.add_argument("--fallback-config", action="append",
                       help=f"An alternative config with the same inputs (repeatable; default for the smoke config: "
                       f"{SMOKE_FALLBACK}).")
    build.add_argument("--name", default=DEFAULT_NAME)
    build.add_argument("--output-dir", default="dist/kaggle", help="Gitignored (dist/).")
    verify = sub.add_parser("verify")
    verify.add_argument("--dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            output = Path(args.output_dir)
            fallback = args.fallback_config
            if fallback is None:
                fallback = [SMOKE_FALLBACK] if args.config == SMOKE_CONFIG else []
            build_bundle(args.config, root=Path("."), output_dir=output, name=args.name, extra_configs=fallback)
            template = output / "dataset-metadata.json"
            if not template.exists():  # for `kaggle datasets create -p <output>`; never overwritten
                slug = args.name.replace("_", "-")
                template.write_text(json.dumps({
                    "title": slug,
                    "id": f"YOUR_KAGGLE_USERNAME/{slug}",
                    "licenses": [{"name": "other"}],
                }, indent=2) + "\n", encoding="utf-8")
                print(f"[smoke_bundle] wrote {template} (set your Kaggle username; datasets are private by default)")
        else:
            print(verify_bundle(Path(args.dir)))
    except BundleError as exc:
        print(f"[smoke_bundle] refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
