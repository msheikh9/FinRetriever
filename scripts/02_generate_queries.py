#!/usr/bin/env python
"""Step 2: leakage-safe training-data preparation and synthetic query generation.

Subcommands (see ``finretriever.training.data_cli``):

* ``prepare``: overlap groups, evaluation blocklist, train/validation split by overlap group,
  tokenizer-visible text and a manifest, all offline;
* ``generate --split train|validation``: synthetic analyst queries from approved passages through
  the configured Groq model. It prints the plan and sends nothing unless ``--confirm-api-use`` is
  given and the API key is in the environment (``GROQ_API_KEY`` by default). Resumable;
* ``build-pairs``: training pairs from the generation records, with duplicate, leakage and quality
  checks and a pairs manifest;
* ``check``: re-run the quality checks on written pairs.

Everything is written under ``output_dir`` in ``data/processed/`` (gitignored). Exit codes: 0 done;
1 refused or a check failed; 2 configuration, corpus or credential error.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root without ``pip install -e .``.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.training.data_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
