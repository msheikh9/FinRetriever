#!/usr/bin/env python
"""Step 3: build the hand-annotated gold evaluation set.

Terminal tool for writing evaluation questions, pooling candidate passages
(BM25, manual lookup, seeded samples) and grading them 0/1/2, bound to the
frozen corpus named in ``configs/eval/gold_set_v1.yaml``. Progress is saved
after every change in ``data/eval/annotation/``; ``export`` writes
``data/eval/gold_queries.jsonl`` and ``data/eval/gold_qrels.jsonl``.

Run with ``--help`` (or ``COMMAND --help``) for the commands, and see
``docs/eval_methodology.md`` for the annotation rules. Exit codes: 0 done;
1 action refused; 2 config, corpus or workspace binding error.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root without ``pip install -e .``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.eval_set.annotation_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
