#!/usr/bin/env python
"""See ``finretriever.training.gpu_rehearsal`` (run from the bundle directory on the GPU; takes no optimizer step)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the bundle or repository root without ``pip install -e .``.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.training.gpu_rehearsal import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
