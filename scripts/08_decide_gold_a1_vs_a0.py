#!/usr/bin/env python
"""Step 8: the A1-versus-A0 decision on the human gold set, with the rule frozen before any gold label existed.

See ``finretriever.evaluation.gold_decision`` and docs/gold_set_v1_protocol.md.
- **The test:** a one-sided paired sign-flip test (10,000 flips, PCG64 seed 20261112) of nDCG@10. A1 is the mean of
  the three official checkpoints, and A0 is step 0 of the same runs.
- **The decision:** A1 improves on A0 only if p <= 0.05 and every seed improves.
- **Unjudged passages:** they count as non-relevant, which can bias A1 downwards, so a negative result is never
  evidence of no improvement.

It reads the gold export and the benchmark's run directories, and writes the decision record once.

Exit codes: 0 decided (either way); 1 refused (a precondition failed, or a different record exists).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root without ``pip install -e .``.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.evaluation.gold_decision import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
