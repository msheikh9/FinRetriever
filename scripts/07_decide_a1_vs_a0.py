#!/usr/bin/env python
"""Step 7: the A0-versus-A1 decision from the seeds' selection records.

See ``finretriever.training.decision_rule``: a selection-aware sign-flip test (10,000 flips, PCG64 seed 20261109) of
the selected checkpoints against step 0 on the synthetic validation set, with the fixed selection rule re-applied
to every flip; A1 beats A0 only if p <= 0.05 and every seed selected a checkpoint after step 0. It reads only the
``selection.json`` records given and writes the decision record once.

Exit codes: 0 decided (either way); 1 refused (a precondition failed, or a different record exists).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root without ``pip install -e .``.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.training.decision_rule import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
