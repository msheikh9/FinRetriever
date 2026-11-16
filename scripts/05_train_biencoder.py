#!/usr/bin/env python
"""Step 5: fine-tune the FinRetriever bi-encoder on the verified training candidates.

See ``finretriever.training.biencoder_trainer`` (objective, false-negative protection, checkpoints)
and ``finretriever.training.trainer_setup`` (configuration and inputs). Without
``--confirm-training`` the inputs, batches and schedule are verified and printed, and nothing is
loaded or written. With it, the pinned base model is loaded (downloaded from Hugging Face unless
``--local-files-only``) and trained; ``--resume`` continues a run from its newest checkpoint.

The trainer reads no evaluation file and computes no retrieval metric. Exit codes: 0 done; 1 refused
(configuration, data or contract); 2 the model or device cannot be used, or a run would be overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root without ``pip install -e .``.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from finretriever.training.biencoder_trainer import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
