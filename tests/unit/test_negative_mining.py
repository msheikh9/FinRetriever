"""Placeholder tests for hard-negative mining.

Hard-negative mining is not implemented in the v1 scaffold. This test simply
verifies the module is importable so the test suite stays green.
"""

from __future__ import annotations

import importlib


def test_negative_mining_module_importable() -> None:
    module = importlib.import_module("finretriever.training.negative_mining")
    assert module is not None
