"""Shared pytest fixtures for FinRetriever."""

from __future__ import annotations

from typing import NoReturn

import pytest

from finretriever.retrieval import biencoder


@pytest.fixture(autouse=True)
def _no_real_dense_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test loads (or downloads) a real Hugging Face model.

    Importing the model backend fails, so a dense retriever without an
    injected fake encoder reports a load error instead of fetching weights.
    Tests that need a model pass a fake encoder or patch the loader.
    """

    def refuse() -> NoReturn:
        raise ImportError("tests never load real models; inject a fake encoder")

    monkeypatch.setattr(biencoder, "_import_backend", refuse)
