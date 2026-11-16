"""Building the v2 retrieval pipeline from its experiment configuration (configs/experiments/*.yaml).

Shared by the experiment runner (scripts/11_v2_experiments.py) and the search demo (scripts/12_search_demo.py),
so both build exactly the same components:

* :func:`experiment_config`: an experiment entry (with the config's defaults) as a
  :class:`~finretriever.retrieval.hybrid.HybridConfig`;
* :func:`build_bm25`: a BM25 component (plain or enriched view), memoised per query;
* :func:`build_seed_components`: the A1 components of one seed (plain or enriched view, MaxP windows). The
  checkpoint is verified against its selection record and its encoder loaded once; embeddings come from (or go
  to) the embedding cache.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from finretriever.data.schemas import Passage
from finretriever.retrieval.bm25 import BM25Retriever
from finretriever.retrieval.checkpoint import (
    CheckpointRetriever,
    inspect_checkpoint,
    load_checkpoint_encoder,
    tokenizer_sha256,
)
from finretriever.retrieval.fusion import FusionConfig
from finretriever.retrieval.hybrid import CachedScorer, HybridConfig, MaxPScorer
from finretriever.retrieval.metadata_scope import ScopePolicy
from finretriever.retrieval.windows import sliding_windows, tail_windows, windows_sha256
from finretriever.training.checkpoint_selection import verify_selected_checkpoint

SEEDLESS = "-"


def _log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def experiment_config(raw: dict[str, Any], defaults: dict[str, Any]) -> HybridConfig:
    fusion = {**defaults["fusion"], **raw.get("fusion", {})}
    scope = {**defaults["scope"], **raw.get("scope", {})}
    return HybridConfig(
        name=raw["id"],
        components=tuple(raw["components"]),
        fusion=FusionConfig(fusion["method"], k=fusion["k"], weights=dict(fusion["weights"]), depth=fusion["depth"]),
        policy=ScopePolicy(**scope),
        metadata_weight=float(raw.get("metadata_weight", defaults["metadata_weight"])),
    )


def component_seeds(name: str, cfg: dict[str, Any]) -> list[str]:
    spec = cfg["components"][name]
    if spec["type"] == "bm25":
        return [SEEDLESS]
    return list(spec.get("seeds") or cfg["seeds"])


def experiment_seeds(raw: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    seeded = [set(component_seeds(c, cfg)) for c in raw["components"] if cfg["components"][c]["type"] != "bm25"]
    if not seeded:
        return [SEEDLESS]
    return [s for s in cfg["seeds"] if all(s in group for group in seeded)]


def build_bm25(name: str, spec: dict[str, Any], passages: list[Passage]) -> tuple[CachedScorer, dict[str, Any]]:
    retriever = BM25Retriever(retriever_name=name, tokenizer=spec.get("tokenizer", "word"),
                              backend=spec.get("backend", "python"), passage_view=spec.get("passage_view", "text"))
    retriever.index(passages)
    return CachedScorer(retriever), retriever.describe()


def build_seed_components(seed: str, names: list[str], cfg: dict[str, Any], passages: list[Passage],
                          position: dict[str, int], window_cache: dict[str, Any],
                          log: Callable[[str], None] = _log) -> tuple[dict, dict]:
    """The A1 components of one seed; the checkpoint is verified and its encoder loaded once."""
    spec_seed = cfg["seeds"][seed]
    selection = verify_selected_checkpoint(spec_seed["selection_record"], spec_seed["checkpoint_path"])
    checkpoint = inspect_checkpoint(spec_seed["checkpoint_path"])
    encoder = load_checkpoint_encoder(checkpoint, device=cfg["device"])
    common = dict(retriever_name=None, cache_dir=cfg["embedding_cache_dir"], device=cfg["device"],
                  encoder=encoder, selection=selection, role="final")
    scorers: dict[str, Any] = {}
    described: dict[str, Any] = {}

    def dense(view: str, label: str, docs: list[Passage], corpus_sha: str) -> CheckpointRetriever:
        r = CheckpointRetriever(checkpoint, **{**common, "retriever_name": f"{seed}:{label}"},
                                corpus_sha256=corpus_sha, passage_view=view)
        t0 = time.time()
        r.index(docs)
        log(f"{seed}:{label} indexed {len(docs)} texts in {time.time() - t0:.0f}s")
        return r

    head: CheckpointRetriever | None = None
    for name in names:
        spec = cfg["components"][name]
        if spec["type"] == "a1":
            r = dense(spec.get("passage_view", "text"), name, passages, cfg["corpus_sha256"])
            if spec.get("passage_view", "text") == "text":
                head = r
            scorers[name] = r
            described[name] = r.describe()
    for name in names:
        spec = cfg["components"][name]
        if spec["type"] != "a1_windows":
            continue
        if head is None:
            head = dense("text", "a1", passages, cfg["corpus_sha256"])
        w = spec["windows"]
        key = f"{w['kind']}:{w['size']}:{w['overlap']}"
        tokenizer = encoder.model.tokenizer if hasattr(encoder, "model") else None
        if tokenizer is None:
            from transformers import AutoTokenizer  # the checkpoint's own tokenizer files
            tokenizer = AutoTokenizer.from_pretrained(spec_seed["checkpoint_path"], local_files_only=True)
        tok_sha = tokenizer_sha256(tokenizer)
        if key not in window_cache or window_cache[key]["tokenizer_sha256"] != tok_sha:
            head_tokens = encoder.max_seq_length - 2
            if w["kind"] == "tail":
                windows = tail_windows(passages, tokenizer, head_tokens=head_tokens, size=w["size"],
                                       overlap=w["overlap"])
            else:
                windows = sliding_windows(passages, tokenizer, size=w["size"], overlap=w["overlap"])
            window_cache[key] = {"windows": windows, "sha256": windows_sha256(windows), "tokenizer_sha256": tok_sha,
                                 "head_tokens": head_tokens}
        entry = window_cache[key]
        windows = entry["windows"]
        wr = dense("text", f"{name}_windows", windows, entry["sha256"])
        parents = [position[x.metadata["parent_passage_id"]] for x in windows]
        scorers[name] = MaxPScorer(head, wr, parents)
        described[name] = {"type": "maxp", "head": head.describe(), "windows": wr.describe(),
                           "window_spec": w, "window_corpus_sha256": entry["sha256"], "windows_count": len(windows),
                           "head_tokens": entry["head_tokens"], "tokenizer_sha256": tok_sha,
                           "passages_with_windows": len(set(parents))}
    return scorers, described


__all__ = ["SEEDLESS", "build_bm25", "build_seed_components", "component_seeds", "experiment_config",
           "experiment_seeds"]
