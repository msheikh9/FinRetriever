"""Validation evaluation for checkpoint selection: synthetic validation questions ranked over the whole corpus.

The trainer calls this at the steps of the selection rule (``checkpoint_selection``). Everything it uses is passed
in memory: the validation pairs (loaded and isolation-checked by ``trainer_setup.prepare_inputs`` from the
preparation's ``validation_sets/``), the corpus passages and the preparation's ``NegativeGuard``. This module opens
no file, so it cannot read evaluation data.

**Relevance**, per question, from the pair's own records (never from evaluation labels):

* relevant: the positive and its exact equivalents (each holds the evidence: ``NegativeGuard.positives``);
* removed from the ranking: every other passage the guard keeps from being a negative for the pair (uncertain
  equivalents, the overlap groups of the relevant passages, evidence-reuse links), so a passage that may answer
  the question counts neither for nor against a checkpoint;
* every other passage is non-relevant.

**Ranking.** Cosine similarity of L2-normalised embeddings: questions with the model's query prefix, passages with
its passage prefix, truncated to ``passage_max_length`` tokens as in training; ties keep corpus order. ``global``
ranks the whole corpus, ``company`` only the passages of the question's company. The metrics are the benchmark's
own kernels (``evaluation.metrics``): MRR@10, nDCG@10 and Recall@100.

**No side effects.** The model is evaluated in eval mode without gradients, its training mode and the random
state are restored afterwards, and no weight changes. The validation set chooses among checkpoints only; it is
never training data and never evaluation data.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from finretriever.data.schemas import Passage
from finretriever.evaluation.metrics import _mrr_at_k, _ndcg_at_k, _recall_at_k
from finretriever.training.pair_contract import TrainingPair
from finretriever.training.training_data import NegativeGuard

MODES = ("global", "company")
METRICS = {"mrr@10": (_mrr_at_k, 10), "ndcg@10": (_ndcg_at_k, 10), "recall@100": (_recall_at_k, 100)}
DEPTH = 100


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], max_length: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class ValidationTargets:
    """What validation ranks and what counts, built once per run."""

    pair_ids: tuple[str, ...]
    queries: tuple[str, ...]  # with the query prefix
    tickers: tuple[str, ...]
    relevant: tuple[frozenset[str], ...]
    removed: tuple[frozenset[str], ...]
    passage_ids: tuple[str, ...]  # corpus order
    passage_texts: tuple[str, ...]  # with the passage prefix
    passage_tickers: tuple[str, ...]


def build_targets(
    pairs: Sequence[TrainingPair],
    guard: NegativeGuard,
    corpus: Mapping[str, Passage],
    *,
    query_prefix: str,
    passage_prefix: str,
) -> ValidationTargets:
    if not pairs:
        raise ValueError("no validation pairs")
    relevant = tuple(guard.positives(p) for p in pairs)
    removed = tuple(guard.exclusions(p) - r for p, r in zip(pairs, relevant, strict=True))
    if missing := sorted({pid for r in relevant for pid in r} - set(corpus)):
        raise ValueError(f"validation positives missing from the corpus: {missing[:3]}")
    return ValidationTargets(
        pair_ids=tuple(p.pair_id for p in pairs),
        queries=tuple(query_prefix + p.query for p in pairs),
        tickers=tuple(p.ticker for p in pairs),
        relevant=relevant,
        removed=removed,
        passage_ids=tuple(corpus),
        passage_texts=tuple(passage_prefix + p.text for p in corpus.values()),
        passage_tickers=tuple(p.ticker for p in corpus.values()),
    )


def _embed(
    encoder: Encoder, texts: Sequence[str], max_length: int, batch_size: int, device: torch.device, fp16: bool
) -> torch.Tensor:
    out = []
    for start in range(0, len(texts), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            out.append(F.normalize(encoder.encode(texts[start : start + batch_size], max_length).float(), dim=-1))
    return torch.cat(out)


def rank(scores: torch.Tensor, keep: torch.Tensor, passage_ids: Sequence[str], depth: int = DEPTH) -> list[str]:
    """The top ``depth`` passage IDs among ``keep``, by score, ties in corpus order."""
    index = torch.nonzero(keep, as_tuple=True)[0]
    order = torch.sort(scores[index], descending=True, stable=True).indices[:depth]
    return [passage_ids[i] for i in index[order].tolist()]


def question_metrics(ranked: Sequence[str], relevant: frozenset[str]) -> dict[str, float]:
    qrels = dict.fromkeys(relevant, 1)
    return {name: kernel(qrels, list(ranked), k) for name, (kernel, k) in METRICS.items()}


def evaluate(
    encoder: Encoder,
    targets: ValidationTargets,
    *,
    query_max_length: int,
    passage_max_length: int,
    batch_size: int,
    device: torch.device,
    fp16: bool = False,
) -> dict[str, Any]:
    """Every validation question ranked in both modes; aggregate and per-question metrics (see the module docstring)."""
    started = time.perf_counter()
    module = encoder if isinstance(encoder, torch.nn.Module) else None
    was_training = module.training if module is not None else False
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        if module is not None:
            module.eval()
        try:
            passages = _embed(encoder, targets.passage_texts, passage_max_length, batch_size, device, fp16)
            queries = _embed(encoder, targets.queries, query_max_length, batch_size, device, fp16)
        finally:
            if module is not None:
                module.train(was_training)
        scores = (queries @ passages.T).double().cpu()
    ids = targets.passage_ids
    position = {pid: i for i, pid in enumerate(ids)}
    company = {t: torch.tensor([p == t for p in targets.passage_tickers], dtype=torch.bool) for t in set(targets.tickers)}
    per_question = []
    for i, pair_id in enumerate(targets.pair_ids):
        keep = torch.ones(len(ids), dtype=torch.bool)
        for pid in targets.removed[i]:
            if pid in position:
                keep[position[pid]] = False
        own = company[targets.tickers[i]]
        row: dict[str, Any] = {"pair_id": pair_id, "relevant": len(targets.relevant[i]),
                               "removed": int((~keep).sum())}
        for mode, mask in (("global", keep), ("company", keep & own)):
            ranked = rank(scores[i], mask, ids)
            first = next((r for r, pid in enumerate(ranked, start=1) if pid in targets.relevant[i]), None)
            row[mode] = {**question_metrics(ranked, targets.relevant[i]), "first_relevant_rank": first}
        per_question.append(row)
    aggregate = {
        mode: {name: sum(q[mode][name] for q in per_question) / len(per_question) for name in METRICS}
        for mode in MODES
    }
    return {
        **aggregate,
        "questions": len(per_question),
        "passages_ranked": len(ids),
        "per_question": per_question,
        "seconds": round(time.perf_counter() - started, 3),
    }


__all__ = ["DEPTH", "METRICS", "MODES", "ValidationTargets", "build_targets", "evaluate", "question_metrics", "rank"]
