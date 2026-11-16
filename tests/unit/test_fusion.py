"""Reciprocal Rank Fusion (finretriever.retrieval.fusion)."""

import pytest

from finretriever.retrieval.fusion import FusionConfig, reciprocal_rank_fusion


def test_rrf_scores_are_sums_of_reciprocal_ranks():
    fused = dict(reciprocal_rank_fusion({"bm25": [1, 2, 3], "dense": [3, 1]}, k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)


def test_rrf_order_ties_and_depth():
    # 5 and 7 tie on fused score; 7 has the better best rank... both rank 1 once: corpus position decides
    out = reciprocal_rank_fusion({"a": [7, 5], "b": [5, 7]}, k=10)
    assert [p for p, _ in out] == [5, 7]
    cut = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [4]}, k=60, depth=1)
    assert sorted(p for p, _ in cut) == [1, 4]


def test_weights_change_the_order():
    rankings = {"bm25": [1, 2], "dense": [2, 1]}
    assert reciprocal_rank_fusion(rankings, k=60, weights={"bm25": 2.0})[0][0] == 1
    assert reciprocal_rank_fusion(rankings, k=60, weights={"dense": 2.0})[0][0] == 2


def test_rrf_is_deterministic_and_rejects_duplicates():
    rankings = {"x": [3, 1, 2], "y": [2, 3, 1], "z": [1, 2, 3]}
    assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(dict(reversed(rankings.items())))
    with pytest.raises(ValueError, match="twice"):
        reciprocal_rank_fusion({"x": [1, 1]})


def test_fusion_config_validation():
    assert FusionConfig("rrf", k=60, weights={"bm25": 1.0}).describe()["k"] == 60
    for bad in (dict(method="max"), dict(k=-1), dict(depth=0), dict(weights={"a": -1.0})):
        with pytest.raises(ValueError):
            FusionConfig(**bad)
