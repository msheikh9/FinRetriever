"""The v2 experiment configuration (configs/experiments/v2_metadata_hybrid.yaml) builds and is well formed."""

from pathlib import Path

import pytest

from finretriever.retrieval.pipeline import (
    SEEDLESS,
    component_seeds,
    experiment_config,
    experiment_seeds,
)
from finretriever.utils.io import read_yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/v2_metadata_hybrid.yaml"
ROLES = {"baseline", "primary", "reference", "exploratory"}


@pytest.fixture(scope="module")
def cfg():
    return read_yaml(CONFIG)


def test_every_experiment_builds_with_a_unique_id_and_a_known_role(cfg):
    ids = [e["id"] for e in cfg["experiments"]]
    assert len(ids) == len(set(ids))
    for e in cfg["experiments"]:
        assert e["role"] in ROLES, e["id"]
        assert set(e["components"]) <= set(cfg["components"]), e["id"]
        config = experiment_config(e, cfg["defaults"])
        assert config.name == e["id"]


def test_the_requested_experiments_are_present_with_their_roles(cfg):
    roles = {e["id"]: e["role"] for e in cfg["experiments"]}
    assert roles["E1_a1"] == "baseline"
    for primary in ("E2_a1_company", "E3_a1_year", "E4_hybrid_rrf", "E5_hybrid_rrf_meta", "E6_a1_enriched",
                    "E7_combined", "E8_a1_tail_windows"):
        assert roles[primary] == "primary"


def test_pre_registered_settings(cfg):
    by_id = {e["id"]: experiment_config(e, cfg["defaults"]) for e in cfg["experiments"]}
    e1 = by_id["E1_a1"]
    assert e1.components == ("a1",) and e1.fusion.method == "single" and e1.policy.company == "none"
    e4 = by_id["E4_hybrid_rrf"]
    assert e4.fusion.method == "rrf" and e4.fusion.k == 60 and dict(e4.fusion.weights) == {}
    e7 = by_id["E7_combined"]
    assert e7.components == ("bm25", "a1_enriched")
    assert (e7.policy.company, e7.policy.year, e7.policy.cells) == ("filter", "filter", "interleave")
    assert (e7.policy.after, e7.policy.before_explicit, e7.policy.before_implicit) == (2, 0, 1)
    assert all(by_id[r].policy.source == "oracle" for r in by_id if "oracle" in r)
    assert all(c.policy.source == "parsed" for i, c in by_id.items() if "oracle" not in i)


def test_seed_resolution(cfg):
    assert component_seeds("bm25", cfg) == [SEEDLESS]
    assert component_seeds("a1", cfg) == ["a1_seed42", "a1_seed1337", "a1_seed2026"]
    assert component_seeds("a1_sliding256", cfg) == ["a1_seed42"]
    by_id = {e["id"]: e for e in cfg["experiments"]}
    assert experiment_seeds(by_id["R_bm25"], cfg) == [SEEDLESS]
    assert experiment_seeds(by_id["E4_hybrid_rrf"], cfg) == ["a1_seed42", "a1_seed1337", "a1_seed2026"]
    assert experiment_seeds(by_id["X_a1_sliding256"], cfg) == ["a1_seed42"]


V21 = CONFIG.parent / "v2_1_year_roles.yaml"


def test_the_v2_config_runs_parser_qu_v1_and_v2_1_changes_only_the_parser(cfg):
    """v2.1 re-runs v2 experiments with parser qu-v2: same ids, components, settings, inputs and seeds."""
    v21 = read_yaml(V21)
    assert "version" not in cfg["parser"]  # the runner's default: qu-v1, as the published v2 results were run
    assert v21["parser"] == {**cfg["parser"], "version": "qu-v2"}
    for key in ("corpus_path", "corpus_sha256", "inputs", "seeds", "defaults", "top_k", "device"):
        assert v21[key] == cfg[key], key
    v2_experiments = {e["id"]: e for e in cfg["experiments"]}
    for e in v21["experiments"]:
        assert e["id"] in v2_experiments, e["id"]
        assert experiment_config(e, v21["defaults"]).describe() == \
            experiment_config(v2_experiments[e["id"]], cfg["defaults"]).describe(), e["id"]
        assert e["role"] == v2_experiments[e["id"]]["role"]
        for name in e["components"]:
            assert v21["components"][name] == cfg["components"][name], name
    assert v21["reference_results"]["path"] == "results/experiments/v2_metadata_hybrid"
    roles = {e["id"]: e for e in v21["experiments"]}
    assert roles["E5_hybrid_rrf_meta"]["scope"]["year"] == "boost"  # the release default: no hard year filter
