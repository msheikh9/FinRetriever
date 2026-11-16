"""Tests for the portable smoke-test bundle, its preflight and the results collector (offline, no training of a
real model, no downloads)."""

from __future__ import annotations

import hashlib
import io
import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")  # the bundle's discovery imports the trainer

import yaml  # noqa: E402

from finretriever.training.data_cli import main as data_main  # noqa: E402
from finretriever.training.smoke_bundle import (  # noqa: E402
    MANIFEST_NAME,
    SUMS_NAME,
    BundleError,
    build_bundle,
    extract_bundle,
    forbidden_reason,
    secret_found,
    verify_bundle,
)
from tests.unit.eval_set_fixtures import build_expanded_inputs  # noqa: E402
from tests.unit.training_fixtures import (  # noqa: E402
    SECRET,
    MockTransport,
    WhitespaceTokenizer,
    generic_candidates,
    training_extra_passages,
    write_prep_config,
)

PREP = "configs/train/data_prep_test.yaml"
SMOKE = "configs/train/smoke_test.yaml"
EVALUATION = ("annotation", "dev_ai_v3", "audit", "gold_qrels.jsonl", "gold_queries.jsonl", "gold_eval.metadata.json")
TOKEN_WARNING = ("Token indices sequence length is longer than the specified maximum sequence length for this model "
                 "(739 > 512). Running this sequence through the model will result in indexing errors")
# A console like the first T4 smoke run's: transformers 5.3.0's model-load output (captured locally), the warning
# that run's console showed, and trainer lines.
KAGGLE_CONSOLE = (
    "\rLoading weights:   0%|          | 0/199 [00:00<?, ?it/s]"
    "\rLoading weights: 100%|██████████| 199/199 "
    "[00:00<00:00, 6489.25it/s]\n"
    "\x1b[1mBertModel LOAD REPORT\x1b[0m from: BAAI/bge-base-en-v1.5\n"
    "Key                     | Status     |  | \n"
    "------------------------+------------+--+-\n"
    "embeddings.position_ids | UNEXPECTED |  | \n"
    "\n"
    "\x1b[3mNotes:\n"
    "- UNEXPECTED\x1b[3m\t:can be ignored when loading from different task/architecture; not ok if you expect "
    "identical arch.\x1b[0m\n"
    f"{TOKEN_WARNING}\n"
    "[05_train_biencoder] run bge_base_ft_v2_smoke-seed42: 48 training candidates from train_candidates_v1 (66 "
    "available)\n"
    "[05_train_biencoder] ESTIMATED peak GPU memory (not measured): 8.61 GiB\n"
    "[05_train_biencoder] step 1/15 epoch 0 loss 0.6985 lr 1.00e-05\n"
    "[05_train_biencoder] step 15/15 epoch 4 loss 0.0381 lr 0.00e+00\n"
    "[05_train_biencoder] done: 15/15 steps; checkpoints ['checkpoint-step-000005', 'checkpoint-step-000010', "
    "'checkpoint-step-000015']\n"
)


def data_cli(*argv: str, transport: Any = None) -> int:
    return data_main([argv[0], "--config", PREP, *argv[1:]], tokenizer_loader=lambda s: WhitespaceTokenizer(),
                     transport=transport, sleep=lambda seconds: None, out=io.StringIO(), err=io.StringIO())


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A repository-like directory with relative paths: corpus, preparation, pairs, candidates, smoke config."""
    root = tmp_path_factory.mktemp("repo")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network")))
        mp.setenv("GROQ_API_KEY", SECRET)
        mp.chdir(root)
        build_expanded_inputs(root, training_extra_passages())
        prep = root / PREP
        prep.parent.mkdir(parents=True)
        write_prep_config(root, prep)
        prep.write_text(prep.read_text().replace(f"{root}/", ""))  # paths relative to the repository
        transport = MockTransport(lambda payload: {"questions": generic_candidates(payload)})
        for argv in (("prepare",), ("generate", "--split", "train", "--confirm-api-use"), ("build-pairs",)):
            assert data_cli(*argv, transport=transport) == 0
        pairs = [json.loads(line) for line in
                 (root / "data/processed/train/test_v1/pairs/train_pairs.jsonl").read_text().splitlines()]
        decisions = root / "data/train_review/test_decisions.jsonl"
        decisions.parent.mkdir(parents=True)
        decisions.write_text("".join(json.dumps({
            "pair_id": p["pair_id"], "passage_id": p["positive_passage_id"], "decision": "accept", "issues": [],
            "question_sha256": hashlib.sha256(p["query"].encode()).hexdigest(), "reviewer": "test_review",
            "gold": False}) + "\n" for p in pairs))
        (root / "configs/train/candidates_test.yaml").write_text(yaml.safe_dump({
            "version": "cand_v1", "data_prep_config": PREP,
            "sources": [{"name": "fixture", "decisions": "data/train_review/test_decisions.jsonl"}]}))
        assert data_cli("build-candidates", "--candidates", "configs/train/candidates_test.yaml") == 0
        (root / SMOKE).write_text(yaml.safe_dump({
            "name": "tiny_smoke",
            "base_model": {"name": "fake/whitespace", "revision": "0" * 40, "query_prefix": "Find: ",
                           "passage_prefix": "", "pooling": "cls"},
            "data": {"data_prep_config": PREP, "candidates": "cand_v1", "max_pairs": 6, "subset_seed": 3},
            "training": {"seed": 42, "batch_size": 3, "learning_rate": 0.01, "epochs": 4, "max_steps": 4,
                         "temperature": 0.05, "query_max_length": 64, "passage_max_length": 40, "device": "cpu"},
            "checkpoints": {"output_dir": "models/tiny", "save_every_steps": 2},
        }))
        yield root


@pytest.fixture(scope="module")
def built(repo: Path) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(repo)
        return build_bundle(SMOKE, root=Path("."), output_dir=repo / "dist/kaggle", name="tiny_bundle",
                            say=lambda text: None)


def extracted(built: dict[str, Any], tmp_path: Path) -> Path:
    return extract_bundle(Path(built["archive"]), tmp_path / "x")


def test_the_bundle_holds_exactly_what_the_run_reads(built: dict[str, Any]) -> None:
    manifest = built["manifest"]
    paths = {f["path"]: f for f in manifest["files"]}
    roles = {f["role"] for f in manifest["files"]}
    assert {"code", "script", "config", "corpus", "preparation", "training_candidates", "source_pairs",
            "generation_records", "quality_decisions", "instructions"} <= roles
    for required in (SMOKE, PREP, "data/processed/train/test_v1/manifest.json",
                     "data/processed/train/test_v1/candidates/cand_v1/train_candidates.jsonl",
                     "data/processed/train/test_v1/pairs/pairs.meta.json",
                     "data/processed/train/test_v1/generation/train.records.jsonl",
                     "data/train_review/test_decisions.jsonl", "scripts/05_train_biencoder.py",
                     "scripts/smoke_preflight.py", "src/finretriever/training/biencoder_trainer.py",
                     "requirements-kaggle.txt", "docs/kaggle_smoke_test.md"):
        assert required in paths, required
    # Not read by the run: never bundled.
    assert not any(p.startswith(EVALUATION) or p.startswith(("models/", "dist/", "data/eval")) for p in paths)
    assert "configs/train/candidates_test.yaml" not in paths and "data_prep.yaml" not in paths
    assert not any("/plans/" in p or "/carryover/" in p for p in paths)
    assert all(len(f["sha256"]) == 64 and f["source"] == f["path"] for f in manifest["files"])
    expected = manifest["expected"]["configs"][SMOKE]
    assert (expected["pairs"], expected["steps"], expected["checkpoint_steps"]) == (6, 4, [2, 4])
    assert expected["estimated_gpu_memory_gib"]["total_gib"] > 0
    assert set(manifest["evaluation_sources"]) >= {"annotation", "gold_qrels.jsonl"}
    assert manifest["expected"]["candidates"]["provenance"]["candidates"] == manifest["expected"]["candidates"][
        "available"]


def test_the_archive_is_deterministic_and_extracts_to_a_verified_copy(repo: Path, built: dict[str, Any],
                                                                      tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    again = build_bundle(SMOKE, root=Path("."), output_dir=tmp_path / "again", name="tiny_bundle",
                         say=lambda text: None)
    assert again["sha256"] == built["sha256"]  # the same inputs give the same bytes
    root = extracted(built, tmp_path)
    report = verify_bundle(root)
    assert report["files"] == len(built["manifest"]["files"])
    sums = (root / SUMS_NAME).read_text().splitlines()
    assert len(sums) == report["files"] + 1 and any(line.endswith(f"  {MANIFEST_NAME}") for line in sums)
    for source in built["manifest"]["files"]:  # the originals are untouched
        origin = repo / source["path"] if (repo / source["path"]).exists() else None
        if origin is not None:
            assert hashlib.sha256(origin.read_bytes()).hexdigest() == source["sha256"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: (r / "data/train_review/test_decisions.jsonl").write_text("{}\n"), "SHA-256 differs"),
        (lambda r: (r / "notes.txt").write_text("extra"), "not in the manifest"),
        (lambda r: (r / "annotation").mkdir(), "evaluation data present"),
        (lambda r: (r / "data/eval").mkdir(parents=True), "evaluation data present"),
        (lambda r: (r / "configs/train/data_prep_test.yaml").unlink(), "missing"),
    ],
)
def test_a_changed_copy_is_refused(built: dict[str, Any], tmp_path: Path, change: Any, message: str) -> None:
    root = extracted(built, tmp_path)
    (root / "models" / "tiny").mkdir(parents=True)  # runtime outputs are allowed
    (root / "smoke_outputs").mkdir()
    (root / "smoke_outputs" / "preflight.json").write_text("{}")
    verify_bundle(root)
    change(root)
    with pytest.raises(BundleError, match=message):
        verify_bundle(root)


def test_evaluation_data_and_credentials_are_never_bundled(repo: Path, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    for path in ("data/eval/gold_qrels.jsonl", "data/raw/AAPL/x.html", "data/cache/embeddings/a.npy", ".env",
                 "results/dev/run.json", "models/x/model.safetensors", "/etc/passwd", "../outside.txt",
                 "configs/kaggle.json", "keys/server.pem"):
        assert forbidden_reason(path) is not None, path
    assert forbidden_reason("data/processed/corpus_v1.jsonl") is None
    assert secret_found(("gsk_" + "A" * 30).encode()) and secret_found(("hf_" + "b" * 34).encode())
    assert secret_found(b"api_key_env: GROQ_API_KEY") is None  # naming the variable is fine
    monkeypatch.chdir(repo)
    leaky = repo / "configs/train/smoke_leaky.yaml"
    leaky.write_text((repo / SMOKE).read_text() + f"# key: {SECRET}\n")
    try:
        with pytest.raises(BundleError, match="looks like a credential"):
            build_bundle("configs/train/smoke_leaky.yaml", root=Path("."), output_dir=tmp_path / "out",
                         name="leaky", say=lambda text: None)
    finally:
        leaky.unlink()
    assert not (tmp_path / "out" / "leaky.tar.gz").exists()


def _tiny_encoder(sha: str | None) -> Any:
    from tests.unit.trainer_fixtures import TinyEncoder

    return TinyEncoder(tokenizer_sha256=sha)


def test_the_preflight_checks_everything_and_trains_nothing(built: dict[str, Any], tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.training import smoke_preflight

    root = extracted(built, tmp_path)
    monkeypatch.chdir(root)
    snapshot = tmp_path / "hub" / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    huggingface_hub = pytest.importorskip("huggingface_hub")  # not installed in CI
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(snapshot / "config.json"))
    sha = json.loads((root / "data/processed/train/test_v1/manifest.json").read_text())["visible_text"][
        "identity"]["tokenizer_sha256"]
    report = smoke_preflight.run_preflight(SMOKE, skip_cuda=True, offline=True,
                                           encoder_loader=lambda model: _tiny_encoder(sha))
    status = {c["name"]: c["status"] for c in report["checks"]}
    assert status == {"bundle": "pass", "config": "pass", "corpus_and_preparation": "pass", "candidates": "pass",
                      "provenance": "pass", "smoke_subset": "pass", "schedule": "pass",
                      "evaluation_isolation": "pass", "model_and_tokenizer": "pass", "cuda_fp16": "skipped",
                      "gpu_memory": "skipped"}
    assert report["failed"] == [] and not report["ready_for_training"] and not report["trained"]
    assert not (root / "models").exists()  # nothing trained, no model written
    isolation = next(c for c in report["checks"] if c["name"] == "evaluation_isolation")["data"]
    assert isolation["opened"] == [] and isolation["files_opened"] > 0
    # Another tokenizer, or an edited config, is a failure.
    report = smoke_preflight.run_preflight(SMOKE, skip_cuda=True, offline=True,
                                           encoder_loader=lambda model: _tiny_encoder("0" * 64))
    assert "model_and_tokenizer" in report["failed"]
    (root / SMOKE).write_text((root / SMOKE).read_text().replace("max_pairs: 6", "max_pairs: 5"))
    report = smoke_preflight.run_preflight(SMOKE, skip_cuda=True, offline=True, skip_model_load=True)
    assert {"bundle", "config"} <= set(report["failed"])
    out = io.StringIO()
    code = smoke_preflight.main(["--config", SMOKE, "--skip-cuda", "--offline", "--skip-model-load",
                                 "--report", str(tmp_path / "pre.json")], out=out)
    assert code == 1 and "NOT READY" in out.getvalue() and json.loads((tmp_path / "pre.json").read_text())["failed"]


def test_the_results_summary_reports_mechanics_and_claims_nothing(built: dict[str, Any], tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.training.biencoder_trainer import train
    from finretriever.training.smoke_results import (
        BENIGN_LOAD_KEYS,
        STATEMENT,
        collect_results,
        summary_markdown,
    )
    from finretriever.training.trainer_setup import load_trainer_config, prepare_inputs

    root = extracted(built, tmp_path)
    monkeypatch.chdir(root)
    config, sha = load_trainer_config(SMOKE)
    result = train(prepare_inputs(config, sha), _tiny_encoder(None))
    console = tmp_path / "console.log"
    console.write_text(KAGGLE_CONSOLE, encoding="utf-8")
    found = collect_results(result.run_dir, console_log=console, verify_checkpoints=False, compare_base=False)
    assert found["steps"] == {"planned": 4, "completed": 4, "complete": True, "checkpoints_expected": [2, 4],
                              "checkpoints_found": [2, 4]}
    assert found["loss"]["initial"] == result.history[0]["loss"] and found["loss"]["final"] == result.history[-1][
        "loss"]
    assert found["loss"]["non_finite"] == 0 and found["gradients"]["non_finite_norms"] == 0
    assert found["gpu_memory"] is None and found["run"]["seed"] == 42 and found["run"]["revision"] == "0" * 40
    assert found["resume_state"] == {"step": 4, "schedule_matches": True, "older_checkpoints_hold_state": False}
    warnings = found["warnings"]
    assert warnings["needs_review"] == 1 and warnings["needs_review_by_category"] == {"tokenizer": 1}
    assert warnings["lines"][0] == {"line": TOKEN_WARNING, "category": "tokenizer", "count": 1, "known_benign": False}
    assert warnings["known_benign"] == 6 and all(w["category"] == "model_load" for w in warnings["lines"][1:])
    assert warnings["load_reports"] == [[{"key": "embeddings.position_ids", "status": "UNEXPECTED",
                                          "known_benign": BENIGN_LOAD_KEYS[("embeddings.position_ids", "UNEXPECTED")]}]]
    assert not [w for w in warnings["lines"] if "05_train_biencoder" in w["line"] or "Loading weights" in w["line"]]
    # The tiny model is not a Hugging Face checkpoint, so the mechanics check cannot pass on it.
    assert found["checkpoints"][0]["missing_files"] and found["mechanics_ok"] is False
    text = summary_markdown(found)
    assert STATEMENT in text and "| Steps completed | 4 of 4 |" in text
    assert "1 distinct lines need review (tokenizer 1)" in text and f"- tokenizer: `{TOKEN_WARNING}` (1x)" in text
    assert "embeddings.position_ids UNEXPECTED (known benign)" in text and "none need review" not in text
    assert "improve" not in text.replace("no retrieval improvement is claimed", "")


def test_the_console_scan_reports_every_library_line(tmp_path: Path) -> None:
    from finretriever.training.smoke_results import console_warnings

    def scan(text: str) -> dict[str, Any]:
        log = tmp_path / "console.log"
        log.write_text(text, encoding="utf-8")
        return console_warnings(log)

    # The smoke run's warning names no warning, error or CUDA: the keyword scan it replaces found nothing.
    assert scan(TOKEN_WARNING + "\n")["lines"][0]["category"] == "tokenizer"
    clean = KAGGLE_CONSOLE.replace(TOKEN_WARNING + "\n", "")
    assert scan(clean)["needs_review"] == 0 and scan(clean)["known_benign"] == 6
    missing = scan(clean.replace("embeddings.position_ids | UNEXPECTED", "pooler.dense.weight     | MISSING   "))
    assert missing["needs_review"] == 6 and missing["load_reports"][0] == [
        {"key": "pooler.dense.weight", "status": "MISSING", "known_benign": None}]
    found = scan("[05_train_biencoder] error: CUDA out of memory\n/x.py:1: UserWarning: autocast is deprecated\n"
                 "RuntimeWarning: overflow encountered in exp\nloss is nan\nsomething else\n"
                 "[05_train_biencoder] step 2/15 epoch 0 loss nan lr 1e-5\n")
    assert {w["line"]: w["category"] for w in found["lines"]} == {
        "[05_train_biencoder] error: CUDA out of memory": "cuda_or_memory",
        "/x.py:1: UserWarning: autocast is deprecated": "warning",
        "RuntimeWarning: overflow encountered in exp": "numerical",
        "loss is nan": "numerical",
        "something else": "other",
    }
    assert console_warnings(None)["note"] == "no console log given"
    assert console_warnings(tmp_path / "absent.log")["note"] == "console log not found"


def test_probe_encoding_of_a_hugging_face_checkpoint(tmp_path: Path) -> None:
    """A tiny BERT saved as the trainer saves checkpoints reloads, encodes finitely and matches
    sentence-transformers (skipped where transformers cannot load BERT)."""
    torch = pytest.importorskip("torch")
    try:
        from transformers import BertConfig, BertModel
        try:
            from transformers import BertTokenizerFast as BertTokenizer
        except ImportError:
            from transformers import BertTokenizer  # type: ignore[assignment]
    except Exception as exc:
        pytest.skip(f"transformers BERT unavailable: {exc}")
    from finretriever.training.biencoder_trainer import HFEncoder
    from finretriever.training.smoke_results import _encode_probe

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "net", "income", "the", "of", "cash"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(words) + "\n")
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=600)
    out = tmp_path / "checkpoint-step-000001"
    out.mkdir()
    HFEncoder(BertModel(config), BertTokenizer(vocab_file=str(vocab)), "cls").save(out, max_seq_length=512)
    probe = _encode_probe(out, "cls")
    assert probe["finite"] and probe["dimension"] == 16
    if "sentence_transformers_max_abs_diff" in probe:
        assert probe["sentence_transformers"] == "loads"


def test_the_guide_and_requirements_never_install_torch_or_name_secrets() -> None:
    root = Path(__file__).parents[2]
    requirements = (root / "requirements-kaggle.txt").read_text()
    assert not any(line.split("#")[0].strip().lower().startswith("torch") for line in requirements.splitlines())
    assert "tokenizers==0.22.2" in requirements and "transformers==5.3.0" in requirements
    guide = (root / "docs/kaggle_smoke_test.md").read_text()
    assert "--confirm-training" in guide and "Private" in guide and "-c /tmp/torch.txt" in guide
    assert secret_found(guide.encode()) is None and secret_found(requirements.encode()) is None


# --- a validated fine-tuning run: validation set, seeds, GradCache ---------------------------------

TRAIN = "configs/train/train_test.yaml"
VALIDATION = "configs/train/validation_test.yaml"


@pytest.fixture(scope="module")
def training_repo(repo: Path) -> Path:
    """The smoke repository plus a validation set (mocked generation, AI-review decisions) and a validated config
    with three seeds and GradCache, all with relative paths."""
    from finretriever.training.validation_set import REVIEW_ASPECTS, pair_id_for
    from tests.unit.validation_fixtures import validation_reply

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network")))
        mp.setenv("GROQ_API_KEY", SECRET)
        mp.chdir(repo)
        assert data_cli("plan", "--split", "validation", "--name", "val_a", "--size", "3", "--method",
                        "balanced_years", "--max-company-share", "1.0") == 0
        assert data_cli("generate", "--split", "validation", "--plan", "val_a", "--max-requests", "3",
                        "--confirm-api-use", transport=MockTransport(validation_reply)) == 0
        prepared = repo / "data/processed/train/test_v1"
        records = {r["passage_id"]: r for r in map(json.loads, (prepared / "generation/validation.records.jsonl")
                                                   .read_text().splitlines())}
        plan = json.loads((prepared / "plans/validation.val_a.json").read_text())["passages"]
        decisions = repo / "data/validation_review/test_val_decisions.jsonl"
        decisions.parent.mkdir(parents=True, exist_ok=True)
        decisions.write_text("".join(json.dumps({
            "pair_id": pair_id_for(position), "passage_id": p["passage_id"], "decision": "accept", "issues": [],
            "question_sha256": hashlib.sha256(c["question"].encode()).hexdigest(),
            "aspects": dict.fromkeys(REVIEW_ASPECTS, "pass"), "reviewer": "ai_review", "gold": False}) + "\n"
            for position, p in enumerate(plan, start=1) for c in records[p["passage_id"]]["accepted"]))
        (repo / VALIDATION).write_text(yaml.safe_dump({
            "version": "val_v1", "data_prep_config": PREP, "plan": "val_a",
            "decisions": "data/validation_review/test_val_decisions.jsonl", "training_candidates": ["cand_v1"]}))
        assert data_cli("build-validation", "--validation", VALIDATION) == 0
        (repo / TRAIN).write_text(yaml.safe_dump({
            "name": "tiny_train",
            "base_model": {"name": "fake/whitespace", "revision": "0" * 40, "query_prefix": "Find: ",
                           "passage_prefix": "", "pooling": "cls"},
            "data": {"data_prep_config": PREP, "candidates": "cand_v1"},
            "training": {"seeds": [42, 1337, 2026], "batch_size": 3, "mini_batch_size": 2, "learning_rate": 0.01,
                         "epochs": 10, "max_steps": 6, "temperature": 0.05, "query_max_length": 64,
                         "passage_max_length": 40, "device": "cpu"},
            "checkpoints": {"output_dir": "models/tiny_train"},
            "validation": {"version": "val_v1", "batch_size": 4},
        }))
    return repo


@pytest.fixture(scope="module")
def training_built(training_repo: Path) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(training_repo)
        return build_bundle(TRAIN, root=Path("."), output_dir=training_repo / "dist/kaggle", name="tiny_train_bundle",
                            say=lambda text: None)


def test_a_training_bundle_holds_the_validation_set_and_every_seeds_schedule(training_built: dict[str, Any]) -> None:
    from finretriever.training.checkpoint_selection import evaluation_steps

    manifest = training_built["manifest"]
    assert manifest["kind"] == "finretriever_training_bundle" and "validation set" in manifest["purpose"]
    paths = {f["path"]: f["role"] for f in manifest["files"]}
    prepared = "data/processed/train/test_v1"
    for path, role in ((f"{prepared}/validation_sets/val_v1/validation_pairs.jsonl", "validation_set"),
                       (f"{prepared}/validation_sets/val_v1/manifest.json", "validation_set"),
                       ("data/validation_review/test_val_decisions.jsonl", "validation_decisions"),
                       (f"{prepared}/plans/validation.val_a.json", "sampling_plan"),
                       (f"{prepared}/generation/validation.records.jsonl", "generation_records"),
                       ("docs/kaggle_training.md", "instructions"),
                       ("src/finretriever/training/validation_eval.py", "code"),
                       ("src/finretriever/training/checkpoint_selection.py", "code")):
        assert paths.get(path) == role, path
    assert "docs/kaggle_smoke_test.md" not in paths and VALIDATION not in paths
    assert not any(p.startswith(EVALUATION) or p.startswith("data/eval") for p in paths)
    expected = manifest["expected"]["configs"][TRAIN]
    assert expected["validation"]["version"] == "val_v1" and expected["validation"]["pairs"] == 2
    assert expected["mini_batch_size"] == 2 and sorted(expected["seeds"]) == ["1337", "2026", "42"]
    for seed, run in expected["seeds"].items():
        assert run["run"] == f"tiny_train-seed{seed}" and run["checkpoint_steps"] == evaluation_steps(run["steps"])
    assert len({run["schedule_sha256"] for run in expected["seeds"].values()}) == 3


def test_the_training_preflight_checks_validation_seeds_and_gradcache(training_built: dict[str, Any], tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.training import smoke_preflight

    root = extract_bundle(Path(training_built["archive"]), tmp_path / "x")
    monkeypatch.chdir(root)
    snapshot = tmp_path / "hub" / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    huggingface_hub = pytest.importorskip("huggingface_hub")  # not installed in CI
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(snapshot / "config.json"))
    sha = json.loads((root / "data/processed/train/test_v1/manifest.json").read_text())["visible_text"][
        "identity"]["tokenizer_sha256"]
    report = smoke_preflight.run_preflight(TRAIN, skip_cuda=True, offline=True,
                                           encoder_loader=lambda model: _tiny_encoder(sha))
    status = {c["name"]: c["status"] for c in report["checks"]}
    assert status == {"bundle": "pass", "config": "pass", "corpus_and_preparation": "pass", "candidates": "pass",
                      "provenance": "pass", "smoke_subset": "pass", "schedule": "pass", "validation_set": "pass",
                      "evaluation_isolation": "pass", "model_and_tokenizer": "pass", "cuda_fp16": "skipped",
                      "gradcache": "pass", "gpu_memory": "skipped"}
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["gradcache"]["data"]["relative_difference"] < 1e-5 and checks["gradcache"]["data"]["device"] == "cpu"
    assert set(checks["schedule"]["data"]["seeds"]) == {"42", "1337", "2026"}
    assert "isolated from the" in checks["validation_set"]["detail"] and not report["trained"]
    assert not (root / "models").exists()


def test_a_validated_run_in_the_bundle_is_collected_with_its_selection(training_built: dict[str, Any], tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.training.biencoder_trainer import main
    from finretriever.training.checkpoint_selection import read_selection_record
    from finretriever.training.smoke_results import (
        VALIDATED_STATEMENT,
        collect_results,
        summary_markdown,
    )

    root = extract_bundle(Path(training_built["archive"]), tmp_path / "x")
    monkeypatch.chdir(root)
    out = io.StringIO()
    assert main(["--config", TRAIN, "--seed", "1337", "--confirm-training"],
                encoder_loader=lambda model: _tiny_encoder(None), out=out) == 0
    run_dir = root / "models/tiny_train/tiny_train-seed1337"
    verify_bundle(root)  # the run's outputs are runtime outputs: the bundle still verifies
    record = read_selection_record(run_dir / "selection.json")
    found = collect_results(run_dir, verify_checkpoints=False, compare_base=False)
    assert found["statement"] == VALIDATED_STATEMENT and found["steps"]["complete"]
    assert found["steps"]["checkpoints_expected"] == [e["step"] for e in record["evaluations"]]
    assert found["selection"]["hashes_verified"] is True and found["selection"]["selected"] == record["selected"]
    text = summary_markdown(found)
    assert "# Trainer run: tiny_train-seed1337" in text and "Selected checkpoint" in text
    assert "improve" not in text.replace("no retrieval improvement is claimed", "")
    other = next(e["checkpoint"] for e in record["evaluations"] if e["step"] != record["selected"]["step"])
    (run_dir / other / "encoder.json").write_text("{}")
    assert collect_results(run_dir, verify_checkpoints=False, compare_base=False)["selection"]["changed_checkpoints"] == [
        other]


def test_the_gpu_rehearsal_ships_in_the_bundle_and_runs_from_it(training_built: dict[str, Any], tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from finretriever.training.gpu_rehearsal import main
    from tests.unit.trainer_fixtures import TinyAttentionEncoder

    paths = {f["path"]: f["role"] for f in training_built["manifest"]["files"]}
    assert "scripts/gpu_rehearsal.py" in paths and paths.get("src/finretriever/training/gpu_rehearsal.py") == "code"
    assert paths.get("src/finretriever/retrieval/checkpoint.py") == "code"  # check 9's sentence-transformers reload
    root = extract_bundle(Path(training_built["archive"]), tmp_path / "x")
    monkeypatch.chdir(root)
    out = io.StringIO()
    assert main(["--config", TRAIN, "--allow-cpu"], out=out, encoder_loader=lambda m: TinyAttentionEncoder()) == 3
    report = json.loads((root / "rehearsal_outputs" / "rehearsal.json").read_text())
    assert not report["failed"] and report["trained"] is False and not (root / "models").exists()
    verify_bundle(root)  # the rehearsal's outputs are runtime outputs: the bundle still verifies
