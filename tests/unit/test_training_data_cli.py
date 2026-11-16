"""End-to-end tests of the training-data pipeline CLI with a mocked Groq API (synthetic corpus, offline)."""

from __future__ import annotations

import hashlib
import io
import json
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from finretriever.eval_set.annotation_store import load_corpus_passages
from finretriever.training.candidates import load_candidates, verify_candidate_provenance
from finretriever.training.data_cli import main
from finretriever.training.data_prep import (
    load_data_prep_config,
    open_prepared,
    verify_frozen_corpus,
)
from finretriever.training.evidence_reuse import Equivalent, EvidenceLocator
from finretriever.training.pair_quality import check_pairs, load_evaluation_questions, read_pairs
from finretriever.training.query_generation import PROMPT_VERSION, TransportResponse, prompt_sha256
from tests.unit.eval_set_fixtures import ExpandedInputs, build_expanded_inputs
from tests.unit.training_fixtures import (
    SECRET,
    MockTransport,
    WhitespaceTokenizer,
    accept_all,
    extra_ids,
    generic_candidates,
    training_extra_passages,
    write_prep_config,
)

EXTRA = extra_ids()
SHARED = "How did Gamma Energy describe the drivers of its revenue change?"
EVAL_COPY = "How large was Gamma's committed credit facility at December 31, 2024?"  # ~ fixture question q_0002


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


def respond(payload: Mapping[str, Any]) -> Any:
    """Canned model replies for a few passages; a generic valid + copied pair otherwise."""
    user = payload["messages"][-1]["content"]
    if "$900 million of borrowings" in user:
        return {"questions": [
            {"question": EVAL_COPY, "type": "debt_liquidity", "answer": "$2,000 million",
             "evidence": "$900 million of borrowings outstanding under its $2,000 million revolving credit facility"},
            {"question": "How much did Gamma Energy borrow under its credit facility at the end of 2024?",
             "type": "debt_liquidity", "answer": "$900 million",
             "evidence": "At December 31, 2024, Gamma Energy had $900 million of borrowings outstanding"},
        ]}
    if "revenue of $3,400 million" in user:
        return {"questions": [
            {"question": SHARED, "type": "revenue", "answer": "higher natural gas prices",
             "evidence": "driven by higher natural gas prices"},
            {"question": "Did Gamma Energy revenue exceed $3.4 billion in fiscal 2024?", "type": "revenue",
             "answer": "yes, $3.4 billion", "evidence": "Gamma Energy reported revenue of $3,400 million for 2024"},
        ]}
    if "revenue of $3,036 million for 2023" in user:
        return {"questions": [{"question": SHARED, "type": "revenue", "answer": "lower oil prices",
                               "evidence": "as lower oil prices offset higher production volumes"}]}
    return {"questions": generic_candidates(payload)}


@pytest.fixture
def setup(tmp_path: Path) -> tuple[ExpandedInputs, Path]:
    inputs = build_expanded_inputs(tmp_path, training_extra_passages())
    return inputs, write_prep_config(tmp_path)


def cli(config: Path, *argv: str, transport: Any = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main([argv[0], "--config", str(config), *argv[1:]], tokenizer_loader=lambda settings: WhitespaceTokenizer(),
                 transport=transport, sleep=lambda seconds: None, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_pipeline_end_to_end_with_mocked_generation(setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, config_path = setup
    root = inputs.root
    protected = {name: tree(root / name) for name in ("annotation", "audit", "dev_ai_v3", "corpus")}
    outputs: list[str] = []

    def run(*argv: str, transport: Any = None) -> tuple[int, str, str]:
        result = cli(config_path, *argv, transport=transport)
        outputs.extend(result[1:])
        return result

    code, out, err = run("prepare")
    assert code == 0, err
    assert "blocked: 13 passages (12 evaluation candidates + 1 in their groups)" in out
    assert run("prepare")[1].startswith("[02_generate_queries] unchanged")

    transport = MockTransport(respond)
    code, out, _ = run("generate", "--split", "train", transport=transport)
    assert code == 0 and "dry run: no request sent" in out and transport.calls == []
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    code, _, err = run("generate", "--split", "train", "--confirm-api-use", transport=transport)
    assert code == 2 and "set GROQ_API_KEY" in err and transport.calls == []

    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert run("generate", "--split", "train", "--confirm-api-use", "--limit", "2", transport=transport)[0] == 0
    assert len(transport.calls) == 2
    assert run("generate", "--split", "train", "--confirm-api-use", transport=transport)[0] == 0
    assert run("generate", "--split", "validation", "--confirm-api-use", transport=transport)[0] == 0
    config, _ = load_data_prep_config(config_path)
    corpus = verify_frozen_corpus(config)
    passages = load_corpus_passages(corpus)
    prepared = open_prepared(config.output_dir, corpus, passages)
    approved = [p for s in ("train", "validation") for p in prepared.source_passages(s, passages, min_words=5)]
    assert len(transport.calls) == len(approved) == 10  # each approved passage requested once across resumes
    assert transport.calls[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert run("generate", "--split", "train", "--confirm-api-use", transport=transport)[0] == 0
    assert len(transport.calls) == 10  # nothing left to do

    code, out, err = run("build-pairs")
    assert code == 0, err
    pairs = read_pairs(prepared)
    meta = json.loads((Path(config.output_dir) / "pairs" / "pairs.meta.json").read_text())
    rejected = meta["stats"]["rejected_by_reason"]
    assert rejected["evaluation_overlap"] == 1 and rejected["duplicate_query"] == 1
    assert rejected["number_not_in_passage"] == 1 and rejected["copied_from_passage"] >= 5
    queries = [p["query"] for s in pairs for p in pairs[s]]
    assert EVAL_COPY not in queries and queries.count(SHARED) == 1
    for split, rows in pairs.items():
        for pair in rows:
            row = prepared.rows[pair["positive_passage_id"]]
            assert row["split"] == split != "blocked" and pair["group_id"] == row["group_id"]
            assert (pair["provenance"]["model"], pair["provenance"]["prompt_version"]) == ("test-model", PROMPT_VERSION)
    assert EXTRA["acme22_competition"] not in {p["positive_passage_id"] for s in pairs for p in pairs[s]}
    chain = {prepared.rows[EXTRA[n]]["split"] for n in ("gamma24_mdna", "gamma23_mdna", "gamma22_mdna")}
    assert len(chain) == 1

    manifest = prepared.manifest
    assert meta["corpus"]["sha256"] == corpus.corpus_sha256
    assert meta["preparation"]["blocklist_sha256"] == manifest["files"]["blocklist.jsonl"]
    assert meta["preparation"]["split"]["seed"] == 7 and meta["preparation"]["overlap"]["containment_threshold"] == 0.5
    assert meta["generation"]["prompt_sha256"] == prompt_sha256() and meta["generation"]["model"] == "test-model"
    assert meta["generation"]["params"]["temperature"] == 0.7 and set(meta["generation"]["runs"]) == {"train", "validation"}
    assert "semantic leakage is not ruled out" in meta["leakage"]["note"]
    for name, digest in meta["files"].items():
        assert hashlib.sha256((Path(config.output_dir) / "pairs" / name).read_bytes()).hexdigest() == digest
    first = tree(Path(config.output_dir) / "pairs")
    code, out, _ = run("build-pairs")  # deterministic: an identical rebuild writes nothing
    assert code == 0 and "unchanged (identical pairs present)" in out and tree(Path(config.output_dir) / "pairs") == first
    code, out, _ = run("check")
    assert code == 0 and "quality checks passed" in out

    every_file = "".join(p.read_text(errors="ignore") for p in root.rglob("*") if p.is_file())
    assert SECRET not in every_file and SECRET not in "".join(outputs)  # the key is never written or printed
    assert {name: tree(root / name) for name in protected} == protected

    pairs_file = Path(config.output_dir) / "pairs" / "train_pairs.jsonl"
    pairs_file.write_text(pairs_file.read_text().replace("Gamma", "Gamma Energy", 1))
    code, _, err = run("check")
    assert code == 1 and "does not match" in err


def test_quality_checks_catch_leaked_split_blocked_and_duplicate_pairs(setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, config_path = setup
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    for argv in (("prepare",), ("generate", "--split", "train", "--confirm-api-use"),
                 ("generate", "--split", "validation", "--confirm-api-use"), ("build-pairs",)):
        assert cli(config_path, *argv, transport=MockTransport(respond))[0] == 0
    config, _ = load_data_prep_config(config_path)
    corpus = verify_frozen_corpus(config)
    passages = load_corpus_passages(corpus)
    prepared = open_prepared(config.output_dir, corpus, passages)
    questions = load_evaluation_questions(config.leakage.evaluation_question_files)
    pairs = read_pairs(prepared)
    assert check_pairs(pairs, config, prepared, passages, questions) == []
    train = pairs["train"]

    def violations(changed: dict[str, list[dict[str, Any]]]) -> str:
        return " | ".join(check_pairs(changed, config, prepared, passages, questions))

    moved = {"train": train[1:], "validation": [*pairs["validation"], {**train[0], "split": "validation"}]}
    assert "belongs to train" in violations(moved)
    blocked = {**pairs, "train": [{**train[0], "positive_passage_id": EXTRA["acme22_competition"]}, *train[1:]]}
    assert "evaluation blocklist" in violations(blocked)
    doubled = {**pairs, "train": [*train, {**train[0], "pair_id": "train_999999"}]}
    assert "duplicate query-passage pair" in violations(doubled)
    invented = {**pairs, "train": [{**train[0], "answer": "$123,456 million"}, *train[1:]]}
    assert "number_not_in_passage" in violations(invented) or "answer_not_supported" in violations(invented)
    copied = {**pairs, "train": [{**train[0], "query": "How large was Beta's committed credit facility at December 31, 2024?"}, *train[1:]]}
    assert "evaluation question" in violations(copied) or "candidate check" in violations(copied)


def test_generation_stops_on_rejected_credentials_and_leaks_nothing(setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, config_path = setup
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert cli(config_path, "prepare")[0] == 0
    denied = MockTransport(lambda payload: TransportResponse(401, {}, f"invalid api key {SECRET}"))
    code, out, err = cli(config_path, "generate", "--split", "train", "--confirm-api-use", transport=denied)
    assert code == 2 and "rejected the credentials" in err and len(denied.calls) == 1
    assert SECRET not in out + err
    generation = Path(load_data_prep_config(config_path)[0].output_dir) / "generation"
    assert (generation / "train.records.jsonl").read_text() == ""  # nothing recorded for the refused request
    code, _, err = cli(config_path, "build-pairs")
    assert code == 0  # an empty run builds empty pairs
    write_prep_config(inputs.root, config_path, leakage={"evaluation_question_files": [str(inputs.root / "missing.jsonl")]})
    code, _, err = cli(config_path, "build-pairs")
    assert code == 1 and "missing" in err


def test_a_pilot_uses_a_company_spread_sample_and_never_exceeds_its_request_budget(
    setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config_path = setup
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert cli(config_path, "prepare")[0] == 0

    def with_usage(payload: Mapping[str, Any]) -> TransportResponse:
        body = {"choices": [{"message": {"content": json.dumps({"questions": generic_candidates(payload)})}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 90, "total_tokens": 390}}
        return TransportResponse(200, {"x-ratelimit-remaining-requests": "14000"}, json.dumps(body))

    transport = MockTransport(with_usage)
    code, out, _ = cli(config_path, "generate", "--split", "train", "--sample", "4", transport=transport)
    assert code == 0 and "sample of 4 passages (seed 7): GAMMA 4" in out and "request budget none" in out
    assert "dry run" in out and transport.calls == []
    argv = ("generate", "--split", "train", "--sample", "4", "--confirm-api-use")
    code, out, err = cli(config_path, *argv, "--max-requests", "3", transport=transport)
    assert code == 0, err
    assert len(transport.calls) == 3 and "requests sent: 3" in out and "stopped at the request budget" in out
    assert "'total_tokens': 1170" in out and "x-ratelimit-remaining-requests" in out
    code, out, _ = cli(config_path, *argv, "--max-requests", "3", transport=transport)
    assert code == 0 and len(transport.calls) == 4 and "'skipped_done': 3" in out  # resumed, not repeated
    records = Path(load_data_prep_config(config_path)[0].output_dir) / "generation" / "train.records.jsonl"
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    assert len(rows) == len({r["passage_id"] for r in rows}) == 4 and all(r["usage"]["total_tokens"] == 390 for r in rows)
    assert SECRET not in records.read_text() + out


def test_the_pilot_incident_now_stops_after_one_request_and_a_new_model_needs_a_new_run(
    setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, config_path = setup
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert cli(config_path, "prepare")[0] == 0
    body = ('{"error":{"message":"The model `test-model` does not exist or you do not have access to it.",'
            '"type":"invalid_request_error","code":"model_not_found"}}')
    missing = MockTransport(lambda payload: TransportResponse(404, {}, body))
    argv = ("generate", "--split", "train", "--sample", "4", "--confirm-api-use")
    code, out, err = cli(config_path, *argv, "--max-requests", "50", transport=missing)
    assert code == 2 and len(missing.calls) == 1 and "requests sent: 1" in out
    assert "model_not_found" in err and "generation.model" in err and SECRET not in out + err
    generation = Path(load_data_prep_config(config_path)[0].output_dir) / "generation"
    assert (generation / "train.records.jsonl").read_text() == ""

    # Records of an earlier failed run (as the pilot left them) bind the directory to the old model.
    failed = MockTransport(lambda payload: TransportResponse(400, {}, '{"error":{"code":"invalid_value"}}'))
    cli(config_path, *argv, "--limit", "1", transport=failed)
    write_prep_config(inputs.root, config_path, generation={"model": "other-model"})
    code, _, err = cli(config_path, *argv, "--max-requests", "1", transport=MockTransport(respond))
    assert code == 1 and "different run" in err
    generation.rename(generation.with_name("generation.failed-test-model"))  # kept, not deleted
    working = MockTransport(respond)
    code, out, err = cli(config_path, *argv, "--limit", "1", "--max-requests", "1", transport=working)
    assert code == 0, err
    assert len(working.calls) == 1 and working.calls[0]["payload"]["model"] == "other-model"
    assert "'done': 1" in out


def test_the_one_request_qwen_pilot_sends_one_request_with_the_qwen_settings(
    setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, config_path = setup
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    qwen = {"model": "qwen/qwen3.8-27b", "temperature": 0.7, "max_completion_tokens": 2048,
            "reasoning_format": "parsed", "reasoning_effort": "none"}
    write_prep_config(inputs.root, config_path, generation=qwen)
    assert cli(config_path, "prepare")[0] == 0

    def qwen_reply(payload: Mapping[str, Any]) -> TransportResponse:
        message = {"content": json.dumps({"questions": generic_candidates(payload)}), "reasoning": "ignored"}
        usage = {"prompt_tokens": 400, "completion_tokens": 110, "total_tokens": 510,
                 "completion_tokens_details": {"reasoning_tokens": 0}}
        return TransportResponse(200, {}, json.dumps({"choices": [{"message": message, "finish_reason": "stop"}],
                                                      "usage": usage}))

    transport = MockTransport(qwen_reply)
    argv = ("generate", "--split", "train", "--sample", "4", "--limit", "1", "--max-requests", "1")
    code, out, _ = cli(config_path, *argv, transport=transport)
    assert code == 0 and transport.calls == [] and "dry run" in out and "(qwen/qwen3.8-27b)" in out
    assert "max_completion_tokens 2048" in out and "reasoning_format parsed, reasoning_effort none" in out
    code, out, err = cli(config_path, *argv, "--confirm-api-use", transport=transport)
    assert code == 0, err
    assert len(transport.calls) == 1 and "requests sent: 1" in out and "'reasoning_tokens': 0" in out
    payload = transport.calls[0]["payload"]
    assert payload["max_completion_tokens"] == 2048 and payload["reasoning_effort"] == "none"
    assert payload["reasoning_format"] == "parsed" and "max_tokens" not in payload
    records = Path(load_data_prep_config(config_path)[0].output_dir) / "generation" / "train.records.jsonl"
    (row,) = [json.loads(line) for line in records.read_text().splitlines()]
    assert row["status"] == "done" and row["usage"]["total_tokens"] == 510
    assert SECRET not in records.read_text() + out + err


def test_a_versioned_plan_drives_a_one_request_pilot_and_records_can_be_revalidated(
    setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, config_path = setup
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert cli(config_path, "prepare")[0] == 0
    code, out, _ = cli(config_path, "sources", "--split", "train")
    assert code == 0 and "excluded by rule (first match)" in out and "kept by company" in out
    plan_argv = ("plan", "--split", "train", "--name", "pilot_v2", "--size", "3")
    code, out, err = cli(config_path, *plan_argv)
    assert code == 0 and "wrote plan pilot_v2" in out, err
    prepared_dir = Path(load_data_prep_config(config_path)[0].output_dir)
    plan_file = prepared_dir / "plans" / "train.pilot_v2.json"
    plan = json.loads(plan_file.read_text())
    assert cli(config_path, *plan_argv)[1].count("unchanged (identical plan present)") == 1
    code, _, err = cli(config_path, *plan_argv, "--seed", "99")
    assert code == 1 and "never changed" in err and json.loads(plan_file.read_text()) == plan

    transport = MockTransport(respond)
    argv = ("generate", "--split", "train", "--plan", "pilot_v2", "--limit", "1", "--max-requests", "1")
    code, out, _ = cli(config_path, *argv, transport=transport)
    assert code == 0 and "plan pilot_v2: 3 passages" in out and "dry run" in out and transport.calls == []
    code, out, err = cli(config_path, *argv, "--confirm-api-use", transport=transport)
    assert code == 0 and len(transport.calls) == 1 and "requests sent: 1" in out, err
    records = prepared_dir / "generation" / "train.records.jsonl"
    (row,) = [json.loads(line) for line in records.read_text().splitlines()]
    assert row["passage_id"] == plan["passages"][0]["passage_id"]  # plan order
    assert row["validator_version"] == "sec-qcheck-v3" and len(row["candidates"]) == row["proposed"]
    code, _, err = cli(config_path, "generate", "--split", "train", "--plan", "pilot_v2", "--sample", "2")
    assert code == 1 and "either --plan or --sample" in err
    code, _, err = cli(config_path, "generate", "--split", "train", "--plan", "pilot_v9")
    assert code == 1 and "no plan 'pilot_v9'" in err

    legacy = {"passage_id": plan["passages"][1]["passage_id"], "split": "train", "status": "done", "proposed": 1,
              "accepted": [], "rejected": [{"question": "Q?", "reason": "number_not_in_passage"}]}
    with records.open("a") as out_file:
        out_file.write(json.dumps(legacy, sort_keys=True) + "\n")
    before = records.read_bytes()
    code, out, _ = cli(config_path, "revalidate", "--split", "train")
    assert code == 0 and "'up_to_date': 1" in out and "'revalidated': 0" in out
    assert f"cannot revalidate {legacy['passage_id']}" in out and "answers and evidence were not kept" in out
    assert records.read_bytes() == before  # nothing guessed, nothing written
    assert SECRET not in records.read_text() + out


def test_a_full_run_samples_only_passages_the_source_filter_keeps(
    setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, config_path = setup
    assert cli(config_path, "prepare")[0] == 0
    strict = {"protected_items": [], "min_finance_term_share": 0.5, "min_figures_per_100_words": 100.0,
              "min_table_numbers_per_100_words": 100.0}
    write_prep_config(inputs.root, config_path, source_quality=strict)
    code, out, _ = cli(config_path, "generate", "--split", "train")
    assert code == 0 and "source quality:" in out and "'low_financial_content'" in out
    code, out, _ = cli(config_path, "sources", "--split", "train")
    kept = int(out.split("source quality keeps ")[1].split(",")[0])
    assert f"train: {kept} approved passages" not in out  # some were excluded
    assert f"{kept} to request" in cli(config_path, "generate", "--split", "train")[1]


def test_changed_request_settings_continue_the_run_and_pairs_name_their_parameters(
    setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, config_path = setup
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    write_prep_config(inputs.root, config_path, generation={"max_completion_tokens": 2048})
    assert cli(config_path, "prepare")[0] == 0
    first = load_data_prep_config(config_path)[0]
    generation = Path(first.output_dir) / "generation"
    argv = ("generate", "--split", "train", "--sample", "4")
    transport = MockTransport(respond)
    assert cli(config_path, *argv, "--limit", "2", "--confirm-api-use", transport=transport)[0] == 0
    before = tree(generation)
    records_before = (generation / "train.records.jsonl").read_bytes()

    write_prep_config(inputs.root, config_path, generation={"max_completion_tokens": 768, "output_tokens_per_minute": 1000})
    paced = load_data_prep_config(config_path)[0]
    code, out, _ = cli(config_path, *argv, transport=transport)
    assert code == 0 and "request settings differ from the recorded run: max_completion_tokens 2048 -> 768" in out
    assert "add --continue-run" in out and "pacing: 600 requests/min and 1000 output tokens/min" in out
    code, _, err = cli(config_path, *argv, "--confirm-api-use", transport=transport)
    assert code == 1 and "only request settings differ" in err and len(transport.calls) == 2
    code, out, _ = cli(config_path, *argv, "--continue-run", transport=transport)
    assert code == 0 and "records continuation 1" in out and tree(generation) == before  # a dry run writes nothing
    code, out, err = cli(config_path, *argv, "--continue-run", "--confirm-api-use", transport=transport)
    assert code == 0, err
    assert "recorded continuation 1: max_completion_tokens 2048 -> 768" in out and "'skipped_done': 2" in out
    assert len(transport.calls) == 4 and transport.calls[-1]["payload"]["max_completion_tokens"] == 768
    after = tree(generation)
    assert after["train.run.json"] == before["train.run.json"] and "train.run.continuations.jsonl" in after
    assert (generation / "train.records.jsonl").read_bytes().startswith(records_before)

    code, out, err = cli(config_path, "build-pairs")
    assert code == 0, err
    rows = [json.loads(line) for line in (generation / "train.records.jsonl").read_text().splitlines()]
    params_of = {r["passage_id"]: r["params_sha256"] for r in rows}
    assert sorted(set(params_of.values())) == sorted({first.generation.params_sha256(), paced.generation.params_sha256()})
    prepared_dir = Path(paced.output_dir)
    meta = json.loads((prepared_dir / "pairs" / "pairs.meta.json").read_text())
    run = meta["generation"]["runs"]["train"]
    assert run["params_sha256"] == [first.generation.params_sha256(), paced.generation.params_sha256()]
    assert run["continuations_sha256"] == after["train.run.continuations.jsonl"]
    train_pairs = [json.loads(line) for line in (prepared_dir / "pairs" / "train_pairs.jsonl").read_text().splitlines()]
    assert train_pairs and all(p["provenance"]["generation_params_sha256"] == params_of[p["positive_passage_id"]]
                               for p in train_pairs)
    assert cli(config_path, "check")[0] == 0

    # A rate limit asking for longer than the cap stops cleanly; an oversized request is refused at once.
    limited = MockTransport(lambda payload: TransportResponse(429, {"Retry-After": "900"}, "daily limit"))
    code, out, err = cli(config_path, "generate", "--split", "validation", "--confirm-api-use", transport=limited)
    assert code == 0, err
    assert "stopped by a rate limit (the API asks to wait 901s" in out and "HTTP 429 responses: 1" in out
    assert len(limited.calls) == 1 and (generation / "validation.records.jsonl").read_text() == ""
    oversized = MockTransport(lambda payload: TransportResponse(413, {}, "Request Entity Too Large"))
    code, _, err = cli(config_path, "generate", "--split", "validation", "--confirm-api-use", transport=oversized)
    assert code == 2 and "larger than the account's limits" in err and len(oversized.calls) == 1
    code, out, _ = cli(config_path, "generate", "--split", "validation", "--confirm-api-use", transport=MockTransport(respond))
    assert code == 0 and "'failed': 0" in out and "'skipped_failed': 0" in out  # nothing was marked failed
    assert SECRET not in "".join(p.read_text() for p in generation.iterdir()) + out + err


# --- evidence reuse, fail-closed leakage checks and carry-over (data_prep_v2) ---------------------

STRATEGY = ("Acme reported net sales of $1,200 for fiscal 2024 and plans to widen its distribution network across "
            "Europe and Latin America over the next three years while keeping capital spending within cash flow.")


def reuse_passages() -> list[Any]:
    from tests.unit.training_fixtures import make_passage
    return [
        *training_extra_passages(),
        # Repeats an evaluation passage's sentence (acme24_supplier) among new text: blocked by evidence reuse.
        make_passage("acme25_outlook", "ACME", "Acme Devices Inc.", "2025-12-31",
                     "We depend on a single supplier in Taiwan for advanced semiconductor components. Acme plans to "
                     "open a second assembly plant in Ohio during 2025 to expand capacity.",
                     section="Business", item="1", accession="0000000001-26-000001"),
        # Shares one evaluation figure (net sales of 1,200) but little else: usable, yet that figure is no evidence.
        make_passage("acme25_strategy", "ACME", "Acme Devices Inc.", "2025-12-31", STRATEGY,
                     section="Business", item="1", section_index=1, accession="0000000001-26-000001"),
    ]


def respond_with_strategy(payload: Mapping[str, Any]) -> Any:
    if "widen its distribution network" in payload["messages"][-1]["content"]:
        return {"questions": [
            {"question": "What net sales did Acme report for fiscal 2024?", "type": "revenue", "answer": "$1,200",
             "evidence": "Acme reported net sales of $1,200 for fiscal 2024"},
            {"question": "Where does Acme plan to widen its distribution network in coming years?", "type": "other",
             "answer": "Europe and Latin America", "evidence": "widen its distribution network across Europe and Latin America"},
        ]}
    return respond(payload)


def test_evidence_reuse_blocks_passages_and_rejects_leaked_evidence_per_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_expanded_inputs(tmp_path, reuse_passages())
    config_path = write_prep_config(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    code, out, err = cli(config_path, "prepare")
    assert code == 0, err
    assert "evidence reuse: 1 more passages reuse evaluation evidence {'spans': 1}" in out and "verified:" in out
    config = load_data_prep_config(config_path)[0]
    corpus = verify_frozen_corpus(config)
    passages = load_corpus_passages(corpus)
    prepared = open_prepared(config.output_dir, corpus, passages)
    names = {p.metadata["fixture_name"]: pid for pid, p in passages.items() if "fixture_name" in p.metadata}
    assert prepared.manifest["format_version"] == 2
    assert prepared.rows[names["acme25_outlook"]]["split"] == "blocked"
    assert prepared.rows[names["acme25_strategy"]]["split"] != "blocked"
    blocked = {json.loads(line)["passage_id"]: json.loads(line)["reasons"]
               for line in (Path(config.output_dir) / "blocklist.jsonl").read_text().splitlines()}
    assert blocked[names["acme25_outlook"]] == ["evidence_reuse"]
    assert cli(config_path, "prepare")[1].startswith("[02_generate_queries] unchanged")  # deterministic

    transport = MockTransport(respond_with_strategy)
    for split in ("train", "validation"):
        assert cli(config_path, "generate", "--split", split, "--confirm-api-use", transport=transport)[0] == 0
    code, out, err = cli(config_path, "build-pairs")
    assert code == 0, err
    meta = json.loads((Path(config.output_dir) / "pairs" / "pairs.meta.json").read_text())
    assert meta["stats"]["rejected_by_reason"]["evaluation_evidence"] >= 1
    assert meta["leakage"]["evidence"]["index_digest"] == prepared.manifest["reuse"]["index"]["digest"]
    pairs = read_pairs(prepared)
    rows = [p for s in pairs for p in pairs[s]]
    assert rows and all("equivalents" in p for p in rows)
    assert not any("$1,200" in p["evidence"] for p in rows)  # the evaluation figure never becomes evidence
    assert any(p["positive_passage_id"] == names["acme25_strategy"] for p in rows)  # its other question survives
    assert cli(config_path, "check")[0] == 0

    # Fail closed: a recorded index that the corpus no longer reproduces stops every leakage check.
    manifest_path = Path(config.output_dir) / "manifest.json"
    original = manifest_path.read_text()
    manifest_path.write_text(original.replace(prepared.manifest["reuse"]["index"]["digest"], "0" * 64))
    code, _, err = cli(config_path, "build-pairs")
    assert code == 1 and "cannot be completed" in err
    manifest_path.write_text(original)


def test_a_preparation_without_evidence_reuse_checks_cannot_build_pairs(setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, config_path = setup
    write_prep_config(inputs.root, config_path, reuse=None)
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert cli(config_path, "prepare")[0] == 0
    config = load_data_prep_config(config_path)[0]
    manifest = json.loads((Path(config.output_dir) / "manifest.json").read_text())
    assert manifest["format_version"] == 1 and "reuse" not in manifest
    assert cli(config_path, "generate", "--split", "train", "--confirm-api-use", transport=MockTransport(respond))[0] == 0
    code, _, err = cli(config_path, "build-pairs")
    assert code == 1 and "without evidence-reuse checks" in err
    assert not (Path(config.output_dir) / "pairs").exists()  # nothing written


def test_carry_over_rechecks_earlier_pairs_and_never_rebinds_them(setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib as _hashlib
    inputs, config_path = setup
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    for argv in (("prepare",), ("generate", "--split", "train", "--confirm-api-use"), ("build-pairs",)):
        assert cli(config_path, *argv, transport=MockTransport(respond))[0] == 0
    source = load_data_prep_config(config_path)[0]
    source_dir = Path(source.output_dir)
    before = tree(source_dir)
    train = [json.loads(line) for line in (source_dir / "pairs" / "train_pairs.jsonl").read_text().splitlines()]
    assert len(train) >= 2
    decisions = inputs.root / "decisions.jsonl"

    def decide(rows: list[dict[str, Any]]) -> None:
        decisions.write_text("".join(json.dumps(r) + "\n" for r in rows))

    base = [{"pair_id": p["pair_id"], "passage_id": p["positive_passage_id"], "reviewer": "ai_review", "gold": False,
             "question_sha256": _hashlib.sha256(p["query"].encode()).hexdigest(), "decision": "accept", "issues": []}
            for p in train]
    base[0] = {**base[0], "decision": "reject", "issues": ["weak_financial_relevance"], "reason": "not financial"}
    decide(base)
    target_path = inputs.root / "target.yaml"
    write_prep_config(inputs.root, target_path, output_dir=str(inputs.root / "data" / "processed" / "train" / "target"),
                      split={"seed": 8, "validation_share": 0.3})
    assert cli(target_path, "prepare")[0] == 0
    argv = ("carry-over", "--from-config", str(config_path), "--decisions", str(decisions), "--name", "pilot")
    code, out, err = cli(target_path, *argv)
    assert code == 0, err
    assert f"of {len(train)} pairs eligible" in out and "'quality_reject': 1" in out
    target_dir = inputs.root / "data" / "processed" / "train" / "target"
    rows = [json.loads(line) for line in (target_dir / "carryover" / "pilot.jsonl").read_text().splitlines()]
    meta = json.loads((target_dir / "carryover" / "pilot.meta.json").read_text())
    assert [r["source_pair_id"] for r in rows] == [p["pair_id"] for p in train]
    assert rows[0]["eligible"] is False and rows[0]["reasons"][0] == "quality_reject"
    assert all(r["eligible"] == (not r["reasons"]) for r in rows) and meta["stats"]["eligible"] == sum(r["eligible"] for r in rows)
    target_rows = {json.loads(line)["passage_id"]: json.loads(line) for line in (target_dir / "passages.jsonl").read_text().splitlines()}
    for r in rows:  # a moved positive is reported, never silently kept
        moved = target_rows[r["positive_passage_id"]]["split"] != "train"
        assert moved == any(x.startswith("positive_now_") for x in r["reasons"])
    assert "not gold labels" in meta["notice"] and meta["decisions"]["sha256"] == _hashlib.sha256(decisions.read_bytes()).hexdigest()
    assert not (target_dir / "pairs").exists()  # candidates only; the target's pairs are not touched
    assert cli(target_path, *argv)[1].count("unchanged (identical copy present)") == 1
    assert tree(source_dir) == before  # the earlier preparation, records and pairs are untouched

    decide([{**base[1], "question_sha256": "0" * 64}, *base[2:]])  # a decision bound to another question
    code, _, err = cli(target_path, *argv[:-1], "pilot_b")
    assert code == 1 and "never move between questions" in err
    decide([{**base[1], "gold": True}])
    code, _, err = cli(target_path, *argv[:-1], "pilot_c")
    assert code == 1 and "never gold labels" in err
    code, _, err = cli(config_path, "carry-over", "--from-config", str(config_path), "--decisions", str(decisions), "--name", "x")
    assert code == 1


def test_a_balanced_prompt_v2_pilot_never_exceeds_its_request_cap(setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, config_path = setup
    write_prep_config(inputs.root, config_path, generation={"prompt_version": "sec-qgen-v2"})
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert cli(config_path, "prepare")[0] == 0
    code, _, err = cli(config_path, "plan", "--split", "train", "--name", "pilot_a", "--size", "4", "--method",
                       "balanced", "--max-company-share", "0.5")
    assert code == 1 and "at most 50% per company" in err  # one company has train sources: a cap is never stretched
    plan_argv = ("plan", "--split", "train", "--name", "pilot_b", "--size", "3", "--method", "balanced",
                 "--max-company-share", "1.0")
    code, out, err = cli(config_path, *plan_argv)
    assert code == 0 and "one passage per overlap group and split unit" in out, err
    plan = json.loads((Path(load_data_prep_config(config_path)[0].output_dir) / "plans" / "train.pilot_b.json").read_text())
    assert len({p["split_unit"] for p in plan["passages"]}) == len({p["group_id"] for p in plan["passages"]}) == 3
    capped_argv = ("plan", "--split", "train", "--name", "pilot_c", "--size", "1", "--max-per-company-year", "1")
    code, _, err = cli(config_path, *capped_argv)
    assert code == 1 and "--max-per-company-year" in err and "apply to the balanced and balanced_years methods only" in err
    capped_argv = (*capped_argv, "--method", "balanced", "--max-company-share", "1.0", "--max-per-unit", "1")
    code, _, err = cli(config_path, *capped_argv)
    assert code == 1 and "one_per='group'" in err  # one passage per unit is already the default
    code, out, err = cli(config_path, *capped_argv, "--one-per", "group")
    assert code == 0 and "at most 1 per company and fiscal year, at most 1 per split unit" in out, err
    assert "concentration: 1 split units (at most 1 passages from one)" in out
    argv = ("generate", "--split", "train", "--plan", "pilot_b", "--max-requests", "2")
    transport = MockTransport(respond)
    code, out, _ = cli(config_path, *argv, transport=transport)
    assert code == 0 and "prompt sec-qgen-v2" in out and transport.calls == []  # the dry run sends nothing
    code, out, err = cli(config_path, *argv, "--confirm-api-use", transport=transport)
    assert code == 0, err
    assert len(transport.calls) == 2 and "requests sent: 2" in out and "stopped at the request budget" in out
    assert all("for fiscal" in c["payload"]["messages"][-1]["content"] for c in transport.calls)
    run = json.loads((Path(load_data_prep_config(config_path)[0].output_dir) / "generation" / "train.run.json").read_text())
    assert (run["prompt_version"], run["prompt_sha256"]) == ("sec-qgen-v2", prompt_sha256("sec-qgen-v2"))


# --- a validation pilot: planned and generated apart from the training data -------------------------


def test_a_validation_pilot_is_generated_apart_from_the_training_data_and_reads_no_evaluation_file(
    setup: tuple[ExpandedInputs, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, config_path = setup
    root = inputs.root
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    for argv in (("prepare",), ("generate", "--split", "train", "--confirm-api-use"), ("build-pairs",)):
        assert cli(config_path, *argv, transport=MockTransport())[0] == 0
    config = load_data_prep_config(config_path)[0]
    prepared_dir = Path(config.output_dir)
    (root / "cand_v1.yaml").write_text(yaml.safe_dump({
        "version": "cand_v1", "data_prep_config": str(config_path),
        "sources": [{"name": "fixture", "decisions": str(accept_all(prepared_dir, root / "decisions.jsonl"))}]}))
    assert cli(config_path, "build-candidates", "--candidates", str(root / "cand_v1.yaml"))[0] == 0
    corpus = verify_frozen_corpus(config)
    passages = load_corpus_passages(corpus)
    prepared = open_prepared(config.output_dir, corpus, passages)
    validation = {p.passage_id for p in prepared.source_passages("validation", passages, min_words=5)}
    held = EXTRA["gamma24_debt"]  # a validation passage that, say, may answer every training candidate's question
    assert held in validation and len(validation) == 3
    locate = EvidenceLocator.locate
    monkeypatch.setattr(EvidenceLocator, "locate", lambda self, *args: [*locate(self, *args), Equivalent(held, "span", "span")])
    questions = [q.text for q in load_evaluation_questions(config.leakage.evaluation_question_files)]
    training = {name: tree(prepared_dir / name) for name in ("pairs", "candidates")}
    train_generation = {n: d for n, d in tree(prepared_dir / "generation").items() if n.startswith("train.")}
    # Evaluation questions, labels, pools, audits and runs are out of reach while the pilot is planned and generated.
    hidden = root / "hidden"
    hidden.mkdir()
    kept = {"corpus", "data", "data_prep.yaml", "cand_v1.yaml", "decisions.jsonl", "hidden"}
    evaluation = [p for p in root.iterdir() if p.name not in kept]
    assert {"annotation", "dev_ai_v3", "gold_queries.jsonl"} <= {p.name for p in evaluation}
    for path in evaluation:
        path.rename(hidden / path.name)

    plan_argv = ("plan", "--split", "validation", "--name", "val_pilot_v1", "--size", "2", "--max-company-share", "1.0",
                 "--exclude-evidence-of", "cand_v1")
    code, _, err = cli(config_path, *plan_argv)  # the default method would ignore the exclusion: refused
    assert code == 1 and "exclusions apply to the balanced and balanced_years methods only" in err
    code, out, err = cli(config_path, *plan_argv, "--method", "balanced_years")
    assert code == 0, err
    assert "fiscal years covered of those available: GAMMA 2/2" in out
    plan = json.loads((prepared_dir / "plans" / "validation.val_pilot_v1.json").read_text())
    assert {p["passage_id"] for p in plan["passages"]} == validation - {held}  # it may answer a training question
    assert plan["constraints"]["excluded_from"] == ["candidates:cand_v1"] and plan["constraints"]["excluded_groups"] == 1
    assert plan["coverage"] == {"GAMMA": {"available": ["2022", "2023"], "covered": ["2022", "2023"]}}

    argv = ("generate", "--split", "validation", "--plan", "val_pilot_v1", "--max-requests", "2")
    sent: list[Mapping[str, Any]] = []

    def flaky(payload: Mapping[str, Any]) -> Any:  # the first request fails once; its retry uses budget too
        sent.append(payload)
        return TransportResponse(503, {}, "unavailable") if len(sent) == 1 else {"questions": generic_candidates(payload)}

    code, out, _ = cli(config_path, *argv, transport=MockTransport(flaky))
    assert code == 0 and "dry run: no request sent" in out and "requests: 2 if every reply is usable" in out and not sent
    code, out, err = cli(config_path, *argv, "--confirm-api-use", transport=MockTransport(flaky))
    assert code == 0, err
    assert len(sent) == 2 and "requests sent: 2" in out and "stopped at the request budget" in out
    prompts = [payload["messages"][-1]["content"] for payload in sent]
    assert all("Write 1 different questions" in p for p in prompts)  # one question per validation passage
    assert not any(q in p for q in questions for p in prompts)  # no evaluation question reaches a prompt
    generation = tree(prepared_dir / "generation")
    assert {n for n in generation if not n.startswith("train.")} == {"validation.records.jsonl", "validation.run.json"}
    assert {n: d for n, d in generation.items() if n.startswith("train.")} == train_generation
    records = [json.loads(line) for line in (prepared_dir / "generation" / "validation.records.jsonl").read_text().splitlines()]
    assert [(r["split"], r["status"]) for r in records] == [("validation", "done")]
    assert json.loads((prepared_dir / "generation" / "validation.run.json").read_text())["queries_per_passage"] == 1
    for path in evaluation:
        (hidden / path.name).rename(path)

    # The training data is untouched: its candidates still verify, and a pair build that would rewrite it is refused.
    assert {name: tree(prepared_dir / name) for name in ("pairs", "candidates")} == training
    assert len(load_candidates(prepared, passages, config, "cand_v1", "train")) == len(
        read_pairs(prepared)["train"]) and verify_candidate_provenance(prepared, "cand_v1")["candidates"]
    code, _, err = cli(config_path, "build-pairs")
    assert code == 1 and "already holds different pairs" in err and "nothing was written" in err
    assert {name: tree(prepared_dir / name) for name in ("pairs", "candidates")} == training


def test_a_training_plan_skips_every_group_an_existing_candidate_uses(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """--exclude-candidate-groups keeps new questions away from the passages the existing candidates use (positives
    and exact and uncertain equivalents), so those candidates stay as they are and none is asked about again."""
    from finretriever.training.candidates import load_candidates as load
    from tests.unit.validation_fixtures import build_world
    from tests.unit.validation_fixtures import cli as world_cli

    world = build_world(tmp_path, monkeypatch)
    candidates = load(world.prepared, world.passages, world.config, "cand_v1", "train")
    used = {world.prepared.rows[pid]["group_id"] for c in candidates
            for pid in (c.positive_passage_id, *c.equivalents.exact, *c.equivalents.uncertain)}
    before = tree(world.directory / "candidates")
    argv = ("plan", "--split", "train", "--size", "1", "--method", "balanced_years", "--max-company-share", "1.0",
            "--one-per", "group", "--exclude-candidate-groups", "cand_v1")
    code, _, err = world_cli(world.config_path, "plan", "--split", "train", "--name", "t1", *argv[3:])
    assert code == 0, err
    plan = json.loads((world.directory / "plans" / "train.t1.json").read_text())
    assert not {world.prepared.rows[p["passage_id"]]["group_id"] for p in plan["passages"]} & used
    assert plan["constraints"]["excluded_from"] == [f"candidate_groups:cand_v1:{len(candidates)} pairs"]
    assert plan["constraints"]["excluded_groups"] == len(used)
    code, _, err = world_cli(world.config_path, "plan", "--split", "train", "--name", "t2", "--size", "2",
                             *argv[5:])  # the same options, one passage more
    assert code == 1 and "only 1 passages fit 2" in err  # one group is left; nothing is shortened
    code, _, err = world_cli(world.config_path, "plan", "--split", "train", "--name", "t3", "--size", "1",
                             "--exclude-candidate-groups", "cand_v1")
    assert code == 1 and "apply to the balanced and balanced_years methods only" in err
    assert tree(world.directory / "candidates") == before  # the candidates are only read
