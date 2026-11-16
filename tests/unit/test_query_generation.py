"""Tests for synthetic query generation: candidate checks, the mocked Groq client, resumable runs (offline)."""

from __future__ import annotations

import email.utils
import hashlib
import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from finretriever.training.data_prep import load_data_prep_config
from finretriever.training.query_generation import (
    PACING_MARGIN_SECONDS,
    PROMPT_VERSION,
    RECORD_FORMAT_VERSION,
    VALIDATOR_VERSION,
    AuthError,
    CompanyAliases,
    CredentialError,
    GenerationError,
    GenerationSettings,
    GroqClient,
    MalformedResponse,
    ModelUnavailable,
    QueryGenerator,
    RateLimited,
    RequestBudgetExhausted,
    RequestRejected,
    RequestTooLarge,
    ResponseTruncated,
    SourcePassage,
    TransientError,
    TransportResponse,
    api_key_from_env,
    build_messages,
    check_candidate,
    check_reply,
    check_run,
    continuations_path,
    dates_in,
    describe_changes,
    generation_run_info,
    latest_records,
    numbers,
    parse_candidates,
    parse_duration,
    proposed_fields,
    read_records,
    record_sha256,
    records_path,
    retry_after_seconds,
    revalidate_records,
    run_path,
    sample_sources,
)
from tests.unit.training_fixtures import SECRET, MockTransport, ok

TEXT = (
    "Gamma Energy reported revenue of $3,400 million for 2024, an increase of 12% from $3,036 million in 2023, "
    "driven by higher natural gas prices and new pipeline capacity in the Permian basin."
)
PASSAGE = SourcePassage(
    passage_id="p1", split="train", group_id="grp_00001", ticker="GAMMA", fiscal_year="2024",
    company="Gamma Energy Corp.", form_type="10-K", filing_date="2025-02-20", report_date="2024-12-31",
    section="Item 7 MD&A", visible_text=TEXT,
)
ALIASES = CompanyAliases.from_companies({"GAMMA": "Gamma Energy Corp.", "ACME": "Acme Devices Inc."})
SETTINGS = GenerationSettings(model="test-model", requests_per_minute=60, max_retries=2, backoff_seconds=1.0)
GOOD = {
    "question": "How much revenue did Gamma Energy report for fiscal 2024?", "type": "revenue",
    "answer": "$3,400 million", "evidence": "Gamma Energy reported revenue of $3,400 million for 2024",
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)


def test_a_grounded_question_is_accepted() -> None:
    candidate, reason = check_candidate(GOOD, PASSAGE, SETTINGS, ALIASES)
    assert reason is None and candidate is not None and candidate.to_dict() == GOOD


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"evidence": None}, "malformed"),
        ({"type": "gossip"}, "malformed"),
        ({"question": "How much revenue did Gamma Energy report for fiscal 2024"}, "question_format"),
        ({"question": "Gamma revenue 2024?"}, "question_format"),
        ({"question": "What does the passage say about Gamma Energy revenue in 2024?"}, "refers_to_context"),
        ({"question": "According to the table, what was Gamma Energy revenue in 2024?"}, "refers_to_context"),
        ({"question": "How much revenue was reported for fiscal 2024 by the company?"}, "missing_company"),
        ({"question": "How did Gamma Energy revenue in 2024 compare with Acme Devices?"}, "other_company"),
        ({"question": "Gamma Energy reported revenue of $3,400 million for 2024, right?"}, "copied_from_passage"),
        ({"evidence": "revenue"}, "evidence_length"),
        ({"evidence": "Gamma Energy reported revenue of $9,999 million for 2024"}, "evidence_not_in_visible_text"),
        ({"question": "Did Gamma Energy revenue reach $3.4 billion in fiscal 2024?", "answer": "yes"}, "number_not_in_passage"),
        ({"answer": "$3,036 million"}, "answer_not_supported"),
        ({"answer": "offshore wind turbines"}, "answer_not_supported"),
    ],
)
def test_unsupported_or_malformed_candidates_are_rejected(change: dict[str, Any], reason: str) -> None:
    candidate, found = check_candidate({**GOOD, **change}, PASSAGE, SETTINGS, ALIASES)
    assert (candidate, found) == (None, reason)


def test_evidence_and_numbers_must_lie_in_the_tokenizer_visible_text() -> None:
    visible = SourcePassage(**{**PASSAGE.__dict__, "visible_text": TEXT[: TEXT.index(" driven")]})  # tail truncated
    tail = {**GOOD, "question": "Why did Gamma Energy revenue rise in fiscal 2024?", "type": "mdna",
            "answer": "higher natural gas prices", "evidence": "driven by higher natural gas prices"}
    assert check_candidate(tail, PASSAGE, SETTINGS, ALIASES)[1] is None
    assert check_candidate(tail, visible, SETTINGS, ALIASES)[1] == "evidence_not_in_visible_text"
    assert "driven by" not in build_messages(visible, 2)[1]["content"]  # the model never sees truncated text


def test_numbers_are_normalised_and_form_names_ignored() -> None:
    assert numbers("$1,200.50 and 12% in Q4 of fiscal 2024 (Form 10-K, Item 7)") == {"1200.5", "12", "2024"}
    assert numbers("0.50 and 3,036") == {"0.5", "3036"}


def test_replies_are_parsed_strictly_and_checked_together() -> None:
    assert parse_candidates('```json\n{"questions": [{"q": 1}]}\n```') == [{"q": 1}]
    with pytest.raises(MalformedResponse):
        parse_candidates("Here are your questions!")
    with pytest.raises(MalformedResponse):
        parse_candidates('{"items": []}')
    second = {**GOOD, "question": "What revenue growth rate did Gamma Energy report for fiscal 2024?", "answer": "12%",
              "evidence": "an increase of 12% from $3,036 million in 2023"}
    near = {**GOOD, "question": "How much revenue did Gamma Energy report, for fiscal 2024?"}
    accepted, rejected = check_reply([GOOD, near, second, "junk"], PASSAGE, SETTINGS, ALIASES, n=1)
    assert [a.question for a in accepted] == [GOOD["question"]]
    assert [r["reason"] for r in rejected] == ["duplicate_in_passage", "excess_question", "malformed"]


def test_prompts_carry_only_the_visible_passage_and_its_filing() -> None:
    table = SourcePassage(**{**PASSAGE.__dict__, "visible_text": "[TABLE] Revenue | 2024: $3,400 [/TABLE]"})
    system, user = build_messages(table, 2)
    assert system["role"] == "system" and "JSON" in system["content"]
    assert "Gamma Energy Corp. (GAMMA)" in user["content"] and "2024-12-31" in user["content"]
    assert "[TABLE] Revenue | 2024: $3,400 [/TABLE]" in user["content"] and "ask for a figure" in user["content"]
    assert "Write 2 different questions" in user["content"] and PROMPT_VERSION == "sec-qgen-v1"


# --- the Groq client ------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now, self.sleeps = 0.0, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def client(transport: Any, clock: Clock, **settings: Any) -> GroqClient:
    return GroqClient(SETTINGS.model_copy(update=settings), api_key=SECRET, transport=transport, sleep=clock.sleep, clock=clock)


def test_client_spaces_requests_and_sends_the_configured_payload() -> None:
    transport, clock = MockTransport(lambda payload: {"questions": []}), Clock()
    c = client(transport, clock)
    assert c.complete([{"role": "user", "content": "hi"}]) == '{"questions": []}'
    c.complete([{"role": "user", "content": "again"}])
    assert clock.sleeps == [1.0]  # 60 requests per minute
    call = transport.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert call["payload"]["model"] == "test-model" and call["payload"]["response_format"] == {"type": "json_object"}
    assert call["payload"]["temperature"] == 0.7 and call["timeout"] == 60.0


def test_client_retries_rate_limits_and_server_errors_with_backoff() -> None:
    replies = iter([TransportResponse(429, {"Retry-After": "3"}, "slow down"), ok({"questions": []})])
    clock = Clock()
    assert client(lambda *a: next(replies), clock, requests_per_minute=6000).complete([]) == '{"questions": []}'
    assert clock.sleeps == [3.0 + PACING_MARGIN_SECONDS]  # Retry-After honoured, with a margin
    failures, clock = MockTransport(lambda payload: TransportResponse(503, {}, "unavailable")), Clock()
    with pytest.raises(TransientError, match="after 3 attempts: HTTP 503"):
        client(failures, clock, requests_per_minute=6000).complete([])
    assert len(failures.calls) == 3 and [s for s in clock.sleeps if s >= 1] == [1.0, 2.0]
    leaky = MockTransport(lambda payload: ConnectionError(f"reset while sending Authorization: Bearer {SECRET}"))
    with pytest.raises(TransientError) as caught:
        client(leaky, Clock()).complete([])
    assert SECRET not in str(caught.value) and "[redacted]" in str(caught.value)


def test_client_stops_at_once_on_rejected_credentials_and_bad_requests() -> None:
    denied = MockTransport(lambda payload: TransportResponse(401, {}, f"invalid key {SECRET}"))
    with pytest.raises(AuthError) as caught:
        client(denied, Clock()).complete([])
    assert len(denied.calls) == 1 and SECRET not in str(caught.value) and "GROQ_API_KEY" in str(caught.value)
    bad = MockTransport(lambda payload: TransportResponse(400, {}, "model not found"))
    with pytest.raises(RequestRejected, match="HTTP 400"):
        client(bad, Clock()).complete([])
    assert len(bad.calls) == 1
    with pytest.raises(MalformedResponse):
        client(MockTransport(lambda payload: TransportResponse(200, {}, "{}")), Clock()).complete([])


def test_the_key_comes_from_the_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(CredentialError, match="set GROQ_API_KEY"):
        api_key_from_env(SETTINGS)
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert api_key_from_env(SETTINGS) == SECRET
    assert SECRET not in json.dumps(SETTINGS.params())


# --- resumable generation --------------------------------------------------------------------------


class Scripted:
    """A chat client replying from a script (strings or exceptions), then with ``default``."""

    def __init__(self, *replies: Any, default: Any = None) -> None:
        self.replies, self.default, self.calls = list(replies), default, 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else self.default
        if isinstance(reply, Exception):
            raise reply
        return reply


def passages(n: int) -> list[SourcePassage]:
    return [SourcePassage(**{**PASSAGE.__dict__, "passage_id": f"p{i}"}) for i in range(1, n + 1)]


def generator(chat: Any, directory: Path, settings: GenerationSettings = SETTINGS, **kw: Any) -> QueryGenerator:
    return QueryGenerator(chat, settings, ALIASES, directory=directory, prepared_manifest_sha256="a" * 64,
                          clock=lambda: "2026-01-01T00:00:00+00:00", secret=SECRET, **kw)


GOOD_REPLY = json.dumps({"questions": [GOOD]})


def test_generation_resumes_without_duplicating_completed_passages(tmp_path: Path) -> None:
    chat = Scripted(default=GOOD_REPLY)
    first = generator(chat, tmp_path).run("train", passages(3), limit=2)
    assert first == {"skipped_done": 0, "skipped_failed": 0, "done": 2, "failed": 0, "stopped_by_budget": 0,
                     "stopped_by_rate_limit": 0} and chat.calls == 2
    second = generator(chat, tmp_path).run("train", passages(3))
    assert second["skipped_done"] == 2 and second["done"] == 1 and chat.calls == 3
    third = generator(chat, tmp_path).run("train", passages(3))
    assert third == {"skipped_done": 3, "skipped_failed": 0, "done": 0, "failed": 0, "stopped_by_budget": 0,
                     "stopped_by_rate_limit": 0} and chat.calls == 3
    records = read_records(records_path(tmp_path, "train"))
    assert [r["passage_id"] for r in records] == ["p1", "p2", "p3"] and all(r["status"] == "done" for r in records)
    assert records[0]["accepted"] == [GOOD] and records[0]["proposed"] == 1
    changed = SETTINGS.model_copy(update={"temperature": 0.2})
    with pytest.raises(GenerationError, match="different run"):
        generator(chat, tmp_path, changed).run("train", passages(3))
    with pytest.raises(GenerationError, match="must belong to the validation split"):
        generator(chat, tmp_path).run("validation", passages(1))


def test_failures_are_recorded_retried_on_request_and_stop_repeated_outages(tmp_path: Path) -> None:
    chat = Scripted("not json", GOOD_REPLY, RequestRejected("HTTP 400: bad"), TransientError(f"HTTP 503 {SECRET}"),
                    default=GOOD_REPLY)
    counts = generator(chat, tmp_path).run("train", passages(4))
    assert counts == {"skipped_done": 0, "skipped_failed": 0, "done": 2, "failed": 2, "stopped_by_budget": 0,
                      "stopped_by_rate_limit": 0}
    records = {r["passage_id"]: r for r in read_records(records_path(tmp_path, "train"))}
    assert records["p1"]["attempts"] == 2 and records["p1"]["status"] == "done"  # a malformed reply was retried
    assert records["p2"]["status"] == "failed" and records["p2"]["error"] == "HTTP 400: bad"
    assert records["p3"]["error"] == "HTTP 503 [redacted]" and records["p3"]["request_failed"]
    assert generator(Scripted(), tmp_path).run("train", passages(4))["skipped_failed"] == 2  # not retried by default
    retried = generator(Scripted(default=GOOD_REPLY), tmp_path).run("train", passages(4), retry_failed=True)
    assert retried["done"] == 2 and retried["skipped_done"] == 2
    assert SECRET not in records_path(tmp_path, "train").read_text()

    outage = Scripted(default=TransientError("HTTP 503"))
    with pytest.raises(GenerationError, match="3 consecutive request failures"):
        generator(outage, tmp_path / "outage").run("train", passages(5))
    assert len(read_records(records_path(tmp_path / "outage", "train"))) == 3
    bad = Scripted("nope", "still not json")
    generator(bad, tmp_path / "bad").run("train", passages(1))
    (record,) = read_records(records_path(tmp_path / "bad", "train"))
    assert (record["status"], record["attempts"], record["error"]) == ("failed", 2, "the reply is not JSON")


def test_rejected_credentials_stop_generation_before_anything_is_recorded(tmp_path: Path) -> None:
    with pytest.raises(AuthError):
        generator(Scripted(AuthError("HTTP 401")), tmp_path).run("train", passages(2))
    assert read_records(records_path(tmp_path, "train")) == []
    (tmp_path / "train.records.jsonl").write_text('{"passage_id": "p1", "status": "do')
    with pytest.raises(GenerationError, match="not a complete record"):
        generator(Scripted(default=GOOD_REPLY), tmp_path).run("train", passages(2))


# --- request budget, token usage and pilot sampling -------------------------------------------------


def with_usage(content: Any, prompt: int = 100, completion: int = 40) -> TransportResponse:
    body = {"choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}
    headers = {"x-ratelimit-remaining-requests": "999", "x-ratelimit-remaining-tokens": "5000", "Authorization": SECRET}
    return TransportResponse(200, headers, json.dumps(body))


def test_the_request_budget_counts_every_attempt_and_stops_the_run_cleanly(tmp_path: Path) -> None:
    failing = MockTransport(lambda payload: TransportResponse(503, {}, "busy"))
    c = GroqClient(SETTINGS.model_copy(update={"max_retries": 4}), api_key=SECRET, transport=failing,
                   sleep=Clock().sleep, clock=Clock(), max_requests=3)
    with pytest.raises(RequestBudgetExhausted, match="budget of 3"):
        c.complete([])
    assert len(failing.calls) == c.requests_sent == 3  # retries count against the budget

    transport = MockTransport(lambda payload: with_usage({"questions": [GOOD]}))
    budget = GroqClient(SETTINGS, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock(), max_requests=2)
    counts = generator(budget, tmp_path).run("train", passages(3))
    assert counts["done"] == 2 and counts["stopped_by_budget"] == 1 and len(transport.calls) == 2
    assert [r["passage_id"] for r in read_records(records_path(tmp_path, "train"))] == ["p1", "p2"]
    resumed = GroqClient(SETTINGS, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock(), max_requests=5)
    again = generator(resumed, tmp_path).run("train", passages(3))
    assert again["done"] == 1 and again["skipped_done"] == 2 and resumed.requests_sent == 1
    with pytest.raises(GenerationError, match="at least 1"):
        GroqClient(SETTINGS, api_key=SECRET, max_requests=0)


def test_token_usage_and_rate_limit_headers_are_recorded_without_secrets(tmp_path: Path) -> None:
    transport = MockTransport(lambda payload: with_usage({"questions": [GOOD]}))
    c = GroqClient(SETTINGS, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock())
    generator(c, tmp_path).run("train", passages(2))
    assert dict(c.usage) == {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}
    assert c.rate_limit == {"x-ratelimit-remaining-requests": "999", "x-ratelimit-remaining-tokens": "5000"}
    records = read_records(records_path(tmp_path, "train"))
    assert records[0]["usage"] == {"completion_tokens": 40, "prompt_tokens": 100, "total_tokens": 140}
    assert SECRET not in records_path(tmp_path, "train").read_text()


def test_pilot_samples_spread_over_companies_deterministically() -> None:
    pool = [SourcePassage(**{**PASSAGE.__dict__, "passage_id": f"{t}{i}", "ticker": t}) for t in ("AAA", "BBB", "CCC")
            for i in range(5)]
    sample = sample_sources(pool, 7, seed=1)
    assert [p.ticker for p in sample] == ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC", "AAA"]
    assert sample == sample_sources(list(reversed(pool)), 7, seed=1)  # independent of input order
    assert sample != sample_sources(pool, 7, seed=2)
    assert len(sample_sources(pool, 50, seed=1)) == 15


# --- the pilot incident: HTTP 404 model_not_found for every request ------------------------------

MODEL_NOT_FOUND = (
    '{"error":{"message":"The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.",'
    '"type":"invalid_request_error","code":"model_not_found"}}'
)


def test_an_unavailable_model_stops_the_client_at_the_first_response() -> None:
    transport = MockTransport(lambda payload: TransportResponse(404, {}, MODEL_NOT_FOUND))
    with pytest.raises(ModelUnavailable) as caught:
        client(transport, Clock()).complete([])
    assert len(transport.calls) == 1 and (caught.value.status, caught.value.code) == (404, "model_not_found")
    assert "'test-model'" in str(caught.value) and "generation.model" in str(caught.value)
    assert SECRET not in str(caught.value)
    retired = MockTransport(lambda payload: TransportResponse(400, {}, '{"error":{"code":"model_decommissioned"}}'))
    with pytest.raises(ModelUnavailable):
        client(retired, Clock()).complete([])
    other = MockTransport(lambda payload: TransportResponse(400, {}, '{"error":{"code":"invalid_value"}}'))
    with pytest.raises(RequestRejected) as rejected:
        client(other, Clock()).complete([])
    assert (rejected.value.status, rejected.value.code) == (400, "invalid_value")


def test_an_unavailable_model_costs_one_request_and_records_nothing(tmp_path: Path) -> None:
    transport = MockTransport(lambda payload: TransportResponse(404, {}, MODEL_NOT_FOUND))
    c = GroqClient(SETTINGS, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock(), max_requests=50)
    with pytest.raises(ModelUnavailable):
        generator(c, tmp_path).run("train", passages(25))
    assert c.requests_sent == 1 and read_records(records_path(tmp_path, "train")) == []


def test_repeated_rejected_requests_trip_the_breaker_and_keep_their_status(tmp_path: Path) -> None:
    transport = MockTransport(lambda payload: TransportResponse(400, {}, '{"error":{"code":"invalid_value"}}'))
    c = GroqClient(SETTINGS, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock())
    with pytest.raises(GenerationError, match="3 consecutive request failures"):
        generator(c, tmp_path).run("train", passages(25))
    records = read_records(records_path(tmp_path, "train"))
    assert len(transport.calls) == len(records) == 3
    assert {(r["http_status"], r["error_code"], r["request_failed"]) for r in records} == {(400, "invalid_value", True)}


# --- reasoning models: Qwen 3.8 on Groq -----------------------------------------------------------

QWEN = SETTINGS.model_copy(update={"model": "qwen/qwen3.8-27b", "max_completion_tokens": 2048,
                                   "reasoning_format": "parsed", "reasoning_effort": "none"})


def reply(message: dict[str, Any], finish_reason: str = "stop", reasoning_tokens: int | None = None) -> TransportResponse:
    usage: dict[str, Any] = {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    body = {"choices": [{"message": message, "finish_reason": finish_reason}], "usage": usage}
    return TransportResponse(200, {}, json.dumps(body))


def test_the_repo_config_targets_qwen_with_a_configurable_completion_limit() -> None:
    config, _ = load_data_prep_config(Path(__file__).parents[2] / "configs" / "train" / "data_prep_v1.yaml")
    g = config.generation
    assert (g.model, g.max_completion_tokens, g.json_mode) == ("qwen/qwen3.8-27b", 768, True)
    assert g.output_tokens_per_minute == 1000 and g.max_completion_tokens < g.output_tokens_per_minute
    assert (g.reasoning_format, g.reasoning_effort) == ("parsed", "none") and 0.5 <= g.temperature <= 0.7
    assert {"max_completion_tokens", "reasoning_format", "reasoning_effort"} <= set(g.params())
    with pytest.raises(ValidationError):
        GenerationSettings(model="m", max_tokens=900)  # the old key is refused, not silently ignored


def test_qwen_requests_carry_the_completion_limit_and_reasoning_settings() -> None:
    transport = MockTransport(lambda payload: {"questions": []})
    GroqClient(QWEN, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock()).complete([])
    payload = transport.calls[0]["payload"]
    assert payload["model"] == "qwen/qwen3.8-27b" and payload["max_completion_tokens"] == 2048
    assert payload["reasoning_format"] == "parsed" and payload["reasoning_effort"] == "none"
    assert payload["response_format"] == {"type": "json_object"} and "max_tokens" not in payload
    plain = MockTransport(lambda payload: {"questions": []})
    client(plain, Clock()).complete([])
    assert not {"reasoning_format", "reasoning_effort"} & set(plain.calls[0]["payload"])  # unset: not sent


def test_raw_reasoning_cannot_be_combined_with_json_mode() -> None:
    with pytest.raises(ValidationError, match="raw cannot be combined with JSON mode"):
        GenerationSettings(model="m", reasoning_format="raw")
    assert GenerationSettings(model="m", reasoning_format="raw", json_mode=False).reasoning_format == "raw"
    with pytest.raises(ValidationError):
        GenerationSettings(model="m", reasoning_effort="extreme")


def test_reasoning_is_never_taken_for_the_reply() -> None:
    answer = json.dumps({"questions": [GOOD]})
    decoy = '{"questions": [{"question": "a draft from the reasoning"}]}'
    parsed = MockTransport(lambda payload: reply({"content": answer, "reasoning": decoy}, reasoning_tokens=80))
    c = GroqClient(QWEN, api_key=SECRET, transport=parsed, sleep=Clock().sleep, clock=Clock())
    assert c.complete([]) == answer
    assert c.usage["reasoning_tokens"] == 80 and c.usage["completion_tokens"] == 120
    tagged = MockTransport(lambda payload: reply({"content": f"<think>\n{decoy}\n</think>\n\n{answer}"}))
    assert parse_candidates(client(tagged, Clock()).complete([])) == [GOOD]
    for message in ({"content": f"<think>{decoy}"}, {"content": None, "reasoning": decoy},
                    {"content": "<think>all reasoning</think>"}):
        with pytest.raises(MalformedResponse):
            client(MockTransport(lambda payload, m=message: reply(m)), Clock()).complete([])


def test_a_reply_cut_off_at_the_token_limit_is_recorded_once_without_a_retry(tmp_path: Path) -> None:
    cut = MockTransport(lambda payload: reply({"content": '{"questions": [{"question": "How much'}, "length"))
    with pytest.raises(ResponseTruncated, match="max_completion_tokens") as caught:
        client(cut, Clock()).complete([])
    assert (caught.value.status, caught.value.code) == (200, "length") and len(cut.calls) == 1
    c = GroqClient(QWEN, api_key=SECRET, transport=cut, sleep=Clock().sleep, clock=Clock(), max_requests=10)
    with pytest.raises(GenerationError, match="3 consecutive request failures"):
        generator(c, tmp_path, QWEN).run("train", passages(5))
    records = read_records(records_path(tmp_path, "train"))
    assert c.requests_sent == len(records) == 3 and all(r["attempts"] == 1 for r in records)
    assert {(r["status"], r["http_status"], r["error_code"]) for r in records} == {("failed", 200, "length")}
    assert records[0]["usage"]["completion_tokens"] == 120  # the tokens the cut-off reply still cost


def test_a_one_request_pilot_keeps_the_diagnosis_of_an_unusable_reply(tmp_path: Path) -> None:
    transport = MockTransport(lambda payload: reply({"content": "Sure! Here are two questions."}))
    c = GroqClient(QWEN, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock(), max_requests=1)
    counts = generator(c, tmp_path, QWEN).run("train", passages(3), limit=1)
    assert counts["failed"] == 1 and counts["stopped_by_budget"] == 1 and c.requests_sent == 1
    (record,) = read_records(records_path(tmp_path, "train"))
    assert (record["error"], record["attempts"], record["budget_exhausted"]) == ("the reply is not JSON", 1, True)
    assert record["usage"]["total_tokens"] == 620 and not record["request_failed"]
    good = MockTransport(lambda payload: reply({"content": json.dumps({"questions": [GOOD]})}))
    one = GroqClient(QWEN, api_key=SECRET, transport=good, sleep=Clock().sleep, clock=Clock(), max_requests=1)
    counts = generator(one, tmp_path / "ok", QWEN).run("train", passages(3), limit=1)
    assert counts["done"] == 1 and counts["stopped_by_budget"] == 0 and one.requests_sent == 1


# --- date-aware number check: the first Qwen pilot reply ---------------------------------------
# The two questions are verbatim from the first real Qwen reply (Apple FY2025 exhibit index), which
# validator v1 rejected as number_not_in_passage. Their answers and evidence were not recorded, so
# the ones below are written for this test from the passage, not taken from the model.

APPLE_TEXT = (
    "Exhibit Number: 4.24 | Exhibit Description: Officer's Certificate of the Registrant, dated as of February 8, 2021, "
    "including forms of global notes representing the 0.700% Notes due 2026, 1.200% Notes due 2028, 1.650% Notes due "
    "2031, 2.375% Notes due 2041, 2.650% Notes due 2051 and 2.800% Notes due 2061. | Incorporated by Reference Form: "
    "8-K | Incorporated by Reference Exhibit: 4.1 | Incorporated by Reference Filing Date/ Period End Date: 2/8/21 "
    "Exhibit Number: 4.26 | Exhibit Description: Indenture, dated as of October 28, 2021, between the Registrant and "
    "The Bank of New York Mellon Trust Company, N.A., as Trustee. | Incorporated by Reference"
)
APPLE = SourcePassage(
    passage_id="aapl", split="train", group_id="grp_00092", ticker="AAPL", fiscal_year="2025", company="Apple Inc.",
    form_type="10-K", filing_date="2025-10-31", report_date="2025-09-27",
    section="Item 15 Exhibit and Financial Statement Schedules", visible_text=APPLE_TEXT, item="15",
)
APPLE_ALIASES = CompanyAliases.from_companies({"AAPL": "Apple Inc.", "ACME": "Acme Devices Inc."})
QWEN_Q1 = ("What are the specific coupon rates and maturity years listed for the global notes described in Apple Inc.'s "
           "Exhibit 4.24 for the fiscal year ended 2025-09-27?")
QWEN_Q2 = ("Which trustee is named in the indenture dated October 28, 2021, referenced in Apple Inc.'s Form 10-K for the "
           "fiscal year ended 2025-09-27?")
NOTES = {"type": "debt_liquidity", "answer": "0.700% due 2026 through 2.800% due 2061",
         "evidence": "0.700% Notes due 2026, 1.200% Notes due 2028, 1.650% Notes due 2031, 2.375% Notes due 2041, "
                     "2.650% Notes due 2051 and 2.800% Notes due 2061"}
TRUSTEE = {"type": "debt_liquidity", "answer": "The Bank of New York Mellon Trust Company, N.A.",
           "evidence": "Indenture, dated as of October 28, 2021, between the Registrant and The Bank of New York Mellon "
                       "Trust Company, N.A., as Trustee."}


def apple(question: str, fields: dict[str, str] = NOTES) -> str | None:
    return check_candidate({"question": question, **fields}, APPLE, SETTINGS, APPLE_ALIASES)[1]


def test_the_rejected_qwen_questions_pass_once_the_fiscal_year_end_is_read_as_a_date() -> None:
    assert VALIDATOR_VERSION == "sec-qcheck-v3"
    # Why v1 rejected them: the ISO date split into 2025, 09 and 27, and 09 and 27 are not in the passage.
    assert {"09", "27"} <= numbers(QWEN_Q1) and not {"09", "27"} & numbers(APPLE_TEXT)
    assert apple(QWEN_Q1) is None
    assert apple(QWEN_Q2, TRUSTEE) is None
    accepted, rejected = check_reply([{"question": QWEN_Q1, **NOTES}, {"question": QWEN_Q2, **TRUSTEE}], APPLE,
                                     SETTINGS, APPLE_ALIASES, n=2)
    assert [c.question for c in accepted] == [QWEN_Q1, QWEN_Q2] and rejected == []


@pytest.mark.parametrize(
    "written",
    ["2025-09-27", "September 27, 2025", "Sept. 27, 2025", "Sep 27 2025", "September 27th, 2025", "27 September 2025",
     "9/27/2025", "09/27/25"],
)
def test_the_report_date_is_accepted_in_common_forms(written: str) -> None:
    assert dates_in(written) == [(0, len(written), "2025-09-27")]
    assert apple(f"Which coupon rates did Apple Inc. list under Exhibit 4.24 for the fiscal year ended {written}?") is None


def test_only_complete_verified_dates_are_exempt() -> None:
    q = "Which coupon rates did Apple Inc. list under Exhibit 4.24 {}?"
    assert apple(q.format("in the Form 10-K filed October 31, 2025")) is None  # the filing date is verified too
    assert apple(q.format("in the officer's certificate dated February 8, 2021")) is None  # a date in the passage
    for unsupported, reason in (
        ("for the fiscal year ended September 28, 2025", "date_not_in_passage"),  # 28 and 2025 alone would pass
        ("for the fiscal year ended 2025-09-28", "number_not_in_passage"),
        ("for the fiscal year ended 9/28/25", "number_not_in_passage"),
        ("for the period ended 2025-27-09", "number_not_in_passage"),  # not a date: its numbers are checked
    ):
        assert apple(q.format(unsupported)) == reason, unsupported
    # The day and month of the verified date are not allowed on their own, and invented figures still fail.
    assert apple("Did Apple Inc. list 27 global notes under Exhibit 4.24 for the fiscal year ended 2025-09-27?") == (
        "number_not_in_passage")
    assert apple("Did Apple Inc. issue $9.9 billion of notes under Exhibit 4.24 for the fiscal year ended "
                 "September 27, 2025?") == "number_not_in_passage"
    assert dates_in("2025-02-30 and 13/45/2025") == []  # impossible dates are not dates
    # Answers are not exempt: a metadata date the evidence does not contain is unsupported.
    assert apple(QWEN_Q2, {**TRUSTEE, "answer": "2025-09-27"}) == "number_not_in_passage"


# --- rejected candidates are kept, and records can be revalidated locally -----------------------

ISO_QUESTION = {**GOOD, "question": "How much revenue did Gamma Energy report for the fiscal year ended 2024-12-31?"}
MIXED_REPLY = [ISO_QUESTION, {**GOOD, "question": "Did Gamma Energy report revenue of $9,999 million for fiscal 2024?"},
               "not an object", {"question": GOOD["question"], "type": "revenue", "answer": 3400}]


def test_checking_stored_fields_matches_checking_the_reply() -> None:
    raw = check_reply(MIXED_REPLY, PASSAGE, SETTINGS, ALIASES, n=2)
    stored = check_reply([proposed_fields(c) for c in MIXED_REPLY], PASSAGE, SETTINGS, ALIASES, n=2)
    assert raw == stored
    assert [r["reason"] for r in raw[1]] == ["number_not_in_passage", "malformed", "malformed"]
    assert proposed_fields("not an object") == dict.fromkeys(("question", "type", "answer", "evidence"))


def test_records_keep_every_proposed_candidate_with_versions_and_no_reasoning(tmp_path: Path) -> None:
    content = json.dumps({"questions": MIXED_REPLY})
    transport = MockTransport(lambda payload: reply(
        {"content": f"<think>DECOY_THINK {SECRET}</think>{content}", "reasoning": "DECOY_REASONING"}))
    c = GroqClient(QWEN, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock())
    generator(c, tmp_path, QWEN).run("train", passages(1))
    (record,) = read_records(records_path(tmp_path, "train"))
    assert (record["model"], record["prompt_version"], record["validator_version"], record["record_format"]) == (
        "qwen/qwen3.8-27b", PROMPT_VERSION, VALIDATOR_VERSION, RECORD_FORMAT_VERSION)
    assert record["params_sha256"] == QWEN.params_sha256()
    assert record["candidates"] == [proposed_fields(item) for item in MIXED_REPLY]
    assert record["accepted"] == [ISO_QUESTION]
    bad = record["rejected"][0]
    assert bad == {"index": 1, **MIXED_REPLY[1], "reason": "number_not_in_passage"}  # question, answer, evidence kept
    assert [r["index"] for r in record["rejected"]] == [1, 2, 3]
    text = records_path(tmp_path, "train").read_text()
    assert "DECOY" not in text and SECRET not in text and "choices" not in text and "<think>" not in text


def write_history(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """A v1-validated record with stored candidates (p1) and a pre-format-2 record without them (p2)."""
    generator(Scripted(), directory).run("train", [])  # binds the directory to this run
    old = {
        "passage_id": "p1", "split": "train", "generated_at": "2026-01-01T00:00:00+00:00", "status": "done",
        "attempts": 1, "error": None, "proposed": 1, "record_format": 2, "model": "test-model",
        "prompt_version": PROMPT_VERSION, "validator_version": "sec-qcheck-v1", "response_sha256": "b" * 64,
        "usage": {"total_tokens": 900}, "candidates": [ISO_QUESTION], "accepted": [],
        "rejected": [{"index": 0, **ISO_QUESTION, "reason": "number_not_in_passage"}],
    }
    legacy = {"passage_id": "p2", "split": "train", "generated_at": "2026-01-01T00:00:00+00:00", "status": "done",
              "attempts": 1, "error": None, "proposed": 1, "usage": {"total_tokens": 800}, "accepted": [],
              "rejected": [{"question": ISO_QUESTION["question"], "reason": "number_not_in_passage"}]}
    with records_path(directory, "train").open("a") as out:
        for record in (old, legacy):
            out.write(json.dumps(record, sort_keys=True) + "\n")
    return old, legacy


def revalidate(directory: Path, **kw: Any) -> dict[str, Any]:
    return revalidate_records(directory, "train", {p.passage_id: p for p in passages(2)}, SETTINGS, ALIASES,
                              prepared_manifest_sha256="a" * 64, clock=lambda: "2026-02-02T00:00:00+00:00", **kw)


def test_revalidation_appends_an_auditable_revision_and_never_rewrites_history(tmp_path: Path) -> None:
    old, _ = write_history(tmp_path)
    path = records_path(tmp_path, "train")
    before = path.read_bytes()
    dry = revalidate(tmp_path, dry_run=True)
    assert dry == {"up_to_date": 0, "revalidated": 1, "changed": 1, "failed_records": 0, "not_revalidatable": ["p2"]}
    assert path.read_bytes() == before
    assert revalidate(tmp_path) == dry
    after = path.read_bytes()
    assert after.startswith(before) and after.count(b"\n") == before.count(b"\n") + 1  # appended, not rewritten
    revision = read_records(path)[-1]
    assert revision["accepted"] == [ISO_QUESTION] and revision["rejected"] == []
    assert revision["validator_version"] == VALIDATOR_VERSION and revision["usage"] == {}
    assert revision["candidates"] == old["candidates"] and revision["generated_at"] == old["generated_at"]
    assert revision["revision"] == {"kind": "revalidation", "of_record_sha256": record_sha256(old),
                                    "previous_validator_version": "sec-qcheck-v1", "changed": True,
                                    "revised_at": "2026-02-02T00:00:00+00:00"}
    assert record_sha256(old) == hashlib.sha256(before.splitlines()[0]).hexdigest()  # the line as written
    assert latest_records(read_records(path))["p1"] == revision
    again = revalidate(tmp_path)
    assert again["up_to_date"] == 1 and again["revalidated"] == 0 and path.read_bytes() == after  # idempotent
    resumed = generator(Scripted(default=GOOD_REPLY), tmp_path).run("train", passages(2))
    assert resumed["skipped_done"] == 2  # revalidation never sends a request or re-queues a passage


def test_revalidation_is_deterministic_and_bound_to_its_run(tmp_path: Path) -> None:
    for name in ("a", "b"):
        write_history(tmp_path / name)
        revalidate(tmp_path / name)
    assert (records_path(tmp_path / "a", "train").read_bytes() == records_path(tmp_path / "b", "train").read_bytes())
    run_path(tmp_path / "a", "train").write_text("{}")
    with pytest.raises(GenerationError, match="different run"):
        revalidate(tmp_path / "a")
    assert revalidate(tmp_path / "empty")["revalidated"] == 0  # nothing generated yet


# --- output-token budgeting and rate limits (Groq: 1,000 output tokens per minute for this account) ----------

PACED = QWEN.model_copy(update={"max_completion_tokens": 768, "output_tokens_per_minute": 1000})


def timed(clock: Clock, times: list[float], response: Any) -> Any:
    """A transport answering ``response(payload)`` and noting the clock at every request."""
    def send(url: str, headers: Any, payload: Any, timeout: float) -> TransportResponse:
        times.append(clock.now)
        return response(payload)
    return send


def used(completion: int | None) -> TransportResponse:
    """A good reply reporting ``completion`` output tokens (or no usage at all)."""
    body: dict[str, Any] = {"choices": [{"message": {"content": GOOD_REPLY}, "finish_reason": "stop"}]}
    if completion is not None:
        body["usage"] = {"prompt_tokens": 900, "completion_tokens": completion, "total_tokens": 900 + completion}
    return TransportResponse(200, {}, json.dumps(body))


def test_a_completion_limit_above_the_output_token_limit_is_never_sent() -> None:
    with pytest.raises(ValidationError, match="exceeds output_tokens_per_minute 1000"):
        GenerationSettings(model="m", max_completion_tokens=2048, output_tokens_per_minute=1000)
    assert GenerationSettings(model="m", max_completion_tokens=768, output_tokens_per_minute=1000)
    unchecked = QWEN.model_copy(update={"output_tokens_per_minute": 1000})  # copies skip validation
    transport = MockTransport()
    with pytest.raises(RequestTooLarge, match="lower generation.max_completion_tokens"):
        GroqClient(unchecked, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock())
    assert transport.calls == []


def test_requests_are_paced_to_the_output_tokens_of_the_last_minute() -> None:
    clock, times = Clock(), []
    c = GroqClient(PACED, api_key=SECRET, transport=timed(clock, times, lambda p: used(230)), sleep=clock.sleep, clock=clock)
    for _ in range(5):
        c.complete([])
    # 230 + 768 fits in 1,000; a third reply does not until the first has been out of the window for 61s.
    assert times == [0.0, 1.0, 61.0, 62.0, 122.0]
    for i, at in enumerate(times):  # every request found room for its whole reservation
        assert sum(230 for earlier in times[:i] if at - earlier < 60) + 768 <= 1000
    assert c.waited_seconds == pytest.approx(122.0)
    clock, times = Clock(), []
    unknown = GroqClient(PACED, api_key=SECRET, transport=timed(clock, times, lambda p: used(None)), sleep=clock.sleep,
                         clock=clock)
    for _ in range(3):
        unknown.complete([])
    assert times == [0.0, 61.0, 122.0]  # without reported usage a reply is charged its whole reservation


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m3s", 3723.0), ("120ms", 0.12), ("7", 7.0), (" 3.5 ", 3.5),
     ("soon", None), ("", None), ("5x", None), ("inf", None)],
)
def test_rate_limit_durations_are_read_as_groq_writes_them(text: str, seconds: float | None) -> None:
    assert parse_duration(text) == (pytest.approx(seconds) if seconds is not None else None)


def test_a_rate_limit_waits_as_long_as_the_api_asks() -> None:
    def wait_for(first: TransportResponse) -> list[float]:
        replies, clock = iter([first, ok({"questions": []})]), Clock()
        client(lambda *a: next(replies), clock, requests_per_minute=6000).complete([])
        return clock.sleeps
    assert wait_for(TransportResponse(429, {"retry-after": "90"}, "")) == [91.0]  # not cut to max_backoff_seconds (60)
    otpm = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` on output tokens per minute "
            "(OTPM): Limit 1000, Used 980, Requested 768. Please try again in 45.5s.")
    assert wait_for(TransportResponse(429, {}, otpm)) == [46.5]  # reached, not exceeded: temporary
    reset = {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "7.66s"}
    assert wait_for(TransportResponse(429, reset, "")) == [pytest.approx(8.66)]
    assert wait_for(TransportResponse(429, {}, "slow down")) == [1.0]  # no wait named: backoff
    date = email.utils.format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 25 < retry_after_seconds({"Retry-After": date}) <= 30 and retry_after_seconds({}) is None


def test_a_rate_limit_that_asks_too_much_or_outlasts_the_retries_stops_cleanly() -> None:
    long = MockTransport(lambda payload: TransportResponse(429, {"Retry-After": "600"}, "daily limit"))
    c = client(long, Clock())
    with pytest.raises(RateLimited, match="wait 601s, longer than generation.max_rate_limit_wait_seconds") as caught:
        c.complete([])
    assert len(long.calls) == c.requests_sent == 1 and not isinstance(caught.value, TransientError)
    assert "601s" in (c.stop_reason or "") and c.rate_limited == 1
    persistent = MockTransport(lambda payload: TransportResponse(429, {"Retry-After": "2"}, f"busy {SECRET}"))
    c = client(persistent, Clock())
    with pytest.raises(RateLimited, match="still rate limited after 3 attempts") as caught:
        c.complete([])
    assert len(persistent.calls) == 3 and caught.value.status == 429 and SECRET not in str(caught.value)
    # A daily limit seen on a good reply stops before the next request rather than waiting hours.
    daily = MockTransport(lambda payload: TransportResponse(
        200, {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "7h30m"},
        json.dumps({"choices": [{"message": {"content": "{}"}}]})))
    c = client(daily, Clock())
    c.complete([])
    with pytest.raises(RateLimited, match="wait 27001s"):
        c.complete([])
    assert len(daily.calls) == 1
    soon = MockTransport(lambda payload: TransportResponse(
        200, {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m30s"},
        json.dumps({"choices": [{"message": {"content": "{}"}}]})))
    clock = Clock()
    c = client(soon, clock)
    c.complete([])
    c.complete([])
    assert len(soon.calls) == 2 and clock.sleeps == [151.0]


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (413, "Request Entity Too Large"),
        (429, "Rate limit reached on output tokens per minute (OTPM): Limit 1000, Requested 2048. Please try again "
              "in 1m5s."),
        (400, '{"error":{"message":"Request too large for model `qwen/qwen3.8-27b` on tokens per minute (TPM): '
              'Limit 6000, Requested 8100","type":"tokens","code":"rate_limit_exceeded"}}'),
    ],
)
def test_oversized_requests_are_never_retried(status: int, body: str, tmp_path: Path) -> None:
    transport = MockTransport(lambda payload: TransportResponse(status, {"Retry-After": "1"}, f"{body} {SECRET}"))
    with pytest.raises(RequestTooLarge, match="max_completion_tokens") as caught:
        client(transport, Clock()).complete([])
    assert len(transport.calls) == 1 and caught.value.status == status and SECRET not in str(caught.value)
    c = GroqClient(SETTINGS, api_key=SECRET, transport=transport, sleep=Clock().sleep, clock=Clock(), max_requests=50)
    with pytest.raises(RequestTooLarge):
        generator(c, tmp_path).run("train", passages(3))
    assert c.requests_sent == 1 and read_records(records_path(tmp_path, "train")) == []  # nothing marked failed


def test_rate_limit_and_budget_stops_record_nothing_and_the_run_resumes(tmp_path: Path) -> None:
    replies = iter([with_usage({"questions": [GOOD]}), TransportResponse(429, {"Retry-After": "900"}, "")])
    c = GroqClient(SETTINGS, api_key=SECRET, transport=lambda *a: next(replies), sleep=Clock().sleep, clock=Clock())
    counts = generator(c, tmp_path).run("train", passages(3))
    assert (counts["done"], counts["failed"], counts["stopped_by_rate_limit"]) == (1, 0, 1)
    assert [r["passage_id"] for r in read_records(records_path(tmp_path, "train"))] == ["p1"]

    always = MockTransport(lambda payload: TransportResponse(429, {"Retry-After": "1"}, "busy"))
    spent = GroqClient(SETTINGS, api_key=SECRET, transport=always, sleep=Clock().sleep, clock=Clock(), max_requests=2)
    counts = generator(spent, tmp_path).run("train", passages(3))
    assert counts["stopped_by_budget"] == 1 and spent.requests_sent == 2 and spent.rate_limited == 2
    assert len(read_records(records_path(tmp_path, "train"))) == 1  # retried 429s are not failures

    resumed = GroqClient(SETTINGS, api_key=SECRET, transport=MockTransport(lambda p: with_usage({"questions": [GOOD]})),
                         sleep=Clock().sleep, clock=Clock())
    counts = generator(resumed, tmp_path).run("train", passages(3))
    assert (counts["skipped_done"], counts["done"], resumed.requests_sent) == (1, 2, 2)

    # After an unusable reply, the stop keeps that diagnosis like the budget does.
    generator(Scripted("not json", RateLimited("wait")), tmp_path / "kept").run("train", passages(2))
    (record,) = read_records(records_path(tmp_path / "kept", "train"))
    assert (record["status"], record["rate_limited"], record["budget_exhausted"]) == ("failed", True, False)


# --- continuing a run whose request settings changed -------------------------------------------

LEGACY_KEYS = ("output_tokens_per_minute", "max_rate_limit_wait_seconds")


def legacy_run(directory: Path) -> tuple[bytes, bytes]:
    """A run recorded before the pacing settings existed, as the real pilot's was: 2048 tokens, two done records."""
    info = generation_run_info(QWEN, "train", "a" * 64)
    params = {k: v for k, v in info["params"].items() if k not in LEGACY_KEYS}
    info = {**info, "params": params,
            "params_sha256": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}
    directory.mkdir(parents=True, exist_ok=True)
    run_path(directory, "train").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    with records_path(directory, "train").open("w") as out:
        for pid in ("p1", "p2"):
            out.write(json.dumps({
                "passage_id": pid, "split": "train", "generated_at": "2026-01-01T00:00:00+00:00", "status": "done",
                "attempts": 1, "error": None, "proposed": 1, "record_format": 2, "model": QWEN.model,
                "prompt_version": PROMPT_VERSION, "validator_version": VALIDATOR_VERSION, "response_sha256": "b" * 64,
                "usage": {"completion_tokens": 230}, "candidates": [GOOD], "accepted": [GOOD], "rejected": [],
            }, sort_keys=True) + "\n")
    return run_path(directory, "train").read_bytes(), records_path(directory, "train").read_bytes()


def test_changed_request_settings_continue_the_run_without_touching_its_records(tmp_path: Path) -> None:
    run_bytes, records_bytes = legacy_run(tmp_path)
    legacy_sha = json.loads(run_bytes)["params_sha256"]
    chat = Scripted(default=GOOD_REPLY)
    with pytest.raises(GenerationError, match="only request settings differ.*--continue-run"):
        generator(chat, tmp_path, PACED).run("train", passages(3))
    assert chat.calls == 0 and not continuations_path(tmp_path, "train").exists()

    check = check_run(tmp_path, "train", PACED, "a" * 64)
    assert check.state == "continuable" and check.changed == {
        "max_completion_tokens": {"from": 2048, "to": 768}, "max_rate_limit_wait_seconds": {"to": 180.0},
        "output_tokens_per_minute": {"to": 1000}}
    assert describe_changes(check.changed) == (
        "max_completion_tokens 2048 -> 768, max_rate_limit_wait_seconds unset -> 180.0, "
        "output_tokens_per_minute unset -> 1000")

    gen = generator(chat, tmp_path, PACED, continue_run=True)
    counts = gen.run("train", passages(3))
    assert (counts["skipped_done"], counts["done"], chat.calls) == (2, 1, 1)  # completed passages are not regenerated
    assert run_path(tmp_path, "train").read_bytes() == run_bytes  # the original binding is never rewritten
    after = records_path(tmp_path, "train").read_bytes()
    assert after.startswith(records_bytes) and after.count(b"\n") == 3
    new = read_records(records_path(tmp_path, "train"))[-1]
    assert (new["passage_id"], new["params_sha256"], new["record_format"]) == ("p3", PACED.params_sha256(), 3)
    (entry,) = read_records(continuations_path(tmp_path, "train"))
    assert entry == gen.continuation == {
        "format_version": 1, "kind": "continuation", "split": "train", "continuation": 1,
        "of_run_sha256": hashlib.sha256(run_bytes).hexdigest(), "previous_params_sha256": legacy_sha,
        "params": PACED.params(), "params_sha256": PACED.params_sha256(), "changed": check.changed,
        "records_before": 2, "records_sha256_before": hashlib.sha256(records_bytes).hexdigest(),
        "created_at": "2026-01-01T00:00:00+00:00"}

    again = generator(chat, tmp_path, PACED).run("train", passages(4))  # bound now: no flag, no new continuation
    assert again["skipped_done"] == 3 and again["done"] == 1
    assert check_run(tmp_path, "train", PACED, "a" * 64).state == "bound"
    assert len(read_records(continuations_path(tmp_path, "train"))) == 1
    sources = {p.passage_id: p for p in passages(4)}
    assert revalidate_records(tmp_path, "train", sources, PACED, ALIASES, prepared_manifest_sha256="a" * 64)[
        "up_to_date"] == 4
    hotter = PACED.model_copy(update={"temperature": 0.2})  # shapes replies: never a continuation
    with pytest.raises(GenerationError, match="different run.*temperature changed"):
        generator(chat, tmp_path, hotter, continue_run=True).run("train", passages(4))


def test_a_continuation_detects_rewritten_records(tmp_path: Path) -> None:
    legacy_run(tmp_path)
    generator(Scripted(default=GOOD_REPLY), tmp_path, PACED, continue_run=True).run("train", passages(3))
    path = records_path(tmp_path, "train")
    path.write_bytes(path.read_bytes().replace(b'"p1"', b'"p9"', 1))
    with pytest.raises(GenerationError, match="does not match the run or its records"):
        check_run(tmp_path, "train", PACED, "a" * 64)
