"""Tests for versioned generation prompts: sec-qgen-v1 stays as recorded, sec-qgen-v2 is a new run (offline)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from finretriever.training.query_generation import (
    LATEST_PROMPT_VERSION,
    PROMPT_VERSION,
    PROMPTS,
    V1_QUESTION_TYPES,
    GenerationError,
    GenerationSettings,
    QueryGenerator,
    SourcePassage,
    build_messages,
    check_run,
    generation_run_info,
    prompt_sha256,
    read_records,
    records_path,
)
from tests.unit.training_fixtures import SECRET

ROOT = Path(__file__).parents[2]
TABLE = SourcePassage(
    passage_id="p1", split="train", group_id="grp_00001", ticker="AAPL", fiscal_year="2024", company="Apple Inc.",
    form_type="10-K", filing_date="2024-11-01", report_date="2024-09-28", section="Item 8 Financial Statements",
    visible_text="[TABLE] Total net sales | 2024: $391,035 | 2023: $383,285 [/TABLE] Net sales rose on Services.",
)
PROSE = SourcePassage(**{**TABLE.__dict__, "visible_text": "Services net sales rose because of advertising and cloud."})
V1_SHA256 = "81626417e69c071f6b156eb31a7810702ad0d56e38bf19990bf471d51237049a"  # recorded by the first Qwen run


def test_the_first_prompt_is_unchanged_and_stays_the_default() -> None:
    assert PROMPT_VERSION == "sec-qgen-v1" and GenerationSettings(model="m").prompt_version == PROMPT_VERSION
    assert prompt_sha256("sec-qgen-v1") == prompt_sha256() == V1_SHA256  # the identity runs are bound to
    system, user = build_messages(TABLE, 2)
    assert "for the fiscal year ended 2024-09-28, filed 2024-11-01" in user["content"]
    assert f"(one of: {', '.join(V1_QUESTION_TYPES)})" in user["content"]  # v1's own types, never the newer ones
    assert "profitability" not in user["content"] and system["content"] == PROMPTS["sec-qgen-v1"].system


def test_v2_asks_for_natural_periods_substance_units_and_no_calculation() -> None:
    assert LATEST_PROMPT_VERSION == "sec-qgen-v2" and prompt_sha256("sec-qgen-v2") != V1_SHA256
    system, user = build_messages(TABLE, 2, "sec-qgen-v2")
    text = user["content"]
    assert "for fiscal 2024 (period ended September 28, 2024), filed November 1, 2024" in text
    assert "2024-09-28" not in text and "2024-11-01" not in text  # no ISO dates for the model to copy
    for rule in ('"fiscal 2024" or "FY2024"', "exhibit or note numbers", "Do not add, subtract, divide, convert",
                 "keep each figure with its unit", "Never copy more than five consecutive words",
                 'never shorten a quote with "..."', "profitability, accounting_policy, tax"):
        assert rule in text, rule
    assert "naming its line item, unit and period" in text  # the v2 table hint
    assert "naming its line item" not in build_messages(PROSE, 2, "sec-qgen-v2")[1]["content"]
    assert "[TABLE] Total net sales" in text and "Write 2 different questions" in text
    with pytest.raises(ValueError, match="unknown prompt version"):
        build_messages(TABLE, 2, "sec-qgen-v9")


def test_no_prompt_carries_example_questions_or_evaluation_text() -> None:
    for prompt in PROMPTS.values():
        # The only "questions" in a template are the JSON field placeholders.
        assert not re.search(r'"question": "[^.]', prompt.user_template)
    evaluation = ROOT / "data/eval/annotation/questions.jsonl"
    if not evaluation.exists():  # the human-written questions can be withheld from a public checkout
        pytest.skip("the evaluation questions are not in this checkout")
    questions = [json.loads(line)["text"] for line in evaluation.read_text().splitlines() if line.strip()]
    assert questions
    for prompt in PROMPTS.values():
        templates = " ".join((prompt.system, prompt.user_template, prompt.table_hint)).casefold()
        assert not any(q.casefold()[:40] in templates for q in questions)


def test_prompt_versions_bind_runs_and_never_mix(tmp_path: Path) -> None:
    v1 = GenerationSettings(model="test-model")
    v2 = v1.model_copy(update={"prompt_version": "sec-qgen-v2"})
    assert v1.params() == v2.params() and "prompt_version" not in v1.params()  # recorded with the run instead
    info = generation_run_info(v2, "train", "a" * 64)
    assert (info["prompt_version"], info["prompt_sha256"]) == ("sec-qgen-v2", prompt_sha256("sec-qgen-v2"))

    class Chat:
        def complete(self, messages: list[dict[str, str]]) -> str:
            assert "for fiscal 2024" in messages[1]["content"]
            return json.dumps({"questions": []})

    gen = QueryGenerator(Chat(), v2, None, directory=tmp_path / "v2", prepared_manifest_sha256="a" * 64,  # type: ignore[arg-type]
                         clock=lambda: "2026-01-01T00:00:00+00:00", secret=SECRET)
    gen.run("train", [TABLE])
    (record,) = read_records(records_path(tmp_path / "v2", "train"))
    assert record["prompt_version"] == "sec-qgen-v2"
    assert check_run(tmp_path / "v2", "train", v2, "a" * 64).state == "bound"
    with pytest.raises(GenerationError, match="different run"):  # a prompt change is never a continuation
        check_run(tmp_path / "v2", "train", v1, "a" * 64)
