"""Synthetic analyst queries for approved training passages, through Groq's chat API.

For each approved passage (train or validation, never both, never blocked) the model sees only
the tokenizer-visible part of the passage (``finretriever.training.visible_text``) with its
company, filing and section, and is asked for a few questions with a short answer and a
verbatim evidence quote. Every candidate is checked locally before it is accepted:

* well-formed JSON fields and an allowed question type;
* a question of 6-40 words ending in "?", that names the passage's company, names no other
  corpus company, does not refer to "the passage" or "the table", and copies no run of
  ``max_copied_words`` words from the passage. Since v3, words of a table title or row label
  in the passage, and standard line items and accounting terms (``STANDARD_TERMS``), do not
  count towards a copied run: an analyst types "allowance for credit losses" as it is;
* evidence (15-300 characters) located in the visible text, so the answer never depends on
  truncated text (lightweight answerability check). v2 required a verbatim quote; v3 also
  accepts verbatim fragments joined by an ellipsis, cells of one table row with cells skipped,
  and a row label restated with the parent the flattened table shortened
  (``finretriever.training.quotes``). Invented cells, labels or words still fail;
* every number in the question and answer present in the visible text (or the filing's
  year), and the answer's numbers in its evidence, so no figure is invented or converted. A
  complete date in the question equal to the filing's report date or filing date (ISO, US
  numeric or written form) is verified metadata and exempt as a whole; its day and month are
  never allowed on their own;
* every other complete date in the question or answer found in the visible text (in any of
  those forms, or as the same month and day with a year the passage or filing gives).

Every proposed candidate is recorded with its fields and, if rejected, the reason, so the checks
can be re-run locally after a validator change (``revalidate_records``): revalidation appends a
revision record and never rewrites an earlier one.

Global checks (duplicates across passages, overlap with evaluation questions) run when the
pairs are built (``finretriever.training.pair_quality``). Evaluation questions and labels are
never part of a prompt.

Generation is resumable: each passage's outcome is appended to ``<split>.records.jsonl`` and
flushed before the next request; a rerun skips passages already done. ``<split>.run.json`` binds
the directory to the prompt, model, parameters and preparation, and a rerun with other settings
is refused, except that request settings (``REQUEST_PARAMS``: token limit, pacing, retries) may
change through a recorded continuation (``check_run``): one line in
``<split>.run.continuations.jsonl`` with the changes and a checksum of the records so far.
Nothing earlier is rewritten, and each later record names the parameters it was generated with.

Requests are spaced to ``requests_per_minute`` and, when ``output_tokens_per_minute`` is set,
paced so that the output tokens of the last minute plus the next request's
``max_completion_tokens`` stay within it; a limit that alone exceeds it is refused before any
request. HTTP 429 waits as long as ``Retry-After``, an exhausted ``x-ratelimit-reset-*`` header or
the error message asks, then retries; a wait longer than ``max_rate_limit_wait_seconds``, or a
rate limit that outlasts the retries, stops the run cleanly without recording the passage. A
request refused as too large (HTTP 413, or more tokens requested than a limit allows) is never
retried and stops the run. 5xx and network errors are retried with backoff; HTTP 401/403 stop the
run at once. The API key is read from the environment, sent only in the Authorization header,
and redacted from every recorded error.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finretriever.eval_set.annotation_store import utc_now
from finretriever.training.quotes import locate_quote, table_label_grams

logger = logging.getLogger(__name__)

# Prompts are versioned (``PROMPTS``). A config chooses one with ``generation.prompt_version``; without it a
# config keeps the first prompt, so runs recorded before prompt versions existed stay bound to theirs.
PROMPT_VERSION = "sec-qgen-v1"  # the default
LATEST_PROMPT_VERSION = "sec-qgen-v2"
# The candidate checks. v1 (records without the field) treated a metadata date's day and month as
# unsupported numbers; v2 exempts complete report and filing dates; v3 locates quotes that skip table
# cells, join fragments with an ellipsis or restate a shortened row label, and does not count table
# labels or standard accounting terms as copying.
VALIDATOR_VERSION = "sec-qcheck-v3"
LEGACY_VALIDATOR_VERSION = "sec-qcheck-v1"
# Record format 2 adds every proposed candidate, the model, prompt and validator versions; format 3 adds the
# parameters' SHA-256 (``params_sha256``), which a continuation can change, and ``rate_limited`` on failures.
RECORD_FORMAT_VERSION = 3
CANDIDATE_FIELDS: tuple[str, ...] = ("question", "type", "answer", "evidence")
# The types sec-qgen-v1 offers; fixed, since the prompt text sent under that version must never change.
V1_QUESTION_TYPES: tuple[str, ...] = (
    "risk",
    "mdna",
    "financial_table",
    "debt_liquidity",
    "revenue",
    "other",
)
V2_QUESTION_TYPES: tuple[str, ...] = (
    "risk",
    "mdna",
    "financial_table",
    "debt_liquidity",
    "revenue",
    "profitability",
    "accounting_policy",
    "tax",
    "other",
)
QUESTION_TYPES: tuple[str, ...] = tuple(dict.fromkeys(V1_QUESTION_TYPES + V2_QUESTION_TYPES))
SYSTEM_PROMPT = (
    "You write search queries that a financial analyst would type to find one specific passage in a "
    "company's SEC Form 10-K. You answer with a single JSON object and nothing else."
)
USER_TEMPLATE = """Company: {company} ({ticker})
Filing: Form {form_type} for the fiscal year ended {report_date}, filed {filing_date}
Section: {section}

Passage (it may stop mid-sentence; [TABLE] ... [/TABLE] marks a table whose cells are separated by |):
<<<
{text}
>>>

Write {n} different questions that this passage answers.
Rules:
1. Each question must be answerable from this passage alone, without any other document.
2. Name the company and the fiscal year or period in every question.
3. Use your own words: never copy more than five consecutive words from the passage.
4. Use only numbers, dates and names that appear in the passage; do not convert units or compute new figures.
5. Do not mention "the passage", "the text", "the excerpt" or "the table".
6. Do not ask about the last sentence if it is cut off.
7. Prefer different kinds of question.{table_hint}
For each question give "question", "type" (one of: {types}), "answer" (a short answer from the passage) and
"evidence" (an exact quote from the passage, at most 300 characters, that contains the answer).
Return {{"questions": [{{"question": "...", "type": "...", "answer": "...", "evidence": "..."}}]}}"""
TABLE_HINT = " The passage contains a financial table: at least one question should ask for a figure from it."
SYSTEM_PROMPT_V2 = (
    "You write the questions a financial analyst would type to find one specific passage in a company's SEC "
    "Form 10-K. You answer with a single JSON object and nothing else."
)
# No example questions: nothing from the evaluation set (or anywhere else) can leak into replies through the prompt.
USER_TEMPLATE_V2 = """Company: {company} ({ticker})
Filing: Form {form_type} for fiscal {fiscal_year} (period ended {period_end}), filed {filed}
Section: {section}

Passage (it may stop mid-sentence; [TABLE] ... [/TABLE] marks a table whose cells are separated by |):
<<<
{text}
>>>

Write {n} different questions that an analyst could answer from this passage alone.
Rules:
1. Ask about substance: reported figures, financial tables, revenue, profitability, liquidity and debt, risks and
their possible effects, or accounting policies. Do not ask for exhibit or note numbers, form or item names, filing
dates, page references, or where documents can be found.
2. Name the company and the period as an analyst would: "fiscal 2024" or "FY2024", or a period the passage states
(for example "the year ended December 31, 2023"). Write a full date only when the question depends on it, never as
YYYY-MM-DD. A figure for an earlier year shown in the passage belongs to that year: name that year.
3. Write natural questions in your own words. Never copy more than five consecutive words from the passage; the
name of a line item or an accounting term may be used as written.
4. Use only figures, units and periods stated in the passage, and keep each figure with its unit (thousands,
millions, billions, percent) and its period. Do not add, subtract, divide, convert or otherwise calculate new
figures.
5. Ask only what the passage states; do not ask about the last sentence if it is cut off.
6. Do not mention "the passage", "the text", "the excerpt" or "the table".
7. Prefer different kinds of question.{table_hint}
For each question give "question", "type" (one of: risk, mdna, financial_table, debt_liquidity, revenue,
profitability, accounting_policy, tax, other), "answer" (a short answer stated in the passage, with its unit and
period) and "evidence" (one quote copied exactly from the passage, at most 300 characters, that contains the
answer; for a table row, copy its label and every cell up to the answer, and never shorten a quote with "...").
Return {{"questions": [{{"question": "...", "type": "...", "answer": "...", "evidence": "..."}}]}}"""
TABLE_HINT_V2 = (
    " The passage contains a financial table: at least one question should ask for a figure from it, naming its "
    "line item, unit and period."
)


@dataclass(frozen=True)
class Prompt:
    """One prompt version: its system message, user template and table hint."""

    version: str
    system: str
    user_template: str
    table_hint: str
    types: tuple[str, ...] = ()  # filled into a ``{types}`` placeholder (sec-qgen-v1 only)

    def sha256(self) -> str:
        """Identity of the prompt: its version and templates (sec-qgen-v1's as recorded since its first run)."""
        return hashlib.sha256(
            f"{self.version}\n{self.system}\n{self.user_template}\n{self.table_hint}".encode()
        ).hexdigest()


PROMPTS: dict[str, Prompt] = {
    "sec-qgen-v1": Prompt(
        "sec-qgen-v1", SYSTEM_PROMPT, USER_TEMPLATE, TABLE_HINT, V1_QUESTION_TYPES
    ),
    "sec-qgen-v2": Prompt("sec-qgen-v2", SYSTEM_PROMPT_V2, USER_TEMPLATE_V2, TABLE_HINT_V2),
}
PROMPT_VERSIONS: tuple[str, ...] = tuple(PROMPTS)
REJECTION_REASONS: tuple[str, ...] = (
    "malformed",
    "question_format",
    "refers_to_context",
    "missing_company",
    "other_company",
    "copied_from_passage",
    "evidence_length",
    "evidence_not_in_visible_text",
    "number_not_in_passage",
    "date_not_in_passage",
    "answer_not_supported",
    "duplicate_in_passage",
    "excess_question",
)


def prompt_sha256(version: str = PROMPT_VERSION) -> str:
    """Identity of a prompt version: its version and templates."""
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version {version!r}; choose from {', '.join(PROMPTS)}")
    return PROMPTS[version].sha256()


class GenerationSettings(BaseModel):
    """Provider, model, request and validation settings; recorded with every output (no secrets)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["groq"] = "groq"
    model: str
    # Recorded with the run as prompt_version and prompt_sha256, not among ``params``: configs written before
    # prompt versions existed keep their parameter hash, and a new prompt is a new run (the identity differs).
    prompt_version: Literal["sec-qgen-v1", "sec-qgen-v2"] = PROMPT_VERSION
    api_base: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"
    temperature: float = Field(default=0.7, ge=0, le=2)
    # Sent as ``max_completion_tokens``; on reasoning models the limit can be spent on reasoning first.
    max_completion_tokens: int = Field(default=900, ge=64)
    # The account's output-token limit per minute for this model (Groq's OTPM). Requests are paced so that the
    # output tokens of the last minute plus one more ``max_completion_tokens`` never exceed it. Unset: not paced.
    output_tokens_per_minute: int | None = Field(default=None, ge=1)
    json_mode: bool = True
    # Reasoning models (e.g. Qwen 3.8 on Groq). Unset values are not sent. ``raw`` puts reasoning in the
    # reply inside <think> tags and is refused with JSON mode (Groq answers HTTP 400).
    reasoning_format: Literal["parsed", "hidden", "raw"] | None = None
    reasoning_effort: Literal["none", "default", "low", "medium", "high"] | None = None
    queries_per_passage: dict[Literal["train", "validation"], int] = Field(
        default_factory=lambda: {"train": 2, "validation": 1}
    )
    requests_per_minute: float = Field(default=20.0, gt=0)
    # The longest wait a rate limit may ask for (Retry-After, x-ratelimit-reset-*) before the run stops instead.
    max_rate_limit_wait_seconds: float = Field(default=180.0, ge=0)
    max_retries: int = Field(default=4, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    backoff_seconds: float = Field(default=2.0, ge=0)
    max_backoff_seconds: float = Field(default=60.0, ge=0)
    max_response_attempts: int = Field(default=2, ge=1)
    max_consecutive_failures: int = Field(default=3, ge=1)
    min_question_words: int = 6
    max_question_words: int = 40
    max_copied_words: int = 8
    min_evidence_chars: int = 15
    max_evidence_chars: int = 300
    near_duplicate_threshold: float = 0.9
    company_aliases: dict[str, list[str]] = Field(default_factory=dict)  # extra names per ticker

    @model_validator(mode="after")
    def _reasoning_and_json(self) -> GenerationSettings:
        if self.json_mode and self.reasoning_format == "raw":
            raise ValueError(
                "reasoning_format raw cannot be combined with JSON mode (the API answers HTTP 400); "
                "use parsed or hidden"
            )
        return self

    @model_validator(mode="after")
    def _completion_fits_output_limit(self) -> GenerationSettings:
        limit = self.output_tokens_per_minute
        if limit is not None and self.max_completion_tokens > limit:
            raise ValueError(
                f"max_completion_tokens {self.max_completion_tokens} exceeds output_tokens_per_minute {limit}: the "
                "API could never admit such a request; lower max_completion_tokens"
            )
        return self

    def params(self) -> dict[str, Any]:
        """Every setting but the prompt version, as recorded with outputs; ``REQUEST_PARAMS`` are the ones that
        do not shape replies. The prompt is recorded separately, with its SHA-256."""
        return self.model_dump(mode="json", exclude={"prompt_version"})

    def params_sha256(self) -> str:
        return _params_sha256(self.params())


def _params_sha256(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


# Settings that decide how and how fast requests are sent, never what an accepted reply says, so a run may change
# them through a recorded continuation (``check_run``). ``max_completion_tokens`` only ends a reply early, and a
# reply that reaches it is recorded as failed (finish_reason length), never accepted.
REQUEST_PARAMS = frozenset(
    {
        "max_completion_tokens",
        "output_tokens_per_minute",
        "requests_per_minute",
        "max_rate_limit_wait_seconds",
        "max_retries",
        "timeout_seconds",
        "backoff_seconds",
        "max_backoff_seconds",
        "max_consecutive_failures",
    }
)


# --- errors and the API client --------------------------------------------------------------------


class GenerationError(RuntimeError):
    """Generation cannot continue (configuration, records, or repeated request failures).

    ``status`` and ``code`` carry the HTTP status and the API's error code when a response gave them.
    """

    def __init__(
        self, message: str = "", *, status: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class CredentialError(GenerationError):
    """No usable API key: missing from the environment, or rejected by the API."""


class AuthError(CredentialError):
    """The API rejected the credentials; the run stops at once."""


class TransientError(GenerationError):
    """Rate limits, server errors or network failures outlasted every retry."""


class RequestRejected(GenerationError):
    """The API refused one request (HTTP 4xx other than 401/403/404/429)."""


class ResponseTruncated(GenerationError):
    """The reply stopped at the token limit (``finish_reason`` "length"); retrying with the same limit would
    usually stop again, so the passage is recorded as failed instead."""


class ModelUnavailable(GenerationError):
    """The configured model does not exist or this key cannot use it: every request would fail, so the run
    stops at once (HTTP 404, or an error code in ``MODEL_ERROR_CODES``)."""


MODEL_ERROR_CODES = frozenset({"model_not_found", "model_decommissioned"})


class MalformedResponse(GenerationError):
    """The reply was not the JSON the prompt asks for."""


class RequestBudgetExhausted(GenerationError):
    """The run's request budget (every HTTP attempt, retries included) is spent; the run stops cleanly."""


class RateLimited(GenerationError):
    """A rate limit asks for a longer wait than ``max_rate_limit_wait_seconds``, or outlasted every retry; the run
    stops cleanly and a rerun resumes the passage."""


class RequestTooLarge(GenerationError):
    """The request can never be served under the account's limits (HTTP 413, "request too large", or more tokens
    requested than a limit allows), so waiting cannot help: the run stops without recording the passage.
    """


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: Mapping[str, str]
    body: str


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], TransportResponse]


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


def api_key_from_env(settings: GenerationSettings) -> str:
    """The API key from ``settings.api_key_env``; never logged or recorded."""
    key = os.environ.get(settings.api_key_env, "").strip()
    if not key:
        raise CredentialError(
            f"set {settings.api_key_env} in the environment to call the {settings.provider} API"
        )
    return key


def redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "[redacted]") if secret else text


def _requests_transport(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> TransportResponse:
    import requests

    response = requests.post(url, headers=dict(headers), json=dict(payload), timeout=timeout)
    return TransportResponse(response.status_code, dict(response.headers), response.text)


OUTPUT_WINDOW_SECONDS = 60.0
# Added to every wait the client computes or is asked for, for clock and network differences with the API.
PACING_MARGIN_SECONDS = 1.0
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_SCALE = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
_TRY_AGAIN_RE = re.compile(r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", flags=re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit:?\s+(\d+)", flags=re.IGNORECASE)
_REQUESTED_RE = re.compile(r"\brequested:?\s+(\d+)", flags=re.IGNORECASE)


def parse_duration(text: str) -> float | None:
    """Seconds in a rate-limit duration ("7.66s", "2m59.56s", "1h2m", "120ms" or plain seconds); None if unreadable."""
    text = text.strip().lower()
    try:
        seconds = float(text)
    except ValueError:
        parts = _DURATION_PART_RE.findall(text)
        if not parts or "".join(n + unit for n, unit in parts) != text:
            return None
        seconds = sum(float(n) * _DURATION_SCALE[unit] for n, unit in parts)
    return max(seconds, 0.0) if math.isfinite(seconds) else None


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """The wait a ``Retry-After`` header asks for (seconds or an HTTP date), if there is one."""
    value = next((str(v).strip() for k, v in headers.items() if k.lower() == "retry-after"), "")
    if not value:
        return None
    seconds = parse_duration(value)
    if seconds is not None:
        return seconds
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(when.timestamp() - time.time(), 0.0)


def _too_large(status: int, body: str) -> bool:
    """Whether the API refused a request as larger than a limit allows at all: HTTP 413, "request too large", or more
    tokens requested than the limit (Groq: "Limit 1000, Requested 2048"). A rate limit that was merely reached
    ("Limit 1000, Used 900, Requested 300") is temporary instead."""
    if status == 413 or "request too large" in body.lower():
        return True
    limit, requested = _LIMIT_RE.search(body), _REQUESTED_RE.search(body)
    return bool(limit and requested and int(requested.group(1)) > int(limit.group(1)))


class GroqClient:
    """OpenAI-compatible chat completions on Groq, with request and output-token pacing, retries and backoff."""

    RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        settings: GenerationSettings,
        *,
        api_key: str,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_requests: int | None = None,
    ) -> None:
        if not api_key:
            raise GenerationError("an API key is required")
        if max_requests is not None and max_requests < 1:
            raise GenerationError("max_requests must be at least 1")
        limit = settings.output_tokens_per_minute
        if (
            limit is not None and settings.max_completion_tokens > limit
        ):  # settings copied without validation
            raise RequestTooLarge(
                f"max_completion_tokens {settings.max_completion_tokens} exceeds output_tokens_per_minute {limit}: "
                "the API could never admit such a request; lower generation.max_completion_tokens"
            )
        self.settings = settings
        self._key = api_key
        self._transport = transport or _requests_transport
        self._sleep = sleep
        self._clock = clock
        self._next_request = 0.0
        self._not_before = 0.0  # the earliest next request a rate limit allows
        self._output: deque[tuple[float, int]] = deque()  # (time, output tokens) of recent requests
        self.max_requests = max_requests
        self.requests_sent = 0  # every HTTP attempt, retries included
        self.usage: Counter[str] = Counter()  # token counts the API reported, summed
        self.rate_limit: dict[str, str] = {}  # the last x-ratelimit-* headers seen (no secrets)
        self.rate_limited = 0  # HTTP 429 responses
        self.waited_seconds = 0.0  # time spent pacing and waiting for rate limits
        self.stop_reason: str | None = None  # why a rate limit stopped the run

    def _output_wait(self, now: float) -> float:
        """Seconds until the output tokens of the last minute leave room for one more ``max_completion_tokens``."""
        limit = self.settings.output_tokens_per_minute
        if limit is None:
            return 0.0
        horizon = (
            OUTPUT_WINDOW_SECONDS + PACING_MARGIN_SECONDS
        )  # tokens count against the limit this long
        while self._output and self._output[0][0] + horizon <= now:
            self._output.popleft()
        room = limit - self.settings.max_completion_tokens  # what earlier requests may still hold
        used, wait = sum(tokens for _, tokens in self._output), 0.0
        for at, tokens in self._output:  # oldest first
            if used <= room:
                break
            used -= tokens
            wait = at + horizon - now
        return wait

    def _spend(self, tokens: int) -> None:
        if self.settings.output_tokens_per_minute is not None and tokens > 0:
            self._output.append((self._clock(), tokens))

    def _wait_turn(self) -> None:
        s, now = self.settings, self._clock()
        asked = self._not_before - now
        if asked > s.max_rate_limit_wait_seconds:
            self.stop_reason = (
                f"the API asks to wait {asked:.0f}s, longer than generation.max_rate_limit_wait_seconds "
                f"({s.max_rate_limit_wait_seconds:g}s)"
            )
            raise RateLimited(f"{self.stop_reason}; rerun after that to resume")
        wait = max(self._next_request - now, asked, self._output_wait(now))
        if wait > 0:
            self._sleep(wait)
            self.waited_seconds += wait
        self._next_request = self._clock() + 60.0 / s.requests_per_minute

    def _read_limits(self, headers: Mapping[str, str]) -> None:
        """Keep the x-ratelimit-* headers, and wait for the reset of any limit left too low for another request."""
        now = self._clock()
        found = {
            k.lower(): str(v) for k, v in headers.items() if k.lower().startswith("x-ratelimit-")
        }
        self.rate_limit.update(found)
        for key, value in found.items():
            if not key.startswith("x-ratelimit-remaining-"):
                continue
            kind = key.removeprefix("x-ratelimit-remaining-")
            need = 1 if "request" in kind else self.settings.max_completion_tokens
            reset = parse_duration(found.get(f"x-ratelimit-reset-{kind}", ""))
            try:
                low = float(value) < need
            except ValueError:
                continue
            if low and reset is not None:
                self._not_before = max(self._not_before, now + reset + PACING_MARGIN_SECONDS)

    def _rate_limited(self, response: TransportResponse, attempt: int) -> None:
        """Schedule the retry of an HTTP 429 as the response asks, or with backoff if it gives no wait."""
        s, now = self.settings, self._clock()
        self.rate_limited += 1
        asked = retry_after_seconds(response.headers)
        if asked is None:
            found = _TRY_AGAIN_RE.search(response.body)
            asked = parse_duration(found.group(1)) if found else None
        if asked is not None:
            self._not_before = max(self._not_before, now + asked + PACING_MARGIN_SECONDS)
        elif self._not_before <= now:  # no wait named anywhere
            self._not_before = now + min(s.backoff_seconds * 2**attempt, s.max_backoff_seconds)
        logger.warning("rate limited (HTTP 429); next attempt in %.1fs", self._not_before - now)

    def complete(self, messages: list[dict[str, str]]) -> str:
        s = self.settings
        payload: dict[str, Any] = {
            "model": s.model,
            "messages": messages,
            "temperature": s.temperature,
            "max_completion_tokens": s.max_completion_tokens,
        }
        if s.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if s.reasoning_format is not None:
            payload["reasoning_format"] = s.reasoning_format
        if s.reasoning_effort is not None:
            payload["reasoning_effort"] = s.reasoning_effort
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        url = f"{s.api_base.rstrip('/')}/chat/completions"
        last, last_status, limited = "", None, False
        for attempt in range(s.max_retries + 1):
            if self.max_requests is not None and self.requests_sent >= self.max_requests:
                raise RequestBudgetExhausted(f"the request budget of {self.max_requests} is spent")
            self._wait_turn()
            self.requests_sent += 1
            spent, delay = (
                s.max_completion_tokens,
                0.0,
            )  # output tokens: the whole limit until a reply says less
            try:
                response = self._transport(url, headers, payload, s.timeout_seconds)
            except Exception as exc:  # noqa: BLE001 - any network failure is retried, then reported
                last, last_status, limited = (
                    redact(f"{type(exc).__name__}: {exc}", self._key),
                    None,
                    False,
                )
                delay = min(s.backoff_seconds * 2**attempt, s.max_backoff_seconds)
            else:
                self._read_limits(response.headers)
                if response.status < 500 and response.status != 200:
                    spent = 0  # refused before generating anything
                if response.status in (401, 403):
                    raise AuthError(
                        f"the {s.provider} API rejected the credentials (HTTP {response.status}); check {s.api_key_env}"
                    )
                if response.status == 200:
                    data = _json_body(response.body)
                    usage = _reported_usage(data)
                    self.usage.update(usage)  # counted even if the reply is unusable
                    spent = usage.get("completion_tokens", spent)
                    return _reply_text(data)
                last, last_status = (
                    redact(f"HTTP {response.status}: {response.body[:200]}", self._key),
                    response.status,
                )
                code = _error_code(response.body)
                if response.status == 404 or code in MODEL_ERROR_CODES:
                    raise ModelUnavailable(
                        f"the {s.provider} API refused model {s.model!r} for this API key (HTTP {response.status}, "
                        f"{code or 'no error code'}): it does not exist or the account cannot use it. Set "
                        "generation.model to a model your account lists; a new model is a new run, so move the old "
                        "generation directory aside first",
                        status=response.status,
                        code=code,
                    )
                if _too_large(response.status, response.body):
                    raise RequestTooLarge(
                        f"the {s.provider} API refused the request as larger than the account's limits allow ({last}); "
                        "waiting cannot help. Lower generation.max_completion_tokens (every request reserves it) or set "
                        "generation.output_tokens_per_minute to the account's limit; nothing was recorded for this passage",
                        status=response.status,
                        code=code,
                    )
                if response.status not in self.RETRY_STATUSES:
                    raise RequestRejected(last, status=response.status, code=code)
                limited = response.status == 429
                if limited:
                    self._rate_limited(
                        response, attempt
                    )  # the wait happens in _wait_turn, which caps it
                else:
                    delay = min(s.backoff_seconds * 2**attempt, s.max_backoff_seconds)
            finally:
                self._spend(spent)
            if attempt < s.max_retries and delay > 0:
                logger.warning("request failed (%s); retrying in %.1fs", last, delay)
                self._sleep(delay)
        if limited:
            self.stop_reason = f"still rate limited after {s.max_retries + 1} attempts ({last})"
            raise RateLimited(f"{self.stop_reason}; rerun later to resume", status=429)
        raise TransientError(
            f"request failed after {s.max_retries + 1} attempts: {last}", status=last_status
        )


def _error_code(body: str) -> str | None:
    """The ``error.code`` of an OpenAI-style error body, if any."""
    try:
        code = json.loads(body)["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return None
    return code if isinstance(code, str) else None


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _json_body(body: str) -> dict[str, Any]:
    try:
        data = json.loads(body)
    except ValueError:
        raise MalformedResponse("unexpected API response (not JSON)") from None
    if not isinstance(data, dict):
        raise MalformedResponse("unexpected API response (not a JSON object)")
    return data


def _reported_usage(data: Mapping[str, Any]) -> dict[str, int]:
    """The token counts the API reported (``usage``), if any, including reasoning tokens."""
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    tokens = {
        k: int(v)
        for k, v in usage.items()
        if k in ("prompt_tokens", "completion_tokens", "total_tokens") and isinstance(v, int)
    }
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if isinstance(reasoning, int):
        tokens["reasoning_tokens"] = reasoning
    return tokens


def _reply_text(data: Mapping[str, Any]) -> str:
    """The reply text, without reasoning.

    A separate ``message.reasoning`` field is ignored and ``<think>...</think>`` blocks are removed
    from the text. A reply cut off at the token limit raises ``ResponseTruncated``.
    """
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise MalformedResponse(f"unexpected API response ({type(exc).__name__})") from None
    if choice.get("finish_reason") == "length":
        raise ResponseTruncated(
            "the reply stopped at the token limit (finish_reason length); raise "
            "generation.max_completion_tokens or reduce reasoning",
            status=200,
            code="length",
        )
    if not isinstance(content, str):
        raise MalformedResponse("the API response has no text content")
    content = _THINK_RE.sub("", content).strip()
    if "<think>" in content:
        raise MalformedResponse(
            "the reply contains unfinished reasoning (<think> without </think>)"
        )
    if not content:
        raise MalformedResponse("the API response has no text content")
    return content


# --- prompts and candidate checks -----------------------------------------------------------------


@dataclass(frozen=True)
class SourcePassage:
    """An approved passage as the generator sees it: metadata and the visible text only."""

    passage_id: str
    split: Literal["train", "validation"]
    group_id: str
    ticker: str
    fiscal_year: str
    company: str
    form_type: str
    filing_date: str
    report_date: str
    section: str
    visible_text: str
    item: str = ""  # the 10-K item number, if known (used by the source-quality filter)


def company_short(company: str) -> str:
    match = re.search(r"[A-Za-z]+", company)
    return match.group(0).capitalize() if match else company


def written_date(iso: str) -> str:
    """``2024-09-28`` as ``September 28, 2024`` (the text unchanged if it is not an ISO date)."""
    try:
        day = date.fromisoformat(iso[:10])
    except ValueError:
        return iso
    return f"{day.strftime('%B')} {day.day}, {day.year}"


def build_messages(
    passage: SourcePassage, n: int, prompt_version: str = PROMPT_VERSION
) -> list[dict[str, str]]:
    """The messages of ``prompt_version`` for ``passage``: its metadata and visible text only."""
    if prompt_version not in PROMPTS:
        raise ValueError(f"unknown prompt version {prompt_version!r}")
    prompt = PROMPTS[prompt_version]
    has_table = "[TABLE]" in passage.visible_text
    user = prompt.user_template.format(
        company=passage.company,
        ticker=passage.ticker,
        form_type=passage.form_type,
        report_date=passage.report_date,
        filing_date=passage.filing_date,
        fiscal_year=passage.fiscal_year,
        period_end=written_date(passage.report_date),
        filed=written_date(passage.filing_date),
        section=passage.section,
        text=passage.visible_text,
        n=n,
        table_hint=prompt.table_hint if has_table else "",
        types=", ".join(prompt.types),
    )
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]


def parse_candidates(content: str) -> list[Any]:
    """The ``questions`` list of a reply (code fences tolerated); ``MalformedResponse`` otherwise."""
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except ValueError:
        raise MalformedResponse("the reply is not JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise MalformedResponse('the reply has no "questions" list')
    return data["questions"]


_WORD_RE = re.compile(r"[a-z0-9]+")
_CONTEXT_RE = re.compile(
    r"\b(?:this|that|these|the|above|following|given|provided)\s+"
    r"(?:passage|excerpt|text|paragraph|table|section|document)s?\b"
)
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NOT_FIGURES_RE = re.compile(r"\b(?:form\s+)?(?:10-k|10-q|8-k)\b|\bitem\s+\d+[a-z]?\b|\bq[1-4]\b")
_STOPWORDS = frozenset(
    "the a an and or of to in for on by with from at as is are was were be its their it this that which "
    "million billion percent company year fiscal".split()
)


def normalise_for_match(text: str) -> str:
    """Case-folded text with quotes and dashes unified, table pipes dropped and spaces collapsed."""
    text = text.casefold().translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "|": " "})
    )
    return re.sub(r"\s+", " ", text).strip()


def numbers(text: str) -> set[str]:
    """Numbers in ``text`` without separators or trailing zeros (form names and item numbers ignored)."""
    out = set()
    for raw in _NUMBER_RE.findall(_NOT_FIGURES_RE.sub(" ", text.casefold())):
        value = raw.replace(",", "")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        out.add(value)
    return out


_MONTHS = {
    m: i
    for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_DATE_RES = (
    ("ymd", re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")),
    ("mdy", re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")),
    ("month_day_year", re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")),
    ("day_month_year", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH},?\s+(\d{{4}})\b")),
)


def dates_in(text: str) -> list[tuple[int, int, str]]:
    """Complete calendar dates in the case-folded ``text`` as (start, end, ISO date).

    Recognised forms: 2025-09-27, 9/27/2025 or 9/27/25 (US month first), September 27, 2025 (also
    abbreviated, with or without the comma or an ordinal suffix) and 27 September 2025. Impossible
    dates are ignored.
    """
    found = []
    for kind, pattern in _DATE_RES:
        for m in pattern.finditer(text.casefold()):
            g = m.groups()
            if kind == "ymd":
                year, month, day = int(g[0]), int(g[1]), int(g[2])
            elif kind == "mdy":
                month, day, year = int(g[0]), int(g[1]), int(g[2])
                if len(g[2]) == 2:
                    year += 2000 if year < 70 else 1900
            elif kind == "month_day_year":
                month, day, year = _MONTHS[g[0][:3]], int(g[1]), int(g[2])
            else:
                day, month, year = int(g[0]), _MONTHS[g[1][:3]], int(g[2])
            try:
                found.append((m.start(), m.end(), date(year, month, day).isoformat()))
            except ValueError:
                continue
    return sorted(found)


def verified_dates(passage: SourcePassage) -> frozenset[str]:
    """The filing's report (fiscal year-end) date and filing date, from the corpus metadata."""
    out = set()
    for value in (passage.report_date, passage.filing_date):
        try:
            out.add(date.fromisoformat(value[:10]).isoformat())
        except ValueError:
            continue
    return frozenset(out)


_MONTH_NAMES = (
    r"jan(?:uary)?",
    r"feb(?:ruary)?",
    r"mar(?:ch)?",
    r"apr(?:il)?",
    r"may",
    r"june?",
    r"july?",
    r"aug(?:ust)?",
    r"sep(?:t(?:ember)?)?",
    r"oct(?:ober)?",
    r"nov(?:ember)?",
    r"dec(?:ember)?",
)
_MONTH_TEXT = {
    i: re.compile(rf"\b{name}\.?\s+0?(\d{{1,2}})\b") for i, name in enumerate(_MONTH_NAMES, 1)
}


def date_supported(iso: str, text: str, allowed_years: set[str]) -> bool:
    """Whether a date not in the filing metadata is backed by ``text``: the same date in any
    recognised form, or the same month and day (e.g. "December 31, 2024 and 2023", or a table
    header "December 31, | 2023") with a year from ``allowed_years``."""
    found = {d for *_, d in dates_in(text)}
    if iso in found:
        return True
    year, month, day = iso.split("-")
    if year not in allowed_years:
        return False
    if any(d[5:] == iso[5:] for d in found):
        return True
    return any(
        int(m.group(1)) == int(day) for m in _MONTH_TEXT[int(month)].finditer(text.casefold())
    )


def without_dates(text: str, dates: frozenset[str]) -> str:
    """The case-folded ``text`` with every complete date in ``dates`` blanked out (other dates kept)."""
    chars = list(text.casefold())
    for start, end, iso in dates_in(text):
        if iso in dates:
            chars[start:end] = " " * (end - start)
    return "".join(chars)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.casefold())


# Standard financial-statement line items and accounting terms. A question names them as written, so
# they never count towards a copied run (validator v3); sentences around them still do.
STANDARD_TERMS: tuple[str, ...] = (
    "accounts receivable",
    "accrued interest receivable",
    "accumulated benefit obligation",
    "accumulated other comprehensive income",
    "allowance for credit losses",
    "allowance for loan losses",
    "cash and cash equivalents",
    "cash paid for income taxes, net of refunds",
    "cash paid for interest",
    "cost of revenue",
    "cost of sales",
    "critical audit matter",
    "deferred tax assets",
    "deferred tax liabilities",
    "depreciation and amortization",
    "depreciation and depletion",
    "depreciation, depletion and amortization",
    "diluted earnings per share",
    "earnings per share",
    "effective tax rate",
    "fair value of plan assets",
    "finance lease liabilities",
    "free cash flow",
    "income from continuing operations",
    "net cash provided by operating activities",
    "net cash used in financing activities",
    "net cash used in investing activities",
    "net income attributable to",
    "operating lease liabilities",
    "projected benefit obligation",
    "property, plant and equipment",
    "proved oil and natural gas reserves",
    "research and development",
    "retained earnings",
    "right-of-use assets",
    "selling, general and administrative",
    "share-based compensation",
    "statutory income tax rate",
    "stock-based compensation",
    "total operating expenses",
    "unrecognized tax benefits",
)
_TERM_GRAMS = frozenset(tuple(_WORD_RE.findall(t)) for t in STANDARD_TERMS)


def _copies(
    question: str, passage_grams: set[tuple[str, ...]], n: int, exempt: set[tuple[str, ...]]
) -> bool:
    """Whether the question copies ``n`` consecutive passage words, not counting ``exempt`` phrases."""
    q = _words(question)
    masked = [False] * len(q)
    for i in range(len(q)):
        for k in range(min(10, len(q) - i), 1, -1):  # the longest exempt phrase starting here
            if tuple(q[i : i + k]) in exempt:
                masked[i : i + k] = [True] * k
                break
    return any(
        not any(masked[i : i + n]) and tuple(q[i : i + n]) in passage_grams
        for i in range(len(q) - n + 1)
    )


@dataclass(frozen=True)
class CompanyAliases:
    """Names that identify each corpus company in a question (lower case)."""

    by_ticker: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_companies(
        cls, companies: Mapping[str, str], extra: Mapping[str, Sequence[str]] | None = None
    ) -> CompanyAliases:
        out = {}
        for ticker, name in companies.items():
            names = {ticker.casefold(), company_short(name).casefold()}
            names.update(a.casefold() for a in (extra or {}).get(ticker, ()))
            out[ticker] = tuple(sorted(names))
        return cls(out)

    def mentions(self, ticker: str, text: str) -> bool:
        lowered = text.casefold()
        return any(
            re.search(rf"(?<![a-z0-9]){re.escape(a)}", lowered)
            for a in self.by_ticker.get(ticker, ())
        )


@dataclass(frozen=True)
class Candidate:
    question: str
    type: str
    answer: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {
            "question": self.question,
            "type": self.type,
            "answer": self.answer,
            "evidence": self.evidence,
        }


def check_candidate(
    raw: Any, passage: SourcePassage, settings: GenerationSettings, aliases: CompanyAliases
) -> tuple[Candidate | None, str | None]:
    """The accepted candidate, or the first rejection reason (see the module docstring)."""
    if not isinstance(raw, dict) or any(
        not isinstance(raw.get(f), str) or not raw[f].strip() for f in CANDIDATE_FIELDS
    ):
        return None, "malformed"
    candidate = Candidate(*(raw[f].strip() for f in CANDIDATE_FIELDS))
    if candidate.type not in QUESTION_TYPES:
        return None, "malformed"
    question = candidate.question
    count = len(_words(question))
    if (
        not question.endswith("?")
        or "\n" in question
        or not settings.min_question_words <= count <= settings.max_question_words
    ):
        return None, "question_format"
    if _CONTEXT_RE.search(question.casefold()):
        return None, "refers_to_context"
    if not aliases.mentions(passage.ticker, question):
        return None, "missing_company"
    if any(t != passage.ticker and aliases.mentions(t, question) for t in aliases.by_ticker):
        return None, "other_company"
    visible_words = _words(passage.visible_text)
    n = settings.max_copied_words
    if _copies(
        question,
        {tuple(visible_words[i : i + n]) for i in range(len(visible_words) - n + 1)},
        n,
        table_label_grams(passage.visible_text) | _TERM_GRAMS,
    ):
        return None, "copied_from_passage"
    if not settings.min_evidence_chars <= len(candidate.evidence) <= settings.max_evidence_chars:
        return None, "evidence_length"
    if locate_quote(candidate.evidence, passage.visible_text) is None:
        return None, "evidence_not_in_visible_text"
    allowed = numbers(passage.visible_text) | {
        passage.fiscal_year,
        passage.report_date[:4],
        passage.filing_date[:4],
    }
    # A complete report or filing date in the question is verified metadata, not a figure.
    verified = verified_dates(passage)
    question_numbers = numbers(without_dates(question, verified))
    if not question_numbers | numbers(candidate.answer) <= allowed:
        return None, "number_not_in_passage"
    for text in (question, candidate.answer):
        for *_, iso in dates_in(text):
            if iso not in verified and not date_supported(iso, passage.visible_text, allowed):
                return None, "date_not_in_passage"
    answer_numbers = numbers(candidate.answer)
    evidence = normalise_for_match(candidate.evidence)
    if answer_numbers:
        if not answer_numbers <= numbers(candidate.evidence):
            return None, "answer_not_supported"
    else:
        content = [w for w in _words(candidate.answer) if len(w) > 2 and w not in _STOPWORDS]
        if content and not any(w in evidence for w in content):
            return None, "answer_not_supported"
    return candidate, None


def question_key(question: str) -> str:
    return " ".join(_words(question))


def _near_duplicate(a: str, b: str, threshold: float) -> bool:
    x, y = set(_words(a)), set(_words(b))
    return bool(x | y) and len(x & y) / len(x | y) >= threshold


def check_reply(
    candidates: Sequence[Any],
    passage: SourcePassage,
    settings: GenerationSettings,
    aliases: CompanyAliases,
    n: int,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Accepted candidates (at most ``n``, no near-duplicates) and rejections with reasons."""
    accepted: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        candidate, reason = check_candidate(raw, passage, settings, aliases)
        if candidate is not None and any(
            _near_duplicate(candidate.question, a.question, settings.near_duplicate_threshold)
            for a in accepted
        ):
            candidate, reason = None, "duplicate_in_passage"
        if candidate is not None and len(accepted) >= n:
            candidate, reason = None, "excess_question"
        if candidate is None:
            rejected.append({"index": index, **proposed_fields(raw), "reason": reason})
        else:
            accepted.append(candidate)
    return accepted, rejected


def proposed_fields(raw: Any) -> dict[str, str | None]:
    """A proposed candidate's four fields exactly as the model gave them (anything else as None).

    Checking these gives the same result as checking the original item, so records can be
    revalidated locally without the reply.
    """
    item = raw if isinstance(raw, dict) else {}
    return {f: item[f] if isinstance(item.get(f), str) else None for f in CANDIDATE_FIELDS}


def sample_sources(passages: Sequence[SourcePassage], n: int, seed: int) -> list[SourcePassage]:
    """``n`` passages spread over companies, for a pilot.

    One passage per company in turn (tickers in sorted order); each company's passages are ordered
    by SHA-256 of ``f"{seed}:{passage_id}"``. Deterministic, so a resumed pilot selects the same
    passages, and a later full run skips the ones already done.
    """
    by_ticker: dict[str, list[SourcePassage]] = {}
    for passage in passages:
        by_ticker.setdefault(passage.ticker, []).append(passage)
    queues = {
        ticker: sorted(
            group, key=lambda p: hashlib.sha256(f"{seed}:{p.passage_id}".encode()).hexdigest()
        )
        for ticker, group in sorted(by_ticker.items())
    }
    chosen: list[SourcePassage] = []
    while len(chosen) < n and any(queues.values()):
        for queue in queues.values():
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen


# --- resumable generation --------------------------------------------------------------------------


def records_path(directory: Path, split: str) -> Path:
    return directory / f"{split}.records.jsonl"


def run_path(directory: Path, split: str) -> Path:
    return directory / f"{split}.run.json"


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every record line; a corrupt line stops (records are append-only and fsynced)."""
    if not path.exists():
        return []
    out = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            out.append(json.loads(line))
        except ValueError:
            raise GenerationError(
                f"{path}:{line_no} is not a complete record; inspect the file before resuming"
            ) from None
    return out


def generation_run_info(
    settings: GenerationSettings, split: str, prepared_manifest_sha256: str
) -> dict[str, Any]:
    """What a split's generation run is bound to: prompt, model, parameters and preparation."""
    return {
        "format_version": 1,
        "split": split,
        "prompt_version": settings.prompt_version,
        "prompt_sha256": prompt_sha256(settings.prompt_version),
        "provider": settings.provider,
        "model": settings.model,
        "params": settings.params(),
        "params_sha256": settings.params_sha256(),
        "queries_per_passage": settings.queries_per_passage[split],  # type: ignore[index]
        "prepared_manifest_sha256": prepared_manifest_sha256,
    }


def continuations_path(directory: Path, split: str) -> Path:
    return directory / f"{split}.run.continuations.jsonl"


def run_bindings(directory: str | Path, split: str) -> list[dict[str, Any]]:
    """The split's run (``run.json``) followed by its continuations, oldest first; empty before the first run."""
    directory = Path(directory)
    path = run_path(directory, split)
    if not path.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8")),
        *read_records(continuations_path(directory, split)),
    ]


@dataclass(frozen=True)
class RunCheck:
    """How settings relate to a split's recorded run; see ``check_run``."""

    state: Literal["new", "bound", "continuable"]
    bindings: list[dict[str, Any]]
    changed: dict[
        str, dict[str, Any]
    ]  # request settings that differ from the latest binding: {"from", "to"}


def _changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for key in sorted(set(old) | set(new)):
        if key in old and key in new and old[key] == new[key]:
            continue
        out[key] = {
            **({"from": old[key]} if key in old else {}),
            **({"to": new[key]} if key in new else {}),
        }
    return out


def describe_changes(changed: Mapping[str, Mapping[str, Any]]) -> str:
    """For example ``max_completion_tokens 2048 -> 768, output_tokens_per_minute unset -> 1000``."""

    def side(change: Mapping[str, Any], key: str) -> str:
        return json.dumps(change[key]) if key in change else "unset"

    return ", ".join(f"{k} {side(c, 'from')} -> {side(c, 'to')}" for k, c in changed.items())


def _identity(info: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in info.items() if k not in ("params", "params_sha256")}


def _verify_continuations(
    directory: Path, split: str, bindings: Sequence[Mapping[str, Any]]
) -> None:
    run_sha256 = hashlib.sha256(run_path(directory, split).read_bytes()).hexdigest()
    records = records_path(directory, split)
    lines = records.read_bytes().splitlines(keepends=True) if records.exists() else []
    for previous, entry in zip(bindings, bindings[1:], strict=False):
        n = entry.get("records_before")
        intact = (
            entry.get("of_run_sha256") == run_sha256
            and entry.get("previous_params_sha256") == previous.get("params_sha256")
            and entry.get("params_sha256") == _params_sha256(entry.get("params", {}))
            and isinstance(n, int)
            and len(lines) >= n
            and hashlib.sha256(b"".join(lines[:n])).hexdigest()
            == entry.get("records_sha256_before")
        )
        if not intact:
            raise GenerationError(
                f"{continuations_path(directory, split)} does not match the run or its records (continuation "
                f"{entry.get('continuation')}); inspect the files before resuming"
            )


def check_run(
    directory: str | Path,
    split: str,
    settings: GenerationSettings,
    prepared_manifest_sha256: str,
) -> RunCheck:
    """Whether ``settings`` may generate into ``directory`` for ``split``.

    ``new``: nothing is recorded yet. ``bound``: the settings are the latest binding (the run or its last
    continuation). ``continuable``: only request settings (``REQUEST_PARAMS``) differ, so the run may go on once a
    continuation is recorded (``record_continuation``). Any other difference (prompt, model, sampling or
    validation settings, preparation) is a different run and raises. Each continuation must name the run, the
    binding before it and its own parameters, and the records it counted must still begin the records file.
    """
    directory = Path(directory)
    bindings = run_bindings(directory, split)
    if not bindings:
        return RunCheck("new", [], {})
    info = generation_run_info(settings, split, prepared_manifest_sha256)
    path = run_path(directory, split)
    if _identity(bindings[0]) != _identity(info):
        raise GenerationError(
            f"{path} records a different run (prompt, model, parameters or preparation); generate into a new "
            "directory instead of mixing runs"
        )
    _verify_continuations(directory, split, bindings)
    changed = _changes(bindings[-1]["params"], info["params"])
    if not changed:
        return RunCheck("bound", bindings, {})
    fixed = [k for k in changed if k not in REQUEST_PARAMS]
    if fixed:
        raise GenerationError(
            f"{path} records a different run (prompt, model, parameters or preparation: {', '.join(fixed)} "
            "changed); generate into a new directory instead of mixing runs"
        )
    return RunCheck("continuable", bindings, changed)


def record_continuation(
    directory: str | Path,
    split: str,
    settings: GenerationSettings,
    prepared_manifest_sha256: str,
    check: RunCheck,
    *,
    clock: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    """Append the continuation that binds the run to ``settings`` from now on; nothing earlier is rewritten.

    It records the changed request settings and the number and SHA-256 of the records so far, so a later
    check can confirm that those records were only ever appended to.
    """
    if check.state != "continuable":
        raise GenerationError(f"a {check.state} run takes no continuation")
    directory = Path(directory)
    info = generation_run_info(settings, split, prepared_manifest_sha256)
    records = records_path(directory, split)
    data = records.read_bytes() if records.exists() else b""
    entry = {
        "format_version": 1,
        "kind": "continuation",
        "split": split,
        "continuation": len(check.bindings),
        "of_run_sha256": hashlib.sha256(run_path(directory, split).read_bytes()).hexdigest(),
        "previous_params_sha256": check.bindings[-1]["params_sha256"],
        "params": info["params"],
        "params_sha256": info["params_sha256"],
        "changed": check.changed,
        "records_before": len(data.splitlines()),
        "records_sha256_before": hashlib.sha256(data).hexdigest(),
        "created_at": clock(),
    }
    with continuations_path(directory, split).open("a", encoding="utf-8") as out:
        _append(out, entry)
    return entry


def latest_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Each passage's outcome: the latest done record (a revalidation revises an earlier one);
    otherwise the latest failure."""
    out: dict[str, Mapping[str, Any]] = {}
    for record in records:
        pid = record["passage_id"]
        if record.get("status") == "done" or out.get(pid, {}).get("status") != "done":
            out[pid] = record
    return out


def record_sha256(record: Mapping[str, Any]) -> str:
    """SHA-256 of a record as it is written (one JSON line, keys sorted)."""
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _append(out: Any, record: Mapping[str, Any], secret: str | None = None) -> None:
    out.write(redact(json.dumps(record, ensure_ascii=False, sort_keys=True), secret) + "\n")
    out.flush()
    os.fsync(out.fileno())


class QueryGenerator:
    """Runs the client over approved passages of one split, resumably; see the module docstring."""

    def __init__(
        self,
        client: ChatClient,
        settings: GenerationSettings,
        aliases: CompanyAliases,
        *,
        directory: str | Path,
        prepared_manifest_sha256: str,
        clock: Callable[[], str] = utc_now,
        secret: str | None = None,
        continue_run: bool = False,
    ) -> None:
        self.client = client
        self.settings = settings
        self.aliases = aliases
        self.directory = Path(directory)
        self.prepared_manifest_sha256 = prepared_manifest_sha256
        self._clock = clock
        self._secret = secret
        self.continue_run = continue_run  # record a continuation if only request settings changed
        self.continuation: dict[str, Any] | None = None  # the one this run recorded

    def run_info(self, split: str) -> dict[str, Any]:
        return generation_run_info(self.settings, split, self.prepared_manifest_sha256)

    def _start(self, split: str) -> None:
        check = check_run(self.directory, split, self.settings, self.prepared_manifest_sha256)
        if check.state == "bound":
            return
        if check.state == "continuable":
            if not self.continue_run:
                raise GenerationError(
                    f"only request settings differ from the run recorded in {self.directory} "
                    f"({describe_changes(check.changed)}); continue it with continue_run (generate --continue-run), "
                    "which records a continuation and keeps every record, or restore the recorded settings"
                )
            self.continuation = record_continuation(
                self.directory,
                split,
                self.settings,
                self.prepared_manifest_sha256,
                check,
                clock=self._clock,
            )
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        run_path(self.directory, split).write_text(
            json.dumps(self.run_info(split), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def run(
        self,
        split: Literal["train", "validation"],
        passages: Sequence[SourcePassage],
        *,
        limit: int | None = None,
        retry_failed: bool = False,
    ) -> dict[str, int]:
        """Generate for passages not done yet (failed ones only with ``retry_failed``); returns counts."""
        if any(p.split != split for p in passages):
            raise GenerationError(
                f"every passage given for {split} must belong to the {split} split"
            )
        self._start(split)
        path = records_path(self.directory, split)
        state = latest_records(read_records(path))
        n = self.settings.queries_per_passage[split]
        counts = {
            "skipped_done": 0,
            "skipped_failed": 0,
            "done": 0,
            "failed": 0,
            "stopped_by_budget": 0,
            "stopped_by_rate_limit": 0,
        }
        consecutive = 0
        attempted = 0
        with path.open("a", encoding="utf-8") as out:
            for passage in passages:
                previous = state.get(passage.passage_id, {}).get("status")
                if previous == "done":
                    counts["skipped_done"] += 1
                    continue
                if previous == "failed" and not retry_failed:
                    counts["skipped_failed"] += 1
                    continue
                if limit is not None and attempted >= limit:
                    break
                attempted += 1
                # A clean stop records nothing for a passage that has spent nothing; a rerun resumes it.
                try:
                    record = self._generate_one(passage, n)
                except RequestBudgetExhausted:
                    counts["stopped_by_budget"] = 1
                    break
                except RateLimited:
                    counts["stopped_by_rate_limit"] = 1
                    break
                _append(out, record, self._secret)
                counts[record["status"]] += 1
                if record.get("budget_exhausted"):
                    counts["stopped_by_budget"] = 1
                    break
                if record.get("rate_limited"):
                    counts["stopped_by_rate_limit"] = 1
                    break
                consecutive = consecutive + 1 if record.get("request_failed") else 0
                if consecutive >= self.settings.max_consecutive_failures:
                    raise GenerationError(
                        f"stopping after {consecutive} consecutive request failures (last: {record['error']}); "
                        "rerun later to resume"
                    )
        return counts

    def _usage_since(self, before: Counter) -> Counter:
        usage = Counter(getattr(self.client, "usage", {}))
        usage.subtract(before)
        return usage

    def _generate_one(self, passage: SourcePassage, n: int) -> dict[str, Any]:
        messages = build_messages(passage, n, self.settings.prompt_version)
        base = {
            "passage_id": passage.passage_id,
            "split": passage.split,
            "generated_at": self._clock(),
            "record_format": RECORD_FORMAT_VERSION,
            "model": self.settings.model,
            "prompt_version": self.settings.prompt_version,
            "validator_version": VALIDATOR_VERSION,
            "params_sha256": self.settings.params_sha256(),
        }
        error, request_failed, status, code = None, False, None, None
        stopped: GenerationError | None = None
        usage_before = Counter(getattr(self.client, "usage", {}))
        for attempt in range(1, self.settings.max_response_attempts + 1):
            try:
                content = self.client.complete(messages)
                candidates = parse_candidates(content)
            except (RequestBudgetExhausted, RateLimited) as exc:
                if error is None:
                    raise  # nothing spent on this passage yet: stop without recording it
                # An earlier reply was unusable; keep that diagnosis rather than discard it.
                stopped, attempt = exc, attempt - 1
                break
            except (AuthError, ModelUnavailable, RequestTooLarge):
                raise  # run-level: stop without recording this passage
            except MalformedResponse as exc:
                error = redact(str(exc), self._secret)
                logger.warning("unusable reply for %s: %s", passage.passage_id, error)
                continue
            except (TransientError, RequestRejected, ResponseTruncated) as exc:
                # Either way no usable reply came back; repeated failures trip the breaker in ``run``.
                # A truncated reply is not retried: the same token limit would usually cut it again.
                error, request_failed = redact(str(exc), self._secret), True
                status, code = exc.status, exc.code
                break
            accepted, rejected = check_reply(candidates, passage, self.settings, self.aliases, n)
            return {
                **base,
                "usage": {k: v for k, v in sorted(self._usage_since(usage_before).items()) if v},
                "status": "done",
                "attempts": attempt,
                "error": None,
                "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "proposed": len(candidates),
                # The parsed candidates only: no reasoning, no raw reply, no provider payload.
                "candidates": [proposed_fields(raw) for raw in candidates],
                "accepted": [c.to_dict() for c in accepted],
                "rejected": rejected,
            }
        return {
            **base,
            "status": "failed",
            "attempts": attempt,
            "error": error,
            "request_failed": request_failed,
            "http_status": status,
            "error_code": code,
            "budget_exhausted": isinstance(stopped, RequestBudgetExhausted),
            "rate_limited": isinstance(stopped, RateLimited),
            "usage": {k: v for k, v in sorted(self._usage_since(usage_before).items()) if v},
            "proposed": 0,
            "candidates": [],
            "accepted": [],
            "rejected": [],
        }


def revalidate_records(
    directory: str | Path,
    split: Literal["train", "validation"],
    sources: Mapping[str, SourcePassage],
    settings: GenerationSettings,
    aliases: CompanyAliases,
    *,
    prepared_manifest_sha256: str,
    clock: Callable[[], str] = utc_now,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-run the current candidate checks on recorded candidates; no request is sent.

    Each passage's latest done record checked by an older validator is re-checked from its stored
    candidates, and a revision record is appended (an unchanged outcome too, so the record shows
    which validator passed it). Earlier records are never rewritten. Records from before format 2
    keep only the rejected questions, not the answers or evidence, and cannot be revalidated; they
    are listed, not guessed at. The settings must match the recorded run (``check_run``); request
    settings may differ, since the checks do not use them.
    """
    directory = Path(directory)
    path = records_path(directory, split)
    counts: dict[str, Any] = {
        "up_to_date": 0,
        "revalidated": 0,
        "changed": 0,
        "failed_records": 0,
        "not_revalidatable": [],
    }
    if check_run(directory, split, settings, prepared_manifest_sha256).state == "new":
        return counts
    n = settings.queries_per_passage[split]
    revisions = []
    for pid, record in latest_records(read_records(path)).items():
        if record["status"] != "done":
            counts["failed_records"] += 1
            continue
        if record.get("validator_version") == VALIDATOR_VERSION:
            counts["up_to_date"] += 1
            continue
        if "candidates" not in record:
            counts["not_revalidatable"].append(pid)
            continue
        if pid not in sources:
            raise GenerationError(f"record for {pid}, which is not an approved {split} passage")
        accepted, rejected = check_reply(record["candidates"], sources[pid], settings, aliases, n)
        new = {"accepted": [c.to_dict() for c in accepted], "rejected": rejected}
        changed = new != {"accepted": record["accepted"], "rejected": record["rejected"]}
        revisions.append(
            {
                **record,
                **new,
                "validator_version": VALIDATOR_VERSION,
                "usage": {},  # no request: the tokens belong to the original record
                "revision": {
                    "kind": "revalidation",
                    "of_record_sha256": record_sha256(record),
                    "previous_validator_version": record.get(
                        "validator_version", LEGACY_VALIDATOR_VERSION
                    ),
                    "changed": changed,
                    "revised_at": clock(),
                },
            }
        )
        counts["revalidated"] += 1
        counts["changed"] += changed
    if revisions and not dry_run:
        with path.open("a", encoding="utf-8") as out:
            for revision in revisions:
                _append(out, revision)
    return counts


__all__ = [
    "LATEST_PROMPT_VERSION",
    "PROMPTS",
    "PROMPT_VERSION",
    "PROMPT_VERSIONS",
    "QUESTION_TYPES",
    "V1_QUESTION_TYPES",
    "V2_QUESTION_TYPES",
    "Prompt",
    "REJECTION_REASONS",
    "STANDARD_TERMS",
    "CANDIDATE_FIELDS",
    "LEGACY_VALIDATOR_VERSION",
    "OUTPUT_WINDOW_SECONDS",
    "PACING_MARGIN_SECONDS",
    "RECORD_FORMAT_VERSION",
    "REQUEST_PARAMS",
    "VALIDATOR_VERSION",
    "AuthError",
    "Candidate",
    "ChatClient",
    "CompanyAliases",
    "CredentialError",
    "GenerationError",
    "GenerationSettings",
    "GroqClient",
    "MODEL_ERROR_CODES",
    "MalformedResponse",
    "ModelUnavailable",
    "QueryGenerator",
    "RateLimited",
    "RequestBudgetExhausted",
    "RequestRejected",
    "RequestTooLarge",
    "ResponseTruncated",
    "RunCheck",
    "SourcePassage",
    "TransientError",
    "TransportResponse",
    "api_key_from_env",
    "build_messages",
    "check_candidate",
    "check_reply",
    "check_run",
    "continuations_path",
    "date_supported",
    "dates_in",
    "describe_changes",
    "generation_run_info",
    "latest_records",
    "normalise_for_match",
    "numbers",
    "parse_candidates",
    "parse_duration",
    "proposed_fields",
    "prompt_sha256",
    "question_key",
    "read_records",
    "record_continuation",
    "record_sha256",
    "records_path",
    "redact",
    "retry_after_seconds",
    "revalidate_records",
    "run_bindings",
    "run_path",
    "sample_sources",
    "verified_dates",
    "without_dates",
]
