"""LLM provider layer. The Gemini client is always a fake: no network in tests."""

from pathlib import Path
from typing import Any

import pytest
from google.genai import errors, types
from pydantic import SecretStr

from cidoc.config import MissingGeminiKeyError, Settings
from cidoc.llm.base import LLMError, RateLimitedError, StructuredResponse, Usage
from cidoc.llm.factory import create_provider
from cidoc.llm.gemini import GeminiProvider, retry_delay
from cidoc.llm.pricing import Price
from cidoc.llm.ratelimit import MAX_ATTEMPTS, RateLimiter, call_with_backoff


@pytest.fixture(autouse=True)
def no_real_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read the developer's .env or real keys from the environment."""
    monkeypatch.chdir(tmp_path)
    for name in ("GEMINI_API_KEY", "CIDOC_MODEL", "CIDOC_RPM", "CIDOC_PRICES"):
        monkeypatch.delenv(name, raising=False)


RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"
SCHEMA: dict[str, Any] = {"type": "object", "properties": {"category": {"type": "string"}}}


class FakeTime:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def response(
    text: str = '{"category": "other"}',
    finish: types.FinishReason | None = types.FinishReason.STOP,
    block: types.BlockedReason | None = None,
    prompt_tokens: int = 100,
    output_tokens: int = 20,
    thought_tokens: int | None = None,
) -> types.GenerateContentResponse:
    candidates = None
    if finish is not None:
        candidates = [
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=finish,
            )
        ]
    return types.GenerateContentResponse(
        candidates=candidates,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block)
        if block
        else None,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thought_tokens,
        ),
    )


def rate_limit_error(delay: str | None = "7s") -> errors.ClientError:
    details: list[dict[str, Any]] = [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []}
    ]
    if delay:
        details.append({"@type": RETRY_INFO, "retryDelay": delay})
    body = {
        "error": {
            "code": 429,
            "message": "Resource has been exhausted (e.g. check quota).",
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        }
    }
    return errors.ClientError(429, body)


class FakeModels:
    def __init__(self, outcomes: list[types.GenerateContentResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self, *, model: str, contents: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes: types.GenerateContentResponse | Exception) -> None:
        self.models = FakeModels(list(outcomes))


def provider(client: FakeClient, fake: FakeTime, rpm: int = 10) -> GeminiProvider:
    return GeminiProvider(
        model="gemini-test", rpm=rpm, client=client, clock=fake.clock, sleep=fake.sleep
    )


# --- rate limiter -------------------------------------------------------------------


def test_rate_limiter_allows_rpm_requests_per_minute_then_waits() -> None:
    fake = FakeTime()
    limiter = RateLimiter(3, clock=fake.clock, sleep=fake.sleep)

    for _ in range(3):
        limiter.acquire()
    assert fake.sleeps == []

    limiter.acquire()
    assert fake.sleeps == [60.0]


def test_rate_limiter_window_slides() -> None:
    fake = FakeTime()
    limiter = RateLimiter(2, clock=fake.clock, sleep=fake.sleep)
    limiter.acquire()
    fake.now += 50
    limiter.acquire()
    fake.now += 15  # the first request is now 65s old

    limiter.acquire()

    assert fake.sleeps == []
    limiter.acquire()  # waits until the second (at +50s) leaves the window
    assert fake.sleeps == [pytest.approx(45.0)]


def test_rate_limiter_rejects_zero_rpm() -> None:
    with pytest.raises(ValueError, match="rpm"):
        RateLimiter(0)


# --- backoff ------------------------------------------------------------------------


def test_backoff_retries_rate_limits_exponentially() -> None:
    fake = FakeTime()
    limiter = RateLimiter(100, clock=fake.clock, sleep=fake.sleep)
    failures = [RateLimitedError("429"), RateLimitedError("429"), RateLimitedError("429")]

    def fn() -> str:
        if failures:
            raise failures.pop(0)
        return "ok"

    result, attempts = call_with_backoff(fn, limiter, fake.sleep)

    assert (result, attempts) == ("ok", 4)
    assert fake.sleeps == [1.0, 2.0, 4.0]


def test_backoff_honors_longer_retry_delay_from_the_api() -> None:
    fake = FakeTime()
    limiter = RateLimiter(100, clock=fake.clock, sleep=fake.sleep)
    failures = [RateLimitedError("429", retry_after=30.0)]

    def fn() -> str:
        if failures:
            raise failures.pop(0)
        return "ok"

    call_with_backoff(fn, limiter, fake.sleep)

    assert fake.sleeps == [30.0]


def test_backoff_gives_up_after_max_attempts() -> None:
    fake = FakeTime()
    limiter = RateLimiter(100, clock=fake.clock, sleep=fake.sleep)
    calls = 0

    def fn() -> str:
        nonlocal calls
        calls += 1
        raise RateLimitedError("429")

    with pytest.raises(RateLimitedError):
        call_with_backoff(fn, limiter, fake.sleep)
    assert calls == MAX_ATTEMPTS == 5


def test_backoff_does_not_retry_other_errors() -> None:
    fake = FakeTime()
    limiter = RateLimiter(100, clock=fake.clock, sleep=fake.sleep)
    calls = 0

    def fn() -> str:
        nonlocal calls
        calls += 1
        raise LLMError("400 bad request")

    with pytest.raises(LLMError):
        call_with_backoff(fn, limiter, fake.sleep)
    assert calls == 1
    assert fake.sleeps == []


# --- Gemini provider ----------------------------------------------------------------


def test_gemini_sends_json_mode_config_at_temperature_zero() -> None:
    client = FakeClient(response())

    provider(client, FakeTime()).generate_structured("SYSTEM", "USER", SCHEMA)

    call = client.models.calls[0]
    config = call["config"]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "USER"
    assert config.system_instruction == "SYSTEM"
    assert config.temperature == 0.0
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SCHEMA
    assert config.tools is None


def test_gemini_reads_text_finish_reason_and_usage() -> None:
    client = FakeClient(response(prompt_tokens=1200, output_tokens=80, thought_tokens=40))

    result = provider(client, FakeTime()).generate_structured("s", "u", SCHEMA)

    assert result.json() == {"category": "other"}
    assert result.complete
    assert result.finish_reason == "STOP"
    assert result.block_reason is None
    assert result.usage.input_tokens == 1200
    assert result.usage.output_tokens == 120  # thinking is billed as output
    assert result.usage.attempts == 1


@pytest.mark.parametrize(
    ("resp", "finish", "block"),
    [
        (response(finish=types.FinishReason.MAX_TOKENS), "MAX_TOKENS", None),
        (response(finish=types.FinishReason.SAFETY, text=""), "SAFETY", None),
        (
            response(finish=None, block=types.BlockedReason.PROHIBITED_CONTENT),
            None,
            "PROHIBITED_CONTENT",
        ),
    ],
)
def test_gemini_marks_incomplete_and_blocked_responses(
    resp: types.GenerateContentResponse, finish: str | None, block: str | None
) -> None:
    result = provider(FakeClient(resp), FakeTime()).generate_structured("s", "u", SCHEMA)

    assert not result.complete
    assert result.finish_reason == finish
    assert result.block_reason == block


def test_gemini_backs_off_on_429_using_the_api_retry_delay() -> None:
    fake = FakeTime()
    client = FakeClient(rate_limit_error("7s"), rate_limit_error(None), response())

    result = provider(client, fake).generate_structured("s", "u", SCHEMA)

    assert result.usage.attempts == 3
    assert fake.sleeps == [7.0, 2.0]  # API delay, then exponential
    assert len(client.models.calls) == 3


def test_gemini_gives_up_on_persistent_429() -> None:
    client = FakeClient(*[rate_limit_error() for _ in range(MAX_ATTEMPTS)])

    with pytest.raises(RateLimitedError, match="RESOURCE_EXHAUSTED"):
        provider(client, FakeTime()).generate_structured("s", "u", SCHEMA)
    assert len(client.models.calls) == MAX_ATTEMPTS


def test_gemini_does_not_retry_other_api_errors() -> None:
    bad = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    client = FakeClient(bad)

    with pytest.raises(LLMError, match="400 INVALID_ARGUMENT") as exc_info:
        provider(client, FakeTime()).generate_structured("s", "u", SCHEMA)
    assert not isinstance(exc_info.value, RateLimitedError)
    assert len(client.models.calls) == 1


def test_gemini_respects_rpm_between_calls() -> None:
    fake = FakeTime()
    client = FakeClient(response(), response(), response())
    gemini = provider(client, fake, rpm=2)

    for _ in range(3):
        gemini.generate_structured("s", "u", SCHEMA)

    assert fake.sleeps == [60.0]


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ({"error": {"details": [{"@type": RETRY_INFO, "retryDelay": "12s"}]}}, 12.0),
        ({"error": {"details": [{"@type": RETRY_INFO, "retryDelay": "1.5s"}]}}, 1.5),
        ({"error": {"details": [{"@type": "other", "retryDelay": "12s"}]}}, None),
        ({"error": {"message": "no details"}}, None),
        ("not a dict", None),
        (None, None),
    ],
)
def test_retry_delay_parsing(details: Any, expected: float | None) -> None:
    assert retry_delay(details) == expected


# --- shared types and factory -------------------------------------------------------


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]"])
def test_structured_response_json_rejects_non_objects(text: str) -> None:
    result = StructuredResponse(text, True, "STOP", None, Usage(1, 1, 0.1))

    with pytest.raises(ValueError):
        result.json()


def test_price_cost() -> None:
    assert Price(input=0.1, output=0.4).cost(2_000_000, 1_000_000) == pytest.approx(0.6)


def test_create_provider_requires_api_key() -> None:
    settings = Settings(github_token=SecretStr("test-github-token-value"))

    with pytest.raises(MissingGeminiKeyError, match="GEMINI_API_KEY"):
        create_provider(settings)


def test_create_provider_uses_configured_model() -> None:
    settings = Settings(
        github_token=SecretStr("test-github-token-value"),
        gemini_api_key=SecretStr("test-gemini-key-value"),
        cidoc_model="gemini-configured",
    )

    assert create_provider(settings).model == "gemini-configured"
    assert create_provider(settings, model="gemini-override").model == "gemini-override"
