"""Gemini via the official google-genai SDK.

Field names checked against google-genai 3.6.1: GenerateContentConfig(system_instruction,
temperature, response_mime_type, response_json_schema); usage_metadata.prompt_token_count,
.candidates_token_count, .thoughts_token_count; errors.APIError(code, status, details).
The SDK does not retry by default (no HttpRetryOptions), so retries happen only here.
"""

import re
import time
from collections.abc import Callable
from typing import Any, Protocol

from google import genai
from google.genai import errors, types
from pydantic import SecretStr

from cidoc.llm.base import (
    LLMError,
    RateLimitedError,
    ServerUnavailableError,
    StructuredResponse,
    Usage,
)
from cidoc.llm.ratelimit import RateLimiter, call_with_backoff

_RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)s$")


class _Models(Protocol):
    def generate_content(
        self, *, model: str, contents: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse: ...


class GeminiClient(Protocol):
    """The slice of genai.Client used here; tests pass a fake."""

    @property
    def models(self) -> _Models: ...


def retry_delay(details: Any) -> float | None:
    """Seconds from a google.rpc.RetryInfo in an API error body, if present."""
    error = details.get("error", details) if isinstance(details, dict) else None
    if not isinstance(error, dict):
        return None
    for item in error.get("details") or []:
        if isinstance(item, dict) and item.get("@type") == _RETRY_INFO:
            match = _DURATION.match(str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


# Retried with backoff. Anything else (400 INVALID_ARGUMENT, 403, 404, ...) is the
# request's fault and fails at once.
_RATE_LIMIT_CODES = frozenset({429})
_RATE_LIMIT_STATUSES = frozenset({"RESOURCE_EXHAUSTED"})
_SERVER_CODES = frozenset({500, 502, 503, 504})
_SERVER_STATUSES = frozenset({"UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"})


def _is_rate_limit(e: errors.APIError) -> bool:
    return e.code in _RATE_LIMIT_CODES or e.status in _RATE_LIMIT_STATUSES


def _is_server_error(e: errors.APIError) -> bool:
    return e.code in _SERVER_CODES or e.status in _SERVER_STATUSES


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


class GeminiProvider:
    def __init__(
        self,
        model: str,
        rpm: int,
        api_key: SecretStr | None = None,
        client: GeminiClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if client is None:
            if api_key is None:
                raise ValueError("GeminiProvider needs an api_key or a client")
            client = genai.Client(api_key=api_key.get_secret_value())
        self._model = model
        self._client = client
        self._clock = clock
        self._sleep = sleep
        self._limiter = RateLimiter(rpm, clock=clock, sleep=sleep)

    @property
    def model(self) -> str:
        return self._model

    def generate_structured(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> StructuredResponse:
        # No tools: the model can only answer, never act.
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=schema,
        )

        def request() -> tuple[types.GenerateContentResponse, float]:
            start = self._clock()
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=user, config=config
                )
            except errors.APIError as e:
                message = f"Gemini API error {e.code} {e.status}: {e.message}"
                if _is_rate_limit(e):
                    raise RateLimitedError(message, retry_delay(e.details)) from e
                if _is_server_error(e):
                    raise ServerUnavailableError(message, retry_delay(e.details)) from e
                raise LLMError(message) from e
            return response, self._clock() - start

        (response, latency), attempts = call_with_backoff(request, self._limiter, self._sleep)
        return to_structured(response, latency, attempts)


def to_structured(
    response: types.GenerateContentResponse, latency_s: float, attempts: int
) -> StructuredResponse:
    candidate = response.candidates[0] if response.candidates else None
    parts = candidate.content.parts if candidate and candidate.content else None
    text = "".join(p.text for p in parts or [] if p.text and not p.thought)
    finish = _enum_value(candidate.finish_reason) if candidate else None
    feedback = response.prompt_feedback
    block = _enum_value(feedback.block_reason) if feedback else None
    meta = response.usage_metadata
    usage = Usage(
        input_tokens=(meta.prompt_token_count or 0) if meta else 0,
        output_tokens=((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
        if meta
        else 0,
        latency_s=latency_s,
        attempts=attempts,
    )
    return StructuredResponse(
        text=text,
        complete=finish == "STOP" and block is None,
        finish_reason=finish,
        block_reason=block,
        usage=usage,
    )
