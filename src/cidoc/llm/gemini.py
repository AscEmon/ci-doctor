"""Gemini via the official google-genai SDK.

Field names checked against google-genai 3.6.1: GenerateContentConfig(system_instruction,
temperature, response_mime_type, response_json_schema); usage_metadata.prompt_token_count,
.candidates_token_count, .thoughts_token_count; errors.APIError(code, status, details).
The SDK does not retry by default (no HttpRetryOptions), so retries happen only here.

Function calling, from the same SDK's types.py and its own automatic-function-calling
loop (models.py, _extra_utils.py): GenerateContentConfig(tools=[Tool(
function_declarations=[FunctionDeclaration(name, description, parameters_json_schema)])],
tool_config=ToolConfig(function_calling_config=FunctionCallingConfig(mode,
allowed_function_names)), automatic_function_calling=AutomaticFunctionCallingConfig(
disable=True)). Mode ANY with allowed_function_names forces a call to one of them; AUTO
lets the model choose. The model turn goes back as the candidate's Content unchanged
(it carries thought signatures), and the results as a role="user" Content of
Part(function_response=FunctionResponse(id, name, response)). We pass declarations, not
Python callables, so the SDK never runs anything itself; AFC is disabled anyway.
"""

import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from google import genai
from google.genai import errors, types
from pydantic import SecretStr

from cidoc.llm.base import (
    FunctionCall,
    FunctionResultsMessage,
    FunctionSpec,
    LLMError,
    Message,
    ModelMessage,
    RateLimitedError,
    ServerUnavailableError,
    StructuredResponse,
    ToolCallResponse,
    Usage,
    UserMessage,
)
from cidoc.llm.ratelimit import RateLimiter, call_with_backoff

_RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)s$")


class _Models(Protocol):
    def generate_content(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
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
        response, latency, attempts = self._generate(user, config)
        return to_structured(response, latency, attempts)

    def generate_with_tools(
        self,
        system: str,
        messages: Sequence[Message],
        functions: Sequence[FunctionSpec],
        force: Sequence[str] | None = None,
    ) -> ToolCallResponse:
        if force:
            calling = types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=list(force)
            )
        else:
            calling = types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=f.name,
                            description=f.description,
                            parameters_json_schema=f.parameters,
                        )
                        for f in functions
                    ]
                )
            ],
            tool_config=types.ToolConfig(function_calling_config=calling),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response, latency, attempts = self._generate(to_contents(messages), config)
        return to_tool_call_response(response, latency, attempts)

    def _generate(
        self, contents: Any, config: types.GenerateContentConfig
    ) -> tuple[types.GenerateContentResponse, float, int]:
        """One request within the RPM limit, with backoff on 429 and 5xx."""

        def request() -> tuple[types.GenerateContentResponse, float]:
            start = self._clock()
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
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
        return response, latency, attempts


def _usage(response: types.GenerateContentResponse, latency_s: float, attempts: int) -> Usage:
    meta = response.usage_metadata
    return Usage(
        input_tokens=(meta.prompt_token_count or 0) if meta else 0,
        output_tokens=((meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
        if meta
        else 0,
        latency_s=latency_s,
        attempts=attempts,
    )


def _outcome(
    response: types.GenerateContentResponse,
) -> tuple[types.Candidate | None, str | None, str | None]:
    """(first candidate, finish reason, block reason)."""
    candidate = response.candidates[0] if response.candidates else None
    finish = _enum_value(candidate.finish_reason) if candidate else None
    feedback = response.prompt_feedback
    block = _enum_value(feedback.block_reason) if feedback else None
    return candidate, finish, block


def _answer_text(parts: list[types.Part] | None) -> str:
    return "".join(p.text for p in parts or [] if p.text and not p.thought)


def to_structured(
    response: types.GenerateContentResponse, latency_s: float, attempts: int
) -> StructuredResponse:
    candidate, finish, block = _outcome(response)
    parts = candidate.content.parts if candidate and candidate.content else None
    return StructuredResponse(
        text=_answer_text(parts),
        complete=finish == "STOP" and block is None,
        finish_reason=finish,
        block_reason=block,
        usage=_usage(response, latency_s, attempts),
    )


def to_contents(messages: Sequence[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        if isinstance(message, UserMessage):
            contents.append(types.Content(role="user", parts=[types.Part(text=message.text)]))
        elif isinstance(message, ModelMessage):
            if isinstance(message.raw, types.Content):
                if message.raw.parts:  # an empty turn (nothing generated) is left out
                    contents.append(message.raw)  # verbatim: keeps thought signatures
                continue
            parts = [types.Part(text=message.text)] if message.text else []
            parts += [
                types.Part(
                    function_call=types.FunctionCall(id=call.id, name=call.name, args=call.args)
                )
                for call in message.function_calls
            ]
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif isinstance(message, FunctionResultsMessage):
            parts = [
                types.Part(
                    function_response=types.FunctionResponse(
                        id=result.call.id, name=result.call.name, response=result.response
                    )
                )
                for result in message.results
            ]
            contents.append(types.Content(role="user", parts=parts))
    return contents


def to_tool_call_response(
    response: types.GenerateContentResponse, latency_s: float, attempts: int
) -> ToolCallResponse:
    candidate, finish, block = _outcome(response)
    content = candidate.content if candidate and candidate.content else None
    parts = content.parts if content else None
    calls = [
        FunctionCall(
            name=p.function_call.name or "",
            args=dict(p.function_call.args or {}),
            id=p.function_call.id,
        )
        for p in parts or []
        if p.function_call is not None
    ]
    message = ModelMessage(
        text=_answer_text(parts),
        function_calls=calls,
        raw=content,
    )
    return ToolCallResponse(
        message=message,
        complete=finish == "STOP" and block is None,
        finish_reason=finish,
        block_reason=block,
        usage=_usage(response, latency_s, attempts),
    )
