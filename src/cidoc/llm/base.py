"""Provider-agnostic LLM interface: one structured-output call, with usage, and one
function-calling call for the agent.

The analyzer only depends on LLMProvider, the agent on ToolCallingProvider; adding a
provider means one new class.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    """Includes thinking tokens: providers bill them as output."""
    latency_s: float
    """Duration of the successful request, excluding rate-limit waits."""
    attempts: int = 1


@dataclass(frozen=True)
class StructuredResponse:
    text: str
    """The raw JSON text the model returned (may be empty or invalid)."""
    complete: bool
    """True when the model stopped normally and nothing was blocked."""
    finish_reason: str | None
    block_reason: str | None
    usage: Usage

    def json(self) -> dict[str, Any]:
        """The response as a JSON object; ValueError if it is not one."""
        data = json.loads(self.text)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
        return data


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    def generate_structured(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> StructuredResponse:
        """One JSON-mode call constrained by `schema`. Raises LLMError on API failure."""
        ...


@dataclass(frozen=True)
class FunctionSpec:
    """A function the model may call. The caller runs it; the provider never does."""

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON schema of the arguments (type, properties, required, enum, items, ...)."""


@dataclass(frozen=True)
class FunctionCall:
    name: str
    args: dict[str, Any]
    id: str | None = None
    """The provider's call id, echoed back in the matching FunctionResult when set."""


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class ModelMessage:
    """One model turn, as returned by generate_with_tools."""

    text: str
    function_calls: list[FunctionCall] = field(default_factory=list)
    raw: Any = None
    """The provider's own record of the turn, sent back verbatim on the next call (for
    Gemini: the candidate Content, which carries thought signatures that must be kept)."""


@dataclass(frozen=True)
class FunctionResult:
    call: FunctionCall
    response: dict[str, Any]


@dataclass(frozen=True)
class FunctionResultsMessage:
    """The results of every function call of the previous model turn."""

    results: list[FunctionResult]


Message = UserMessage | ModelMessage | FunctionResultsMessage


@dataclass(frozen=True)
class ToolCallResponse:
    message: ModelMessage
    complete: bool
    """True when the model stopped normally and nothing was blocked."""
    finish_reason: str | None
    block_reason: str | None
    usage: Usage

    @property
    def function_calls(self) -> list[FunctionCall]:
        return self.message.function_calls

    @property
    def text(self) -> str:
        return self.message.text


class ToolCallingProvider(Protocol):
    @property
    def model(self) -> str: ...

    def generate_with_tools(
        self,
        system: str,
        messages: Sequence[Message],
        functions: Sequence[FunctionSpec],
        force: Sequence[str] | None = None,
    ) -> ToolCallResponse:
        """One call with `functions` declared. With `force`, the model must call one of
        those functions and cannot answer in text. The provider never runs a function.
        Raises LLMError on API failure."""
        ...


class LLMError(RuntimeError):
    """The provider API failed (after retries, where retrying applies)."""


class TransientError(LLMError):
    """A failure worth retrying with backoff. `retry_after` is the delay the API asked
    for, if any."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitedError(TransientError):
    """HTTP 429 / quota exhausted."""


class ServerUnavailableError(TransientError):
    """HTTP 500/502/503/504: the service failed or was overloaded, not the request."""
