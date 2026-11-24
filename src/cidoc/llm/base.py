"""Provider-agnostic LLM interface: one structured-output call, with usage.

The analyzer only depends on LLMProvider; adding a provider means one new class.
"""

import json
from dataclasses import dataclass
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
