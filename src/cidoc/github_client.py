"""Thin httpx client for the GitHub REST API."""

import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import SecretStr
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

API_VERSION = "2022-11-28"
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
DEFAULT_MAX_ATTEMPTS = 5
# GitHub asks clients to wait at least a minute on a secondary limit without Retry-After.
SECONDARY_RATE_LIMIT_MIN_WAIT = 60.0
# Small buffer so we wake just after the primary rate-limit window resets.
RATE_LIMIT_RESET_BUFFER = 1.0

_backoff = wait_exponential(multiplier=1, min=1, max=60)


class RetryableHTTPError(httpx.HTTPStatusError):
    """A 5xx or rate-limit response that is worth retrying."""


def _is_primary_rate_limit_exhausted(response: httpx.Response) -> bool:
    remaining: str | None = response.headers.get("X-RateLimit-Remaining")
    return remaining == "0"


def _is_primary_rate_limit(response: httpx.Response) -> bool:
    return response.status_code in (403, 429) and _is_primary_rate_limit_exhausted(response)


def _is_secondary_rate_limit(response: httpx.Response) -> bool:
    if response.status_code not in (403, 429) or _is_primary_rate_limit(response):
        return False
    return "Retry-After" in response.headers or "secondary rate limit" in response.text.lower()


def _is_retryable(response: httpx.Response) -> bool:
    return (
        response.is_server_error
        or _is_primary_rate_limit(response)
        or _is_secondary_rate_limit(response)
    )


class GitHubClient:
    def __init__(
        self,
        token: SecretStr,
        base_url: str = "https://api.github.com",
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sleep = sleep
        self._clock = clock
        # Log downloads answer 302 to a signed blob URL on another host;
        # httpx drops the Authorization header on cross-origin redirects.
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {token.get_secret_value()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "ci-doctor",
            },
            follow_redirects=True,
            timeout=DEFAULT_TIMEOUT,
            transport=transport,
        )
        self._retrying = Retrying(
            retry=retry_if_exception_type(RetryableHTTPError),
            wait=self._wait,
            stop=stop_after_attempt(max_attempts),
            sleep=sleep,
            reraise=True,
        )

    def __repr__(self) -> str:
        return f"GitHubClient(base_url={str(self._http.base_url)!r})"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        return self._retrying(self._send, method, url, **kwargs)

    def _seconds_until_reset(self, response: httpx.Response) -> float:
        reset = response.headers.get("X-RateLimit-Reset", "")
        if not reset.isdigit():
            return SECONDARY_RATE_LIMIT_MIN_WAIT
        return max(0.0, float(reset) - self._clock() + RATE_LIMIT_RESET_BUFFER)

    def _wait(self, state: RetryCallState) -> float:
        backoff = _backoff(state)
        exc = state.outcome.exception() if state.outcome else None
        if not isinstance(exc, RetryableHTTPError):
            return backoff
        if _is_primary_rate_limit(exc.response):
            return self._seconds_until_reset(exc.response)
        if not _is_secondary_rate_limit(exc.response):
            return backoff
        retry_after = exc.response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return float(retry_after)
        return max(SECONDARY_RATE_LIMIT_MIN_WAIT, backoff)

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        response = self._http.request(method, url, **kwargs)
        if response.is_success and _is_primary_rate_limit_exhausted(response):
            # Quota just ran out: pause now so the next call doesn't fail.
            self._sleep(self._seconds_until_reset(response))
        if _is_retryable(response):
            raise RetryableHTTPError(
                f"Retryable HTTP {response.status_code} for {method} {response.url}",
                request=response.request,
                response=response,
            )
        response.raise_for_status()
        return response

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", url, params=params).json()

    def download_log(self, url: str) -> bytes:
        return self.request("GET", url).content
