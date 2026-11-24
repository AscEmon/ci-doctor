"""Client-side requests-per-minute limit and backoff on rate-limit errors."""

import time
from collections import deque
from collections.abc import Callable

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cidoc.llm.base import RateLimitedError

WINDOW_S = 60.0
MAX_ATTEMPTS = 5
_exponential = wait_exponential(multiplier=1, min=1, max=60)


class RateLimiter:
    """At most `rpm` requests in any 60-second window."""

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm < 1:
            raise ValueError("rpm must be at least 1")
        self.rpm = rpm
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[float] = deque()

    def acquire(self) -> None:
        """Block until one more request fits in the window, then record it."""
        while True:
            now = self._clock()
            while self._sent and now - self._sent[0] >= WINDOW_S:
                self._sent.popleft()
            if len(self._sent) < self.rpm:
                self._sent.append(now)
                return
            self._sleep(self._sent[0] + WINDOW_S - now)


def backoff_wait(state: RetryCallState) -> float:
    """Exponential backoff, but never shorter than the delay the API asked for."""
    wait = _exponential(state)
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, RateLimitedError) and exc.retry_after:
        wait = max(wait, exc.retry_after)
    return wait


def call_with_backoff[T](
    fn: Callable[[], T],
    limiter: RateLimiter,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[T, int]:
    """Call `fn` within the rate limit, retrying RateLimitedError. Returns (result, attempts).

    Other errors propagate at once; after `max_attempts` the last RateLimitedError does.
    """
    attempts = 0

    def attempt() -> T:
        nonlocal attempts
        attempts += 1
        limiter.acquire()
        return fn()

    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type(RateLimitedError),
        wait=backoff_wait,
        sleep=sleep,
        reraise=True,
    )
    return retrying(attempt), attempts
