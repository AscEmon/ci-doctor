"""Spend limits: estimate each call's cost before making it, and stop before a budget is
exceeded.

The guard wraps any LLMProvider, so it sees every real API call (first attempts and
validation retries alike) and none of the cache hits, which cost nothing. Estimates use
the same chars-per-token rule as the preprocessor's budget; output is a fixed allowance.
After each call the actual usage is charged, so an estimate error can overshoot the
budget by at most one call.
"""

from dataclasses import dataclass
from typing import Any

from cidoc.analyze.preprocess import estimate_tokens
from cidoc.llm.base import LLMProvider, StructuredResponse, Usage
from cidoc.llm.pricing import Price

# A diagnosis JSON is ~100-250 tokens; leave room for thinking tokens, billed as output.
EXPECTED_OUTPUT_TOKENS = 500
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class CallEstimate:
    input_tokens: int
    output_tokens: int = EXPECTED_OUTPUT_TOKENS

    def cost(self, price: Price) -> float:
        return price.cost(self.input_tokens, self.output_tokens)


def estimate_call(system: str, user: str) -> CallEstimate:
    return CallEstimate(input_tokens=estimate_tokens(system) + estimate_tokens(user))


class BudgetExceededError(RuntimeError):
    """The next call would push spending past the budget; it was not made."""

    def __init__(self, spent: float, next_cost: float, max_cost: float) -> None:
        super().__init__(
            f"next call (~${next_cost:.6f}) would exceed --max-cost ${max_cost:.6f}"
            f" (spent ${spent:.6f} so far)"
        )
        self.spent = spent
        self.next_cost = next_cost
        self.max_cost = max_cost


@dataclass
class CostGuard:
    max_cost: float
    price: Price
    spent: float = 0.0
    calls: int = 0

    def check(self, estimate: CallEstimate) -> None:
        next_cost = estimate.cost(self.price)
        if self.spent + next_cost > self.max_cost:
            raise BudgetExceededError(self.spent, next_cost, self.max_cost)

    def charge(self, response: StructuredResponse) -> None:
        self.charge_usage(response.usage)

    def charge_usage(self, usage: Usage) -> None:
        self.spent += self.price.cost(usage.input_tokens, usage.output_tokens)
        self.calls += 1


class BudgetedProvider:
    """An LLMProvider that checks the guard before every call and charges it after."""

    def __init__(self, provider: LLMProvider, guard: CostGuard) -> None:
        self._provider = provider
        self.guard = guard

    @property
    def model(self) -> str:
        return self._provider.model

    def generate_structured(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> StructuredResponse:
        self.guard.check(estimate_call(system, user))
        response = self._provider.generate_structured(system, user, schema)
        self.guard.charge(response)
        return response


@dataclass(frozen=True)
class RunEstimate:
    items: int
    cached: int
    """Items with a cached result: no call, no cost."""
    calls: int
    input_tokens: int
    output_tokens: int
    cost: float | None
    """USD; None without a configured price."""
    duration_s: float
    """At the configured requests-per-minute limit. Validation retries are not included."""


def estimate_run(
    calls: list[CallEstimate],
    cached: int,
    price: Price | None,
    rpm: int,
    items: int | None = None,
) -> RunEstimate:
    """What a run would cost: `calls` for the uncached items (one per item unless
    `items` says how many there are, as for agent runs with several calls each)."""
    input_tokens = sum(c.input_tokens for c in calls)
    output_tokens = sum(c.output_tokens for c in calls)
    return RunEstimate(
        items=items if items is not None else len(calls) + cached,
        cached=cached,
        calls=len(calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=price.cost(input_tokens, output_tokens) if price else None,
        duration_s=len(calls) * SECONDS_PER_MINUTE / rpm,
    )
