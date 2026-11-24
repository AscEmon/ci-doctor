"""Token, cost and latency totals for one analysis or many."""

from collections.abc import Iterable
from dataclasses import dataclass

from cidoc.llm.base import Usage
from cidoc.llm.pricing import Price


@dataclass(frozen=True)
class UsageSummary:
    calls: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost: float | None
    """USD estimate; None when the model has no configured price."""


def summarize(calls: Iterable[Usage], price: Price | None) -> UsageSummary:
    calls = list(calls)
    input_tokens = sum(c.input_tokens for c in calls)
    output_tokens = sum(c.output_tokens for c in calls)
    return UsageSummary(
        calls=len(calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=sum(c.latency_s for c in calls),
        cost=price.cost(input_tokens, output_tokens) if price else None,
    )
