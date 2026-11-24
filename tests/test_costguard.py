from pathlib import Path

import pytest

from cidoc.analyze.analyzer import analyze_log
from cidoc.analyze.cache import ResponseCache, analyze_cached
from cidoc.analyze.costguard import (
    EXPECTED_OUTPUT_TOKENS,
    BudgetedProvider,
    BudgetExceededError,
    CallEstimate,
    CostGuard,
    estimate_call,
    estimate_run,
)
from cidoc.analyze.preprocess import PreprocessedLog, preprocess
from cidoc.analyze.prompt import load_prompt
from cidoc.llm.base import Usage
from cidoc.llm.pricing import Price
from tests.fakes import VALID_DIAGNOSIS, ScriptedProvider, ok

TS = "2026-10-01T12:34:56.1234567Z "
PROMPT = load_prompt()
# $1 per million tokens both ways keeps the arithmetic readable.
PRICE = Price(input=1.0, output=1.0)


def log(error: str = "lib/main.dart:3:5: Error: Undefined name 'foo'.") -> PreprocessedLog:
    # Five lines: VALID_DIAGNOSIS cites line 5.
    lines = ("##[group]Run flutter build", "##[endgroup]", "Compiling", error, "##[error]exit 1")
    return preprocess("".join(f"{TS}{line}\n" for line in lines))


def first_call_estimate() -> CallEstimate:
    return estimate_call(PROMPT.system, PROMPT.user(log()))


def test_estimate_call_counts_system_and_user_text() -> None:
    estimate = estimate_call("a" * 400, "b" * 800)

    assert estimate == CallEstimate(input_tokens=300, output_tokens=EXPECTED_OUTPUT_TOKENS)
    assert estimate.cost(PRICE) == pytest.approx((300 + EXPECTED_OUTPUT_TOKENS) / 1_000_000)


def test_call_within_budget_is_made_and_charged_actual_usage() -> None:
    guard = CostGuard(max_cost=1.0, price=PRICE)
    scripted = ScriptedProvider(ok(VALID_DIAGNOSIS, usage=Usage(3000, 200, 1)))
    provider = BudgetedProvider(scripted, guard)

    result = analyze_log(provider, log(), PROMPT)

    assert result.ok
    assert guard.calls == 1
    assert guard.spent == pytest.approx(3200 / 1_000_000)
    assert provider.model == "fake-model"


def test_call_over_budget_is_not_made() -> None:
    scripted = ScriptedProvider(ok(VALID_DIAGNOSIS))
    too_little = first_call_estimate().cost(PRICE) * 0.99
    provider = BudgetedProvider(scripted, CostGuard(max_cost=too_little, price=PRICE))

    with pytest.raises(BudgetExceededError, match="would exceed --max-cost"):
        analyze_log(provider, log(), PROMPT)
    assert scripted.calls == []


def test_exactly_the_budget_is_allowed() -> None:
    exact = first_call_estimate().cost(PRICE)
    scripted = ScriptedProvider(ok(VALID_DIAGNOSIS, usage=Usage(1, 1, 0.1)))

    analyze_log(BudgetedProvider(scripted, CostGuard(exact, PRICE)), log(), PROMPT)

    assert len(scripted.calls) == 1


def test_retry_is_guarded_too() -> None:
    # Enough for the first call, but its actual usage leaves too little for the retry.
    estimate = first_call_estimate()
    budget = estimate.cost(PRICE) * 1.5
    heavy = Usage(estimate.input_tokens, estimate.output_tokens, 0.1)
    scripted = ScriptedProvider(ok("not json", usage=heavy), ok(VALID_DIAGNOSIS))
    guard = CostGuard(budget, PRICE)

    with pytest.raises(BudgetExceededError) as exc_info:
        analyze_log(BudgetedProvider(scripted, guard), log(), PROMPT)

    assert len(scripted.calls) == 1
    assert exc_info.value.spent == pytest.approx(estimate.cost(PRICE))
    assert guard.spent <= budget


def test_guard_spans_many_analyses() -> None:
    per_call = first_call_estimate().cost(PRICE)
    usage = Usage(first_call_estimate().input_tokens, EXPECTED_OUTPUT_TOKENS, 0.1)
    scripted = ScriptedProvider(*[ok(VALID_DIAGNOSIS, usage=usage) for _ in range(5)])
    provider = BudgetedProvider(scripted, CostGuard(per_call * 3.5, PRICE))

    done = 0
    with pytest.raises(BudgetExceededError):
        for _ in range(5):
            analyze_log(provider, log(), PROMPT)
            done += 1

    assert done == 3
    assert len(scripted.calls) == 3


def test_cache_hits_cost_nothing_even_with_a_zero_budget(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    analyze_cached(ScriptedProvider(ok(VALID_DIAGNOSIS)), log(), PROMPT, cache)
    scripted = ScriptedProvider()
    guard = CostGuard(max_cost=0.0, price=PRICE)

    result = analyze_cached(BudgetedProvider(scripted, guard), log(), PROMPT, cache)

    assert result.cached
    assert guard.spent == 0.0
    assert scripted.calls == []


def test_budget_error_is_not_cached(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    broke = BudgetedProvider(ScriptedProvider(), CostGuard(0.0, PRICE))

    with pytest.raises(BudgetExceededError):
        analyze_cached(broke, log(), PROMPT, cache)

    assert not list(cache.directory.glob("*.json"))


# --- run estimate (what eval --dry-run prints) --------------------------------------


def test_estimate_run_totals_cost_and_duration() -> None:
    calls = [CallEstimate(1000, 500), CallEstimate(3000, 500)]

    run = estimate_run(calls, cached=2, price=Price(input=0.1, output=0.4), rpm=10)

    assert (run.items, run.cached, run.calls) == (4, 2, 2)
    assert (run.input_tokens, run.output_tokens) == (4000, 1000)
    assert run.cost == pytest.approx((4000 * 0.1 + 1000 * 0.4) / 1_000_000)
    assert run.duration_s == pytest.approx(12.0)


def test_estimate_run_without_price_or_calls() -> None:
    assert estimate_run([CallEstimate(10)], cached=0, price=None, rpm=10).cost is None
    empty = estimate_run([], cached=5, price=PRICE, rpm=10)
    assert (empty.calls, empty.cost, empty.duration_s) == (0, 0.0, 0.0)
