import json
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from cidoc.analyze.analyzer import analyze_log
from cidoc.analyze.cache import (
    ResponseCache,
    analyze_cached,
    cache_key,
    log_hash,
)
from cidoc.analyze.display import render_usage
from cidoc.analyze.preprocess import PreprocessedLog, preprocess
from cidoc.analyze.prompt import Prompt, load_prompt
from cidoc.analyze.usage import summarize
from cidoc.llm.base import LLMError, Usage
from cidoc.llm.pricing import Price
from tests.fakes import VALID_DIAGNOSIS, ScriptedProvider, ok

TS = "2026-10-01T12:34:56.1234567Z "
PROMPT = load_prompt()
PRICE = Price(input=0.10, output=0.40)


def log(error: str = "lib/main.dart:3:5: Error: Undefined name 'foo'.") -> PreprocessedLog:
    lines = ("##[group]Run flutter build apk", "##[endgroup]", "Compiling", "Compiling", error)
    return preprocess("".join(f"{TS}{line}\n" for line in (*lines, "##[error]exit 1")))


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache.in_data_dir(tmp_path / "data")


def test_second_run_is_a_cache_hit_without_api_calls(cache: ResponseCache) -> None:
    provider = ScriptedProvider(ok(VALID_DIAGNOSIS))  # one response: a second call would fail

    first = analyze_cached(provider, log(), PROMPT, cache)
    second = analyze_cached(provider, log(), PROMPT, cache)

    assert len(provider.calls) == 1
    assert not first.cached
    assert second.cached
    assert second.diagnosis == first.diagnosis
    assert second.calls == first.calls  # original usage kept for reporting


def test_failures_are_cached_too(cache: ResponseCache) -> None:
    provider = ScriptedProvider(ok("bad"), ok("bad"))

    first = analyze_cached(provider, log(), PROMPT, cache)
    second = analyze_cached(provider, log(), PROMPT, cache)

    assert len(provider.calls) == 2  # first call and its retry, then nothing
    assert second.cached
    assert second.failure == first.failure
    assert second.failure is not None
    assert second.failure.startswith("invalid JSON")


def test_api_errors_are_not_cached(cache: ResponseCache) -> None:
    provider = ScriptedProvider(LLMError("503"), ok(VALID_DIAGNOSIS))

    with pytest.raises(LLMError):
        analyze_cached(provider, log(), PROMPT, cache)
    result = analyze_cached(provider, log(), PROMPT, cache)

    assert result.ok
    assert not result.cached


def test_refresh_calls_again_and_overwrites(cache: ResponseCache) -> None:
    provider = ScriptedProvider(ok(VALID_DIAGNOSIS), ok({**VALID_DIAGNOSIS, "confidence": 0.5}))
    analyze_cached(provider, log(), PROMPT, cache)

    refreshed = analyze_cached(provider, log(), PROMPT, cache, refresh=True)
    again = analyze_cached(provider, log(), PROMPT, cache)

    assert len(provider.calls) == 2
    assert refreshed.diagnosis is not None
    assert refreshed.diagnosis.confidence == 0.5
    assert again.cached
    assert again.diagnosis == refreshed.diagnosis


def test_key_depends_on_prompt_version_log_and_model() -> None:
    digest = log_hash(log())
    base = cache_key("v1", digest, "m1")

    assert cache_key("v1", digest, "m1") == base
    assert cache_key("v2", digest, "m1") != base
    assert cache_key("v1", digest, "m2") != base
    assert cache_key("v1", log_hash(log("lib/b.dart:1:1: Error: other")), "m1") != base


def test_budget_change_is_a_new_log_hash() -> None:
    long = [f"lib/f{i}.dart:1:1: Error: e{i}" for i in range(400)]
    text = "".join(f"{TS}{line}\n" for line in ("##[group]Run x", *long, "##[error]exit 1"))

    assert log_hash(preprocess(text, budget_tokens=500)) != log_hash(
        preprocess(text, budget_tokens=5000)
    )


def test_new_prompt_version_misses_the_cache(cache: ResponseCache) -> None:
    provider = ScriptedProvider(ok(VALID_DIAGNOSIS), ok(VALID_DIAGNOSIS))
    other = Prompt(system=PROMPT.system, guideline_hash="x", version="other-version")

    analyze_cached(provider, log(), PROMPT, cache)
    result = analyze_cached(provider, log(), other, cache)

    assert not result.cached
    assert len(provider.calls) == 2


def test_corrupt_entry_is_a_miss_and_gets_replaced(cache: ResponseCache) -> None:
    provider = ScriptedProvider(ok(VALID_DIAGNOSIS), ok(VALID_DIAGNOSIS))
    analyze_cached(provider, log(), PROMPT, cache)
    [entry] = list(cache.directory.glob("*.json"))
    entry.write_text("{ truncated", encoding="utf-8")

    result = analyze_cached(provider, log(), PROMPT, cache)

    assert not result.cached
    assert analyze_cached(provider, log(), PROMPT, cache).cached
    assert not list(cache.directory.glob("*.tmp"))


def test_entry_records_what_produced_it(cache: ResponseCache) -> None:
    analyze_cached(ScriptedProvider(ok(VALID_DIAGNOSIS)), log(), PROMPT, cache)

    [entry] = list(cache.directory.glob("*.json"))
    text = entry.read_text(encoding="utf-8")

    for value in (PROMPT.version, PROMPT.guideline_hash, log_hash(log()), "fake-model"):
        assert value in text


# --- usage --------------------------------------------------------------------------


def test_summarize_totals_and_cost() -> None:
    calls = [Usage(1000, 100, 0.5), Usage(1200, 300, 0.7, attempts=2)]

    total = summarize(calls, PRICE)

    assert (total.calls, total.input_tokens, total.output_tokens) == (2, 2200, 400)
    assert total.latency_s == pytest.approx(1.2)
    assert total.cost == pytest.approx((2200 * 0.10 + 400 * 0.40) / 1_000_000)


def test_summarize_without_price_has_unknown_cost() -> None:
    assert summarize([Usage(1, 1, 0.1)], None).cost is None


def render(result_cached: bool, price: Price | None) -> str:
    provider = ScriptedProvider(ok(VALID_DIAGNOSIS, usage=Usage(1234, 56, 0.789)))
    result = replace(analyze_log(provider, log(), PROMPT), cached=result_cached)
    out = StringIO()
    render_usage(Console(file=out, width=120), result, price)
    return out.getvalue()


def test_usage_table_shows_tokens_latency_and_cost() -> None:
    out = render(result_cached=False, price=PRICE)

    for expected in ("1234", "56", "0.79s", "$0.000146"):
        assert expected in out


def test_usage_without_price_says_how_to_set_it() -> None:
    out = render(result_cached=False, price=None)

    assert "unknown" in out
    assert "CIDOC_PRICES" in out


def test_cached_usage_says_no_call_was_made() -> None:
    out = render(result_cached=True, price=PRICE)

    assert "cached: no API call, $0 this run" in out
    assert "1234 in" in out


def test_raw_outputs_are_cached(cache: ResponseCache) -> None:
    provider = ScriptedProvider(ok("garbage"), ok(VALID_DIAGNOSIS))
    first = analyze_cached(provider, log(), PROMPT, cache)

    again = analyze_cached(provider, log(), PROMPT, cache)

    assert first.outputs == ["garbage", again.outputs[1]]
    assert again.cached
    assert again.outputs == first.outputs


def test_entries_cached_before_outputs_were_kept_still_load(cache: ResponseCache) -> None:
    analyze_cached(ScriptedProvider(ok(VALID_DIAGNOSIS)), log(), PROMPT, cache)
    [entry] = list(cache.directory.glob("*.json"))
    data = json.loads(entry.read_text(encoding="utf-8"))
    del data["result"]["outputs"]
    entry.write_text(json.dumps(data), encoding="utf-8")

    result = analyze_cached(ScriptedProvider(), log(), PROMPT, cache)

    assert result.cached
    assert result.outputs == []
