import pytest

from cidoc.analyze.analyzer import MAX_ATTEMPTS, analyze_log
from cidoc.analyze.preprocess import PreprocessedLog, preprocess
from cidoc.analyze.prompt import load_prompt
from cidoc.analyze.schema import RESPONSE_SCHEMA, ROOT_CAUSE_MAX_CHARS, Diagnosis
from cidoc.labeling.labels import CATEGORIES
from cidoc.llm.base import LLMError, StructuredResponse, Usage
from tests.fakes import VALID_DIAGNOSIS as VALID
from tests.fakes import ScriptedProvider, ok

TS = "2026-10-01T12:34:56.1234567Z "
PROMPT = load_prompt()


def log() -> PreprocessedLog:
    text = "".join(
        f"{TS}{line}\n"
        for line in (
            "##[group]Run flutter build apk",
            "flutter build apk",
            "##[endgroup]",
            "Compiling lib/main.dart",
            "lib/main.dart:3:5: Error: Undefined name 'foo'.",
            "##[error]Process completed with exit code 1.",
        )
    )
    return preprocess(text)


def test_schema_enum_is_the_taxonomy() -> None:
    assert RESPONSE_SCHEMA["properties"]["category"]["enum"] == list(CATEGORIES)
    assert set(RESPONSE_SCHEMA["required"]) == set(Diagnosis.model_fields)


def test_valid_response_on_first_call() -> None:
    provider = ScriptedProvider(ok(VALID))

    result = analyze_log(provider, log(), PROMPT)

    assert result.ok
    assert result.diagnosis == Diagnosis(**VALID)
    assert result.failure is None
    assert result.model == "fake-model"
    assert len(result.calls) == 1
    system, user, schema = provider.calls[0]
    assert system == PROMPT.system
    assert "L5| lib/main.dart:3:5: Error: Undefined name 'foo'." in user
    assert schema is RESPONSE_SCHEMA


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (ok("{not json"), "invalid JSON"),
        (ok("[1, 2]"), "invalid JSON"),
        (ok(""), "empty response"),
        (ok({**VALID, "category": "made_up"}), "schema mismatch: category"),
        (ok({**VALID, "root_cause": "x" * (ROOT_CAUSE_MAX_CHARS + 1)}), "root_cause"),
        (ok({**VALID, "root_cause": "   "}), "root_cause"),
        (ok({**VALID, "confidence": 1.5}), "confidence"),
        (ok({**VALID, "evidence_lines": [0]}), "evidence_lines"),
        (ok({**VALID, "evidence_lines": [999]}), "past the last log line"),
        (ok({k: v for k, v in VALID.items() if k != "suggested_fix"}), "suggested_fix"),
        (ok({**VALID, "extra": 1}), "extra"),
        (ok("", complete=False, finish_reason="SAFETY"), "incomplete: finish reason SAFETY"),
        (ok(VALID, complete=False, finish_reason="MAX_TOKENS"), "incomplete"),
        (ok("", complete=False, finish_reason=None, block_reason="OTHER"), "blocked: OTHER"),
    ],
)
def test_invalid_response_is_retried_once_with_the_error(
    bad: StructuredResponse, reason: str
) -> None:
    provider = ScriptedProvider(bad, ok(VALID))

    result = analyze_log(provider, log(), PROMPT)

    assert result.ok
    assert len(result.calls) == 2
    first_prompt, retry = provider.calls[0][1], provider.calls[1][1]
    assert retry.startswith(first_prompt)
    assert "previous answer was rejected" in retry
    assert reason in retry


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (ok("", complete=False, finish_reason=None, block_reason="SAFETY"), "blocked: SAFETY"),
        (ok(""), "empty response"),
        (ok("nope"), "invalid JSON"),
        (ok({**VALID, "confidence": -1}), "schema mismatch"),
    ],
)
def test_second_failure_is_recorded_not_raised(bad: StructuredResponse, reason: str) -> None:
    provider = ScriptedProvider(bad, bad)

    result = analyze_log(provider, log(), PROMPT)

    assert not result.ok
    assert result.diagnosis is None
    assert result.failure is not None
    assert result.failure.startswith(reason)
    assert len(result.calls) == MAX_ATTEMPTS == 2


def test_validation_error_does_not_echo_model_output() -> None:
    long_cause = "SECRET-LOOKING-" + "x" * 300
    provider = ScriptedProvider(ok({**VALID, "root_cause": long_cause}), ok(VALID))

    analyze_log(provider, log(), PROMPT)

    assert long_cause not in provider.calls[1][1]


def test_api_errors_propagate() -> None:
    provider = ScriptedProvider(LLMError("500 INTERNAL"))

    with pytest.raises(LLMError):
        analyze_log(provider, log(), PROMPT)


def test_usage_of_every_call_is_kept() -> None:
    first, second = Usage(100, 5, 0.2), Usage(120, 30, 0.4)
    provider = ScriptedProvider(ok("bad", usage=first), ok(VALID, usage=second))

    result = analyze_log(provider, log(), PROMPT)

    assert result.calls == [first, second]
