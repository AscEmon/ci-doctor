"""`cidoc analyze`, with a scripted provider: no network."""

import gzip
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.labeling.labels import LABELS_FILENAME, Label, append_label
from cidoc.llm.base import LLMError
from tests.datadir import log_text, store_job
from tests.fakes import VALID_DIAGNOSIS, ScriptedProvider, ok

# Assembled at runtime so no secret-shaped literal lives in the repo (as in test_redact).
FAKE_GHP = "ghp" + "_" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0UvWxYz"[:36]


def run(*args: str) -> Any:
    return CliRunner().invoke(app, ["analyze", *args])


def test_analyzes_a_stored_job_by_id(data_dir: Path, provider: ScriptedProvider) -> None:
    store_job(data_dir, 42, log_text())

    result = run("42")

    assert result.exit_code == 0, result.output
    assert "dart_compile" in result.output
    assert "0.90" in result.output
    assert "Undefined name 'foo'." in result.output  # evidence line 5 shown
    assert "Masked" not in result.output
    user = provider.calls[0][1]
    assert "<failed_steps>\nRun flutter build apk\n</failed_steps>" in user
    assert "L5| lib/main.dart:3:5: Error: Undefined name 'foo'." in user


def test_stored_logs_are_used_as_stored(data_dir: Path, provider: ScriptedProvider) -> None:
    # Already redacted when mined; re-redacting could shift the line numbers labels cite.
    log_path = store_job(data_dir, 42, log_text("token [REDACTED:github_token] used"))

    result = run(str(data_dir / log_path))

    assert result.exit_code == 0, result.output
    assert "L5| token [REDACTED:github_token] used" in provider.calls[0][1]


def test_external_logs_are_redacted_before_sending(
    data_dir: Path, provider: ScriptedProvider, tmp_path: Path
) -> None:
    path = tmp_path / "downloaded.txt"
    path.write_text(log_text(f"Error: auth failed for {FAKE_GHP}"), encoding="utf-8")

    result = run(str(path))

    assert result.exit_code == 0, result.output
    assert "Masked 1 secret-looking values" in result.output
    sent = provider.calls[0][1]
    assert FAKE_GHP not in sent
    assert "[REDACTED:" in sent


def test_external_gzip_logs_are_read(
    data_dir: Path, provider: ScriptedProvider, tmp_path: Path
) -> None:
    path = tmp_path / "job.log.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(log_text())

    result = run(str(path))

    assert result.exit_code == 0, result.output
    assert "L5| lib/main.dart" in provider.calls[0][1]


def test_job_only_in_labels_is_found(data_dir: Path, provider: ScriptedProvider) -> None:
    log_path = store_job(data_dir, 7, log_text(), in_db=False)
    label = Label(
        job_id=7,
        repo="o/a",
        run_id=7,
        category="dart_compile",
        root_cause="x",
        evidence_lines=[5],
        confidence=3,
        log_path=log_path,
        labeled_at="2026-10-31T00:00:00+00:00",
    )
    append_label(data_dir / LABELS_FILENAME, label)

    result = run("7")

    assert result.exit_code == 0, result.output
    assert "L5| lib/main.dart:3:5: Error" in provider.calls[0][1]


@pytest.mark.parametrize(
    ("target", "message"),
    [("999", "job 999 is neither in the database nor labeled"), ("nope.log", "neither a log")],
)
def test_unknown_target_exits_2(
    data_dir: Path, provider: ScriptedProvider, target: str, message: str
) -> None:
    connect(data_dir / DB_FILENAME).close()

    result = run(target)

    assert result.exit_code == 2
    assert message in result.output
    assert provider.calls == []


def test_missing_api_key_exits_2_without_calling(data_dir: Path) -> None:
    store_job(data_dir, 42, log_text())

    result = run("42")

    assert result.exit_code == 2
    assert "GEMINI_API_KEY is not set" in result.output


def test_failed_analysis_exits_1_with_the_reason(
    data_dir: Path, provider: ScriptedProvider
) -> None:
    provider.responses = [ok("not json"), ok("still not json")]
    store_job(data_dir, 42, log_text())

    result = run("42")

    assert result.exit_code == 1
    assert "analysis failed: invalid JSON" in result.output


def test_api_error_exits_1(data_dir: Path, provider: ScriptedProvider) -> None:
    provider.responses = [LLMError("Gemini API error 500 INTERNAL: boom")]
    store_job(data_dir, 42, log_text())

    result = run("42")

    assert result.exit_code == 1
    assert "500 INTERNAL" in result.output


def test_model_output_is_printed_literally(data_dir: Path, provider: ScriptedProvider) -> None:
    provider.responses = [ok({**VALID_DIAGNOSIS, "root_cause": "[red]boom[/red] in [bold]x"})]
    store_job(data_dir, 42, log_text())

    result = run("42")

    assert "[red]boom[/red] in [bold]x" in result.output


def test_rerun_uses_the_cache_and_prints_usage(
    data_dir: Path, provider: ScriptedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIDOC_PRICES", '{"fake-model": {"input": 0.1, "output": 0.4}}')
    get_settings.cache_clear()
    store_job(data_dir, 42, log_text())

    first = run("42")
    second = run("42")

    assert first.exit_code == second.exit_code == 0
    assert len(provider.calls) == 1
    assert "usage (fake-model)" in first.output
    assert "$0.000018" in first.output  # 100 in, 20 out
    assert "cached: no API call, $0 this run" in second.output
    assert "dart_compile" in second.output


def test_refresh_bypasses_the_cache(data_dir: Path, provider: ScriptedProvider) -> None:
    provider.responses.append(ok(VALID_DIAGNOSIS))
    store_job(data_dir, 42, log_text())

    run("42")
    result = run("42", "--refresh")

    assert result.exit_code == 0, result.output
    assert len(provider.calls) == 2
    assert "cached" not in result.output


def test_max_cost_needs_a_price(data_dir: Path, provider: ScriptedProvider) -> None:
    store_job(data_dir, 42, log_text())

    result = run("42", "--max-cost", "1")

    assert result.exit_code == 2
    assert "--max-cost needs a price for fake-model" in result.output
    assert provider.calls == []


def test_max_cost_stops_before_an_expensive_call(
    data_dir: Path, provider: ScriptedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIDOC_PRICES", '{"fake-model": {"input": 0.1, "output": 0.4}}')
    get_settings.cache_clear()
    store_job(data_dir, 42, log_text())

    result = run("42", "--max-cost", "0.0000001")

    assert result.exit_code == 1
    assert "Stopped: next call" in result.output
    assert provider.calls == []


def test_max_cost_allows_an_affordable_call(
    data_dir: Path, provider: ScriptedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIDOC_PRICES", '{"fake-model": {"input": 0.1, "output": 0.4}}')
    get_settings.cache_clear()
    store_job(data_dir, 42, log_text())

    result = run("42", "--max-cost", "0.01")

    assert result.exit_code == 0, result.output
    assert len(provider.calls) == 1
