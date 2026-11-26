"""`cidoc analyze --agent`, with a scripted model and fake GitHub: no network."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cidoc.agent.cache import CACHE_DIRNAME
from cidoc.cli import app
from cidoc.config import Settings
from tests.agent_fakes import FakeGitHub, ScriptedToolProvider, submit, turn
from tests.datadir import log_text, store_job


def run(*args: str) -> Any:
    return CliRunner().invoke(app, ["analyze", *args])


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> FakeGitHub:
    fake = FakeGitHub()
    monkeypatch.setattr("cidoc.cli.open_tools_github", lambda token, base_url: fake)
    return fake


def script(monkeypatch: pytest.MonkeyPatch, *turns: Any) -> ScriptedToolProvider:
    scripted = ScriptedToolProvider(*turns)

    def create(settings: Settings, model: str | None = None) -> ScriptedToolProvider:
        return scripted

    monkeypatch.setattr("cidoc.cli.create_tool_provider", create)
    return scripted


def test_agent_prints_the_trajectory_and_the_diagnosis(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    store_job(data_dir, 42, log_text())
    provider = script(
        monkeypatch,
        turn(("search_log", {"pattern": "Undefined name"})),
        turn(submit(evidence_files=["lib/main.dart:3"])),
    )

    result = run("42", "--agent")

    assert result.exit_code == 0, result.output
    assert "agent trajectory" in result.output
    assert "search_log" in result.output
    assert 'pattern="Undefined name"' in result.output
    assert "1 lines match" in result.output  # output preview
    assert "1 tool call (max 6); 2 model calls" in result.output
    assert "dart_compile" in result.output
    assert "lib/main.dart:3" in result.output
    assert len(provider.requests) == 2
    assert github.closed
    assert list((data_dir / CACHE_DIRNAME).glob("*.json"))


def test_tools_restricts_what_the_model_is_offered(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    store_job(data_dir, 42, log_text())
    provider = script(monkeypatch, turn(submit()))

    result = run("42", "--agent", "--tools", "search_log, read_log", "--max-steps", "2")

    assert result.exit_code == 0, result.output
    assert provider.requests[0]["functions"] == ["search_log", "read_log", "submit_diagnosis"]
    assert "max 2 steps" in result.output
    assert "no tool calls" in result.output


def test_max_steps_is_enforced_from_the_cli(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    store_job(data_dir, 42, log_text())
    provider = script(
        monkeypatch,
        turn(("read_log", {"start_line": 1, "end_line": 2})),
        turn(submit()),
    )

    result = run("42", "--agent", "--max-steps", "1")

    assert result.exit_code == 0, result.output
    assert provider.requests[1]["force"] == ["submit_diagnosis"]
    assert "forced submit: max_steps" in result.output


def test_a_second_run_is_served_from_the_cache(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    store_job(data_dir, 42, log_text())
    script(monkeypatch, turn(("read_log", {"start_line": 1, "end_line": 2})), turn(submit()))
    assert run("42", "--agent").exit_code == 0

    again = script(monkeypatch)
    result = run("42", "--agent")

    assert result.exit_code == 0, result.output
    assert again.requests == []
    assert "cached: no calls this run" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--tools", "read_file,rm_rf"], "unknown tools: rm_rf"),
        (["--tools", " , "], "at least one tool"),
    ],
)
def test_bad_tool_lists_are_rejected(
    data_dir: Path, github: FakeGitHub, args: list[str], message: str
) -> None:
    store_job(data_dir, 42, log_text())

    result = run("42", "--agent", *args)

    assert result.exit_code == 2
    assert message in result.output


def test_agent_options_need_agent(data_dir: Path) -> None:
    result = run("42", "--max-steps", "3")

    assert result.exit_code == 2
    assert "only apply to --agent" in result.output


def test_agent_needs_a_stored_job(data_dir: Path, tmp_path: Path) -> None:
    path = tmp_path / "job.log"
    path.write_text(log_text(), encoding="utf-8")

    result = run(str(path), "--agent")

    assert result.exit_code == 2
    assert "needs a stored job id" in result.output


def test_no_call_prints_the_agent_input_without_a_key(data_dir: Path) -> None:
    store_job(data_dir, 42, log_text())

    result = run("42", "--agent", "--no-call")

    assert result.exit_code == 0, result.output
    assert "When to use tools:" in result.output
    assert "submit_diagnosis: Submit the final diagnosis" in result.output
    assert "Agent prompt agent-" in result.output


def test_a_failed_agent_run_exits_1(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    store_job(data_dir, 42, log_text())
    script(monkeypatch, turn(text="no"), turn(text="still no"))

    result = run("42", "--agent")

    assert result.exit_code == 1
    assert "analysis failed" in result.output
