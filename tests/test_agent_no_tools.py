"""`--agent --tools none`: the agent instruction and loop with only submit_diagnosis."""

from pathlib import Path
from typing import Any

import pytest
from google.genai import types
from typer.testing import CliRunner

from cidoc.agent.budget import estimate_item_calls
from cidoc.agent.loop import AgentLimits, run_agent
from cidoc.agent.prompt import (
    AGENT_INSTRUCTION,
    AGENT_INSTRUCTION_NO_TOOLS,
    SUBMIT_FUNCTION,
    SUBMIT_SPEC,
)
from cidoc.agent.runtime import agent_setup, parse_tool_names
from cidoc.agent.tools import TOOLS, ToolBox, ToolEnv, UnknownToolError
from cidoc.analyze.preprocess import preprocess
from cidoc.analyze.prompt import load_prompt
from cidoc.cli import app
from cidoc.eval.results import RUNS_DIRNAME, load_run
from cidoc.logview import clean_log
from tests.agent_fakes import FakeGitHub, ScriptedToolProvider, context, submit, turn
from tests.datadir import log_text
from tests.fakes import VALID_DIAGNOSIS
from tests.test_agent_eval import script, three_items
from tests.test_eval import setup_jobs
from tests.test_llm import FakeClient, FakeTime, provider
from tests.test_llm_tools import call_response


def excerpt() -> Any:
    return preprocess(log_text(), ["Run flutter build apk"])


def no_tools_box(github: FakeGitHub | None = None) -> ToolBox:
    return ToolBox(ToolEnv(context(clean_log(log_text())), github or FakeGitHub()), names=[])


# --- parsing and prompt ---------------------------------------------------------------------


def test_tools_none_parses_to_an_empty_tool_list() -> None:
    assert parse_tool_names("none") == []
    assert parse_tool_names(" none ") == []
    with pytest.raises(UnknownToolError, match="cannot be combined"):
        parse_tool_names("none,read_log")


def test_the_no_tools_prompt_offers_only_submit_diagnosis() -> None:
    setup = agent_setup(load_prompt(), [], 6)

    assert setup.tools == []
    assert setup.prompt.functions == [SUBMIT_SPEC]


def test_the_no_tools_instruction_keeps_the_checklist_and_drops_tool_references() -> None:
    base = load_prompt()
    system = agent_setup(base, [], 6).prompt.system

    assert system == base.system + "\n" + AGENT_INSTRUCTION_NO_TOOLS
    for reasoning in (
        "non-stable Flutter channel (master, beta, main), a build matrix, or\n"
        "  --enable-experiment",
        "could be specific to that channel",
        "no assertion or exception text",
        "consider whether the failure is environmental",
        "about to answer ci_config or toolchain_version",
        '"... (N lines omitted)" right before the first error',
        "your confidence must be at most 0.7",
    ):
        assert reasoning in system, reasoning
    for tool in TOOLS:
        assert tool not in system, tool
    assert "<tool_result" not in system
    assert "at most 6 tools" not in system


def test_the_no_tools_prompt_differs_from_the_full_agent_only_in_tool_access() -> None:
    base = load_prompt()
    full = agent_setup(base, None, 6).prompt
    none = agent_setup(base, [], 6).prompt

    assert full.system == base.system + "\n" + AGENT_INSTRUCTION.format(max_steps=6)
    assert none.version != full.version
    assert none.version.startswith("agent-v2-")
    assert "verification list" in none.user(excerpt())
    assert "call the tool" not in none.user(excerpt())


def test_a_no_tools_item_is_estimated_at_one_call() -> None:
    setup = agent_setup(load_prompt(), [], 6)

    assert len(estimate_item_calls(setup.prompt, excerpt(), setup.limits)) == 1


# --- no tool definitions are sent -------------------------------------------------------


def test_no_tool_definitions_are_sent_to_the_model() -> None:
    scripted = ScriptedToolProvider(turn(submit()))

    result = run_agent(
        scripted, excerpt(), agent_setup(load_prompt(), [], 6).prompt, no_tools_box()
    )

    assert result.ok
    assert scripted.requests[0]["functions"] == [SUBMIT_FUNCTION]
    assert result.trajectory is not None
    assert result.trajectory.tools == []
    assert result.trajectory.tool_calls == []


def test_the_gemini_request_declares_only_submit_diagnosis() -> None:
    client = FakeClient(call_response((SUBMIT_FUNCTION, VALID_DIAGNOSIS, "c1")))
    gemini = provider(client, FakeTime())

    result = run_agent(gemini, excerpt(), agent_setup(load_prompt(), [], 6).prompt, no_tools_box())

    assert result.ok
    config: types.GenerateContentConfig = client.models.calls[0]["config"]
    assert config.tools is not None
    declarations = [
        d.name for tool in config.tools if isinstance(tool, types.Tool)
        for d in tool.function_declarations or []
    ]  # fmt: skip
    assert declarations == [SUBMIT_FUNCTION]


def test_a_call_to_an_undeclared_tool_runs_nothing() -> None:
    github = FakeGitHub()
    scripted = ScriptedToolProvider(turn(("read_workflow", {})), turn(submit()))
    limits = AgentLimits()

    run_agent(scripted, excerpt(), agent_setup(load_prompt(), [], 6).prompt,
              no_tools_box(github), limits)  # fmt: skip

    reply = scripted.requests[1]["messages"][-1].results[0].response["output"]
    assert "error: unknown tool 'read_workflow'" in reply
    assert github.calls == []


# --- eval -----------------------------------------------------------------------------------


def run(*args: str) -> Any:
    return CliRunner().invoke(app, ["eval", *args])


def test_eval_tools_none_records_an_empty_tool_set_and_its_own_version(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    three_items(monkeypatch)
    assert run("--agent").exit_code == 0
    scripted = script(
        monkeypatch,
        turn(submit()),
        turn(submit()),
        turn(("submit_diagnosis", {**VALID_DIAGNOSIS, "category": "dependency_resolution"})),
    )

    result = run("--agent", "--tools", "none")

    assert result.exit_code == 0, result.output
    assert all(r["functions"] == [SUBMIT_FUNCTION] for r in scripted.requests)
    runs = [load_run(path) for path in (data_dir / RUNS_DIRNAME).glob("*_agent*.json")]
    assert len(runs) == 2  # the second run never overwrites the first
    (full,) = [r for r in runs if r.info.tools]
    (none,) = [r for r in runs if r.info.tools == []]
    assert none.info.agent is True
    assert none.info.tools == []
    assert none.info.prompt_version != full.info.prompt_version
    assert none.metrics["mean_tool_calls"] == 0.0
    assert github.calls == []
    assert "none" in result.output


def test_dry_run_with_tools_none_counts_one_call_per_item(
    data_dir: Path, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)

    result = run("--agent", "--tools", "none", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "Agent with no tools: one model call per uncached item" in result.output
    rows = {
        cells[1].strip(): cells[2].strip()
        for cells in (line.split("│") for line in result.output.splitlines())
        if len(cells) >= 3
    }
    assert rows["items"] == "3"
    assert rows["model calls (worst case)"] == "3"
