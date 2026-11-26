"""The agent loop with a scripted model and fake GitHub: no network."""

import json
from pathlib import Path
from typing import Any

import pytest

from cidoc.agent.cache import ToolCache
from cidoc.agent.context import load_context
from cidoc.agent.loop import (
    FORCED_MAX_INPUT,
    FORCED_MAX_STEPS,
    FORCED_REPROMPT,
    FORCED_RETRY,
    NOT_RUN_LIMIT,
    REPROMPT,
    AgentLimits,
    run_agent,
    run_agent_cached,
)
from cidoc.agent.prompt import SUBMIT_FUNCTION, AgentPrompt, build_agent_prompt
from cidoc.agent.tools import OUTPUT_CAP_TOKENS, TOOLS, ToolBox, ToolEnv
from cidoc.analyze.cache import ResponseCache
from cidoc.analyze.preprocess import PreprocessedLog, preprocess
from cidoc.analyze.prompt import load_prompt
from cidoc.analyze.schema import AgentDiagnosis
from cidoc.db import connect
from cidoc.llm.base import FunctionResultsMessage, LLMError, Usage, UserMessage
from cidoc.llm.pricing import Price
from cidoc.logview import clean_log
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.runs import FailedJob, insert_failure
from tests.agent_fakes import (
    FIX_SHA,
    JOB_ID,
    REPO,
    RUN_ID,
    SHA,
    WORKFLOW,
    Call,
    FakeGitHub,
    ScriptedToolProvider,
    contents,
    context,
    file_entry,
    submit,
    turn,
)
from tests.datadir import NOW, log_text

LOG = log_text()
READ_PUBSPEC = ("read_file", {"path": "pubspec.yaml"})


def excerpt() -> PreprocessedLog:
    return preprocess(LOG, ["Run flutter build apk"])


def agent_prompt(limits: AgentLimits = AgentLimits()) -> AgentPrompt:  # noqa: B008
    return build_agent_prompt(
        load_prompt(), list(TOOLS.values()), limits.max_steps, limits.max_input_tokens,
        limits.cap_tokens,
    )  # fmt: skip


def toolbox(github: FakeGitHub | None = None, cache: ToolCache | None = None) -> ToolBox:
    github = github or FakeGitHub({contents("pubspec.yaml"): file_entry("name: app\n")})
    return ToolBox(ToolEnv(context(clean_log(LOG)), github), cache=cache)


def agent(
    provider: ScriptedToolProvider,
    limits: AgentLimits = AgentLimits(),  # noqa: B008
    tools: ToolBox | None = None,
    price: Price | None = None,
) -> Any:
    return run_agent(
        provider, excerpt(), agent_prompt(limits), tools or toolbox(), limits, price=price
    )


# --- the happy paths -----------------------------------------------------------------------


def test_an_answer_without_tools_is_one_model_call() -> None:
    provider = ScriptedToolProvider(turn(submit()))

    result = agent(provider)

    assert result.ok
    assert isinstance(result.diagnosis, AgentDiagnosis)
    assert result.diagnosis.category == "dart_compile"
    assert result.diagnosis.evidence_files == []
    assert provider.requests[0]["force"] is None
    assert provider.requests[0]["functions"] == [*TOOLS, SUBMIT_FUNCTION]
    assert result.trajectory is not None
    assert result.trajectory.tool_calls == []
    assert result.trajectory.model_calls == 1
    assert result.trajectory.stop == "submitted"


def test_tool_results_are_sent_back_and_recorded() -> None:
    provider = ScriptedToolProvider(
        turn(READ_PUBSPEC, ("search_log", {"pattern": "Undefined name"})),
        turn(submit(evidence_files=["pubspec.yaml:1"])),
    )

    result = agent(provider)

    assert result.ok
    assert result.diagnosis.evidence_files == ["pubspec.yaml:1"]
    results = provider.requests[1]["messages"][-1]
    assert isinstance(results, FunctionResultsMessage)
    assert [r.call.name for r in results.results] == ["read_file", "search_log"]
    assert "1| name: app" in results.results[0].response["output"]
    assert (
        "L5| lib/main.dart:3:5: Error: Undefined name 'foo'."
        in (results.results[1].response["output"])
    )
    trajectory = result.trajectory
    assert [(r.step, r.turn, r.name) for r in trajectory.tool_calls] == [
        (1, 1, "read_file"),
        (2, 1, "search_log"),
    ]
    first = trajectory.tool_calls[0]
    assert first.args == {"path": "pubspec.yaml", "start_line": 1, "end_line": 200}
    assert first.ok
    assert first.output_chars == len(results.results[0].response["output"])
    assert first.output_tokens > 0
    assert first.preview.startswith("pubspec.yaml @")
    assert trajectory.model_calls == 2


def test_trajectory_totals_tokens_and_cost() -> None:
    provider = ScriptedToolProvider(
        turn(READ_PUBSPEC, usage=Usage(2000, 100, 1.0)),
        turn(submit(), usage=Usage(3000, 200, 1.0)),
    )

    result = agent(provider, price=Price(input=1.0, output=10.0))

    trajectory = result.trajectory
    assert (trajectory.input_tokens, trajectory.output_tokens) == (5000, 300)
    assert trajectory.cost == pytest.approx((5000 * 1.0 + 300 * 10.0) / 1_000_000)
    assert [c.input_tokens for c in result.calls] == [2000, 3000]


def test_a_tool_error_is_shown_to_the_model_and_the_run_goes_on() -> None:
    provider = ScriptedToolProvider(
        turn(("read_file", {"path": "../../etc/passwd"})),
        turn(submit()),
    )

    result = agent(provider)

    assert result.ok
    response = provider.requests[1]["messages"][-1].results[0].response
    assert response["output"].startswith("error: ")
    assert not result.trajectory.tool_calls[0].ok


# --- limits -------------------------------------------------------------------------------


def test_max_steps_forces_submit_diagnosis() -> None:
    limits = AgentLimits(max_steps=2)
    provider = ScriptedToolProvider(
        turn(READ_PUBSPEC),
        turn(("list_dir", {})),
        turn(submit()),
    )

    result = agent(provider, limits)

    assert result.ok
    assert [r["force"] for r in provider.requests] == [None, None, [SUBMIT_FUNCTION]]
    assert result.trajectory.hit_max_steps
    assert result.trajectory.forced == [FORCED_MAX_STEPS]
    assert len(result.trajectory.tool_calls) == 2


def test_calls_past_the_limit_in_one_turn_are_answered_not_run() -> None:
    limits = AgentLimits(max_steps=2)
    provider = ScriptedToolProvider(
        turn(READ_PUBSPEC, ("list_dir", {}), ("read_workflow", {})),
        turn(submit()),
    )

    result = agent(provider, limits)

    results = provider.requests[1]["messages"][-1].results
    assert [r.call.name for r in results] == ["read_file", "list_dir", "read_workflow"]
    assert results[2].response == {"error": NOT_RUN_LIMIT}
    assert len(result.trajectory.tool_calls) == 2
    assert provider.requests[1]["force"] == [SUBMIT_FUNCTION]


def test_max_input_tokens_forces_submit_before_a_tool_round_that_would_not_fit() -> None:
    limits = AgentLimits(max_input_tokens=5_000)  # the prompt alone is ~4k tokens
    provider = ScriptedToolProvider(turn(submit()))

    result = agent(provider, limits)

    assert result.ok
    assert provider.requests[0]["force"] == [SUBMIT_FUNCTION]
    assert result.trajectory.forced == [FORCED_MAX_INPUT]
    assert not result.trajectory.hit_max_steps


def test_max_input_tokens_counts_actual_usage_across_calls() -> None:
    limits = AgentLimits(max_input_tokens=30_000)
    provider = ScriptedToolProvider(
        turn(READ_PUBSPEC, usage=Usage(25_000, 10, 0.1)),  # already most of the budget
        turn(submit()),
    )

    result = agent(provider, limits)

    assert provider.requests[1]["force"] == [SUBMIT_FUNCTION]
    assert result.trajectory.forced == [FORCED_MAX_INPUT]


def test_a_forced_model_that_still_calls_tools_gets_no_tool_run() -> None:
    limits = AgentLimits(max_steps=1)
    provider = ScriptedToolProvider(
        turn(READ_PUBSPEC),
        turn(("list_dir", {})),  # ignored the forced mode
        turn(submit()),
    )

    result = agent(provider, limits)

    assert result.ok
    assert len(result.trajectory.tool_calls) == 1
    assert provider.requests[2]["messages"][-1].results[0].response == {"error": NOT_RUN_LIMIT}


def test_the_model_call_limit_ends_a_run_that_never_submits() -> None:
    limits = AgentLimits(max_steps=1)
    provider = ScriptedToolProvider(*[turn(READ_PUBSPEC) for _ in range(limits.max_model_calls)])

    result = agent(provider, limits)

    assert not result.ok
    assert result.failure == "no valid submit_diagnosis within the model-call limit"
    assert len(provider.requests) == limits.max_model_calls


# --- re-prompts and retries --------------------------------------------------------------


def test_text_instead_of_a_call_is_reprompted_once() -> None:
    provider = ScriptedToolProvider(turn(text="It is a dart_compile error."), turn(submit()))

    result = agent(provider)

    assert result.ok
    second = provider.requests[1]
    assert second["force"] == [SUBMIT_FUNCTION]
    assert isinstance(second["messages"][-1], UserMessage)
    assert second["messages"][-1].text == REPROMPT
    assert result.trajectory.forced == [FORCED_REPROMPT]


def test_text_twice_is_a_failure() -> None:
    provider = ScriptedToolProvider(turn(text="one"), turn(text="two"))

    result = agent(provider)

    assert not result.ok
    assert result.failure == "answered in text instead of calling submit_diagnosis"
    assert len(provider.requests) == 2
    assert result.trajectory.stop == "failed"


def test_a_cut_off_turn_without_a_call_is_reprompted() -> None:
    provider = ScriptedToolProvider(turn(finish="MALFORMED_FUNCTION_CALL"), turn(submit()))

    assert agent(provider).ok


def test_an_invalid_submission_is_rejected_once_then_accepted() -> None:
    provider = ScriptedToolProvider(
        turn(submit(category="flaky")),
        turn(submit()),
    )

    result = agent(provider)

    assert result.ok
    rejection = provider.requests[1]["messages"][-1].results[0].response["error"]
    assert rejection.startswith("rejected: schema mismatch: category")
    assert provider.requests[1]["force"] == [SUBMIT_FUNCTION]
    assert result.trajectory.forced == [FORCED_RETRY]


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"confidence": 2}, "confidence"),
        ({"evidence_lines": [999]}, "past the last log line 6"),
        ({"evidence_files": ["pubspec.yaml"]}, "path:line"),
        ({"root_cause": "x" * 300}, "root_cause"),
    ],
)
def test_two_invalid_submissions_fail_with_the_reason(bad: dict[str, Any], message: str) -> None:
    provider = ScriptedToolProvider(turn(submit(**bad)), turn(submit(**bad)))

    result = agent(provider)

    assert not result.ok
    assert result.failure is not None
    assert message in result.failure
    assert len(provider.requests) == 2


def test_a_rejected_submit_answers_every_call_of_its_turn() -> None:
    provider = ScriptedToolProvider(turn(READ_PUBSPEC, submit(confidence=5)), turn(submit()))

    agent(provider)

    results = provider.requests[1]["messages"][-1].results
    assert [r.call.name for r in results] == ["read_file", SUBMIT_FUNCTION]
    assert results[0].response["error"].startswith("not run")


def test_a_blocked_turn_fails_without_a_retry() -> None:
    provider = ScriptedToolProvider(turn(block="SAFETY"))

    result = agent(provider)

    assert result.failure == "blocked: SAFETY"
    assert len(provider.requests) == 1


def test_api_errors_propagate() -> None:
    provider = ScriptedToolProvider(LLMError("quota"))

    with pytest.raises(LLMError):
        agent(provider)


# --- caching --------------------------------------------------------------------------------


def test_a_cached_agent_result_makes_no_model_or_github_call(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "analyze")
    tool_cache = ToolCache(tmp_path / "tools")
    first = ScriptedToolProvider(turn(READ_PUBSPEC), turn(submit()))
    run_agent_cached(first, excerpt(), agent_prompt(), toolbox(cache=tool_cache), cache)

    github = FakeGitHub()
    again = ScriptedToolProvider()
    result = run_agent_cached(again, excerpt(), agent_prompt(), toolbox(github, tool_cache), cache)

    assert result.cached
    assert result.ok
    assert result.trajectory is not None
    assert [r.name for r in result.trajectory.tool_calls] == ["read_file"]
    assert again.requests == []
    assert github.calls == []


def test_a_refresh_reuses_cached_tool_results(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "analyze")
    tool_cache = ToolCache(tmp_path / "tools")
    first = ScriptedToolProvider(turn(READ_PUBSPEC), turn(submit()))
    run_agent_cached(first, excerpt(), agent_prompt(), toolbox(cache=tool_cache), cache)

    github = FakeGitHub()
    again = ScriptedToolProvider(turn(READ_PUBSPEC), turn(submit()))
    result = run_agent_cached(
        again, excerpt(), agent_prompt(), toolbox(github, tool_cache), cache, refresh=True
    )

    assert result.trajectory is not None
    assert result.trajectory.tool_calls[0].cached
    assert github.calls == []
    assert len(again.requests) == 2


def test_prompt_version_covers_tools_and_limits() -> None:
    base = load_prompt()
    every = build_agent_prompt(base, list(TOOLS.values()), 6, 30_000, OUTPUT_CAP_TOKENS)
    fewer = build_agent_prompt(base, [TOOLS["read_log"]], 6, 30_000, OUTPUT_CAP_TOKENS)
    steps = build_agent_prompt(base, list(TOOLS.values()), 3, 30_000, OUTPUT_CAP_TOKENS)
    cap = build_agent_prompt(base, list(TOOLS.values()), 6, 30_000, 1000)

    versions = {every.version, fewer.version, steps.version, cap.version}
    assert len(versions) == 4
    assert all(v.startswith("agent-") for v in versions)
    assert base.version not in versions
    assert every.system.startswith(base.system)  # the Phase 1 v3 instruction, unchanged
    assert fewer.tools == ["read_log"]


# --- fix_sha ----------------------------------------------------------------------------------


def test_fix_sha_never_reaches_a_tool_a_prompt_or_github(tmp_path: Path) -> None:
    conn = connect(tmp_path / "cidoc.db")
    save_repo(conn, RepoInfo(REPO, 300, "main"), NOW)
    insert_failure(
        conn,
        FailedJob(
            job_id=JOB_ID, repo=REPO, run_id=RUN_ID, workflow_name="CI",
            workflow_path=WORKFLOW, event="push", head_branch="main", head_sha=SHA,
            fix_sha=FIX_SHA, failed_steps=["Run flutter build apk"], runner_os="Linux",
            created_at="2026-10-20T00:00:00Z", log_path="logs/o__a/7.log.gz",
            log_bytes=1, redactions=0,
        ),
    )  # fmt: skip
    ctx = load_context(conn, JOB_ID, LOG)
    created = "2026-10-20T10:00:00Z"
    github = FakeGitHub(
        {
            contents("pubspec.yaml"): file_entry("name: app\n"),
            contents(""): [{"type": "file", "name": "pubspec.yaml", "size": 10}],
            contents(WORKFLOW): file_entry("on: push\n"),
            f"/repos/{REPO}/actions/runs/{RUN_ID}/jobs": {"jobs": [{"id": 8, "name": "b"}]},
            f"/repos/{REPO}/actions/runs/{RUN_ID}": {"workflow_id": 5, "created_at": created},
            f"/repos/{REPO}/actions/workflows/5/runs": {"workflow_runs": []},
            f"/repos/{REPO}/commits/{SHA}": {"commit": {"message": "m"}, "files": []},
        }
    )
    every_tool: list[Call] = [
        READ_PUBSPEC,
        ("list_dir", {}),
        ("read_workflow", {}),
        ("search_log", {"pattern": "error"}),
        ("read_log", {"start_line": 1, "end_line": 6}),
        ("sibling_jobs", {}),
        ("previous_run", {}),
        ("commit_changes", {}),
    ]
    limits = AgentLimits(max_steps=len(every_tool))
    provider = ScriptedToolProvider(turn(*every_tool), turn(submit()))

    result = run_agent(
        provider, excerpt(), agent_prompt(limits), ToolBox(ToolEnv(ctx, github)), limits
    )

    assert result.ok
    assert result.trajectory is not None
    assert all(r.ok for r in result.trajectory.tool_calls), result.trajectory.tool_calls
    sent = json.dumps([[r["system"], [repr(m) for m in r["messages"]]] for r in provider.requests])
    for needle in (FIX_SHA, FIX_SHA[:12], FIX_SHA[:7]):
        assert needle not in sent
        assert needle not in json.dumps(github.calls)
        assert needle not in repr(result)
