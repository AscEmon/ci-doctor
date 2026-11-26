"""`cidoc eval --agent`: run files, metrics, cost estimates, caching. No network."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cidoc.agent.budget import BudgetedToolProvider, estimate_item_calls
from cidoc.agent.runtime import agent_setup
from cidoc.agent.trajectory import ToolCallRecord, Trajectory
from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.costguard import BudgetExceededError, CostGuard
from cidoc.analyze.preprocess import preprocess
from cidoc.analyze.prompt import load_prompt
from cidoc.analyze.schema import AgentDiagnosis
from cidoc.cli import app
from cidoc.config import Settings
from cidoc.eval.metrics import EvalItem, compute_metrics
from cidoc.eval.results import RUNS_DIRNAME, load_run
from cidoc.eval.runner import load_cases
from cidoc.llm.base import Usage, UserMessage
from cidoc.llm.factory import create_tool_provider
from cidoc.llm.gemini import GeminiProvider
from cidoc.llm.pricing import Price
from tests.agent_fakes import FakeGitHub, ScriptedToolProvider, submit, turn
from tests.fakes import VALID_DIAGNOSIS
from tests.test_eval import opener, setup_jobs

READ_LOG = ("read_log", {"start_line": 4, "end_line": 6})
DEPENDENCY = {**VALID_DIAGNOSIS, "category": "dependency_resolution"}


def run(*args: str) -> Any:
    return CliRunner().invoke(app, ["eval", *args])


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> FakeGitHub:
    fake = FakeGitHub()
    monkeypatch.setattr("cidoc.cli.open_tools_github", lambda token, base_url: fake)
    return fake


def script(
    monkeypatch: pytest.MonkeyPatch, *turns: Any, model: str = "fake-model"
) -> ScriptedToolProvider:
    scripted = ScriptedToolProvider(*turns, model=model)

    def create(settings: Settings, model: str | None = None) -> ScriptedToolProvider:
        return scripted

    monkeypatch.setattr("cidoc.cli.create_tool_provider", create)
    return scripted


def three_items(monkeypatch: pytest.MonkeyPatch, model: str = "fake-model") -> ScriptedToolProvider:
    """Job 1: one tool then submit; job 2: no tool; job 3: two tools (one turn)."""
    return script(
        monkeypatch,
        turn(READ_LOG),
        turn(submit()),
        turn(submit()),
        turn(READ_LOG, ("search_log", {"pattern": "version solving"})),
        turn(("submit_diagnosis", DEPENDENCY)),
        model=model,
    )


def dry_rows(output: str) -> dict[str, str]:
    """The dry-run table as {row name: value}."""
    rows = {}
    for line in output.splitlines():
        cells = [c.strip() for c in line.split("│")]
        if len(cells) >= 3:
            rows[cells[1]] = cells[2]
    return rows


def run_files(data_dir: Path) -> list[Path]:
    return sorted((data_dir / RUNS_DIRNAME).glob("*.json"))


def test_agent_run_file_records_agent_tools_and_trajectories(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    three_items(monkeypatch)

    result = run("--agent", "--tools", "read_log,search_log", "--max-steps", "4")

    assert result.exit_code == 0, result.output
    (path,) = run_files(data_dir)
    assert path.name.endswith("_agent.json")
    saved = load_run(path)
    assert saved.info.agent is True
    assert saved.info.tools == ["search_log", "read_log"]
    assert saved.info.max_steps == 4
    assert saved.info.prompt_version is not None
    assert saved.info.prompt_version.startswith("agent-")
    assert saved.info.model == "fake-model"
    trajectories = [item.result.trajectory for item in saved.items]
    assert [len(t.tool_calls) for t in trajectories if t] == [1, 0, 2]
    assert saved.metrics["mean_tool_calls"] == pytest.approx(1.0)
    assert saved.metrics["used_tools_rate"] == pytest.approx(2 / 3)
    assert saved.metrics["tool_usage"] == {"read_log": 2, "search_log": 1}
    assert saved.metrics["hit_max_steps"] == 0
    assert saved.metrics["mean_model_calls"] == pytest.approx(5 / 3)
    assert saved.metrics["accuracy"] == 1.0
    assert "tool calls / item" in result.output
    assert "Tool usage: read_log 2, search_log 1" in result.output
    assert "[1 tool]" in result.output


def test_an_agent_rerun_makes_no_model_or_github_calls(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    three_items(monkeypatch)
    assert run("--agent").exit_code == 0
    github.calls.clear()

    again = script(monkeypatch)
    result = run("--agent")

    assert result.exit_code == 0, result.output
    assert again.requests == []
    assert github.calls == []
    assert result.output.count("(cached)") == 3


def test_items_that_hit_max_steps_are_counted(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    script(
        monkeypatch,
        turn(READ_LOG), turn(submit()),
        turn(submit()),
        turn(("submit_diagnosis", DEPENDENCY)),
    )  # fmt: skip

    result = run("--agent", "--max-steps", "1")

    assert result.exit_code == 0, result.output
    saved = load_run(run_files(data_dir)[0])
    assert saved.metrics["hit_max_steps"] == 1


def test_dry_run_estimates_every_model_call_at_the_worst_case(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)

    result = run("--agent", "--max-steps", "3", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run: gemini-3.5-flash-lite (agent)" in result.output
    setup = agent_setup(load_prompt(), None, 3)
    logs = [
        preprocess(c.source.text, c.source.failed_steps)
        for c in load_cases(data_dir, opener(data_dir)).cases
    ]
    expected = sum(len(estimate_item_calls(setup.prompt, log, setup.limits)) for log in logs)
    assert 3 < expected <= 3 * 4
    assert dry_rows(result.output)["model calls (worst case)"] == str(expected)
    assert "(4 model calls)" in result.output
    assert not run_files(data_dir)


def test_dry_run_counts_cached_agent_items_as_free(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    three_items(monkeypatch, model="gemini-3.5-flash-lite")  # the model dry-run plans for
    assert run("--agent").exit_code == 0

    result = run("--agent", "--dry-run")

    assert result.exit_code == 0, result.output
    rows = dry_rows(result.output)
    assert rows["cached (free)"] == "3"
    assert rows["model calls (worst case)"] == "0"


def test_max_cost_does_not_start_an_item_whose_worst_case_does_not_fit(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    monkeypatch.setenv("CIDOC_PRICES", '{"gemini-3.5-flash-lite": {"input": 1.0, "output": 1.0}}')
    provider = script(monkeypatch, turn(submit()))

    result = run("--agent", "--max-cost", "0.000001")

    assert result.exit_code == 1, result.output
    assert provider.requests == []
    assert "would exceed --max-cost" in result.output
    saved = load_run(run_files(data_dir)[0])
    assert saved.info.complete is False


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--agent", "--model", "regex-baseline"], "--agent needs an LLM model"),
        (["--tools", "read_log"], "only apply to --agent"),
        (["--agent", "--tools", "nope"], "unknown tools: nope"),
    ],
)
def test_agent_option_errors(data_dir: Path, args: list[str], message: str) -> None:
    setup_jobs(data_dir)

    result = run(*args)

    assert result.exit_code == 2
    assert message in result.output


def test_the_tool_provider_uses_the_configured_rpm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    settings = Settings(gemini_api_key="k", cidoc_rpm=3)  # type: ignore[arg-type]

    provider = create_tool_provider(settings)

    assert isinstance(provider, GeminiProvider)
    assert provider._limiter.rpm == 3


# --- metrics ------------------------------------------------------------------------------


def agent_item(job_id: int, tools: list[str], hit: bool = False) -> EvalItem:
    records = [
        ToolCallRecord(n, 1, name, {}, True, False, 10, 3, 0.1, "")
        for n, name in enumerate(tools, 1)
    ]
    diagnosis = AgentDiagnosis.model_validate(VALID_DIAGNOSIS)
    calls = [Usage(1000, 50, 0.5)] * (len(tools) + 1)
    trajectory = Trajectory(tools=["a"], max_steps=2, max_input_tokens=30_000,
                            tool_calls=records, hit_max_steps=hit)  # fmt: skip
    result = AnalysisResult("m", diagnosis, None, calls, trajectory=trajectory)
    return EvalItem(job_id, "dart_compile", 2, [5], result)


def test_agent_metrics() -> None:
    items = [
        agent_item(1, []),
        agent_item(2, ["read_log", "read_file"], hit=True),
        agent_item(3, ["read_log"]),
    ]

    metrics = compute_metrics(items, price=None)

    assert metrics.agent_items == 3
    assert metrics.mean_tool_calls == pytest.approx(1.0)
    assert metrics.used_tools_rate == pytest.approx(2 / 3)
    assert metrics.tool_usage == {"read_log": 2, "read_file": 1}
    assert metrics.hit_max_steps == 1
    assert metrics.mean_model_calls == pytest.approx(2.0)
    assert metrics.mean_input_tokens == pytest.approx(2000.0)


def test_phase_1_runs_have_no_agent_metrics() -> None:
    diagnosis = AgentDiagnosis.model_validate(VALID_DIAGNOSIS)
    result = AnalysisResult("m", diagnosis, None, [Usage(900, 40, 0.5)])

    metrics = compute_metrics([EvalItem(1, "dart_compile", 2, [5], result)], price=None)

    assert metrics.agent_items == 0
    assert metrics.mean_tool_calls is None
    assert metrics.hit_max_steps is None
    assert metrics.mean_model_calls == 1.0


# --- cost guard --------------------------------------------------------------------------


def test_the_guard_checks_and_charges_every_agent_model_call() -> None:
    usage = Usage(1_000_000, 0, 0.1)  # $1 at $1 per million input tokens
    scripted = ScriptedToolProvider(turn(READ_LOG, usage=usage), turn(submit()))
    guard = CostGuard(max_cost=1.0001, price=Price(input=1.0, output=1.0))
    guarded = BudgetedToolProvider(scripted, guard)

    guarded.generate_with_tools("s", [UserMessage("u")], [])
    with pytest.raises(BudgetExceededError):  # ~$0.0005 more (output allowance) won't fit
        guarded.generate_with_tools("s", [UserMessage("u")], [])

    assert guard.calls == 1
    assert guard.spent == pytest.approx(1.0)
    assert len(scripted.requests) == 1  # the second call was never sent
