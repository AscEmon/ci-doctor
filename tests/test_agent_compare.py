"""--show and --compare with agent runs: Phase 1 vs agent on one split. No network."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cidoc.cli import app
from cidoc.config import Settings
from cidoc.eval.compare import compare_runs
from cidoc.eval.results import RUNS_DIRNAME, load_run
from cidoc.labeling.labels import LABELS_FILENAME, load_labels, newest_labels
from tests.agent_fakes import FakeGitHub
from tests.fakes import VALID_DIAGNOSIS, ScriptedProvider, ok
from tests.test_agent_eval import three_items
from tests.test_eval import setup_jobs


def run(*args: str) -> Any:
    return CliRunner().invoke(app, ["eval", *args])


def phase1_run(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Phase 1 on the same three jobs: right, right, wrong (job 3 is dependency_resolution)."""
    scripted = ScriptedProvider(*[ok(VALID_DIAGNOSIS) for _ in range(3)])

    def create(settings: Settings, model: str | None = None) -> ScriptedProvider:
        return scripted

    monkeypatch.setattr("cidoc.cli.create_provider", create)
    assert run("--model", "fake-model").exit_code == 0
    (path,) = sorted((data_dir / RUNS_DIRNAME).glob("*.json"))
    return path


def agent_run(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    three_items(monkeypatch)
    assert run("--agent").exit_code == 0
    (path,) = sorted((data_dir / RUNS_DIRNAME).glob("*_agent.json"))
    return path


def test_compare_phase_1_with_an_agent_run(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    a = phase1_run(data_dir, monkeypatch)
    b = agent_run(data_dir, monkeypatch)

    result = run("--compare", str(a), str(b))

    assert result.exit_code == 0, result.output
    out = result.output
    assert "fake-model agent (max 6 steps, tools read_file," in out
    assert "B is an agent run, A is Phase 1" in out
    assert "only B right 1" in out
    for label in ("model calls per item", "tool calls per item", "items that hit max_steps"):
        assert label in out
    assert "3 shared items" in out


def test_compare_marks_tool_calls_as_neither_better_nor_worse(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    a = load_run(phase1_run(data_dir, monkeypatch))
    b = load_run(agent_run(data_dir, monkeypatch))
    labels = newest_labels(load_labels(data_dir / LABELS_FILENAME))

    comparison = compare_runs(a, b, lambda model: None, labels)

    deltas = {d.key: d for d in comparison.deltas}
    assert deltas["mean_model_calls"].a == pytest.approx(1.0)
    assert deltas["mean_model_calls"].b == pytest.approx(5 / 3)
    assert deltas["mean_model_calls"].improved is False
    assert deltas["mean_tool_calls"].a is None
    assert deltas["mean_tool_calls"].b == pytest.approx(1.0)
    assert deltas["accuracy"].improved is True


def test_show_job_prints_the_trajectory_of_an_agent_item(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    path = agent_run(data_dir, monkeypatch)

    result = run("--show", str(path), "--job", "3")

    assert result.exit_code == 0, result.output
    out = result.output
    assert "trajectory:" in out
    assert "1. read_log(start_line=4, end_line=6)" in out
    assert '2. search_log(pattern="version solving"' in out
    assert "L5|" in out  # output preview
    assert "2 tool calls (max 6); 2 model calls" in out


def test_show_without_job_lists_the_tools_each_item_used(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, github: FakeGitHub
) -> None:
    setup_jobs(data_dir)
    path = agent_run(data_dir, monkeypatch)

    result = run("--show", str(path))

    assert result.exit_code == 0, result.output
    assert "tools:     read_log, search_log" in result.output
    assert "tools:     none" in result.output
    assert "trajectory:" not in result.output


def test_show_on_a_phase_1_run_is_unchanged(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_jobs(data_dir)
    path = phase1_run(data_dir, monkeypatch)

    result = run("--show", str(path), "--job", "1")

    assert result.exit_code == 0, result.output
    assert "tools:" not in result.output
    assert "trajectory" not in result.output
