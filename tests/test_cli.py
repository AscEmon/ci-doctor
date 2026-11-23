from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cidoc import cli
from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.miner.runs import MineResult

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == version("ci-doctor")


def test_mine_runs_prints_skipped_run_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")  # fake, tests only
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    results = {"acme/ok": MineResult(stored=2, skipped_runs=0), "acme/bad": MineResult(1, 3)}
    monkeypatch.setattr(cli, "open_github", lambda: nullcontext())
    monkeypatch.setattr(cli, "mine_repo", lambda _c, _db, _d, name, *_a: results[name])

    result = runner.invoke(
        app, ["mine", "runs", "--repo", "acme/ok", "--repo", "acme/bad"], terminal_width=200
    )
    get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    ok, bad = result.output.split("acme/bad")
    assert "stored 2 new failed jobs" in ok
    assert "skipped" not in ok
    assert "stored 1 new failed jobs" in bad
    assert "skipped 3 runs after errors" in bad


def test_mine_runs_skips_excluded_repos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "exclude_repos.txt").write_text("acme/listed\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")  # fake, tests only
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    mined: list[str] = []

    def fake_mine_repo(_c: object, _db: object, _d: object, name: str, *_a: object) -> MineResult:
        mined.append(name)
        return MineResult(stored=0, skipped_runs=0)

    monkeypatch.setattr(cli, "open_github", lambda: nullcontext())
    monkeypatch.setattr(cli, "mine_repo", fake_mine_repo)

    args = ["mine", "runs", "--repo", "flutter/flutter", "--repo", "acme/listed", "--repo", "o/app"]
    result = runner.invoke(app, args, terminal_width=200)
    get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    assert mined == ["o/app"]
    assert "excluded flutter/flutter (excluded org flutter/*)" in result.output
    assert "excluded acme/listed (listed in exclude_repos.txt)" in result.output


def test_malformed_exclude_file_exits_with_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "exclude_repos.txt").write_text("not a repo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")  # fake, tests only
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()

    result = runner.invoke(app, ["mine", "runs"], terminal_width=200)
    get_settings.cache_clear()

    assert result.exit_code == 2
    assert "exclude_repos.txt line 1" in result.output
