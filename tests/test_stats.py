import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.runs import FailedJob, insert_failure
from cidoc.miner.stats import MatrixDuplicate, RepoStats, collect_stats, pct

NOW = datetime(2026, 10, 31, tzinfo=UTC)


def failure(
    job_id: int,
    repo: str,
    steps: list[str],
    runner_os: str | None = "Linux",
    fix_sha: str | None = None,
    redactions: int = 0,
) -> FailedJob:
    return FailedJob(
        job_id=job_id,
        repo=repo,
        run_id=job_id // 10,
        workflow_name="CI",
        workflow_path=".github/workflows/ci.yml",
        event="push",
        head_branch="main",
        head_sha="abc",
        fix_sha=fix_sha,
        failed_steps=steps,
        runner_os=runner_os,
        created_at="2026-10-20T00:00:00Z",
        log_path=f"logs/x/{job_id}.log.gz",
        log_bytes=10,
        redactions=redactions,
    )


def populate(conn: sqlite3.Connection) -> None:
    for name in ("o/a", "o/b", "o/empty"):
        save_repo(conn, RepoInfo(name, 300, "main"), NOW)
    insert_failure(conn, failure(11, "o/a", ["Build APK"], fix_sha="f1", redactions=2))
    insert_failure(conn, failure(12, "o/a", ["Build APK", "Test"], runner_os="macOS"))
    insert_failure(conn, failure(13, "o/a", ["Test"], fix_sha="f2"))
    insert_failure(conn, failure(21, "o/b", [], runner_os=None, fix_sha="f3"))


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(tmp_path / "cidoc.db")
    yield db
    db.close()


def test_collect_stats(conn: sqlite3.Connection) -> None:
    populate(conn)

    stats = collect_stats(conn)

    assert (stats.repos, stats.jobs, stats.with_fix, stats.redactions) == (3, 4, 3, 2)
    assert stats.per_repo == [
        RepoStats("o/a", 3, 2),
        RepoStats("o/b", 1, 1),
        RepoStats("o/empty", 0, 0),
    ]
    assert stats.per_runner_os == [("Linux", 2), ("(unknown)", 1), ("macOS", 1)]
    assert stats.per_failed_step == [("Build APK", 2), ("Test", 2)]


def test_run_level_stats_and_matrix_duplicates(conn: sqlite3.Connection) -> None:
    populate(conn)  # run 1 (o/a): jobs 11, 12, 13; run 2 (o/b): job 21
    # Same run id in another repo must not merge with o/a's run 1.
    insert_failure(conn, failure(14, "o/b", ["Build APK"]))
    insert_failure(conn, failure(15, "o/b", ["Build APK"]))

    stats = collect_stats(conn)

    assert stats.runs == 3
    assert stats.jobs_per_run == 2.0
    # 11 and 12 both broke first at "Build APK" (12 also failed "Test" afterwards);
    # 13 broke first at "Test", so it is not a repeat. 21 has no step info.
    assert stats.matrix_duplicates == [
        MatrixDuplicate("o/a", 1, "Build APK", 2),
        MatrixDuplicate("o/b", 1, "Build APK", 2),
    ]
    assert stats.duplicate_jobs == 2


def test_collect_stats_limits_steps(conn: sqlite3.Connection) -> None:
    populate(conn)

    assert collect_stats(conn, top_steps=1).per_failed_step == [("Build APK", 2)]


def test_collect_stats_on_empty_db(conn: sqlite3.Connection) -> None:
    stats = collect_stats(conn)

    assert (stats.repos, stats.jobs, stats.with_fix, stats.redactions) == (0, 0, 0, 0)
    assert stats.per_repo == []
    assert stats.per_runner_os == []
    assert stats.per_failed_step == []
    assert (stats.runs, stats.jobs_per_run, stats.matrix_duplicates) == (0, 0.0, [])


def test_pct_handles_zero() -> None:
    assert pct(1, 4) == 25.0
    assert pct(0, 0) == 0.0


def test_stats_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-value")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    db = connect(tmp_path / "data" / DB_FILENAME)
    populate(db)
    db.close()

    result = CliRunner().invoke(app, ["mine", "stats"], terminal_width=200)
    get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    assert "4 failed jobs across 3 repos; 3 (75.0%) have a fix_sha" in result.output
    assert "Build APK" in result.output
    assert "2 unique runs; 2.00 failed jobs per run on average." in result.output
    assert "1 runs have likely matrix duplicates" in result.output
    assert "1 jobs (25.0%) are repeats" in result.output
    assert "Likely matrix duplicates" in result.output
    assert "o/empty" in result.output
