import gzip
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.prune import apply_prune, failure_prune_reason, plan_prune
from cidoc.miner.rules import RepoExcludes
from cidoc.miner.runs import FailedJob, insert_failure, log_relpath, mark_run_mined

NOW = datetime(2026, 10, 31, tzinfo=UTC)
CI = (".github/workflows/ci.yml", "CI")


def failure(job_id: int, repo: str, workflow: tuple[str, str] = CI) -> FailedJob:
    path, name = workflow
    return FailedJob(
        job_id=job_id,
        repo=repo,
        run_id=job_id // 10,
        workflow_name=name,
        workflow_path=path,
        event="push",
        head_branch="main",
        head_sha="abc",
        fix_sha=None,
        failed_steps=["Build"],
        runner_os="Linux",
        created_at="2026-10-20T00:00:00Z",
        log_path=log_relpath(repo, job_id).as_posix(),
        log_bytes=10,
        redactions=0,
    )


def populate(conn: sqlite3.Connection, data_dir: Path) -> None:
    """Two kept jobs, one per prune reason, and an excluded repo with a mined-run marker."""
    for name in ("o/app", "flutter/flutter", "acme/listed"):
        save_repo(conn, RepoInfo(name, 300, "main"), NOW)
    jobs = [
        failure(11, "o/app"),
        failure(12, "o/app"),
        failure(13, "o/app", ("dynamic/pages/pages-build-deployment", "pages build")),
        failure(14, "o/app", (".github/workflows/codeql.yml", "CodeQL")),
        failure(15, "o/app", (".github/workflows/deps.yml", "Dependabot auto-merge")),
        failure(21, "flutter/flutter", ("dynamic/x", "Copilot")),  # repo reason wins
        failure(22, "flutter/flutter"),
        failure(31, "acme/listed"),
    ]
    for job in jobs:
        insert_failure(conn, job)
        log = data_dir / job.log_path
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(gzip.compress(b"error: boom\n"))
    mark_run_mined(conn, "flutter/flutter", 2, NOW)
    mark_run_mined(conn, "o/app", 1, NOW)


EXCLUDES = RepoExcludes(repos=frozenset({"acme/listed"}))


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(tmp_path / DB_FILENAME)
    populate(db, tmp_path)
    yield db
    db.close()


def job_ids(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute("SELECT job_id FROM failures ORDER BY job_id")]


@pytest.mark.parametrize(
    ("repo", "workflow", "expected"),
    [
        ("o/app", CI, None),
        ("o/app", ("dynamic/pages/x", "pages"), "dynamic workflow"),
        ("o/app", (".github/workflows/a.yml", "Copilot setup"), "copilot workflow"),
        ("Dart-Lang/sdk", ("dynamic/x", "CodeQL"), "excluded org dart-lang/*"),
    ],
)
def test_failure_prune_reason_reuses_mining_rules(
    repo: str, workflow: tuple[str, str], expected: str | None
) -> None:
    path, name = workflow
    assert failure_prune_reason(RepoExcludes(), repo, path, name) == expected


def test_plan_counts_per_reason(conn: sqlite3.Connection) -> None:
    plan = plan_prune(conn, EXCLUDES)

    assert sorted(f.job_id for f in plan.failures) == [13, 14, 15, 21, 22, 31]
    assert dict(plan.counts_by_reason()) == {
        "excluded org flutter/*": 2,
        "listed in exclude_repos.txt": 1,
        "dynamic workflow": 1,
        "codeql workflow": 1,
        "dependabot workflow": 1,
    }
    assert plan.repos == [
        ("acme/listed", "listed in exclude_repos.txt"),
        ("flutter/flutter", "excluded org flutter/*"),
    ]


def test_planning_changes_nothing(conn: sqlite3.Connection, tmp_path: Path) -> None:
    plan_prune(conn, EXCLUDES)

    assert job_ids(conn) == [11, 12, 13, 14, 15, 21, 22, 31]
    assert len(list((tmp_path / "logs").rglob("*.log.gz"))) == 8


def test_apply_deletes_rows_and_logs(conn: sqlite3.Connection, tmp_path: Path) -> None:
    result = apply_prune(conn, tmp_path, plan_prune(conn, EXCLUDES))

    assert (result.failures, result.log_files, result.mined_runs, result.repos) == (6, 6, 1, 2)
    assert job_ids(conn) == [11, 12]
    assert [r[0] for r in conn.execute("SELECT full_name FROM repos")] == ["o/app"]
    assert [r[0] for r in conn.execute("SELECT repo FROM mined_runs")] == ["o/app"]
    logs = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.log.gz"))
    assert logs == ["logs/o__app/11.log.gz", "logs/o__app/12.log.gz"]
    # Emptied per-repo log directories are removed too.
    assert not (tmp_path / "logs" / "flutter__flutter").exists()


def test_apply_tolerates_missing_log_files(conn: sqlite3.Connection, tmp_path: Path) -> None:
    (tmp_path / log_relpath("acme/listed", 31)).unlink()

    result = apply_prune(conn, tmp_path, plan_prune(conn, EXCLUDES))

    assert result.failures == 6
    assert result.log_files == 5


def test_apply_never_deletes_outside_data_dir(conn: sqlite3.Connection, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "keep.txt"
    outside.write_text("x", encoding="utf-8")
    conn.execute("UPDATE failures SET log_path = '../keep.txt' WHERE job_id = 13")

    apply_prune(conn, data_dir, plan_prune(conn, EXCLUDES))

    assert outside.exists()


def test_rerun_after_apply_has_nothing_to_do(conn: sqlite3.Connection, tmp_path: Path) -> None:
    apply_prune(conn, tmp_path, plan_prune(conn, EXCLUDES))

    assert plan_prune(conn, EXCLUDES).empty


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    data_dir = tmp_path / "data"
    db = connect(data_dir / DB_FILENAME)
    populate(db, data_dir)
    db.close()
    (data_dir / "exclude_repos.txt").write_text("acme/listed\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")  # fake, tests only
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    yield data_dir
    get_settings.cache_clear()


def stored_job_count(data_dir: Path) -> int:
    db = connect(data_dir / DB_FILENAME)
    try:
        count: int = db.execute("SELECT count(*) FROM failures").fetchone()[0]
    finally:
        db.close()
    return count


def test_cli_prune_is_dry_run_by_default(cli_env: Path) -> None:
    result = CliRunner().invoke(app, ["mine", "prune"], terminal_width=200)

    assert result.exit_code == 0, result.output
    assert "Would remove 6 failed jobs" in result.output
    assert "excluded org flutter/*" in result.output
    assert "Dry run: nothing deleted" in result.output
    assert stored_job_count(cli_env) == 8
    assert len(list(cli_env.rglob("*.log.gz"))) == 8


def test_cli_prune_apply_deletes_and_warns_about_labels(cli_env: Path) -> None:
    label = {
        "job_id": 22,
        "repo": "flutter/flutter",
        "run_id": 2,
        "category": "other",
        "root_cause": "x",
        "evidence_lines": [1],
        "confidence": 1,
        "log_path": "logs/flutter__flutter/22.log.gz",
        "labeled_at": "2026-10-30T00:00:00Z",
    }
    (cli_env / "labels.jsonl").write_text(json.dumps(label) + "\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["mine", "prune", "--apply"], terminal_width=200)

    assert result.exit_code == 0, result.output
    assert "1 of these jobs are in labels.jsonl" in result.output
    assert "Deleted 6 failures, 6 log files, 1 mined-run markers, 2 repos." in result.output
    assert stored_job_count(cli_env) == 2
    # labels.jsonl is hand-made work and is never modified by prune.
    assert json.loads((cli_env / "labels.jsonl").read_text(encoding="utf-8")) == label

    again = CliRunner().invoke(app, ["mine", "prune"], terminal_width=200)
    assert "Nothing to prune." in again.output


# --- Out-of-scope jobs (event, action, aggregator) ---------------------------------------

SEMANTIC_PR_LOG = (
    "##[group]Run amannn/action-semantic-pull-request@v5\n"
    "##[endgroup]\n"
    "No release type found in pull request title\n"
    "##[error]No release type found in pull request title\n"
)


def add_failure(
    conn: sqlite3.Connection,
    data_dir: Path,
    job_id: int,
    log: bytes,
    event: str = "push",
    failed_steps: list[str] | None = None,
) -> None:
    job = replace(failure(job_id, "o/app"), event=event, failed_steps=failed_steps or ["Check"])
    insert_failure(conn, job)
    path = data_dir / job.log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(log)


def test_prune_removes_out_of_scope_jobs(conn: sqlite3.Connection, tmp_path: Path) -> None:
    add_failure(conn, tmp_path, 41, gzip.compress(b"x\n"), event="issue_comment")
    add_failure(conn, tmp_path, 42, gzip.compress(SEMANTIC_PR_LOG.encode()))
    add_failure(conn, tmp_path, 43, b"not a gzip file")

    plan = plan_prune(conn, EXCLUDES, tmp_path)

    reasons = {f.job_id: f.reason for f in plan.failures}
    assert reasons[41] == "issue bot (issue_comment event)"
    assert reasons[42] == "PR hygiene (amannn/action-semantic-pull-request)"
    assert 43 not in reasons  # unreadable log: kept, no crash
    assert 11 not in reasons and 12 not in reasons  # ordinary build failures kept


def test_log_rules_need_data_dir(conn: sqlite3.Connection, tmp_path: Path) -> None:
    add_failure(conn, tmp_path, 42, gzip.compress(SEMANTIC_PR_LOG.encode()))

    without_logs = {f.job_id for f in plan_prune(conn, EXCLUDES).failures}

    assert 42 not in without_logs
