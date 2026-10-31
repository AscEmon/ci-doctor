import sqlite3
from pathlib import Path

import pytest

from cidoc.db import MIGRATIONS, connect, migrate, schema_version


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_connect_creates_db_and_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "cidoc.db"

    conn = connect(db_path)

    assert db_path.exists()
    assert schema_version(conn) == len(MIGRATIONS)
    assert table_columns(conn, "repos") == [
        "full_name",
        "stars",
        "default_branch",
        "discovered_at",
    ]
    assert table_columns(conn, "failures") == [
        "job_id",
        "repo",
        "run_id",
        "workflow_name",
        "workflow_path",
        "event",
        "head_branch",
        "head_sha",
        "fix_sha",
        "failed_steps",
        "runner_os",
        "created_at",
        "log_path",
        "log_bytes",
        "redactions",
    ]
    conn.close()


def test_reconnect_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "cidoc.db"
    connect(db_path).close()

    conn = connect(db_path)

    assert schema_version(conn) == len(MIGRATIONS)
    conn.close()


def test_migrate_applies_only_pending_scripts() -> None:
    conn = sqlite3.connect(":memory:")
    assert migrate(conn, ["CREATE TABLE a (x INTEGER);"]) == 1

    assert migrate(conn, ["CREATE TABLE a (x INTEGER);", "CREATE TABLE b (y INTEGER);"]) == 2

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert tables == {"a", "b"}


def test_mined_runs_table_exists(tmp_path: Path) -> None:
    conn = connect(tmp_path / "cidoc.db")

    assert table_columns(conn, "mined_runs") == ["run_id", "repo", "mined_at"]
    conn.close()


def test_failed_migration_rolls_back() -> None:
    conn = sqlite3.connect(":memory:")
    bad = "CREATE TABLE ok (x INTEGER);\nCREATE TABLE ok (x INTEGER);"

    with pytest.raises(sqlite3.OperationalError):
        migrate(conn, [bad])

    assert schema_version(conn) == 0
    assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0


def test_failures_require_known_repo(tmp_path: Path) -> None:
    conn = connect(tmp_path / "cidoc.db")

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO failures (job_id, repo, run_id, workflow_name, workflow_path, event,"
            " head_sha, failed_steps, created_at, log_path, log_bytes)"
            " VALUES (1, 'nobody/nothing', 1, 'ci', '.github/workflows/ci.yml', 'push',"
            " 'abc', '[]', '2026-10-01T00:00:00Z', 'x.log.gz', 0)"
        )
    conn.close()
