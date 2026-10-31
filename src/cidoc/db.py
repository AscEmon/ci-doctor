"""SQLite storage for mined repos and failed CI jobs.

Schema versions are tracked with PRAGMA user_version. To change the schema,
append a new script to MIGRATIONS; never edit one that has already shipped.
"""

import sqlite3
from collections.abc import Sequence
from pathlib import Path

DB_FILENAME = "cidoc.db"

MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE repos (
        full_name      TEXT PRIMARY KEY,
        stars          INTEGER NOT NULL,
        default_branch TEXT NOT NULL,
        discovered_at  TEXT NOT NULL
    );

    CREATE TABLE failures (
        job_id        INTEGER PRIMARY KEY,
        repo          TEXT NOT NULL REFERENCES repos(full_name),
        run_id        INTEGER NOT NULL,
        workflow_name TEXT NOT NULL,
        workflow_path TEXT NOT NULL,
        event         TEXT NOT NULL,
        head_branch   TEXT,
        head_sha      TEXT NOT NULL,
        fix_sha       TEXT,
        failed_steps  TEXT NOT NULL,  -- JSON array of step names
        runner_os     TEXT,
        created_at    TEXT NOT NULL,
        log_path      TEXT NOT NULL,
        log_bytes     INTEGER NOT NULL
    );

    CREATE INDEX failures_repo_idx ON failures(repo);
    """,
    """
    -- Number of secrets masked in the stored log.
    ALTER TABLE failures ADD COLUMN redactions INTEGER NOT NULL DEFAULT 0;
    """,
    """
    -- Failed runs whose jobs have all been processed (stored or log expired).
    CREATE TABLE mined_runs (
        run_id   INTEGER PRIMARY KEY,
        repo     TEXT NOT NULL REFERENCES repos(full_name),
        mined_at TEXT NOT NULL
    );
    """,
)


def schema_version(conn: sqlite3.Connection) -> int:
    version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    return version


def migrate(conn: sqlite3.Connection, migrations: Sequence[str] = MIGRATIONS) -> int:
    """Apply pending migrations in order, each in its own transaction. Returns the new version."""
    current = schema_version(conn)
    for version, script in enumerate(migrations[current:], start=current + 1):
        try:
            conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;")
        except sqlite3.Error:
            conn.rollback()
            raise
    return schema_version(conn)


def connect(path: Path) -> sqlite3.Connection:
    """Open the database at `path`, creating it and applying migrations as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn
