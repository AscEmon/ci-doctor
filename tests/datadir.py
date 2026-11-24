"""Build a throwaway data dir (database, stored logs, labels) for CLI tests."""

import gzip
from datetime import UTC, datetime
from pathlib import Path

from cidoc.db import DB_FILENAME, connect
from cidoc.labeling.labels import LABELS_FILENAME, Label, append_label
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.runs import FailedJob, insert_failure

TS = "2026-10-01T12:34:56.1234567Z "
NOW = datetime(2026, 10, 31, tzinfo=UTC)
LOG_LINES = (
    "##[group]Run flutter build apk",  # 1
    "flutter build apk",  # 2
    "##[endgroup]",  # 3
    "Compiling lib/main.dart",  # 4
    "lib/main.dart:3:5: Error: Undefined name 'foo'.",  # 5
    "##[error]Process completed with exit code 1.",  # 6
)


def log_text(*extra: str) -> str:
    """LOG_LINES with `extra` lines inserted before the error (line 5 onwards)."""
    return "".join(f"{TS}{line}\n" for line in (*LOG_LINES[:4], *extra, *LOG_LINES[4:]))


def store_job(data_dir: Path, job_id: int, text: str, in_db: bool = True) -> str:
    """Write the job's gzipped log (and its database row); returns the log path."""
    log_path = f"logs/o__a/{job_id}.log.gz"
    (data_dir / "logs" / "o__a").mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / log_path, "wt", encoding="utf-8", newline="") as fh:
        fh.write(text)
    if in_db:
        conn = connect(data_dir / DB_FILENAME)
        save_repo(conn, RepoInfo("o/a", 300, "main"), NOW)
        insert_failure(
            conn,
            FailedJob(
                job_id=job_id,
                repo="o/a",
                run_id=job_id,
                workflow_name="CI",
                workflow_path=".github/workflows/ci.yml",
                event="push",
                head_branch="main",
                head_sha="abc",
                fix_sha=None,
                failed_steps=["Run flutter build apk"],
                runner_os="Linux",
                created_at="2026-10-20T00:00:00Z",
                log_path=log_path,
                log_bytes=1,
                redactions=0,
            ),
        )
        conn.close()
    return log_path


def add_label(
    data_dir: Path,
    job_id: int,
    category: str,
    confidence: int = 2,
    evidence: list[int] | None = None,
) -> Label:
    label = Label(
        job_id=job_id,
        repo="o/a",
        run_id=job_id,
        category=category,
        root_cause="cause",
        evidence_lines=evidence if evidence is not None else [5],
        confidence=confidence,
        log_path=f"logs/o__a/{job_id}.log.gz",
        labeled_at=NOW.isoformat(),
    )
    append_label(data_dir / LABELS_FILENAME, label)
    return label
