"""Collect failed GitHub Actions jobs and their (redacted, gzipped) logs."""

import gzip
import json
import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from cidoc.github_client import GitHubClient
from cidoc.miner.fixpair import FixPairFinder
from cidoc.redact import redact

# GitHub deletes run logs after 90 days by default.
LOG_RETENTION = timedelta(days=90)
RUNS_PAGE_SIZE = 100
LOGS_DIR = "logs"
# Expired or deleted logs answer with these; skip the job rather than fail the repo.
MISSING_LOG_STATUSES = frozenset({404, 410})

_OS_BY_LABEL_PREFIX = (
    ("ubuntu", "Linux"),
    ("linux", "Linux"),
    ("macos", "macOS"),
    ("windows", "Windows"),
)


@dataclass(frozen=True)
class FailedJob:
    job_id: int
    repo: str
    run_id: int
    workflow_name: str
    workflow_path: str
    event: str
    head_branch: str | None
    head_sha: str
    fix_sha: str | None
    failed_steps: list[str]
    runner_os: str | None
    created_at: str
    log_path: str
    log_bytes: int
    redactions: int


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def runner_os_from_labels(labels: Sequence[str]) -> str | None:
    """Guess the runner OS from `runs-on` labels; self-hosted runners may say nothing useful."""
    for label in labels:
        lowered = label.lower()
        for prefix, os_name in _OS_BY_LABEL_PREFIX:
            if lowered.startswith(prefix):
                return os_name
    return None


def failed_step_names(job: dict[str, Any]) -> list[str]:
    return [s["name"] for s in job.get("steps") or [] if s.get("conclusion") == "failure"]


def list_failed_runs(
    client: GitHubClient, repo: str, since: datetime, limit: int
) -> Iterator[dict[str, Any]]:
    """Yield up to `limit` failed runs created after `since`, newest first."""
    seen = 0
    page = 1
    while seen < limit:
        data = client.get_json(
            f"/repos/{repo}/actions/runs",
            params={
                "status": "failure",
                "created": f">={since.date().isoformat()}",
                "per_page": RUNS_PAGE_SIZE,
                "page": page,
            },
        )
        runs = data.get("workflow_runs", [])
        for run in runs:
            # `created` filters by day; trim to the exact cutoff.
            if parse_time(run["created_at"]) < since:
                continue
            yield run
            seen += 1
            if seen >= limit:
                return
        if len(runs) < RUNS_PAGE_SIZE:
            return
        page += 1


def list_failed_jobs(client: GitHubClient, repo: str, run_id: int) -> list[dict[str, Any]]:
    data = client.get_json(
        f"/repos/{repo}/actions/runs/{run_id}/jobs",
        params={"filter": "latest", "per_page": 100},
    )
    return [job for job in data.get("jobs", []) if job.get("conclusion") == "failure"]


def log_relpath(repo: str, job_id: int) -> Path:
    owner, name = repo.split("/", 1)
    return Path(LOGS_DIR) / f"{owner}__{name}" / f"{job_id}.log.gz"


def download_job_log(client: GitHubClient, repo: str, job_id: int) -> bytes | None:
    try:
        return client.download_log(f"/repos/{repo}/actions/jobs/{job_id}/logs")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in MISSING_LOG_STATUSES:
            return None
        raise


def store_log(data_dir: Path, relpath: Path, raw: bytes) -> tuple[int, int]:
    """Redact and gzip a log under `data_dir`. Returns (uncompressed bytes, redaction count).

    Writes to a temp file and renames, so an interrupted run never leaves a half-written log.
    """
    result = redact(raw.decode("utf-8-sig", errors="replace"))
    payload = result.text.encode("utf-8")
    dest = data_dir / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(payload)
    os.replace(tmp, dest)
    return len(payload), result.total


def insert_failure(conn: sqlite3.Connection, job: FailedJob) -> None:
    with conn:
        conn.execute(
            "INSERT INTO failures (job_id, repo, run_id, workflow_name, workflow_path, event,"
            " head_branch, head_sha, fix_sha, failed_steps, runner_os, created_at, log_path,"
            " log_bytes, redactions)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job.job_id,
                job.repo,
                job.run_id,
                job.workflow_name,
                job.workflow_path,
                job.event,
                job.head_branch,
                job.head_sha,
                job.fix_sha,
                json.dumps(job.failed_steps),
                job.runner_os,
                job.created_at,
                job.log_path,
                job.log_bytes,
                job.redactions,
            ),
        )


def stored_job_ids(conn: sqlite3.Connection, repo: str) -> set[int]:
    return {row[0] for row in conn.execute("SELECT job_id FROM failures WHERE repo = ?", (repo,))}


def mined_run_ids(conn: sqlite3.Connection, repo: str) -> set[int]:
    rows = conn.execute("SELECT run_id FROM mined_runs WHERE repo = ?", (repo,))
    return {row[0] for row in rows}


def mark_run_mined(conn: sqlite3.Connection, repo: str, run_id: int, now: datetime) -> None:
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO mined_runs (run_id, repo, mined_at) VALUES (?, ?, ?)",
            (run_id, repo, now.isoformat()),
        )


def mine_repo(
    client: GitHubClient,
    conn: sqlite3.Connection,
    data_dir: Path,
    repo: str,
    now: datetime,
    max_runs: int,
    on_job: Callable[[FailedJob | None, int], None] | None = None,
) -> int:
    """Store every failed job from up to `max_runs` recent failed runs. Returns jobs stored.

    Safe to re-run after an interruption: finished runs and stored jobs are skipped, and
    already-mined runs still count toward `max_runs` so the cap is per repo, not per call.
    `on_job` is called with the stored job, or None and the job id when its log is gone.
    """
    since = now - LOG_RETENTION
    fixes = FixPairFinder(client, repo, since)
    done_runs = mined_run_ids(conn, repo)
    done_jobs = stored_job_ids(conn, repo)
    stored = 0
    for run in list_failed_runs(client, repo, since, max_runs):
        if run["id"] in done_runs:
            continue
        for job in list_failed_jobs(client, repo, run["id"]):
            if job["id"] in done_jobs:
                continue
            raw = download_job_log(client, repo, job["id"])
            if raw is None:
                if on_job:
                    on_job(None, job["id"])
                continue
            relpath = log_relpath(repo, job["id"])
            log_bytes, redactions = store_log(data_dir, relpath, raw)
            failure = FailedJob(
                job_id=job["id"],
                repo=repo,
                run_id=run["id"],
                workflow_name=run.get("name") or "",
                workflow_path=run.get("path") or "",
                event=run["event"],
                head_branch=run.get("head_branch"),
                head_sha=run["head_sha"],
                # Cached per (workflow, branch), so this costs at most one lookup per branch.
                fix_sha=fixes.fix_sha(run),
                failed_steps=failed_step_names(job),
                runner_os=runner_os_from_labels(job.get("labels") or []),
                created_at=run["created_at"],
                log_path=relpath.as_posix(),
                log_bytes=log_bytes,
                redactions=redactions,
            )
            insert_failure(conn, failure)
            stored += 1
            if on_job:
                on_job(failure, failure.job_id)
        mark_run_mined(conn, repo, run["id"], now)
    return stored


def repos_to_mine(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT full_name FROM repos ORDER BY full_name")]
