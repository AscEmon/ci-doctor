"""Remove stored data that the mining rules would no longer collect.

Uses the same rule functions as mining (cidoc.miner.rules), so what prune removes is
exactly what a fresh mine would skip. Planning is read-only; apply_prune does the work.
"""

import contextlib
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cidoc.miner.rules import RepoExcludes, workflow_skip_reason


@dataclass(frozen=True)
class PrunedFailure:
    job_id: int
    repo: str
    log_path: str
    reason: str


@dataclass(frozen=True)
class PrunePlan:
    failures: list[PrunedFailure]
    repos: list[tuple[str, str]]
    """(full_name, reason) for excluded repos still in the repos table."""

    def counts_by_reason(self) -> list[tuple[str, int]]:
        return Counter(f.reason for f in self.failures).most_common()

    @property
    def empty(self) -> bool:
        return not self.failures and not self.repos


@dataclass(frozen=True)
class PruneResult:
    failures: int
    mined_runs: int
    repos: int
    log_files: int


def failure_prune_reason(
    excludes: RepoExcludes, repo: str, workflow_path: str, workflow_name: str
) -> str | None:
    """Why a stored failure should go, or None to keep it. Repo exclusion wins."""
    return excludes.reason(repo) or workflow_skip_reason(workflow_path, workflow_name)


def plan_prune(conn: sqlite3.Connection, excludes: RepoExcludes) -> PrunePlan:
    failures = [
        PrunedFailure(row["job_id"], row["repo"], row["log_path"], reason)
        for row in conn.execute(
            "SELECT job_id, repo, workflow_path, workflow_name, log_path FROM failures"
            " ORDER BY repo, job_id"
        )
        if (
            reason := failure_prune_reason(
                excludes, row["repo"], row["workflow_path"], row["workflow_name"]
            )
        )
    ]
    repos = [
        (name, reason)
        for (name,) in conn.execute("SELECT full_name FROM repos ORDER BY full_name")
        if (reason := excludes.reason(name))
    ]
    return PrunePlan(failures=failures, repos=repos)


def _remove_log(data_dir: Path, log_path: str) -> bool:
    """Delete one stored log. Returns False if it was already gone or lies outside data_dir."""
    root = data_dir.resolve()
    target = (data_dir / log_path).resolve()
    # log_path comes from our own DB, but never delete outside data_dir on a bad row.
    if not target.is_relative_to(root):
        return False
    try:
        target.unlink()
    except FileNotFoundError:
        return False
    # Tidy up a repo's log directory once its last log is gone.
    with contextlib.suppress(OSError):
        target.parent.rmdir()
    return True


def apply_prune(conn: sqlite3.Connection, data_dir: Path, plan: PrunePlan) -> PruneResult:
    """Delete the planned rows in one transaction, then their log files.

    Rows go first: if this stops midway, the worst case is orphaned log files, never rows
    pointing at logs that no longer exist.
    """
    repo_names = [(name,) for name, _ in plan.repos]
    with conn:
        failures = conn.executemany(
            "DELETE FROM failures WHERE job_id = ?", [(f.job_id,) for f in plan.failures]
        ).rowcount
        # The plan holds every failure of an excluded repo (repo exclusion wins), so only
        # mined_runs still references those repos; clear it before the repo rows.
        mined_runs = conn.executemany("DELETE FROM mined_runs WHERE repo = ?", repo_names).rowcount
        repos = conn.executemany("DELETE FROM repos WHERE full_name = ?", repo_names).rowcount
    log_files = sum(_remove_log(data_dir, f.log_path) for f in plan.failures)
    return PruneResult(failures=failures, mined_runs=mined_runs, repos=repos, log_files=log_files)
