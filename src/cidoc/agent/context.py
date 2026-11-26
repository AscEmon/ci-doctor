"""What the agent's tools may know about one failed job.

fix_sha is deliberately absent: it is the commit that later fixed the failure, so it
would leak the answer. It is never selected from the database, so no tool can return it
and no prompt can contain it.
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field

from cidoc.logview import clean_log

# Never add fix_sha here.
_COLUMNS = "repo, run_id, head_sha, head_branch, workflow_path, failed_steps"
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class RunContext:
    job_id: int
    repo: str | None = None
    run_id: int | None = None
    head_sha: str | None = None
    """The failing commit. Every repository read happens at this commit."""
    head_branch: str | None = None
    workflow_path: str | None = None
    failed_steps: list[str] = field(default_factory=list)
    """Failed step names from the API, to find the failed step blocks in the log."""
    log_lines: list[str] = field(default_factory=list)
    """The full stored log, cleaned (timestamps and ANSI codes removed). Line N here is
    "L<N>|" in the excerpt, so numbers from the log tools match the excerpt."""

    @property
    def has_run(self) -> bool:
        """Run metadata is known (the job is in the database), so GitHub tools can work."""
        return self.repo is not None and self.run_id is not None and self.head_sha is not None


def valid_repo(repo: str) -> bool:
    return bool(_REPO.match(repo)) and ".." not in repo


def valid_sha(sha: str) -> bool:
    return bool(_SHA.match(sha))


def load_context(conn: sqlite3.Connection, job_id: int, log_text: str) -> RunContext:
    """The job's run metadata (if it is still in the database) and its full log."""
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM failures WHERE job_id = ?",  # noqa: S608 - constant columns
        (job_id,),
    ).fetchone()
    lines = clean_log(log_text)
    if row is None:
        return RunContext(job_id=job_id, log_lines=lines)
    return RunContext(
        job_id=job_id,
        repo=row["repo"],
        run_id=row["run_id"],
        head_sha=row["head_sha"],
        head_branch=row["head_branch"],
        workflow_path=row["workflow_path"],
        failed_steps=list(json.loads(row["failed_steps"] or "[]")),
        log_lines=lines,
    )
