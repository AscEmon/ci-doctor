"""Summaries of what has been mined so far."""

import sqlite3
from dataclasses import dataclass

UNKNOWN = "(unknown)"


@dataclass(frozen=True)
class RepoStats:
    repo: str
    jobs: int
    with_fix: int


@dataclass(frozen=True)
class MatrixDuplicate:
    """Several failed jobs in one run that failed at the same step: likely matrix legs."""

    repo: str
    run_id: int
    step: str
    jobs: int


@dataclass(frozen=True)
class MiningStats:
    repos: int
    jobs: int
    with_fix: int
    redactions: int
    per_repo: list[RepoStats]
    per_runner_os: list[tuple[str, int]]
    per_failed_step: list[tuple[str, int]]
    runs: int
    matrix_duplicates: list[MatrixDuplicate]

    @property
    def jobs_per_run(self) -> float:
        return self.jobs / self.runs if self.runs else 0.0

    @property
    def duplicate_jobs(self) -> int:
        """Jobs beyond the first in each duplicate group: what one-per-run sampling drops."""
        return sum(d.jobs - 1 for d in self.matrix_duplicates)


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def collect_stats(conn: sqlite3.Connection, top_steps: int = 20) -> MiningStats:
    jobs, with_fix, redactions = conn.execute(
        "SELECT count(*), count(fix_sha), coalesce(sum(redactions), 0) FROM failures"
    ).fetchone()
    per_repo = [
        RepoStats(*row)
        for row in conn.execute(
            "SELECT r.full_name, count(f.job_id), count(f.fix_sha)"
            " FROM repos r LEFT JOIN failures f ON f.repo = r.full_name"
            " GROUP BY r.full_name ORDER BY count(f.job_id) DESC, r.full_name"
        )
    ]
    per_os = [
        (row[0], row[1])
        for row in conn.execute(
            "SELECT coalesce(runner_os, ?), count(*) FROM failures GROUP BY 1 ORDER BY 2 DESC, 1",
            (UNKNOWN,),
        )
    ]
    per_step = [
        (row[0], row[1])
        for row in conn.execute(
            "SELECT step.value, count(*) FROM failures, json_each(failures.failed_steps) AS step"
            " GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT ?",
            (top_steps,),
        )
    ]
    runs: int = conn.execute(
        "SELECT count(*) FROM (SELECT DISTINCT repo, run_id FROM failures)"
    ).fetchone()[0]
    # A job's first failed step is where it broke; later ones (uploads, `if: always()`
    # steps) usually fail as a consequence. Jobs sharing a run and a first failed step
    # are most likely legs of one matrix hitting the same error.
    matrix_duplicates = [
        MatrixDuplicate(*row)
        for row in conn.execute(
            "SELECT repo, run_id, json_extract(failed_steps, '$[0]') AS step, count(*)"
            " FROM failures"
            " WHERE step IS NOT NULL GROUP BY repo, run_id, step HAVING count(*) > 1"
            " ORDER BY count(*) DESC, repo, run_id, step"
        )
    ]
    return MiningStats(
        repos=len(per_repo),
        jobs=jobs,
        with_fix=with_fix,
        redactions=redactions,
        per_repo=per_repo,
        per_runner_os=per_os,
        per_failed_step=per_step,
        runs=runs,
        matrix_duplicates=matrix_duplicates,
    )
