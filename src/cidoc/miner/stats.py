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
class MiningStats:
    repos: int
    jobs: int
    with_fix: int
    redactions: int
    per_repo: list[RepoStats]
    per_runner_os: list[tuple[str, int]]
    per_failed_step: list[tuple[str, int]]


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
    return MiningStats(
        repos=len(per_repo),
        jobs=jobs,
        with_fix=with_fix,
        redactions=redactions,
        per_repo=per_repo,
        per_runner_os=per_os,
        per_failed_step=per_step,
    )
