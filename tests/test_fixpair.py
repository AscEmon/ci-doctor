import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from cidoc.db import connect
from cidoc.github_client import GitHubClient
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.fixpair import FixPairFinder, find_fix_sha, list_success_runs
from cidoc.miner.runs import mine_repo

# Fake value for tests only; never put real credentials here.
FAKE_TOKEN = SecretStr("test-github-token-value")
NOW = datetime(2026, 10, 31, tzinfo=UTC)
SINCE = datetime(2026, 8, 2, tzinfo=UTC)
REPO = "acme/app"


def run(
    run_id: int,
    created_at: str,
    conclusion: str = "success",
    workflow_id: int = 7,
    branch: str | None = "main",
    sha: str | None = None,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": "CI",
        "path": ".github/workflows/ci.yml",
        "workflow_id": workflow_id,
        "event": "push",
        "head_branch": branch,
        "head_sha": sha or f"sha{run_id}",
        "created_at": created_at,
        "conclusion": conclusion,
    }


FAILED = run(10, "2026-10-10T12:00:00Z", conclusion="failure")


def test_picks_earliest_success_after_failure() -> None:
    successes = [
        run(14, "2026-10-12T09:00:00Z"),
        run(12, "2026-10-10T15:00:00Z"),
        run(13, "2026-10-11T08:00:00Z"),
    ]

    assert find_fix_sha(FAILED, successes) == "sha12"


def test_ignores_success_before_or_at_failure_time() -> None:
    successes = [run(9, "2026-10-10T11:59:59Z"), run(11, "2026-10-10T12:00:00Z")]

    assert find_fix_sha(FAILED, successes) is None


def test_ignores_other_workflows_branches_and_non_success() -> None:
    successes = [
        run(11, "2026-10-10T13:00:00Z", workflow_id=8),
        run(12, "2026-10-10T13:00:00Z", branch="dev"),
        run(13, "2026-10-10T13:00:00Z", conclusion="cancelled"),
        run(14, "2026-10-11T13:00:00Z"),
    ]

    assert find_fix_sha(FAILED, successes) == "sha14"


def test_rerun_on_same_sha_is_kept_as_flaky_signal() -> None:
    successes = [run(11, "2026-10-10T12:30:00Z", sha=FAILED["head_sha"])]

    assert find_fix_sha(FAILED, successes) == FAILED["head_sha"]


def test_handles_mixed_utc_offsets() -> None:
    # 13:00+02:00 is 11:00Z, i.e. before the failure.
    successes = [run(11, "2026-10-10T13:00:00+02:00"), run(12, "2026-10-10T12:00:01Z")]

    assert find_fix_sha(FAILED, successes) == "sha12"


def test_list_success_runs_queries_workflow_branch_and_pages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        page = int(request.url.params["page"])
        count = 100 if page == 1 else 3
        runs = [run(page * 1000 + i, "2026-10-11T00:00:00Z") for i in range(count)]
        return httpx.Response(200, json={"workflow_runs": runs})

    with GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler)) as client:
        runs = list_success_runs(client, REPO, 7, "feature/a b", SINCE)

    assert len(runs) == 103
    assert [r.url.path for r in seen] == [f"/repos/{REPO}/actions/workflows/7/runs"] * 2
    params = seen[0].url.params
    assert params["branch"] == "feature/a b"
    assert params["status"] == "success"
    assert params["created"] == ">=2026-08-02"


def test_finder_caches_per_workflow_and_branch() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["branch"])
        return httpx.Response(200, json={"workflow_runs": [run(20, "2026-10-20T00:00:00Z")]})

    with GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler)) as client:
        finder = FixPairFinder(client, REPO, SINCE)
        first = finder.fix_sha(FAILED)
        second = finder.fix_sha(run(15, "2026-10-15T00:00:00Z", conclusion="failure"))
        detached = finder.fix_sha(
            run(16, "2026-10-15T00:00:00Z", conclusion="failure", branch=None)
        )

    assert (first, second, detached) == ("sha20", "sha20", None)
    assert calls == ["main"]


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(tmp_path / "cidoc.db")
    save_repo(db, RepoInfo(REPO, 500, "main"), NOW)
    yield db
    db.close()


def test_mine_repo_records_fix_sha(conn: sqlite3.Connection, tmp_path: Path) -> None:
    failed_runs = [
        run(30, "2026-10-20T10:00:00Z", conclusion="failure"),
        run(10, "2026-10-10T12:00:00Z", conclusion="failure"),
    ]
    successes = [run(40, "2026-10-21T10:00:00Z"), run(12, "2026-10-10T15:00:00Z")]
    failed_job = {"conclusion": "failure", "labels": [], "steps": []}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/repos/{REPO}/actions/runs":
            return httpx.Response(200, json={"workflow_runs": failed_runs})
        if path == f"/repos/{REPO}/actions/workflows/7/runs":
            return httpx.Response(200, json={"workflow_runs": successes})
        if path.endswith("/jobs"):
            run_id = int(path.split("/")[-2])
            return httpx.Response(200, json={"jobs": [{"id": run_id * 10, **failed_job}]})
        return httpx.Response(200, content=b"error: boom\n")

    with GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler)) as client:
        mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30)

    rows = dict(conn.execute("SELECT run_id, fix_sha FROM failures").fetchall())
    assert rows == {30: "sha40", 10: "sha12"}
