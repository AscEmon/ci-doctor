import gzip
import json
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
from cidoc.miner.runs import (
    LOG_RETENTION,
    FailedJob,
    MineResult,
    list_failed_runs,
    log_relpath,
    mine_repo,
    runner_os_from_labels,
)
from cidoc.redact import REDACTOR_VERSION

# Fake value for tests only; never put real credentials here.
FAKE_TOKEN = SecretStr("test-github-token-value")
NOW = datetime(2026, 10, 31, tzinfo=UTC)
REPO = "acme/app"
FAKE_GHP = "ghp" + "_" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0UvWxYz"


def run(run_id: int, created_at: str = "2026-10-20T10:00:00Z", **extra: Any) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": "CI",
        "path": ".github/workflows/ci.yml",
        "head_repository": {"full_name": REPO},
        "workflow_id": 7,
        "event": "pull_request",
        "head_branch": "feature/x",
        "head_sha": f"sha{run_id}",
        "created_at": created_at,
        "conclusion": "failure",
        **extra,
    }


def job(
    job_id: int, conclusion: str = "failure", labels: list[str] | None = None
) -> dict[str, Any]:
    return {
        "id": job_id,
        "conclusion": conclusion,
        "labels": labels if labels is not None else ["ubuntu-latest"],
        "steps": [
            {"name": "Set up job", "conclusion": "success"},
            {"name": "Run flutter build apk", "conclusion": conclusion},
            {"name": "Upload", "conclusion": "skipped"},
        ],
    }


def make_client(
    runs: list[dict[str, Any]],
    jobs: dict[int, list[dict[str, Any]]],
    logs: dict[int, bytes | int | BaseException],
    seen: list[str] | None = None,
) -> GitHubClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if seen is not None:
            seen.append(path)
        if path == f"/repos/{REPO}/actions/runs":
            return httpx.Response(200, json={"workflow_runs": runs})
        if path.startswith(f"/repos/{REPO}/actions/workflows/"):
            return httpx.Response(200, json={"workflow_runs": []})
        if path.endswith("/jobs"):
            run_id = int(path.split("/")[-2])
            return httpx.Response(200, json={"jobs": jobs[run_id]})
        if path.endswith("/logs"):
            log = logs[int(path.split("/")[-2])]
            if isinstance(log, BaseException):
                raise log
            if isinstance(log, int):
                return httpx.Response(log, json={"message": "gone"})
            return httpx.Response(200, content=log)
        raise AssertionError(f"unexpected request {path}")

    return GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler), sleep=lambda _: None)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(tmp_path / "cidoc.db")
    save_repo(db, RepoInfo(REPO, 500, "main"), NOW)
    yield db
    db.close()


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["ubuntu-22.04"], "Linux"),
        (["self-hosted", "macOS-14"], "macOS"),
        (["windows-latest"], "Windows"),
        (["self-hosted", "gpu"], None),
        ([], None),
    ],
)
def test_runner_os_from_labels(labels: list[str], expected: str | None) -> None:
    assert runner_os_from_labels(labels) == expected


def test_list_failed_runs_filters_and_trims_to_cutoff() -> None:
    seen_params: list[httpx.QueryParams] = []
    runs = [run(1), run(2, created_at="2026-08-02T05:00:00Z"), run(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(request.url.params)
        return httpx.Response(200, json={"workflow_runs": runs})

    since = datetime(2026, 8, 2, 12, tzinfo=UTC)
    with GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler)) as client:
        got = [r["id"] for r in list_failed_runs(client, REPO, since, limit=10)]

    assert got == [1, 3]
    assert seen_params[0]["status"] == "failure"
    assert seen_params[0]["created"] == ">=2026-08-02"


def test_list_failed_runs_respects_limit() -> None:
    runs = [run(i) for i in range(100)]
    pages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        pages.append(request.url.params["page"])
        return httpx.Response(200, json={"workflow_runs": runs})

    with GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler)) as client:
        got = list(list_failed_runs(client, REPO, datetime(2026, 1, 1, tzinfo=UTC), limit=120))

    assert len(got) == 120
    assert pages == ["1", "2"]


@pytest.mark.parametrize(
    "extra",
    [
        {"path": "dynamic/pages/pages-build-deployment", "name": "pages build and deployment"},
        {"path": "dynamic/dependabot/dependabot-updates", "name": "Dependabot Updates"},
        {"path": ".github/workflows/codeql.yml", "name": "CodeQL"},
        {"path": "dynamic/copilot-swe-agent/copilot", "name": "Copilot coding agent"},
    ],
)
def test_list_failed_runs_skips_github_managed_workflows(extra: dict[str, str]) -> None:
    runs = [run(1, **extra), run(2), run(3, **extra), run(4)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"workflow_runs": runs})

    with GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler)) as client:
        got = [r["id"] for r in list_failed_runs(client, REPO, NOW - LOG_RETENTION, limit=2)]

    # Skipped runs don't use up the cap.
    assert got == [2, 4]


def test_mine_repo_never_requests_jobs_for_skipped_workflows(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    seen: list[str] = []
    client = make_client(
        runs=[run(100, path="dynamic/pages/pages-build-deployment")],
        jobs={},
        logs={},
        seen=seen,
    )

    with client:
        stored = mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30).stored

    assert stored == 0
    assert seen == [f"/repos/{REPO}/actions/runs"]


@pytest.mark.parametrize(
    "head_repository", [{"full_name": "someone/app-fork"}, None], ids=["fork", "deleted-fork"]
)
def test_fork_pr_run_gets_no_fix_sha_and_no_lookup(
    conn: sqlite3.Connection, tmp_path: Path, head_repository: dict[str, str] | None
) -> None:
    seen: list[str] = []
    client = make_client(
        runs=[run(100, head_repository=head_repository)],
        jobs={100: [job(1001)]},
        logs={1001: b"error: boom\n"},
        seen=seen,
    )

    with client:
        stored = mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30).stored

    assert stored == 1
    assert conn.execute("SELECT fix_sha FROM failures").fetchone()["fix_sha"] is None
    assert not any("/actions/workflows/" in path for path in seen)


def test_same_repo_run_still_looks_up_fix_sha(conn: sqlite3.Connection, tmp_path: Path) -> None:
    seen: list[str] = []
    client = make_client(
        runs=[run(100, head_repository={"full_name": "ACME/App"})],  # names are case-insensitive
        jobs={100: [job(1001)]},
        logs={1001: b"error: boom\n"},
        seen=seen,
    )

    with client:
        mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30)

    assert f"/repos/{REPO}/actions/workflows/7/runs" in seen


def test_mine_repo_stores_failed_jobs_with_redacted_gzipped_logs(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    raw_log = f"\ufeffStep 1\nexport GITHUB_TOKEN={FAKE_GHP}\nBUILD FAILED\n".encode()
    client = make_client(
        runs=[run(100)],
        jobs={100: [job(1001, labels=["macos-14"]), job(1002, conclusion="success")]},
        logs={1001: raw_log},
    )
    reported: list[FailedJob | None] = []

    with client:
        stored = mine_repo(
            client, conn, tmp_path, REPO, NOW, max_runs=30, on_job=lambda j, _: reported.append(j)
        ).stored

    assert stored == 1
    row = conn.execute("SELECT * FROM failures").fetchone()
    assert row["job_id"] == 1001
    assert row["run_id"] == 100
    assert row["workflow_name"] == "CI"
    assert row["workflow_path"] == ".github/workflows/ci.yml"
    assert row["event"] == "pull_request"
    assert row["head_branch"] == "feature/x"
    assert row["head_sha"] == "sha100"
    assert row["fix_sha"] is None
    assert json.loads(row["failed_steps"]) == ["Run flutter build apk"]
    assert row["runner_os"] == "macOS"
    assert row["log_path"] == "logs/acme__app/1001.log.gz"
    assert row["redactions"] == 1
    assert json.loads(row["redaction_counts"]) == {"github_token": 1}
    assert (row["redactor_version"], row["over_redacted"]) == (REDACTOR_VERSION, 0)

    text = gzip.decompress((tmp_path / row["log_path"]).read_bytes()).decode()
    assert text == "Step 1\nexport GITHUB_TOKEN=[REDACTED:github_token]\nBUILD FAILED\n"
    assert row["log_bytes"] == len(text.encode())
    assert FAKE_GHP not in text
    assert not list(tmp_path.rglob("*.tmp"))
    assert [j.job_id if j else None for j in reported] == [1001]


def test_mine_repo_skips_expired_logs(conn: sqlite3.Connection, tmp_path: Path) -> None:
    client = make_client(
        runs=[run(100)],
        jobs={100: [job(1001), job(1002)]},
        logs={1001: 410, 1002: b"error: boom\n"},
    )
    skipped: list[int] = []

    with client:
        stored = mine_repo(
            client,
            conn,
            tmp_path,
            REPO,
            NOW,
            max_runs=30,
            on_job=lambda j, job_id: skipped.append(job_id) if j is None else None,
        ).stored

    assert stored == 1
    assert skipped == [1001]
    assert [r[0] for r in conn.execute("SELECT job_id FROM failures")] == [1002]
    assert not (tmp_path / log_relpath(REPO, 1001)).exists()


def stored_jobs(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute("SELECT job_id FROM failures ORDER BY job_id")]


def test_rerun_after_completion_does_no_work(conn: sqlite3.Connection, tmp_path: Path) -> None:
    runs = [run(100), run(200)]
    jobs = {100: [job(1001)], 200: [job(2001)]}
    logs: dict[int, bytes | int | BaseException] = {1001: b"a\n", 2001: 410}
    with make_client(runs, jobs, logs) as client:
        assert mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30).stored == 1

    seen: list[str] = []
    with make_client(runs, jobs, logs, seen) as client:
        assert mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30).stored == 0

    assert seen == [f"/repos/{REPO}/actions/runs"]
    assert stored_jobs(conn) == [1001]


def test_resume_after_interruption_mid_run(conn: sqlite3.Connection, tmp_path: Path) -> None:
    runs = [run(100)]
    jobs = {100: [job(1001), job(1002)]}
    crashing: dict[int, bytes | int | BaseException] = {1001: b"a\n", 1002: KeyboardInterrupt()}

    with make_client(runs, jobs, crashing) as client, pytest.raises(KeyboardInterrupt):
        mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30)

    assert stored_jobs(conn) == [1001]
    assert conn.execute("SELECT count(*) FROM mined_runs").fetchone()[0] == 0

    seen: list[str] = []
    logs: dict[int, bytes | int | BaseException] = {1001: b"a\n", 1002: b"b\n"}
    with make_client(runs, jobs, logs, seen) as client:
        assert mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30).stored == 1

    assert f"/repos/{REPO}/actions/jobs/1001/logs" not in seen
    assert f"/repos/{REPO}/actions/jobs/1002/logs" in seen
    assert stored_jobs(conn) == [1001, 1002]
    assert [r[0] for r in conn.execute("SELECT run_id FROM mined_runs")] == [100]


def test_error_in_one_run_skips_it_and_continues(
    conn: sqlite3.Connection, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runs = [run(100), run(200)]
    jobs = {100: [job(1001)], 200: [job(2001)]}
    logs: dict[int, bytes | int | BaseException] = {1001: RuntimeError("boom"), 2001: b"b\n"}

    with make_client(runs, jobs, logs) as client:
        result = mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30)

    assert result == MineResult(stored=1, skipped_runs=1)
    assert stored_jobs(conn) == [2001]
    # The failed run stays unmarked so the next pass retries it.
    assert [r[0] for r in conn.execute("SELECT run_id FROM mined_runs")] == [200]
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert record.getMessage() == f"run skipped: {REPO} run 100: RuntimeError"


def test_skipped_run_is_retried_on_next_pass(conn: sqlite3.Connection, tmp_path: Path) -> None:
    runs = [run(100)]
    jobs = {100: [job(1001)]}
    with make_client(runs, jobs, {1001: RuntimeError("boom")}) as client:
        assert mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30).skipped_runs == 1

    with make_client(runs, jobs, {1001: b"a\n"}) as client:
        result = mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=30)

    assert result == MineResult(stored=1, skipped_runs=0)
    assert stored_jobs(conn) == [1001]


def test_mined_runs_count_toward_per_repo_cap(conn: sqlite3.Connection, tmp_path: Path) -> None:
    runs = [run(100), run(200), run(300)]
    jobs = {100: [job(1001)], 200: [job(2001)], 300: [job(3001)]}
    logs: dict[int, bytes | int | BaseException] = {1001: b"a", 2001: b"b", 3001: b"c"}

    with make_client(runs, jobs, logs) as client:
        assert mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=2).stored == 2
        assert mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=2).stored == 0
        assert mine_repo(client, conn, tmp_path, REPO, NOW, max_runs=3).stored == 1

    assert stored_jobs(conn) == [1001, 2001, 3001]
