import gzip
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.github_client import GitHubClient
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.refetch import find_refetch_targets, refetch_logs
from cidoc.miner.runs import FailedJob, insert_failure, log_relpath
from cidoc.redact import REDACTOR_VERSION

# Fake values for tests only; never put real credentials here.
FAKE_TOKEN = SecretStr("test-github-token-value")
FAKE_SECRET = "Xq7" + "Lm2Rp9Tv4Wz8Kd3Hn6Bc1Gs5Jf0Ya"
NOW = datetime(2026, 10, 31, tzinfo=UTC)
REPO = "o/app"
CACHE_KEY = "flutter-linux-stable-3.24.3-x64-8f14e45fceea167a5a36dedd4bea2543"
# What GitHub serves for the job, and what the old rules stored for it.
RAW_LOG = f"Cache restored from key: {CACHE_KEY}\nsigning with SECRET_KEY={FAKE_SECRET}\n"
OLD_LOG = (
    "Cache restored from key: [REDACTED:keyword_secret]\n"
    "signing with SECRET_KEY=[REDACTED:keyword_secret]\n"
)


def stored_failure(
    conn: sqlite3.Connection, data_dir: Path, job_id: int, text: str, version: int = 1
) -> None:
    relpath = log_relpath(REPO, job_id)
    dest = data_dir / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(gzip.compress(text.encode()))
    redactions = text.count("[REDACTED:")
    insert_failure(
        conn,
        FailedJob(
            job_id=job_id,
            repo=REPO,
            run_id=job_id,
            workflow_name="CI",
            workflow_path=".github/workflows/ci.yml",
            event="push",
            head_branch="main",
            head_sha="abc",
            fix_sha=None,
            failed_steps=["Build"],
            runner_os="Linux",
            created_at="2026-10-20T00:00:00Z",
            log_path=relpath.as_posix(),
            log_bytes=len(text),
            redactions=redactions,
            redaction_counts={},
        ),
    )
    conn.execute("UPDATE failures SET redactor_version = ? WHERE job_id = ?", (version, job_id))
    conn.commit()


def populate(conn: sqlite3.Connection, data_dir: Path) -> None:
    save_repo(conn, RepoInfo(REPO, 300, "main"), NOW)
    stored_failure(conn, data_dir, 1, OLD_LOG)  # target
    stored_failure(conn, data_dir, 2, OLD_LOG)  # target
    stored_failure(conn, data_dir, 3, "token: [REDACTED:github_token]\n")  # vendor mask only
    stored_failure(conn, data_dir, 4, "plain log\n")  # nothing masked
    stored_failure(conn, data_dir, 5, OLD_LOG, version=REDACTOR_VERSION)  # already current


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(tmp_path / DB_FILENAME)
    populate(db, tmp_path)
    yield db
    db.close()


def make_client(logs: dict[int, bytes | int | BaseException], seen: list[int]) -> GitHubClient:
    def handler(request: httpx.Request) -> httpx.Response:
        job_id = int(request.url.path.split("/")[-2])
        seen.append(job_id)
        log = logs[job_id]
        if isinstance(log, BaseException):
            raise log
        if isinstance(log, int):
            return httpx.Response(log, json={"message": "gone"})
        return httpx.Response(200, content=log)

    return GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler), sleep=lambda _: None)


def stored_text(data_dir: Path, job_id: int) -> str:
    return gzip.decompress((data_dir / log_relpath(REPO, job_id)).read_bytes()).decode()


def row(conn: sqlite3.Connection, job_id: int) -> sqlite3.Row:
    result: sqlite3.Row = conn.execute(
        "SELECT * FROM failures WHERE job_id = ?", (job_id,)
    ).fetchone()
    return result


def test_targets_are_old_logs_with_keyword_masks(conn: sqlite3.Connection, tmp_path: Path) -> None:
    assert [t.job_id for t in find_refetch_targets(conn, tmp_path)] == [1, 2]


def test_refetch_rewrites_log_and_row(conn: sqlite3.Connection, tmp_path: Path) -> None:
    targets = [t for t in find_refetch_targets(conn, tmp_path) if t.job_id == 1]
    with make_client({1: RAW_LOG.encode()}, []) as client:
        result = refetch_logs(client, conn, tmp_path, targets)

    assert (result.refetched, result.expired, result.failed) == (1, 0, [])
    text = stored_text(tmp_path, 1)
    assert CACHE_KEY in text  # no longer over-masked
    assert FAKE_SECRET not in text  # still masked under the new rules
    assert text == (
        f"Cache restored from key: {CACHE_KEY}\nsigning with SECRET_KEY=[REDACTED:keyword_secret]\n"
    )
    r = row(conn, 1)
    assert json.loads(r["redaction_counts"]) == {"keyword_secret": 1}
    assert r["redactions"] == 1
    assert r["log_bytes"] == len(text.encode())
    assert (r["redactor_version"], r["over_redacted"]) == (REDACTOR_VERSION, 0)


def test_refetched_log_is_not_a_target_again(conn: sqlite3.Connection, tmp_path: Path) -> None:
    targets = find_refetch_targets(conn, tmp_path)
    with make_client({1: RAW_LOG.encode(), 2: RAW_LOG.encode()}, []) as client:
        refetch_logs(client, conn, tmp_path, targets)

    # Both still hold a (genuine) keyword_secret mask, but are on the current version.
    assert find_refetch_targets(conn, tmp_path) == []


@pytest.mark.parametrize("status", [404, 410])
def test_expired_log_is_kept_and_marked(
    conn: sqlite3.Connection, tmp_path: Path, status: int
) -> None:
    targets = find_refetch_targets(conn, tmp_path)
    with make_client({1: status, 2: RAW_LOG.encode()}, []) as client:
        result = refetch_logs(client, conn, tmp_path, targets)

    assert (result.refetched, result.expired) == (1, 1)
    assert stored_text(tmp_path, 1) == OLD_LOG
    r = row(conn, 1)
    assert (r["over_redacted"], r["redactor_version"]) == (1, 1)
    # Marked logs are not retried.
    assert find_refetch_targets(conn, tmp_path) == []


def test_other_errors_leave_log_untouched_and_continue(
    conn: sqlite3.Connection, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    targets = find_refetch_targets(conn, tmp_path)
    with make_client({1: 403, 2: RAW_LOG.encode()}, []) as client:
        result = refetch_logs(client, conn, tmp_path, targets)

    assert result.refetched == 1
    assert result.failed == [(1, "HTTPStatusError")]
    assert stored_text(tmp_path, 1) == OLD_LOG
    assert (row(conn, 1)["over_redacted"], row(conn, 1)["redactor_version"]) == (0, 1)
    assert "refetch failed: o/app job 1: HTTPStatusError" in caplog.text
    # Still a target, so the next run retries it.
    assert [t.job_id for t in find_refetch_targets(conn, tmp_path)] == [1]


def test_keyboard_interrupt_stops(conn: sqlite3.Connection, tmp_path: Path) -> None:
    targets = find_refetch_targets(conn, tmp_path)
    with (
        make_client({1: RAW_LOG.encode(), 2: KeyboardInterrupt()}, []) as client,
        pytest.raises(KeyboardInterrupt),
    ):
        refetch_logs(client, conn, tmp_path, targets)

    assert row(conn, 1)["redactor_version"] == REDACTOR_VERSION
    assert [t.job_id for t in find_refetch_targets(conn, tmp_path)] == [2]


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    data_dir = tmp_path / "data"
    db = connect(data_dir / DB_FILENAME)
    populate(db, data_dir)
    db.close()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")  # fake, tests only
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    yield data_dir
    get_settings.cache_clear()


def test_cli_dry_run_downloads_nothing(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_github() -> GitHubClient:
        raise AssertionError("dry run must not contact GitHub")

    monkeypatch.setattr("cidoc.cli.open_github", no_github)

    result = CliRunner().invoke(app, ["mine", "refetch", "--only-redacted"], terminal_width=200)

    assert result.exit_code == 0, result.output
    assert "Would refetch 2 logs from 1 repos." in result.output
    assert stored_text(cli_env, 1) == OLD_LOG


def test_cli_apply_prints_summary(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    logs: dict[int, bytes | int | BaseException] = {1: 410, 2: RAW_LOG.encode()}
    monkeypatch.setattr("cidoc.cli.open_github", lambda: make_client(logs, seen))

    result = CliRunner().invoke(
        app, ["mine", "refetch", "--only-redacted", "--apply"], terminal_width=200
    )

    assert result.exit_code == 0, result.output
    assert seen == [1, 2]
    assert "expired o/app job 1: kept old log, marked over_redacted" in result.output
    assert "Refetched 1, expired 1, failed 0." in result.output

    again = CliRunner().invoke(app, ["mine", "refetch", "--only-redacted"], terminal_width=200)
    assert "1 logs are already marked over_redacted" in again.output
    assert "Nothing to refetch." in again.output


def test_cli_requires_only_redacted(cli_env: Path) -> None:
    result = CliRunner().invoke(app, ["mine", "refetch"], terminal_width=200)

    assert result.exit_code == 2
    assert "Only --only-redacted is supported" in result.output
