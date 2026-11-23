import gzip
import json
import random
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.redaction_report import collect_redaction_report, snippet_around
from cidoc.miner.runs import FailedJob, insert_failure, log_relpath, store_log

NOW = datetime(2026, 10, 31, tzinfo=UTC)
REPO = "o/app"
# Fake secrets built at runtime so no scanner mistakes them for real ones.
FAKE_GHP = "ghp" + "_" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0UvWxYz"
FAKE_AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
FAKE_LONG = "Xq7" + "Lm2Rp9Tv4Wz8Kd3Hn6Bc1Gs5Jf0Ya"  # random-looking: keyword_secret
FAKE_SECRETS = (FAKE_GHP, FAKE_AWS, FAKE_LONG)


def rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311 - example sampling only, not security


def mine_log(conn: sqlite3.Connection, data_dir: Path, job_id: int, raw: str) -> None:
    """Store a log exactly as mining does (redact + gzip) and insert its row."""
    relpath = log_relpath(REPO, job_id)
    log_bytes, counts = store_log(data_dir, relpath, raw.encode())
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
            failed_steps=[],
            runner_os="Linux",
            created_at="2026-10-20T00:00:00Z",
            log_path=relpath.as_posix(),
            log_bytes=log_bytes,
            redactions=sum(counts.values()),
            redaction_counts=counts,
        ),
    )


LOG_1 = (
    "2026-10-20T10:00:00.1234567Z Run flutter build apk\n"
    f"2026-10-20T10:00:01.0000000Z export GITHUB_TOKEN={FAKE_GHP} && ./gradlew assembleRelease\n"
    f"aws configure set aws_access_key_id {FAKE_AWS}\n"
    f"signing: storePassword={FAKE_LONG}\n"
    "BUILD FAILED\n"
)
LOG_2 = f"token: {FAKE_GHP}\nok\n"


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(tmp_path / DB_FILENAME)
    save_repo(db, RepoInfo(REPO, 300, "main"), NOW)
    mine_log(db, tmp_path, 1, LOG_1)
    mine_log(db, tmp_path, 2, LOG_2)
    mine_log(db, tmp_path, 3, "nothing secret here\n")
    yield db
    db.close()


def report_text(conn: sqlite3.Connection, data_dir: Path) -> str:
    report = collect_redaction_report(conn, data_dir, rng(0))
    return json.dumps(
        [[r.rule, r.redactions, r.logs, [e.snippet for e in r.examples]] for r in report.rules]
    )


def test_counts_per_rule_and_logs(conn: sqlite3.Connection, tmp_path: Path) -> None:
    report = collect_redaction_report(conn, tmp_path, rng(0))

    assert [(r.rule, r.redactions, r.logs) for r in report.rules] == [
        ("github_token", 2, 2),
        ("aws_access_key", 1, 1),
        ("keyword_secret", 1, 1),
    ]
    assert report.logs_with_redactions == 2
    assert report.logs_counted_from_markers == 0
    assert report.logs_missing == []


def test_examples_show_context_with_value_masked(conn: sqlite3.Connection, tmp_path: Path) -> None:
    report = collect_redaction_report(conn, tmp_path, rng(0), context=12)
    by_rule = {r.rule: r for r in report.rules}

    [aws] = by_rule["aws_access_key"].examples
    assert (aws.job_id, aws.repo, aws.line_no) == (1, REPO, 3)
    assert aws.snippet == "…cess_key_id [REDACTED:aws_access_key]"
    gh = {e.snippet for e in by_rule["github_token"].examples}
    # Timestamps are cleaned off, so context starts at the real log text.
    assert gh == {
        "…ITHUB_TOKEN=[REDACTED:github_token] && ./gradle…",
        "token: [REDACTED:github_token]",
    }


def test_never_contains_an_unmasked_value(conn: sqlite3.Connection, tmp_path: Path) -> None:
    text = report_text(conn, tmp_path)

    for secret in FAKE_SECRETS:
        assert secret not in text
        assert secret[:12] not in text


def test_examples_are_capped_and_seeded(tmp_path: Path) -> None:
    db = connect(tmp_path / DB_FILENAME)
    save_repo(db, RepoInfo(REPO, 300, "main"), NOW)
    mine_log(db, tmp_path, 1, "".join(f"line {i} token: {FAKE_GHP}\n" for i in range(25)))

    first = collect_redaction_report(db, tmp_path, rng(7), examples_per_rule=10)
    again = collect_redaction_report(db, tmp_path, rng(7), examples_per_rule=10)
    db.close()

    [rule] = first.rules
    assert rule.redactions == 25
    assert len(rule.examples) == 10
    assert len({e.line_no for e in rule.examples}) == 10
    assert first == again


def test_old_rows_are_counted_from_markers(conn: sqlite3.Connection, tmp_path: Path) -> None:
    # Rows mined before the redaction_counts column existed carry the '{}' default.
    conn.execute("UPDATE failures SET redaction_counts = '{}'")

    report = collect_redaction_report(conn, tmp_path, rng(0))

    assert [(r.rule, r.redactions) for r in report.rules] == [
        ("github_token", 2),
        ("aws_access_key", 1),
        ("keyword_secret", 1),
    ]
    assert report.logs_counted_from_markers == 2


def test_marker_count_includes_text_hidden_by_carriage_returns(tmp_path: Path) -> None:
    db = connect(tmp_path / DB_FILENAME)
    save_repo(db, RepoInfo(REPO, 300, "main"), NOW)
    mine_log(db, tmp_path, 1, f"token: {FAKE_GHP}\rprogress 100%\n")
    db.execute("UPDATE failures SET redaction_counts = '{}'")

    report = collect_redaction_report(db, tmp_path, rng(0))
    db.close()

    [rule] = report.rules
    assert rule.redactions == 1  # counted, though the cleaned line has no example to show


def test_missing_log_is_reported_not_fatal(conn: sqlite3.Connection, tmp_path: Path) -> None:
    (tmp_path / log_relpath(REPO, 2)).unlink()

    report = collect_redaction_report(conn, tmp_path, rng(0))

    assert report.logs_missing == [2]
    assert report.logs_with_redactions == 1


def test_snippet_is_re_redacted() -> None:
    # A log mined under older, weaker rules could still hold a raw value next to a mask.
    line = f"[REDACTED:sk_key] then {FAKE_GHP}"

    snippet = snippet_around(line, 0, len("[REDACTED:sk_key]"), context=60)

    assert FAKE_GHP not in snippet
    assert snippet == "[REDACTED:sk_key] then [REDACTED:github_token]"


def test_stored_log_never_holds_raw_values(conn: sqlite3.Connection, tmp_path: Path) -> None:
    raw = gzip.decompress((tmp_path / log_relpath(REPO, 1)).read_bytes()).decode()

    for secret in FAKE_SECRETS:
        assert secret not in raw


def test_cli_redaction_report(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")  # fake, tests only
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    result = CliRunner().invoke(
        app, ["mine", "redaction-report", "--seed", "1"], terminal_width=200
    )
    get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    assert "Redactions in 2 logs" in result.output
    assert "github_token" in result.output
    # Printed without Rich markup, so masks render literally instead of being eaten as tags.
    assert "export GITHUB_TOKEN=[REDACTED:github_token]" in result.output
    assert f"{REPO} job 1 L3:" in result.output
    for secret in FAKE_SECRETS:
        assert secret not in result.output
