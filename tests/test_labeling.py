import gzip
import json
import random
import sqlite3
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.labeling.labels import (
    CATEGORY_KEYS,
    Label,
    append_label,
    load_labels,
    parse_evidence,
)
from cidoc.labeling.session import Candidate, load_candidates, pick_stratified, run_session
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.runs import FailedJob, insert_failure

NOW = datetime(2026, 10, 31, tzinfo=UTC)
LOG_LINES = 10


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311 - deterministic sampling in tests


def candidate(job_id: int, repo: str) -> Candidate:
    return Candidate(
        job_id=job_id,
        repo=repo,
        run_id=job_id // 10,
        workflow_name="CI",
        workflow_path=".github/workflows/ci.yml",
        event="push",
        head_branch="main",
        failed_steps=["Build"],
        runner_os="Linux",
        log_path=f"logs/{repo.replace('/', '__')}/{job_id}.log.gz",
    )


def make_label(job_id: int, repo: str = "o/a") -> Label:
    return Label(
        job_id=job_id,
        repo=repo,
        run_id=1,
        category="test_failure",
        root_cause="widget test expects old text",
        evidence_lines=[3],
        confidence=2,
        log_path="logs/x.log.gz",
        labeled_at=NOW.isoformat(),
    )


def test_taxonomy_keys_are_unique_single_chars() -> None:
    assert all(len(k) == 1 for k in CATEGORY_KEYS)
    assert len(set(CATEGORY_KEYS.values())) == 12
    assert "s" not in CATEGORY_KEYS
    assert "q" not in CATEGORY_KEYS


def test_parse_evidence() -> None:
    assert parse_evidence(" 12, 3,12 ,", max_line=20) == [3, 12]


@pytest.mark.parametrize(
    ("text", "error"),
    [("", "at least one"), ("4, x", "'x'"), ("0", "outside"), ("21", "outside"), ("-2", "'-2'")],
)
def test_parse_evidence_rejects(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_evidence(text, max_line=20)


def test_append_and_load_round_trip_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "labels.jsonl"
    append_label(path, make_label(1))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"job_id": 2, "repo": "o/a", "categ')  # crash mid-write
        fh.write("\n")
    append_label(path, make_label(3))
    bad: list[int] = []

    labels = load_labels(path, on_bad_line=bad.append)

    assert [lbl.job_id for lbl in labels] == [1, 3]
    assert bad == [2]


def test_load_labels_missing_file(tmp_path: Path) -> None:
    assert load_labels(tmp_path / "nope.jsonl") == []


def test_pick_stratified_round_robins_across_repos() -> None:
    pool = [candidate(i, "o/big") for i in range(10)] + [
        candidate(100, "o/small"),
        candidate(101, "o/small"),
    ]

    picked = pick_stratified(pool, [], n=6, rng=seeded(0))

    assert Counter(c.repo for c in picked) == {"o/big": 4, "o/small": 2}
    assert len({c.job_id for c in picked}) == 6


def test_pick_stratified_accounts_for_earlier_labels() -> None:
    pool = [candidate(i, "o/a") for i in range(5)] + [candidate(10 + i, "o/b") for i in range(5)]

    picked = pick_stratified(pool, ["o/a", "o/a", "o/a"], n=3, rng=seeded(1))

    assert [c.repo for c in picked] == ["o/b", "o/b", "o/b"]


def test_pick_stratified_returns_all_when_n_exceeds_pool() -> None:
    pool = [candidate(1, "o/a"), candidate(2, "o/b")]

    assert len(pick_stratified(pool, [], n=50, rng=seeded(2))) == 2


class Script:
    """Feeds scripted keypresses and typed answers to the session."""

    def __init__(self, keys: str, answers: list[str]) -> None:
        self.keys = list(keys)
        self.answers = list(answers)

    def getch(self) -> str:
        return self.keys.pop(0)

    def ask(self, prompt: str) -> str:
        answer = self.answers.pop(0)
        if answer == "^C":
            raise KeyboardInterrupt
        return answer


def write_log(data_dir: Path, c: Candidate) -> None:
    path = data_dir / c.log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"2026-10-01T00:00:00Z line {i}\n" for i in range(1, LOG_LINES + 1))
    path.write_bytes(gzip.compress(lines.encode()))


def session(
    tmp_path: Path, picks: list[Candidate], script: Script
) -> tuple[Path, int, int, bool, str]:
    for c in picks:
        write_log(tmp_path, c)
    labels_path = tmp_path / "labels.jsonl"
    console = Console(record=True, width=120, color_system=None)
    result = run_session(
        console, picks, tmp_path, labels_path, script.getch, script.ask, clock=lambda: NOW
    )
    return labels_path, result.labeled, result.skipped, result.quit, console.export_text()


def test_session_labels_skips_and_retries_invalid_input(tmp_path: Path) -> None:
    picks = [candidate(1, "o/a"), candidate(2, "o/b"), candidate(3, "o/c")]
    # job 1: bad key 'x' ignored, '6' test_failure, empty cause re-asked, bad evidence re-asked,
    #        bad confidence '9' ignored, then '3'. job 2: skip. job 3: 't' timeout_resource.
    script = Script(
        keys="x6" + "93" + "s" + "t1",
        answers=["", "golden image   mismatch", "99", "4, 2", "runner ran out of time", "10"],
    )

    labels_path, labeled, skipped, quit_, out = session(tmp_path, picks, script)

    assert (labeled, skipped, quit_) == (2, 1, False)
    labels = load_labels(labels_path)
    assert labels[0] == Label(
        job_id=1,
        repo="o/a",
        run_id=0,
        category="test_failure",
        root_cause="golden image mismatch",
        evidence_lines=[2, 4],
        confidence=3,
        log_path="logs/o__a/1.log.gz",
        labeled_at=NOW.isoformat(),
    )
    assert (labels[1].job_id, labels[1].category, labels[1].confidence) == (
        3,
        "timeout_resource",
        1,
    )
    assert "root cause cannot be empty" in out
    assert "99 is outside 1..10" in out
    assert "https://github.com/o/a/actions/runs/0/job/1" in out
    assert "line 10" in out


def test_quit_mid_label_writes_nothing_and_resume_continues(tmp_path: Path) -> None:
    picks = [candidate(1, "o/a"), candidate(2, "o/a")]

    labels_path, labeled, _, quit_, _ = session(
        tmp_path, picks, Script(keys="1" + "2" + "2", answers=["cause", "1", "cause two", "^C"])
    )

    assert (labeled, quit_) == (1, True)
    assert [lbl.job_id for lbl in load_labels(labels_path)] == [1]

    remaining = [
        c for c in picks if c.job_id not in {lbl.job_id for lbl in load_labels(labels_path)}
    ]
    _, labeled, _, quit_, _ = session(tmp_path, remaining, Script(keys="q", answers=[]))

    assert (labeled, quit_) == (0, True)
    assert len(load_labels(labels_path)) == 1


def test_missing_log_is_skipped(tmp_path: Path) -> None:
    c = candidate(1, "o/a")
    console = Console(record=True, color_system=None)

    result = run_session(
        console, [c], tmp_path, tmp_path / "l.jsonl", lambda: "q", lambda _: "", lambda: NOW
    )

    assert (result.labeled, result.skipped, result.quit) == (0, 1, False)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "data" / DB_FILENAME)
    save_repo(conn, RepoInfo("o/a", 300, "main"), NOW)
    for job_id in (11, 12):
        c = candidate(job_id, "o/a")
        insert_failure(
            conn,
            FailedJob(
                job_id=job_id,
                repo="o/a",
                run_id=1,
                workflow_name="CI",
                workflow_path=".github/workflows/ci.yml",
                event="push",
                head_branch="main",
                head_sha="abc",
                fix_sha=None,
                failed_steps=["Build"],
                runner_os="Linux",
                created_at="2026-10-20T00:00:00Z",
                log_path=c.log_path,
                log_bytes=1,
                redactions=0,
            ),
        )
        write_log(tmp_path / "data", c)
    yield conn
    conn.close()


def test_load_candidates_excludes_labeled(db: sqlite3.Connection) -> None:
    got = load_candidates(db, exclude={11})

    assert [c.job_id for c in got] == [12]
    assert got[0].failed_steps == ["Build"]


def test_label_command_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db: sqlite3.Connection
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key-value")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    keys = iter("4" + "2" + "q")
    monkeypatch.setattr(typer, "getchar", lambda: next(keys))
    runner = CliRunner()

    first = runner.invoke(app, ["label", "--n", "5", "--seed", "1"], input="null check\n3\n")
    monkeypatch.setattr(typer, "getchar", lambda: "q")
    second = runner.invoke(app, ["label", "--n", "5"])
    get_settings.cache_clear()

    assert first.exit_code == 0, first.output
    assert "Stopped: 1 labeled, 0 skipped" in first.output
    lines = (tmp_path / "data" / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["category"] == "dart_compile"
    assert second.exit_code == 0, second.output
    assert "1 labeled so far; 1 unlabeled; picked 1." in second.output
    assert "Stopped: 0 labeled, 0 skipped (1 total" in second.output
