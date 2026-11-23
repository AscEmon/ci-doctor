import gzip
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.labeling.labels import (
    LabelNotFoundError,
    find_label,
    load_labels,
    replace_label_line,
)
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.runs import FailedJob, insert_failure

NOW = datetime(2026, 10, 31, tzinfo=UTC)
LOG = "".join(f"2026-10-01T00:00:00Z line {i}\n" for i in range(1, 11))


def record(job_id: int, **extra: object) -> dict[str, object]:
    return {
        "job_id": job_id,
        "repo": "o/a",
        "run_id": job_id,
        "category": "test_failure",
        "root_cause": f"cause {job_id}",
        "evidence_lines": [3],
        "confidence": 2,
        "log_path": f"logs/o__a/{job_id}.log.gz",
        "labeled_at": "2026-10-30T00:00:00+00:00",
        "hint": "test_failure",
        "hint_accepted": True,
        "edited_at": None,
        **extra,
    }


# Two labels, a line a crash cut short, and one with a field from a future version.
ORIGINAL_LINES = [
    json.dumps(record(11)) + "\n",
    '{"job_id": 99, "repo": "o/a", "categ\n',
    json.dumps(record(12, reviewer="someone")) + "\n",
    json.dumps(record(13)) + "\n",
]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    data_dir = tmp_path / "data"
    conn = connect(data_dir / DB_FILENAME)
    save_repo(conn, RepoInfo("o/a", 300, "main"), NOW)
    for job_id in (11, 12, 13):
        log_path = data_dir / f"logs/o__a/{job_id}.log.gz"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(gzip.compress(LOG.encode()))
        if job_id == 13:
            continue  # pruned from the DB; label and log remain
        insert_failure(
            conn,
            FailedJob(
                job_id=job_id,
                repo="o/a",
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
                log_path=f"logs/o__a/{job_id}.log.gz",
                log_bytes=1,
                redactions=0,
            ),
        )
    conn.close()
    (data_dir / "labels.jsonl").write_text("".join(ORIGINAL_LINES), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")  # fake, tests only
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    yield data_dir
    get_settings.cache_clear()


def edit(
    monkeypatch: pytest.MonkeyPatch, args: list[str], keys: str, answers: str = ""
) -> tuple[int, str]:
    pressed = iter(keys)
    monkeypatch.setattr(typer, "getchar", lambda: next(pressed))
    result = CliRunner().invoke(app, ["label", "edit", *args], input=answers, terminal_width=200)
    return result.exit_code, result.output


def lines(data_dir: Path) -> list[str]:
    return (data_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)


def test_edit_last_changes_fields_and_keeps_everything_else(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # category 4 (dart_compile), new root cause, new evidence, confidence 3.
    code, out = edit(monkeypatch, ["--last"], "43", "missing import in main.dart\n5, 2\n")

    assert code == 0, out
    assert "Saved: changed category, root_cause, evidence_lines, confidence for job 13" in out
    after = lines(env)
    assert after[:3] == ORIGINAL_LINES[:3]  # byte for byte, bad line included
    edited = json.loads(after[3])
    assert edited == {
        **record(13),
        "category": "dart_compile",
        "root_cause": "missing import in main.dart",
        "evidence_lines": [2, 5],
        "confidence": 3,
        "edited_at": edited["edited_at"],
    }
    assert edited["hint"] == "test_failure"
    assert edited["hint_accepted"] is True  # reflects the original decision, not the edit
    assert datetime.fromisoformat(edited["edited_at"]).tzinfo is not None


def test_edit_by_job_id_enter_keeps_values_and_unknown_fields(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Enter keeps category, new root cause, Enter keeps evidence, Enter keeps confidence.
    code, out = edit(monkeypatch, ["12"], "\r\r", "flaky golden on macOS\n\n")

    assert code == 0, out
    assert "changed root_cause for job 12" in out
    after = lines(env)
    assert [after[0], after[1], after[3]] == [
        ORIGINAL_LINES[0],
        ORIGINAL_LINES[1],
        ORIGINAL_LINES[3],
    ]
    edited = json.loads(after[2])
    assert edited["root_cause"] == "flaky golden on macOS"
    assert (edited["category"], edited["evidence_lines"], edited["confidence"]) == (
        "test_failure",
        [3],
        2,
    )
    assert edited["reviewer"] == "someone"


def test_unchanged_edit_writes_nothing(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = edit(monkeypatch, ["11"], "\r\r", "\n\n")

    assert code == 0, out
    assert "No changes; label unchanged." in out
    assert lines(env) == ORIGINAL_LINES


def test_quit_leaves_file_unchanged(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = edit(monkeypatch, ["11"], "q")

    assert code == 0
    assert "Quit: label unchanged." in out
    assert lines(env) == ORIGINAL_LINES


def test_unknown_job_id_is_an_error(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = edit(monkeypatch, ["424242"], "")

    assert code == 1
    assert "no label for job 424242" in out
    assert lines(env) == ORIGINAL_LINES


@pytest.mark.parametrize("args", [[], ["11", "--last"]])
def test_needs_exactly_one_of_job_id_or_last(
    env: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    code, out = edit(monkeypatch, args, "")

    assert code == 2
    assert "Give a job id or --last" in out


def test_file_stays_valid_jsonl_after_edits(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    edit(monkeypatch, ["11"], "02", "runner lost connection\n\n")
    edit(monkeypatch, ["--last"], "o1", "unclear\n1\n")

    after = lines(env)
    assert len(after) == len(ORIGINAL_LINES)
    assert all(line.endswith("\n") for line in after)
    parsed = [json.loads(line) for i, line in enumerate(after) if i != 1]  # 1: the bad line
    assert [p["job_id"] for p in parsed] == [11, 12, 13]
    assert [lbl.job_id for lbl in load_labels(env / "labels.jsonl")] == [11, 12, 13]
    assert not list(env.glob("*.tmp"))


def test_labeling_session_still_runs_without_subcommand(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(typer, "getchar", lambda: "q")  # never block on a real keypress
    result = CliRunner().invoke(app, ["label", "--n", "1"])

    assert result.exit_code == 0, result.output
    assert "Nothing left to label." in result.output


# --- store helpers --------------------------------------------------------------------


def test_find_label(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(ORIGINAL_LINES), encoding="utf-8")

    assert find_label(path).line_index == 3
    assert find_label(path, 12).record["reviewer"] == "someone"
    with pytest.raises(LabelNotFoundError, match="no label for job 99"):
        find_label(path, 99)  # only on the unreadable line
    with pytest.raises(LabelNotFoundError, match="does not exist"):
        find_label(tmp_path / "missing.jsonl")


def test_replace_label_line_is_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(ORIGINAL_LINES), encoding="utf-8")

    def crash(_src: object, _dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("cidoc.labeling.labels.os.replace", crash)
    with pytest.raises(OSError, match="disk full"):
        replace_label_line(path, 0, record(11, category="other"))

    # The original is untouched if the rename never happens.
    assert path.read_text(encoding="utf-8") == "".join(ORIGINAL_LINES)


def test_load_labels_ignores_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(record(12, reviewer="someone")) + "\n", encoding="utf-8")

    [label] = load_labels(path)

    assert label.job_id == 12
