"""Agent tools: repository reads at the failing commit. Fake GitHub, no network."""

from pathlib import Path
from typing import Any

import pytest

from cidoc.agent.context import load_context
from cidoc.agent.tools import (
    LIST_DIR_MAX_ENTRIES,
    TRUNCATED,
    ToolBox,
    ToolEnv,
    ToolError,
    UnknownToolError,
    cap_output,
    select_tools,
    validate_repo_path,
)
from cidoc.db import connect
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.runs import FailedJob, insert_failure
from tests.agent_fakes import (
    FIX_SHA,
    SHA,
    WORKFLOW,
    FakeGitHub,
    contents,
    context,
    file_entry,
    http_error,
)
from tests.datadir import NOW


def box(github: FakeGitHub | None = None, **kw: Any) -> ToolBox:
    return ToolBox(ToolEnv(context(), github if github is not None else FakeGitHub()), **kw)


# --- path validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pubspec.yaml", "pubspec.yaml"),
        ("./android//app/build.gradle", "android/app/build.gradle"),
        ("ios/Podfile ", "ios/Podfile"),
        ("a/./b", "a/b"),
    ],
)
def test_valid_paths_are_normalized(path: str, expected: str) -> None:
    assert validate_repo_path(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "~/.ssh/id_rsa",
        "C:/Windows/win.ini",
        "../secrets",
        "lib/../../x",
        "lib/..",
        "a\\b",
        "https://example.com/x",
        "file:///etc/passwd",
        "lib/\x00main.dart",
        "",
        ".",
    ],
)
def test_paths_that_could_leave_the_repository_are_rejected(path: str) -> None:
    with pytest.raises(ToolError):
        validate_repo_path(path)


def test_root_is_allowed_only_where_asked() -> None:
    assert validate_repo_path(".", allow_root=True) == ""


def test_rejected_paths_make_no_request() -> None:
    github = FakeGitHub()

    out = box(github).call("read_file", {"path": "../../etc/passwd"})

    assert not out.ok
    assert out.text.startswith("error: ")
    assert github.calls == []


def test_query_characters_stay_in_the_path() -> None:
    github = FakeGitHub()

    box(github).call("read_file", {"path": "a?ref=main#x"})

    assert github.calls == [(contents("a%3Fref%3Dmain%23x"), {"ref": SHA})]


# --- read_file ------------------------------------------------------------------------


def test_read_file_reads_at_the_failing_commit_with_line_numbers() -> None:
    github = FakeGitHub({contents("pubspec.yaml"): file_entry("name: app\nversion: 1.0.0\n")})

    out = box(github).call("read_file", {"path": "pubspec.yaml"})

    assert out.ok
    assert github.calls == [(contents("pubspec.yaml"), {"ref": SHA})]
    assert out.text.splitlines() == [
        f"pubspec.yaml @ {SHA[:12]}: lines 1-2 of 2",
        "1| name: app",
        "2| version: 1.0.0",
    ]
    assert out.args == {"path": "pubspec.yaml", "start_line": 1, "end_line": 200}


def test_read_file_returns_at_most_200_lines_and_says_how_to_continue() -> None:
    text = "".join(f"line {n}\n" for n in range(1, 501))
    github = FakeGitHub({contents("big.txt"): file_entry(text)})

    out = box(github).call("read_file", {"path": "big.txt", "start_line": 101, "end_line": 900})

    rows = out.text.splitlines()
    assert rows[0].endswith("lines 101-300 of 500")
    assert rows[1] == "101| line 101"
    assert rows[200] == "300| line 300"
    assert rows[-1] == "(200 more lines; read_file start_line=301)"


def test_read_file_stops_at_16_kb() -> None:
    text = "".join(f"{'x' * 199}\n" for _ in range(150))  # 30 KB in 150 lines
    github = FakeGitHub({contents("wide.txt"): file_entry(text)})

    out = box(github, cap_tokens=100_000).call("read_file", {"path": "wide.txt"})

    body = out.text.split("\n", 1)[1]
    assert len(body.encode()) <= 16 * 1024 + 100
    assert "read_file start_line=" in out.text.splitlines()[-1]


def test_read_file_rejects_binary_files() -> None:
    github = FakeGitHub({contents("logo.png"): file_entry(b"\x89PNG\r\n\x1a\n\x00\x00\x00")})

    out = box(github).call("read_file", {"path": "logo.png"})

    assert not out.ok
    assert "binary file" in out.text
    assert "PNG" not in out.text


def test_read_file_reports_files_too_large_for_the_api() -> None:
    entry = {"type": "file", "size": 5_000_000, "encoding": "none", "content": ""}
    github = FakeGitHub({contents("huge.json"): entry})

    out = box(github).call("read_file", {"path": "huge.json"})

    assert out.text == "error: huge.json is too large to read (5000000 bytes)"


def test_read_file_on_a_directory_points_to_list_dir() -> None:
    github = FakeGitHub({contents("lib"): [{"type": "file", "name": "main.dart"}]})

    assert "use list_dir" in box(github).call("read_file", {"path": "lib"}).text


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"path": "a", "start_line": 0}, "start_line"),
        ({"path": "a", "start_line": 5, "end_line": 2}, "end_line is before start_line"),
        ({"start_line": 1}, "path"),
        ({"path": "a", "mode": "w"}, "mode"),
    ],
)
def test_read_file_bad_arguments_are_errors(args: dict[str, Any], message: str) -> None:
    github = FakeGitHub({contents("a"): file_entry("x\n")})

    out = box(github).call("read_file", args)

    assert not out.ok
    assert message in out.text


def test_read_file_past_the_end() -> None:
    github = FakeGitHub({contents("a"): file_entry("x\ny\n")})

    out = box(github).call("read_file", {"path": "a", "start_line": 3})

    assert out.text == "error: start_line 3 is past the end (a has 2 lines)"


# --- list_dir and read_workflow -------------------------------------------------------


def test_list_dir_lists_name_type_and_size() -> None:
    entries = [
        {"type": "dir", "name": "android", "size": 0},
        {"type": "file", "name": "pubspec.yaml", "size": 812},
    ]
    github = FakeGitHub({contents(""): entries})

    out = box(github).call("list_dir", {})

    assert github.calls == [(contents(""), {"ref": SHA})]
    assert out.text.splitlines() == [
        f". @ {SHA[:12]}: 2 entries",
        "dir     android",
        "file    pubspec.yaml (812 bytes)",
    ]


def test_list_dir_caps_entries() -> None:
    entries = [{"type": "file", "name": f"f{n}", "size": 1} for n in range(250)]
    github = FakeGitHub({contents("assets"): entries})

    out = box(github, cap_tokens=100_000).call("list_dir", {"path": "assets"})

    rows = out.text.splitlines()
    assert len(rows) == 1 + LIST_DIR_MAX_ENTRIES + 1
    assert rows[-1] == "... (50 more entries not shown)"


def test_read_workflow_reads_the_runs_workflow_in_full() -> None:
    text = "".join(f"step {n}\n" for n in range(1, 301))
    github = FakeGitHub({contents(WORKFLOW): file_entry(text)})

    out = box(github, cap_tokens=100_000).call("read_workflow", {})

    rows = out.text.splitlines()
    assert rows[0] == f"{WORKFLOW} @ {SHA[:12]}: 300 lines"
    assert rows[-1] == "300| step 300"


# --- errors, caps, redaction ----------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (http_error(404), "error: not found (HTTP 404) at this commit"),
        (http_error(403, retryable=True), "error: GitHub rate limit reached"),
        (http_error(502, retryable=True), "error: GitHub server error (HTTP 502)"),
        (http_error(401), "error: GitHub returned HTTP 401"),
        (RuntimeError("boom"), "error: RuntimeError: boom"),
    ],
)
def test_github_errors_become_short_error_texts(error: Exception, message: str) -> None:
    github = FakeGitHub({contents("pubspec.yaml"): error})

    out = box(github).call("read_file", {"path": "pubspec.yaml"})

    assert not out.ok
    assert out.text.startswith(message)


def test_github_tools_without_run_metadata_say_so() -> None:
    env = ToolEnv(context(repo=None, run_id=None, head_sha=None), FakeGitHub())

    out = ToolBox(env).call("read_workflow", {})

    assert out.text.startswith("error: no run metadata for this job")


def test_unknown_and_disabled_tools_are_errors() -> None:
    tools = ToolBox(ToolEnv(context(), FakeGitHub()), names=["list_dir"])

    assert tools.names == ["list_dir"]
    assert tools.call("rm_rf", {}).text.startswith("error: unknown tool 'rm_rf'")
    assert not tools.call("read_file", {"path": "a"}).ok


def test_select_tools_rejects_unknown_names() -> None:
    with pytest.raises(UnknownToolError, match="nope"):
        select_tools(["read_file", "nope"])


def test_output_is_capped_with_a_truncated_marker() -> None:
    text = "".join(f"{'y' * 70}\n" for _ in range(200))
    github = FakeGitHub({contents("a"): file_entry(text)})

    out = box(github, cap_tokens=500).call("read_file", {"path": "a"})

    assert len(out.text) <= 500 * 4
    assert out.text.endswith(TRUNCATED)


def test_cap_output_leaves_short_text_alone() -> None:
    assert cap_output("short", 10) == "short"


def test_tool_output_is_redacted() -> None:
    token = "ghp" + "_" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0UvWxYz"[:36]
    github = FakeGitHub({contents("ci.sh"): file_entry(f"export TOKEN={token}\n")})

    out = box(github).call("read_file", {"path": "ci.sh"})

    assert token not in out.text
    assert "[REDACTED:" in out.text


# --- context ----------------------------------------------------------------------------


def test_context_never_loads_fix_sha(tmp_path: Path) -> None:
    conn = connect(tmp_path / "cidoc.db")
    save_repo(conn, RepoInfo("o/a", 300, "main"), NOW)
    insert_failure(
        conn,
        FailedJob(
            job_id=7,
            repo="o/a",
            run_id=70,
            workflow_name="CI",
            workflow_path=WORKFLOW,
            event="push",
            head_branch="main",
            head_sha=SHA,
            fix_sha=FIX_SHA,
            failed_steps=[],
            runner_os="Linux",
            created_at="2026-10-20T00:00:00Z",
            log_path="logs/o__a/7.log.gz",
            log_bytes=1,
            redactions=0,
        ),
    )

    ctx = load_context(conn, 7, "2026-10-01T00:00:00.0Z hello\n")

    assert ctx.head_sha == SHA
    assert ctx.log_lines == ["hello"]
    assert FIX_SHA not in repr(ctx)
    assert load_context(conn, 8, "x\n").has_run is False
