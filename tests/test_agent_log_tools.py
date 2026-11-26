"""Agent log tools: search_log and read_log over the full stored log. No network."""

import re
import time

import pytest

from cidoc.agent.tools import (
    READ_LOG_MAX_LINES,
    SEARCH_DEADLINE_S,
    ToolBox,
    ToolEnv,
    ToolError,
    compile_search_pattern,
    search_lines,
)
from cidoc.analyze.preprocess import MAX_LINE_CHARS
from tests.agent_fakes import FakeGitHub, context

LOG = [
    "##[group]Run flutter build apk",
    "Running Gradle task 'assembleRelease'...",
    "FAILURE: Build failed with an exception.",
    "* What went wrong:",
    "Execution failed for task ':app:minifyReleaseWithR8'.",
    "##[error]Process completed with exit code 1.",
]


def box(lines: list[str] | None = None) -> ToolBox:
    github = FakeGitHub()
    tools = ToolBox(ToolEnv(context(lines if lines is not None else LOG), github))
    return tools


# --- search_log -------------------------------------------------------------------------


def test_search_log_returns_numbered_matches_like_the_excerpt() -> None:
    out = box().call("search_log", {"pattern": r"failed (with|for)"})

    assert out.ok
    assert out.text.splitlines() == [
        "2 lines match /failed (with|for)/",
        "L3| FAILURE: Build failed with an exception.",
        "L5| Execution failed for task ':app:minifyReleaseWithR8'.",
    ]
    assert out.args == {"pattern": r"failed (with|for)", "max_matches": 20, "ignore_case": True}


def test_search_log_is_case_insensitive_unless_asked() -> None:
    tools = box()

    assert "L3|" in tools.call("search_log", {"pattern": "failure:"}).text
    sensitive = tools.call("search_log", {"pattern": "failure:", "ignore_case": False})
    assert sensitive.text == "no lines match /failure:/ in 6 log lines"


def test_search_log_searches_the_full_log_and_caps_matches() -> None:
    lines = [f"warning {n}" for n in range(1, 1001)]

    out = box(lines).call("search_log", {"pattern": "warning", "max_matches": 3})

    assert out.text.splitlines() == [
        "first 3 of 1000 lines matching /warning/",
        "L1| warning 1",
        "L2| warning 2",
        "L3| warning 3",
    ]


def test_search_log_rejects_too_many_matches_requested() -> None:
    out = box().call("search_log", {"pattern": "a", "max_matches": 500})

    assert not out.ok
    assert "max_matches" in out.text


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        (r"(a+)+b", "nested repetition"),
        (r"(?:x|xx)*y", "alternation inside a repeated group"),
        (r"(\s*foo)*", "nested repetition"),
        (r"x*x*x*y", "unbounded repeats in a row"),
        (r"x{1,1000}x{1,1000}x{1,1000}y", "unbounded repeats in a row"),
        (r".*a.*b.*c", "unbounded repeats of broad items"),
        (r"(?:a.*){3}", "unbounded repeats of broad items"),
        (r'"[^"]*"\S+[^x]*', "unbounded repeats of broad items"),
        (r"(a)\1", "backreferences"),
        (r"(a", "invalid regex"),
        ("", "pattern is empty"),
        ("a" * 201, "longer than 200"),
    ],
)
def test_catastrophic_and_invalid_patterns_are_refused(pattern: str, message: str) -> None:
    with pytest.raises(ToolError, match=re.escape(message)):
        compile_search_pattern(pattern, ignore_case=True)


@pytest.mark.parametrize(
    "pattern",
    [
        r"error.*failed",
        r".*error.*",
        r"\w+:\w+:\w+:\w+;",
        r"(\d{1,3}\.){3}\d+",
        r"(?:error|warning): .*",
        r"[a-z_]+\.dart:\d+:\d+",
        r"Xcode \d+(\.\d+)?",
    ],
)
def test_ordinary_patterns_are_accepted(pattern: str) -> None:
    compile_search_pattern(pattern, ignore_case=True)


def test_accepted_worst_case_patterns_are_bounded_per_line_and_in_total() -> None:
    lines = ["ab" * 2500] * 400  # matched on their first 500 chars only
    regex = compile_search_pattern(r".*a.*c", ignore_case=False)  # no "c": worst case

    started = time.monotonic()
    regex.search(lines[0][:MAX_LINE_CHARS])
    one_line = time.monotonic() - started
    found = search_lines(lines, regex, max_matches=20)
    elapsed = time.monotonic() - started

    assert one_line < 0.2
    assert found.total == 0
    assert elapsed < SEARCH_DEADLINE_S + 0.5


def test_search_stops_at_the_deadline() -> None:
    ticks = iter(range(100))
    regex = compile_search_pattern("needle", ignore_case=True)

    found = search_lines(["hay"] * 50, regex, 20, deadline_s=5, clock=lambda: next(ticks))

    assert found.stopped_at == 6
    assert found.total == 0


def test_a_refused_pattern_is_an_error_text_not_an_exception() -> None:
    out = box().call("search_log", {"pattern": r"(a+)+$"})

    assert not out.ok
    assert out.text.startswith("error: nested repetition")


# --- read_log ---------------------------------------------------------------------------


def test_read_log_returns_raw_numbered_lines() -> None:
    out = box().call("read_log", {"start_line": 3, "end_line": 4})

    assert out.text.splitlines() == [
        "log lines 3-4 of 6",
        "L3| FAILURE: Build failed with an exception.",
        "L4| * What went wrong:",
    ]


def test_read_log_returns_at_most_150_lines() -> None:
    lines = [f"line {n}" for n in range(1, 1001)]

    out = box(lines).call("read_log", {"start_line": 10, "end_line": 900})

    rows = out.text.splitlines()
    assert rows[0] == f"log lines 10-{9 + READ_LOG_MAX_LINES} of 1000"
    assert rows[-2] == f"L{9 + READ_LOG_MAX_LINES}| line {9 + READ_LOG_MAX_LINES}"
    assert rows[-1] == f"(at most 150 lines per call; continue at {10 + READ_LOG_MAX_LINES})"


def test_read_log_clamps_to_the_end_of_the_log() -> None:
    out = box().call("read_log", {"start_line": 5, "end_line": 99})

    assert out.text.splitlines()[0] == "log lines 5-6 of 6"
    assert "continue" not in out.text


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"start_line": 7, "end_line": 8}, "past the end (the log has 6 lines)"),
        ({"start_line": 4, "end_line": 2}, "end_line is before start_line"),
        ({"start_line": 0, "end_line": 2}, "start_line"),
        ({"start_line": 1}, "end_line"),
    ],
)
def test_read_log_bad_ranges_are_errors(args: dict[str, int], message: str) -> None:
    out = box().call("read_log", args)

    assert not out.ok
    assert message in out.text


def test_log_tools_work_without_run_metadata() -> None:
    env = ToolEnv(context(LOG, repo=None, run_id=None, head_sha=None), None)

    assert ToolBox(env).call("read_log", {"start_line": 1, "end_line": 1}).ok
