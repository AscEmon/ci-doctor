"""Agent log tools: search_log and read_log over the full stored log. No network."""

import re
import time

import pytest

from cidoc.agent.tools import (
    READ_LOG_MAX_LINES,
    SEARCH_DEADLINE_S,
    TOOLS,
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

STEPS = [
    "##[group]Run actions/checkout@v4",  # 1
    "[command]/usr/bin/git fetch --depth=1 origin",  # 2
    "error: checkout printed this but the step passed",  # 3
    "##[endgroup]",  # 4
    "##[group]Run flutter test",  # 5  failed step starts
    "flutter test",  # 6
    "shell: /usr/bin/bash --noprofile --norc -e -o pipefail {0}",  # 7
    "env:",  # 8
    "  FLUTTER_ROOT: /opt/hostedtoolcache/flutter",  # 9
    "  FAIL_FAST: true",  # 10
    "##[endgroup]",  # 11
    "00:01 +1: loads config",  # 12
    "Expected: <3>",  # 13
    "  Actual: <4>",  # 14
    "00:02 +1 -1: Some tests failed.",  # 15
    "Connection reset: ECONNRESET",  # 16
    "00:02 +1 -1: Some tests failed.",  # 17
    "##[error]Process completed with exit code 1.",  # 18
    "Post job cleanup.",  # 19
    "Expected: something during cleanup",  # 20
]


def search(pattern: str, lines: list[str] | None = None, **args: object) -> str:
    ctx = context(lines if lines is not None else STEPS, failed_steps=["Run flutter test"])
    tools = ToolBox(ToolEnv(ctx, None))
    out = tools.call("search_log", {"pattern": pattern, **args})
    assert out.ok, out.text
    return out.text


def test_search_log_returns_numbered_matches_like_the_excerpt() -> None:
    out = box().call("search_log", {"pattern": r"failed (with|for)"})

    assert out.ok
    assert out.text.splitlines()[1:] == [
        "L5| Execution failed for task ':app:minifyReleaseWithR8'.",
        "L3| FAILURE: Build failed with an exception.",
    ]  # closest to the ##[error] on line 6 first
    assert out.args == {
        "pattern": r"failed (with|for)",
        "max_matches": 20,
        "ignore_case": True,
        "scope": "failed_step",
    }


def test_the_failed_step_is_searched_by_default() -> None:
    text = search("Expected:")

    assert text.splitlines() == [
        "1 matching lines (1 distinct) for /Expected:/ in the failed step"
        " (Run flutter test); showing 1 of 1, closest to an ##[error] first",
        "L13| Expected: <3>",
    ]  # not line 20: nothing after "Post job cleanup." is searched


def test_scope_all_searches_the_whole_job_before_cleanup() -> None:
    failed_step = search("error")
    everything = search("error", scope="all")

    assert "L3|" not in failed_step
    assert "L3| error: checkout printed this but the step passed" in everything
    assert "in the whole log" in everything
    assert "Expected: something during cleanup" not in search("Expected", scope="all")


def test_step_headers_commands_shell_and_env_lines_never_match() -> None:
    text = search("flutter|git|bash|FLUTTER_ROOT|checkout", scope="all")

    assert "L6| flutter test" in text  # the command itself is kept
    assert "L3|" in text
    for noise in ("L1|", "L2|", "L5|", "L7|", "L9|"):
        assert noise not in text


def test_fail_does_not_match_pipefail_or_env_names() -> None:
    text = search("fail")

    assert "pipefail" not in text
    assert "FAIL_FAST" not in text
    assert "L15| 00:02 +1 -1: Some tests failed. (x2)" in text
    assert search("pipefail").startswith("no lines match /pipefail/")
    assert search(r"\bfail\b").startswith("no lines match")  # "failed" is not "fail"


def test_indented_test_output_is_not_mistaken_for_env() -> None:
    assert "L14|   Actual: <4>" in search("Actual:")


def test_identical_lines_are_merged_and_ordered_by_distance_to_the_error() -> None:
    text = search("failed|ECONNRESET|Expected")

    assert text.splitlines() == [
        "4 matching lines (3 distinct) for /failed|ECONNRESET|Expected/ in the failed step"
        " (Run flutter test); showing 3 of 3, closest to an ##[error] first",
        "L16| Connection reset: ECONNRESET",
        "L15| 00:02 +1 -1: Some tests failed. (x2)",
        "L13| Expected: <3>",
    ]


def test_total_versus_shown_is_always_reported() -> None:
    text = search("failed|ECONNRESET|Expected", max_matches=1)

    assert "4 matching lines (3 distinct)" in text.splitlines()[0]
    assert "showing 1 of 3" in text.splitlines()[0]
    assert len(text.splitlines()) == 2


def test_a_log_without_errors_or_failed_steps_is_searched_in_order() -> None:
    lines = [f"warning {n}" for n in range(1, 1001)]

    out = box(lines).call("search_log", {"pattern": "warning", "max_matches": 3})

    assert out.text.splitlines() == [
        "1000 matching lines (1000 distinct) for /warning/ in the whole log (no failed step"
        " found); showing 3 of 1000, closest to an ##[error] first",
        "L1| warning 1",
        "L2| warning 2",
        "L3| warning 3",
    ]


def test_search_log_is_case_insensitive_unless_asked() -> None:
    assert "L13|" in search("expected:")
    assert search("expected:", ignore_case=False).startswith("no lines match")


def test_search_log_rejects_bad_arguments() -> None:
    for args in ({"pattern": "a", "max_matches": 500}, {"pattern": "a", "scope": "repo"}):
        out = box().call("search_log", args)
        assert not out.ok
        assert "invalid arguments" in out.text


def test_the_description_teaches_good_patterns() -> None:
    description = TOOLS["search_log"].description

    for hint in ("Expected:|Actual:", "Exception|Error:", r"\[E\]|Some tests failed",
                 "Could not|timed out|429|ECONNRESET", "pipefail", r"\bfail\b"):  # fmt: skip
        assert hint in description


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
    found = search_lines(lines, regex)
    elapsed = time.monotonic() - started

    assert one_line < 0.2
    assert found.matches == []
    assert elapsed < SEARCH_DEADLINE_S + 0.5


def test_search_stops_at_the_deadline() -> None:
    ticks = iter(range(100))
    regex = compile_search_pattern("needle", ignore_case=True)

    found = search_lines(["hay"] * 50, regex, deadline_s=5, clock=lambda: next(ticks))

    assert found.stopped_at == 5  # 0-based: the sixth line
    assert found.matches == []


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
