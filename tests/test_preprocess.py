import pytest

from cidoc.analyze.preprocess import (
    MARKER_CONTEXT,
    Line,
    collapse_repeats,
    preprocess,
)
from cidoc.logview import FULL_STEP_MAX_LINES, is_marker

TS = "2026-10-01T12:34:56.1234567Z "


def log(*lines: str) -> str:
    return "".join(f"{TS}{line}\n" for line in lines)


def kept(text: str) -> dict[int, str]:
    """Line number -> rendered line, for every kept line."""
    return {line.number: line.render() for block in preprocess(text).blocks for line in block}


def failing_step(*body: str) -> str:
    return log(
        "##[group]Run flutter build apk",  # 1
        "flutter build apk",  # 2
        "##[endgroup]",  # 3
        *body,  # 4..
        "##[error]Process completed with exit code 1.",
        "Post job cleanup.",
        "##[error]after cleanup",
    )


def filler(n: int, prefix: str = "compiling") -> list[str]:
    return [f"{prefix} unit {i}" for i in range(n)]


# --- cleaning and line numbers ------------------------------------------------------


def test_keeps_original_line_numbers_and_strips_ansi_and_timestamps() -> None:
    text = log(
        "##[group]Run actions/checkout@v4",  # 1
        "##[endgroup]",  # 2
        *filler(20, "checkout"),  # 3..22, more than 15 lines before the error
        "##[group]Run flutter test",  # 23
        "##[endgroup]",  # 24
        "\x1b[31mSome tests failed.\x1b[0m",  # 25
        "##[error]Process completed with exit code 1.",  # 26
        "Post job cleanup.",  # 27
    )

    out = preprocess(text)

    assert out.failed_titles == ["Run flutter test"]
    assert out.text == "\n".join(
        [
            "... (10 lines omitted)",
            *(f"L{n}| checkout unit {n - 3}" for n in range(11, 23)),
            "L23| ##[group]Run flutter test",
            "L24| ##[endgroup]",
            "L25| Some tests failed.",
            "L26| ##[error]Process completed with exit code 1.",
            "... (1 lines omitted)",
        ]
    )
    assert TS.strip() not in out.text


def test_nothing_after_post_job_cleanup() -> None:
    lines = kept(failing_step("boom"))

    assert all("after cleanup" not in line for line in lines.values())


@pytest.mark.parametrize(
    "marker",
    [
        "##[error]x",
        "FAILURE: Build failed",
        "BUILD FAILED in 3s",
        "lib/a.dart:1:2: error: x",
        "Error: x",
        "java.lang.IllegalStateException",
        "test FAILED",
        "Could not resolve com.x:y:1.0",
        "Could not find com.x:y:1.0",
        "version solving failed.",
        "❌  error: no such module",
        "** ARCHIVE FAILED **",
        "** BUILD FAILED **",
        "[!] CocoaPods could not find compatible versions",
        "fastlane finished with errors",
        "💥",
        "Error (Xcode): No profiles",
        "ERROR ITMS-90717",
        "remote: error: GH013",
        "remote: fatal: denied",
        "Received status code 401 from server",
    ],
)
def test_error_markers(marker: str) -> None:
    assert is_marker(marker)


@pytest.mark.parametrize(
    "line",
    ["Target Integrity (Xcode): error: x", "Warning (Xcode): Error: deprecated"],
)
def test_xcode_warnings_are_not_markers(line: str) -> None:
    assert not is_marker(line)


def test_long_failed_step_keeps_markers_with_15_lines_of_context() -> None:
    before = filler(FULL_STEP_MAX_LINES, "before")
    after = filler(FULL_STEP_MAX_LINES, "after")
    text = failing_step(*before, "e: Unresolved reference: foo error: x", *after)
    marker = 4 + len(before)

    lines = kept(text)

    assert "error: x" in lines[marker]
    assert all(n in lines for n in range(marker - MARKER_CONTEXT, marker + MARKER_CONTEXT + 1))
    assert marker - MARKER_CONTEXT - 1 not in lines
    assert marker + MARKER_CONTEXT + 1 not in lines


def test_without_a_failed_step_keeps_markers_and_the_tail() -> None:
    body = [*filler(200, "a"), "FAILURE: Build failed", *filler(300, "b")]
    marker = 201

    lines = kept(log(*body))

    assert "FAILURE" in lines[marker]
    assert marker - MARKER_CONTEXT in lines
    assert marker - MARKER_CONTEXT - 1 not in lines
    assert len(body) in lines  # the tail


# --- noise --------------------------------------------------------------------------

WELCOME = [
    "  ╔════════════════════════════════════════╗",
    "  ║   Welcome to Flutter! - https://flutter.dev   ║",
    "  ║ The Flutter tool uses Google Analytics  ║",
    "  ║ to anonymously report feature usage.   ║",
    "  ╚════════════════════════════════════════╝",
]


def test_drops_welcome_to_flutter_banner() -> None:
    lines = kept(failing_step(*WELCOME, "Building apk", "error: boom"))

    assert not any("║" in line or "╔" in line or "╚" in line for line in lines.values())
    assert "Building apk" in lines[4 + len(WELCOME)]


@pytest.mark.parametrize("prefix", ["", "    ", "[example]: ", "[  +12 ms] "])
def test_drops_pub_outdated_list_but_keeps_resolution_status(prefix: str) -> None:
    pub = [
        "Resolving dependencies...",
        "  async 2.11.0 (2.12.0 available)",
        "> http 1.2.0 (was 1.1.0) (1.3.0 available)",
        "+ meta 1.16.0",
        "! vector_math 2.2.0 (overridden) (2.4.3 available)",
        "Changed 3 dependencies!",
        "4 packages have newer versions incompatible with dependency constraints.",
        "Try `flutter pub outdated` for more information.",
        "lib/main.dart:3:1: Error: oops",
    ]
    lines = kept(failing_step(*(prefix + line for line in pub)))
    shown = "\n".join(lines.values())

    for gone in ("async", "http", "meta", "vector_math", "newer versions", "pub outdated"):
        assert gone not in shown
    assert "Resolving dependencies..." in shown
    assert "Changed 3 dependencies!" in shown
    assert "Error: oops" in shown


def test_drops_node_deprecation_warnings() -> None:
    text = failing_step(
        "(node:2716) [DEP0040] DeprecationWarning: The `punycode` module is deprecated.",
        "(Use `node --trace-deprecation ...` to show where the warning was created)",
        "Error: upload failed",
    )

    shown = "\n".join(kept(text).values())

    assert "DeprecationWarning" not in shown
    assert "trace-deprecation" not in shown
    assert "Error: upload failed" in shown


def test_fastlane_changelog_dropped_and_40_lines_above_kept() -> None:
    # A long step, so it is condensed and only rule-selected lines are kept.
    head = filler(FULL_STEP_MAX_LINES, "gym")
    lead = filler(45, "lead")
    notice = [
        "#######################################################################",
        "# fastlane 2.228.0 is available. You are on 2.227.0.",
        "# You should use the latest version.",
        "#######################################################################",
        *(f"2.228.0 Improvement {i}" for i in range(30)),
        "To see all new releases, open https://github.com/fastlane/fastlane/releases",
        "Please update using `bundle update fastlane`",
    ]
    text = failing_step(*head, *lead, *notice)
    notice_start = 4 + len(head) + len(lead)

    lines = kept(text)
    shown = "\n".join(lines.values())

    assert "fastlane 2.228.0 is available" not in shown
    assert "Improvement" not in shown
    assert "To see all new releases" not in shown
    assert "Please update using" not in shown
    assert all(n in lines for n in range(notice_start - 40, notice_start))
    assert notice_start - 41 not in lines


# --- cascade collapse ---------------------------------------------------------------


def test_collapses_consecutive_identical_lines() -> None:
    text = failing_step(
        "lib/a.dart:1:1: Error: Type 'Foo' not found.",
        *["warning: unused import"] * 5,
        "lib/a.dart:1:1: Error: Type 'Foo' not found.",
    )

    out = preprocess(text).text

    assert "L5| warning: unused import (repeated 5 times)" in out
    assert out.count("warning: unused import") == 1
    # The same line again later, after something else, is not merged.
    assert "L4| lib/a.dart:1:1: Error" in out
    assert "L10| lib/a.dart:1:1: Error" in out
    # The gap after the collapsed run is not reported as omitted lines.
    assert "lines omitted)\nL10|" not in out


def test_collapse_only_merges_adjacent_line_numbers() -> None:
    lines = list(collapse_repeats([(1, "x"), (2, "x"), (5, "x"), (6, "y")]))

    assert lines == [Line(1, "x", 2), Line(5, "x"), Line(6, "y")]
