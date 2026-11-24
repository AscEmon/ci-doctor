import pytest

from cidoc.analyze.preprocess import (
    DEFAULT_TOKEN_BUDGET,
    LINES_BEFORE_ERROR,
    MARKER_CONTEXT,
    MAX_LINE_CHARS,
    Line,
    collapse_repeats,
    estimate_tokens,
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
        *filler(90, "checkout"),  # 3..92, more than 80 lines before the error
        "##[group]Run flutter test",  # 93
        "##[endgroup]",  # 94
        "\x1b[31mSome tests failed.\x1b[0m",  # 95
        "##[error]Process completed with exit code 1.",  # 96
        "Post job cleanup.",  # 97
    )

    out = preprocess(text)

    assert out.failed_titles == ["Run flutter test"]
    assert out.text == "\n".join(
        [
            "... (15 lines omitted)",
            *(f"L{n}| checkout unit {n - 3}" for n in range(16, 93)),
            "L93| ##[group]Run flutter test",
            "L94| ##[endgroup]",
            "L95| Some tests failed.",
            "L96| ##[error]Process completed with exit code 1.",
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
        # Longer than the 80 lines kept above ##[error], so those all fall inside it.
        *(f"2.228.0 Improvement {i}" for i in range(90)),
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


# --- token budget -------------------------------------------------------------------


def test_estimate_tokens_is_char_based() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_default_budget_holds_for_a_huge_log() -> None:
    body = [f"lib/f{i}.dart:{i}:1: Error: undefined name 'x{i}'" for i in range(5000)]
    text = failing_step(*body)

    out = preprocess(text)

    assert out.budget_tokens == DEFAULT_TOKEN_BUDGET
    assert out.estimated_tokens <= DEFAULT_TOKEN_BUDGET
    assert out.dropped_windows > 0


@pytest.mark.parametrize("budget", [500, 1500, 3000])
def test_output_fits_the_budget(budget: int) -> None:
    body = [*filler(100, "a"), *("Error: x" + str(i) for i in range(300)), *filler(100, "b")]

    out = preprocess(failing_step(*body), budget_tokens=budget)

    assert out.estimated_tokens <= budget


def test_lines_before_error_are_kept_even_over_budget() -> None:
    body = filler(FULL_STEP_MAX_LINES + 100)
    text = failing_step(*body)
    error = 4 + len(body)

    lines = kept_with(text, budget_tokens=10)

    assert set(lines) == set(range(error - LINES_BEFORE_ERROR, error + 1))


def kept_with(text: str, budget_tokens: int) -> dict[int, str]:
    out = preprocess(text, budget_tokens=budget_tokens)
    return {line.number: line.render() for block in out.blocks for line in block}


def test_lowest_priority_windows_are_dropped_first() -> None:
    # A long failed step: head, an early marker, a later marker, then 100 lines of
    # output before ##[error], so neither marker is inside the always-kept lead.
    body = [
        *filler(60, "a"),
        "FAILURE: first",
        *filler(60, "b"),
        "FAILURE: second",
        *filler(100, "c"),
    ]
    text = failing_step(*body)
    head, first, second = 1, 4 + 60, 4 + 60 + 1 + 60
    full = preprocess(text, budget_tokens=10**6).estimated_tokens

    seen_partial = False
    for budget in range(0, full + 50, 25):
        lines = kept_with(text, budget)
        # Each window only appears once everything ranked above it is in.
        if second in lines:
            assert first in lines
        if first in lines:
            assert head in lines
        seen_partial |= head in lines and first in lines and second not in lines
    assert seen_partial
    assert {head, first, second} <= set(kept_with(text, full))


def test_long_lines_are_truncated() -> None:
    blob = "x" * (MAX_LINE_CHARS + 1234)

    rendered = Line(7, blob).render()

    assert rendered == f"L7| {'x' * MAX_LINE_CHARS}... [1234 chars truncated]"


def test_cascades_cost_one_line_against_the_budget() -> None:
    body = ["lib/a.dart:1:1: Error: Type 'R' not found."] * 2000

    out = preprocess(failing_step(*body), budget_tokens=200)

    assert out.text.count("Type 'R' not found.") == 1
    assert "L4| lib/a.dart:1:1: Error: Type 'R' not found. (repeated 2000 times)" in out.text
    assert out.estimated_tokens <= 200
