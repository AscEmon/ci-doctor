import gzip
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console

from cidoc.labeling.display import render_focus_view
from cidoc.labeling.hints import hint_for_log
from cidoc.labeling.session import Candidate, ViewToggle, label_one
from cidoc.logview import (
    FULL_STEP_MAX_LINES,
    LINES_BEFORE_ERROR,
    build_focus_view,
    build_view,
    clean_log,
    fastlane_noise_blocks,
    hidden_action_lines,
    is_marker,
    read_log,
    split_steps,
    visible_line_numbers,
)

TS = "2026-10-01T12:34:56.1234567Z "
NOW = datetime(2026, 10, 31, tzinfo=UTC)


def log(*lines: str) -> str:
    return "".join(f"{TS}{line}\n" for line in lines)


# Shape taken from a real Actions log: set-up groups, a checkout step with inner groups,
# a composite action whose sub-steps succeed or are skipped, then the failing step whose
# output (and ##[error]) follows its header group, then post-job cleanup.
LOG = log(
    "##[group]Runner Image",  # 1
    "Image: ubuntu-24.04",  # 2
    "##[endgroup]",  # 3
    "##[group]Run actions/checkout@v4",  # 4
    "with: repository",  # 5
    "##[endgroup]",  # 6
    "##[group]Fetching the repository",  # 7
    "git fetch origin",  # 8
    "##[endgroup]",  # 9
    "##[start-action display=Setup Flutter;id=__self.__flutter]",  # 10
    "##[group]Run subosito/flutter-action@v2",  # 11
    "##[endgroup]",  # 12
    "##[start-action display=Windows only;id=__self.__flutter.__run]",  # 13
    "##[end-action id=__self.__flutter.__run;outcome=skipped;conclusion=skipped;duration_ms=0]",
    "##[start-action display=Cache Flutter;id=__self.__flutter.cache]",  # 15
    "##[group]Run actions/cache@v4",  # 16
    "##[endgroup]",  # 17
    "Cache restored from key: flutter-linux-stable",  # 18
    "##[end-action id=__self.__flutter.cache;outcome=success;conclusion=success;duration_ms=9]",
    "##[end-action id=__self.__flutter;outcome=success;conclusion=success;duration_ms=99]",  # 20
    "##[group]Run flutter build apk --release",  # 21
    "flutter build apk --release",  # 22
    "shell: /usr/bin/bash -e {0}",  # 23
    "##[endgroup]",  # 24
    "Running Gradle task 'assembleRelease'...",  # 25
    "e: file:///app/MainActivity.kt:12:5 Unresolved reference: foo",  # 26
    "FAILURE: Build failed with an exception.",  # 27
    "BUILD FAILED in 41s",  # 28
    "##[error]Process completed with exit code 1.",  # 29
    "Post job cleanup.",  # 30
    "[command]/usr/bin/git version",  # 31
    "##[error]this error after cleanup is not shown",  # 32
)


def shown_numbers(text: str, failed_steps: list[str] | None = None) -> list[int]:
    view = build_focus_view(text, failed_steps or [])
    return [number for block in view.blocks for number, _ in block]


def test_focus_shows_only_the_failed_step_with_real_line_numbers() -> None:
    view = build_focus_view(LOG)

    assert view.failed_titles == ["Run flutter build apk --release"]
    assert [number for block in view.blocks for number, _ in block] == list(range(21, 30))
    lines = dict(line for block in view.blocks for line in block)
    assert lines[21] == "##[group]Run flutter build apk --release"
    assert lines[29] == "##[error]Process completed with exit code 1."


def test_successful_and_skipped_action_blocks_are_hidden() -> None:
    lines = clean_log(LOG)
    hidden = hidden_action_lines(lines, len(lines))

    # 1-based lines 10-20: the whole composite action, including its nested sub-steps.
    assert hidden == set(range(9, 20))
    # The 15 lines before ##[error] would reach back to line 14, but lines 14-20 sit in
    # a successful action block, so only the failed step's own lines are shown.
    assert min(shown_numbers(LOG)) == 21


def test_nothing_after_post_job_cleanup_is_shown() -> None:
    view = build_focus_view(LOG)

    assert max(number for block in view.blocks for number, _ in block) == 29
    assert view.hidden_after_cleanup == 3


def test_failed_composite_action_is_kept() -> None:
    text = log(
        "##[group]Run ./.github/actions/setup",
        "##[endgroup]",
        "##[start-action display=Install deps;id=a.deps]",
        "##[group]Run flutter pub get",
        "##[endgroup]",
        "Because app depends on foo ^2.0.0, version solving failed.",
        "##[error]Process completed with exit code 1.",
        "##[end-action id=a.deps;outcome=failure;conclusion=failure;duration_ms=5]",
    )

    view = build_focus_view(text)

    assert view.failed_titles == ["Install deps"]
    # Lines 1-2 are within the 15 lines before the error; 8 is the failed end-action.
    assert shown_numbers(text) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_step_named_in_db_is_shown_even_without_error_marker() -> None:
    text = log(
        "##[group]Run dart format --set-exit-if-changed .",
        "##[endgroup]",
        "Formatted 3 files",
        "##[group]Run echo done",
        "##[endgroup]",
        "done",
    )

    view = build_focus_view(text, ["Run dart format --set-exit-if-changed ."])

    assert view.failed_titles == ["Run dart format --set-exit-if-changed ."]
    assert shown_numbers(text, ["Run dart format --set-exit-if-changed ."]) == [1, 2, 3]


def test_split_steps_joins_start_action_with_its_header() -> None:
    lines = clean_log(LOG)
    titles = [s.title for s in split_steps(lines, 29)]

    assert titles == [
        "Set up job",
        "Run actions/checkout@v4",
        "Setup Flutter",
        "Windows only",
        "Cache Flutter",
        "Run flutter build apk --release",
    ]


def test_long_failed_step_is_condensed_around_errors() -> None:
    noise = [f"test {i} passed" for i in range(FULL_STEP_MAX_LINES * 2)]
    text = log(
        "##[group]Run flutter test",  # 1
        "##[endgroup]",  # 2
        *noise[:100],  # 3-102
        "Expected: 'Hello' Actual: 'Hola'",  # 103
        "FAILED: test/widget_test.dart",  # 104
        *noise[100:],  # 105-304
        "Some tests failed.",  # 305
        "##[error]Process completed with exit code 1.",  # 306
    )

    numbers = shown_numbers(text)

    assert numbers[:3] == [1, 2, 3]  # step head
    assert {103, 104} <= set(numbers)  # marker window in the middle
    assert numbers[-LINES_BEFORE_ERROR - 1 :] == list(range(291, 307))  # lead-up to error
    assert len(numbers) < 60


def test_no_failed_step_gives_empty_focus() -> None:
    assert build_focus_view(log("hello", "world")).empty


def test_render_focus_view_marks_gaps_and_hint() -> None:
    text = log(
        "##[group]Run flutter test",
        "##[endgroup]",
        *[f"ok {i}" for i in range(FULL_STEP_MAX_LINES + 20)],
        "##[error]boom",
    )
    console = Console(record=True, width=120)

    render_focus_view(console, build_focus_view(text))

    out = console.export_text()
    assert "failed step focus: Run flutter test" in out
    assert "⋯" in out
    assert "press f for the full view" in out


# --- f toggles between focus and full view ------------------------------------------


def candidate() -> Candidate:
    return Candidate(
        job_id=1,
        repo="o/a",
        run_id=1,
        workflow_name="CI",
        workflow_path=".github/workflows/ci.yml",
        event="push",
        head_branch="main",
        failed_steps=["Build"],
        runner_os="Linux",
        log_path="x.log.gz",
    )


def test_f_toggles_full_view_and_back_then_labels() -> None:
    console = Console(record=True, width=120)
    full = build_view(LOG)
    views = ViewToggle(console, full, build_focus_view(LOG))
    views.render()
    keys = iter(["f", "f", "s"])

    result = label_one(
        console, lambda: next(keys), input, candidate(), full.total_lines, NOW, views
    )

    out = console.export_text()
    assert result is None  # skipped after toggling
    assert out.count("failed step focus") == 2  # initial, then back after the second f
    assert f"last {full.total_lines} of {full.total_lines} lines" in out
    assert "f toggle failed-step focus / full view" in out


def test_without_a_failed_step_full_view_is_shown_and_f_is_off() -> None:
    console = Console(record=True, width=120)
    text = log("hello")
    views = ViewToggle(console, build_view(text), build_focus_view(text))
    views.render()
    keys = iter(["f", "s"])  # f is ignored (not a valid key), s skips

    label_one(console, lambda: next(keys), input, candidate(), 1, NOW, views)

    out = console.export_text()
    assert "no failed step or ##[error] found; showing the full view" in out
    assert "toggle failed-step focus" not in out


def test_focus_view_reads_gzipped_stored_log(tmp_path: Path) -> None:
    path = tmp_path / "x.log.gz"
    path.write_bytes(gzip.compress(LOG.encode()))

    assert build_focus_view(read_log(path)).failed_titles == ["Run flutter build apk --release"]


# --- Xcode / fastlane markers ---------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        '❌ error: Signing for "Runner" requires a development team.',
        "** ARCHIVE FAILED **",
        "** BUILD FAILED **",
        '[!] CocoaPods could not find compatible versions for pod "Firebase/Auth"',
        "[10:20:50]: fastlane finished with errors",
        "[!] Error building the application - see the log above 💥",
        "Error (Xcode): No profiles for 'com.example.app' were found",
        'ERROR ITMS-90189: "Redundant Binary Upload."',
    ],
)
def test_xcode_and_fastlane_error_markers(line: str) -> None:
    assert is_marker(line)


@pytest.mark.parametrize(
    "line",
    [
        'Target Integrity (Xcode): The file "Runner.xcconfig" couldn\'t be opened.',
        "Warning (Xcode): Run script build phase 'Thin Binary' will be run during every build",
        "Warning (Xcode): error: deprecated API used",  # error marker inside: still a warning
    ],
)
def test_xcode_warnings_are_not_error_markers(line: str) -> None:
    assert not is_marker(line)


def fastlane_notice(version: str = "2.242.0", notes: int = 0) -> list[str]:
    banner = "#" * 71
    lines = [
        banner,
        f"# fastlane {version} is available. You are on 2.228.0.",
        "# You should use the latest version.",
        "# Please update using `bundle update fastlane`.",
        banner,
    ]
    if notes:
        lines += [f"* change {i}" for i in range(notes)]
        lines += [
            "To see all new releases, open https://github.com/fastlane/fastlane/releases",
            "",
            "Please update using `bundle update fastlane`",
        ]
    return lines


def test_fastlane_notice_blocks_with_and_without_release_notes() -> None:
    lines = ["a", *fastlane_notice(notes=5), "b", *fastlane_notice(), "c"]

    blocks = fastlane_noise_blocks(lines, len(lines))

    # First: banner (5) + notes (5) + 3 closing lines = indices 1..13. Second: banner only.
    assert blocks == [(1, 14), (15, 20)]
    assert lines[14] == "b" and lines[20] == "c"


FASTLANE_LOG = log(
    "##[group]Run bundle exec fastlane ios release",  # 1
    "##[endgroup]",  # 2
    *[f"build output {i}" for i in range(60)],  # 3-62
    '[!] No signing certificate "iOS Distribution" found',  # 63
    *[f"context {i}" for i in range(30)],  # 64-93
    "[10:20:50]: fastlane finished with errors",  # 94
    *fastlane_notice(notes=150),  # 95-252
    "##[error]Process completed with exit code 1.",  # 253
)


def test_fastlane_notice_is_hidden_and_the_lines_above_it_shown() -> None:
    view = build_focus_view(FASTLANE_LOG)
    shown = {number for block in view.blocks for number, _ in block}
    lines = clean_log(FASTLANE_LOG)
    [(start, stop)] = fastlane_noise_blocks(lines, len(lines))

    assert (start + 1, stop) == (95, 252)
    assert not shown & set(range(95, 253))  # whole notice hidden, release notes included
    assert set(range(55, 95)) <= shown  # the 40 lines above it
    assert {63, 94, 253} <= shown  # the real error, fastlane's verdict, ##[error]


def test_hints_skip_xcode_warnings_and_fastlane_notices() -> None:
    warn = log(
        "Warning (Xcode): xcodebuild: WARNING: Using the first of multiple matching destinations"
    )
    notice = log(*fastlane_notice(notes=3), "plain line")

    assert hint_for_log(warn, build_focus_view(warn)) is None
    # The notice mentions nothing a rule matches today; check it is not even searched.
    assert visible_line_numbers(notice) == [len(clean_log(notice))]
