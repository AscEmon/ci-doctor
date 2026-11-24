import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console

from cidoc.labeling.hints import RULES, Hint, find_hint, hint_for_log
from cidoc.labeling.labels import Label, load_labels
from cidoc.labeling.session import Candidate, label_one
from cidoc.logview import build_focus_view, clean_log

NOW = datetime(2026, 10, 31, tzinfo=UTC)
TS = "2026-10-01T12:34:56.1234567Z "


def log(*lines: str) -> str:
    return "".join(f"{TS}{line}\n" for line in lines)


def hint_of(*lines: str) -> Hint | None:
    cleaned = clean_log(log(*lines))
    return find_hint(cleaned, range(1, len(cleaned) + 1))


@pytest.mark.parametrize(
    ("lines", "category", "matched"),
    [
        (
            [
                "Because app depends on intl ^0.19.0 which doesn't match any versions,"
                " version solving failed."
            ],
            "dependency_resolution",
            "version solving failed",
        ),
        (
            [
                "Run flutter analyze",
                "Analyzing app...",
                "   info • Unused import • lib/main.dart:3:8 • unused_import",
            ],
            "analyzer_lint",
            "info •",
        ),
        (
            [
                "$ dart analyze --fatal-infos",
                "warning • The value of the local variable isn't used • lib/a.dart:9:7",
            ],
            "analyzer_lint",
            "warning •",
        ),
        (
            ["Run flutter analyze", "  error • Undefined name 'foo' • lib/b.dart:1:1"],
            "analyzer_lint",
            "error •",
        ),
        (
            ["fatal: could not read Username for 'https://github.com': terminal prompts disabled"],
            "signing_secrets",
            "could not read Username",
        ),
        (
            [
                "Execution failed for task ':app:validateSigningRelease'.",
                "> Keystore file '/home/runner/upload.jks' not found for signing config 'release'.",
            ],
            "signing_secrets",
            "Keystore",
        ),
        (
            [
                "error: No profiles for 'com.example.app' were found: Xcode couldn't find any iOS"
                " App Development provisioning profiles matching 'com.example.app'."
            ],
            "signing_secrets",
            "provisioning profile",
        ),
        (
            ['Code Signing Error: No signing certificate "iOS Development" found'],
            "signing_secrets",
            "Code Signing",
        ),
        (
            ["Android Gradle plugin requires Java 17 to run. You are currently using Java 11."],
            "toolchain_version",
            "requires Java",
        ),
        (
            [
                "BUG! exception in phase 'semantic analysis':"
                " Unsupported class file major version 65"
            ],
            "toolchain_version",
            "Unsupported class file major version",
        ),
        (
            ["Minimum supported Gradle version is 8.7. Current version is 8.3."],
            "toolchain_version",
            "Minimum supported Gradle",
        ),
        (
            ["00:42 +118 -1: Some tests failed."],
            "test_failure",
            "Some tests failed",
        ),
        (
            ["00:05 +3 -1: test/widget_test.dart: counter increments [E]"],
            "test_failure",
            "[E]",
        ),
        (
            ["❌ test/login_test.dart: login shows error (failed)"],
            "test_failure",
            "test/login_test.dart: login shows error (failed",
        ),
        (
            ["Error running pod install"],
            "ios_build",
            "pod install",
        ),
        (
            ['[!] CocoaPods could not find compatible versions for pod "Firebase/CoreOnly"'],
            "ios_build",
            "CocoaPods",
        ),
        (
            ["Failed to build iOS app", "Error (Xcode): xcodebuild exited with code 65"],
            "ios_build",
            "xcodebuild",
        ),
        (
            ["Gradle download timed out after 120000 ms"],
            "infra_flaky",
            "timed out",
        ),
        (
            ["npm ERR! code ETIMEDOUT"],
            "infra_flaky",
            "ETIMEDOUT",
        ),
        (
            ["Received HTTP 503 Service Unavailable from storage.googleapis.com"],
            "infra_flaky",
            "503",
        ),
        (
            ["Error: read ECONNRESET"],
            "infra_flaky",
            "ECONNRESET",
        ),
    ],
)
def test_each_pattern(lines: list[str], category: str, matched: str) -> None:
    hint = hint_of(*lines)

    assert hint is not None
    assert (hint.category, hint.matched) == (category, matched)


def test_every_rule_has_a_fixture() -> None:
    assert [rule.category for rule in RULES] == [
        "dependency_resolution",
        "analyzer_lint",
        "signing_secrets",
        "toolchain_version",
        "gradle_agp_config",
        "test_failure",
        "ios_build",
        "infra_flaky",
    ]


REAL_KOTLIN_LINE = "> Could not find org.jetbrains.kotlin:kotlin-stdlib:2.0.21."


@pytest.mark.parametrize(
    ("line", "matched"),
    [
        (REAL_KOTLIN_LINE, "Could not find org.jetbrains.kotlin:kotlin-stdlib:2.0.21."),
        (
            "> Could not resolve all artifacts for configuration ':classpath'.",
            "Could not resolve all artifacts",
        ),
        (
            "> Could not resolve all dependencies for configuration ':app:debugRuntimeClasspath'.",
            "Could not resolve all dependencies",
        ),
        (
            "Execution failed for task ':app:checkDebugAarMetadata'."
            " > Could not resolve all files for configuration",
            "Could not resolve all files",
        ),
        (
            "   > Could not resolve com.google.firebase:firebase-bom:33.1.0.",
            "Could not resolve com.google.firebase:firebase-bom:33.1.0.",
        ),
        (
            "      > Could not GET 'https://dl.google.com/dl/android/maven2/com/android/x.pom'.",
            "Could not GET 'https://",
        ),
        (
            "Could not download kotlin-compiler-embeddable-1.9.22.jar"
            " (org.jetbrains.kotlin:kotlin-compiler-embeddable:1.9.22)",
            "Could not download",
        ),
    ],
)
def test_gradle_maven_resolution_is_dependency_resolution(line: str, matched: str) -> None:
    assert hint_of(line) == Hint("dependency_resolution", matched, 1)


@pytest.mark.parametrize(
    ("line", "matched"),
    [
        (
            "Namespace not specified. Specify a namespace in the module's build file.",
            "Namespace not specified",
        ),
        (
            "Plugin [id: 'dev.flutter.flutter-plugin-loader'] was not found in any of the"
            " following sources:",
            "Plugin [id: 'dev.flutter.flutter-plugin-loader'] was not found",
        ),
        (
            "Plugin [id: 'com.android.application', version: '8.1.0', apply: false]"
            " was not found in any of the following sources:",
            "Plugin [id: 'com.android.application', version: '8.1.0', apply: false] was ",
        ),
        (
            "> Could not find method implementation() for arguments [project ':core']",
            "Could not find method implementation() for arguments",
        ),
        (
            "* Where: Build file '/home/runner/work/app/android/app/build.gradle' line: 27",
            "Build file '/home/runner/work/app/android/app/build.gradle' l",
        ),
    ],
)
def test_gradle_config_errors(line: str, matched: str) -> None:
    hint = hint_of(line)

    assert hint is not None
    assert (hint.category, hint.matched) == ("gradle_agp_config", matched[:60])


def test_resolution_failure_inside_build_file_error_is_dependency_resolution() -> None:
    # Gradle's usual layout: the "Where" points at a build file, the cause is resolution.
    hint = hint_of(
        "FAILURE: Build failed with an exception.",
        "* Where:",
        "Build file '/home/runner/work/app/android/build.gradle' line: 12",
        "* What went wrong:",
        "A problem occurred configuring root project 'android'.",
        "> Could not resolve all artifacts for configuration ':classpath'.",
        f"   {REAL_KOTLIN_LINE}",
    )

    assert hint == Hint("dependency_resolution", "Could not resolve all artifacts", 6)


def test_signing_and_toolchain_win_over_build_file_location() -> None:
    signing = hint_of(
        "Build file '/app/android/app/build.gradle' line: 40",
        "> Keystore file '/app/upload.jks' not found for signing config 'release'.",
    )
    toolchain = hint_of(
        "Build file '/app/android/build.gradle' line: 3",
        "> Android Gradle plugin requires Java 17 to run.",
    )

    assert signing is not None and signing.category == "signing_secrets"
    assert toolchain is not None and toolchain.category == "toolchain_version"


def test_could_not_find_without_coordinates_is_not_dependency_resolution() -> None:
    assert hint_of("Could not find a file named 'pubspec.yaml'") is None


@pytest.mark.parametrize(
    "line",
    [
        "BUILD FAILED in 2m 13s",
        "   info • Unused import • lib/main.dart:3:8",  # no flutter/dart analyze before it
        "Fetching latest packages failed",  # 'test' inside 'latest' is not a test failure
        "sha256: 9f86d081884c7d6585030a",  # 503 inside a hash
        "All tests passed!",
    ],
)
def test_no_hint(line: str) -> None:
    assert hint_of(line) is None


def test_analyzer_issue_before_the_analyze_command_does_not_count() -> None:
    assert hint_of("info • stale message", "Run flutter analyze", "No issues found!") is None


def test_first_rule_in_order_wins_and_line_number_is_reported() -> None:
    hint = hint_of(
        "Running pod install...",
        "Because app depends on foo ^2.0.0, version solving failed.",
    )

    assert hint == Hint("dependency_resolution", "version solving failed", 2)


# Setup step mentions keystore and xcodebuild; the failed step is a test failure.
STEPPED_LOG = log(
    "##[group]Run ./scripts/decode_keystore.sh",
    "##[endgroup]",
    "Decoded keystore to android/app/upload.jks",
    "xcodebuild -version: Xcode 16.0",
    *[f"setup line {i}" for i in range(20)],
    "##[group]Run flutter test",
    "##[endgroup]",
    "00:42 +118 -1: Some tests failed.",
    "##[error]Process completed with exit code 1.",
    "Post job cleanup.",
    "Request timed out",
)


def test_hint_searches_only_the_failed_step() -> None:
    hint = hint_for_log(STEPPED_LOG, build_focus_view(STEPPED_LOG))

    assert hint is not None
    assert hint.category == "test_failure"


def test_without_a_focus_view_whole_log_before_cleanup_is_searched() -> None:
    text = log("Error running pod install", "Post job cleanup.", "ETIMEDOUT")

    focus = build_focus_view(text)
    assert focus.empty
    hint = hint_for_log(text, focus)

    assert hint is not None
    assert hint.category == "ios_build"
    assert hint_for_log(log("Post job cleanup.", "ETIMEDOUT"), build_focus_view("")) is None


# --- shown, never auto-filled, recorded in the label -------------------------------


def candidate() -> Candidate:
    return Candidate(
        job_id=1,
        repo="o/a",
        run_id=7,
        workflow_name="CI",
        workflow_path=".github/workflows/ci.yml",
        event="push",
        head_branch="main",
        failed_steps=["Build"],
        runner_os="Linux",
        log_path="x.log.gz",
    )


def run_label(keys: str, hint: Hint | None) -> tuple[Label | None, str]:
    console = Console(record=True, width=120)
    pressed = iter(keys)
    answers = iter(["pubspec pins intl too low", "3"])
    label = label_one(
        console, lambda: next(pressed), lambda _: next(answers), candidate(), 10, NOW, hint=hint
    )
    return label, console.export_text()


HINT = Hint("dependency_resolution", "version solving failed", 3)


def test_hint_is_shown_above_the_prompt() -> None:
    _, out = run_label("12", HINT)

    hint_line = "hint: dependency_resolution (matched: 'version solving failed' on L3)"
    assert hint_line in out
    assert out.index(hint_line) < out.index("category>")


@pytest.mark.parametrize(
    ("keys", "category", "accepted"),
    [("12", "dependency_resolution", True), ("62", "test_failure", False)],
)
def test_label_records_hint_and_whether_it_was_accepted(
    keys: str, category: str, accepted: bool
) -> None:
    label, _ = run_label(keys, HINT)

    assert label is not None
    assert (label.category, label.hint, label.hint_accepted) == (category, HINT.category, accepted)


def test_hint_never_auto_fills() -> None:
    # Enter, space and other non-category keys are ignored; only a typed category counts.
    label, _ = run_label("\r \n62", HINT)

    assert label is not None
    assert label.category == "test_failure"


def test_no_hint_records_none() -> None:
    label, out = run_label("12", None)

    assert label is not None
    assert (label.hint, label.hint_accepted) == (None, None)
    assert "hint:" not in out


def test_old_labels_without_hint_fields_still_load(tmp_path: Path) -> None:
    old = {
        "job_id": 1,
        "repo": "o/a",
        "run_id": 7,
        "category": "other",
        "root_cause": "x",
        "evidence_lines": [1],
        "confidence": 1,
        "log_path": "x.log.gz",
        "labeled_at": NOW.isoformat(),
    }
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")

    [label] = load_labels(path)

    assert (label.hint, label.hint_accepted) == (None, None)
    assert asdict(label)["hint"] is None
