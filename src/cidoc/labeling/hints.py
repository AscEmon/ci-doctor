"""Deterministic, regex-based category hints for the labeler. No LLM involved.

A hint is shown above the category prompt and never fills in the answer. Labels record
the hint and whether the chosen category matched it, so anchoring on hints can be
measured later.

Rules are tried in order and the first match wins, so the broad patterns (test
failures, network errors) come last. They are searched only in the failed step's
lines when the focus view found one: setup steps mention keystores, Xcode and HTTP
codes all the time without that being why the job failed.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from cidoc.labeling.display import (
    NON_ERROR_MARKERS,
    FocusView,
    clean_log,
    visible_line_numbers,
)


@dataclass(frozen=True)
class HintRule:
    category: str
    pattern: re.Pattern[str]
    after: re.Pattern[str] | None = None
    """If set, only match on lines after the first line matching this."""


@dataclass(frozen=True)
class Hint:
    category: str
    matched: str
    """The text the rule matched."""
    line_no: int


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


RULES: tuple[HintRule, ...] = (
    # pub, then Gradle/Maven artifacts that could not be resolved or downloaded. First, so
    # it wins over gradle_agp_config: Gradle wraps resolution failures in config errors.
    HintRule(
        "dependency_resolution",
        _rx(
            r"version solving failed"
            r"|Could not resolve all (?:artifacts|dependencies|files)"
            r"|Could not find [\w.\-]+:[\w.\-]+:[\w.\-]+"
            r"|Could not resolve [\w.\-]+:[\w.\-]+:[\w.\-]+"
            r"|Could not GET 'https?://"
            r"|Could not download"
        ),
    ),
    HintRule(
        "analyzer_lint",
        _rx(r"\b(?:info|warning|error) •"),
        after=_rx(r"\b(?:flutter|dart) analyze\b"),
    ),
    HintRule(
        "signing_secrets",
        _rx(r"could not read Username|keystore|provisioning profile|code signing"),
    ),
    HintRule(
        "toolchain_version",
        _rx(r"requires Java|Unsupported class file major version|minimum supported Gradle"),
    ),
    # After signing and toolchain: "Build file '...' line: N" also wraps keystore and
    # Java/Gradle version errors, which the guideline files under those categories.
    # The plugin pattern allows ", version: '...'" etc. after the id, as Gradle prints it.
    HintRule(
        "gradle_agp_config",
        _rx(
            r"Namespace not specified"
            r"|Plugin \[id: '[^']+'[^\]]*\] was not found"
            r"|Could not find method \w+\(\) for arguments"
            r"|Build file '.*' line: \d+"
        ),
    ),
    # [E] is flutter test's failed-test marker. \btest keeps "latest ... failed" out.
    HintRule("test_failure", _rx(r"Some tests failed|\[E\]|\btest.*failed")),
    HintRule("ios_build", _rx(r"pod install|CocoaPods|xcodebuild")),
    # \b keeps 503 from matching inside longer numbers and hashes.
    HintRule("infra_flaky", _rx(r"timed out|ETIMEDOUT|\b503\b|ECONNRESET")),
)
MAX_MATCH_CHARS = 60


def find_hint(lines: list[str], search: Iterable[int]) -> Hint | None:
    """First rule (in RULES order) matching any of the 1-based line numbers in `search`."""
    numbers = sorted(set(search))
    for rule in RULES:
        first_after: int | None = None
        if rule.after is not None:
            first_after = next(
                (n for n, line in enumerate(lines, start=1) if rule.after.search(line)), None
            )
            if first_after is None:
                continue
        for n in numbers:
            if first_after is not None and n <= first_after:
                continue
            if any(marker in lines[n - 1] for marker in NON_ERROR_MARKERS):
                continue  # e.g. "Warning (Xcode): ... xcodebuild ..." is not an ios_build cause
            match = rule.pattern.search(lines[n - 1])
            if match:
                return Hint(rule.category, match.group(0)[:MAX_MATCH_CHARS], n)
    return None


def hint_for_log(text: str, focus: FocusView) -> Hint | None:
    """Hint from the focus view's lines, or, without one, every line before cleanup that
    is not known noise (successful action blocks, fastlane update notices)."""
    lines = clean_log(text)
    if focus.empty:
        search: Iterable[int] = visible_line_numbers(text)
    else:
        search = (number for block in focus.blocks for number, _ in block)
    return find_hint(lines, search)
