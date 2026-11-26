"""Deterministic views of a stored job log: cleaning, error markers, failed-step focus.

Shared by the labeling UI, the miner and the analyzer preprocessor; no rendering here.

Cleaning keeps exactly one output line per input line, so the 1-based line numbers
shown here are the real line numbers in the stored log. Labelers cite them as evidence.
"""

import gzip
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TAIL_LINES = 120
CONTEXT_LINES = 3
# The first errors are usually the root cause; analyzer runs can emit thousands more.
MAX_SNIPPETS = 20

ERROR_MARKERS = (
    "##[error]",
    "FAILURE:",
    "BUILD FAILED",
    "error:",
    "Error:",
    "Exception",
    "FAILED",
    "Could not resolve",
    "Could not find",
    "version solving failed",
    # git push/fetch and HTTP uploads.
    "remote: error",
    "remote: fatal",
    "Received status code",
    # Xcode, CocoaPods and fastlane.
    "❌",
    "** ARCHIVE FAILED **",
    "** BUILD FAILED **",
    "[!]",
    "fastlane finished with errors",
    "💥",
    "Error (Xcode):",
    "ERROR ITMS-",
)
# Xcode diagnostics that look alarming but never fail a build. A line with one of these
# is not a marker even if it also contains an error marker.
NON_ERROR_MARKERS = (
    "Target Integrity (Xcode)",
    "Warning (Xcode)",
)

# CSI sequences (colors, cursor moves) and OSC sequences (titles, hyperlinks).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
# GitHub prefixes every log line with an RFC 3339 timestamp.
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?")

NumberedLine = tuple[int, str]


def split_lines(text: str) -> list[str]:
    """Split on LF only, so numbering matches `zcat log.gz | cat -n` (unlike str.splitlines)."""
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line.removesuffix("\r") for line in lines]


def clean_line(line: str) -> str:
    # A bare CR redraws the line (progress bars); keep what a terminal would end up showing.
    line = line.rsplit("\r", 1)[-1]
    return _TIMESTAMP_RE.sub("", _ANSI_RE.sub("", line.lstrip("\ufeff")))


def clean_log(text: str) -> list[str]:
    return [clean_line(line) for line in split_lines(text)]


def is_marker(line: str) -> bool:
    if any(marker in line for marker in NON_ERROR_MARKERS):
        return False
    return any(marker in line for marker in ERROR_MARKERS)


# fastlane's "update available" notice: a #### banner around "fastlane X is available",
# optionally followed by the release notes, ending "To see all new releases ..." /
# "Please update using ...". It is printed after a failure, right where the error is
# expected, and is never a root cause.
_FASTLANE_AVAILABLE = re.compile(r"\bfastlane [\d.]+ is available\b")
_BANNER = re.compile(r"^#{10,}\s*$")
_CHANGELOG_END = "To see all new releases"
_UPDATE_HINT = "Please update using"
# How far the release notes may run after the banner (real ones: ~150 lines).
_CHANGELOG_MAX_LINES = 400
LINES_BEFORE_NOISE = 40


def fastlane_noise_blocks(lines: list[str], end: int) -> list[tuple[int, int]]:
    """Half-open [start, stop) index ranges of fastlane update notices in lines[:end]."""
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < end:
        line = lines[i]
        if _FASTLANE_AVAILABLE.search(line):
            start = i - 1 if i > 0 and _BANNER.match(lines[i - 1]) else i
            stop = i + 1
            for j in range(i + 1, min(end, i + 6)):  # closing banner
                if _BANNER.match(lines[j]):
                    stop = j + 1
                    break
        elif _CHANGELOG_END in line:
            start, stop = i, i + 1
        else:
            i += 1
            continue
        # Extend over the release notes to "To see all new releases" + "Please update".
        for j in range(stop, min(end, stop + _CHANGELOG_MAX_LINES)):
            if _FASTLANE_AVAILABLE.search(lines[j]):
                break  # the next notice starts; this one had no release notes
            if _CHANGELOG_END in lines[j]:
                stop = j + 1
                break
        for j in range(stop, min(end, stop + 3)):
            if _UPDATE_HINT in lines[j]:
                stop = j + 1
                break
        blocks.append((start, stop))
        i = stop
    return blocks


def marker_windows(lines: list[str], end: int, context: int) -> list[tuple[int, int]]:
    """Half-open [start, stop) index ranges around marker lines before `end`, merged."""
    windows: list[tuple[int, int]] = []
    for i in range(end):
        if not is_marker(lines[i]):
            continue
        start, stop = max(0, i - context), min(end, i + context + 1)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], stop)
        else:
            windows.append((start, stop))
    return windows


@dataclass(frozen=True)
class LogView:
    total_lines: int
    snippets: list[list[NumberedLine]]
    omitted_snippets: int
    tail: list[NumberedLine]
    marker_lines: frozenset[int]


def build_view(
    text: str,
    tail: int = TAIL_LINES,
    context: int = CONTEXT_LINES,
    max_snippets: int = MAX_SNIPPETS,
) -> LogView:
    lines = clean_log(text)
    tail_start = max(0, len(lines) - tail)
    windows = marker_windows(lines, tail_start, context)

    def numbered(start: int, stop: int) -> list[NumberedLine]:
        return [(i + 1, lines[i]) for i in range(start, stop)]

    return LogView(
        total_lines=len(lines),
        snippets=[numbered(start, stop) for start, stop in windows[:max_snippets]],
        omitted_snippets=max(0, len(windows) - max_snippets),
        tail=numbered(tail_start, len(lines)),
        marker_lines=frozenset(i + 1 for i, line in enumerate(lines) if is_marker(line)),
    )


def read_log(path: Path) -> str:
    # newline="" keeps bare CRs intact so line numbers are not shifted.
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


# --- Failed-step focus view ---------------------------------------------------------
#
# GitHub job logs mark each step's start with "##[group]Run <command>"; the matching
# "##[endgroup]" only closes the collapsed header (command, env), and the step's output,
# including its "##[error]" lines, follows it. So a step runs from one "Run" header to
# the next. Composite actions add "##[start-action ...;id=X]" ... "##[end-action id=X;
# outcome=...]" pairs around their sub-steps, which tell us which parts succeeded.

STEP_HEADER = "##[group]Run "
START_ACTION = "##[start-action "
END_ACTION = "##[end-action "
ERROR_PREFIX = "##[error]"
CLEANUP_LINE = "Post job cleanup."
# Composite sub-steps with these outcomes carry nothing to diagnose.
HIDDEN_OUTCOMES = frozenset({"success", "skipped"})
LINES_BEFORE_ERROR = 15
# A failed step up to this long is shown whole; longer ones are condensed.
FULL_STEP_MAX_LINES = 150
STEP_HEAD_LINES = 10


def _action_attrs(line: str, prefix: str) -> dict[str, str]:
    """Parse 'display=X;id=Y;outcome=Z' out of a start-/end-action line."""
    body = line[len(prefix) :].removesuffix("]")
    attrs: dict[str, str] = {}
    for part in body.split(";"):
        name, sep, value = part.partition("=")
        if sep:
            attrs[name.strip()] = value
    return attrs


@dataclass(frozen=True)
class Step:
    start: int
    """0-based index of the step's first line."""
    stop: int
    """0-based index one past its last line."""
    title: str


def hidden_action_lines(lines: list[str], end: int) -> set[int]:
    """Indices inside start-/end-action pairs whose outcome is success or skipped."""
    hidden: set[int] = set()
    open_actions: list[tuple[str, int]] = []
    for i in range(end):
        line = lines[i]
        if line.startswith(START_ACTION):
            open_actions.append((_action_attrs(line, START_ACTION).get("id", ""), i))
        elif line.startswith(END_ACTION):
            attrs = _action_attrs(line, END_ACTION)
            for depth in range(len(open_actions) - 1, -1, -1):
                if open_actions[depth][0] == attrs.get("id"):
                    start = open_actions[depth][1]
                    del open_actions[depth:]
                    if attrs.get("outcome") in HIDDEN_OUTCOMES:
                        hidden.update(range(start, i + 1))
                    break
    return hidden


def split_steps(lines: list[str], end: int) -> list[Step]:
    """Split lines[:end] at step headers; a start-action directly above a header joins it."""
    starts: list[tuple[int, str]] = [(0, "Set up job")]
    for i in range(end):
        line = lines[i]
        if line.startswith(START_ACTION):
            title = _action_attrs(line, START_ACTION).get("display") or line
            starts.append((i, title))
        elif line.startswith(STEP_HEADER):
            if starts and starts[-1][0] == i - 1 and lines[i - 1].startswith(START_ACTION):
                continue  # header of the sub-step its start-action already opened
            starts.append((i, line.removeprefix("##[group]")))
    steps = []
    for (start, title), (stop, _) in zip(starts, [*starts[1:], (end, "")], strict=True):
        if stop > start:
            steps.append(Step(start, stop, title))
    return steps


def _matches_failed_step(step: Step, failed_steps: Iterable[str]) -> bool:
    title = step.title.strip().lower()
    return any(name.strip().lower() == title for name in failed_steps if name.strip())


@dataclass(frozen=True)
class FocusView:
    total_lines: int
    blocks: list[list[NumberedLine]]
    """Contiguous runs of shown lines, in log order."""
    failed_titles: list[str]
    hidden_after_cleanup: int
    marker_lines: frozenset[int]

    @property
    def empty(self) -> bool:
        return not self.blocks


@dataclass(frozen=True)
class LogSteps:
    lines: list[str]
    end: int
    """Index of 'Post job cleanup.' (or len(lines)): nothing from here on is considered."""
    hidden: set[int]
    """Successful/skipped action blocks and fastlane update notices."""
    errors: list[int]
    failed: list[Step]
    noise_starts: list[int]
    """Where each fastlane update notice begins; the real error is usually just above."""


def locate_failed_steps(text: str, failed_steps: Iterable[str]) -> LogSteps:
    """Steps that contain an ##[error] or match a failed step name from the API."""
    return locate_failed_steps_in_lines(clean_log(text), failed_steps)


def locate_failed_steps_in_lines(lines: list[str], failed_steps: Iterable[str]) -> LogSteps:
    """locate_failed_steps for lines already cleaned with clean_log."""
    end = next((i for i, line in enumerate(lines) if line.startswith(CLEANUP_LINE)), len(lines))
    hidden = hidden_action_lines(lines, end)
    noise = fastlane_noise_blocks(lines, end)
    for start, stop in noise:
        hidden.update(range(start, stop))
    errors = [i for i in range(end) if lines[i].startswith(ERROR_PREFIX) and i not in hidden]
    names = list(failed_steps)
    failed = [
        step
        for step in split_steps(lines, end)
        if any(step.start <= e < step.stop for e in errors) or _matches_failed_step(step, names)
    ]
    return LogSteps(lines, end, hidden, errors, failed, [start for start, _ in noise])


def visible_line_numbers(text: str) -> list[int]:
    """1-based numbers of lines before 'Post job cleanup.' that are not known noise."""
    located = locate_failed_steps(text, ())
    return [i + 1 for i in range(located.end) if i not in located.hidden]


def failed_step_lines(text: str, failed_steps: Iterable[str] = ()) -> list[str]:
    """All visible lines of the failed step(s), uncondensed, in log order.

    For rules that need to read what the failing step ran (its header, env and output).
    Empty when no failed step can be located.
    """
    located = locate_failed_steps(text, failed_steps)
    return [
        located.lines[i]
        for step in located.failed
        for i in range(step.start, step.stop)
        if i not in located.hidden
    ]


WindowKind = Literal["error_lead", "step_head", "noise_lead", "marker", "step"]


@dataclass(frozen=True)
class FocusWindow:
    kind: WindowKind
    indices: tuple[int, ...]
    """0-based, visible (not hidden), ascending."""


def focus_windows(
    located: LogSteps,
    context: int = CONTEXT_LINES,
    max_snippets: int | None = MAX_SNIPPETS,
    lines_before_error: int = LINES_BEFORE_ERROR,
    hidden: set[int] | None = None,
) -> list[FocusWindow]:
    """What the focus view shows, as typed windows (they may overlap).

    - "step": a whole failed step, if it has at most FULL_STEP_MAX_LINES visible lines
    - "step_head": the first lines of a longer failed step
    - "marker": error-marker lines of a failed step with `context` lines around them
      (the first `max_snippets` per step)
    - "error_lead": each ##[error] line and the `lines_before_error` lines above it
    - "noise_lead": the lines above a fastlane update notice in a failed step

    `hidden` overrides `located.hidden` (callers may hide more noise).
    """
    lines = located.lines
    if hidden is None:
        hidden = located.hidden

    def window(kind: WindowKind, start: int, stop: int) -> FocusWindow:
        return FocusWindow(kind, tuple(i for i in range(max(0, start), stop) if i not in hidden))

    windows: list[FocusWindow] = []
    for e in located.errors:
        windows.append(window("error_lead", e - lines_before_error, e + 1))
    for step in located.failed:
        visible = [i for i in range(step.start, step.stop) if i not in hidden]
        if len(visible) <= FULL_STEP_MAX_LINES:
            windows.append(FocusWindow("step", tuple(visible)))
        else:
            windows.append(FocusWindow("step_head", tuple(visible[:STEP_HEAD_LINES])))
        step_lines = lines[step.start : step.stop]
        for start, stop in marker_windows(step_lines, len(step_lines), context)[:max_snippets]:
            windows.append(window("marker", step.start + start, step.start + stop))
    # fastlane prints its update notice right after the failure: show what came before it.
    for n in located.noise_starts:
        if any(step.start <= n < step.stop for step in located.failed):
            windows.append(window("noise_lead", n - LINES_BEFORE_NOISE, n))
    return [w for w in windows if w.indices]


def focus_indices(
    located: LogSteps,
    context: int = CONTEXT_LINES,
    max_snippets: int | None = MAX_SNIPPETS,
    hidden: set[int] | None = None,
) -> set[int]:
    """0-based indices the focus view shows: the union of its windows."""
    windows = focus_windows(located, context, max_snippets, hidden=hidden)
    return {i for w in windows for i in w.indices}


def build_focus_view(text: str, failed_steps: Iterable[str] = ()) -> FocusView:
    """Only the failed step(s), condensed if long, plus the lines leading up to each error.

    Line numbers stay those of the stored log, so they can be cited as evidence.
    """
    located = locate_failed_steps(text, failed_steps)
    lines, end = located.lines, located.end
    shown = focus_indices(located)
    titles = [step.title for step in located.failed]

    blocks: list[list[NumberedLine]] = []
    for i in sorted(shown):
        # Line numbers are 1-based, so the previous index i-1 shows as number i.
        if blocks and blocks[-1][-1][0] == i:
            blocks[-1].append((i + 1, lines[i]))
        else:
            blocks.append([(i + 1, lines[i])])
    return FocusView(
        total_lines=len(lines),
        blocks=blocks,
        failed_titles=titles,
        hidden_after_cleanup=len(lines) - end,
        marker_lines=frozenset(i + 1 for i, line in enumerate(lines) if is_marker(line)),
    )
