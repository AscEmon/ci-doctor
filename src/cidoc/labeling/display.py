"""Turn a stored job log into a compact view for a human labeler.

Cleaning keeps exactly one output line per input line, so the 1-based line numbers
shown here are the real line numbers in the stored log. Labelers cite them as evidence.
"""

import gzip
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.text import Text

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
    return any(marker in line for marker in ERROR_MARKERS)


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


def _render_lines(console: Console, lines: list[NumberedLine], view: LogView) -> None:
    # Same gutter width everywhere so snippets and tail line up.
    width = len(str(view.total_lines))
    for number, line in lines:
        # Text, not markup: log content must never be interpreted as rich tags.
        row = Text(f"{number:>{width}} │ ", style="dim")
        row.append(line, style="bold red" if number in view.marker_lines else "")
        console.print(row, overflow="fold", highlight=False)


def render_view(console: Console, view: LogView) -> None:
    if view.snippets:
        console.rule("error markers before the tail", style="red")
        for i, snippet in enumerate(view.snippets):
            if i:
                console.print(Text("   ⋯", style="dim"))
            _render_lines(console, snippet, view)
        if view.omitted_snippets:
            console.print(Text(f"   ({view.omitted_snippets} more snippets omitted)", style="dim"))
    console.rule(f"last {len(view.tail)} of {view.total_lines} lines")
    _render_lines(console, view.tail, view)


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


def build_focus_view(text: str, failed_steps: Iterable[str] = ()) -> FocusView:
    """Only the failed step(s), condensed if long, plus the lines leading up to each error.

    Line numbers stay those of the stored log, so they can be cited as evidence.
    """
    lines = clean_log(text)
    end = next((i for i, line in enumerate(lines) if line.startswith(CLEANUP_LINE)), len(lines))
    hidden = hidden_action_lines(lines, end)
    errors = [i for i in range(end) if lines[i].startswith(ERROR_PREFIX) and i not in hidden]
    failed_steps = list(failed_steps)

    shown: set[int] = set()
    titles: list[str] = []
    for step in split_steps(lines, end):
        has_error = any(step.start <= e < step.stop for e in errors)
        if not has_error and not _matches_failed_step(step, failed_steps):
            continue
        titles.append(step.title)
        visible = [i for i in range(step.start, step.stop) if i not in hidden]
        if len(visible) <= FULL_STEP_MAX_LINES:
            shown.update(visible)
            continue
        shown.update(visible[:STEP_HEAD_LINES])
        step_lines = lines[step.start : step.stop]
        windows = marker_windows(step_lines, len(step_lines), CONTEXT_LINES)[:MAX_SNIPPETS]
        for start, stop in windows:
            shown.update(i for i in range(step.start + start, step.start + stop) if i not in hidden)
    for e in errors:
        shown.update(i for i in range(max(0, e - LINES_BEFORE_ERROR), e + 1) if i not in hidden)

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


def render_focus_view(console: Console, view: FocusView) -> None:
    title = "; ".join(view.failed_titles) or "lines before errors"
    console.rule(f"failed step focus: {title}", style="red")
    width = len(str(view.total_lines))
    for n, block in enumerate(view.blocks):
        if n:
            gap = block[0][0] - view.blocks[n - 1][-1][0] - 1
            console.print(Text(f"   ⋯ {gap} lines", style="dim"))
        for number, line in block:
            row = Text(f"{number:>{width}} │ ", style="dim")
            row.append(line, style="bold red" if number in view.marker_lines else "")
            console.print(row, overflow="fold", highlight=False)
    note = f"{view.total_lines} lines total"
    if view.hidden_after_cleanup:
        note += f"; {view.hidden_after_cleanup} after 'Post job cleanup.' hidden"
    console.print(Text(f"   ({note}; press f for the full view)", style="dim"))
