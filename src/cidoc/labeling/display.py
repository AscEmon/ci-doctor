"""Turn a stored job log into a compact view for a human labeler.

Cleaning keeps exactly one output line per input line, so the 1-based line numbers
shown here are the real line numbers in the stored log. Labelers cite them as evidence.
"""

import gzip
import re
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
