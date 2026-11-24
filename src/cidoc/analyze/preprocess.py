"""Turn a stored job log into compact, line-numbered text for the analyzer.

Deterministic, no LLM. Line selection reuses the labeling focus view (failed steps,
nothing after "Post job cleanup.", successful composite sub-steps hidden) with wider
context around error markers, then drops known noise and collapses repeated lines.
Every kept line is prefixed with its 1-based number in the stored log ("L1234| "), so
the model can cite original line numbers as evidence.

The result fits a token budget: candidate windows are ranked, and the lowest-ranked
ones are dropped first. The lines leading up to each ##[error] are always kept.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from cidoc.logview import TAIL_LINES, focus_windows, is_marker, locate_failed_steps

MARKER_CONTEXT = 15
LINES_BEFORE_ERROR = 80
DEFAULT_TOKEN_BUDGET = 6000
# Rough estimate for English and code; logs with long hashes or URLs run a bit higher.
CHARS_PER_TOKEN = 4
# Single lines can be huge (minified JS, base64 blobs); never let one eat the budget.
MAX_LINE_CHARS = 500

# Lower is more important. error_lead is always kept, whatever the budget.
PRIORITY = {
    "error_lead": 0,
    "step_head": 1,
    "noise_lead": 2,
    "tail": 2,
    "marker": 3,
    "step": 4,
}

# Flutter's first-run analytics notice, drawn in a ╔═╗ box.
_WELCOME = "Welcome to Flutter"
_BOX_TOP = "╔"
_BOX_BOTTOM = "╚"
_BOX_MAX_LINES = 40
# `pub get` lists the resolved packages above a one-line summary of outdated ones. Build
# scripts may nest it (indented), so these match after stripping leading whitespace.
_OUTDATED_SUMMARY = re.compile(r"^\d+ packages? ha(?:s|ve) newer versions? incompatible")
_OUTDATED_HINT = re.compile(r"^Try `(?:flutter|dart) pub outdated`")
# "  yaml 3.1.3 (3.1.4 available)", "> http 1.2.0 (was 1.1.0)", "+ meta 1.16.0", ...
_OUTDATED_ENTRY = re.compile(r"^(?:[!>*+~-] )?[a-z][a-z0-9_]* \d\S*(?: .*)?$")
_PUB_STATUS = re.compile(
    r"^(?:Got dependencies!|Changed \d+ dependenc(?:y|ies)!|No dependencies changed)"
)
_NODE_DEPRECATION = re.compile(r"^\(node:\d+\) .*DeprecationWarning:")
_NODE_TRACE_HINT = "(Use `node --trace-deprecation"
# `flutter -v` prefixes lines with a timing column ("[  +12 ms] "), melos with the
# package name ("[example]: "). Only stripped for noise matching, never from output.
_VERBOSE_PREFIX = re.compile(r"^\[[^\]]{0,40}\]:? ")


def _body(line: str) -> str:
    return _VERBOSE_PREFIX.sub("", line).strip()


def _welcome_banner(lines: list[str], i: int, end: int) -> range:
    start = next(
        (j for j in range(i, max(-1, i - 4), -1) if lines[j].lstrip().startswith(_BOX_TOP)), None
    )
    if start is None:
        return range(i, i + 1)
    stop = next(
        (
            j + 1
            for j in range(i, min(end, i + _BOX_MAX_LINES))
            if lines[j].lstrip().startswith(_BOX_BOTTOM)
        ),
        i + 1,
    )
    return range(start, stop)


def _outdated_list(lines: list[str], i: int, end: int) -> set[int]:
    """The summary at `i`, the hint below it and the package list above it."""
    noise = {i}
    if i + 1 < end and _OUTDATED_HINT.match(_body(lines[i + 1])):
        noise.add(i + 1)
    j = i - 1
    while j >= 0 and _PUB_STATUS.match(_body(lines[j])):
        j -= 1  # keep "Got dependencies!" etc.; it can say what changed
    while j >= 0 and _OUTDATED_ENTRY.match(_body(lines[j])):
        noise.add(j)
        j -= 1
    return noise


def noise_lines(lines: list[str], end: int) -> set[int]:
    """0-based indices of lines in lines[:end] that never help a diagnosis.

    fastlane update notices are already hidden by the focus view.
    """
    noise: set[int] = set()
    for i in range(end):
        body = _body(lines[i])
        if _WELCOME in body:
            noise.update(_welcome_banner(lines, i, end))
        elif _OUTDATED_SUMMARY.match(body):
            noise.update(_outdated_list(lines, i, end))
        elif _NODE_DEPRECATION.match(body) or body.startswith(_NODE_TRACE_HINT):
            noise.add(i)
    return noise


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Line:
    number: int
    """1-based line number in the stored log."""
    text: str
    repeat: int = 1

    def render(self) -> str:
        text = self.text
        if len(text) > MAX_LINE_CHARS:
            text = f"{text[:MAX_LINE_CHARS]}... [{len(text) - MAX_LINE_CHARS} chars truncated]"
        suffix = f" (repeated {self.repeat} times)" if self.repeat > 1 else ""
        return f"L{self.number}| {text}{suffix}"


def collapse_repeats(numbered: Iterable[tuple[int, str]]) -> Iterator[Line]:
    """Merge runs of identical consecutive lines (error cascades) into one Line."""
    current: Line | None = None
    for number, text in numbered:
        if current and text == current.text and number == current.number + current.repeat:
            current = Line(current.number, text, current.repeat + 1)
            continue
        if current:
            yield current
        current = Line(number, text)
    if current:
        yield current


def _blocks(lines: list[str], keep: set[int]) -> list[list[Line]]:
    runs: list[list[tuple[int, str]]] = []
    for i in sorted(keep):
        if runs and runs[-1][-1][0] == i:  # previous index i-1 has number i
            runs[-1].append((i + 1, lines[i]))
        else:
            runs.append([(i + 1, lines[i])])
    return [list(collapse_repeats(run)) for run in runs]


def render(blocks: list[list[Line]], total_lines: int) -> str:
    out: list[str] = []
    previous = 0
    for block in blocks:
        gap = block[0].number - previous - 1
        if gap:
            out.append(f"... ({gap} lines omitted)")
        out.extend(line.render() for line in block)
        last = block[-1]
        previous = last.number + last.repeat - 1
    if blocks and previous < total_lines:
        out.append(f"... ({total_lines - previous} lines omitted)")
    return "\n".join(out)


@dataclass(frozen=True)
class PreprocessedLog:
    total_lines: int
    failed_titles: list[str]
    blocks: list[list[Line]]
    """Contiguous runs of kept lines, in log order, with repeats collapsed."""
    budget_tokens: int
    dropped_windows: int
    """Candidate windows left out to stay within the budget."""

    @property
    def text(self) -> str:
        return render(self.blocks, self.total_lines)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.text)


Window = tuple[int, tuple[int, ...]]
"""(priority, 0-based indices)"""


def candidate_windows(
    text: str, failed_steps: Iterable[str] = ()
) -> tuple[list[str], list[Window], list[str]]:
    """Cleaned lines, ranked candidate windows (most important first), failed step titles."""
    located = locate_failed_steps(text, failed_steps)
    lines, end = located.lines, located.end
    dropped = located.hidden | noise_lines(lines, end)
    windows: list[Window] = [
        (PRIORITY[w.kind], w.indices)
        for w in focus_windows(
            located,
            context=MARKER_CONTEXT,
            max_snippets=None,
            lines_before_error=LINES_BEFORE_ERROR,
            hidden=dropped,
        )
    ]
    if not located.failed:
        # No failed step found: every marker in the log, plus its last lines.
        visible = [i for i in range(end) if i not in dropped]
        windows.append((PRIORITY["tail"], tuple(visible[-TAIL_LINES:])))
        for i in visible:
            if is_marker(lines[i]):
                lo, hi = i - MARKER_CONTEXT, i + MARKER_CONTEXT + 1
                window = tuple(j for j in range(max(0, lo), min(end, hi)) if j not in dropped)
                windows.append((PRIORITY["marker"], window))
    windows.sort(key=lambda w: w[0])  # stable: same priority keeps log order
    return lines, windows, [step.title for step in located.failed]


def preprocess(
    text: str,
    failed_steps: Iterable[str] = (),
    budget_tokens: int = DEFAULT_TOKEN_BUDGET,
) -> PreprocessedLog:
    lines, windows, titles = candidate_windows(text, failed_steps)
    total = len(lines)
    keep: set[int] = set()
    taken = 0
    for priority, indices in windows:
        if not set(indices) - keep:
            taken += 1
            continue
        candidate = keep | set(indices)
        fits = estimate_tokens(render(_blocks(lines, candidate), total)) <= budget_tokens
        if priority > PRIORITY["error_lead"] and not fits:
            break  # everything from here on ranks lower
        keep = candidate
        taken += 1
    return PreprocessedLog(
        total_lines=total,
        failed_titles=titles,
        blocks=_blocks(lines, keep),
        budget_tokens=budget_tokens,
        dropped_windows=len(windows) - taken,
    )
