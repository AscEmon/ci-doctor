"""Turn a stored job log into compact, line-numbered text for the analyzer.

Deterministic, no LLM. Line selection reuses the labeling focus view (failed steps,
nothing after "Post job cleanup.", successful composite sub-steps hidden) with wider
context around error markers, then drops known noise and collapses repeated lines.
Every kept line is prefixed with its 1-based number in the stored log ("L1234| "), so
the model can cite original line numbers as evidence.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from cidoc.logview import TAIL_LINES, focus_indices, is_marker, locate_failed_steps

MARKER_CONTEXT = 15

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


@dataclass(frozen=True)
class Line:
    number: int
    """1-based line number in the stored log."""
    text: str
    repeat: int = 1

    def render(self) -> str:
        suffix = f" (repeated {self.repeat} times)" if self.repeat > 1 else ""
        return f"L{self.number}| {self.text}{suffix}"


@dataclass(frozen=True)
class PreprocessedLog:
    total_lines: int
    failed_titles: list[str]
    blocks: list[list[Line]]
    """Contiguous runs of kept lines, in log order, with repeats collapsed."""

    @property
    def text(self) -> str:
        out: list[str] = []
        previous = 0
        for block in self.blocks:
            gap = block[0].number - previous - 1
            if gap:
                out.append(f"... ({gap} lines omitted)")
            out.extend(line.render() for line in block)
            last = block[-1]
            previous = last.number + last.repeat - 1
        if self.blocks and previous < self.total_lines:
            out.append(f"... ({self.total_lines - previous} lines omitted)")
        return "\n".join(out)


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


def select_indices(
    text: str, failed_steps: Iterable[str] = ()
) -> tuple[list[str], set[int], list[str]]:
    """Cleaned lines, 0-based indices to keep, and the titles of the failed steps."""
    located = locate_failed_steps(text, failed_steps)
    lines, end = located.lines, located.end
    dropped = located.hidden | noise_lines(lines, end)
    keep = focus_indices(located, context=MARKER_CONTEXT, max_snippets=None, hidden=dropped)
    if not located.failed:
        # No failed step found: every marker in the log, plus its last lines.
        for i in range(end):
            if is_marker(lines[i]) and i not in dropped:
                keep.update(range(max(0, i - MARKER_CONTEXT), min(end, i + MARKER_CONTEXT + 1)))
        visible = [i for i in range(end) if i not in dropped]
        keep.update(visible[-TAIL_LINES:])
    return lines, keep - dropped, [step.title for step in located.failed]


def preprocess(text: str, failed_steps: Iterable[str] = ()) -> PreprocessedLog:
    lines, keep, titles = select_indices(text, failed_steps)
    blocks: list[list[tuple[int, str]]] = []
    for i in sorted(keep):
        if blocks and blocks[-1][-1][0] == i:  # previous index i-1 has number i
            blocks[-1].append((i + 1, lines[i]))
        else:
            blocks.append([(i + 1, lines[i])])
    return PreprocessedLog(
        total_lines=len(lines),
        failed_titles=titles,
        blocks=[list(collapse_repeats(block)) for block in blocks],
    )
