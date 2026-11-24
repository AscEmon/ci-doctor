"""Render log views for a human labeler in the terminal."""

from rich.console import Console
from rich.text import Text

from cidoc.logview import FocusView, LogView, NumberedLine


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
