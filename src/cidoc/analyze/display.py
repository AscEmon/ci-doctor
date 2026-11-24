"""Print an analysis result in the terminal.

Model output and log lines are untrusted text: they are always rich Text, never markup.
"""

from rich.console import Console
from rich.text import Text

from cidoc.analyze.analyzer import AnalysisResult

EVIDENCE_MAX_CHARS = 160


def render_result(console: Console, result: AnalysisResult, lines: list[str]) -> None:
    """`lines` are the cleaned log lines, to show the text of each evidence line."""
    diagnosis = result.diagnosis
    if diagnosis is None:
        console.print(Text.assemble(("analysis failed: ", "bold red"), result.failure or "?"))
        return

    def field(name: str, value: str, style: str = "") -> None:
        console.print(Text.assemble((f"{name:<11}", "bold"), (value, style)))

    field("category", diagnosis.category, "bold cyan")
    field("confidence", f"{diagnosis.confidence:.2f}")
    field("root cause", diagnosis.root_cause)
    field("fix", diagnosis.suggested_fix)
    console.print(Text("evidence", style="bold"))
    width = len(str(len(lines)))
    for number in diagnosis.evidence_lines:
        text = lines[number - 1] if number <= len(lines) else "(past the end of the log)"
        if len(text) > EVIDENCE_MAX_CHARS:
            text = text[:EVIDENCE_MAX_CHARS] + "..."
        row = Text(f"  {number:>{width}} │ ", style="dim")
        row.append(text)
        console.print(row, overflow="fold", highlight=False)
