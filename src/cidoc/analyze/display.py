"""Print an analysis result in the terminal.

Model output and log lines are untrusted text: they are always rich Text, never markup.
"""

from rich.console import Console
from rich.table import Table
from rich.text import Text

from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.costguard import EXPECTED_OUTPUT_TOKENS
from cidoc.analyze.preprocess import PreprocessedLog, estimate_tokens
from cidoc.analyze.prompt import Prompt
from cidoc.analyze.usage import summarize
from cidoc.llm.pricing import Price

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


def format_cost(cost: float | None) -> str:
    return "unknown" if cost is None else f"${cost:.6f}"


def render_usage(console: Console, result: AnalysisResult, price: Price | None) -> None:
    total = summarize(result.calls, price)
    if result.cached:
        console.print(
            Text(
                f"cached: no API call, $0 this run (originally {total.input_tokens} in /"
                f" {total.output_tokens} out tokens, {format_cost(total.cost)})",
                style="dim",
            )
        )
        return
    table = Table(title=f"usage ({result.model})", title_justify="left", show_edge=False)
    for column in ("call", "input tok", "output tok", "latency", "attempts", "cost"):
        table.add_column(column, justify="left" if column == "call" else "right")
    for n, call in enumerate(result.calls, start=1):
        single = summarize([call], price)
        table.add_row(
            "first" if n == 1 else "retry",
            str(call.input_tokens),
            str(call.output_tokens),
            f"{call.latency_s:.2f}s",
            str(call.attempts),
            format_cost(single.cost),
        )
    if len(result.calls) > 1:
        table.add_row(
            "total",
            str(total.input_tokens),
            str(total.output_tokens),
            f"{total.latency_s:.2f}s",
            "",
            format_cost(total.cost),
            style="bold",
        )
    console.print(table)
    if price is None:
        console.print(
            Text(f"No price configured for {result.model}: set CIDOC_PRICES.", style="dim")
        )


def render_input(console: Console, prompt: Prompt, log: PreprocessedLog, model: str) -> None:
    """Exactly what is sent: the system instruction, then the user message with the log."""
    user = prompt.user(log)
    system_tokens, user_tokens = estimate_tokens(prompt.system), estimate_tokens(user)
    console.rule(f"system instruction (~{system_tokens} tokens)", style="cyan")
    # soft_wrap: no inserted line breaks, so copied text is exactly what was sent.
    console.print(Text(prompt.system), soft_wrap=True, highlight=False)
    console.rule(
        f"user message (~{user_tokens} tokens; log excerpt ~{log.estimated_tokens}"
        f" of {log.budget_tokens} budget)",
        style="cyan",
    )
    console.print(Text(user), soft_wrap=True, highlight=False)
    console.rule(style="cyan")
    console.print(
        Text(
            f"Input ~{system_tokens + user_tokens} tokens (4 chars per token) to {model},"
            f" prompt {prompt.version}; output allowance {EXPECTED_OUTPUT_TOKENS}."
            " A retry resends the user message with the rejection reason appended.",
            style="dim",
        )
    )


def render_outputs(console: Console, result: AnalysisResult) -> None:
    """The model's raw text for each call, before validation."""
    if not result.outputs:
        note = "(no raw output"
        if result.cached:
            note += ": this result was cached before raw outputs were kept; use --refresh"
        console.print(Text(note + ")", style="dim"))
        return
    for n, output in enumerate(result.outputs, start=1):
        title = "model output" if len(result.outputs) == 1 else f"model output, call {n}"
        if result.cached:
            title += " (cached)"
        console.rule(title, style="magenta")
        console.print(Text(output or "(empty)"), soft_wrap=True, highlight=False)
    console.rule(style="magenta")
