"""Print an agent run in the terminal: its tool calls, totals and evidence files.

Tool output, arguments and model text are untrusted: always rich Text, never markup.
"""

import json
from typing import Any

from rich.console import Console
from rich.text import Text

from cidoc.agent.prompt import AgentPrompt
from cidoc.agent.trajectory import Trajectory
from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.preprocess import PreprocessedLog, estimate_tokens
from cidoc.analyze.schema import AgentDiagnosis

ARGS_MAX_CHARS = 60


def format_args(args: dict[str, Any]) -> str:
    text = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return text if len(text) <= ARGS_MAX_CHARS else text[:ARGS_MAX_CHARS] + "..."


def _cost(value: float | None) -> str:
    return "cost unknown" if value is None else f"${value:.6f}"


def trajectory_summary(trajectory: Trajectory) -> str:
    calls = len(trajectory.tool_calls)
    parts = [
        f"{calls} tool call{'s' if calls != 1 else ''} (max {trajectory.max_steps})",
        f"{trajectory.model_calls} model calls",
        f"{trajectory.input_tokens:,} in / {trajectory.output_tokens:,} out tokens",
        _cost(trajectory.cost),
    ]
    if trajectory.call_input_tokens:
        steps = " → ".join(f"{n:,}" for n in trajectory.call_input_tokens)
        parts.append(f"input per call {steps} (of {trajectory.max_input_tokens:,})")
    if trajectory.forced:
        parts.append("forced submit: " + ", ".join(trajectory.forced))
    return "; ".join(parts)


def render_tool_calls(console: Console, trajectory: Trajectory, indent: str = "") -> None:
    """One line per tool call (name, args, status, size, latency), its preview below."""
    if not trajectory.tool_calls:
        console.print(Text(f"{indent}no tool calls: answered from the log excerpt", "dim"))
        return
    for call in trajectory.tool_calls:
        row = Text(f"{indent}{call.step}. ")
        row.append(call.name, style="bold")
        row.append(f"({format_args(call.args)})  ")
        row.append("ok" if call.ok else "error", style="green" if call.ok else "yellow")
        detail = f" · {call.output_tokens} tok · {call.latency_s:.2f}s"
        row.append(detail + (" · cached" if call.cached else ""), style="dim")
        console.print(row, soft_wrap=True, highlight=False)
        console.print(Text(f"{indent}   {call.preview}", style="dim"), soft_wrap=True)


def render_warnings(console: Console, trajectory: Trajectory, indent: str = "") -> None:
    for warning in trajectory.warnings:
        console.print(Text(f"{indent}warning: {warning}", style="yellow"), soft_wrap=True)


def render_trajectory(console: Console, trajectory: Trajectory, cached: bool = False) -> None:
    title = "agent trajectory" + (" (cached: no calls this run)" if cached else "")
    console.rule(title, style="cyan")
    render_tool_calls(console, trajectory)
    console.print(Text(trajectory_summary(trajectory), style="dim"), soft_wrap=True)
    render_warnings(console, trajectory)


def render_evidence_files(console: Console, result: AnalysisResult) -> None:
    diagnosis = result.diagnosis
    if isinstance(diagnosis, AgentDiagnosis) and diagnosis.evidence_files:
        console.print(Text("files", style="bold"))
        for item in diagnosis.evidence_files:
            console.print(Text(f"  {item}"), highlight=False)


def render_agent_input(
    console: Console, prompt: AgentPrompt, log: PreprocessedLog, model: str
) -> None:
    user = prompt.user(log)
    console.rule(f"system instruction (~{estimate_tokens(prompt.system)} tokens)", style="cyan")
    console.print(Text(prompt.system), soft_wrap=True, highlight=False)
    console.rule(f"user message (~{estimate_tokens(user)} tokens)", style="cyan")
    console.print(Text(user), soft_wrap=True, highlight=False)
    console.rule("functions", style="cyan")
    for function in prompt.functions:
        console.print(Text.assemble((function.name, "bold"), f": {function.description}"))
    console.rule(style="cyan")
    console.print(
        Text(
            f"Agent prompt {prompt.version} (Phase 1 {prompt.base.version}) to {model}.",
            style="dim",
        )
    )
