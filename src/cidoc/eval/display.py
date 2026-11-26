"""Print eval results in the terminal."""

from rich.console import Console
from rich.table import Table
from rich.text import Text

from cidoc.agent.display import render_tool_calls, trajectory_summary
from cidoc.analyze.costguard import CallEstimate, RunEstimate
from cidoc.analyze.schema import AgentDiagnosis
from cidoc.eval.compare import Comparison, MetricDelta
from cidoc.eval.metrics import FAILED, EvalItem, Metrics
from cidoc.eval.results import RunFile, RunInfo
from cidoc.llm.pricing import Price

TOP_CONFUSIONS = 8


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def usd(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.6f}"


def seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def render_item(console: Console, n: int, total: int, item: EvalItem) -> None:
    if item.error is not None:
        row = Text(f"[{n}/{total}] job {item.job_id} ", style="dim")
        row.append("!", style="yellow")
        row.append(f" {item.gold} → not scored: {item.error}", style="yellow")
        console.print(row, overflow="ellipsis", no_wrap=True, highlight=False)
        return
    mark = ("✓", "green") if item.correct else ("✗", "red")
    predicted = item.predicted or f"failed: {item.result.failure}"
    row = Text(f"[{n}/{total}] job {item.job_id} ", style="dim")
    row.append(mark[0], style=mark[1])
    row.append(f" {item.gold} → {predicted}")
    trajectory = item.result.trajectory
    if trajectory is not None:
        tools = len(trajectory.tool_calls)
        row.append(f" [{tools} tool{'s' if tools != 1 else ''}]", style="dim")
    if item.result.cached:
        row.append(" (cached)", style="dim")
    console.print(row, overflow="ellipsis", no_wrap=True, highlight=False)


def render_summary(console: Console, info: RunInfo, metrics: Metrics) -> None:
    headline = Table(title=f"eval: {info.model}", title_justify="left", show_header=False)
    headline.add_column(style="bold")
    headline.add_column(justify="right")
    rows = [
        ("split", info.split),
        ("items scored", f"{metrics.n} of {info.labels_total} labels"),
        ("accuracy", f"{pct(metrics.accuracy)} ({metrics.correct}/{metrics.n})"),
        ("macro-F1", f"{pct(metrics.macro_f1)} ({len(metrics.macro_f1_categories)} categories)"),
        ("accuracy, confidence 3", f"{pct(metrics.confident_accuracy)} (n={metrics.confident_n})"),
        (
            "evidence hit rate (±3)",
            f"{pct(metrics.evidence_hit_rate)} ({metrics.evidence_hits}/{metrics.evidence_n})",
        ),
        ("mean cost per item", usd(metrics.mean_cost)),
        ("latency p50 / p95", f"{seconds(metrics.latency_p50)} / {seconds(metrics.latency_p95)}"),
        ("failed / blocked", f"{metrics.failed} / {metrics.blocked}"),
        ("errored (API, not scored)", str(metrics.errored)),
        ("prompt version", info.prompt_version or "—"),
        ("guideline", info.guideline_hash),
        ("model calls / item", number(metrics.mean_model_calls)),
        (
            "tokens / item (in / out)",
            f"{number(metrics.mean_input_tokens, 0)} / {number(metrics.mean_output_tokens, 0)}",
        ),
    ]
    if info.agent:
        rows += [
            ("agent tools", ", ".join(info.tools or []) or "none"),
            ("max steps", str(info.max_steps)),
            ("tool calls / item", number(metrics.mean_tool_calls)),
            ("items that used a tool", f"{pct(metrics.used_tools_rate)} of {metrics.agent_items}"),
            ("items that hit max_steps", str(metrics.hit_max_steps)),
        ]
    for name, value in rows:
        headline.add_row(name, value)
    console.print(headline)
    if metrics.tool_usage:
        usage = ", ".join(f"{name} {count}" for name, count in metrics.tool_usage.items())
        console.print(Text(f"Tool usage: {usage}"), soft_wrap=True)

    per = Table(title="per category", title_justify="left")
    for column in ("category", "support", "predicted", "precision", "recall", "F1"):
        per.add_column(column, justify="left" if column == "category" else "right")
    for category, stats in metrics.per_category.items():
        in_macro = category in metrics.macro_f1_categories
        per.add_row(
            category,
            str(stats.support),
            str(stats.predicted),
            pct(stats.precision),
            pct(stats.recall),
            pct(stats.f1),
            style="" if in_macro else "dim",
        )
    console.print(per)
    excluded = ", ".join(f"{c} ({n})" for c, n in metrics.excluded_categories.items())
    console.print(Text(f"Not in macro-F1 (< 5 labels): {excluded or 'none'}", style="dim"))

    confusions = sorted(
        (
            (count, gold, predicted)
            for gold, row in metrics.confusion.items()
            for predicted, count in row.items()
            if predicted != gold
        ),
        reverse=True,
    )[:TOP_CONFUSIONS]
    if confusions:
        console.print(Text("Most common confusions (label → prediction):", style="bold"))
        for count, gold, predicted in confusions:
            console.print(Text(f"  {count}x {gold} → {predicted}"))


def duration(value: float) -> str:
    minutes, secs = divmod(round(value), 60)
    return f"~{minutes}m {secs:02d}s" if minutes else f"~{secs}s"


def calls_within(calls: list[CallEstimate], price: Price, max_cost: float) -> int:
    """How many of `calls`, in order, fit in `max_cost`."""
    spent = 0.0
    for n, call in enumerate(calls):
        spent += call.cost(price)
        if spent > max_cost:
            return n
    return len(calls)


def render_estimate(
    console: Console,
    model: str,
    estimate: RunEstimate,
    rpm: int,
    calls: list[CallEstimate],
    price: Price | None,
    max_cost: float | None,
    max_steps: int | None = None,
) -> None:
    """`max_steps` is set for agent runs: their items are estimated at the worst case."""
    title = f"dry run: {model}" + (" (agent)" if max_steps is not None else "")
    table = Table(title=title, title_justify="left")
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.show_header = False
    rows = [
        ("items", str(estimate.items)),
        ("cached (free)", str(estimate.cached)),
        ("API calls" if max_steps is None else "model calls (worst case)", str(estimate.calls)),
        ("input tokens (est.)", f"{estimate.input_tokens:,}"),
        ("output tokens (allowance)", f"{estimate.output_tokens:,}"),
        ("cost (est.)", usd(estimate.cost)),
        (f"duration at {rpm} RPM", duration(estimate.duration_s)),
    ]
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)
    notes = [
        "No API calls were made.",
        "Estimates use 4 characters per token; validation retries are not included.",
    ]
    if max_steps == 0:
        notes.append(
            "Agent with no tools: one model call per uncached item (a text re-prompt or"
            " validation retry, if needed, is not counted)."
        )
    elif max_steps is not None:
        notes.append(
            f"Agent: each uncached item is counted at its worst case: up to {max_steps}"
            f" tool rounds and the final submit (at most {max_steps + 1} model calls), each"
            " resending the conversation, until the per-item input-token cap is reached;"
            " most items stop sooner. GitHub calls for tools are free and not counted."
        )
    if price is None and estimate.calls:
        notes.append(f"No price configured for {model}: set CIDOC_PRICES.")
    if max_cost is not None and price is not None:
        fit = calls_within(calls, price, max_cost)
        if fit < len(calls):
            notes.append(
                f"--max-cost ${max_cost} covers about {fit} of {len(calls)} calls;"
                " the run would stop there and save what it scored."
            )
    for note in notes:
        console.print(Text(note, style="dim"))


_COUNTS = ("failed", "blocked", "hit_max_steps")
_MEANS = ("mean_model_calls", "mean_tool_calls")


def _metric(delta: MetricDelta, value: float | None) -> str:
    if delta.key == "mean_cost":
        return usd(value)
    if delta.key.startswith("latency"):
        return seconds(value)
    if delta.key in _COUNTS:
        return "—" if value is None else str(int(value))
    if delta.key in _MEANS:
        return number(value)
    if delta.key == "mean_input_tokens":
        return number(value, 0)
    return pct(value)


def _delta(delta: MetricDelta) -> Text:
    change = delta.delta
    if change is None:
        return Text("—", style="dim")
    if delta.key == "mean_cost":
        shown = f"{change:+.6f}"
    elif delta.key.startswith("latency"):
        shown = f"{change:+.2f}s"
    elif delta.key in _COUNTS:
        shown = f"{int(change):+d}"
    elif delta.key in _MEANS:
        shown = f"{change:+.2f}"
    elif delta.key == "mean_input_tokens":
        shown = f"{change:+,.0f}"
    else:
        shown = f"{change * 100:+.1f} pts"
    style = {True: "green", False: "red", None: "dim"}[delta.improved]
    return Text(shown, style=style)


def _run_label(run: RunFile) -> str:
    version = run.info.prompt_version or "no prompt"
    model = run.info.model
    if run.info.agent:
        tools = ",".join(run.info.tools or []) or "none"
        model += f" agent (max {run.info.max_steps} steps, tools {tools})"
    return f"{model} ({version}, {run.info.split} split, {run.info.created_at[:16]})"


def render_comparison(
    console: Console, a: RunFile, b: RunFile, comparison: Comparison, max_rows: int = 50
) -> None:
    console.print(Text(f"A: {_run_label(a)}"), soft_wrap=True)
    console.print(Text(f"B: {_run_label(b)}"), soft_wrap=True)
    scope = f"{comparison.shared} shared items"
    if comparison.recomputed:
        scope += (
            f"; metrics recomputed on those (only in A: {len(comparison.only_a)},"
            f" only in B: {len(comparison.only_b)})"
        )
    console.print(Text(scope, style="dim"))
    console.print(Text("Both runs scored against the current labels.jsonl.", style="dim"))
    if comparison.label_changes:
        console.print(
            Text(
                f"Warning: labels changed between the runs for"
                f" {len(comparison.label_changes)} jobs (older run → newer run):",
                style="yellow",
            )
        )
        for change in comparison.label_changes:
            console.print(
                Text(f"  job {change.job_id}: {change.old} → {change.new}", style="yellow")
            )
    for note in comparison.notes:
        console.print(Text(note, style="yellow"))

    table = Table(title="metrics (Δ = B - A)", title_justify="left")
    for column in ("metric", "A", "B", "Δ"):
        table.add_column(column, justify="left" if column == "metric" else "right")
    for delta in comparison.deltas:
        table.add_row(delta.label, _metric(delta, delta.a), _metric(delta, delta.b), _delta(delta))
    console.print(table)
    console.print(
        Text(
            f"Both right {comparison.both_right}, only A right {comparison.only_a_right},"
            f" only B right {comparison.only_b_right}, both wrong {comparison.both_wrong}."
        )
    )

    if not comparison.disagreements:
        console.print(Text("The runs agree on every shared item.", style="dim"))
        return
    rows = Table(
        title=f"{len(comparison.disagreements)} items where A and B disagree",
        title_justify="left",
    )
    for column in ("job", "label", "A", "B"):
        rows.add_column(column)
    for d in comparison.disagreements[:max_rows]:
        rows.add_row(
            str(d.job_id),
            d.gold,
            Text(d.a or FAILED, style="green" if d.a_correct else ""),
            Text(d.b or FAILED, style="green" if d.b_correct else ""),
        )
    console.print(rows)
    if len(comparison.disagreements) > max_rows:
        hidden = len(comparison.disagreements) - max_rows
        console.print(Text(f"({hidden} more not shown)", style="dim"))


def select_items(items: list[EvalItem], wrong: bool, job: int | None) -> list[EvalItem]:
    """--job keeps one job; --wrong keeps scored items that are not correct (an
    item that errored was never scored, so it is neither right nor wrong)."""
    selected = [item for item in items if job is None or item.job_id == job]
    if wrong:
        selected = [item for item in selected if item.error is None and not item.correct]
    return selected


def render_items(
    console: Console,
    run: RunFile,
    items: list[EvalItem],
    fallback_root_causes: dict[int, str],
    show_trajectory: bool = False,
) -> None:
    """One block per item: the human label next to the prediction. Agent items list the
    tools they used; with `show_trajectory` (--show --job) every tool call is printed.

    `fallback_root_causes` (from labels.jsonl) fills in the human root cause for runs
    saved before run files stored it; such causes are marked as possibly newer.
    """
    console.print(Text(f"{_run_label(run)}", style="bold"))
    for item in items:
        diagnosis = item.result.diagnosis
        if item.error is not None:
            mark, mark_style = "!", "yellow"
        elif item.correct:
            mark, mark_style = "✓", "green"
        else:
            mark, mark_style = "✗", "red"
        header = Text(f"job {item.job_id}  ")
        header.append(mark, style=mark_style)
        header.append(f"  label {item.gold} (confidence {item.gold_confidence})  →  ")
        if item.error is not None:
            header.append(f"not scored: {item.error}", style="yellow")
        elif diagnosis is None:
            header.append(f"failed: {item.result.failure}", style="red")
        else:
            header.append(f"{diagnosis.category} ({diagnosis.confidence:.2f})", style="bold")
        console.rule(style="dim")
        console.print(header, soft_wrap=True, highlight=False)

        human = item.gold_root_cause
        note = ""
        if human is None and item.job_id in fallback_root_causes:
            human, note = fallback_root_causes[item.job_id], "  (from current labels.jsonl)"
        rows = [
            ("human", human or "(not recorded)", note),
            ("predicted", diagnosis.root_cause if diagnosis else "—", ""),
            (
                "evidence",
                f"predicted {diagnosis.evidence_lines if diagnosis else []}"
                f"  human {item.gold_evidence}",
                "",
            ),
        ]
        if isinstance(diagnosis, AgentDiagnosis) and diagnosis.evidence_files:
            rows.append(("files", ", ".join(diagnosis.evidence_files), ""))
        trajectory = item.result.trajectory
        if trajectory is not None and not show_trajectory:
            used = ", ".join(call.name for call in trajectory.tool_calls) or "none"
            rows.append(("tools", used, f"  ({trajectory_summary(trajectory)})"))
        for name, value, row_note in rows:
            row = Text(f"  {name + ':':<11}", style="dim")
            row.append(value)
            row.append(row_note, style="dim")
            console.print(row, soft_wrap=True, highlight=False)
        if trajectory is not None and show_trajectory:
            console.print(Text("  trajectory:", style="dim"))
            render_tool_calls(console, trajectory, indent="    ")
            console.print(
                Text(f"    {trajectory_summary(trajectory)}", style="dim"), soft_wrap=True
            )
    console.rule(style="dim")
