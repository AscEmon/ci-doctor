"""Print eval results in the terminal."""

from rich.console import Console
from rich.table import Table
from rich.text import Text

from cidoc.analyze.costguard import CallEstimate, RunEstimate
from cidoc.eval.compare import Comparison, MetricDelta
from cidoc.eval.metrics import FAILED, EvalItem, Metrics
from cidoc.eval.results import RunFile, RunInfo
from cidoc.llm.pricing import Price

TOP_CONFUSIONS = 8


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def usd(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.6f}"


def seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def render_item(console: Console, n: int, total: int, item: EvalItem) -> None:
    mark = ("✓", "green") if item.correct else ("✗", "red")
    predicted = item.predicted or f"failed: {item.result.failure}"
    row = Text(f"[{n}/{total}] job {item.job_id} ", style="dim")
    row.append(mark[0], style=mark[1])
    row.append(f" {item.gold} → {predicted}")
    if item.result.cached:
        row.append(" (cached)", style="dim")
    console.print(row, overflow="ellipsis", no_wrap=True, highlight=False)


def render_summary(console: Console, info: RunInfo, metrics: Metrics) -> None:
    headline = Table(title=f"eval: {info.model}", title_justify="left", show_header=False)
    headline.add_column(style="bold")
    headline.add_column(justify="right")
    rows = [
        ("items", f"{metrics.n} of {info.labels_total} labels"),
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
        ("prompt version", info.prompt_version or "—"),
        ("guideline", info.guideline_hash),
    ]
    for name, value in rows:
        headline.add_row(name, value)
    console.print(headline)

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
) -> None:
    table = Table(title=f"dry run: {model}", title_justify="left")
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.show_header = False
    rows = [
        ("items", str(estimate.items)),
        ("cached (free)", str(estimate.cached)),
        ("API calls", str(estimate.calls)),
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


def _metric(delta: MetricDelta, value: float | None) -> str:
    if delta.key == "mean_cost":
        return usd(value)
    if delta.key.startswith("latency"):
        return seconds(value)
    if delta.key in ("failed", "blocked"):
        return "—" if value is None else str(int(value))
    return pct(value)


def _delta(delta: MetricDelta) -> Text:
    change = delta.delta
    if change is None:
        return Text("—", style="dim")
    if delta.key == "mean_cost":
        shown = f"{change:+.6f}"
    elif delta.key.startswith("latency"):
        shown = f"{change:+.2f}s"
    elif delta.key in ("failed", "blocked"):
        shown = f"{int(change):+d}"
    else:
        shown = f"{change * 100:+.1f} pts"
    style = {True: "green", False: "red", None: "dim"}[delta.improved]
    return Text(shown, style=style)


def _run_label(run: RunFile) -> str:
    version = run.info.prompt_version or "no prompt"
    return f"{run.info.model} ({version}, {run.info.created_at[:16]})"


def render_comparison(
    console: Console, a: RunFile, b: RunFile, comparison: Comparison, max_rows: int = 50
) -> None:
    console.print(Text(f"A: {_run_label(a)}"))
    console.print(Text(f"B: {_run_label(b)}"))
    scope = f"{comparison.shared} shared items"
    if comparison.recomputed:
        scope += (
            f"; metrics recomputed on those (only in A: {len(comparison.only_a)},"
            f" only in B: {len(comparison.only_b)})"
        )
    console.print(Text(scope, style="dim"))
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
