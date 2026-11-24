"""Print eval results in the terminal."""

from rich.console import Console
from rich.table import Table
from rich.text import Text

from cidoc.eval.metrics import EvalItem, Metrics
from cidoc.eval.results import RunInfo

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
