"""Compare two eval runs: metric deltas and the items where they disagree.

Runs are compared on the jobs both scored. When those are all of each run's jobs, each
side keeps the metrics it saved; otherwise (a --limit run, a run stopped by --max-cost)
both sides are recomputed on the shared jobs, so the numbers measure the same thing.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from cidoc.eval.metrics import EvalItem, compute_metrics
from cidoc.eval.results import RunFile
from cidoc.llm.pricing import Price

# (key in the metrics dict, label, higher is better)
COMPARED_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("accuracy", "accuracy", True),
    ("macro_f1", "macro-F1", True),
    ("confident_accuracy", "accuracy, confidence 3", True),
    ("evidence_hit_rate", "evidence hit rate (±3)", True),
    ("mean_cost", "mean cost per item", False),
    ("latency_p50", "latency p50", False),
    ("latency_p95", "latency p95", False),
    ("failed", "failed", False),
    ("blocked", "blocked", False),
)


@dataclass(frozen=True)
class MetricDelta:
    key: str
    label: str
    a: float | None
    b: float | None
    higher_is_better: bool

    @property
    def delta(self) -> float | None:
        return None if self.a is None or self.b is None else self.b - self.a

    @property
    def improved(self) -> bool | None:
        """Whether B is better than A; None when equal or unknown."""
        if self.delta is None or self.delta == 0:
            return None
        return (self.delta > 0) == self.higher_is_better


@dataclass(frozen=True)
class Disagreement:
    job_id: int
    gold: str
    a: str | None
    """A's predicted category; None when A failed on this item."""
    b: str | None

    @property
    def a_correct(self) -> bool:
        return self.a == self.gold

    @property
    def b_correct(self) -> bool:
        return self.b == self.gold


@dataclass(frozen=True)
class Comparison:
    shared: int
    only_a: list[int]
    only_b: list[int]
    recomputed: bool
    """True when the runs scored different jobs and metrics were recomputed on shared ones."""
    deltas: list[MetricDelta]
    disagreements: list[Disagreement]
    """Shared jobs with different predictions: one side right first, then both wrong."""
    both_right: int
    only_a_right: int
    only_b_right: int
    both_wrong: int
    notes: list[str]


def _shared_metrics(items: list[EvalItem], shared: set[int], price: Price | None) -> dict[str, Any]:
    return asdict(compute_metrics([i for i in items if i.job_id in shared], price))


def compare_runs(a: RunFile, b: RunFile, price_for: Callable[[str], Price | None]) -> Comparison:
    # Errored items were never scored, so they are not compared either.
    a_items = {item.job_id: item for item in a.items if item.error is None}
    b_items = {item.job_id: item for item in b.items if item.error is None}
    shared = set(a_items) & set(b_items)
    recomputed = set(a_items) != set(b_items)
    if recomputed:
        a_metrics = _shared_metrics(a.items, shared, price_for(a.info.model))
        b_metrics = _shared_metrics(b.items, shared, price_for(b.info.model))
    else:
        a_metrics, b_metrics = a.metrics, b.metrics
    deltas = [
        MetricDelta(key, label, a_metrics.get(key), b_metrics.get(key), higher)
        for key, label, higher in COMPARED_METRICS
    ]

    disagreements: list[Disagreement] = []
    counts = {"both_right": 0, "only_a_right": 0, "only_b_right": 0, "both_wrong": 0}
    for job_id in [j for j in a_items if j in shared]:  # A's order
        ia, ib = a_items[job_id], b_items[job_id]
        if ia.correct and ib.correct:
            counts["both_right"] += 1
        elif ia.correct:
            counts["only_a_right"] += 1
        elif ib.correct:
            counts["only_b_right"] += 1
        else:
            counts["both_wrong"] += 1
        if ia.predicted != ib.predicted:
            disagreements.append(Disagreement(job_id, ia.gold, ia.predicted, ib.predicted))
    disagreements.sort(key=lambda d: not (d.a_correct or d.b_correct))  # stable

    notes: list[str] = []
    if a.info.guideline_hash != b.info.guideline_hash:
        notes.append("The runs used different labeling guidelines (guideline hash differs).")
    if any(a_items[j].gold != b_items[j].gold for j in shared):
        notes.append(
            "Some labels changed between the runs; each side is scored against its own,"
            " and disagreements show run A's label."
        )
    for name, run in (("A", a), ("B", b)):
        if not run.info.complete:
            notes.append(f"Run {name} is partial: {run.info.stopped}")

    return Comparison(
        shared=len(shared),
        only_a=sorted(set(a_items) - shared),
        only_b=sorted(set(b_items) - shared),
        recomputed=recomputed,
        deltas=deltas,
        disagreements=disagreements,
        notes=notes,
        **counts,
    )
