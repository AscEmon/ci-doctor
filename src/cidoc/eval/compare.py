"""Compare two eval runs: metric deltas and the items where they disagree.

Both runs are re-scored against one label set, the current labels.jsonl, not the labels
each run file stored: otherwise a relabeled job could count as right in one run and
wrong in the other for the same prediction. Jobs with no current label are left out.

Runs are compared on the jobs both scored (a --limit run or one stopped by --max-cost
scores fewer), and both sides' metrics are recomputed on those jobs.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from cidoc.eval.metrics import EvalItem, compute_metrics
from cidoc.eval.results import RunFile
from cidoc.labeling.labels import Label
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
class LabelChange:
    job_id: int
    old: str
    """Label stored in the older run."""
    new: str
    """Label stored in the newer run."""


@dataclass(frozen=True)
class Comparison:
    shared: int
    only_a: list[int]
    only_b: list[int]
    recomputed: bool
    """True when the runs scored different jobs (metrics cover only the shared ones)."""
    deltas: list[MetricDelta]
    disagreements: list[Disagreement]
    """Shared jobs with different predictions: one side right first, then both wrong."""
    both_right: int
    only_a_right: int
    only_b_right: int
    both_wrong: int
    notes: list[str]
    label_changes: list[LabelChange]
    """Shared jobs whose stored label differs between the runs (older run's → newer's)."""
    unlabeled: list[int]
    """Jobs either run scored that have no current label; left out of the comparison."""


def _shared_metrics(items: list[EvalItem], shared: set[int], price: Price | None) -> dict[str, Any]:
    return asdict(compute_metrics([i for i in items if i.job_id in shared], price))


def relabel(item: EvalItem, label: Label) -> EvalItem:
    """The item scored against `label` instead of the label its run stored."""
    return replace(
        item,
        gold=label.category,
        gold_confidence=label.confidence,
        gold_evidence=list(label.evidence_lines),
        gold_root_cause=label.root_cause,
    )


def _label_changes(
    a: RunFile, b: RunFile, a_items: dict[int, EvalItem], b_items: dict[int, EvalItem]
) -> list[LabelChange]:
    a_is_older = a.info.created_at <= b.info.created_at
    older, newer = (a_items, b_items) if a_is_older else (b_items, a_items)
    return [
        LabelChange(job_id, older[job_id].gold, newer[job_id].gold)
        for job_id in sorted(set(older) & set(newer))
        if older[job_id].gold != newer[job_id].gold
    ]


def compare_runs(
    a: RunFile,
    b: RunFile,
    price_for: Callable[[str], Price | None],
    labels: Mapping[int, Label],
) -> Comparison:
    """Compare A and B, scoring both against `labels` (job id -> current label)."""
    # Errored items were never scored, so they are not compared either.
    a_stored = {item.job_id: item for item in a.items if item.error is None}
    b_stored = {item.job_id: item for item in b.items if item.error is None}
    label_changes = _label_changes(a, b, a_stored, b_stored)
    unlabeled = sorted((set(a_stored) | set(b_stored)) - set(labels))
    a_items = {j: relabel(i, labels[j]) for j, i in a_stored.items() if j in labels}
    b_items = {j: relabel(i, labels[j]) for j, i in b_stored.items() if j in labels}
    shared = set(a_items) & set(b_items)
    recomputed = set(a_items) != set(b_items)
    a_metrics = _shared_metrics(list(a_items.values()), shared, price_for(a.info.model))
    b_metrics = _shared_metrics(list(b_items.values()), shared, price_for(b.info.model))
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
    if a.info.split != b.info.split:
        notes.append(f"The runs scored different splits ({a.info.split} vs {b.info.split}).")
    if a.info.guideline_hash != b.info.guideline_hash:
        notes.append("The runs used different labeling guidelines (guideline hash differs).")
    if unlabeled:
        notes.append(
            f"Left out {len(unlabeled)} jobs that have no label in labels.jsonl now:"
            f" {', '.join(map(str, unlabeled))}"
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
        label_changes=label_changes,
        unlabeled=unlabeled,
        **counts,
    )
