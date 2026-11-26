"""Score predictions against human labels. Pure functions, no I/O.

A result without a diagnosis (failed validation, blocked) has no prediction: it counts as
wrong everywhere and shows up as FAILED in the confusion matrix. That is model behavior.

An item whose API call still failed after retries (EvalItem.error) says nothing about the
model: it is left out of every metric except the `errored` count, and the next run
retries it.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.usage import summarize
from cidoc.labeling.labels import CATEGORIES
from cidoc.llm.pricing import Price

MIN_SUPPORT = 5
"""Categories with fewer human labels are left out of macro-F1 (too few to score)."""
EVIDENCE_WINDOW = 3
CONFIDENT = 3
FAILED = "(failed)"
BLOCKED_PREFIX = "blocked"


@dataclass(frozen=True)
class EvalItem:
    job_id: int
    gold: str
    gold_confidence: int
    gold_evidence: list[int]
    result: AnalysisResult
    error: str | None = None
    """The API failed after retries ("ErrorType: message"): not scored."""
    gold_root_cause: str | None = None
    """The labeler's root cause (for display; not used by any metric)."""

    @property
    def predicted(self) -> str | None:
        return self.result.diagnosis.category if self.result.diagnosis else None

    @property
    def correct(self) -> bool:
        return self.predicted == self.gold


@dataclass(frozen=True)
class CategoryStats:
    support: int
    """Human labels with this category."""
    predicted: int
    """Predictions of this category."""
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class Metrics:
    n: int
    correct: int
    accuracy: float
    macro_f1: float | None
    """Over the categories in macro_f1_categories; None if there are none."""
    macro_f1_categories: list[str]
    excluded_categories: dict[str, int]
    """Categories left out of macro-F1, with their (too small) support."""
    per_category: dict[str, CategoryStats]
    confusion: dict[str, dict[str, int]]
    """confusion[gold][predicted] = count; predicted may be FAILED."""
    confident_n: int
    confident_accuracy: float | None
    """Accuracy on labels with confidence 3."""
    evidence_n: int
    """Items whose label cites evidence lines."""
    evidence_hits: int
    evidence_hit_rate: float | None
    mean_cost: float | None
    """USD per item, from each result's original calls; None without a price."""
    latency_p50: float | None
    latency_p95: float | None
    """Seconds per item, over items that made API calls."""
    failed: int
    """Scored items whose answer was unusable after one retry; they count as wrong."""
    blocked: int
    errored: int
    """Items left out because the API failed after retries."""
    mean_model_calls: float | None = None
    """Model calls per scored item (from each result's original calls); None if none."""
    mean_input_tokens: float | None = None
    mean_output_tokens: float | None = None
    agent_items: int = 0
    """Scored items with an agent trajectory; the agent metrics below cover these."""
    mean_tool_calls: float | None = None
    """None for runs without agent items (Phase 1, the baseline)."""
    used_tools_rate: float | None = None
    """Share of agent items that made at least one tool call."""
    tool_usage: dict[str, int] = field(default_factory=dict)
    """Tool name -> calls, over all agent items, most used first."""
    hit_max_steps: int | None = None
    """Agent items that used every allowed tool call (submit was forced)."""


def _ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default); None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def evidence_hit(item: EvalItem, window: int = EVIDENCE_WINDOW) -> bool:
    """Any predicted line within `window` lines of any human evidence line."""
    predicted = item.result.diagnosis.evidence_lines if item.result.diagnosis else []
    return any(abs(p - g) <= window for p in predicted for g in item.gold_evidence)


def category_stats(items: Sequence[EvalItem]) -> dict[str, CategoryStats]:
    """Per category in taxonomy order, for every category labeled or predicted."""
    support = Counter(item.gold for item in items)
    predicted = Counter(item.predicted for item in items if item.predicted is not None)
    hits = Counter(item.gold for item in items if item.correct)
    stats: dict[str, CategoryStats] = {}
    for category in CATEGORIES:
        if not support[category] and not predicted[category]:
            continue
        precision = _ratio(hits[category], predicted[category])
        recall = _ratio(hits[category], support[category])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stats[category] = CategoryStats(
            support[category], predicted[category], precision, recall, f1
        )
    return stats


def confusion_matrix(items: Sequence[EvalItem]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for item in items:
        row = matrix.setdefault(item.gold, {})
        column = item.predicted or FAILED
        row[column] = row.get(column, 0) + 1
    return matrix


def compute_metrics(
    all_items: Sequence[EvalItem],
    price: Price | None,
    min_support: int = MIN_SUPPORT,
    evidence_window: int = EVIDENCE_WINDOW,
) -> Metrics:
    """Metrics over the scored items; errored items are only counted."""
    items = [item for item in all_items if item.error is None]
    per_category = category_stats(items)
    scored = [c for c, s in per_category.items() if s.support >= min_support]
    support = Counter(item.gold for item in items)
    excluded = {c: support[c] for c in CATEGORIES if support[c] < min_support}

    confident = [item for item in items if item.gold_confidence == CONFIDENT]
    with_evidence = [item for item in items if item.gold_evidence]
    hits = sum(evidence_hit(item, evidence_window) for item in with_evidence)

    paid = [item for item in items if item.result.calls]
    costs = [summarize(item.result.calls, price).cost for item in items]
    if paid and price is None:
        mean_cost: float | None = None
    else:
        mean_cost = _ratio(sum(c or 0.0 for c in costs), len(items))
    latencies = [summarize(item.result.calls, price).latency_s for item in paid]

    failures = [item.result.failure for item in items if item.result.failure is not None]
    correct = sum(item.correct for item in items)
    usage = _usage_means(items)
    agent = _agent_metrics(items)
    return Metrics(
        n=len(items),
        correct=correct,
        accuracy=_ratio(correct, len(items)),
        macro_f1=(sum(per_category[c].f1 for c in scored) / len(scored)) if scored else None,
        macro_f1_categories=scored,
        excluded_categories=excluded,
        per_category=per_category,
        confusion=confusion_matrix(items),
        confident_n=len(confident),
        confident_accuracy=(
            _ratio(sum(item.correct for item in confident), len(confident)) if confident else None
        ),
        evidence_n=len(with_evidence),
        evidence_hits=hits,
        evidence_hit_rate=_ratio(hits, len(with_evidence)) if with_evidence else None,
        mean_cost=mean_cost,
        latency_p50=percentile(latencies, 50),
        latency_p95=percentile(latencies, 95),
        failed=len(failures),
        blocked=sum(f.startswith(BLOCKED_PREFIX) for f in failures),
        errored=len(all_items) - len(items),
        mean_model_calls=usage[0],
        mean_input_tokens=usage[1],
        mean_output_tokens=usage[2],
        agent_items=agent.items,
        mean_tool_calls=agent.mean_tool_calls,
        used_tools_rate=agent.used_tools_rate,
        tool_usage=agent.tool_usage,
        hit_max_steps=agent.hit_max_steps,
    )


def _usage_means(items: Sequence[EvalItem]) -> tuple[float | None, float | None, float | None]:
    """Mean model calls, input and output tokens per item (the regex baseline has 0)."""
    if not items:
        return None, None, None
    n = len(items)
    calls = sum(len(item.result.calls) for item in items)
    tokens_in = sum(c.input_tokens for item in items for c in item.result.calls)
    tokens_out = sum(c.output_tokens for item in items for c in item.result.calls)
    return calls / n, tokens_in / n, tokens_out / n


@dataclass(frozen=True)
class AgentMetrics:
    items: int
    mean_tool_calls: float | None
    used_tools_rate: float | None
    tool_usage: dict[str, int]
    hit_max_steps: int | None


def _agent_metrics(items: Sequence[EvalItem]) -> AgentMetrics:
    trajectories = [i.result.trajectory for i in items if i.result.trajectory is not None]
    if not trajectories:
        return AgentMetrics(0, None, None, {}, None)
    usage = Counter(call.name for t in trajectories for call in t.tool_calls)
    n = len(trajectories)
    return AgentMetrics(
        items=n,
        mean_tool_calls=sum(len(t.tool_calls) for t in trajectories) / n,
        used_tools_rate=sum(t.used_tools for t in trajectories) / n,
        tool_usage=dict(usage.most_common()),
        hit_max_steps=sum(t.hit_max_steps for t in trajectories),
    )
