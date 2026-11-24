"""Metrics on a fixed fixture; every expected number is worked out by hand below."""

from dataclasses import replace

import pytest

from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.schema import Diagnosis
from cidoc.eval.metrics import (
    FAILED,
    EvalItem,
    Metrics,
    compute_metrics,
    evidence_hit,
    percentile,
)
from cidoc.labeling.labels import CATEGORIES
from cidoc.llm.base import Usage
from cidoc.llm.pricing import Price

PRICE = Price(input=1.0, output=1.0)  # 1000 in + 100 out = $0.0011 per call
CALL = Usage(1000, 100, 1.0)


def item(
    gold: str,
    predicted: str | None,
    *,
    confidence: int = 2,
    gold_evidence: list[int] | None = None,
    evidence: list[int] | None = None,
    failure: str = "invalid JSON: x",
    calls: list[Usage] | None = None,
) -> EvalItem:
    diagnosis = None
    if predicted is not None:
        diagnosis = Diagnosis(
            category=predicted,
            root_cause="cause",
            evidence_lines=evidence or [],
            suggested_fix="fix",
            confidence=0.5,
        )
    result = AnalysisResult(
        model="m",
        diagnosis=diagnosis,
        failure=None if predicted is not None else failure,
        calls=calls if calls is not None else ([CALL] if predicted is not None else [CALL, CALL]),
    )
    return EvalItem(1, gold, confidence, gold_evidence or [], result)


DEP, TEST, INFRA = "dependency_resolution", "test_failure", "infra_flaky"

# 12 items, 8 correct.
FIXTURE = [
    # dependency_resolution: 5 labels, 3 right, 1 called infra_flaky, 1 failed.
    item(DEP, DEP, confidence=3, gold_evidence=[10], evidence=[12]),  # evidence hit (2 away)
    item(DEP, DEP, gold_evidence=[10], evidence=[14]),  # evidence miss (4 away)
    item(DEP, DEP),
    item(DEP, INFRA, confidence=3),
    item(DEP, None, gold_evidence=[5]),  # failed: evidence miss
    # test_failure: 5 labels, 4 right, 1 called dependency_resolution.
    item(TEST, TEST, confidence=3, gold_evidence=[20]),  # no predicted evidence: miss
    item(TEST, TEST),
    item(TEST, TEST),
    item(TEST, TEST),
    item(TEST, DEP),
    # infra_flaky: 2 labels (below min support), 1 right, 1 blocked.
    item(INFRA, INFRA),
    item(INFRA, None, confidence=3, failure="blocked: SAFETY"),
]


@pytest.fixture
def metrics() -> Metrics:
    return compute_metrics(FIXTURE, PRICE)


def test_accuracy(metrics: Metrics) -> None:
    assert (metrics.n, metrics.correct) == (12, 8)
    assert metrics.accuracy == pytest.approx(8 / 12)


def test_per_category_precision_recall_support(metrics: Metrics) -> None:
    dep, test, infra = (metrics.per_category[c] for c in (DEP, TEST, INFRA))

    # dep: 3 right of 4 predicted (3 + the test_failure one), of 5 labeled.
    assert (dep.support, dep.predicted) == (5, 4)
    assert dep.precision == pytest.approx(0.75)
    assert dep.recall == pytest.approx(0.6)
    assert dep.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert (test.support, test.predicted, test.precision, test.recall) == (5, 4, 1.0, 0.8)
    assert test.f1 == pytest.approx(2 * 0.8 / 1.8)
    assert (infra.support, infra.predicted, infra.precision, infra.recall) == (2, 2, 0.5, 0.5)
    # Only categories that were labeled or predicted, in taxonomy order.
    assert list(metrics.per_category) == [DEP, TEST, INFRA]


def test_macro_f1_only_over_categories_with_enough_labels(metrics: Metrics) -> None:
    assert metrics.macro_f1_categories == [DEP, TEST]
    expected = (2 * 0.75 * 0.6 / 1.35 + 2 * 0.8 / 1.8) / 2
    assert metrics.macro_f1 == pytest.approx(expected)
    assert metrics.excluded_categories[INFRA] == 2
    assert metrics.excluded_categories["ci_config"] == 0
    assert set(metrics.excluded_categories) == set(CATEGORIES) - {DEP, TEST}


def test_confusion_matrix(metrics: Metrics) -> None:
    assert metrics.confusion == {
        DEP: {DEP: 3, INFRA: 1, FAILED: 1},
        TEST: {TEST: 4, DEP: 1},
        INFRA: {INFRA: 1, FAILED: 1},
    }


def test_confidence_3_subset(metrics: Metrics) -> None:
    # Right: the first dep and the first test; wrong: dep->infra, blocked infra.
    assert metrics.confident_n == 4
    assert metrics.confident_accuracy == pytest.approx(0.5)


def test_evidence_hit_rate(metrics: Metrics) -> None:
    assert (metrics.evidence_n, metrics.evidence_hits) == (4, 1)
    assert metrics.evidence_hit_rate == pytest.approx(0.25)


def test_cost_latency_and_failures(metrics: Metrics) -> None:
    # 10 answered items with 1 call each, 2 failed items with 2 calls each.
    assert metrics.mean_cost == pytest.approx(14 * 0.0011 / 12)
    assert metrics.latency_p50 == pytest.approx(1.0)
    assert metrics.latency_p95 == pytest.approx(2.0)
    assert (metrics.failed, metrics.blocked) == (2, 1)


@pytest.mark.parametrize(("predicted", "hit"), [([7], True), ([13], True), ([14], False)])
def test_evidence_window_is_plus_minus_3(predicted: list[int], hit: bool) -> None:
    assert evidence_hit(item(DEP, DEP, gold_evidence=[10], evidence=predicted)) is hit


def test_any_predicted_line_near_any_human_line_counts() -> None:
    assert evidence_hit(item(DEP, DEP, gold_evidence=[50, 100], evidence=[1, 98]))


def test_zero_cost_model_has_no_latency_and_needs_no_price() -> None:
    baseline = [item(DEP, DEP, calls=[]), item(TEST, DEP, calls=[])]

    metrics = compute_metrics(baseline, price=None)

    assert metrics.mean_cost == 0.0
    assert metrics.latency_p50 is None
    assert metrics.latency_p95 is None


def test_paid_model_without_price_has_unknown_cost() -> None:
    assert compute_metrics([item(DEP, DEP)], price=None).mean_cost is None


def test_empty_input() -> None:
    metrics = compute_metrics([], PRICE)

    assert (metrics.n, metrics.accuracy, metrics.macro_f1) == (0, 0.0, None)
    assert metrics.confident_accuracy is None
    assert metrics.evidence_hit_rate is None


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], 50, 2.5),
        ([float(v) for v in range(1, 21)], 95, 19.05),
        ([5.0], 95, 5.0),
        ([3.0, 1.0, 2.0], 50, 2.0),
    ],
)
def test_percentile(values: list[float], q: float, expected: float) -> None:
    assert percentile(values, q) == pytest.approx(expected)


def test_percentile_of_nothing() -> None:
    assert percentile([], 50) is None


def test_errored_items_are_left_out_of_every_metric_but_counted() -> None:
    errored = replace(item(DEP, None, confidence=3, gold_evidence=[5]), error="LLMError: 503")

    with_error = compute_metrics([*FIXTURE, errored], PRICE)
    without = compute_metrics(FIXTURE, PRICE)

    assert with_error.errored == 1
    assert without.errored == 0
    for field in (
        "n",
        "correct",
        "accuracy",
        "macro_f1",
        "per_category",
        "confusion",
        "confident_n",
        "evidence_n",
        "mean_cost",
        "latency_p95",
        "failed",
        "blocked",
    ):
        assert getattr(with_error, field) == getattr(without, field), field


def test_unusable_answers_still_count_as_wrong() -> None:
    # A model answer that failed validation is model behavior, unlike an API error.
    metrics = compute_metrics([item(DEP, DEP), item(DEP, None)], PRICE)

    assert (metrics.n, metrics.accuracy, metrics.failed, metrics.errored) == (2, 0.5, 1, 0)
