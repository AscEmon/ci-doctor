from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.schema import Diagnosis
from cidoc.cli import app
from cidoc.eval.compare import compare_runs
from cidoc.eval.metrics import EvalItem, compute_metrics
from cidoc.eval.results import RunFile, RunInfo, load_run, write_run
from cidoc.llm.base import Usage
from cidoc.llm.pricing import Price

DEP, TEST, INFRA = "dependency_resolution", "test_failure", "infra_flaky"
PRICE = Price(input=1.0, output=1.0)


def item(job_id: int, gold: str, predicted: str | None, cost_tokens: int = 1000) -> EvalItem:
    diagnosis = (
        Diagnosis(
            category=predicted,
            root_cause="cause",
            evidence_lines=[1],
            suggested_fix="fix",
            confidence=0.5,
        )
        if predicted
        else None
    )
    result = AnalysisResult(
        model="m",
        diagnosis=diagnosis,
        failure=None if predicted else "invalid JSON: x",
        calls=[Usage(cost_tokens, 0, 1.0)],
    )
    return EvalItem(job_id, gold, 2, [1], result)


def run_file(model: str, items: list[EvalItem], **info: Any) -> RunFile:
    fields: dict[str, Any] = {
        "model": model,
        "created_at": "2026-11-24T08:00:00+00:00",
        "prompt_version": "p1",
        "guideline_hash": "g1",
        "token_budget": 6000,
        "labels_total": len(items),
        "complete": True,
    }
    fields.update(info)
    return RunFile(RunInfo(**fields), items, asdict(compute_metrics(items, PRICE)))


# A gets 1 of 3 right; B gets 2 of 3 right, at twice the cost.
A = run_file("model-a", [item(1, DEP, DEP), item(2, TEST, DEP), item(3, INFRA, None)])
B = run_file(
    "model-b",
    [item(1, DEP, DEP, 2000), item(2, TEST, TEST, 2000), item(3, INFRA, TEST, 2000)],
)


def prices(model: str) -> Price | None:
    return PRICE


def delta(comparison: Any, key: str) -> Any:
    return next(d for d in comparison.deltas if d.key == key)


def test_metric_deltas_and_direction() -> None:
    comparison = compare_runs(A, B, prices)

    accuracy = delta(comparison, "accuracy")
    assert accuracy.a == pytest.approx(1 / 3)
    assert accuracy.b == pytest.approx(2 / 3)
    assert accuracy.delta == pytest.approx(1 / 3)
    assert accuracy.improved is True
    cost = delta(comparison, "mean_cost")
    assert cost.delta == pytest.approx(0.002 - 0.001)  # one call per item on both sides
    assert cost.improved is False  # more expensive
    failed = delta(comparison, "failed")
    assert (failed.a, failed.b, failed.improved) == (1, 0, True)
    assert comparison.recomputed is False


def test_disagreements_and_outcome_counts() -> None:
    comparison = compare_runs(A, B, prices)

    assert [(d.job_id, d.a, d.b) for d in comparison.disagreements] == [
        (2, DEP, TEST),  # only B right: listed first
        (3, None, TEST),  # both wrong
    ]
    assert comparison.disagreements[0].b_correct
    assert not comparison.disagreements[1].a_correct
    assert (
        comparison.both_right,
        comparison.only_a_right,
        comparison.only_b_right,
        comparison.both_wrong,
    ) == (1, 0, 1, 1)


def test_different_item_sets_are_compared_on_shared_jobs() -> None:
    partial_b = run_file(
        "model-b",
        [item(1, DEP, DEP), item(2, TEST, TEST), item(4, DEP, DEP)],
        complete=False,
        stopped="job 3: budget",
    )

    comparison = compare_runs(A, partial_b, prices)

    assert comparison.recomputed
    assert comparison.shared == 2
    assert (comparison.only_a, comparison.only_b) == ([3], [4])
    accuracy = delta(comparison, "accuracy")
    assert (accuracy.a, accuracy.b) == (0.5, 1.0)  # on jobs 1 and 2 only
    assert any("Run B is partial: job 3: budget" in note for note in comparison.notes)


def test_notes_for_changed_guideline_and_labels() -> None:
    relabeled = run_file(
        "model-b",
        [item(1, DEP, DEP), item(2, INFRA, DEP), item(3, INFRA, None)],
        guideline_hash="g2",
    )

    notes = compare_runs(A, relabeled, prices).notes

    assert any("different labeling guidelines" in note for note in notes)
    assert any("labels changed" in note for note in notes)


def test_identical_runs_agree() -> None:
    comparison = compare_runs(A, A, prices)

    assert comparison.disagreements == []
    assert all(d.delta in (0, None) for d in comparison.deltas)


# --- CLI ----------------------------------------------------------------------------


def write(path: Path, run: RunFile) -> Path:
    write_run(path, run.info, run.items, compute_metrics(run.items, PRICE))
    return path


def test_compare_command(data_dir: Path, tmp_path: Path) -> None:
    a = write(tmp_path / "a.json", A)
    b = write(tmp_path / "b.json", B)

    result = CliRunner().invoke(app, ["eval", "--compare", str(a), str(b)])

    assert result.exit_code == 0, result.output
    assert "A: model-a" in result.output
    assert "B: model-b" in result.output
    assert "+33.3 pts" in result.output
    assert "2 items where A and B disagree" in result.output
    assert "only B right 1" in result.output


def test_compare_needs_no_key_or_prompt(data_dir: Path, tmp_path: Path) -> None:
    a = write(tmp_path / "a.json", A)

    result = CliRunner().invoke(app, ["eval", "--compare", str(a), str(a)])

    assert result.exit_code == 0, result.output
    assert "agree on every shared item" in result.output


def test_compare_round_trips_through_files(tmp_path: Path) -> None:
    path = write(tmp_path / "a.json", A)

    loaded = load_run(path)

    assert [replace(i.result, calls=[]) for i in loaded.items] == [
        replace(i.result, calls=[]) for i in A.items
    ]
    assert loaded.metrics["accuracy"] == A.metrics["accuracy"]


def test_compare_bad_file_exits_2(data_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(app, ["eval", "--compare", str(bad), str(tmp_path / "x.json")])

    assert result.exit_code == 2
    assert "cannot read run file" in result.output
