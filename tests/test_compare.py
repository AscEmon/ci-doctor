import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.schema import Diagnosis
from cidoc.cli import app
from cidoc.eval.compare import LabelChange, compare_runs
from cidoc.eval.metrics import EvalItem, compute_metrics
from cidoc.eval.results import RunFile, RunInfo, load_run, write_run
from cidoc.labeling.labels import Label
from cidoc.llm.base import Usage
from cidoc.llm.pricing import Price
from tests.datadir import add_label

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


def labels(**categories: str) -> dict[int, Label]:
    """Current labels, e.g. labels(j1=DEP, j2=TEST)."""
    return {
        int(key[1:]): Label(
            job_id=int(key[1:]),
            repo="o/a",
            run_id=int(key[1:]),
            category=category,
            root_cause="cause",
            evidence_lines=[1],
            confidence=2,
            log_path="x",
            labeled_at="2026-11-24T00:00:00+00:00",
        )
        for key, category in categories.items()
    }


CURRENT = labels(j1=DEP, j2=TEST, j3=INFRA, j4=DEP)


def prices(model: str) -> Price | None:
    return PRICE


def delta(comparison: Any, key: str) -> Any:
    return next(d for d in comparison.deltas if d.key == key)


def test_metric_deltas_and_direction() -> None:
    comparison = compare_runs(A, B, prices, CURRENT)

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
    comparison = compare_runs(A, B, prices, CURRENT)

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

    comparison = compare_runs(A, partial_b, prices, CURRENT)

    assert comparison.recomputed
    assert comparison.shared == 2
    assert (comparison.only_a, comparison.only_b) == ([3], [4])
    accuracy = delta(comparison, "accuracy")
    assert (accuracy.a, accuracy.b) == (0.5, 1.0)  # on jobs 1 and 2 only
    assert any("Run B is partial: job 3: budget" in note for note in comparison.notes)


def test_note_for_changed_guideline() -> None:
    other_guideline = run_file("model-b", B.items, guideline_hash="g2")

    notes = compare_runs(A, other_guideline, prices, CURRENT).notes

    assert any("different labeling guidelines" in note for note in notes)


# Job 2's label was test_failure when A ran, infra_flaky when B ran, and is
# dependency_resolution now. Both models predicted the same thing on job 2.
OLD_RUN = run_file(
    "model-a",
    [item(1, DEP, DEP), item(2, TEST, DEP), item(3, INFRA, INFRA)],
    created_at="2026-11-20T08:00:00+00:00",
)
NEW_RUN = run_file(
    "model-b",
    [item(1, DEP, DEP), item(2, INFRA, DEP), item(3, INFRA, INFRA)],
    created_at="2026-11-24T08:00:00+00:00",
)
RELABELED = labels(j1=DEP, j2=DEP, j3=INFRA)


def test_both_runs_are_rescored_against_the_current_labels() -> None:
    # Against their stored labels both got job 2 wrong; against the current one both are
    # right, and identical predictions give identical metrics.
    comparison = compare_runs(OLD_RUN, NEW_RUN, prices, RELABELED)

    accuracy = delta(comparison, "accuracy")
    assert (accuracy.a, accuracy.b) == (1.0, 1.0)
    assert comparison.both_right == 3
    assert comparison.disagreements == []


def test_label_changes_between_runs_are_listed_old_to_new() -> None:
    expected = [LabelChange(2, TEST, INFRA)]

    assert compare_runs(OLD_RUN, NEW_RUN, prices, RELABELED).label_changes == expected
    # Old → new follows the runs' timestamps, not the A/B order.
    assert compare_runs(NEW_RUN, OLD_RUN, prices, RELABELED).label_changes == expected


def test_disagreements_show_the_current_label() -> None:
    b = run_file("model-b", [item(1, DEP, TEST), item(2, INFRA, TEST), item(3, INFRA, INFRA)])

    comparison = compare_runs(OLD_RUN, b, prices, RELABELED)

    assert [(d.job_id, d.gold, d.a_correct) for d in comparison.disagreements] == [
        (1, DEP, True),
        (2, DEP, True),
    ]


def test_jobs_no_longer_labeled_are_left_out() -> None:
    comparison = compare_runs(A, B, prices, labels(j1=DEP, j2=TEST))

    assert comparison.shared == 2
    assert comparison.unlabeled == [3]
    assert any("no label in labels.jsonl now: 3" in note for note in comparison.notes)


def test_identical_runs_agree() -> None:
    comparison = compare_runs(A, A, prices, CURRENT)

    assert comparison.disagreements == []
    assert all(d.delta in (0, None) for d in comparison.deltas)


# --- CLI ----------------------------------------------------------------------------


def write(path: Path, run: RunFile) -> Path:
    write_run(path, run.info, run.items, compute_metrics(run.items, PRICE))
    return path


def label_jobs(data_dir: Path) -> None:
    for job_id, category in ((1, DEP), (2, TEST), (3, INFRA)):
        add_label(data_dir, job_id, category)


def test_compare_command(data_dir: Path, tmp_path: Path) -> None:
    label_jobs(data_dir)
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
    label_jobs(data_dir)
    a = write(tmp_path / "a.json", A)

    result = CliRunner().invoke(app, ["eval", "--compare", str(a), str(a)])

    assert result.exit_code == 0, result.output
    assert "agree on every shared item" in result.output


def test_compare_command_warns_about_changed_labels(data_dir: Path, tmp_path: Path) -> None:
    add_label(data_dir, 1, DEP)
    add_label(data_dir, 2, TEST)
    add_label(data_dir, 2, DEP)  # relabeled after both runs
    add_label(data_dir, 3, INFRA)
    old = write(tmp_path / "old.json", OLD_RUN)
    new = write(tmp_path / "new.json", NEW_RUN)

    result = CliRunner().invoke(app, ["eval", "--compare", str(old), str(new)])

    assert result.exit_code == 0, result.output
    assert "current labels.jsonl" in result.output
    assert "labels changed between the runs for 1 jobs" in result.output
    assert "job 2: test_failure → infra_flaky" in result.output
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


def test_errored_items_are_not_compared() -> None:
    errored = replace(item(2, TEST, None), error="ServerUnavailableError: 503")
    b_with_error = run_file("model-b", [item(1, DEP, DEP), errored, item(3, INFRA, TEST)])

    comparison = compare_runs(A, b_with_error, prices, CURRENT)

    assert comparison.shared == 2
    assert comparison.recomputed
    assert comparison.only_a == [2]
    assert [d.job_id for d in comparison.disagreements] == [3]


def test_run_files_without_an_error_field_still_load(tmp_path: Path) -> None:
    path = write(tmp_path / "old.json", A)
    data = json.loads(path.read_text(encoding="utf-8"))
    for saved in data["items"]:
        del saved["error"]
    path.write_text(json.dumps(data), encoding="utf-8")

    assert all(i.error is None for i in load_run(path).items)


def test_error_round_trips_through_run_files(tmp_path: Path) -> None:
    errored = replace(item(2, TEST, None), error="LLMError: boom")
    path = write(tmp_path / "e.json", run_file("m", [item(1, DEP, DEP), errored]))

    assert [i.error for i in load_run(path).items] == [None, "LLMError: boom"]
