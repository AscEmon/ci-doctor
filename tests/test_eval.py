"""`cidoc eval`: runner, result files, resuming. Scripted provider, no network."""

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cidoc.analyze.prompt import load_prompt
from cidoc.cli import app
from cidoc.db import DB_FILENAME, connect
from cidoc.eval.metrics import compute_metrics
from cidoc.eval.results import RUNS_DIRNAME, load_run
from cidoc.eval.runner import load_cases
from cidoc.labeling.labels import LABELS_FILENAME
from cidoc.llm.base import LLMError
from tests.datadir import add_label, log_text, store_job
from tests.fakes import VALID_DIAGNOSIS, ScriptedProvider, ok

PUB_FAILURE = (
    "Because app depends on intl ^9.0.0 which doesn't match any versions, version solving failed."
)


def run(*args: str) -> Any:
    return CliRunner().invoke(app, ["eval", *args])


def run_files(data_dir: Path) -> list[Path]:
    return sorted((data_dir / RUNS_DIRNAME).glob("*.json"))


def setup_jobs(data_dir: Path) -> None:
    """Three labeled jobs: two dart_compile, one dependency_resolution.

    Logs differ: the cache is keyed by content, so identical logs share one result.
    """
    store_job(data_dir, 1, log_text("Building flavor one"))
    store_job(data_dir, 2, log_text("Building flavor two"))
    store_job(data_dir, 3, log_text(PUB_FAILURE))
    add_label(data_dir, 1, "dart_compile", confidence=3)
    add_label(data_dir, 2, "dart_compile")
    add_label(data_dir, 3, "dependency_resolution")


def opener(data_dir: Path) -> Callable[[], sqlite3.Connection]:
    return lambda: connect(data_dir / DB_FILENAME)


# --- loading labels -----------------------------------------------------------------


def test_load_cases_keeps_the_newest_label_per_job_in_file_order(data_dir: Path) -> None:
    setup_jobs(data_dir)
    add_label(data_dir, 1, "native_compile")  # relabeled later

    loaded = load_cases(data_dir, opener(data_dir))

    assert [(c.label.job_id, c.label.category) for c in loaded.cases] == [
        (2, "dart_compile"),
        (3, "dependency_resolution"),
        (1, "native_compile"),
    ]


def test_load_cases_skips_non_taxonomy_records_and_reports_missing_logs(
    data_dir: Path,
) -> None:
    setup_jobs(data_dir)
    with (data_dir / LABELS_FILENAME).open("a", encoding="utf-8") as fh:
        record = json.loads((data_dir / LABELS_FILENAME).read_text().splitlines()[0])
        fh.write(json.dumps({**record, "job_id": 9, "category": "skip"}) + "\n")
    add_label(data_dir, 8, "other")  # labeled, but its log is gone

    loaded = load_cases(data_dir, opener(data_dir))

    assert [c.label.job_id for c in loaded.cases] == [1, 2, 3]
    assert loaded.missing == [8]
    assert loaded.labels_total == 4


def test_limit_takes_the_first_labels(data_dir: Path) -> None:
    setup_jobs(data_dir)

    assert [c.label.job_id for c in load_cases(data_dir, opener(data_dir), 2).cases] == [1, 2]


# --- regex baseline -----------------------------------------------------------------


def test_baseline_run_needs_no_key_and_writes_a_result_file(data_dir: Path) -> None:
    setup_jobs(data_dir)

    result = run("--model", "regex-baseline")

    assert result.exit_code == 0, result.output
    [path] = run_files(data_dir)
    assert path.name.endswith("_regex-baseline.json")
    saved = load_run(path)
    assert saved.info.model == "regex-baseline"
    assert saved.info.prompt_version is None
    assert saved.info.guideline_hash == load_prompt().guideline_hash
    assert saved.info.complete
    assert [item.job_id for item in saved.items] == [1, 2, 3]
    # The pub failure matches a hint rule; the Dart error does not.
    assert [item.predicted for item in saved.items] == ["other", "other", "dependency_resolution"]
    assert saved.metrics["correct"] == 1
    assert "eval: regex-baseline" in result.output
    assert "accuracy" in result.output


# --- LLM ----------------------------------------------------------------------------


def test_llm_run_records_prompt_version_and_scores(
    data_dir: Path, provider: ScriptedProvider
) -> None:
    setup_jobs(data_dir)
    provider.responses = [
        ok(VALID_DIAGNOSIS),
        ok(VALID_DIAGNOSIS),
        ok({**VALID_DIAGNOSIS, "category": "dependency_resolution"}),
    ]

    result = run("--model", "fake-model")

    assert result.exit_code == 0, result.output
    saved = load_run(run_files(data_dir)[0])
    prompt = load_prompt()
    assert (saved.info.model, saved.info.prompt_version) == ("fake-model", prompt.version)
    assert saved.info.guideline_hash == prompt.guideline_hash
    assert saved.info.token_budget == 6000
    assert saved.metrics["accuracy"] == 1.0
    assert saved.metrics["evidence_hit_rate"] == 1.0
    assert len(provider.calls) == 3


def test_saved_metrics_can_be_recomputed_from_items(
    data_dir: Path, provider: ScriptedProvider
) -> None:
    setup_jobs(data_dir)
    provider.responses = [ok(VALID_DIAGNOSIS), ok("bad"), ok("bad"), ok(VALID_DIAGNOSIS)]

    run("--model", "fake-model")

    saved = load_run(run_files(data_dir)[0])
    recomputed = compute_metrics(saved.items, price=None)
    for key in ("n", "correct", "accuracy", "failed", "evidence_hits", "confusion"):
        assert saved.metrics[key] == getattr(recomputed, key)
    assert saved.metrics["failed"] == 1


def test_rerun_is_resumed_from_the_cache(data_dir: Path, provider: ScriptedProvider) -> None:
    setup_jobs(data_dir)
    provider.responses = [ok(VALID_DIAGNOSIS) for _ in range(3)]
    run("--model", "fake-model")

    second = run("--model", "fake-model")

    assert second.exit_code == 0, second.output
    assert len(provider.calls) == 3  # none for the second run
    assert second.output.count("(cached)") == 3
    assert all(item.result.cached for item in load_run(run_files(data_dir)[-1]).items)


def test_api_error_stops_cleanly_saves_partial_and_resumes(
    data_dir: Path, provider: ScriptedProvider
) -> None:
    setup_jobs(data_dir)
    provider.responses = [ok(VALID_DIAGNOSIS), LLMError("Gemini API error 429 quota")]

    first = run("--model", "fake-model")

    assert first.exit_code == 1
    assert "Stopped after 1 of 3" in first.output
    partial = load_run(run_files(data_dir)[0])
    assert not partial.info.complete
    assert partial.info.stopped is not None
    assert "job 2" in partial.info.stopped
    assert [item.job_id for item in partial.items] == [1]

    provider.responses = [ok(VALID_DIAGNOSIS), ok(VALID_DIAGNOSIS)]
    second = run("--model", "fake-model")

    assert second.exit_code == 0, second.output
    assert len(provider.calls) == 2 + 2  # job 1 came from the cache
    resumed = load_run(run_files(data_dir)[-1])
    assert resumed.info.complete
    assert [item.result.cached for item in resumed.items] == [True, False, False]


def test_missing_key_for_an_llm_model_exits_2(data_dir: Path) -> None:
    setup_jobs(data_dir)

    result = run("--model", "gemini-anything")

    assert result.exit_code == 2
    assert "GEMINI_API_KEY is not set" in result.output
    assert run_files(data_dir) == []


def test_no_labels_exits_1(data_dir: Path) -> None:
    result = run("--model", "regex-baseline")

    assert result.exit_code == 1
    assert "No labeled jobs" in result.output


@pytest.mark.parametrize("model", ["gemini/3.5:flash lite"])
def test_model_name_is_made_safe_for_the_file_name(
    data_dir: Path, provider: ScriptedProvider, model: str
) -> None:
    setup_jobs(data_dir)
    provider.responses = [ok(VALID_DIAGNOSIS) for _ in range(3)]

    run("--model", model, "--limit", "1")

    [path] = run_files(data_dir)
    assert path.parent == data_dir / RUNS_DIRNAME
    assert "/" not in path.name.split("_", 1)[1]
