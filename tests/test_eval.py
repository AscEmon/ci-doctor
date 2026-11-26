"""`cidoc eval`: runner, result files, resuming. Scripted provider, no network."""

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cidoc.analyze.cache import ResponseCache
from cidoc.analyze.prompt import load_prompt
from cidoc.cli import app
from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.eval.metrics import compute_metrics
from cidoc.eval.results import RUNS_DIRNAME, load_run
from cidoc.eval.runner import load_cases, plan_run
from cidoc.labeling.labels import LABELS_FILENAME
from cidoc.llm.base import LLMError, ServerUnavailableError, Usage
from cidoc.llm.pricing import Price
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


def test_load_cases_filters_by_split_using_the_newest_label(data_dir: Path) -> None:
    setup_jobs(data_dir)  # all dev
    store_job(data_dir, 4, log_text("Building flavor four"))
    add_label(data_dir, 4, "other", split="test1")
    add_label(data_dir, 2, "dart_compile", split="test1")  # relabeled into test1

    def ids(split: str) -> list[int]:
        return [c.label.job_id for c in load_cases(data_dir, opener(data_dir), split=split).cases]

    assert ids("dev") == [1, 3]
    assert ids("test1") == [4, 2]
    assert ids("test2") == []
    assert ids("all") == [1, 3, 4, 2]


def test_eval_defaults_to_dev_and_records_the_split(data_dir: Path) -> None:
    setup_jobs(data_dir)
    store_job(data_dir, 4, log_text("Building flavor four"))
    add_label(data_dir, 4, "other", split="test1")

    result = run("--model", "regex-baseline")

    assert result.exit_code == 0, result.output
    assert "split" in result.output and "dev" in result.output
    saved = load_run(run_files(data_dir)[-1])
    assert saved.info.split == "dev"
    assert sorted(item.job_id for item in saved.items) == [1, 2, 3]


def test_eval_test_split(data_dir: Path) -> None:
    setup_jobs(data_dir)
    store_job(data_dir, 4, log_text("Building flavor four"))
    add_label(data_dir, 4, "other", split="test1")

    result = run("--model", "regex-baseline", "--split", "test1")

    assert result.exit_code == 0, result.output
    saved = load_run(run_files(data_dir)[-1])
    assert (saved.info.split, [item.job_id for item in saved.items]) == ("test1", [4])


def test_eval_rejects_unknown_split(data_dir: Path) -> None:
    assert run("--split", "train").exit_code == 2
    assert run("--split", "test").exit_code == 2  # renamed to test1


def test_eval_test2_split(data_dir: Path) -> None:
    setup_jobs(data_dir)
    store_job(data_dir, 4, log_text("Building flavor four"))
    add_label(data_dir, 4, "other", split="test2")

    result = run("--model", "regex-baseline", "--split", "test2")

    assert result.exit_code == 0, result.output
    saved = load_run(run_files(data_dir)[-1])
    assert (saved.info.split, [item.job_id for item in saved.items]) == ("test2", [4])


def test_run_file_with_legacy_test_split_reads_as_test1(data_dir: Path) -> None:
    setup_jobs(data_dir)
    run("--model", "regex-baseline")
    path = run_files(data_dir)[-1]
    data = json.loads(path.read_text())
    data["split"] = "test"
    path.write_text(json.dumps(data))

    assert load_run(path).info.split == "test1"


def test_old_run_file_without_split_reads_as_dev(data_dir: Path) -> None:
    setup_jobs(data_dir)
    run("--model", "regex-baseline")
    path = run_files(data_dir)[-1]
    data = json.loads(path.read_text())
    del data["split"]
    path.write_text(json.dumps(data))

    assert load_run(path).info.split == "dev"


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


def test_failing_item_is_recorded_and_the_run_continues(
    data_dir: Path, provider: ScriptedProvider
) -> None:
    setup_jobs(data_dir)
    error = ServerUnavailableError("Gemini API error 503 UNAVAILABLE: overloaded")
    provider.responses = [ok(VALID_DIAGNOSIS), error, ok(VALID_DIAGNOSIS)]

    first = run("--model", "fake-model")

    assert first.exit_code == 1  # finished, but not everything was scored
    assert "job 2 ! dart_compile → not scored: ServerUnavailableError" in first.output
    assert "1 items failed after retries and were not scored" in first.output
    saved = load_run(run_files(data_dir)[0])
    assert saved.info.complete
    assert saved.info.stopped is None
    assert [item.job_id for item in saved.items] == [1, 2, 3]
    assert saved.items[1].error == "ServerUnavailableError: " + str(error)
    assert saved.metrics["errored"] == 1
    assert saved.metrics["n"] == 2  # job 2 is left out of the metrics
    assert saved.metrics["accuracy"] == 0.5  # job 3: dart_compile for a pub failure
    assert len(list((data_dir / "cache" / "analyze").glob("*.json"))) == 2  # not cached


def test_errored_items_are_retried_on_the_next_run(
    data_dir: Path, provider: ScriptedProvider
) -> None:
    setup_jobs(data_dir)
    provider.responses = [ok(VALID_DIAGNOSIS), LLMError("boom"), ok(VALID_DIAGNOSIS)]
    run("--model", "fake-model")
    provider.responses = [ok(VALID_DIAGNOSIS)]

    second = run("--model", "fake-model")

    assert second.exit_code == 0, second.output
    assert len(provider.calls) == 3 + 1  # only job 2 is called again
    resumed = load_run(run_files(data_dir)[-1])
    assert [item.result.cached for item in resumed.items] == [True, False, True]
    assert resumed.metrics["errored"] == 0


def test_unexpected_errors_do_not_stop_the_run_either(
    data_dir: Path, provider: ScriptedProvider
) -> None:
    setup_jobs(data_dir)
    reset = ConnectionResetError("peer reset")
    provider.responses = [reset, ok(VALID_DIAGNOSIS), ok(VALID_DIAGNOSIS)]

    result = run("--model", "fake-model")

    assert result.exit_code == 1
    saved = load_run(run_files(data_dir)[0])
    assert saved.items[0].error == "ConnectionResetError: peer reset"
    assert [item.error for item in saved.items[1:]] == [None, None]


def test_ctrl_c_stops_the_run_and_saves_what_was_scored(
    data_dir: Path, provider: ScriptedProvider
) -> None:
    setup_jobs(data_dir)
    provider.responses = [ok(VALID_DIAGNOSIS), KeyboardInterrupt()]

    result = run("--model", "fake-model")

    assert result.exit_code == 130
    saved = load_run(run_files(data_dir)[0])
    assert not saved.info.complete
    assert saved.info.stopped == "interrupted (Ctrl-C)"
    assert [item.job_id for item in saved.items] == [1]


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


# --- dry run and cost guard ---------------------------------------------------------

PRICES = '{"fake-model": {"input": 0.1, "output": 0.4}}'


@pytest.fixture
def no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if anything tries to create a provider."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run must not create a provider")

    monkeypatch.setattr("cidoc.cli.create_provider", refuse)


def test_dry_run_estimates_without_a_key_or_any_call(
    data_dir: Path, no_provider: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIDOC_PRICES", PRICES)
    monkeypatch.setenv("CIDOC_RPM", "2")
    get_settings.cache_clear()
    setup_jobs(data_dir)

    result = run("--model", "fake-model", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run: fake-model" in result.output
    assert "No API calls were made." in result.output
    assert "duration at 2 RPM" in result.output
    assert "~1m 30s" in result.output  # 3 calls at 2 per minute
    plan = plan_run(
        load_cases(data_dir, opener(data_dir)).cases,
        load_prompt(),
        ResponseCache.in_data_dir(data_dir),
        "fake-model",
        6000,
    )
    expected = sum(call.cost(Price(input=0.1, output=0.4)) for call in plan.calls)
    assert f"${expected:.6f}" in result.output
    assert run_files(data_dir) == []
    assert not (data_dir / "cache").exists()


def test_dry_run_counts_cached_items_as_free(
    data_dir: Path, provider: ScriptedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_jobs(data_dir)
    provider.responses = [ok(VALID_DIAGNOSIS)]
    run("--model", "fake-model", "--limit", "1")

    loaded = load_cases(data_dir, opener(data_dir))
    plan = plan_run(
        loaded.cases,
        load_prompt(),
        ResponseCache.in_data_dir(data_dir),
        "fake-model",
        6000,
    )

    assert plan.cached == 1
    assert len(plan.calls) == 2
    assert all(call.input_tokens > 0 for call in plan.calls)
    refreshed = plan_run(
        loaded.cases,
        load_prompt(),
        ResponseCache.in_data_dir(data_dir),
        "fake-model",
        6000,
        refresh=True,
    )
    assert (refreshed.cached, len(refreshed.calls)) == (0, 3)


def test_dry_run_warns_when_max_cost_covers_only_part(
    data_dir: Path, no_provider: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIDOC_PRICES", PRICES)
    get_settings.cache_clear()
    setup_jobs(data_dir)
    loaded = load_cases(data_dir, opener(data_dir))
    plan = plan_run(
        loaded.cases, load_prompt(), ResponseCache.in_data_dir(data_dir), "fake-model", 6000
    )
    per_call = plan.calls[0].cost(Price(input=0.1, output=0.4))

    result = run("--model", "fake-model", "--dry-run", "--max-cost", str(per_call * 1.5))

    assert result.exit_code == 0, result.output
    assert "covers about 1 of 3 calls" in result.output


def test_baseline_dry_run(data_dir: Path, no_provider: None) -> None:
    setup_jobs(data_dir)

    result = run("--model", "regex-baseline", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "3 items, no API calls, $0" in result.output
    assert run_files(data_dir) == []


def test_max_cost_needs_a_price(data_dir: Path, provider: ScriptedProvider) -> None:
    setup_jobs(data_dir)

    result = run("--model", "fake-model", "--max-cost", "1")

    assert result.exit_code == 2
    assert "--max-cost needs a price for fake-model" in result.output
    assert provider.calls == []


def test_max_cost_stops_cleanly_saves_and_resumes(
    data_dir: Path, provider: ScriptedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CIDOC_PRICES", PRICES)
    get_settings.cache_clear()
    setup_jobs(data_dir)
    loaded = load_cases(data_dir, opener(data_dir))
    plan = plan_run(
        loaded.cases, load_prompt(), ResponseCache.in_data_dir(data_dir), "fake-model", 6000
    )
    price = Price(input=0.1, output=0.4)
    # Each call really costs what was estimated; the budget fits one call, not two.
    budget = plan.calls[0].cost(price) * 1.5
    provider.responses = [
        ok(VALID_DIAGNOSIS, usage=Usage(call.input_tokens, call.output_tokens, 0.1))
        for call in plan.calls
    ]

    first = run("--model", "fake-model", "--max-cost", str(budget))

    assert first.exit_code == 1
    assert "would exceed --max-cost" in first.output
    partial = load_run(run_files(data_dir)[0])
    assert not partial.info.complete
    assert len(partial.items) == len(provider.calls) == 1

    second = run("--model", "fake-model")  # no limit this time

    assert second.exit_code == 0, second.output
    assert len(provider.calls) == 3
    assert load_run(run_files(data_dir)[-1]).info.complete


def test_baseline_ignores_max_cost(data_dir: Path) -> None:
    setup_jobs(data_dir)

    result = run("--model", "regex-baseline", "--max-cost", "0")

    assert result.exit_code == 0, result.output


# --- --show -------------------------------------------------------------------------


def scored_run(data_dir: Path, provider: ScriptedProvider) -> Path:
    """Jobs 1 and 2 right (dart_compile), job 3 wrong (dart_compile for a pub failure)."""
    setup_jobs(data_dir)
    provider.responses = [ok(VALID_DIAGNOSIS) for _ in range(3)]
    run("--model", "fake-model")
    return run_files(data_dir)[-1]


def show(*args: str) -> Any:
    return CliRunner().invoke(app, ["eval", "--show", *args])


def test_show_prints_label_next_to_prediction(data_dir: Path, provider: ScriptedProvider) -> None:
    path = scored_run(data_dir, provider)

    result = show(str(path))

    assert result.exit_code == 0, result.output
    out = result.output
    for job_id in (1, 2, 3):
        assert f"job {job_id}  " in out
    assert "label dependency_resolution (confidence 2)  →  dart_compile (0.90)" in out
    assert "label dart_compile (confidence 3)  →  dart_compile (0.90)" in out
    assert "human:     cause" in out  # stored in the run file
    assert "from current labels.jsonl" not in out
    assert f"predicted: {VALID_DIAGNOSIS['root_cause']}" in out
    assert "evidence:  predicted [5]  human [5]" in out
    assert "Showing 3 of 3 items (1 wrong, 0 not scored)." in out


def test_show_wrong_only(data_dir: Path, provider: ScriptedProvider) -> None:
    path = scored_run(data_dir, provider)

    result = show(str(path), "--wrong")

    assert result.exit_code == 0, result.output
    assert "job 3  " in result.output
    assert "job 1  " not in result.output
    assert "job 2  " not in result.output
    assert "Showing 1 of 3 items" in result.output


def test_show_one_job(data_dir: Path, provider: ScriptedProvider) -> None:
    path = scored_run(data_dir, provider)

    result = show(str(path), "--job", "2")

    assert result.exit_code == 0, result.output
    assert "job 2  " in result.output
    assert "job 1  " not in result.output
    assert "job 3  " not in result.output


def test_show_unknown_job_exits_2(data_dir: Path, provider: ScriptedProvider) -> None:
    path = scored_run(data_dir, provider)

    result = show(str(path), "--job", "999")

    assert result.exit_code == 2
    assert "job 999 is not in" in result.output


def test_show_failed_and_errored_items(data_dir: Path, provider: ScriptedProvider) -> None:
    setup_jobs(data_dir)
    provider.responses = [ok("bad"), ok("bad"), LLMError("boom"), ok(VALID_DIAGNOSIS)]
    run("--model", "fake-model")
    path = run_files(data_dir)[-1]

    everything = show(str(path))
    wrong = show(str(path), "--wrong")

    assert "→  failed: invalid JSON" in everything.output  # job 1: counts as wrong
    assert "→  not scored: LLMError: boom" in everything.output  # job 2
    assert "job 1  " in wrong.output
    assert "job 2  " not in wrong.output  # never scored, so not "wrong"
    assert "(2 wrong, 1 not scored)" in wrong.output


def test_show_old_run_file_falls_back_to_labels(data_dir: Path, provider: ScriptedProvider) -> None:
    path = scored_run(data_dir, provider)
    data = json.loads(path.read_text(encoding="utf-8"))
    for saved in data["items"]:
        del saved["gold_root_cause"]
    path.write_text(json.dumps(data), encoding="utf-8")

    result = show(str(path), "--job", "1")

    assert result.exit_code == 0, result.output
    assert "human:     cause  (from current labels.jsonl)" in result.output


@pytest.mark.parametrize(
    "args",
    [["--wrong"], ["--job", "1"], ["--show", "a.json", "--compare", "a.json", "b.json"]],
)
def test_show_option_misuse_exits_2(data_dir: Path, args: list[str]) -> None:
    result = run(*args)

    assert result.exit_code == 2
    assert "--show" in result.output


def test_show_bad_file_exits_2(data_dir: Path) -> None:
    result = show("missing.json")

    assert result.exit_code == 2
    assert "cannot read run file" in result.output


def test_run_path_never_returns_an_existing_file(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from cidoc.eval.results import run_path

    now = datetime(2026, 11, 26, 8, 0, 0, tzinfo=UTC)
    first = run_path(tmp_path, "m", now)
    first.parent.mkdir(parents=True)
    first.write_text("{}")
    second = run_path(tmp_path, "m", now)
    second.write_text("{}")

    assert first.name == "20261126T080000Z_m.json"
    assert second.name == "20261126T080000Z_m_2.json"
    assert run_path(tmp_path, "m", now).name == "20261126T080000Z_m_3.json"
