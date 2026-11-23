from pathlib import Path

import pytest

from cidoc.miner.rules import (
    EXCLUDE_FILENAME,
    JobFacts,
    RepoExcludes,
    event_skip_reason,
    is_fork_run,
    job_skip_reason,
    load_repo_excludes,
    parse_exclude_file,
    run_skip_reason,
    workflow_skip_reason,
)


@pytest.mark.parametrize(
    ("path", "name", "expected"),
    [
        (".github/workflows/ci.yml", "CI", None),
        (".github/workflows/build.yml", "Build & Test", None),
        ("dynamic/pages/pages-build-deployment", "pages build and deployment", "dynamic workflow"),
        (".github/workflows/deps.yml", "Dependabot auto-merge", "dependabot workflow"),
        (".github/workflows/codeql-analysis.yml", "CodeQL", "codeql workflow"),
        (".github/workflows/x.yml", "copilot-setup-steps", "copilot workflow"),
    ],
)
def test_workflow_skip_reason(path: str, name: str, expected: str | None) -> None:
    assert workflow_skip_reason(path, name) == expected


@pytest.mark.parametrize(
    ("head_repository", "expected"),
    [
        ({"full_name": "acme/app"}, False),
        ({"full_name": "Acme/App"}, False),
        ({"full_name": "someone/app"}, True),
        (None, True),
    ],
)
def test_is_fork_run(head_repository: dict[str, str] | None, expected: bool) -> None:
    assert is_fork_run({"head_repository": head_repository}, "acme/app") is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("flutter/flutter", "excluded org flutter/*"),
        ("Flutter/Engine", "excluded org flutter/*"),
        ("dart-lang/sdk", "excluded org dart-lang/*"),
        ("acme/flutter", None),
        ("flutter-community/plus_plugins", None),
    ],
)
def test_default_excludes(name: str, expected: str | None) -> None:
    assert RepoExcludes().reason(name) == expected


def test_parse_exclude_file() -> None:
    text = "# extra\nAcme/Big-Monorepo\n\nsome-org/*  # whole org\nacme/big-monorepo\n"

    owners, repos = parse_exclude_file(text)

    assert owners == {"some-org"}
    assert repos == {"acme/big-monorepo"}


@pytest.mark.parametrize("line", ["just-a-name", "*/repo", "o/r/extra", "https://github.com/o/r"])
def test_parse_exclude_file_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(ValueError, match=r"exclude_repos\.txt line 2"):
        parse_exclude_file(f"ok/repo\n{line}\n")


def test_load_repo_excludes_without_file_uses_defaults(tmp_path: Path) -> None:
    assert load_repo_excludes(tmp_path) == RepoExcludes()


def test_load_repo_excludes_adds_file_entries_to_defaults(tmp_path: Path) -> None:
    (tmp_path / EXCLUDE_FILENAME).write_text("acme/huge\nother-org/*\n", encoding="utf-8")

    excludes = load_repo_excludes(tmp_path)

    assert excludes.reason("flutter/packages") == "excluded org flutter/*"
    assert excludes.reason("ACME/huge") == "listed in exclude_repos.txt"
    assert excludes.reason("other-org/x") == "excluded org other-org/*"
    assert excludes.reason("acme/small") is None


# --- Out-of-scope jobs: issue-bot events ------------------------------------------------


@pytest.mark.parametrize("event", ["issues", "issue_comment"])
def test_issue_events_are_out_of_scope(event: str) -> None:
    assert event_skip_reason(event) == f"issue bot ({event} event)"
    assert run_skip_reason({"event": event, "path": ".github/workflows/ci.yml", "name": "CI"})


@pytest.mark.parametrize("event", ["push", "pull_request", "pull_request_target", "schedule"])
def test_build_events_are_in_scope(event: str) -> None:
    assert event_skip_reason(event) is None


# --- Out-of-scope jobs: PR hygiene and issue-bot actions -------------------------------


@pytest.mark.parametrize(
    ("step", "reason"),
    [
        ("Run amannn/action-semantic-pull-request@v5", "PR hygiene"),
        ("Run actions/labeler@v5", "PR hygiene"),
        ("Run actions/stale@v9", "PR hygiene"),
        ("Run actions/first-interaction@v1", "PR hygiene"),
        ("Run release-drafter/release-drafter@v6", "PR hygiene"),
        ("Run someone/release-drafter@main", "PR hygiene"),
        ("Run CodelyTV/pr-size-labeler@v1", "PR hygiene"),
        ("Run actions/add-to-project@v1.0.2", "issue bot"),
        (
            "##[group]Run amannn/action-semantic-pull-request@0723387faaf9b38adef4775cd",
            "PR hygiene",
        ),
    ],
)
def test_out_of_scope_action_in_failed_step(step: str, reason: str) -> None:
    by_name = job_skip_reason(JobFacts(failed_steps=(step.removeprefix("##[group]"),)))
    by_log = job_skip_reason(JobFacts(failed_steps=("Check title",), failed_step_text=step))

    assert by_name is not None and by_name.startswith(reason)
    assert by_log is not None and by_log.startswith(reason)


@pytest.mark.parametrize(
    "step",
    [
        "Run actions/checkout@v4",
        "Run subosito/flutter-action@v2",
        "Run flutter build apk  # uses actions/labeler@v5 elsewhere",
        "Run echo amannn/action-semantic-pull-request@v5",  # not a step header for it
    ],
)
def test_build_actions_are_in_scope(step: str) -> None:
    assert job_skip_reason(JobFacts(failed_steps=(step,), failed_step_text=step)) is None


@pytest.mark.parametrize(
    "name", ["Validate PR title", "Check PR Title", "pr-title", "Label PR", "labeler", "Welcome"]
)
def test_pr_hygiene_job_names(name: str) -> None:
    assert job_skip_reason(JobFacts(job_name=name)) == f"PR hygiene (job '{name}')"


@pytest.mark.parametrize("name", ["Build", "Label app strings", "Welcome screen tests", "lint"])
def test_build_job_names_are_in_scope(name: str) -> None:
    facts = JobFacts(job_name=name, failed_steps=("Run flutter test",))

    assert job_skip_reason(facts) is None


# --- Out-of-scope jobs: aggregators ------------------------------------------------------

# The failing step of a real aggregator, as GitHub logs it: needs results arrive as JSON
# in an env var (expressions are substituted before logging).
AGGREGATOR_STEP = """##[group]Run node .github/scripts/ci/check.mjs result
node .github/scripts/ci/check.mjs result
shell: /usr/bin/bash -e {0}
env:
  NEEDS: {
  "build": {
    "result": "failure",
    "outputs": {}
  }
}
##[endgroup]
Job build failed.
##[error]Process completed with exit code 1."""


def test_aggregator_reading_needs_json_is_out_of_scope() -> None:
    facts = JobFacts(
        job_name="ci",
        failed_steps=("Run node .github/scripts/ci/check.mjs result",),
        failed_step_text=AGGREGATOR_STEP,
    )

    assert job_skip_reason(facts) == "aggregator (reads needs results)"


@pytest.mark.parametrize(
    "text",
    [
        "##[group]Run echo '${{ toJSON(needs) }}'",
        'if [ "${{ needs.build.result }}" != success ]; then exit 1; fi',
        "Run re-actors/alls-green@release/v1",
    ],
)
def test_other_aggregator_traces(text: str) -> None:
    reason = job_skip_reason(JobFacts(failed_steps=("Decide",), failed_step_text=text))

    assert reason is not None and reason.startswith("aggregator")


@pytest.mark.parametrize("name", ["ci-success", "All green", "gate", "result", "Checks"])
def test_aggregator_by_name_with_one_real_step(name: str) -> None:
    facts = JobFacts(
        job_name=name,
        failed_steps=("Run exit 1",),
        step_names=(
            "Set up job",
            "Run actions/checkout@v4",
            "Run exit 1",
            "Post Run actions/checkout@v4",
            "Complete job",
        ),
    )

    assert job_skip_reason(facts) == f"aggregator (job '{name}')"


def test_check_job_that_builds_is_not_an_aggregator() -> None:
    # A job named "check" that sets up Flutter and runs the analyzer is a real check.
    facts = JobFacts(
        job_name="check",
        failed_steps=("Run flutter analyze",),
        step_names=(
            "Set up job",
            "Run actions/checkout@v4",
            "Run subosito/flutter-action@v2",
            "Run flutter pub get",
            "Run flutter analyze",
            "Complete job",
        ),
    )

    assert job_skip_reason(facts) is None


def test_aggregator_name_needs_known_steps() -> None:
    # Rows mined before step names were stored: the name alone is not enough.
    assert job_skip_reason(JobFacts(job_name="gate", failed_steps=("Run exit 1",))) is None


def test_needs_evidence_requires_a_single_failed_step() -> None:
    facts = JobFacts(failed_steps=("Build", "Test"), failed_step_text=AGGREGATOR_STEP)

    assert job_skip_reason(facts) is None
