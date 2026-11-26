"""Agent run-context tools: sibling_jobs, previous_run, commit_changes. Fake GitHub."""

from typing import Any

import pytest

from cidoc.agent.tools import PATCH_MAX_LINES, ToolBox, ToolEnv, versions_in_name
from tests.agent_fakes import FIX_SHA, JOB_ID, REPO, RUN_ID, SHA, FakeGitHub, context

JOBS_URL = f"/repos/{REPO}/actions/runs/{RUN_ID}/jobs"
RUN_URL = f"/repos/{REPO}/actions/runs/{RUN_ID}"
WORKFLOW_RUNS_URL = f"/repos/{REPO}/actions/workflows/555/runs"
COMMIT_URL = f"/repos/{REPO}/commits/{SHA}"
CREATED = "2026-10-20T10:00:00Z"


def call(github: FakeGitHub, name: str, **ctx: Any) -> Any:
    return ToolBox(ToolEnv(context(**ctx), github), cap_tokens=100_000).call(name, {})


def job(
    job_id: int, name: str, conclusion: str, labels: list[str], failed: tuple[str, ...] = ()
) -> Any:
    steps = [{"name": "Checkout", "conclusion": "success"}]
    steps += [{"name": step, "conclusion": "failure"} for step in failed]
    return {"id": job_id, "name": name, "conclusion": conclusion, "labels": labels, "steps": steps}


# --- sibling_jobs -----------------------------------------------------------------------


def test_sibling_jobs_lists_the_other_jobs_without_logs() -> None:
    jobs = {
        "total_count": 3,
        "jobs": [
            job(JOB_ID, "build (macos-14, 3.24.0)", "failure", ["macos-14"], ("Build iOS",)),
            job(8, "build (ubuntu-latest, 3.24.0)", "success", ["ubuntu-latest"]),
            job(9, "test Flutter 3.27.1 java 17", "failure", ["windows-2022"], ("Run tests",)),
        ],
    }
    github = FakeGitHub({JOBS_URL: jobs})

    out = call(github, "sibling_jobs")

    assert github.calls == [(JOBS_URL, {"filter": "latest", "per_page": 100})]
    assert out.text.splitlines() == [
        "this job: build (macos-14, 3.24.0) | failure | macOS | failed steps: Build iOS"
        " | versions: matrix (macos-14, 3.24.0)",
        "2 other jobs in this run (1 failure, 1 success):",
        "- build (ubuntu-latest, 3.24.0) | success | Linux"
        " | versions: matrix (ubuntu-latest, 3.24.0)",
        "- test Flutter 3.27.1 java 17 | failure | Windows | failed steps: Run tests"
        " | versions: flutter 3.27.1, java 17",
    ]


def test_sibling_jobs_says_when_the_run_has_more_jobs_than_listed() -> None:
    github = FakeGitHub({JOBS_URL: {"total_count": 150, "jobs": [job(8, "a", "success", [])]}})

    assert call(github, "sibling_jobs").text.endswith("... (149 more jobs not listed)")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("build", []),
        ("Xcode 15.4 build", ["xcode 15.4"]),
        ("jdk-21 / flutter:3.22", ["jdk 21", "flutter 3.22"]),
        ("ci (stable)", ["matrix (stable)"]),
    ],
)
def test_versions_in_name(name: str, expected: list[str]) -> None:
    assert versions_in_name(name) == expected


# --- previous_run ---------------------------------------------------------------------


def previous_runs(*runs: dict[str, Any]) -> FakeGitHub:
    return FakeGitHub(
        {
            RUN_URL: {"id": RUN_ID, "workflow_id": 555, "created_at": CREATED},
            WORKFLOW_RUNS_URL: {"workflow_runs": list(runs)},
        }
    )


def test_previous_run_returns_the_latest_earlier_run_on_the_branch() -> None:
    github = previous_runs(
        {
            "id": 60,
            "conclusion": "success",
            "head_sha": "b" * 40,
            "created_at": "2026-10-19T09:00:00Z",
            "event": "push",
        },
        {
            "id": 50,
            "conclusion": "failure",
            "head_sha": "c" * 40,
            "created_at": "2026-10-18T09:00:00Z",
            "event": "push",
        },
    )

    out = call(github, "previous_run")

    assert github.calls[1] == (
        WORKFLOW_RUNS_URL,
        {"branch": "main", "status": "completed", "created": f"<{CREATED}", "per_page": 30},
    )
    assert out.text.splitlines() == [
        "latest earlier run of this workflow on branch main:",
        "conclusion: success",
        f"head_sha: {'b' * 12} (a different commit)",
        f"created_at: 2026-10-19T09:00:00Z (failing run: {CREATED})",
        "event: push",
    ]


def test_previous_run_never_returns_this_run_or_a_later_one() -> None:
    # Even if the API ignored the created filter: a later run could hold the fix.
    github = previous_runs(
        {
            "id": 80,
            "conclusion": "success",
            "head_sha": FIX_SHA,
            "created_at": "2026-10-21T00:00:00Z",
        },
        {"id": RUN_ID, "conclusion": "failure", "head_sha": SHA, "created_at": CREATED},
    )

    out = call(github, "previous_run")

    assert out.text == "no earlier completed run of this workflow on branch main"
    assert FIX_SHA[:12] not in out.text


def test_previous_run_on_the_same_commit_says_so() -> None:
    github = previous_runs(
        {"id": 60, "conclusion": "success", "head_sha": SHA, "created_at": "2026-10-19T00:00:00Z"},
    )

    assert "(same commit as the failing run)" in call(github, "previous_run").text


def test_previous_run_needs_a_branch() -> None:
    out = call(previous_runs(), "previous_run", head_branch=None)

    assert out.text == "error: the branch of this run is unknown"


# --- commit_changes -------------------------------------------------------------------


def changed(name: str, additions: int, deletions: int, patch: str | None) -> dict[str, Any]:
    return {
        "filename": name,
        "status": "modified",
        "additions": additions,
        "deletions": deletions,
        "patch": patch,
    }


def test_commit_changes_lists_files_and_a_patch_excerpt() -> None:
    commit = {
        "sha": SHA,
        "commit": {"message": "Bump Flutter to 3.27\n\nLonger body."},
        "parents": [{"sha": "d" * 40}],
        "files": [
            changed("pubspec.yaml", 1, 1, "@@ -1 +1 @@\n-sdk: 3.24.0\n+sdk: 3.27.0"),
            changed("assets/logo.png", 0, 0, None),
        ],
    }
    github = FakeGitHub({COMMIT_URL: commit})

    out = call(github, "commit_changes")

    assert out.text.splitlines() == [
        f"commit {SHA[:12]}: Bump Flutter to 3.27",
        "changes vs its parent: 2 files, +1 -1",
        "modified  pubspec.yaml (+1 -1)",
        "modified  assets/logo.png (+0 -0)",
        "patch excerpt (at most 200 lines):",
        "--- pubspec.yaml",
        "@@ -1 +1 @@",
        "-sdk: 3.24.0",
        "+sdk: 3.27.0",
    ]


def test_commit_changes_truncates_the_patch_to_200_lines_in_total() -> None:
    big = "\n".join(f"+line {n}" for n in range(150))
    commit = {
        "commit": {"message": "x"},
        "parents": [{"sha": "d" * 40}],
        "files": [changed("a.dart", 150, 0, big), changed("b.dart", 150, 0, big)],
    }

    out = call(FakeGitHub({COMMIT_URL: commit}), "commit_changes")

    patch = out.text.split("patch excerpt (at most 200 lines):\n", 1)[1].splitlines()
    assert len(patch) == PATCH_MAX_LINES + 1
    assert patch[-1] == "... (patch truncated: 102 more lines)"  # headers count


def test_commit_changes_diffs_a_merge_against_its_first_parent() -> None:
    first, second = "d" * 40, "e" * 40
    compare_url = f"/repos/{REPO}/compare/{first}...{SHA}"
    github = FakeGitHub(
        {
            COMMIT_URL: {
                "commit": {"message": "Merge"},
                "parents": [{"sha": first}, {"sha": second}],
                "files": [changed("wrong.txt", 1, 0, None)],
            },
            compare_url: {"files": [changed("right.txt", 2, 0, None)]},
        }
    )

    out = call(github, "commit_changes")

    assert github.calls[1][0] == compare_url
    assert f"merge commit; changes vs its first parent {first[:12]}" in out.text
    assert "right.txt" in out.text
    assert "wrong.txt" not in out.text


def test_run_tools_report_github_errors_as_text() -> None:
    out = call(FakeGitHub(), "sibling_jobs")

    assert not out.ok
    assert out.text == "error: not found (HTTP 404) at this commit"
