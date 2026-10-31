"""Pair each failed run with the next successful run of the same workflow on the same branch.

The pair's head SHAs give a ground-truth diff (failure_sha..fix_sha) for evaluating fixes.
A fix_sha equal to the failing head_sha means a re-run passed with no code change, which
is itself a strong hint that the failure was flaky.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from cidoc.github_client import GitHubClient

RUNS_PAGE_SIZE = 100
# Busy branches can have many green runs; we only need the earliest few after a failure.
MAX_SUCCESS_PAGES = 5


def _created(run: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(run["created_at"])


def find_fix_sha(failed_run: dict[str, Any], success_runs: Iterable[dict[str, Any]]) -> str | None:
    """Return the head_sha of the earliest success created strictly after `failed_run`."""
    failed_at = _created(failed_run)
    later = [
        r
        for r in success_runs
        if r["id"] != failed_run["id"]
        and r.get("workflow_id") == failed_run["workflow_id"]
        and r.get("head_branch") == failed_run["head_branch"]
        and r.get("conclusion") == "success"
        and _created(r) > failed_at
    ]
    if not later:
        return None
    sha: str = min(later, key=_created)["head_sha"]
    return sha


def list_success_runs(
    client: GitHubClient, repo: str, workflow_id: int, branch: str, since: datetime
) -> list[dict[str, Any]]:
    """Successful runs of one workflow on one branch created on or after `since`'s date."""
    runs: list[dict[str, Any]] = []
    for page in range(1, MAX_SUCCESS_PAGES + 1):
        data = client.get_json(
            f"/repos/{repo}/actions/workflows/{workflow_id}/runs",
            params={
                "branch": branch,
                "status": "success",
                "created": f">={since.date().isoformat()}",
                "per_page": RUNS_PAGE_SIZE,
                "page": page,
            },
        )
        batch = data.get("workflow_runs", [])
        runs.extend(batch)
        if len(batch) < RUNS_PAGE_SIZE:
            break
    return runs


class FixPairFinder:
    """Look up fix SHAs, fetching each (workflow, branch) success list only once."""

    def __init__(self, client: GitHubClient, repo: str, since: datetime) -> None:
        self._client = client
        self._repo = repo
        self._since = since
        self._cache: dict[tuple[int, str], list[dict[str, Any]]] = {}

    def fix_sha(self, failed_run: dict[str, Any]) -> str | None:
        branch = failed_run.get("head_branch")
        if not branch:
            return None
        key = (failed_run["workflow_id"], branch)
        if key not in self._cache:
            self._cache[key] = list_success_runs(self._client, self._repo, *key, self._since)
        return find_fix_sha(failed_run, self._cache[key])
