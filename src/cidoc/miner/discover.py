"""Find popular Dart repos whose GitHub Actions workflows build Flutter."""

import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import quote

import httpx

from cidoc.github_client import GitHubClient

MIN_STARS = 200
PUSHED_WITHIN = timedelta(days=182)  # ~6 months
SEARCH_PAGE_SIZE = 100
# The search API never returns more than 1000 results for one query.
SEARCH_RESULT_CAP = 1000
WORKFLOW_DIR = ".github/workflows/"
RAW_CONTENT_ACCEPT = "application/vnd.github.raw+json"
# Repos with Actions disabled, empty, or blocked answer with these instead of data.
SKIPPABLE_STATUSES = frozenset({404, 409, 451})


@dataclass(frozen=True)
class RepoInfo:
    full_name: str
    stars: int
    default_branch: str


def build_search_query(now: datetime) -> str:
    pushed_since = (now - PUSHED_WITHIN).date().isoformat()
    return f"language:Dart stars:>={MIN_STARS} pushed:>={pushed_since} archived:false fork:false"


def search_repos(client: GitHubClient, query: str, limit: int) -> Iterator[RepoInfo]:
    """Yield up to `limit` search hits, most-starred first."""
    limit = min(limit, SEARCH_RESULT_CAP)
    seen = 0
    page = 1
    while seen < limit:
        data = client.get_json(
            "/search/repositories",
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": SEARCH_PAGE_SIZE,
                "page": page,
            },
        )
        items = data.get("items", [])
        for item in items:
            # The query already filters these; double-check in case search is stale.
            if item.get("archived") or item.get("fork"):
                continue
            yield RepoInfo(item["full_name"], item["stargazers_count"], item["default_branch"])
            seen += 1
            if seen >= limit:
                return
        if len(items) < SEARCH_PAGE_SIZE:
            return
        page += 1


def list_workflow_paths(client: GitHubClient, full_name: str) -> list[str]:
    """Return workflow file paths, ignoring GitHub-managed dynamic workflows."""
    data = client.get_json(f"/repos/{full_name}/actions/workflows", params={"per_page": 100})
    paths = [wf["path"] for wf in data.get("workflows", [])]
    return [p for p in paths if p.startswith(WORKFLOW_DIR)]


def fetch_file_text(client: GitHubClient, full_name: str, path: str) -> str:
    response = client.request(
        "GET",
        f"/repos/{full_name}/contents/{quote(path)}",
        headers={"Accept": RAW_CONTENT_ACCEPT},
    )
    return response.text


def uses_flutter_workflow(client: GitHubClient, full_name: str) -> bool:
    """True if any workflow file in the repo mentions flutter."""
    try:
        for path in list_workflow_paths(client, full_name):
            if "flutter" in fetch_file_text(client, full_name, path).lower():
                return True
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in SKIPPABLE_STATUSES:
            return False
        raise
    return False


def known_repos(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT full_name FROM repos")}


def save_repo(conn: sqlite3.Connection, repo: RepoInfo, discovered_at: datetime) -> None:
    """Insert a repo, or refresh stars/branch if it is already stored."""
    with conn:
        conn.execute(
            "INSERT INTO repos (full_name, stars, default_branch, discovered_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(full_name) DO UPDATE SET"
            " stars = excluded.stars, default_branch = excluded.default_branch",
            (repo.full_name, repo.stars, repo.default_branch, discovered_at.isoformat()),
        )


def discover(
    client: GitHubClient,
    conn: sqlite3.Connection,
    now: datetime,
    limit: int,
    on_checked: Callable[[RepoInfo, bool], None] | None = None,
) -> int:
    """Search, keep repos with a Flutter workflow, and store them. Returns how many were added.

    Repos already in the table are skipped, so an interrupted run can be restarted cheaply.
    """
    already = known_repos(conn)
    added = 0
    for repo in search_repos(client, build_search_query(now), limit):
        if repo.full_name in already:
            continue
        keep = uses_flutter_workflow(client, repo.full_name)
        if keep:
            save_repo(conn, repo, now)
            added += 1
        if on_checked:
            on_checked(repo, keep)
    return added
