"""Hand-picked repos to mine in addition to search results.

The seed file has one `owner/repo` per line. Blank lines and `#` comments are ignored.
Seed repos skip the Flutter-workflow check: listing one is an explicit choice.
The exclude list still wins, so a seed can't pull an excluded framework repo back in.
"""

import re
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx

from cidoc.github_client import GitHubClient
from cidoc.miner.discover import SKIPPABLE_STATUSES, RepoInfo, known_repos, save_repo
from cidoc.miner.rules import RepoExcludes

SEED_FILENAME = "seed_repos.txt"
FULL_NAME_RE = re.compile(r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")


def parse_seed_file(text: str) -> list[str]:
    """Return unique `owner/repo` names in file order. Raises ValueError on a malformed line."""
    names: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not FULL_NAME_RE.match(line):
            raise ValueError(f"line {lineno}: expected owner/repo, got {line!r}")
        if line not in names:
            names.append(line)
    return names


def fetch_repo(client: GitHubClient, full_name: str) -> RepoInfo | None:
    """Look up a repo, or return None if it is gone or inaccessible."""
    try:
        data = client.get_json(f"/repos/{full_name}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in SKIPPABLE_STATUSES:
            return None
        raise
    # Use the canonical name: GitHub redirects renamed repos.
    return RepoInfo(data["full_name"], data["stargazers_count"], data["default_branch"])


def add_seed_repos(
    client: GitHubClient,
    conn: sqlite3.Connection,
    seed_file: Path,
    now: datetime,
    on_missing: Callable[[str], None] | None = None,
    excludes: RepoExcludes | None = None,
    on_excluded: Callable[[str, str], None] | None = None,
) -> int:
    """Store every seed repo not already in the table. Returns how many were added."""
    excludes = excludes or RepoExcludes()
    already = known_repos(conn)
    added = 0
    for name in parse_seed_file(seed_file.read_text(encoding="utf-8")):
        if name in already:
            continue
        reason = excludes.reason(name)
        if reason:
            if on_excluded:
                on_excluded(name, reason)
            continue
        repo = fetch_repo(client, name)
        if repo is None:
            if on_missing:
                on_missing(name)
            continue
        # Check the canonical name too: a renamed repo may now live in an excluded org.
        reason = excludes.reason(repo.full_name)
        if reason:
            if on_excluded:
                on_excluded(repo.full_name, reason)
            continue
        if repo.full_name not in already:
            save_repo(conn, repo, now)
            already.add(repo.full_name)
            added += 1
    return added
