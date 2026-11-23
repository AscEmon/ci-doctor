"""Rules deciding which repos and runs are worth mining. Shared by mining and pruning.

Rules take plain fields rather than API payloads, so the same functions apply to a
GitHub run object during mining and to a stored `failures` row during pruning.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXCLUDE_FILENAME = "exclude_repos.txt"
# Flutter/Dart framework and tooling repos: their CI failures are about building the SDK
# itself, not the app-level Flutter builds ci-doctor targets. Excluded by org.
DEFAULT_EXCLUDED_OWNERS = frozenset({"flutter", "dart-lang"})
_EXCLUDE_LINE_RE = re.compile(r"^[A-Za-z0-9-]+/(\*|[A-Za-z0-9._-]+)$")

# GitHub-managed workflows (Pages builds, Dependabot, CodeQL default setup, Copilot
# agent) live under this prefix instead of .github/workflows/.
DYNAMIC_WORKFLOW_PREFIX = "dynamic/"
# Bot/security workflows whose failures say nothing about the project's build.
# Matched case-insensitively against the workflow name.
BOT_WORKFLOW_NAME_MARKERS = ("dependabot", "codeql", "copilot")


def workflow_skip_reason(path: str, name: str) -> str | None:
    """Why a workflow's runs are not build failures worth mining, or None to keep them."""
    if path.startswith(DYNAMIC_WORKFLOW_PREFIX):
        return "dynamic workflow"
    lowered = name.lower()
    for marker in BOT_WORKFLOW_NAME_MARKERS:
        if marker in lowered:
            return f"{marker} workflow"
    return None


def run_skip_reason(run: dict[str, Any]) -> str | None:
    return workflow_skip_reason(run.get("path") or "", run.get("name") or "")


def is_fork_run(run: dict[str, Any], repo: str) -> bool:
    """True when the run's code came from another repository (a fork PR).

    A null head_repository means the source fork was deleted; treat it as a fork too.
    Its branch names belong to the fork, so pairing them with the base repo's runs
    would match unrelated branches that happen to share a name (e.g. "main").
    """
    head = run.get("head_repository")
    if head is None:
        return True
    full_name: str = head.get("full_name") or ""
    return full_name.lower() != repo.lower()


@dataclass(frozen=True)
class RepoExcludes:
    """Repos never to mine. Names are compared case-insensitively, like GitHub does."""

    owners: frozenset[str] = DEFAULT_EXCLUDED_OWNERS
    repos: frozenset[str] = frozenset()

    def reason(self, full_name: str) -> str | None:
        """Why `full_name` is excluded, or None to keep it."""
        lowered = full_name.lower()
        owner = lowered.split("/", 1)[0]
        if owner in self.owners:
            return f"excluded org {owner}/*"
        if lowered in self.repos:
            return f"listed in {EXCLUDE_FILENAME}"
        return None


def parse_exclude_file(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return (owners, repos) from an exclude file, lowercased.

    One `owner/repo` per line; `owner/*` excludes the whole org. Blank lines and `#`
    comments are ignored. Raises ValueError on a malformed line.
    """
    owners: set[str] = set()
    repos: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not _EXCLUDE_LINE_RE.match(line):
            raise ValueError(f"{EXCLUDE_FILENAME} line {lineno}: expected owner/repo, got {line!r}")
        owner, name = line.lower().split("/", 1)
        if name == "*":
            owners.add(owner)
        else:
            repos.add(f"{owner}/{name}")
    return frozenset(owners), frozenset(repos)


def load_repo_excludes(data_dir: Path) -> RepoExcludes:
    """Default excludes plus `<data_dir>/exclude_repos.txt` if it exists."""
    path = data_dir / EXCLUDE_FILENAME
    if not path.exists():
        return RepoExcludes()
    owners, repos = parse_exclude_file(path.read_text(encoding="utf-8"))
    return RepoExcludes(owners=DEFAULT_EXCLUDED_OWNERS | owners, repos=repos)
