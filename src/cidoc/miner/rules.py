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


# Runs triggered by issue activity are issue bots (triage, project boards), never builds.
ISSUE_BOT_EVENTS = frozenset({"issues", "issue_comment"})


def event_skip_reason(event: str) -> str | None:
    if event in ISSUE_BOT_EVENTS:
        return f"issue bot ({event} event)"
    return None


def run_skip_reason(run: dict[str, Any]) -> str | None:
    return event_skip_reason(run.get("event") or "") or workflow_skip_reason(
        run.get("path") or "", run.get("name") or ""
    )


# --- Job-level scope: PR hygiene, issue bots, aggregators -----------------------------
#
# These jobs fail by design (a PR title doesn't match a convention, a required job
# failed elsewhere) and say nothing about building the app.

# Actions whose failure is the whole job's purpose. "*/name" matches any owner (forks
# and re-publications of the same action are common).
OUT_OF_SCOPE_ACTIONS: dict[str, str] = {
    "amannn/action-semantic-pull-request": "PR hygiene",
    "actions/labeler": "PR hygiene",
    "actions/stale": "PR hygiene",
    "actions/first-interaction": "PR hygiene",
    "*/release-drafter": "PR hygiene",
    "*/pr-size-labeler": "PR hygiene",
    "actions/add-to-project": "issue bot",
    "re-actors/alls-green": "aggregator",
}
# 'Run owner/name@ref' as a step title (API step name or log step header). Sub-paths
# like owner/name/sub@ref count as owner/name.
_ACTION_STEP = re.compile(r"^(?:##\[group\])?Run ([\w.-]+)/([\w.-]+)(?:/[^@\s]*)?@", re.MULTILINE)
PR_HYGIENE_JOB_NAME = re.compile(
    r"\bpr[ _-]?title\b|^label(?:er|[ _-]?prs?)$|^welcome$", re.IGNORECASE
)
AGGREGATOR_JOB_NAME = re.compile(
    r"^(?:ci[ _-]?success|all[ _-]?green|gate|results?|checks?)$", re.IGNORECASE
)
# What an aggregator's failing step shows. GitHub substitutes ${{ needs... }} before
# logging, so the usual trace is the job's NEEDS (or similar) env printed as JSON.
_NEEDS_EVIDENCE = re.compile(
    r"^\s*[A-Z0-9_]*NEEDS[A-Z0-9_]*:\s*\{|toJSON\(needs\)|\bneeds\.[\w*-]+\.result\b",
    re.MULTILINE,
)
# Steps every job has; they don't count when judging how much a job does.
_BOILERPLATE_STEP = re.compile(r"^(?:set up job|complete job|post .*)$", re.IGNORECASE)
# An aggregator by name runs at most a checkout and the one check step.
AGGREGATOR_MAX_REAL_STEPS = 2


@dataclass(frozen=True)
class JobFacts:
    """What is known about a failed job. Empty / None fields mean 'not known'."""

    job_name: str = ""
    failed_steps: tuple[str, ...] = ()
    step_names: tuple[str, ...] = ()
    """All step names of the job; empty when unknown (rows mined before they were kept)."""
    failed_step_text: str | None = None
    """The failed step(s)' log lines; None before the log is downloaded."""


def action_skip_reason(text: str) -> str | None:
    """Reason if `text` holds a 'Run owner/name@ref' step for an out-of-scope action."""
    for owner, name in _ACTION_STEP.findall(text):
        ref = f"{owner}/{name}".lower()
        reason = OUT_OF_SCOPE_ACTIONS.get(ref) or OUT_OF_SCOPE_ACTIONS.get(f"*/{name.lower()}")
        if reason:
            return f"{reason} ({ref})"
    return None


def job_skip_reason(facts: JobFacts) -> str | None:
    """Why a failed job is out of scope, or None to keep it."""
    name = facts.job_name.strip()
    if PR_HYGIENE_JOB_NAME.search(name):
        return f"PR hygiene (job '{name}')"
    for text in (*facts.failed_steps, facts.failed_step_text or ""):
        reason = action_skip_reason(text)
        if reason:
            return reason
    if len(facts.failed_steps) != 1:
        return None  # an aggregator fails in exactly one step
    if facts.failed_step_text and _NEEDS_EVIDENCE.search(facts.failed_step_text):
        return "aggregator (reads needs results)"
    real_steps = [s for s in facts.step_names if not _BOILERPLATE_STEP.match(s.strip())]
    if AGGREGATOR_JOB_NAME.match(name) and 0 < len(real_steps) <= AGGREGATOR_MAX_REAL_STEPS:
        return f"aggregator (job '{name}')"
    return None


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
