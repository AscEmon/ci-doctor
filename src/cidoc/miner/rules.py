"""Rules deciding which runs are worth mining. Shared by mining and pruning.

Rules take plain fields rather than API payloads, so the same functions apply to a
GitHub run object during mining and to a stored `failures` row during pruning.
"""

from typing import Any

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
