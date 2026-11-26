"""Test doubles for the agent: a fake GitHub API and a fixed run context. No network."""

import base64
from collections.abc import Callable
from typing import Any

import httpx

from cidoc.agent.context import RunContext
from cidoc.github_client import RetryableHTTPError

SHA = "a1b2c3d4e5f6" + "0" * 28
FIX_SHA = "f1f1f1f1f1f1" + "9" * 28
"""The fixing commit. Lives in the test database only; must never reach the agent."""
REPO = "o/a"
RUN_ID = 70
JOB_ID = 7
WORKFLOW = ".github/workflows/ci.yml"


def http_error(status: int, retryable: bool = False) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.github.test/x")
    response = httpx.Response(status, request=request)
    cls = RetryableHTTPError if retryable else httpx.HTTPStatusError
    return cls(f"HTTP {status}", request=request, response=response)


def file_entry(content: str | bytes, name: str = "f") -> dict[str, Any]:
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return {
        "type": "file",
        "name": name,
        "size": len(raw),
        "encoding": "base64",
        "content": base64.b64encode(raw).decode("ascii"),
    }


Route = Any | BaseException | Callable[[dict[str, Any] | None], Any]


class FakeGitHub:
    """Answers GET JSON from `routes` (URL path -> value, exception, or callable of the
    params); anything else is a 404. Records every call."""

    def __init__(self, routes: dict[str, Route] | None = None) -> None:
        self.routes: dict[str, Route] = dict(routes or {})
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((url, params))
        if url not in self.routes:
            raise http_error(404)
        value = self.routes[url]
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            return value(params)
        return value


def context(log_lines: list[str] | None = None, **overrides: Any) -> RunContext:
    fields: dict[str, Any] = {
        "job_id": JOB_ID,
        "repo": REPO,
        "run_id": RUN_ID,
        "head_sha": SHA,
        "head_branch": "main",
        "workflow_path": WORKFLOW,
        "log_lines": log_lines if log_lines is not None else ["line one", "line two"],
    }
    fields.update(overrides)
    return RunContext(**fields)


def contents(path: str) -> str:
    return f"/repos/{REPO}/contents/{path}" if path else f"/repos/{REPO}/contents"
