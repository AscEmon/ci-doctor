"""Test doubles for the agent: a fake GitHub API and a fixed run context. No network."""

import base64
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from cidoc.agent.context import RunContext
from cidoc.github_client import RetryableHTTPError
from cidoc.llm.base import (
    FunctionCall,
    FunctionSpec,
    Message,
    ModelMessage,
    ToolCallResponse,
    Usage,
)
from tests.fakes import VALID_DIAGNOSIS

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
        self.closed = False

    def close(self) -> None:
        self.closed = True

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


# --- scripted tool-calling model ------------------------------------------------------

Call = tuple[str, dict[str, Any]]


def turn(
    *calls: Call,
    text: str = "",
    usage: Usage | None = None,
    finish: str = "STOP",
    block: str | None = None,
) -> ToolCallResponse:
    """One model turn: function calls (ids c1, c2, ...) and/or text."""
    function_calls = [FunctionCall(name, args, f"c{n}") for n, (name, args) in enumerate(calls, 1)]
    return ToolCallResponse(
        message=ModelMessage(text, function_calls),
        complete=finish == "STOP" and block is None,
        finish_reason=finish,
        block_reason=block,
        usage=usage or Usage(1000, 50, 0.4),
    )


def submit(**overrides: Any) -> Call:
    return ("submit_diagnosis", {**VALID_DIAGNOSIS, **overrides})


class ScriptedToolProvider:
    """Answers generate_with_tools from a script of turns; records every request."""

    def __init__(self, *turns: ToolCallResponse | BaseException, model: str = "fake-model") -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate_with_tools(
        self,
        system: str,
        messages: Sequence[Message],
        functions: Sequence[FunctionSpec],
        force: Sequence[str] | None = None,
    ) -> ToolCallResponse:
        self.requests.append(
            {
                "system": system,
                "messages": list(messages),
                "functions": [f.name for f in functions],
                "force": list(force) if force else None,
            }
        )
        outcome = self.turns.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
