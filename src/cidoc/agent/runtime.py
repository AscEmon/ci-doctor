"""Putting an agent run together for one job: prompt, limits, GitHub client, tools.

Shared by `cidoc analyze --agent` and `cidoc eval --agent`, so both run the same thing.
"""

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr

from cidoc.agent.cache import ToolCache
from cidoc.agent.context import RunContext, load_context
from cidoc.agent.loop import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_STEPS, AgentLimits
from cidoc.agent.prompt import AgentPrompt, build_agent_prompt
from cidoc.agent.tools import (
    OUTPUT_CAP_TOKENS,
    GitHubReader,
    ToolBox,
    ToolEnv,
    UnknownToolError,
    select_tools,
)
from cidoc.analyze.prompt import Prompt
from cidoc.github_client import GitHubClient

TOOLS_MAX_ATTEMPTS = 1
"""No retries for tool requests: a rate limit or server error comes straight back to
the model as a short error text instead of sleeping until the quota resets."""


def parse_tool_names(value: str | None) -> list[str] | None:
    """`--tools a,b` as a list (validated); None (every tool) when not given."""
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise UnknownToolError("--tools needs at least one tool name")
    select_tools(names)  # raises UnknownToolError
    return names


@dataclass(frozen=True)
class AgentSetup:
    prompt: AgentPrompt
    limits: AgentLimits
    tool_names: list[str] | None
    """None: every tool."""

    @property
    def tools(self) -> list[str]:
        return self.prompt.tools


def agent_setup(
    base: Prompt,
    tool_names: Sequence[str] | None = None,
    max_steps: int | None = None,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    cap_tokens: int = OUTPUT_CAP_TOKENS,
) -> AgentSetup:
    limits = AgentLimits(
        max_steps=max_steps or DEFAULT_MAX_STEPS,
        max_input_tokens=max_input_tokens,
        cap_tokens=cap_tokens,
    )
    specs = select_tools(tool_names)
    prompt = build_agent_prompt(
        base, specs, limits.max_steps, limits.max_input_tokens, limits.cap_tokens
    )
    names = list(tool_names) if tool_names is not None else None
    return AgentSetup(prompt=prompt, limits=limits, tool_names=names)


def open_tools_github(token: SecretStr, base_url: str) -> GitHubClient:
    """Our GitHub client with the existing read-only token: the only network access the
    tools have."""
    return GitHubClient(token, base_url=base_url, max_attempts=TOOLS_MAX_ATTEMPTS)


def job_context(
    job_id: int, log_text: str, open_db: Callable[[], sqlite3.Connection]
) -> RunContext:
    conn = open_db()
    try:
        return load_context(conn, job_id, log_text)
    finally:
        conn.close()


def make_toolbox(
    ctx: RunContext, github: GitHubReader | None, setup: AgentSetup, data_dir: Path
) -> ToolBox:
    return ToolBox(
        ToolEnv(ctx, github),
        names=setup.tool_names,
        cap_tokens=setup.limits.cap_tokens,
        cache=ToolCache.in_data_dir(data_dir),
    )
