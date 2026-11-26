"""What an agent run did for one item: each tool call, and the model calls around them.

Plain data with no dependencies, so AnalysisResult (and the response cache and eval run
files that store it) can carry it.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

PREVIEW_CHARS = 240

SUBMITTED = "submitted"
FAILED = "failed"


@dataclass(frozen=True)
class ToolCallRecord:
    step: int
    """1-based number of this tool call within the item."""
    turn: int
    """1-based number of the model call that asked for it."""
    name: str
    args: dict[str, Any]
    ok: bool
    cached: bool
    output_chars: int
    output_tokens: int
    """Char-based estimate of what the model received."""
    latency_s: float
    preview: str
    """The first PREVIEW_CHARS characters of the output (redacted, untrusted)."""


@dataclass(frozen=True)
class Trajectory:
    tools: list[str]
    """Tools offered to the model (submit_diagnosis not included)."""
    max_steps: int
    max_input_tokens: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    call_input_tokens: list[int] = field(default_factory=list)
    """Actual input tokens of each model call, in order (sums to input_tokens); empty in
    trajectories recorded before it was kept."""
    output_tokens: int = 0
    cost: float | None = None
    """USD for the model calls; None without a configured price."""
    hit_max_steps: bool = False
    """The tool-call limit was reached, so submit_diagnosis was forced."""
    forced: list[str] = field(default_factory=list)
    """Why submit_diagnosis was forced, in order: max_steps, max_input_tokens,
    text_reprompt, validation_retry."""
    warnings: list[str] = field(default_factory=list)
    """Problems in the accepted answer that were fixed by dropping parts of it (invalid
    evidence_files entries)."""
    stop: str = FAILED
    """SUBMITTED when a valid submit_diagnosis ended the run, else FAILED."""

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


def preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS] + "..."


def trajectory_to_dict(trajectory: Trajectory) -> dict[str, Any]:
    return asdict(trajectory)


def trajectory_from_dict(data: dict[str, Any]) -> Trajectory:
    fields = set(Trajectory.__dataclass_fields__)
    values = {k: v for k, v in data.items() if k in fields}
    values["tool_calls"] = [ToolCallRecord(**call) for call in data.get("tool_calls", [])]
    return Trajectory(**values)
