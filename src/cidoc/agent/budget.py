"""--max-cost for agent runs: the same CostGuard as Phase 1, on every model call.

An item makes several model calls, so the guard checks each one before it is made, and
`check_item` refuses to start an item whose worst case (max_steps + 1 calls) would not
fit, so the budget is not spent on an item that would be cut off half way.
"""

import json
from collections.abc import Sequence

from cidoc.agent.loop import COMPACT_CHARS, AgentLimits, estimate_input
from cidoc.agent.prompt import AgentPrompt
from cidoc.analyze.costguard import CallEstimate, CostGuard
from cidoc.analyze.preprocess import PreprocessedLog, estimate_tokens
from cidoc.llm.base import FunctionSpec, Message, ToolCallingProvider, ToolCallResponse

# A compacted result: COMPACT_CHARS of text, the size note and the <tool_result> tags.
COMPACT_RESULT_TOKENS = estimate_tokens("x" * (COMPACT_CHARS + 120))


def _functions_json(functions: Sequence[FunctionSpec]) -> str:
    return json.dumps([[f.name, f.description, f.parameters] for f in functions])


def estimate_item_calls(
    prompt: AgentPrompt, log: PreprocessedLog, limits: AgentLimits
) -> list[CallEstimate]:
    """Worst-case model calls for one item: one per tool step plus the final submit.

    Each call resends the conversation: call k carries the latest tool result in full
    (at most cap_tokens) and the k - 1 earlier ones compacted; the total stays within
    max_input_tokens, as the loop enforces.
    """
    first = (
        estimate_tokens(prompt.system)
        + estimate_tokens(_functions_json(prompt.functions))
        + estimate_tokens(prompt.user(log))
    )
    calls: list[CallEstimate] = []
    spent = 0
    for k in range(limits.max_steps + 1):
        size = first + (limits.cap_tokens + (k - 1) * COMPACT_RESULT_TOKENS if k else 0)
        if calls and spent + size > limits.max_input_tokens:
            break
        calls.append(CallEstimate(input_tokens=size))
        spent += size
    return calls


class BudgetedToolProvider:
    """A ToolCallingProvider that checks the guard before every call and charges it after."""

    def __init__(self, provider: ToolCallingProvider, guard: CostGuard) -> None:
        self._provider = provider
        self.guard = guard

    @property
    def model(self) -> str:
        return self._provider.model

    def check_item(self, calls: Sequence[CallEstimate]) -> None:
        """BudgetExceededError if the worst case of an item would pass the budget."""
        total = sum(c.input_tokens for c in calls)
        output = sum(c.output_tokens for c in calls)
        self.guard.check(CallEstimate(input_tokens=total, output_tokens=output))

    def generate_with_tools(
        self,
        system: str,
        messages: Sequence[Message],
        functions: Sequence[FunctionSpec],
        force: Sequence[str] | None = None,
    ) -> ToolCallResponse:
        estimate = estimate_input(system, _functions_json(functions), list(messages))
        self.guard.check(CallEstimate(input_tokens=estimate))
        response = self._provider.generate_with_tools(system, messages, functions, force)
        self.guard.charge_usage(response.usage)
        return response
