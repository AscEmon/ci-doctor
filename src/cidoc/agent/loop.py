"""The agent loop, written out by hand: model call, run the tools it asks for, repeat,
until it calls submit_diagnosis.

Limits, per item:
- max_steps tool calls (default 6). Once they are used up, the next model call is
  forced to submit_diagnosis (function calling mode ANY, only that function allowed).
  Extra calls in the same turn are answered "not run" so every call gets a result.
- max_input_tokens over all model calls (default 30,000, actual usage). A model call
  that might be followed by a tool round is made only if a forced submit after that
  round would still fit; otherwise this call is the forced submit.
- A turn with text and no function call is re-prompted once, forcing submit_diagnosis.
- submit_diagnosis arguments are validated with the Phase 1 Diagnosis model (plus
  evidence_files); an invalid answer is rejected back to the model once, forced again.
The validation retry and text re-prompt may push input past max_input_tokens by one call.

API errors (after the provider's backoff) propagate, as in Phase 1.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from cidoc.agent.prompt import SUBMIT_FUNCTION, AgentPrompt
from cidoc.agent.tools import OUTPUT_CAP_TOKENS, ToolBox, ToolOutput
from cidoc.agent.trajectory import (
    FAILED,
    SUBMITTED,
    ToolCallRecord,
    Trajectory,
    preview,
)
from cidoc.analyze.analyzer import AnalysisResult, InvalidResponseError
from cidoc.analyze.cache import ResponseCache, cache_key, log_hash
from cidoc.analyze.preprocess import PreprocessedLog, estimate_tokens
from cidoc.analyze.schema import AgentDiagnosis
from cidoc.llm.base import (
    FunctionCall,
    FunctionResult,
    FunctionResultsMessage,
    Message,
    ToolCallingProvider,
    ToolCallResponse,
    Usage,
    UserMessage,
)
from cidoc.llm.pricing import Price

DEFAULT_MAX_STEPS = 6
DEFAULT_MAX_INPUT_TOKENS = 30_000
EXTRA_MODEL_CALLS = 3
"""Model calls beyond max_steps: the forced submit, a text re-prompt, a validation retry."""

FORCED_MAX_STEPS = "max_steps"
FORCED_MAX_INPUT = "max_input_tokens"
FORCED_REPROMPT = "text_reprompt"
FORCED_RETRY = "validation_retry"

REPROMPT = (
    "You answered in text. Call submit_diagnosis now with your diagnosis; do not answer"
    " in plain text."
)
NOT_RUN_LIMIT = "not run: the tool-call limit is reached; call submit_diagnosis now"
NOT_RUN_SUBMITTED = "not run: submit_diagnosis was called in the same turn"


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = DEFAULT_MAX_STEPS
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    cap_tokens: int = OUTPUT_CAP_TOKENS

    @property
    def max_model_calls(self) -> int:
        return self.max_steps + EXTRA_MODEL_CALLS


ResultWrapper = Callable[[ToolOutput], str]


def plain_result(output: ToolOutput) -> str:
    return output.text


def parse_submit(args: dict[str, Any], total_lines: int) -> AgentDiagnosis:
    """submit_diagnosis arguments as a validated diagnosis, or InvalidResponseError."""
    try:
        diagnosis = AgentDiagnosis.model_validate(args)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
            for err in exc.errors(include_input=False)
        )
        raise InvalidResponseError(f"schema mismatch: {problems}") from None
    beyond = [n for n in diagnosis.evidence_lines if n > total_lines]
    if beyond:
        raise InvalidResponseError(
            f"schema mismatch: evidence_lines {beyond} are past the last log line {total_lines}"
        )
    return diagnosis


def _message_tokens(message: Message) -> int:
    if isinstance(message, UserMessage):
        return estimate_tokens(message.text)
    if isinstance(message, FunctionResultsMessage):
        return sum(estimate_tokens(json.dumps(r.response)) for r in message.results)
    calls = json.dumps([[c.name, c.args] for c in message.function_calls])
    return estimate_tokens(message.text) + estimate_tokens(calls)


def estimate_input(system: str, functions_json: str, messages: list[Message]) -> int:
    """Char-based estimate of the next call's input tokens."""
    return (
        estimate_tokens(system)
        + estimate_tokens(functions_json)
        + sum(_message_tokens(m) for m in messages)
    )


def _raw_output(response: ToolCallResponse) -> str:
    """The model's turn, for inspection: its text and function calls."""
    return json.dumps(
        {"text": response.text, "calls": [[c.name, c.args] for c in response.function_calls]},
        ensure_ascii=False,
    )


@dataclass
class _Run:
    """Mutable state of one agent run."""

    calls: list[Usage] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    records: list[ToolCallRecord] = field(default_factory=list)
    forced: list[str] = field(default_factory=list)
    steps: int = 0
    hit_max_steps: bool = False

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)


def run_agent(
    provider: ToolCallingProvider,
    log: PreprocessedLog,
    prompt: AgentPrompt,
    tools: ToolBox,
    limits: AgentLimits = AgentLimits(),  # noqa: B008 - frozen
    price: Price | None = None,
    wrap: ResultWrapper = plain_result,
) -> AnalysisResult:
    run = _Run()
    messages: list[Message] = [UserMessage(prompt.user(log))]
    functions_json = json.dumps([[f.name, f.description, f.parameters] for f in prompt.functions])
    force_reason: str | None = None
    reprompted = retried = False
    failure: str | None = None
    diagnosis: AgentDiagnosis | None = None

    for turn in range(1, limits.max_model_calls + 1):
        if force_reason is None:
            if run.steps >= limits.max_steps:
                force_reason, run.hit_max_steps = FORCED_MAX_STEPS, True
            else:
                next_input = estimate_input(prompt.system, functions_json, messages)
                after_round = next_input + limits.cap_tokens
                if run.input_tokens + next_input + after_round > limits.max_input_tokens:
                    force_reason = FORCED_MAX_INPUT
            if force_reason is not None:
                run.forced.append(force_reason)
        forcing = force_reason is not None
        response = provider.generate_with_tools(
            prompt.system,
            messages,
            prompt.functions,
            force=[SUBMIT_FUNCTION] if forcing else None,
        )
        run.calls.append(response.usage)
        run.outputs.append(_raw_output(response))
        if response.block_reason:
            failure = f"blocked: {response.block_reason}"
            break
        messages.append(response.message)

        submit = next((c for c in response.function_calls if c.name == SUBMIT_FUNCTION), None)
        if submit is not None:
            try:
                diagnosis = parse_submit(submit.args, log.total_lines)
                break
            except InvalidResponseError as exc:
                failure = str(exc)
                if retried:
                    break
                retried = True
                rejection = (
                    f"rejected: {exc}. Call submit_diagnosis again with corrected arguments."
                )
                messages.append(_results(response.function_calls, submit, rejection))
                force_reason = FORCED_RETRY
                run.forced.append(FORCED_RETRY)
                continue

        if not response.function_calls:
            failure = (
                f"no submit_diagnosis call (finish reason {response.finish_reason})"
                if not response.complete
                else "answered in text instead of calling submit_diagnosis"
            )
            if reprompted:
                break
            reprompted = True
            messages.append(UserMessage(REPROMPT))
            force_reason = FORCED_REPROMPT
            run.forced.append(FORCED_REPROMPT)
            continue

        results: list[FunctionResult] = []
        for call in response.function_calls:
            if forcing or run.steps >= limits.max_steps:
                results.append(FunctionResult(call, {"error": NOT_RUN_LIMIT}))
                continue
            run.steps += 1
            output = tools.call(call.name, call.args)
            run.records.append(_record(run.steps, turn, output))
            results.append(FunctionResult(call, {"output": wrap(output)}))
        messages.append(FunctionResultsMessage(results))
    else:
        failure = failure or "no valid submit_diagnosis within the model-call limit"

    trajectory = Trajectory(
        tools=prompt.tools,
        max_steps=limits.max_steps,
        max_input_tokens=limits.max_input_tokens,
        tool_calls=run.records,
        model_calls=len(run.calls),
        input_tokens=run.input_tokens,
        output_tokens=sum(c.output_tokens for c in run.calls),
        cost=price.cost(run.input_tokens, sum(c.output_tokens for c in run.calls))
        if price
        else None,
        hit_max_steps=run.hit_max_steps,
        forced=run.forced,
        stop=SUBMITTED if diagnosis is not None else FAILED,
    )
    return AnalysisResult(
        model=provider.model,
        diagnosis=diagnosis,
        failure=None if diagnosis is not None else failure,
        calls=run.calls,
        outputs=run.outputs,
        trajectory=trajectory,
    )


def _results(
    calls: list[FunctionCall], submit: FunctionCall, rejection: str
) -> FunctionResultsMessage:
    """A result for every call of the turn: the rejection for submit, "not run" for the
    rest (the model must get an answer to each call it made)."""
    return FunctionResultsMessage(
        [
            FunctionResult(c, {"error": rejection if c is submit else NOT_RUN_SUBMITTED})
            for c in calls
        ]
    )


def _record(step: int, turn: int, output: ToolOutput) -> ToolCallRecord:
    return ToolCallRecord(
        step=step,
        turn=turn,
        name=output.name,
        args=output.args,
        ok=output.ok,
        cached=output.cached,
        output_chars=len(output.text),
        output_tokens=estimate_tokens(output.text),
        latency_s=output.latency_s,
        preview=preview(output.text),
    )


# --- cached runs ------------------------------------------------------------------------


def agent_digest(log: PreprocessedLog, tools: ToolBox) -> str:
    """What an agent result depends on besides the prompt and model: the excerpt and
    the job its tools read (two jobs with one log can still differ in their repos)."""
    ctx = tools.env.ctx
    payload = json.dumps([log_hash(log), ctx.job_id, ctx.repo, ctx.run_id, ctx.head_sha])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_agent_cached(
    provider: ToolCallingProvider,
    log: PreprocessedLog,
    prompt: AgentPrompt,
    tools: ToolBox,
    cache: ResponseCache,
    limits: AgentLimits = AgentLimits(),  # noqa: B008 - frozen
    price: Price | None = None,
    wrap: ResultWrapper = plain_result,
    refresh: bool = False,
) -> AnalysisResult:
    """The cached result if there is one (unless `refresh`), else run and store it."""
    digest = agent_digest(log, tools)
    key = cache_key(prompt.version, digest, provider.model)
    if not refresh:
        hit = cache.get(key)
        if hit is not None:
            return hit
    result = run_agent(provider, log, prompt, tools, limits, price, wrap)
    meta = {
        "prompt_version": prompt.version,
        "guideline_hash": prompt.guideline_hash,
        "log_hash": log_hash(log),
        "agent_digest": digest,
        "model": provider.model,
    }
    cache.put(key, result, meta)
    return result
