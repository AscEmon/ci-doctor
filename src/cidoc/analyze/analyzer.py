"""One analysis: preprocessed log in, validated Diagnosis (or a recorded failure) out.

A response that is blocked, cut off, empty, not JSON, or does not match the schema is
retried once, telling the model what was wrong. If the retry fails too, the result
carries no diagnosis and the reason why. API errors (after the provider's own backoff)
propagate: they say nothing about the log.
"""

from dataclasses import dataclass

from pydantic import ValidationError

from cidoc.analyze.preprocess import PreprocessedLog
from cidoc.analyze.prompt import SYSTEM_INSTRUCTION, user_prompt
from cidoc.analyze.schema import RESPONSE_SCHEMA, Diagnosis
from cidoc.llm.base import LLMProvider, StructuredResponse, Usage

MAX_ATTEMPTS = 2  # the first call and one retry


class InvalidResponseError(ValueError):
    """The model's answer cannot be used; the message says why, for the retry prompt."""


@dataclass(frozen=True)
class AnalysisResult:
    model: str
    diagnosis: Diagnosis | None
    failure: str | None
    """Why there is no diagnosis (from the last attempt); None on success."""
    calls: list[Usage]

    @property
    def ok(self) -> bool:
        return self.diagnosis is not None


def _format_validation_error(exc: ValidationError) -> str:
    # include_input=False: never echo the model's (possibly long) text back.
    problems = [
        f"{'.'.join(str(p) for p in err['loc']) or 'response'}: {err['msg']}"
        for err in exc.errors(include_input=False)
    ]
    return "; ".join(problems)


def parse_response(response: StructuredResponse, total_lines: int) -> Diagnosis:
    """The response as a Diagnosis, or InvalidResponseError saying what is wrong."""
    if response.block_reason:
        raise InvalidResponseError(f"blocked: {response.block_reason}")
    if not response.complete:
        raise InvalidResponseError(f"incomplete: finish reason {response.finish_reason}")
    if not response.text.strip():
        raise InvalidResponseError("empty response")
    try:
        data = response.json()
    except ValueError as exc:
        raise InvalidResponseError(f"invalid JSON: {exc}") from None
    try:
        diagnosis = Diagnosis.model_validate(data)
    except ValidationError as exc:
        raise InvalidResponseError(f"schema mismatch: {_format_validation_error(exc)}") from None
    beyond = [n for n in diagnosis.evidence_lines if n > total_lines]
    if beyond:
        raise InvalidResponseError(
            f"schema mismatch: evidence_lines {beyond} are past the last log line {total_lines}"
        )
    return diagnosis


def retry_prompt(prompt: str, error: str) -> str:
    return (
        f"{prompt}\n\nYour previous answer was rejected: {error}. "
        "Answer again with only a JSON object that matches the response schema."
    )


def analyze_log(provider: LLMProvider, log: PreprocessedLog) -> AnalysisResult:
    prompt = user_prompt(log)
    calls: list[Usage] = []
    error = ""
    for attempt in range(MAX_ATTEMPTS):
        text = prompt if attempt == 0 else retry_prompt(prompt, error)
        response = provider.generate_structured(SYSTEM_INSTRUCTION, text, RESPONSE_SCHEMA)
        calls.append(response.usage)
        try:
            diagnosis = parse_response(response, log.total_lines)
        except InvalidResponseError as exc:
            error = str(exc)
            continue
        return AnalysisResult(provider.model, diagnosis, None, calls)
    return AnalysisResult(provider.model, None, error, calls)
