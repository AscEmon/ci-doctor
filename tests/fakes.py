"""Shared test doubles."""

import json
from typing import Any

from cidoc.llm.base import StructuredResponse, Usage

VALID_DIAGNOSIS: dict[str, Any] = {
    "category": "dart_compile",
    "root_cause": "lib/main.dart references an undefined name 'foo'.",
    "evidence_lines": [5],
    "suggested_fix": "Define or import 'foo' in lib/main.dart.",
    "confidence": 0.9,
}


def ok(data: dict[str, Any] | str, **kw: Any) -> StructuredResponse:
    text = data if isinstance(data, str) else json.dumps(data)
    fields: dict[str, Any] = {
        "complete": True,
        "finish_reason": "STOP",
        "block_reason": None,
        "usage": Usage(100, 20, 0.5),
    }
    fields.update(kw)
    return StructuredResponse(text=text, **fields)


class ScriptedProvider:
    def __init__(self, *responses: StructuredResponse | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def generate_structured(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> StructuredResponse:
        self.calls.append((system, user, schema))
        outcome = self.responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
