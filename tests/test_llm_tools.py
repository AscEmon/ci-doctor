"""Gemini function calling through the provider interface. Fake client, no network."""

from typing import Any

import pytest
from google.genai import errors, types

from cidoc.llm.base import (
    FunctionCall,
    FunctionResult,
    FunctionResultsMessage,
    FunctionSpec,
    LLMError,
    Message,
    ModelMessage,
    UserMessage,
)
from cidoc.llm.gemini import to_contents
from tests.test_llm import FakeClient, FakeTime, provider, rate_limit_error

READ_FILE = FunctionSpec(
    name="read_file",
    description="Read a file.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)
SUBMIT = FunctionSpec(name="submit_diagnosis", description="Answer.", parameters={})


def call_response(
    *calls: tuple[str, dict[str, Any], str | None],
    text: str | None = None,
    finish: types.FinishReason = types.FinishReason.STOP,
    signature: bytes | None = None,
) -> types.GenerateContentResponse:
    parts = [types.Part(text=text)] if text else []
    parts += [
        types.Part(
            function_call=types.FunctionCall(name=name, args=args, id=call_id),
            thought_signature=signature,
        )
        for name, args, call_id in calls
    ]
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000, candidates_token_count=30, thoughts_token_count=70
        ),
    )


def config_of(client: FakeClient, n: int = 0) -> types.GenerateContentConfig:
    config: types.GenerateContentConfig = client.models.calls[n]["config"]
    return config


def test_declares_functions_and_lets_the_model_choose_by_default() -> None:
    client = FakeClient(call_response(("read_file", {"path": "pubspec.yaml"}, "c1")))
    gemini = provider(client, FakeTime())

    gemini.generate_with_tools("SYSTEM", [UserMessage("log")], [READ_FILE, SUBMIT])

    config = config_of(client)
    assert config.system_instruction == "SYSTEM"
    assert config.temperature == 0.0
    assert config.tools is not None
    tool = config.tools[0]
    assert isinstance(tool, types.Tool)
    declarations = tool.function_declarations or []
    assert [d.name for d in declarations] == ["read_file", "submit_diagnosis"]
    assert declarations[0].parameters_json_schema == READ_FILE.parameters
    assert declarations[0].description == "Read a file."
    assert config.tool_config is not None
    calling = config.tool_config.function_calling_config
    assert calling is not None
    assert calling.mode == types.FunctionCallingConfigMode.AUTO
    assert calling.allowed_function_names is None
    assert config.automatic_function_calling is not None
    assert config.automatic_function_calling.disable is True
    assert config.response_mime_type is None  # no JSON mode with tools


def test_force_requires_a_call_to_the_named_function() -> None:
    client = FakeClient(call_response(("submit_diagnosis", {"category": "other"}, None)))
    gemini = provider(client, FakeTime())

    gemini.generate_with_tools(
        "S", [UserMessage("u")], [READ_FILE, SUBMIT], force=["submit_diagnosis"]
    )

    tool_config = config_of(client).tool_config
    assert tool_config is not None
    calling = tool_config.function_calling_config
    assert calling is not None
    assert calling.mode == types.FunctionCallingConfigMode.ANY
    assert calling.allowed_function_names == ["submit_diagnosis"]


def test_reads_function_calls_text_and_usage() -> None:
    client = FakeClient(
        call_response(
            ("read_file", {"path": "a"}, "c1"),
            ("search_log", {"pattern": "x"}, "c2"),
            text="Let me look.",
        )
    )
    gemini = provider(client, FakeTime())

    response = gemini.generate_with_tools("S", [UserMessage("u")], [READ_FILE])

    assert response.function_calls == [
        FunctionCall("read_file", {"path": "a"}, "c1"),
        FunctionCall("search_log", {"pattern": "x"}, "c2"),
    ]
    assert response.text == "Let me look."
    assert response.complete
    assert response.usage.input_tokens == 1000
    assert response.usage.output_tokens == 100  # thinking tokens are billed as output


def test_the_model_turn_is_replayed_verbatim_with_its_thought_signature() -> None:
    client = FakeClient(
        call_response(("read_file", {"path": "a"}, "c1"), signature=b"sig"),
        call_response(("submit_diagnosis", {}, None)),
    )
    gemini = provider(client, FakeTime())
    first = gemini.generate_with_tools("S", [UserMessage("u")], [READ_FILE])
    call = first.function_calls[0]

    history: list[Message] = [
        UserMessage("u"),
        first.message,
        FunctionResultsMessage([FunctionResult(call, {"output": "1| name: app"})]),
    ]
    gemini.generate_with_tools("S", history, [READ_FILE])

    contents = client.models.calls[1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1] is first.message.raw
    assert contents[1].parts[0].thought_signature == b"sig"
    response_part = contents[2].parts[0].function_response
    assert response_part.id == "c1"
    assert response_part.name == "read_file"
    assert response_part.response == {"output": "1| name: app"}


def test_model_messages_without_a_raw_turn_are_rebuilt() -> None:
    message = ModelMessage("thinking", [FunctionCall("read_file", {"path": "a"}, "c9")])

    (content,) = to_contents([message])

    assert content.role == "model"
    assert content.parts is not None
    assert content.parts[0].text == "thinking"
    call = content.parts[1].function_call
    assert call is not None
    assert (call.name, call.args, call.id) == ("read_file", {"path": "a"}, "c9")


def test_an_empty_model_turn_is_not_replayed() -> None:
    empty = types.Content(role="model", parts=[])

    assert to_contents([UserMessage("u"), ModelMessage("", [], raw=empty)])[1:] == []
    assert to_contents([ModelMessage("", [])]) == []


def test_a_response_without_candidates_is_incomplete() -> None:
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        )
    )
    gemini = provider(FakeClient(blocked), FakeTime())

    response = gemini.generate_with_tools("S", [UserMessage("u")], [READ_FILE])

    assert not response.complete
    assert response.block_reason == "SAFETY"
    assert response.function_calls == []
    assert response.message.raw is None


def test_tool_calls_share_the_rate_limit_and_backoff() -> None:
    fake = FakeTime()
    client = FakeClient(
        call_response(("read_file", {}, None)),
        rate_limit_error("7s"),
        call_response(("read_file", {}, None)),
        call_response(("read_file", {}, None)),
    )
    gemini = provider(client, fake, rpm=2)

    for _ in range(3):
        gemini.generate_with_tools("S", [UserMessage("u")], [READ_FILE])

    # A retry after 7s, then the third request waits for the 60 s window.
    assert fake.sleeps[0] == pytest.approx(7.0)
    assert sum(fake.sleeps) >= 60.0
    assert len(client.models.calls) == 4


def test_request_errors_are_not_retried() -> None:
    bad = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    gemini = provider(FakeClient(bad), FakeTime())

    with pytest.raises(LLMError, match="400"):
        gemini.generate_with_tools("S", [UserMessage("u")], [READ_FILE])
