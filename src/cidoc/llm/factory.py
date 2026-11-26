"""Build the configured provider. The only place that knows which providers exist."""

from cidoc.config import Settings, require_gemini_key
from cidoc.llm.base import LLMProvider, ToolCallingProvider
from cidoc.llm.gemini import GeminiProvider


def _gemini(settings: Settings, model: str | None) -> GeminiProvider:
    return GeminiProvider(
        model=model or settings.cidoc_model,
        rpm=settings.cidoc_rpm,
        api_key=require_gemini_key(settings),
    )


def create_provider(settings: Settings, model: str | None = None) -> LLMProvider:
    """The provider for `model` (default CIDOC_MODEL). Fails clearly without an API key."""
    return _gemini(settings, model)


def create_tool_provider(settings: Settings, model: str | None = None) -> ToolCallingProvider:
    """The function-calling provider for the agent; same model, key and RPM limit."""
    return _gemini(settings, model)
