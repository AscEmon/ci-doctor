"""Application settings loaded from environment variables and an optional .env file.

Secrets are stored as SecretStr so they render as '**********' in repr/str/logs.
Call .get_secret_value() only at the point of use (e.g. building an auth header).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from cidoc.analyze.preprocess import DEFAULT_TOKEN_BUDGET
from cidoc.llm.pricing import Price

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    github_token: SecretStr
    # Optional: mining/labeling never call the LLM. Commands that do (analyze/eval)
    # must go through require_gemini_key().
    gemini_api_key: SecretStr | None = None
    github_api_url: str = "https://api.github.com"
    data_dir: Path = Path("data")

    # Analyzer settings, all from CIDOC_* env vars.
    cidoc_model: str = DEFAULT_MODEL
    cidoc_rpm: PositiveInt = 10
    """Requests per minute sent to the LLM API."""
    cidoc_token_budget: PositiveInt = DEFAULT_TOKEN_BUDGET
    """Estimated tokens of preprocessed log sent per analysis."""
    cidoc_agent_max_input: PositiveInt = 80_000
    """Agent: total input tokens over all model calls of one item (agent.loop default)."""
    cidoc_prices: dict[str, Price] = {}
    """USD per million tokens by model, as JSON. Never hardcoded: prices change."""

    def price_for(self, model: str) -> Price | None:
        return self.cidoc_prices.get(model)


class MissingGeminiKeyError(RuntimeError):
    """Raised when a command needs the Gemini API but no key is configured."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def require_gemini_key(settings: Settings | None = None) -> SecretStr:
    """Return the Gemini API key, or fail clearly if it is not configured.

    Call this only where the Gemini client is created, so commands that don't
    talk to Gemini keep working without the key. The key stays wrapped in
    SecretStr; unwrap it with .get_secret_value() when building the client.
    """
    key = (settings or get_settings()).gemini_api_key
    if key is None or not key.get_secret_value():
        raise MissingGeminiKeyError("GEMINI_API_KEY is not set; required for analyze/eval")
    return key
