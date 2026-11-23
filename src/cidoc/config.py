"""Application settings loaded from environment variables and an optional .env file.

Secrets are stored as SecretStr so they render as '**********' in repr/str/logs.
Call .get_secret_value() only at the point of use (e.g. building an auth header).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    github_token: SecretStr
    # Optional: Phase 0 mining/labeling never calls Anthropic. Commands that do
    # (analyze/eval) must go through require_anthropic_key().
    anthropic_api_key: SecretStr | None = None
    github_api_url: str = "https://api.github.com"
    data_dir: Path = Path("data")


class MissingAnthropicKeyError(RuntimeError):
    """Raised when a command needs the Anthropic API but no key is configured."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def require_anthropic_key(settings: Settings | None = None) -> SecretStr:
    """Return the Anthropic API key, or fail clearly if it is not configured.

    Call this only where the Anthropic client is created, so commands that don't
    talk to Anthropic keep working without the key. The key stays wrapped in
    SecretStr; unwrap it with .get_secret_value() when building the client.
    """
    key = (settings or get_settings()).anthropic_api_key
    if key is None or not key.get_secret_value():
        raise MissingAnthropicKeyError("ANTHROPIC_API_KEY is not set; required for analyze/eval")
    return key
