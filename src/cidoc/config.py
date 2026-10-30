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
    anthropic_api_key: SecretStr
    github_api_url: str = "https://api.github.com"
    data_dir: Path = Path("data")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
