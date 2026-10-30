from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from cidoc.config import Settings, get_settings

# Fake values for tests only; never put real credentials here.
FAKE_GITHUB_TOKEN = "test-github-token-value"  # noqa: S105
FAKE_ANTHROPIC_KEY = "test-anthropic-key-value"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run each test in an empty dir with no real secrets in the environment."""
    for name in ("GITHUB_TOKEN", "ANTHROPIC_API_KEY", "GITHUB_API_URL", "DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_loads_from_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)

    settings = get_settings()

    assert settings.github_token.get_secret_value() == FAKE_GITHUB_TOKEN
    assert settings.anthropic_api_key.get_secret_value() == FAKE_ANTHROPIC_KEY
    assert settings.github_api_url == "https://api.github.com"
    assert settings.data_dir == Path("data")


def test_loads_from_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"GITHUB_TOKEN={FAKE_GITHUB_TOKEN}\nANTHROPIC_API_KEY={FAKE_ANTHROPIC_KEY}\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.github_token.get_secret_value() == FAKE_GITHUB_TOKEN
    assert settings.anthropic_api_key.get_secret_value() == FAKE_ANTHROPIC_KEY


def test_env_var_overrides_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"GITHUB_TOKEN=from-dotenv\nANTHROPIC_API_KEY={FAKE_ANTHROPIC_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)

    assert Settings().github_token.get_secret_value() == FAKE_GITHUB_TOKEN


def test_missing_secrets_raise() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings()

    missing = {err["loc"][0] for err in exc_info.value.errors()}
    assert missing == {"github_token", "anthropic_api_key"}


def test_secrets_masked_in_repr_and_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)

    settings = Settings()

    for rendered in (repr(settings), str(settings), settings.model_dump_json()):
        assert FAKE_GITHUB_TOKEN not in rendered
        assert FAKE_ANTHROPIC_KEY not in rendered
        assert "**********" in rendered
