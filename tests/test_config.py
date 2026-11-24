from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from cidoc.config import (
    DEFAULT_MODEL,
    MissingGeminiKeyError,
    Settings,
    get_settings,
    require_gemini_key,
)

# Fake values for tests only; never put real credentials here.
FAKE_GITHUB_TOKEN = "test-github-token-value"  # noqa: S105
FAKE_GEMINI_KEY = "test-gemini-key-value"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run each test in an empty dir with no real secrets in the environment."""
    for name in (
        "GITHUB_TOKEN",
        "GEMINI_API_KEY",
        "GITHUB_API_URL",
        "DATA_DIR",
        "CIDOC_MODEL",
        "CIDOC_RPM",
        "CIDOC_TOKEN_BUDGET",
        "CIDOC_PRICES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_loads_from_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)

    settings = get_settings()

    assert settings.github_token.get_secret_value() == FAKE_GITHUB_TOKEN
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == FAKE_GEMINI_KEY
    assert settings.github_api_url == "https://api.github.com"
    assert settings.data_dir == Path("data")


def test_loads_from_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"GITHUB_TOKEN={FAKE_GITHUB_TOKEN}\nGEMINI_API_KEY={FAKE_GEMINI_KEY}\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.github_token.get_secret_value() == FAKE_GITHUB_TOKEN
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == FAKE_GEMINI_KEY


def test_env_var_overrides_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"GITHUB_TOKEN=from-dotenv\nGEMINI_API_KEY={FAKE_GEMINI_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)

    assert Settings().github_token.get_secret_value() == FAKE_GITHUB_TOKEN


def test_missing_github_token_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings()

    missing = {err["loc"][0] for err in exc_info.value.errors()}
    assert missing == {"github_token"}


def test_loads_with_only_github_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)

    settings = Settings()

    assert settings.github_token.get_secret_value() == FAKE_GITHUB_TOKEN
    assert settings.gemini_api_key is None


def test_require_gemini_key_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)

    with pytest.raises(MissingGeminiKeyError, match="GEMINI_API_KEY is not set"):
        require_gemini_key()


def test_require_gemini_key_raises_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "")

    with pytest.raises(MissingGeminiKeyError):
        require_gemini_key()


def test_require_gemini_key_returns_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)

    key = require_gemini_key()

    assert key.get_secret_value() == FAKE_GEMINI_KEY


def test_secrets_masked_in_repr_and_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)

    settings = Settings()

    for rendered in (repr(settings), str(settings), settings.model_dump_json()):
        assert FAKE_GITHUB_TOKEN not in rendered
        assert FAKE_GEMINI_KEY not in rendered
        assert "**********" in rendered


def test_analyzer_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)

    settings = Settings()

    assert settings.cidoc_model == DEFAULT_MODEL == "gemini-3.5-flash-lite"
    assert settings.cidoc_rpm == 10
    assert settings.cidoc_token_budget == 6000
    assert settings.cidoc_prices == {}
    assert settings.price_for(DEFAULT_MODEL) is None


def test_analyzer_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv("CIDOC_MODEL", "some-model")
    monkeypatch.setenv("CIDOC_RPM", "3")
    monkeypatch.setenv("CIDOC_TOKEN_BUDGET", "1234")
    monkeypatch.setenv("CIDOC_PRICES", '{"some-model": {"input": 0.5, "output": 2}}')

    settings = Settings()

    assert settings.cidoc_model == "some-model"
    assert settings.cidoc_rpm == 3
    assert settings.cidoc_token_budget == 1234
    price = settings.price_for("some-model")
    assert price is not None
    assert price.cost(1_000_000, 500_000) == pytest.approx(0.5 + 1.0)
    assert settings.price_for("other-model") is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CIDOC_RPM", "0"),
        ("CIDOC_TOKEN_BUDGET", "-1"),
        ("CIDOC_PRICES", '{"m": {"input": -1, "output": 1}}'),
        ("CIDOC_PRICES", '{"m": {"input": 1}}'),
        ("CIDOC_PRICES", '{"m": {"input": "[PLACEHOLDER]", "output": 1}}'),
    ],
)
def test_invalid_analyzer_settings_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings()
