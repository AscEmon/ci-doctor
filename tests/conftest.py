"""Fixtures shared by the CLI tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from cidoc.config import Settings, get_settings
from tests.agent_fakes import FakeGitHub
from tests.fakes import VALID_DIAGNOSIS, ScriptedProvider, ok


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """An empty data dir and settings that never see the developer's .env or keys."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token-value")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    for name in ("GEMINI_API_KEY", "CIDOC_MODEL", "CIDOC_RPM", "CIDOC_TOKEN_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CIDOC_PRICES", raising=False)
    get_settings.cache_clear()
    yield tmp_path / "data"
    get_settings.cache_clear()


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> ScriptedProvider:
    """The provider every CLI command gets; answers VALID_DIAGNOSIS once by default."""
    scripted = ScriptedProvider(ok(VALID_DIAGNOSIS))

    def create(settings: Settings, model: str | None = None) -> ScriptedProvider:
        return scripted

    monkeypatch.setattr("cidoc.cli.create_provider", create)
    return scripted


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> FakeGitHub:
    """The GitHub client agent tools get in CLI commands; 404s anything not routed."""
    fake = FakeGitHub()
    monkeypatch.setattr("cidoc.cli.open_tools_github", lambda token, base_url: fake)
    return fake
