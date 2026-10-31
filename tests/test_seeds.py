import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from cidoc.db import connect
from cidoc.github_client import GitHubClient
from cidoc.miner.discover import RepoInfo, save_repo
from cidoc.miner.seeds import add_seed_repos, parse_seed_file

# Fake value for tests only; never put real credentials here.
FAKE_TOKEN = SecretStr("test-github-token-value")
NOW = datetime(2026, 10, 31, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(tmp_path / "cidoc.db")
    yield db
    db.close()


def test_parse_skips_blanks_comments_and_duplicates() -> None:
    text = "# seeds\nflutter/gallery\n\n  foo-org/my.app  # trailing note\nflutter/gallery\n"

    assert parse_seed_file(text) == ["flutter/gallery", "foo-org/my.app"]


@pytest.mark.parametrize("line", ["just-a-name", "https://github.com/o/r", "o/r/extra", "o / r"])
def test_parse_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(ValueError, match="line 2"):
        parse_seed_file(f"ok/repo\n{line}\n")


def test_add_seed_repos(conn: sqlite3.Connection, tmp_path: Path) -> None:
    save_repo(conn, RepoInfo("o/known", 10, "main"), NOW)
    seed_file = tmp_path / "seed_repos.txt"
    seed_file.write_text("o/known\no/new\no/gone\no/old-name\n", encoding="utf-8")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.removeprefix("/repos/")
        requested.append(name)
        if name == "o/gone":
            return httpx.Response(404, json={"message": "Not Found"})
        canonical = "o/new-name" if name == "o/old-name" else name
        return httpx.Response(
            200, json={"full_name": canonical, "stargazers_count": 42, "default_branch": "dev"}
        )

    missing: list[str] = []
    client = GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler))
    with client:
        added = add_seed_repos(client, conn, seed_file, NOW, on_missing=missing.append)

    assert added == 2
    assert requested == ["o/new", "o/gone", "o/old-name"]
    assert missing == ["o/gone"]
    rows = {tuple(r) for r in conn.execute("SELECT full_name, stars, default_branch FROM repos")}
    assert rows == {("o/known", 10, "main"), ("o/new", 42, "dev"), ("o/new-name", 42, "dev")}
