import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from cidoc.db import connect
from cidoc.github_client import GitHubClient
from cidoc.miner.discover import (
    RepoInfo,
    build_search_query,
    discover,
    save_repo,
    search_repos,
    uses_flutter_workflow,
)

# Fake value for tests only; never put real credentials here.
FAKE_TOKEN = SecretStr("test-github-token-value")
NOW = datetime(2026, 10, 31, tzinfo=UTC)

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler) -> GitHubClient:
    return GitHubClient(FAKE_TOKEN, transport=httpx.MockTransport(handler), sleep=lambda _: None)


def search_item(name: str, stars: int = 500, **extra: Any) -> dict[str, Any]:
    return {"full_name": name, "stargazers_count": stars, "default_branch": "main", **extra}


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(tmp_path / "cidoc.db")
    yield db
    db.close()


def test_search_query_filters() -> None:
    assert build_search_query(NOW) == (
        "language:Dart stars:>=200 pushed:>=2026-05-02 archived:false fork:false"
    )


def test_search_paginates_and_respects_limit() -> None:
    pages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params["page"]
        pages.append(page)
        offset = (int(page) - 1) * 100
        return httpx.Response(
            200, json={"items": [search_item(f"o/r{offset + i}") for i in range(100)]}
        )

    with make_client(handler) as client:
        repos = list(search_repos(client, "q", limit=150))

    assert len(repos) == 150
    assert repos[-1].full_name == "o/r149"
    assert pages == ["1", "2"]


def test_search_stops_on_short_page_and_drops_forks() -> None:
    items = [search_item("o/a"), search_item("o/fork", fork=True), search_item("o/b")]

    with make_client(lambda r: httpx.Response(200, json={"items": items})) as client:
        repos = list(search_repos(client, "q", limit=50))

    assert [r.full_name for r in repos] == ["o/a", "o/b"]


def workflow_handler(files: dict[str, str], status: int = 200) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"message": "nope"})
        if request.url.path.endswith("/actions/workflows"):
            workflows = [{"path": p} for p in files] + [
                {"path": "dynamic/pages/pages-build-deployment"}
            ]
            return httpx.Response(200, json={"workflows": workflows})
        path = request.url.path.split("/contents/", 1)[1]
        assert request.headers["Accept"] == "application/vnd.github.raw+json"
        return httpx.Response(200, text=files[path])

    return handler


def test_flutter_workflow_detected() -> None:
    files = {
        ".github/workflows/lint.yml": "steps:\n  - run: dart format .\n",
        ".github/workflows/build.yml": "uses: subosito/Flutter-action@v2\n",
    }
    with make_client(workflow_handler(files)) as client:
        assert uses_flutter_workflow(client, "o/r")


def test_non_flutter_workflows_rejected() -> None:
    files = {".github/workflows/ci.yml": "run: dart test\n"}
    with make_client(workflow_handler(files)) as client:
        assert not uses_flutter_workflow(client, "o/r")


def test_missing_actions_is_not_an_error() -> None:
    with make_client(workflow_handler({}, status=404)) as client:
        assert not uses_flutter_workflow(client, "o/r")


def test_save_repo_upserts_without_touching_discovered_at(conn: sqlite3.Connection) -> None:
    save_repo(conn, RepoInfo("o/r", 300, "main"), NOW)
    save_repo(conn, RepoInfo("o/r", 450, "dev"), datetime(2026, 11, 5, tzinfo=UTC))

    row = conn.execute("SELECT * FROM repos").fetchone()
    assert tuple(row) == ("o/r", 450, "dev", NOW.isoformat())


def test_discover_stores_flutter_repos_and_skips_known(conn: sqlite3.Connection) -> None:
    save_repo(conn, RepoInfo("o/known", 900, "main"), NOW)
    workflow_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/repositories":
            items = [search_item("o/known"), search_item("o/flutter"), search_item("o/plain")]
            return httpx.Response(200, json={"items": items})
        repo = "/".join(path.split("/")[2:4])
        workflow_calls.append(repo)
        if path.endswith("/actions/workflows"):
            return httpx.Response(200, json={"workflows": [{"path": ".github/workflows/ci.yml"}]})
        body = "run: flutter test" if repo == "o/flutter" else "run: dart test"
        return httpx.Response(200, text=body)

    with make_client(handler) as client:
        added = discover(client, conn, NOW, limit=10)

    assert added == 1
    names = {row[0] for row in conn.execute("SELECT full_name FROM repos")}
    assert names == {"o/known", "o/flutter"}
    assert "o/known" not in workflow_calls
