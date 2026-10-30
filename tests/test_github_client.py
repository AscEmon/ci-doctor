import httpx
from pydantic import SecretStr

from cidoc.github_client import GitHubClient

# Fake value for tests only; never put real credentials here.
FAKE_TOKEN = SecretStr("test-github-token-value")


def make_client(transport: httpx.MockTransport) -> GitHubClient:
    return GitHubClient(FAKE_TOKEN, transport=transport)


def test_sends_auth_and_api_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    with make_client(httpx.MockTransport(handler)) as client:
        assert client.get_json("/repos/o/r") == {"ok": True}

    headers = seen[0].headers
    assert headers["Authorization"] == "Bearer test-github-token-value"
    assert headers["Accept"] == "application/vnd.github+json"
    assert headers["X-GitHub-Api-Version"] == "2022-11-28"


def test_log_download_follows_redirect_without_leaking_auth() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "api.github.com":
            return httpx.Response(302, headers={"Location": "https://blob.example.com/log.zip"})
        return httpx.Response(200, content=b"log-bytes")

    with make_client(httpx.MockTransport(handler)) as client:
        assert client.download_log("/repos/o/r/actions/runs/1/logs") == b"log-bytes"

    assert [r.url.host for r in seen] == ["api.github.com", "blob.example.com"]
    assert "Authorization" not in seen[1].headers


def test_repr_does_not_expose_token() -> None:
    client = make_client(httpx.MockTransport(lambda r: httpx.Response(200)))
    assert "test-github-token-value" not in repr(client)
    client.close()
